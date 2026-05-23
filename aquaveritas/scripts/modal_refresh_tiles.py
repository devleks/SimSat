"""
Modal scheduled wrapper for scripts/refresh_tiles.py — Option 2 deployment.

Architecture:
    Modal cron (Mon 06:00 UTC weekly)
        │
        ▼
    Container: aquaveritas-refresh
        ├─ Clones git repo at HEAD
        ├─ Starts vendored SimSat API (uvicorn api:api on :9005)
        │   — no docker-in-docker; sources baked into the image at build time
        ├─ Pulls fine-tuned GGUF + mmproj from HuggingFace
        │   (Arty1001/aquaveritas-lfm-GGUF, cached on Modal volume)
        ├─ Boots llama-server with the GGUF on :8080 (--ctx-size 8192)
        ├─ Runs `python scripts/refresh_tiles.py` (writes WebPs + JSON)
        └─ Commits changes on branch `refresh/<ISO_DATE>` + opens a PR via `gh`

Why we vendor SimSat instead of running its docker-compose:
    Modal functions don't have a Docker daemon inside them, so we can't
    spin up the upstream `fakesat-sim` / `fakesat-dashboard` containers.
    Instead we vendor just the FastAPI surface (`scripts/vendored_sim/`,
    ~60kB) and run it with uvicorn. The endpoint refresh_tiles.py uses
    (`/data/image/sentinel`) is stateless and needs neither the orbital
    simulator nor the Django dashboard.

Setup (one-time):
    modal token new                                       # auth Modal
    modal secret create gh-token-aquaveritas GH_TOKEN=... # for PR opening
    modal deploy scripts/modal_refresh_tiles.py           # ships the cron

Manual trigger (no waiting for Monday):
    modal run scripts/modal_refresh_tiles.py::main
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "aquaveritas-refresh"

# Container image. The build chain is:
#   1. system deps for GDAL/rasterio (SentinelProvider uses pystac-client +
#      odc-stac under the hood, both of which need GDAL bindings)
#   2. Python deps for both the SimSat API and refresh_tiles.py
#   3. llama.cpp built from source (CPU-only — model is 450M params)
#   4. Vendored sim sources copied into /opt/sim
#   5. expat workaround — same one Dockerfile.sim uses, otherwise rasterio
#      fails to load on debian-slim
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "curl", "ca-certificates",
        "build-essential", "cmake", "gcc", "g++",
        "libgdal-dev", "libexpat1", "libexpat1-dev",
    )
    .pip_install(
        # refresh_tiles + evaluator deps
        "requests>=2.31",
        "pillow>=10.0",
        "openai>=1.40",
        "anthropic>=0.34",       # imported by evaluator.py even if unused here
        "psycopg2-binary>=2.9",  # imported by db.py even if unused here
        "huggingface_hub>=0.24",
        # SimSat API runtime — slimmed subset (no pyqt6/cartopy: those are
        # GUI / map-rendering side and not pulled in by /data/image/sentinel).
        # matplotlib IS needed despite seeming GUI-only — sentinel_provider
        # uses its colour helpers for the SWIR composite.
        "fastapi>=0.110",
        "uvicorn>=0.27",
        "numpy>=1.26",
        "matplotlib>=3.8",
        "pyorbital>=1.8",
        "pystac-client>=0.7",
        "odc-stac>=0.3",
        "rasterio>=1.3",
    )
    .run_commands(
        # Build llama.cpp (CPU-only is fine; 450M Q8_0 inferences in <2s/call).
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build && cmake --build build --config Release -j",
        # Same rasterio/expat workaround as the upstream Dockerfile.sim.
        # Without this, SentinelProvider raises 'libexpat.so.1 not found'.
        "LIB_PATH=$(python3 -c \"import rasterio; from pathlib import Path; "
        "print(Path(rasterio.__file__).parent / '../rasterio.libs')\") && "
        "mkdir -p $LIB_PATH && "
        "ln -sf /usr/lib/x86_64-linux-gnu/libexpat.so.1 $LIB_PATH/libexpat.so.1",
        "ldconfig",
    )
    .add_local_dir(
        local_path="scripts/vendored_sim",
        remote_path="/opt/sim",
        copy=True,  # bake into image, not mount at runtime
    )
)

app = modal.App(APP_NAME, image=image)

# HF cache volume — first run downloads ~640MB, subsequent runs reuse it.
hf_cache_vol = modal.Volume.from_name(
    "aquaveritas-hf-cache", create_if_missing=True,
)
HF_REPO     = "Arty1001/aquaveritas-lfm-GGUF"
GGUF_FILE   = "aquaveritas-lfm-q8_0.gguf"          # 451 MB Q8_0 backbone
MMPROJ_FILE = "mmproj-aquaveritas-lfm-F16.gguf"    # 189 MB vision encoder
GH_SECRET   = modal.Secret.from_name("gh-token-aquaveritas")  # exports GH_TOKEN

REPO_URL    = "https://github.com/devleks/AquaVeritas.git"  # adjust if forked
REPO_BRANCH = "main"


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    """Subprocess helper that prints + checks. Returns stdout."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout)
    return result.stdout


def _wait_for_http(url: str, timeout_s: int, what: str) -> None:
    """Poll URL until it returns 2xx or timeout. Raises on timeout."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if 200 <= r.status < 300:
                    print(f"  {what} ready at {url} (HTTP {r.status})")
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError(f"{what} did not become ready at {url} within {timeout_s}s")


@app.function(
    timeout=60 * 60,                       # 1 hour upper bound
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[GH_SECRET],
    cpu=4.0,
    memory=8192,
    schedule=modal.Cron("0 6 * * 1"),      # Mondays 06:00 UTC
)
def refresh() -> dict:
    """The cron-fired entrypoint. Returns a summary dict."""
    run_id = f"refresh-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    workdir = Path(f"/tmp/{run_id}")
    workdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Clone repo ────────────────────────────────────────────────────────
    _run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, "repo"],
         cwd=workdir)
    repo = workdir / "repo" / "aquaveritas"

    # ── 2. Start vendored SimSat API (uvicorn) in the background ─────────────
    sim_proc = subprocess.Popen(
        ["uvicorn", "api:api", "--host", "127.0.0.1", "--port", "9005"],
        cwd="/opt/sim",
    )
    _wait_for_http(
        # /data/image/sentinel is the endpoint we use — but its 503-without-
        # explicit-params behaviour means a GET probe is a fine readiness
        # signal (any 4xx/5xx response means uvicorn is up).
        "http://127.0.0.1:9005/docs", timeout_s=30, what="SimSat API",
    )

    # ── 3. Pull GGUF + mmproj from HuggingFace (cached on volume) ────────────
    from huggingface_hub import hf_hub_download
    print(f"Pulling {HF_REPO} (cached: {hf_cache_vol})…")
    gguf_path   = hf_hub_download(repo_id=HF_REPO, filename=GGUF_FILE)
    mmproj_path = hf_hub_download(repo_id=HF_REPO, filename=MMPROJ_FILE)
    print(f"  backbone: {gguf_path}")
    print(f"  mmproj:   {mmproj_path}")

    # ── 4. Boot llama-server with the fine-tuned GGUF + mmproj ───────────────
    # --ctx-size 8192 because two Sentinel-2 images tokenize to ~5600 tokens
    # together; 4096 (the default) overflows. Proven during the 2026-05-23
    # local smoke test.
    llama_proc = subprocess.Popen([
        "/opt/llama.cpp/build/bin/llama-server",
        "--model",    gguf_path,
        "--mmproj",   mmproj_path,
        "--port",     "8080",
        "--host",     "127.0.0.1",
        "--ctx-size", "8192",
    ])
    _wait_for_http(
        "http://127.0.0.1:8080/health", timeout_s=90, what="llama-server",
    )

    try:
        # ── 5. Run the refresh ───────────────────────────────────────────────
        _run(
            ["python", "scripts/refresh_tiles.py",
             "--llama-url",  "http://127.0.0.1:8080",
             "--simsat-url", "http://127.0.0.1:9005"],
            cwd=repo,
        )

        # ── 6. Commit + push branch + open PR ────────────────────────────────
        branch = f"refresh/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        env = {**os.environ,
               "GIT_AUTHOR_NAME":     "AquaVeritas Refresh Bot",
               "GIT_AUTHOR_EMAIL":    "refresh-bot@aquaveritas.local",
               "GIT_COMMITTER_NAME":  "AquaVeritas Refresh Bot",
               "GIT_COMMITTER_EMAIL": "refresh-bot@aquaveritas.local"}
        _run(["git", "checkout", "-b", branch], cwd=repo, env=env)
        _run(["git", "add",
              "app_web/public/sample_tiles/",
              "app_web/src/lib/predictions.json",
              "app_web/src/lib/sites.ts"], cwd=repo, env=env)

        # Empty diff means nothing changed → exit gracefully without a PR.
        diff = subprocess.run(["git", "diff", "--cached", "--stat"],
                              cwd=repo, capture_output=True, text=True, check=True)
        if not diff.stdout.strip():
            print("No diff — skipping commit and PR.")
            return {"run_id": run_id, "pr_opened": False, "reason": "no diff"}

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _run(["git", "commit", "-m",
              f"Weekly refresh: 20-site tiles + predictions ({today})"],
             cwd=repo, env=env)
        push_url = REPO_URL.replace(
            "https://", f"https://x-access-token:{os.environ['GH_TOKEN']}@",
        )
        _run(["git", "push", push_url, branch], cwd=repo, env=env)
        pr_url = _run([
            "gh", "pr", "create",
            "--repo",  REPO_URL.removesuffix(".git").removeprefix("https://github.com/"),
            "--head",  branch,
            "--base",  REPO_BRANCH,
            "--title", f"Weekly tile refresh — {today}",
            "--body",  f"Automated weekly refresh from Modal run `{run_id}`.\n\n"
                       f"See `scripts/modal_refresh_tiles.py` for the pipeline.",
        ], cwd=repo, env=env).strip()
        return {"run_id": run_id, "pr_opened": True, "pr_url": pr_url}

    finally:
        for proc, name in ((llama_proc, "llama-server"), (sim_proc, "uvicorn")):
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception as exc:  # noqa: BLE001
                print(f"  warning: {name} cleanup raised: {exc}")


@app.local_entrypoint()
def main():
    """Allows `modal run scripts/modal_refresh_tiles.py::main` for manual fire."""
    result = refresh.remote()
    print("Refresh complete:", result)
