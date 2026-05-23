"""
Modal scheduled wrapper for scripts/refresh_tiles.py — Option 2 deployment.

Architecture:
    Modal cron (Mon 06:00 UTC weekly)
        │
        ▼
    Container: aquaveritas-refresh
        ├─ Clones git repo at HEAD
        ├─ Boots SimSat docker-compose stack (postgres + simsat API)
        ├─ Pulls fine-tuned GGUF + mmproj from HuggingFace
        │   (Arty1001/aquaveritas-lfm-GGUF, cached on Modal volume)
        ├─ Boots llama-server with the GGUF on :8080
        ├─ Runs `python scripts/refresh_tiles.py` (writes WebPs + JSON)
        └─ Commits changes on branch `refresh/<ISO_DATE>` + opens a PR via `gh`

Why one container does it all:
    - Stateless. Anything we need lives in the image or in a Modal Volume.
    - The fetch path requires the SimSat backend, and SimSat itself is the
      thing that proxies Sentinel-2 imagery. Running it inside the container
      avoids exposing a long-lived public SimSat URL.
    - llama-server is the inference backend `refresh_tiles.py` already
      targets — no separate code path needed.

Setup (one-time):
    modal token new                                       # auth Modal
    modal secret create gh-token-aquaveritas GH_TOKEN=... # for PR opening
    # Model is pulled from HuggingFace on container cold-start — no Modal
    # volume needed. The download is cached in /root/.cache/huggingface
    # which lives on a small persistent volume so the second run is fast.
    modal deploy scripts/modal_refresh_tiles.py           # ships the cron

Manual trigger (no waiting for Monday):
    modal run scripts/modal_refresh_tiles.py::refresh
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "aquaveritas-refresh"

# Container image — pinned tags so weekly runs don't drift.
# llama.cpp built from source inside the image: the Modal base image gives
# us CUDA toolkit, we layer the build on top. ~2GB image; acceptable for
# a weekly job.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential", "cmake", "ca-certificates",
                 "docker.io", "docker-compose-plugin")
    .pip_install(
        "requests>=2.31",
        "pillow>=10.0",
        "openai>=1.40",          # llama-server is OpenAI-compatible
        "anthropic>=0.34",       # imported by evaluator.py even if unused here
        "psycopg2-binary>=2.9",  # imported by db.py even if unused here
        "huggingface_hub>=0.24", # for pulling the GGUF + mmproj from HF
    )
    .run_commands(
        # Build llama.cpp (CPU-only; the model is 450M params, fine on CPU).
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build && cmake --build build --config Release -j",
    )
)

app = modal.App(APP_NAME, image=image)

# Model is pulled from HF on cold-start. Cache lives on a small persistent
# volume so subsequent weekly runs don't re-download the ~640MB total.
hf_cache_vol = modal.Volume.from_name(
    "aquaveritas-hf-cache", create_if_missing=True,
)
HF_REPO    = "Arty1001/aquaveritas-lfm-GGUF"
GGUF_FILE  = "aquaveritas-lfm-q8_0.gguf"          # 451 MB, Q8_0 backbone
MMPROJ_FILE = "mmproj-aquaveritas-lfm-F16.gguf"   # 189 MB, vision encoder
GH_SECRET  = modal.Secret.from_name("gh-token-aquaveritas")  # exports GH_TOKEN

REPO_URL  = "https://github.com/devleks/AquaVeritas.git"  # adjust if forked
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


@app.function(
    timeout=60 * 60,                       # 1 hour upper bound
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[GH_SECRET],
    # cpu/memory: 4 vCPU + 8GB is plenty for 450M Q8_0 inference at 20 sites.
    cpu=4.0,
    memory=8192,
    schedule=modal.Cron("0 6 * * 1"),      # Mondays 06:00 UTC
)
def refresh() -> dict:
    """The actual cron-fired entrypoint. Returns a summary dict."""
    run_id = f"refresh-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    workdir = Path(f"/tmp/{run_id}")
    workdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Clone repo ────────────────────────────────────────────────────────
    _run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, "repo"],
         cwd=workdir)
    repo = workdir / "repo" / "aquaveritas"

    # ── 2. Boot the SimSat stack so refresh_tiles.py has a fetch source ──────
    # The repo's docker-compose.yml binds postgres:5433 + simsat:9005 to the
    # container's own loopback — fine for a single-container run.
    _run(["docker", "compose", "up", "-d"], cwd=repo)
    # SimSat takes ~10s to boot; healthcheck would be cleaner, this is fine
    # for a once-a-week job.
    time.sleep(15)

    # ── 3. Pull GGUF + mmproj from HuggingFace (cached on volume) ────────────
    from huggingface_hub import hf_hub_download
    print(f"Pulling {HF_REPO} (cached: {hf_cache_vol})…")
    gguf_path = hf_hub_download(repo_id=HF_REPO, filename=GGUF_FILE)
    mmproj_path = hf_hub_download(repo_id=HF_REPO, filename=MMPROJ_FILE)
    print(f"  backbone: {gguf_path}")
    print(f"  mmproj:   {mmproj_path}")

    # ── 4. Boot llama-server with the fine-tuned GGUF + mmproj ───────────────
    llama_proc = subprocess.Popen([
        "/opt/llama.cpp/build/bin/llama-server",
        "--model",  gguf_path,
        "--mmproj", mmproj_path,
        "--port",   "8080",
        "--host",   "127.0.0.1",
        "--ctx-size", "4096",
    ])
    # Give it ~20s for the model to memory-map; the fine-tuned LFM2.5-VL
    # 450M Q8_0 loads in roughly that on 4 vCPUs.
    time.sleep(20)

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
        env = {**os.environ, "GIT_AUTHOR_NAME": "AquaVeritas Refresh Bot",
               "GIT_AUTHOR_EMAIL": "refresh-bot@aquaveritas.local",
               "GIT_COMMITTER_NAME": "AquaVeritas Refresh Bot",
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
        # Push uses GH_TOKEN from the Modal secret.
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
        llama_proc.terminate()
        llama_proc.wait(timeout=10)
        subprocess.run(["docker", "compose", "down"], cwd=repo, check=False)


@app.local_entrypoint()
def main():
    """Allows `modal run scripts/modal_refresh_tiles.py::main` for manual fire."""
    result = refresh.remote()
    print("Refresh complete:", result)
