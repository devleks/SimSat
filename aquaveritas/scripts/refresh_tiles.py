"""
Weekly tile + prediction refresh — Option 2 from docs/LIVE_DATA_OPTIONS.md.

For each of the 20 monitored sites:
  1. Fetch a fresh Sentinel-2 RGB+SWIR pair via SimSatClient
  2. Run inference through llama-server (fine-tuned LFM2.5-VL GGUF)
  3. Save a resized WebP to app_web/public/sample_tiles/<site>.webp
  4. Update app_web/src/lib/predictions.json with the merged 11-field prediction
  5. Patch app_web/src/lib/sites.ts `capturedAt` to today's UTC date

The Modal wrapper (scripts/modal_refresh_tiles.py) runs this on a weekly cron
inside a container that has llama-server + the GGUF + the SimSat stack — then
commits the diff and opens a PR. Run locally to test the same flow:

    # Prerequisites: docker compose up -d; llama-server running on :8080
    python scripts/refresh_tiles.py --dry-run        # plan, no writes
    python scripts/refresh_tiles.py                  # full refresh
    python scripts/refresh_tiles.py --site lake_chad # one site only

Failure policy: per-site exceptions are logged and skipped; the script always
exits 0 unless ZERO sites refreshed successfully. That way a single CDSE miss
doesn't void the rest of the week's update.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image  # noqa: E402 — sys.path append must precede

from aquaveritas.evaluator import LlamaBackend  # noqa: E402
from aquaveritas.locations import LOCATIONS, LOCATIONS_BY_ID  # noqa: E402
from aquaveritas.sentinel_fetcher import fetch_location_images  # noqa: E402

REPO_ROOT       = Path(__file__).parent.parent
WEB_PUBLIC_DIR  = REPO_ROOT / "app_web" / "public" / "sample_tiles"
PREDICTIONS_JSON = REPO_ROOT / "app_web" / "src" / "lib" / "predictions.json"
SITES_TS         = REPO_ROOT / "app_web" / "src" / "lib" / "sites.ts"

# WebP encoding: 640px square is enough for the /globe panel preview at
# any zoom. Quality 78 keeps each file ~50-80kB (matches copy_web_samples).
WEBP_SIZE_PX = 640
WEBP_QUALITY = 78


def _merged_prediction(core: dict | None, buffer: dict | None) -> dict | None:
    """
    Merge core (4 fields) + buffer (6 fields) into the 11-field shape the
    web app expects. Returns None if either inference call failed — a
    partial prediction would silently mislead the /globe panel.
    """
    if not core or not buffer:
        return None
    return {
        # Core
        "water_extent_status":    core["water_extent_status"],
        "flood_risk":             core["flood_risk"],
        "water_clarity":          core["water_clarity"],
        "shoreline_encroachment": bool(core["shoreline_encroachment"]),
        # Buffer
        "agriculture_present":              bool(buffer["agriculture_present"]),
        "crop_stress_level":                buffer["crop_stress_level"],
        "crop_stress_type":                 buffer["crop_stress_type"],
        "cultivation_expanding_toward_water": bool(buffer["cultivation_expanding_toward_water"]),
        "settlement_visible":               bool(buffer["settlement_visible"]),
        "bare_soil_expansion":              bool(buffer["bare_soil_expansion"]),
        # 11th field — model emits it on core; default false if missing.
        "image_quality_limited": bool(core.get("image_quality_limited", False)),
    }


def _save_webp(png_bytes: bytes, dest: Path) -> int:
    """Convert PNG bytes → resized WebP. Returns file size in bytes."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img.thumbnail((WEBP_SIZE_PX, WEBP_SIZE_PX), Image.Resampling.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="WEBP", quality=WEBP_QUALITY, method=6)
    return dest.stat().st_size


def _patch_captured_at(site_id: str, iso_date: str) -> bool:
    """
    Rewrite the `capturedAt: "YYYY-MM-DD"` line for one site in sites.ts.
    Returns True if the file was modified. Naive regex on purpose: keeps
    the patch surgical and avoids round-tripping the whole TS file.
    """
    text = SITES_TS.read_text()
    # Look for the site's id line, then the following capturedAt within
    # the same object literal (next ~20 lines).
    pattern = re.compile(
        rf'(id:\s*"{re.escape(site_id)}".*?capturedAt:\s*")\d{{4}}-\d{{2}}-\d{{2}}(")',
        re.DOTALL,
    )
    new_text, n = pattern.subn(rf"\g<1>{iso_date}\g<2>", text, count=1)
    if n == 0:
        return False
    SITES_TS.write_text(new_text)
    return True


def refresh_one(
    location_id: str,
    backend: LlamaBackend | None,
    today_iso: str,
    dry_run: bool,
) -> tuple[dict | None, str]:
    """
    Returns (prediction-or-None, status_message) for one site.
    A None prediction means the site was skipped — caller decides whether
    that counts as failure (real run) or success (dry run).
    """
    loc = LOCATIONS_BY_ID[location_id]
    timestamp = datetime.now(timezone.utc).isoformat()

    images = fetch_location_images(
        lon=loc.lon, lat=loc.lat, timestamp=timestamp, location_id=loc.id,
    )
    if not images.any_core_available:
        return None, "no core image (CDSE returned nothing in window)"

    if dry_run or backend is None:
        return None, f"[dry-run] would refresh from pass at {timestamp[:16]}Z"

    core_out = backend.infer_core(images.rgb_core.image, images.swir_core.image, loc)
    buffer_out = None
    if images.any_buffer_available:
        buffer_out = backend.infer_buffer(
            images.rgb_buffer.image, images.swir_buffer.image, loc,
        )
    prediction = _merged_prediction(core_out, buffer_out)
    if prediction is None:
        return None, f"inference returned None (core={bool(core_out)}, buffer={bool(buffer_out)})"

    # Save WebP for the /globe panel.
    webp_path = WEB_PUBLIC_DIR / f"{location_id}.webp"
    size_kb = _save_webp(images.rgb_core.image, webp_path) / 1024

    # Patch sites.ts capturedAt.
    _patch_captured_at(location_id, today_iso)

    return prediction, (
        f"✓ {prediction['water_extent_status']:<10} "
        f"crop={prediction['crop_stress_level']:<8} "
        f"webp={size_kb:4.0f}kB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", help="Refresh only this site id (default: all 20)")
    parser.add_argument("--llama-url", default="http://localhost:8080",
                        help="llama-server base URL")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only: fetch images, skip inference and writes")
    args = parser.parse_args()

    targets = [LOCATIONS_BY_ID[args.site]] if args.site else LOCATIONS
    today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"=== AquaVeritas weekly tile refresh — {today_iso} ===")
    print(f"  sites:      {len(targets)}")
    print(f"  llama:      {args.llama_url}")
    print(f"  fetcher:    direct STAC (Element84 Earth Search v1)")
    print(f"  dry-run:    {args.dry_run}")
    print()

    backend = LlamaBackend(base_url=args.llama_url) if not args.dry_run else None

    # Load + (optionally) update predictions.json in one pass.
    predictions = json.loads(PREDICTIONS_JSON.read_text())
    refresh_run_id = f"refresh-{today_iso}-{int(time.time())}"
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for loc in targets:
        print(f"  → {loc.name:<24} ", end="", flush=True)
        try:
            prediction, msg = refresh_one(loc.id, backend, today_iso, args.dry_run)
        except Exception as exc:  # noqa: BLE001 — per-site isolation is the point
            prediction, msg = None, f"exception: {type(exc).__name__}: {exc}"
        print(msg)

        # Success is "we got what we wanted" — a prediction on a real run,
        # or an image-available marker on a dry run.
        if args.dry_run:
            ok = msg.startswith("[dry-run]")
        else:
            ok = prediction is not None
            if ok:
                predictions[loc.id] = prediction

        (succeeded if ok else failed).append(loc.id if ok else (loc.id, msg))

    # Persist predictions.json (only on real run, with at least one success).
    if not args.dry_run and succeeded:
        predictions["_meta"]["last_refreshed"] = today_iso
        predictions["_meta"]["refresh_run_id"] = refresh_run_id
        PREDICTIONS_JSON.write_text(
            json.dumps(predictions, indent=2) + "\n",
        )
        print(f"\nWrote {PREDICTIONS_JSON.relative_to(REPO_ROOT)}")

    # Summary
    print()
    print(f"=== Done: {len(succeeded)}/{len(targets)} succeeded ===")
    if failed:
        print("Failures:")
        for sid, msg in failed:
            print(f"  ✗ {sid}: {msg}")
    # Exit non-zero only if everything failed — single-site failures are tolerable.
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
