# Vendored SimSat API

Source: `/Users/ml_labs/claudey/SimSat/src/sim/` (the DPhi-Space SimSat repo).
Vendored so the Modal weekly-refresh container can stand up a SimSat API
endpoint without docker-in-docker. Only the bits we actually call are here:

| File                              | Why we need it                          |
|-----------------------------------|-----------------------------------------|
| `api.py`                          | FastAPI app — serves `/data/image/sentinel` |
| `ImagingProviders/sentinel_provider.py` | Hits the public Sentinel-2 STAC catalogue (no auth) |
| `ImagingProviders/mapbox_provider.py`   | Imported at module-load; unused by the refresh path |
| `ImagingProviders/__init__.py`    | Makes the package importable           |

Not vendored: `main.py`, `simulator.py`, `gui.py`, `camera.py` — these
power the orbital simulation + dashboard telemetry, neither of which the
refresh needs (we always pass explicit lat/lon to `/data/image/sentinel`).

## Running it

```bash
cd scripts/vendored_sim
uvicorn api:api --host 127.0.0.1 --port 9005
```

## Re-syncing from upstream

When the SimSat repo's `src/sim/api.py` or `ImagingProviders/` change in a
way we want to pick up:

```bash
cp /Users/ml_labs/claudey/SimSat/src/sim/api.py scripts/vendored_sim/api.py
cp /Users/ml_labs/claudey/SimSat/src/sim/ImagingProviders/*.py \
   scripts/vendored_sim/ImagingProviders/
```

Then re-run the Modal smoke test before deploying.
