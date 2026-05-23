"""
Direct Sentinel-2 L2A fetcher — no HTTP, no SimSat, no AGPL.

Built on:
  - pystac-client  (BSD-3) for STAC catalogue search
  - odc-stac       (Apache-2) for COG mosaic loading
  - rasterio       (BSD-3, transitively via odc-stac)
  - Pillow         (HPND) for PNG encoding
  - numpy          (BSD-3) for array ops

Catalogue used: Element84's public Earth Search v1
(https://earth-search.aws.element84.com/v1), no auth required, no
per-request rate limit on the search endpoint, COGs served from a free
public S3 bucket.

This module exists to replace `SimSatClient.fetch_location_images()`
without bringing the upstream SimSat code (AGPLv3) into the AquaVeritas
codebase. The public interface mirrors what `refresh_tiles.py` already
expects from `simsat.LocationImages`, so the call site just swaps the
import.

Band names use Earth Search v1's asset keys directly:
  RGB  composite:  red / green / blue        (true colour)
  SWIR composite:  swir16 / nir / red        (false colour — water vs land)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

# These imports are heavy (GDAL transitively). Defer them so importing
# this module from a unit test doesn't drag in the world.

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION       = "sentinel-2-l2a"
CLOUD_CEILING    = 60      # max % cloud cover we'll consider per scene
SEARCH_LIMIT     = 12      # how many candidate scenes to rank
TARGET_PX        = 768     # output PNG side length (3 sub-tiles × 256)


# ── Public interface (mirrors aquaveritas.simsat.LocationImages) ──────────────

@dataclass
class ImageResult:
    image: Optional[bytes]
    metadata: dict[str, Any] = field(default_factory=dict)
    available: bool = False


@dataclass
class LocationImages:
    rgb_core:    ImageResult
    swir_core:   ImageResult
    rgb_buffer:  ImageResult
    swir_buffer: ImageResult
    timestamp:   str
    location_id: str

    @property
    def any_core_available(self) -> bool:
        return self.rgb_core.available and self.swir_core.available

    @property
    def any_buffer_available(self) -> bool:
        return self.rgb_buffer.available and self.swir_buffer.available


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _bbox_from_centre(lon: float, lat: float, size_km: float) -> tuple[float, float, float, float]:
    """
    Return (west, south, east, north) WGS84 bbox of half-extent size_km/2
    around (lon, lat). Latitude → degrees uses 111.32 km / deg; longitude
    scales by cos(lat). Good enough for tiles ≤ 30km away from the equator,
    which covers all 20 monitored sites.
    """
    half_km = size_km / 2.0
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * max(0.1, np.cos(np.radians(lat))))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


# ── STAC search ───────────────────────────────────────────────────────────────

def _search_recent_scene(
    lon: float,
    lat: float,
    timestamp: str,
    window_seconds: int,
) -> Optional[Any]:
    """
    Return the least-cloudy Sentinel-2 L2A item that:
      • contains the (lon, lat) point
      • has acquisition date within `window_seconds` BEFORE `timestamp`
      • has cloud cover ≤ CLOUD_CEILING

    Returns None if no scene matches.
    """
    from pystac_client import Client  # local import — heavy

    end = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(seconds=window_seconds)
    dt_range = f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    client = Client.open(EARTH_SEARCH_URL)
    search = client.search(
        collections=[COLLECTION],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=dt_range,
        query={"eo:cloud_cover": {"lt": CLOUD_CEILING}},
        limit=SEARCH_LIMIT,
    )
    items = list(search.items())
    if not items:
        return None
    # Sort by cloud cover ascending — clearest first.
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items[0]


# ── Raster load + PNG encoding ────────────────────────────────────────────────

def _load_band_stack(
    item: Any,
    bands: list[str],
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Mosaic the requested bands from one STAC item over `bbox`. Returns a
    (H, W, 3) uint8 RGB array ready for PNG encoding.

    Sentinel-2 L2A surface reflectance is 0-10000 in the COGs. We clip at
    a per-channel high-percentile to stretch contrast (otherwise glare and
    cloud edges flatten everything), then linearly remap to 0-255.
    """
    from odc.stac import load  # local import — heavy

    ds = load(
        [item],
        bands=bands,
        bbox=bbox,
        resolution=max((bbox[2] - bbox[0]) / TARGET_PX, 0.0001),  # in degrees
        crs="EPSG:4326",
        chunks={},  # eager numpy, no dask
    )

    # ds is an xarray.Dataset; each band is a (time, y, x) DataArray.
    # We expect time=1 since we passed one item.
    channels = []
    for b in bands:
        arr = ds[b].isel(time=0).values.astype(np.float32)
        # Drop NaN/no-data sentinels to 0 before stretching.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # 98th-percentile clip — keeps highlights from blowing the lows.
        hi = float(np.percentile(arr[arr > 0], 98)) if (arr > 0).any() else 1.0
        if hi <= 0:
            hi = 1.0
        arr = np.clip(arr / hi, 0.0, 1.0)
        channels.append((arr * 255).astype(np.uint8))

    # Stack as (H, W, 3) — odc-stac gives us (y, x) in row-major already.
    return np.stack(channels, axis=-1)


def _array_to_png(arr: np.ndarray) -> bytes:
    from PIL import Image  # local import — heavy
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ── Public function ───────────────────────────────────────────────────────────

# Band constants — match aquaveritas.simsat for drop-in compatibility.
RGB_BANDS   = ["red", "green", "blue"]
SWIR_BANDS  = ["swir16", "nir", "red"]
CORE_KM     = 15.0
BUFFER_KM   = 15.0
DEFAULT_WINDOW_SECONDS = 2_592_000   # 30 days


def fetch_location_images(
    lon: float,
    lat: float,
    timestamp: str,
    location_id: str = "",
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> LocationImages:
    """
    Return the four images refresh_tiles.py needs (RGB+SWIR × core+buffer)
    for one site. Uses one STAC item — picks the least-cloudy match in
    the window — and renders the four composites from it.

    The core and buffer share a tile in our scheme (CORE_KM == BUFFER_KM)
    because each site's coordinate already sits at the water-body / land
    transition. If you split them in future, just call once per size_km.
    """
    bbox_core   = _bbox_from_centre(lon, lat, CORE_KM)
    bbox_buffer = _bbox_from_centre(lon, lat, BUFFER_KM)

    item = _search_recent_scene(lon, lat, timestamp, window_seconds)
    if item is None:
        empty = ImageResult(image=None, metadata={"error": "no scene matched"})
        return LocationImages(
            rgb_core=empty, swir_core=empty,
            rgb_buffer=empty, swir_buffer=empty,
            timestamp=timestamp, location_id=location_id,
        )

    cloud = float(item.properties.get("eo:cloud_cover", -1))
    acq   = item.properties.get("datetime", "")
    meta_common = {
        "image_available": True,
        "scene_id":        item.id,
        "acquisition_date": acq,
        "cloud_cover":     cloud,
    }

    def _shot(bands: list[str], bbox: tuple[float, float, float, float]) -> ImageResult:
        try:
            arr = _load_band_stack(item, bands, bbox)
            png = _array_to_png(arr)
            return ImageResult(image=png, metadata=meta_common, available=True)
        except Exception as exc:  # noqa: BLE001 — one band miss shouldn't sink the site
            return ImageResult(
                image=None,
                metadata={**meta_common, "error": f"{type(exc).__name__}: {exc}"},
                available=False,
            )

    return LocationImages(
        rgb_core   = _shot(RGB_BANDS,  bbox_core),
        swir_core  = _shot(SWIR_BANDS, bbox_core),
        rgb_buffer = _shot(RGB_BANDS,  bbox_buffer),
        swir_buffer= _shot(SWIR_BANDS, bbox_buffer),
        timestamp  = timestamp,
        location_id= location_id,
    )
