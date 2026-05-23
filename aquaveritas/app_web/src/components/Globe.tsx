"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl, {
  GeoJSONSource,
  Map as MLMap,
  MapLayerMouseEvent,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  SITES,
  CATEGORY_COLOR,
  CATEGORY_LABEL,
  type SiteCategory,
} from "@/lib/sites";

/**
 * Globe — MapLibre GL with native globe projection (v5+).
 *
 * Controlled component: parent owns `selectedId`. We emit changes through
 * `onSelectChange`. The detail / inference panel is rendered by the parent
 * (/globe page) using GlobeInferencePanel.
 *
 * Basemap: NASA EOSDIS GIBS — VIIRS_SNPP_CorrectedReflectance_TrueColor
 * (daily global imagery, ~250m, no auth, CORS-friendly). Genuinely "live":
 * the URL is date-parameterized and we request today-2-days to guarantee
 * the global mosaic is fully processed. See docs/LIVE_DATA_OPTIONS.md for
 * the full provider comparison and why GIBS won.
 *
 * Upgrade paths (one URL swap each):
 *   - EOX s2cloudless     — crisper Sentinel-2 10m, static yearly composite
 *   - Sentinel Hub WMS    — live Sentinel-2 10m, requires free account
 *
 * Container sizing note: the inner ref div uses explicit `h-full w-full`
 * (NOT `absolute inset-0`). MapLibre adds `.maplibregl-map { position:
 * relative }` to the container, silently overriding position:absolute and
 * collapsing the element to height:0 if there's no explicit height. Took a
 * long debug session to find this; do not "simplify" back to absolute
 * positioning without re-verifying tile rendering.
 */

/**
 * Date string for the GIBS VIIRS tile URL. Two-day lag guarantees the
 * global mosaic is fully processed (VIIRS has ~3h latency plus processing).
 */
function getLiveBasemapDate(): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - 2);
  return d.toISOString().slice(0, 10);
}

export interface GlobeProps {
  selectedId: string | null;
  onSelectChange: (id: string | null) => void;
}

export default function Globe({ selectedId, onSelectChange }: GlobeProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [styleLoaded, setStyleLoaded] = useState(false);
  const [fatal, setFatal] = useState<string | null>(null);

  const allCategories = useMemo(
    () => Object.keys(CATEGORY_LABEL) as SiteCategory[],
    [],
  );
  const [active, setActive] = useState<Set<SiteCategory>>(
    () => new Set(allCategories),
  );

  const onSelectRef = useRef(onSelectChange);
  useEffect(() => {
    onSelectRef.current = onSelectChange;
  }, [onSelectChange]);

  // ── Init the map ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    // Hard precheck: WebGL must be available. Safari Private mode and some
    // privacy extensions block the WebGL context entirely; MapLibre then
    // fails silently. Catching it ourselves means we can show the user a
    // clear explanation rather than a blank canvas.
    const testCanvas = document.createElement("canvas");
    const hasWebGL = Boolean(
      testCanvas.getContext("webgl2") ||
        testCanvas.getContext("webgl") ||
        (testCanvas.getContext("experimental-webgl") as RenderingContext | null),
    );
    if (!hasWebGL) {
      setFatal(
        "WebGL is not available in this browser session. This commonly happens in Safari Private windows or with strict tracking-prevention. Switch to a normal window (or Chrome / Brave / Edge) to see the globe.",
      );
      return;
    }

    const liveDate = getLiveBasemapDate();
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          viirs: {
            type: "raster",
            tiles: [
              `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_SNPP_CorrectedReflectance_TrueColor/default/${liveDate}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`,
            ],
            tileSize: 256,
            // VIIRS Level9 maxzoom = 9. Beyond that we'd need a different
            // GIBS layer or a higher-resolution provider.
            maxzoom: 9,
            attribution: `NASA EOSDIS GIBS · VIIRS SNPP True Color · ${liveDate}`,
          },
        },
        layers: [{ id: "viirs", type: "raster", source: "viirs" }],
        glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
      },
      center: [14.25, 12.95], // Lake Chad
      zoom: 1.5,
      pitch: 0,
      bearing: 0,
      attributionControl: { compact: true },
    });
    mapRef.current = map;

    map.on("error", (e) => {
      console.error("[globe] maplibre error", e?.error ?? e);
    });

    map.on("load", () => {
      // Globe projection — sphere rendering for the orbital aesthetic.
      try {
        map.setProjection({ type: "globe" });
      } catch {
        /* mercator fallback on older clients */
      }
      // Belt-and-suspenders resize after first frame.
      requestAnimationFrame(() => map.resize());
    });

    map.on("style.load", () => {
      try {
        map.addSource("sites", {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
        // Dot styling adapted for the bright VIIRS true-color basemap:
        // - Dark soft shadow underneath for contrast against bright land
        // - Slightly larger dot than the CARTO version
        // - Thicker white stroke so the colour reads against any backdrop
        map.addLayer({
          id: "sites-halo",
          type: "circle",
          source: "sites",
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 10, 5, 22],
            "circle-color": "#000000",
            "circle-opacity": 0.35,
            "circle-blur": 0.5,
            "circle-stroke-width": 0,
          },
        });
        map.addLayer({
          id: "sites-dot",
          type: "circle",
          source: "sites",
          paint: {
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 1, 5, 5, 9],
            "circle-color": ["get", "color"],
            "circle-stroke-color": "#FFFFFF",
            "circle-stroke-width": 2,
            "circle-stroke-opacity": 1,
          },
        });

        map.on("mouseenter", "sites-dot", () => {
          map.getCanvas().style.cursor = "pointer";
        });
        map.on("mouseleave", "sites-dot", () => {
          map.getCanvas().style.cursor = "";
        });
        map.on("click", "sites-dot", (e: MapLayerMouseEvent) => {
          const f = e.features?.[0];
          if (!f) return;
          const id = (f.properties as { id: string }).id;
          onSelectRef.current(id);
        });
      } catch (err) {
        console.error("[globe] failed adding sources/layers:", err);
      }

      setStyleLoaded(true);
    });

    // ResizeObserver — MapLibre needs to know if its container resizes
    // (font load, parent flex-grow, viewport rotation). Without this the
    // canvas stays at its first-measure size and lags on layout shifts.
    const ro = new ResizeObserver(() => {
      map.resize();
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      map.remove();
      mapRef.current = null;
      setStyleLoaded(false);
    };
  }, []);

  // ── Push filtered features into the source ───────────────────────────────
  useEffect(() => {
    if (!styleLoaded || !mapRef.current) return;
    const features = SITES.filter((s) => active.has(s.category)).map((s) => ({
      type: "Feature" as const,
      geometry: { type: "Point" as const, coordinates: [s.lon, s.lat] },
      properties: {
        id: s.id,
        name: s.name,
        category: s.category,
        color: CATEGORY_COLOR[s.category],
      },
    }));
    const source = mapRef.current.getSource("sites") as GeoJSONSource | undefined;
    source?.setData({ type: "FeatureCollection", features });

    // If the currently-selected site got filtered out, clear it upstream.
    if (selectedId) {
      const stillVisible = features.some((f) => f.properties.id === selectedId);
      if (!stillVisible) onSelectRef.current(null);
    }
  }, [active, styleLoaded, selectedId]);

  // ── External selection (e.g. clicked from site index) → fly to it ────────
  useEffect(() => {
    if (!styleLoaded || !mapRef.current || !selectedId) return;
    const site = SITES.find((s) => s.id === selectedId);
    if (!site) return;
    mapRef.current.flyTo({
      center: [site.lon, site.lat],
      zoom: Math.max(mapRef.current.getZoom(), 3.2),
      speed: 0.7,
      curve: 1.4,
      essential: true,
    });
  }, [selectedId, styleLoaded]);

  const toggleCategory = useCallback(
    (cat: SiteCategory) => {
      setActive((prev) => {
        const next = new Set(prev);
        if (next.has(cat)) {
          if (next.size === 1) return new Set(allCategories);
          next.delete(cat);
        } else {
          next.add(cat);
        }
        return next;
      });
    },
    [allCategories],
  );

  const categoryCounts = useMemo(() => {
    const counts: Record<SiteCategory, number> = {
      shrinkage: 0,
      flooding: 0,
      mixed: 0,
    };
    for (const s of SITES) counts[s.category] += 1;
    return counts;
  }, []);

  return (
    <div className="relative h-full w-full overflow-hidden">
      {/*
        Container MUST use explicit h-full w-full, NOT absolute inset-0.
        See module comment above for why. The brand surface-alt background
        is what shows for the brief moment before tiles paint.
      */}
      <div
        ref={containerRef}
        className="h-full w-full bg-[color:var(--color-surface-alt)]"
      />

      {/* Fatal-error overlay — visible explanation when WebGL is blocked. */}
      {fatal && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-[color:var(--color-surface)]/95 p-12">
          <div className="max-w-md text-center">
            <p className="text-xs uppercase tracking-[0.18em] text-[color:var(--color-ochre-deep)]">
              Map unavailable in this browser
            </p>
            <p className="mt-3 text-sm leading-relaxed text-[color:var(--color-ink)]">
              {fatal}
            </p>
            <p className="mt-6 text-xs text-[color:var(--color-ink-muted)]">
              The site index below still works — click any &ldquo;Run
              inference →&rdquo; row to see the model&rsquo;s assessment
              of that site.
            </p>
          </div>
        </div>
      )}

      {/* Legend / category filter */}
      <div className="pointer-events-none absolute left-6 top-6 z-10 flex flex-col gap-2 rounded-sm border border-[color:var(--color-rule)] bg-[color:var(--color-surface)]/95 p-4 text-xs shadow-sm backdrop-blur-sm">
        <p className="font-medium uppercase tracking-[0.18em] text-[color:var(--color-ink-faint)]">
          Click to filter
        </p>
        {allCategories.map((k) => {
          const isOn = active.has(k);
          return (
            <button
              key={k}
              type="button"
              onClick={() => toggleCategory(k)}
              className={`pointer-events-auto flex items-center gap-2 text-left transition-opacity duration-300 ${
                isOn ? "opacity-100" : "opacity-30 hover:opacity-60"
              }`}
              style={{ transitionTimingFunction: "var(--ease-out-quint)" }}
              aria-pressed={isOn}
            >
              <span
                aria-hidden
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ background: CATEGORY_COLOR[k] }}
              />
              <span className="text-[color:var(--color-ink-muted)]">
                {CATEGORY_LABEL[k]}
              </span>
              <span className="ml-auto pl-3 font-[family-name:var(--font-mono)] text-[10px] text-[color:var(--color-ink-faint)]">
                {categoryCounts[k]}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
