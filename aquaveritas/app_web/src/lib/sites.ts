/**
 * The 20 monitored freshwater sites.
 *
 * Source of truth: aquaveritas/src/aquaveritas/locations.py
 * Keep in sync if the Python list changes. (Future: generate from a shared JSON.)
 *
 * Categories:
 *   shrinkage  — chronic loss of surface area (Aral, Lake Chad, Dead Sea, Urmia, Salton)
 *   flooding   — seasonal or active flooding regime (Victoria, Tonle Sap, deltas)
 *   mixed      — agricultural mixed-use, seasonal swings
 */
export type SiteCategory = "shrinkage" | "flooding" | "mixed";

export interface Site {
  id: string;
  name: string;
  lat: number;
  lon: number;
  category: SiteCategory;
  blurb?: string;
  /**
   * ISO date the bundled Sentinel-2 tile was captured. Must stay in sync
   * with the date column in scripts/copy_web_samples.py — the script is
   * the source of truth for which capture date each tile represents.
   */
  capturedAt?: string;
}

export const SITES: Site[] = [
  // ── Chronic shrinkage ─────────────────────────────────────────────────────
  {
    id: "lake_chad",
    name: "Lake Chad",
    lat: 12.95,
    lon: 14.25,
    category: "shrinkage",
    capturedAt: "2026-05-23",
    blurb:
      "Lost ~90% of its surface area since the 1960s. 30M people across Chad, Niger, Nigeria, and Cameroon depend on it.",
  },
  {
    id: "aral_sea",
    name: "Aral Sea South Basin",
    lat: 43.8,
    lon: 59.1,
    category: "shrinkage",
    capturedAt: "2026-05-23",
    blurb:
      "Once the world's 4th-largest lake. Diverted for Soviet cotton irrigation; southern basin nearly dry.",
  },
  {
    id: "lake_urmia",
    name: "Lake Urmia",
    lat: 37.6,
    lon: 46.0,
    category: "shrinkage",
    capturedAt: "2026-05-23",
    blurb:
      "Hypersaline lake, NW Iran. Lost ~80% volume since the 1970s due to upstream damming.",
  },
  {
    id: "dead_sea",
    name: "Dead Sea",
    lat: 31.8,
    lon: 35.55,
    category: "shrinkage",
    capturedAt: "2026-05-23",
    blurb:
      "Dropping ~1m/year. Jordan River diversion and mineral extraction. Lowest point on Earth.",
  },
  {
    id: "salton_sea",
    name: "Salton Sea",
    lat: 33.05,
    lon: -115.7,
    category: "shrinkage",
    capturedAt: "2026-05-23",
    blurb:
      "California's largest lake, accidentally created in 1905. Agricultural runoff source declining; shoreline retreats.",
  },
  // ── Flooding / seasonal ──────────────────────────────────────────────────
  {
    id: "lake_victoria",
    name: "Lake Victoria",
    lat: 0.05,
    lon: 34.2,
    category: "flooding",
    capturedAt: "2026-05-23",
    blurb:
      "Africa's largest lake. Recent flood episodes have submerged shoreline communities and infrastructure.",
  },
  {
    id: "tonle_sap",
    name: "Tonle Sap",
    lat: 13.05,
    lon: 103.85,
    category: "flooding",
    capturedAt: "2026-05-23",
    blurb:
      "Cambodia's life-source lake. Annual flood reversal of the Mekong tributary; fisheries collapsing.",
  },
  {
    id: "okavango",
    name: "Okavango Delta",
    lat: -19.8,
    lon: 23.1,
    category: "flooding",
    capturedAt: "2026-05-23",
    blurb:
      "Inland delta in Botswana. Pulse-flooded every dry season. UNESCO World Heritage site.",
  },
  {
    id: "tana_river",
    name: "Tana River Delta",
    lat: -2.55,
    lon: 40.52,
    category: "flooding",
    capturedAt: "2026-05-23",
    blurb:
      "Kenya's longest river. Delta supports pastoralist communities; large-scale irrigation pressure.",
  },
  {
    id: "amazon_delta",
    name: "Amazon Delta",
    lat: -0.8,
    lon: -50.2,
    category: "flooding",
    capturedAt: "2026-05-23",
    blurb:
      "Mouth of the Amazon. Tidal-influenced; deforestation upstream is changing sediment regimes.",
  },
  {
    id: "danube_delta",
    name: "Danube Delta",
    lat: 45.2,
    lon: 29.4,
    category: "flooding",
    capturedAt: "2026-05-23",
    blurb: "Europe's largest river delta. Romania/Ukraine border; protected wetland complex.",
  },
  {
    id: "congo_delta",
    name: "Lower Congo River",
    lat: -5.85,
    lon: 13.05,
    category: "flooding",
    capturedAt: "2026-05-23",
    blurb:
      "Lower Congo gorge near the Atlantic. Persistent cloud cover; one of the world's most data-poor large rivers.",
  },
  // ── Mixed / agricultural ─────────────────────────────────────────────────
  {
    id: "po_valley",
    name: "Lake Garda (Po Valley)",
    lat: 45.55,
    lon: 10.68,
    category: "mixed",
    capturedAt: "2026-05-23",
    blurb: "Italy's largest lake. Drought-stressed alpine fed; Po Valley irrigation downstream.",
  },
  {
    id: "mekong_delta",
    name: "Mekong Delta",
    lat: 10.05,
    lon: 105.65,
    category: "mixed",
    capturedAt: "2026-05-23",
    blurb:
      "Vietnam's rice bowl. Saltwater intrusion advancing 90km inland from upstream damming and sea-level rise.",
  },
  {
    id: "lake_turkana",
    name: "Lake Turkana",
    lat: 3.5,
    lon: 36.05,
    category: "mixed",
    capturedAt: "2026-05-23",
    blurb:
      "World's largest desert lake. Threatened by Ethiopia's Gibe III dam reducing Omo River inflow.",
  },
  {
    id: "lake_titicaca",
    name: "Lake Titicaca",
    lat: -15.85,
    lon: -70.02,
    category: "mixed",
    capturedAt: "2026-05-23",
    blurb:
      "Highest commercially navigable lake on Earth. Glacial-fed; receding ice and pollution stress.",
  },
  {
    id: "nile_delta",
    name: "Nile Delta",
    lat: 31.4,
    lon: 30.4,
    category: "mixed",
    capturedAt: "2026-05-23",
    blurb:
      "Source of Egypt's agriculture. Sea-level rise, subsidence, salinization; population pressure.",
  },
  {
    id: "omo_river",
    name: "Omo River Delta",
    lat: 4.6,
    lon: 36.15,
    category: "mixed",
    capturedAt: "2026-05-23",
    blurb:
      "Flows into Lake Turkana from Ethiopia. Gibe III dam altering downstream hydrology since 2015.",
  },
  {
    id: "mesopotamian_marshes",
    name: "Mesopotamian Marshes",
    lat: 30.9,
    lon: 47.4,
    category: "mixed",
    capturedAt: "2026-05-23",
    blurb:
      "Iraq's southern marshlands. Drained under Saddam Hussein; partial restoration ongoing.",
  },
  {
    id: "niger_delta",
    name: "Niger Delta",
    lat: 5.3,
    lon: 5.3,
    category: "mixed",
    capturedAt: "2026-05-23",
    blurb:
      "Nigeria's oil-producing delta. Mangrove loss, oil pollution, and population pressure.",
  },
];

export const CATEGORY_LABEL: Record<SiteCategory, string> = {
  shrinkage: "Chronic shrinkage",
  flooding: "Flooding / seasonal",
  mixed: "Mixed / agricultural",
};

/**
 * Marker colors tied to the brand palette (see globals.css).
 * Restrained: ochre is reserved for the loss category — the eye is drawn to
 * the same color used for "lost their lake" on the hero.
 *
 * Hex (not OKLCH): MapLibre's color parser only accepts hex/rgb/hsl. These
 * values are hex equivalents of the OKLCH brand tokens in globals.css:
 *   shrinkage  oklch(0.62 0.130 50)   → #CA7420  arid ochre, deepened
 *   flooding   oklch(0.50 0.090 220)  → #2C7190  deep ocean
 *   mixed      oklch(0.55 0.020 240)  → #7F8490  muted blue-grey
 */
export const CATEGORY_COLOR: Record<SiteCategory, string> = {
  shrinkage: "#CA7420",
  flooding: "#2C7190",
  mixed: "#7F8490",
};
