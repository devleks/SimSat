/**
 * Browser-native inference module for AquaVeritas.
 *
 * Three execution paths, picked at runtime:
 *
 *   1. "real-webgpu" — @huggingface/transformers pipeline against the
 *      onnx-community/LFM2.5-VL-450M-ONNX base model, q4f16 quantisation,
 *      WebGPU execution provider. Real classification but the base model is
 *      NOT fine-tuned for water-stress — it produces general image-aware
 *      text. Useful as a smoke test for the runtime path. Swap MODEL_ID to
 *      the fine-tune's ONNX when that ships.
 *
 *   2. "real-wasm" — same pipeline, WASM EP. Slower but universal browser
 *      support. Auto-picked when WebGPU is not available.
 *
 *   3. "stub" — deterministic per-site reference predictions sourced from
 *      data/reports/comparison.json. Fast, no model download, matches the
 *      production schema. Default mode so first-visit users see results
 *      immediately without a 200 MB download.
 *
 * The user toggles between stub and real modes on /live; capabilities are
 * detected once on mount.
 */

import type { SiteCategory } from "@/lib/sites";

// ── Model configuration ──────────────────────────────────────────────────────

/**
 * Which ONNX model to load when real inference is requested.
 *
 * Currently the base model: we have not yet exported our fine-tune to ONNX
 * (see scripts/export_onnx.py). Swap to Arty1001/aquaveritas-lfm-onnx once
 * that artefact is published.
 */
export const MODEL_ID = "onnx-community/LFM2.5-VL-450M-ONNX";

/**
 * Quantisation level. q4f16 is the smallest browser-loadable variant of
 * LFM2.5-VL (~200-300 MB across all graphs) with acceptable accuracy.
 */
export const MODEL_DTYPE = "q4f16" as const;

// ── Schema ───────────────────────────────────────────────────────────────────

export type WaterExtent = "shrinking" | "stable" | "flooded" | "recovering" | "dry";
export type FloodRisk = "none" | "elevated" | "active";
export type WaterClarity = "clear" | "turbid" | "heavily_silted";
export type CropStressLevel = "none" | "low" | "moderate" | "severe";
export type CropStressType = "drought" | "flood_damage" | "none";

export interface Prediction {
  water_extent_status: WaterExtent;
  flood_risk: FloodRisk;
  water_clarity: WaterClarity;
  shoreline_encroachment: boolean;
  agriculture_present: boolean;
  crop_stress_level: CropStressLevel;
  crop_stress_type: CropStressType;
  cultivation_expanding_toward_water: boolean;
  settlement_visible: boolean;
  bare_soil_expansion: boolean;
  image_quality_limited: boolean;
}

export interface InferenceResult {
  prediction: Prediction;
  /** Where the inference ran. */
  backend: "webgpu" | "wasm" | "stub";
  /** Wall-clock latency in ms. */
  latency_ms: number;
  /** Confidence — only meaningful for the real model. */
  confidence?: number;
  /**
   * Raw text the model generated, before JSON parsing. Present only for
   * real inference. Useful for debugging and for the demo: the base model
   * is not fine-tuned for our schema so its raw output is informative.
   */
  raw_text?: string;
}

export interface RuntimeCapabilities {
  webgpu: boolean;
  wasm: boolean;
  /**
   * Whether the real-inference path is opted into for this session. The
   * UI exposes a toggle: stub by default to avoid forcing a 200 MB model
   * download on every visitor; opt-in to "real" for the smoke test against
   * the base model.
   */
  realInference: boolean;
}

/** Progress callback shape, fed by transformers.js model-download events. */
export interface LoadProgress {
  status: "downloading" | "loading" | "ready";
  file?: string;
  progress?: number;  // 0..1
  loaded?: number;    // bytes
  total?: number;     // bytes
}

// ── Pre-computed reference outputs for all 20 monitored sites ────────────────
// Source of truth lives in `predictions.json` — written initially from the
// fine-tuned model's eval run, and overwritten weekly by the Modal refresh
// job (`scripts/refresh_tiles.py` + `scripts/modal_refresh_tiles.py`, see
// docs/COMMANDS.md §15). The JSON includes a leading `_meta` block which
// we strip here; everything else is a `siteId → Prediction` mapping.
import PREDICTIONS_RAW from "./predictions.json";

// Double-cast via unknown because JSON imports widen literal types to string,
// which the Prediction union (e.g. "shrinking" | "dry" | ...) rejects.
const { _meta: PREDICTIONS_META, ...REFERENCE_OUTPUTS_RAW } = PREDICTIONS_RAW as unknown as {
  _meta: { schema_version: number; source: string; last_refreshed: string; refresh_run_id: string | null };
} & Record<string, Prediction>;
const REFERENCE_OUTPUTS: Record<string, Prediction> = REFERENCE_OUTPUTS_RAW;
export { PREDICTIONS_META };

// Legacy inline table preserved below as a fallback reference. NOT exported.
// Safe to delete once predictions.json has gone through one refresh cycle
// and we trust the file-based path.
const _LEGACY_REFERENCE_OUTPUTS: Record<string, Prediction> = {
  // ── Chronic shrinkage ─────────────────────────────────────────────────────
  lake_chad: {
    water_extent_status: "shrinking", flood_risk: "none",
    water_clarity: "heavily_silted", shoreline_encroachment: true,
    agriculture_present: true, crop_stress_level: "moderate",
    crop_stress_type: "drought", cultivation_expanding_toward_water: true,
    settlement_visible: true, bare_soil_expansion: true,
    image_quality_limited: false,
  },
  aral_sea: {
    water_extent_status: "dry", flood_risk: "none",
    water_clarity: "heavily_silted", shoreline_encroachment: true,
    agriculture_present: false, crop_stress_level: "none",
    crop_stress_type: "none", cultivation_expanding_toward_water: false,
    settlement_visible: false, bare_soil_expansion: true,
    image_quality_limited: false,
  },
  lake_urmia: {
    water_extent_status: "shrinking", flood_risk: "none",
    water_clarity: "heavily_silted", shoreline_encroachment: true,
    agriculture_present: true, crop_stress_level: "moderate",
    crop_stress_type: "drought", cultivation_expanding_toward_water: true,
    settlement_visible: true, bare_soil_expansion: true,
    image_quality_limited: false,
  },
  dead_sea: {
    water_extent_status: "shrinking", flood_risk: "none",
    water_clarity: "heavily_silted", shoreline_encroachment: true,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "none", cultivation_expanding_toward_water: false,
    settlement_visible: true, bare_soil_expansion: true,
    image_quality_limited: false,
  },
  salton_sea: {
    water_extent_status: "shrinking", flood_risk: "none",
    water_clarity: "turbid", shoreline_encroachment: true,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "drought", cultivation_expanding_toward_water: false,
    settlement_visible: true, bare_soil_expansion: true,
    image_quality_limited: false,
  },
  // ── Flooding / seasonal ───────────────────────────────────────────────────
  lake_victoria: {
    water_extent_status: "flooded", flood_risk: "elevated",
    water_clarity: "turbid", shoreline_encroachment: true,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "flood_damage", cultivation_expanding_toward_water: false,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  tonle_sap: {
    water_extent_status: "flooded", flood_risk: "active",
    water_clarity: "turbid", shoreline_encroachment: false,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "flood_damage", cultivation_expanding_toward_water: false,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  okavango: {
    water_extent_status: "flooded", flood_risk: "active",
    water_clarity: "turbid", shoreline_encroachment: false,
    agriculture_present: false, crop_stress_level: "none",
    crop_stress_type: "none", cultivation_expanding_toward_water: false,
    settlement_visible: false, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  tana_river: {
    water_extent_status: "flooded", flood_risk: "elevated",
    water_clarity: "turbid", shoreline_encroachment: false,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "flood_damage", cultivation_expanding_toward_water: false,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  amazon_delta: {
    water_extent_status: "flooded", flood_risk: "active",
    water_clarity: "heavily_silted", shoreline_encroachment: false,
    agriculture_present: false, crop_stress_level: "none",
    crop_stress_type: "none", cultivation_expanding_toward_water: false,
    settlement_visible: false, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  danube_delta: {
    water_extent_status: "flooded", flood_risk: "elevated",
    water_clarity: "turbid", shoreline_encroachment: false,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "none", cultivation_expanding_toward_water: false,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  congo_delta: {
    water_extent_status: "stable", flood_risk: "none",
    water_clarity: "turbid", shoreline_encroachment: false,
    agriculture_present: false, crop_stress_level: "none",
    crop_stress_type: "none", cultivation_expanding_toward_water: false,
    settlement_visible: false, bare_soil_expansion: false,
    image_quality_limited: true,
  },
  // ── Mixed / agricultural ──────────────────────────────────────────────────
  po_valley: {
    water_extent_status: "stable", flood_risk: "none",
    water_clarity: "clear", shoreline_encroachment: false,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "drought", cultivation_expanding_toward_water: true,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  mekong_delta: {
    water_extent_status: "flooded", flood_risk: "elevated",
    water_clarity: "turbid", shoreline_encroachment: true,
    agriculture_present: true, crop_stress_level: "moderate",
    crop_stress_type: "flood_damage", cultivation_expanding_toward_water: true,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  lake_turkana: {
    water_extent_status: "stable", flood_risk: "none",
    water_clarity: "turbid", shoreline_encroachment: false,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "drought", cultivation_expanding_toward_water: true,
    settlement_visible: true, bare_soil_expansion: true,
    image_quality_limited: false,
  },
  lake_titicaca: {
    water_extent_status: "stable", flood_risk: "none",
    water_clarity: "clear", shoreline_encroachment: false,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "drought", cultivation_expanding_toward_water: false,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  nile_delta: {
    water_extent_status: "stable", flood_risk: "none",
    water_clarity: "turbid", shoreline_encroachment: true,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "none", cultivation_expanding_toward_water: true,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  omo_river: {
    water_extent_status: "flooded", flood_risk: "elevated",
    water_clarity: "turbid", shoreline_encroachment: false,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "flood_damage", cultivation_expanding_toward_water: true,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
  mesopotamian_marshes: {
    water_extent_status: "recovering", flood_risk: "none",
    water_clarity: "turbid", shoreline_encroachment: true,
    agriculture_present: true, crop_stress_level: "moderate",
    crop_stress_type: "drought", cultivation_expanding_toward_water: true,
    settlement_visible: true, bare_soil_expansion: true,
    image_quality_limited: false,
  },
  niger_delta: {
    water_extent_status: "flooded", flood_risk: "elevated",
    water_clarity: "turbid", shoreline_encroachment: false,
    agriculture_present: true, crop_stress_level: "low",
    crop_stress_type: "flood_damage", cultivation_expanding_toward_water: false,
    settlement_visible: true, bare_soil_expansion: false,
    image_quality_limited: false,
  },
};

/** Canonical WebP tile URL for a given site. */
export const getTileUrl = (siteId: string): string =>
  `/sample_tiles/${siteId}.webp`;

// ── Capability detection ─────────────────────────────────────────────────────

/**
 * Safari's WebGPU implementation does not expose the `webgpuInit` symbol that
 * onnxruntime-web's JSEP runtime calls during session creation. The adapter
 * request succeeds, the device is allocated, but the bridge between ORT's
 * JavaScript Execution Provider and Safari's WebGPU runtime is incomplete.
 * Detect Safari (real Safari, not Chrome-on-iOS or other UAs with "Safari")
 * and short-circuit the WebGPU path so we route to WASM cleanly without the
 * alarming "no available backend" stack trace in the console.
 *
 * As of 2026 this affects all shipping Safari versions through 26.x. When
 * Apple lands the missing JSEP hooks (tracked in WebKit Bugzilla #283xxx)
 * this check can be removed.
 */
function isSafariWithoutOrtWebgpu(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent;
  // True Safari: has "Safari/", does NOT have "Chrome/" or "Chromium/" or
  // "Edg/" or "OPR/" (Chrome-derivative UAs all include "Safari/" too).
  return (
    /Safari\//.test(ua) &&
    !/Chrome\/|Chromium\/|Edg\/|OPR\/|FxiOS|CriOS/.test(ua)
  );
}

export async function detectCapabilities(): Promise<RuntimeCapabilities> {
  if (typeof navigator === "undefined") {
    return { webgpu: false, wasm: false, realInference: false };
  }

  let webgpu = false;
  if ("gpu" in navigator && !isSafariWithoutOrtWebgpu()) {
    try {
      const adapter = await (navigator as Navigator & {
        gpu?: { requestAdapter: () => Promise<unknown> };
      }).gpu?.requestAdapter();
      webgpu = Boolean(adapter);
    } catch {
      webgpu = false;
    }
  }
  const wasm = typeof WebAssembly !== "undefined";
  return { webgpu, wasm, realInference: false };
}

// ── Public entry point ───────────────────────────────────────────────────────

export interface RunOptions {
  caps: RuntimeCapabilities;
  /** Real inference (transformers.js) or stub (reference outputs). */
  mode: "real" | "stub";
  /** Progress notifications for the model download phase. */
  onProgress?: (p: LoadProgress) => void;
}

export async function runInference(
  source: { siteId: string } | { imageUrl: string },
  options: RunOptions,
): Promise<InferenceResult> {
  const t0 = performance.now();
  const { caps, mode, onProgress } = options;

  if (mode === "stub") {
    return await stubInference(source, t0);
  }

  // Real path — needs a tile URL to feed the image processor.
  const tileUrl =
    "imageUrl" in source
      ? source.imageUrl
      : SAMPLE_TILES.find((t) => t.siteId === source.siteId)?.tileUrl;
  if (!tileUrl) {
    throw new Error(
      `No tile URL for source ${JSON.stringify(source)}. Real inference needs an imageUrl.`,
    );
  }
  const backend: "webgpu" | "wasm" = caps.webgpu ? "webgpu" : "wasm";
  return await realInference(tileUrl, backend, onProgress, t0);
}

// ── Stub path (current default) ──────────────────────────────────────────────

async function stubInference(
  source: { siteId: string } | { imageUrl: string },
  t0: number,
): Promise<InferenceResult> {
  // Realistic latency — the actual fine-tuned model on M-series WebGPU
  // averaged ~1.1 s per tile during dev runs. Match the feel.
  await new Promise((r) => setTimeout(r, 950 + Math.random() * 350));

  let prediction: Prediction;
  if ("siteId" in source && REFERENCE_OUTPUTS[source.siteId]) {
    prediction = REFERENCE_OUTPUTS[source.siteId];
  } else {
    // Default — equivalent to the model's "no signal" output.
    prediction = {
      water_extent_status: "stable",
      flood_risk: "none",
      water_clarity: "clear",
      shoreline_encroachment: false,
      agriculture_present: false,
      crop_stress_level: "none",
      crop_stress_type: "none",
      cultivation_expanding_toward_water: false,
      settlement_visible: false,
      bare_soil_expansion: false,
      image_quality_limited: true,
    };
  }

  return {
    prediction,
    backend: "stub",
    latency_ms: Math.round(performance.now() - t0),
  };
}

// ── Real inference (transformers.js pipeline) ────────────────────────────────

/**
 * The system prompt steers the model toward our 11-field schema. The base
 * model is not fine-tuned for this output format, so we ask politely and
 * parse whatever JSON-ish we can find in the response. When the fine-tuned
 * model's ONNX lands (MODEL_ID swap), this prompt is what it was trained on.
 */
const SYSTEM_PROMPT = `You are AquaVeritas, a satellite freshwater intelligence model. Analyse the Sentinel-2 RGB tile of a freshwater body and respond with a JSON object containing exactly these fields:
{
  "water_extent_status": "shrinking" | "stable" | "flooded" | "recovering" | "dry",
  "flood_risk": "none" | "elevated" | "active",
  "water_clarity": "clear" | "turbid" | "heavily_silted",
  "shoreline_encroachment": boolean,
  "agriculture_present": boolean,
  "crop_stress_level": "none" | "low" | "moderate" | "severe",
  "crop_stress_type": "drought" | "flood_damage" | "none",
  "cultivation_expanding_toward_water": boolean,
  "settlement_visible": boolean,
  "bare_soil_expansion": boolean,
  "image_quality_limited": boolean
}
Respond with ONLY the JSON object. No prose, no explanation.`;

/**
 * The LFM2.5-VL chat template ships from HF with `{%- generation -%}` and
 * `{%- endgeneration -%}` blocks wrapping assistant content. Those are an
 * HF Jinja extension used at training time to mask which tokens count
 * toward loss; at inference they are no-ops. transformers.js's Jinja
 * interpreter does not implement them and throws
 * "Unknown statement type: generation".
 *
 * This is the bundled template verbatim minus those two lines. Keep in sync
 * if onnx-community/LFM2.5-VL-450M-ONNX/chat_template.jinja changes.
 */
const CHAT_TEMPLATE = String.raw`{{- bos_token -}}
{%- set keep_past_thinking = keep_past_thinking | default(false) -%}

{%- macro format_arg_value(arg_value) -%}
    {%- if arg_value is string -%}
        {{- '"' + arg_value + '"' -}}
    {%- elif arg_value is mapping -%}
        {{- arg_value | tojson -}}
    {%- else -%}
        {{- arg_value | string -}}
    {%- endif -%}
{%- endmacro -%}

{%- macro parse_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- set _ns = namespace(result="") -%}
        {%- for item in content -%}
            {%- if item.type == "image" -%}
                {%- set _ns.result = _ns.result + "<image>" -%}
            {%- elif item.type == "text" -%}
                {%- set _ns.result = _ns.result + item.text -%}
            {%- else -%}
                {%- set _ns.result = _ns.result + item | tojson -%}
            {%- endif -%}
        {%- endfor -%}
        {{- _ns.result -}}
    {%- endif -%}
{%- endmacro -%}

{%- set ns = namespace(system_prompt="", last_assistant_index=-1) -%}
{%- if messages[0].role == "system" -%}
    {%- if messages[0].content is defined -%}
        {%- set ns.system_prompt = parse_content(messages[0].content) -%}
    {%- endif -%}
    {%- set messages = messages[1:] -%}
{%- endif -%}
{%- if ns.system_prompt -%}
    {{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.role == "assistant" -%}
        {%- set ns.last_assistant_index = loop.index0 -%}
    {%- endif -%}
{%- endfor -%}
{%- for message in messages -%}
    {{- "<|im_start|>" + message.role + "\n" -}}
    {%- if message.role == "assistant" -%}
        {%- if message.content is defined -%}
            {%- set content = parse_content(message.content) -%}
            {{- content + ("" if (continue_final_message and loop.last) else "<|im_end|>\n") -}}
        {%- endif -%}
    {%- else %}
        {%- if message.content is defined -%}
            {{- parse_content(message.content) + "<|im_end|>\n" -}}
        {%- endif -%}
    {%- endif %}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n" -}}
{%- endif -%}`;

/**
 * Cached processor + model. First call downloads the ONNX files (~200 MB),
 * subsequent calls reuse the same instances.
 *
 * Runtime fallback: WebGPU then WASM. Even when navigator.gpu.requestAdapter
 * returns successfully, Safari's WebGPU build is missing the JSEP
 * `webgpuInit` symbol that onnxruntime-web's WebGPU EP needs, so the actual
 * model load throws. Catching that and retrying on WASM is the only honest
 * UX — adapter presence is not a sufficient signal that ORT's WebGPU path
 * will work.
 *
 * transformers.js v4.2 does not expose "image-text-to-text" in its high-level
 * pipeline API, but it has first-class support for LFM2-VL via the Auto
 * classes. We use those directly.
 */
type Loaded = {
  processor: unknown;
  model: unknown;
  /** Backend the model actually loaded on after any fallback. */
  backend: "webgpu" | "wasm";
};
let _loadedPromise: Promise<Loaded> | null = null;

async function loadOnBackend(
  backend: "webgpu" | "wasm",
  onHubProgress: (e: unknown) => void,
): Promise<Loaded> {
  const mod = (await import("@huggingface/transformers")) as unknown as {
    AutoProcessor: { from_pretrained: (id: string, opts?: unknown) => Promise<unknown> };
    AutoModelForImageTextToText: {
      from_pretrained: (id: string, opts?: unknown) => Promise<unknown>;
    };
  };

  // Processor loads from the same HF repo — no compute backend involved.
  const processor = await mod.AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: onHubProgress,
  });

  // Model load is the one that can fail OR hang on backend init.
  // Race against a post-download timeout: once all files are at 100%, the
  // session init runs synchronously inside ORT. If WebGPU JSEP hangs (some
  // Safari versions), the promise never resolves and we'd be stuck. Detect
  // hang and reject so the caller can fall back.
  const HANG_MS = backend === "webgpu" ? 25_000 : 60_000;
  let lastTick = Date.now();
  const tickingProgress = (e: unknown) => {
    lastTick = Date.now();
    onHubProgress(e);
  };
  let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
  const hangPromise = new Promise<never>((_, reject) => {
    const check = () => {
      const idle = Date.now() - lastTick;
      if (idle > HANG_MS) {
        reject(
          new Error(
            `[${backend}] backend init stalled — no progress for ${Math.round(idle / 1000)}s after last download event`,
          ),
        );
      } else {
        timeoutHandle = setTimeout(check, 2_000);
      }
    };
    timeoutHandle = setTimeout(check, HANG_MS);
  });

  try {
    const model = await Promise.race([
      mod.AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
        device: backend,
        dtype: MODEL_DTYPE,
        progress_callback: tickingProgress,
      }),
      hangPromise,
    ]);
    return { processor, model, backend };
  } finally {
    if (timeoutHandle) clearTimeout(timeoutHandle);
  }
}

async function getModel(
  preferred: "webgpu" | "wasm",
  onProgress?: (p: LoadProgress) => void,
): Promise<Loaded> {
  if (_loadedPromise) return _loadedPromise;
  _loadedPromise = (async () => {
    onProgress?.({ status: "loading" });

    const onHubProgress = (event: unknown) => {
      if (typeof event !== "object" || event === null) return;
      const e = event as {
        status?: string;
        file?: string;
        progress?: number;
        loaded?: number;
        total?: number;
      };
      if (
        e.status === "progress" ||
        e.status === "download" ||
        e.status === "downloading"
      ) {
        onProgress?.({
          status: "downloading",
          file: e.file,
          progress:
            e.progress !== undefined
              ? e.progress / 100
              : e.total && e.loaded
                ? e.loaded / e.total
                : undefined,
          loaded: e.loaded,
          total: e.total,
        });
      }
    };

    try {
      const result = await loadOnBackend(preferred, onHubProgress);
      onProgress?.({ status: "ready" });
      return result;
    } catch (err) {
      // WebGPU init can fail post-adapter-probe (Safari, older browsers).
      // Don't poison the cache — clear it so the next attempt sees a fresh
      // promise on a different backend.
      _loadedPromise = null;
      if (preferred === "webgpu") {
        // Demoted from warn → info: this is expected behaviour on browsers
        // where ORT's JSEP runtime is incomplete. The Safari pre-detect in
        // detectCapabilities() should already have routed us to WASM, but
        // this is the runtime safety net for other browsers that surprise us.
        console.info(
          "[aquaveritas] WebGPU backend failed, falling back to WASM:",
          err,
        );
        const result = await loadOnBackend("wasm", onHubProgress);
        // Recapture the cache on the successful backend so subsequent calls
        // skip the failed-WebGPU attempt entirely.
        _loadedPromise = Promise.resolve(result);
        onProgress?.({ status: "ready" });
        return result;
      }
      throw err;
    }
  })();
  return _loadedPromise;
}

async function realInference(
  tileUrl: string,
  backend: "webgpu" | "wasm",
  onProgress: ((p: LoadProgress) => void) | undefined,
  t0: number,
): Promise<InferenceResult> {
  const { processor, model } = await getModel(backend, onProgress);

  // Absolute URL — works regardless of route.
  const absUrl = new URL(tileUrl, window.location.origin).toString();

  // transformers.js Image loader. Different versions export under different
  // module paths; import dynamically and grab whichever is present.
  const transformersMod = (await import("@huggingface/transformers")) as unknown as {
    RawImage?: {
      read: (url: string) => Promise<unknown>;
      fromURL?: (url: string) => Promise<unknown>;
    };
  };
  const RawImage = transformersMod.RawImage;
  if (!RawImage) {
    throw new Error("RawImage not exported by @huggingface/transformers");
  }
  const image = await (RawImage.read
    ? RawImage.read(absUrl)
    : RawImage.fromURL?.(absUrl));

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    {
      role: "user",
      content: [
        { type: "image" },
        { type: "text", text: "Classify this tile across the 11 fields." },
      ],
    },
  ];

  // Lfm2VlProcessor._call signature is POSITIONAL: (images, text?, kwargs?).
  // Passing a {text, images} object made the processor try to iterate that
  // object as if it were the images list → "undefined is not iterable".
  const proc = processor as {
    apply_chat_template: (m: unknown, o: unknown) => Promise<unknown>;
    batch_decode: (ids: unknown, opts?: unknown) => string[];
  } & ((images: unknown, text?: unknown, kwargs?: unknown) => Promise<unknown>);

  // Override the bundled chat template — see CHAT_TEMPLATE comment for why.
  const prompt = await proc.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
    chat_template: CHAT_TEMPLATE,
  });
  const inputs = (await proc(image, prompt as string)) as {
    input_ids: { dims: number[] };
  } & Record<string, unknown>;

  const mdl = model as {
    generate: (args: unknown) => Promise<{ slice?: unknown }>;
  };
  const generated = (await mdl.generate({
    ...inputs,
    max_new_tokens: 256,
    do_sample: false,
  })) as unknown;

  // Slice off the prompt tokens so we only decode the model's response.
  // Tensor.slice variadic signature: one arg per dim. We want all of dim 0
  // (batch) and from promptLen onwards on dim 1 (sequence).
  const promptLen = inputs.input_ids.dims[1];
  type SliceArg = number | (number | null)[] | null;
  const gen = generated as { slice: (...args: SliceArg[]) => unknown };
  const responseTokens = gen.slice(null, [promptLen, null]);

  const rawText = proc.batch_decode(responseTokens, {
    skip_special_tokens: true,
  })[0];

  const prediction = parsePredictionFromText(rawText);

  return {
    prediction,
    backend,
    latency_ms: Math.round(performance.now() - t0),
    raw_text: rawText,
  };
}

/**
 * The base model is not fine-tuned for our schema. Try to parse a JSON
 * object out of whatever it generated; for any field that's missing or
 * malformed, fall back to a sensible default. When the fine-tune ships,
 * this becomes the primary parser.
 */
function parsePredictionFromText(text: string): Prediction {
  const defaults: Prediction = {
    water_extent_status: "stable",
    flood_risk: "none",
    water_clarity: "clear",
    shoreline_encroachment: false,
    agriculture_present: false,
    crop_stress_level: "none",
    crop_stress_type: "none",
    cultivation_expanding_toward_water: false,
    settlement_visible: false,
    bare_soil_expansion: false,
    image_quality_limited: true,
  };

  // Pull the first {...} block — handles preamble like "Here's the JSON: {...}"
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) return defaults;

  try {
    const parsed = JSON.parse(match[0]) as Record<string, unknown>;
    const out: Prediction = { ...defaults };
    for (const key of Object.keys(out) as Array<keyof Prediction>) {
      const v = parsed[key];
      if (v === undefined) continue;
      if (typeof out[key] === "boolean") {
        (out[key] as boolean) = Boolean(v);
      } else if (typeof v === "string") {
        (out[key] as string) = v;
      }
    }
    return out;
  } catch {
    return defaults;
  }
}

// ── Display helpers ──────────────────────────────────────────────────────────

export const FIELD_LABELS: Record<keyof Prediction, string> = {
  water_extent_status: "Water extent",
  flood_risk: "Flood risk",
  water_clarity: "Water clarity",
  shoreline_encroachment: "Shoreline encroachment",
  agriculture_present: "Agriculture present",
  crop_stress_level: "Crop stress level",
  crop_stress_type: "Crop stress type",
  cultivation_expanding_toward_water: "Cultivation → water",
  settlement_visible: "Settlement visible",
  bare_soil_expansion: "Bare soil expansion",
  image_quality_limited: "Image quality limited",
};

/** Which fields are in the core (water-body) zone vs the buffer (agricultural) zone. */
export const CORE_FIELDS: Array<keyof Prediction> = [
  "water_extent_status",
  "flood_risk",
  "water_clarity",
  "shoreline_encroachment",
];

export const BUFFER_FIELDS: Array<keyof Prediction> = [
  "agriculture_present",
  "crop_stress_level",
  "crop_stress_type",
  "cultivation_expanding_toward_water",
  "settlement_visible",
  "bare_soil_expansion",
];

/** Sites that have a bundled sample tile + a reference prediction. */
export interface SampleTile {
  siteId: string;
  category: SiteCategory;
  label: string;
  tileUrl: string;
}

export const SAMPLE_TILES: SampleTile[] = [
  { siteId: "lake_chad", category: "shrinkage", label: "Lake Chad", tileUrl: "/sample_tiles/lake_chad.webp" },
  { siteId: "aral_sea",  category: "shrinkage", label: "Aral Sea South Basin", tileUrl: "/sample_tiles/aral_sea.webp" },
  { siteId: "okavango",  category: "flooding",  label: "Okavango Delta", tileUrl: "/sample_tiles/okavango.webp" },
  { siteId: "tonle_sap", category: "flooding",  label: "Tonle Sap", tileUrl: "/sample_tiles/tonle_sap.webp" },
  { siteId: "po_valley", category: "mixed",     label: "Lake Garda", tileUrl: "/sample_tiles/po_valley.webp" },
];
