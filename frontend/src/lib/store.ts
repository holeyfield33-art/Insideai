"use client";

/**
 * Central simulation state. Every websocket event lands in `applyEvent`;
 * components subscribe to slices, and R3F animation loops read transiently
 * via useSimStore.getState() so per-frame work never re-renders React.
 */
import { create } from "zustand";

import { applyTheme, type ThemeName } from "./palette";
import type {
  LogitsData,
  ModelInfo,
  RunSummary,
  SampledToken,
  ServerEvent,
  Stage,
  TokenView,
  Zone,
} from "./types";

const THEME_KEY = "insideai-theme";

function initialTheme(): ThemeName {
  if (typeof window !== "undefined") {
    const saved = window.localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  }
  return "dark";
}

export interface AttentionData {
  layer: number;
  step: number;
  mean: number[][];
  lastRow: number[][];
  headEntropy: number[];
  heads: number[][][] | null;
}

export interface FfnData {
  layer: number;
  step: number;
  activations: number[];
  pooledFrom: number;
  meanAbs: number;
  max: number;
  activeFrac: number;
  inputPooled: number[] | null;
  outputPooled: number[] | null;
}

export interface LayerMeta {
  hiddenNorm: number;
  residualDelta: number;
}

export interface AnomalyPoint {
  step: number;
  layer: number;
  zetaProxy: number;
  flagged: boolean;
}

export interface GenerationParamsUI {
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  chat: "auto" | "raw";
  seed: string; // free text; "" = random
  speed: number;
}

export const DEFAULT_PARAMS: GenerationParamsUI = {
  max_new_tokens: 48,
  temperature: 0.8,
  top_k: 40,
  top_p: 0.95,
  repetition_penalty: 1.15,
  chat: "auto",
  seed: "",
  speed: 0.4,
};

interface SimState {
  connection: "connecting" | "open" | "closed";
  modelInfo: ModelInfo | null;
  /** Whether the backend has a live model (false = replay-only mode). null
      until the server_mode handshake arrives. */
  liveModel: boolean | null;

  running: boolean;
  /** The currently-shown run is a recorded replay, not a live generation.
      Set the moment the client requests a replay; cleared when a live run
      starts. Drives the unmistakable REPLAY indicator. */
  replaying: boolean;
  /** Recorded runs available for replay (GET /runs). */
  runs: RunSummary[];
  stage: Stage;
  step: number;
  activeLayer: number; // -1 when not inside the layer stack

  prompt: string;
  tokens: TokenView[];
  points: [number, number, number][];
  pointNorms: number[];
  positional: { norms: number[]; sample: number[][] } | null;

  attention: (AttentionData | null)[];
  ffn: (FfnData | null)[];
  layerMeta: (LayerMeta | null)[];
  /** Latest zeta_proxy reading per layer, for the layer-chip row. */
  anomalyLatest: (AnomalyPoint | null)[];
  /** Rolling window across all layers/steps, oldest first — the EKG trace.
      Capped so a long generation doesn't grow this unbounded. */
  anomalyHistory: AnomalyPoint[];

  logits: LogitsData | null;
  sampled: SampledToken | null;
  /** Sampled token of the in-flight step, cleared once step_end confirms it. */
  pendingSample: SampledToken | null;
  generated: SampledToken[];
  text: string;
  endReason: string | null;
  stats: { computeMs: number | null; durationS: number | null; tokensPerS: number | null };
  error: string | null;

  // UI state
  params: GenerationParamsUI;
  selectedLayer: number;
  selectedHead: number; // -1 = head mean
  focusZone: Zone;
  /** Bumped on every zone click so re-clicking the same zone re-centers. */
  focusNonce: number;
  cinematic: boolean;
  theme: ThemeName;

  // actions
  applyEvent: (ev: ServerEvent) => void;
  setConnection: (c: SimState["connection"]) => void;
  setReplaying: (on: boolean) => void;
  setRuns: (runs: RunSummary[]) => void;
  setParams: (p: Partial<GenerationParamsUI>) => void;
  setSelectedLayer: (l: number) => void;
  setSelectedHead: (h: number) => void;
  setFocusZone: (z: Zone) => void;
  setCinematic: (on: boolean) => void;
  setTheme: (t: ThemeName) => void;
}

const emptyLayers = <T,>(n: number): (T | null)[] => Array.from({ length: n }, () => null);

const startTheme = initialTheme();
applyTheme(startTheme);

export const useSimStore = create<SimState>((set, get) => ({
  connection: "connecting",
  modelInfo: null,
  liveModel: null,

  running: false,
  replaying: false,
  runs: [],
  stage: "idle",
  step: 0,
  activeLayer: -1,

  prompt: "",
  tokens: [],
  points: [],
  pointNorms: [],
  positional: null,

  attention: [],
  ffn: [],
  layerMeta: [],
  anomalyLatest: [],
  anomalyHistory: [],

  logits: null,
  sampled: null,
  pendingSample: null,
  generated: [],
  text: "",
  endReason: null,
  stats: { computeMs: null, durationS: null, tokensPerS: null },
  error: null,

  params: DEFAULT_PARAMS,
  selectedLayer: 0,
  selectedHead: -1,
  focusZone: "overview",
  focusNonce: 0,
  cinematic: true,
  theme: startTheme,

  applyEvent: (ev) => {
    switch (ev.type) {
      case "server_mode":
        set({ liveModel: ev.live });
        break;

      case "model_info": {
        const { type: _t, ...info } = ev;
        set({
          modelInfo: info,
          attention: emptyLayers(info.n_layer),
          ffn: emptyLayers(info.n_layer),
          layerMeta: emptyLayers(info.n_layer),
          anomalyLatest: emptyLayers(info.n_layer),
          anomalyHistory: [],
          // Late layers are where the "meaning" work shows most clearly.
          selectedLayer: Math.max(0, info.n_layer - 1),
        });
        break;
      }

      case "generation_start": {
        const n = get().modelInfo?.n_layer ?? 0;
        set({
          running: true,
          error: null,
          endReason: null,
          prompt: ev.prompt,
          stage: "tokenize",
          step: 0,
          activeLayer: -1,
          generated: [],
          text: "",
          sampled: null,
          pendingSample: null,
          logits: null,
          attention: emptyLayers(n),
          ffn: emptyLayers(n),
          layerMeta: emptyLayers(n),
          anomalyLatest: emptyLayers(n),
          anomalyHistory: [],
          stats: { computeMs: null, durationS: null, tokensPerS: null },
        });
        break;
      }

      case "tokenize":
        set({ tokens: ev.tokens, stage: "tokenize", step: ev.step });
        break;

      case "embeddings":
        set({ points: ev.points, pointNorms: ev.norms, stage: "embeddings" });
        break;

      case "positional":
        set({ positional: { norms: ev.norms, sample: ev.sample }, stage: "positional" });
        break;

      case "layer_start":
        // The tower follows the live sweep via activeLayer; the heatmap,
        // arcs and neuron diagram stay on the user's selectedLayer so they
        // update once per token instead of flickering 24x per token.
        set({ activeLayer: ev.layer, stage: "layers" });
        break;

      case "attention": {
        const attention = get().attention.slice();
        if (ev.mode === "full") {
          attention[ev.layer] = {
            layer: ev.layer,
            step: ev.step,
            mean: ev.mean,
            lastRow: ev.last_row,
            headEntropy: ev.head_entropy_bits,
            heads: ev.heads,
          };
        } else {
          // Decode row: append to the accumulated causal matrix. Earlier rows
          // are shorter by exactly their causal reach, so the ragged matrix
          // stays exact (missing cells are true zeros).
          const prev = attention[ev.layer];
          if (!prev) break;
          attention[ev.layer] = {
            layer: ev.layer,
            step: ev.step,
            mean: [...prev.mean, ev.mean_row],
            lastRow: ev.head_rows,
            headEntropy: ev.head_entropy_bits,
            heads: prev.heads ? prev.heads.map((m, h) => [...m, ev.head_rows[h]]) : null,
          };
        }
        set({ attention });
        break;
      }

      case "ffn": {
        const ffn = get().ffn.slice();
        ffn[ev.layer] = {
          layer: ev.layer,
          step: ev.step,
          activations: ev.activations,
          pooledFrom: ev.pooled_from,
          meanAbs: ev.mean_abs,
          max: ev.max,
          activeFrac: ev.active_frac,
          inputPooled: ev.input_pooled ?? null,
          outputPooled: ev.output_pooled ?? null,
        };
        set({ ffn });
        break;
      }

      case "layer_end": {
        const layerMeta = get().layerMeta.slice();
        layerMeta[ev.layer] = { hiddenNorm: ev.hidden_norm, residualDelta: ev.residual_delta };
        set({ layerMeta });
        break;
      }

      case "anomaly": {
        const point: AnomalyPoint = {
          step: ev.step,
          layer: ev.layer,
          zetaProxy: ev.zeta_proxy,
          flagged: ev.flagged,
        };
        const anomalyLatest = get().anomalyLatest.slice();
        anomalyLatest[ev.layer] = point;
        const HISTORY_CAP = 600; // ~ a few full generations of layer ticks
        const anomalyHistory = [...get().anomalyHistory, point].slice(-HISTORY_CAP);
        set({ anomalyLatest, anomalyHistory });
        break;
      }

      case "logits": {
        const { type: _t, step: _s, ...logits } = ev;
        // A fresh distribution invalidates the previous step's sampled marker.
        set({ logits, stage: "logits", activeLayer: -1, sampled: null });
        break;
      }

      case "sampled": {
        const { type: _t, step: _s, ...sampled } = ev;
        set({ sampled, pendingSample: sampled, stage: "sampled" });
        break;
      }

      case "step_end":
        set({
          pendingSample: null,
          generated: [...get().generated, ev.token],
          text: ev.text,
          step: ev.step + 1,
          stage: "streaming",
          stats: { ...get().stats, computeMs: ev.compute_ms },
        });
        break;

      case "generation_end":
        set({
          running: false,
          stage: "done",
          text: ev.text,
          endReason: ev.reason,
          activeLayer: -1,
          stats: {
            ...get().stats,
            durationS: ev.duration_s,
            tokensPerS: ev.tokens_per_s,
          },
        });
        break;

      case "error":
        set({ error: ev.message, running: false, stage: "error", activeLayer: -1 });
        break;

      case "pong":
        break;
    }
  },

  setConnection: (connection) => set({ connection }),
  setReplaying: (replaying) => set({ replaying }),
  setRuns: (runs) => set({ runs }),
  setParams: (p) => set({ params: { ...get().params, ...p } }),
  setSelectedLayer: (selectedLayer) => set({ selectedLayer }),
  setSelectedHead: (selectedHead) => set({ selectedHead }),
  setFocusZone: (focusZone) =>
    set({ focusZone, cinematic: false, focusNonce: get().focusNonce + 1 }),
  setCinematic: (cinematic) => set({ cinematic }),
  setTheme: (theme) => {
    applyTheme(theme);
    if (typeof window !== "undefined") window.localStorage.setItem(THEME_KEY, theme);
    set({ theme });
  },
}));
