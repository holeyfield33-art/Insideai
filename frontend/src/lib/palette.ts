/**
 * Educational dark palette — "chalkboard diagram" style.
 *
 * Validated with the dataviz six-checks validator against the card surface
 * (#131722, dark mode):
 *   - identity pair green #2aa35d (input/prompt) + red #e0564f
 *     (output/generated) — ALL CHECKS PASS, worst CVD ΔE 14.4 (deutan),
 *     with INPUT/OUTPUT section labels as secondary encoding
 *   - accent blue #4d84ea (single-series marks) — ALL CHECKS PASS
 *   - amber #cc7d0a ("processing now" state, always with a label) — PASS
 * Sequential ramps are single-hue with monotone lightness, dark-anchored
 * (near-surface dark = 0, bright = 1) for the dark surface.
 */
import { scaleLinear } from "d3-scale";

export const palette = {
  page: "#0b0e17",
  surface: "#131722",
  raised: "#1c2333",
  ink: "#e9edf9",
  ink2: "#a9b3ce",
  ink3: "#6e7893",
  grid: "#29314a",
  border: "rgba(255,255,255,0.1)",

  /** Single-series marks: probability bars, meters, UI accent. */
  accent: "#4d84ea",
  /** Meter track = dark step of the accent's own ramp. */
  accentTrack: "#21315a",

  /** Identity pair (validated): input/prompt vs output/generated. */
  promptToken: "#2aa35d",
  generatedToken: "#e0564f",

  /** "This is computing right now" highlight — always labeled. */
  active: "#cc7d0a",

  danger: "#e0564f",
} as const;

/** Sequential blue ramp — attention weights (0 → 1, dark → bright). */
export const ATTN_RAMP = [
  "#161b2e",
  "#1d2a4c",
  "#243d74",
  "#2c53a1",
  "#3a6cc9",
  "#5b89e6",
  "#86abf1",
  "#b6d0f9",
  "#e4eefd",
];

/** Second sequential context — activation magnitude (teal, dark → bright). */
export const ACT_RAMP = [
  "#13211c",
  "#183a2f",
  "#1e5343",
  "#256d56",
  "#2e8869",
  "#48a37f",
  "#73bd9c",
  "#a6d7c0",
  "#def3e8",
];

function rampScale(ramp: string[]) {
  return scaleLinear<string>()
    .domain(ramp.map((_, i) => i / (ramp.length - 1)))
    .range(ramp)
    .clamp(true);
}

export const attnScale = rampScale(ATTN_RAMP);
export const actScale = rampScale(ACT_RAMP);

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

/** Precomputed 256-entry RGB lookup table for fast canvas heatmaps. */
export function rampLUT(ramp: string[], n = 256): Uint8ClampedArray {
  const stops = ramp.map(hexToRgb);
  const lut = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (stops.length - 1);
    const seg = Math.min(Math.floor(t), stops.length - 2);
    const f = t - seg;
    for (let c = 0; c < 3; c++) {
      lut[i * 3 + c] = stops[seg][c] + (stops[seg + 1][c] - stops[seg][c]) * f;
    }
  }
  return lut;
}

export const attnLUT = rampLUT(ATTN_RAMP);

/** Scene styling (3D diagram look, not chart marks). */
export const scene = {
  background: "#0b0e17",
  nodeRim: "#dfe5f5", // light outline around every node — chalkboard diagram
  edge: "#86abf1", // strong connection (bright on dark)
  edgeFaint: "#2b3552", // weak connection
  slabFill: "#1a2135",
  slabEdge: "#8ea0c6",
  beam: "#41547f",
  axis: "#5d6890",
} as const;
