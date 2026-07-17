/**
 * Two validated educational palettes — dark ("chalkboard") and light
 * ("textbook") — same semantic color language in both:
 * green = input, blue = processing, red = output, amber = computing now.
 *
 * Both were checked with the dataviz six-checks validator against their card
 * surface (CVD ΔE of the green/red identity pair ≥ 12 under protan/deutan,
 * ≥ 3:1 contrast, lightness bands, chroma floor):
 *   dark  (#131722): #2aa35d / #e0564f (ΔE 14.4) · accent #4d84ea · amber #cc7d0a
 *   light (#ffffff): #0f8a3d / #d94343 (ΔE 12.9) · accent #2a5bd7 · amber #d97706
 * Sequential ramps are single-hue, monotone lightness, anchored at the
 * surface (dark-anchored on dark, light-anchored on light).
 *
 * Exports are live bindings switched by applyTheme(); the 3D scene remounts
 * on theme change so every material picks up the new values.
 */
import { scaleLinear } from "d3-scale";

export type ThemeName = "dark" | "light";

export interface Palette {
  page: string;
  surface: string;
  raised: string;
  ink: string;
  ink2: string;
  ink3: string;
  grid: string;
  accent: string;
  accentTrack: string;
  promptToken: string;
  generatedToken: string;
  active: string;
  danger: string;
}

export interface SceneStyle {
  background: string;
  nodeRim: string;
  edge: string;
  edgeFaint: string;
  slabFill: string;
  slabEdge: string;
  towerBlue: string;
  beam: string;
  axis: string;
  dropline: string;
  gridMajor: string;
  gridMinor: string;
  ambient: number;
  directional: number;
}

const PALETTES: Record<ThemeName, Palette> = {
  dark: {
    page: "#0b0e17",
    surface: "#131722",
    raised: "#1c2333",
    ink: "#e9edf9",
    ink2: "#a9b3ce",
    ink3: "#6e7893",
    grid: "#29314a",
    accent: "#4d84ea",
    accentTrack: "#21315a",
    promptToken: "#2aa35d",
    generatedToken: "#e0564f",
    active: "#cc7d0a",
    danger: "#e0564f",
  },
  light: {
    page: "#f4f6fb",
    surface: "#ffffff",
    raised: "#f1f4fa",
    ink: "#17203a",
    ink2: "#4c5670",
    ink3: "#8a93ab",
    grid: "#dfe4f0",
    accent: "#2a5bd7",
    accentTrack: "#dbe5fb",
    promptToken: "#0f8a3d",
    generatedToken: "#d94343",
    active: "#d97706",
    danger: "#c02626",
  },
};

const SCENES: Record<ThemeName, SceneStyle> = {
  dark: {
    background: "#0b0e17",
    nodeRim: "#dfe5f5",
    edge: "#86abf1",
    edgeFaint: "#2b3552",
    slabFill: "#1a2135",
    slabEdge: "#8ea0c6",
    towerBlue: "#5b89e6",
    beam: "#41547f",
    axis: "#5d6890",
    dropline: "#3a4663",
    gridMajor: "#232b42",
    gridMinor: "#151b2a",
    ambient: 0.85,
    directional: 0.5,
  },
  light: {
    background: "#f6f8fc",
    nodeRim: "#26304f",
    edge: "#2f3e9e",
    edgeFaint: "#ccd3ea",
    slabFill: "#e9edf9",
    slabEdge: "#3a4468",
    towerBlue: "#4a7ade",
    beam: "#9db1e8",
    axis: "#8a93ab",
    dropline: "#aeb9d6",
    gridMajor: "#ccd5ea",
    gridMinor: "#e4e9f5",
    ambient: 1.05,
    directional: 0.65,
  },
};

const RAMPS: Record<ThemeName, { attn: string[]; act: string[] }> = {
  // dark-anchored: near-surface dark = 0, bright = 1
  dark: {
    attn: ["#161b2e", "#1d2a4c", "#243d74", "#2c53a1", "#3a6cc9", "#5b89e6", "#86abf1", "#b6d0f9", "#e4eefd"],
    act: ["#13211c", "#183a2f", "#1e5343", "#256d56", "#2e8869", "#48a37f", "#73bd9c", "#a6d7c0", "#def3e8"],
  },
  // light-anchored: near-white = 0, dark = 1
  light: {
    attn: ["#f3f7fe", "#dfe9fc", "#c2d4f9", "#9dbaf3", "#749aea", "#4e7bdf", "#2a5bd7", "#1d44a9", "#132f78"],
    act: ["#effbf7", "#d2f1e6", "#ace2d0", "#7fceb5", "#52b497", "#2c977b", "#127a61", "#0a604c", "#064a3b"],
  },
};

function rampScale(ramp: string[]) {
  return scaleLinear<string>()
    .domain(ramp.map((_, i) => i / (ramp.length - 1)))
    .range(ramp)
    .clamp(true);
}

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

// Live bindings — importers always see the current theme's values.
export let palette: Palette = PALETTES.dark;
export let scene: SceneStyle = SCENES.dark;
export let ATTN_RAMP: string[] = RAMPS.dark.attn;
export let ACT_RAMP: string[] = RAMPS.dark.act;
export let attnScale = rampScale(ATTN_RAMP);
export let actScale = rampScale(ACT_RAMP);
export let attnLUT = rampLUT(ATTN_RAMP);

export function applyTheme(name: ThemeName): void {
  palette = PALETTES[name];
  scene = SCENES[name];
  ATTN_RAMP = RAMPS[name].attn;
  ACT_RAMP = RAMPS[name].act;
  attnScale = rampScale(ATTN_RAMP);
  actScale = rampScale(ACT_RAMP);
  attnLUT = rampLUT(ATTN_RAMP);
}
