/** Instrument palette. Semantic input / processing / output colors use
 * labels as a second encoding; no colorblind-validation claim is made here. */
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
    page: "#101412",
    surface: "#191c1b",
    raised: "#242927",
    ink: "#f0f2ee",
    ink2: "#bfc7c0",
    ink3: "#929d95",
    grid: "#333a35",
    accent: "#df9c6c",
    accentTrack: "#493326",
    promptToken: "#64b5a0",
    generatedToken: "#e3bf61",
    active: "#df9c6c",
    danger: "#f28579",
  },
  light: {
    page: "#eff2ef",
    surface: "#ffffff",
    raised: "#e4eae5",
    ink: "#19251e",
    ink2: "#46574b",
    ink3: "#5b6d61",
    grid: "#c9d3cc",
    accent: "#9a471b",
    accentTrack: "#efd9c9",
    promptToken: "#167561",
    generatedToken: "#8b650b",
    active: "#9a471b",
    danger: "#b83429",
  },
};

const SCENES: Record<ThemeName, SceneStyle> = {
  dark: {
    background: "#101412",
    nodeRim: "#e9eee8",
    edge: "#edb894",
    edgeFaint: "#3b423c",
    slabFill: "#222c26",
    slabEdge: "#708378",
    towerBlue: "#cb875a",
    beam: "#66796c",
    axis: "#6e8073",
    dropline: "#35493c",
    gridMajor: "#293a2e",
    gridMinor: "#1a241d",
    ambient: 0.85,
    directional: 0.5,
  },
  light: {
    background: "#e8efea",
    nodeRim: "#213b2b",
    edge: "#9a471b",
    edgeFaint: "#b4c6b9",
    slabFill: "#f0f2ee",
    slabEdge: "#536c5c",
    towerBlue: "#b46736",
    beam: "#a2b5a8",
    axis: "#5b6d61",
    dropline: "#a2b5a8",
    gridMajor: "#bccdc1",
    gridMinor: "#d5e0d8",
    ambient: 1.05,
    directional: 0.65,
  },
};

const RAMPS: Record<ThemeName, { attn: string[]; act: string[] }> = {
  dark: {
    attn: ["#191c1b", "#392a20", "#5e3c26", "#875532", "#b07445", "#ce956b", "#e4b18a", "#f4cfad", "#ffe9d6"],
    act: ["#14221c", "#1d372a", "#294e3b", "#37674c", "#47835e", "#639d76", "#88b894", "#b2d5b7", "#e0f0df"],
  },
  light: {
    attn: ["#fff7ef", "#f7dfc8", "#ebbe98", "#d89a6d", "#bd7947", "#a35a2f", "#813f1c", "#602c13", "#411b0c"],
    act: ["#f1faf5", "#d5eddd", "#aed7bc", "#87bc9a", "#639f7a", "#41815c", "#26613f", "#154829", "#0c321c"],
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
