/** Small display helpers shared by HUD components. */

export function pct(p: number, digits = 1): string {
  const v = p * 100;
  if (v > 0 && v < 0.1) return "<0.1%";
  return `${v.toFixed(digits)}%`;
}

/** Make whitespace visible in token chips without changing the glyphs. */
export function visualizeToken(text: string): string {
  return text
    .replace(/^ /, "␣") // leading space -> ␣
    .replace(/\n/g, "⏎") // newline -> ⏎
    .replace(/\t/g, "⇥");
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
