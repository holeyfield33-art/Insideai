/** Shared 3D layout math (used by TokenVisualizer and AttentionVisualizer). */

/**
 * Arrange n tokens on a straight line in reading order (left → right), like
 * the classic attention arc diagrams. Returns positions relative to the
 * attention zone origin.
 */
export function rowPositions(n: number): [number, number, number][] {
  if (n === 0) return [];
  // Cap total width to 26 world units so long sequences never reach the
  // neighbouring tower zone (x=18 minus margin).
  const spacing = Math.min(1.35, 26 / Math.max(n, 1));
  const total = (n - 1) * spacing;
  const positions: [number, number, number][] = [];
  for (let i = 0; i < n; i++) {
    positions.push([i * spacing - total / 2, 2.1, 0]);
  }
  return positions;
}
