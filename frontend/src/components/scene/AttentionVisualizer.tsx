"use client";

/**
 * Attention as a classic arc diagram: tokens sit on a line, and every arc is
 * one real entry of the attention matrix for the selected layer/head — a
 * later token "looking back" at an earlier one. Darker blue = stronger
 * attention (sequential ramp); the newest token's arcs are drawn in red
 * (it is the output being computed right now).
 */
import { Html } from "@react-three/drei";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

import { rowPositions } from "@/lib/layout";
import { attnScale, palette, scene } from "@/lib/palette";
import { useSimStore } from "@/lib/store";

const THRESHOLD = 0.05;
const MAX_EDGES = 300;
const CURVE_SEGMENTS = 8;

function buildArcGeometry(
  matrix: number[][],
  positions: [number, number, number][],
  lastOnly: boolean
): THREE.BufferGeometry | null {
  const n = Math.min(matrix.length, positions.length);
  if (n < 2) return null;

  const edges: { i: number; j: number; w: number }[] = [];
  const start = lastOnly ? n - 1 : 1;
  for (let i = start; i < n; i++) {
    const row = matrix[i];
    for (let j = 0; j < i; j++) {
      const w = (row[j] ?? 0) / 255;
      if (w >= THRESHOLD) edges.push({ i, j, w });
    }
    if (lastOnly) break;
  }
  if (edges.length === 0) return null;
  edges.sort((a, b) => b.w - a.w);
  const kept = edges.slice(0, MAX_EDGES);

  const verts: number[] = [];
  const colors: number[] = [];
  const p0 = new THREE.Vector3();
  const p2 = new THREE.Vector3();
  const mid = new THREE.Vector3();
  const color = new THREE.Color();
  const red = new THREE.Color(palette.generatedToken);
  const faint = new THREE.Color(scene.edgeFaint);

  for (const { i, j, w } of kept) {
    p0.set(...positions[i]);
    p2.set(...positions[j]);
    mid.lerpVectors(p0, p2, 0.5);
    mid.y += 0.6 + p0.distanceTo(p2) * 0.3; // arc height grows with span
    const curve = new THREE.QuadraticBezierCurve3(p0.clone(), mid.clone(), p2.clone());
    const pts = curve.getPoints(CURVE_SEGMENTS);

    if (lastOnly) {
      color.copy(faint).lerp(red, 0.45 + 0.55 * w);
    } else {
      // Floor at ramp step 0.4 so even modest weights stay visible.
      color.set(attnScale(0.4 + 0.6 * w));
    }
    for (let s = 0; s < pts.length - 1; s++) {
      verts.push(pts[s].x, pts[s].y, pts[s].z, pts[s + 1].x, pts[s + 1].y, pts[s + 1].z);
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return geo;
}

export default function AttentionVisualizer({
  origin,
}: {
  origin: readonly [number, number, number];
}) {
  const attention = useSimStore((s) => s.attention);
  const selectedLayer = useSimStore((s) => s.selectedLayer);
  const selectedHead = useSimStore((s) => s.selectedHead);
  const tokenCount = useSimStore((s) => s.tokens.length);

  const att = attention[selectedLayer] ?? null;
  const matrix = useMemo(() => {
    if (!att) return null;
    if (selectedHead >= 0 && att.heads) return att.heads[selectedHead] ?? att.mean;
    return att.mean;
  }, [att, selectedHead]);

  const positions = useMemo(() => rowPositions(tokenCount), [tokenCount]);

  const geoAll = useMemo(
    () => (matrix ? buildArcGeometry(matrix, positions, false) : null),
    [matrix, positions]
  );
  const geoLast = useMemo(
    () => (matrix ? buildArcGeometry(matrix, positions, true) : null),
    [matrix, positions]
  );

  useEffect(() => {
    return () => {
      geoAll?.dispose();
    };
  }, [geoAll]);
  useEffect(() => {
    return () => {
      geoLast?.dispose();
    };
  }, [geoLast]);

  return (
    <group position={[origin[0], origin[1], origin[2]]}>
      <Html center position={[0, 9.6, 0]} distanceFactor={26} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none whitespace-nowrap text-center">
          <div className="text-[14px] font-semibold uppercase tracking-[0.28em] text-ink">
            self-attention
          </div>
          <div className="mt-0.5 text-[11px] text-ink2">
            each arc = one token looking back at an earlier token · darker = stronger
          </div>
          <div className="font-mono text-[10px] text-ink3">
            {att
              ? `layer ${selectedLayer} · ${
                  selectedHead === -1 || !att.heads ? "average of all heads" : `head ${selectedHead}`
                } · red arcs = the token being generated`
              : "arcs appear while the model reads your prompt"}
          </div>
        </div>
      </Html>

      {geoAll && (
        <lineSegments geometry={geoAll} frustumCulled={false}>
          <lineBasicMaterial vertexColors transparent opacity={0.9} />
        </lineSegments>
      )}
      {geoLast && (
        <lineSegments geometry={geoLast} frustumCulled={false}>
          <lineBasicMaterial vertexColors transparent opacity={0.95} />
        </lineSegments>
      )}
    </group>
  );
}
