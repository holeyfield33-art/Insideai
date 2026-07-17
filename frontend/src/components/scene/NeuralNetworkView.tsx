"use client";

/**
 * The classic layered network diagram — live, and real. For the selected
 * layer: a green input column (the token's hidden state entering the MLP), a
 * teal-blue hidden column (actual GELU/SiLU activations), a red output column
 * (what the MLP hands back).
 *
 * Wiring = the model's true weight matrices (pooled |W| block-means).
 * Live signal = wiring strength × the source node's current activation, so
 * you can watch each token's signal flow along the wires once per token —
 * no random blinking, one clean pulse per generated token.
 */
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { visualizeToken } from "@/lib/format";
import { palette, scene } from "@/lib/palette";
import { useSimStore } from "@/lib/store";

const X_IN = -6.6;
const X_HID = 0;
const X_OUT = 6.6;
const CY = 5.4; // vertical center of the diagram

const EDGE_MIN = 0.06;

/** The three columns light up in order per token — the signal visibly
 *  travels input → neurons → output instead of everything blinking at once. */
const COLUMN_DELAY_S = { in: 0, hid: 0.3, out: 0.6 };

function columnPositions(count: number, x: number): [number, number, number][] {
  const spacing = Math.min(0.95, 11 / Math.max(count, 1));
  const total = (count - 1) * spacing;
  return Array.from({ length: count }, (_, i) => [x, CY + total / 2 - i * spacing, 0]);
}

/** Contiguous block means — reduce an array to n values without inventing data. */
function poolTo(arr: number[], n: number): number[] {
  if (arr.length <= n) return arr.slice();
  const out = new Array(n).fill(0);
  const size = arr.length / n;
  for (let g = 0; g < n; g++) {
    const start = Math.floor(g * size);
    const end = Math.max(start + 1, Math.floor((g + 1) * size));
    let sum = 0;
    for (let i = start; i < end; i++) sum += arr[i];
    out[g] = sum / (end - start);
  }
  return out;
}

interface Wiring {
  geo: THREE.BufferGeometry;
  /** Per edge: index of its source node in the flat node array. */
  src: number[];
  /** Per edge: pooled |W| strength 0..1. */
  base: number[];
}

function buildWiring(
  wIn: number[][],
  wOut: number[][],
  pIn: [number, number, number][],
  pHid: [number, number, number][],
  pOut: [number, number, number][],
  IN: number
): Wiring {
  const verts: number[] = [];
  const src: number[] = [];
  const base: number[] = [];

  const push = (a: [number, number, number], b: [number, number, number], w: number, s: number) => {
    verts.push(a[0], a[1], a[2] - 0.05, b[0], b[1], b[2] - 0.05);
    src.push(s);
    base.push(w);
  };

  for (let h = 0; h < wIn.length; h++) {
    for (let i = 0; i < (wIn[h]?.length ?? 0); i++) {
      const w = wIn[h][i];
      if (w >= EDGE_MIN && pIn[i] && pHid[h]) push(pIn[i], pHid[h], w, i);
    }
  }
  for (let o = 0; o < wOut.length; o++) {
    for (let h = 0; h < (wOut[o]?.length ?? 0); h++) {
      const w = wOut[o][h];
      if (w >= EDGE_MIN && pHid[h] && pOut[o]) push(pHid[h], pOut[o], w, IN + h);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(new Array(verts.length).fill(0.2), 3));
  return { geo, src, base };
}

export default function NeuralNetworkView({ origin }: { origin: readonly [number, number, number] }) {
  const modelInfo = useSimStore((s) => s.modelInfo);
  const selectedLayer = useSimStore((s) => s.selectedLayer);
  const ffnAll = useSimStore((s) => s.ffn);
  const lastToken = useSimStore((s) => s.tokens[s.tokens.length - 1]?.text ?? null);

  const wiringInfo = modelInfo?.mlp_wiring ?? null;
  const IN = wiringInfo?.in_nodes ?? 12;
  const HID = wiringInfo?.hidden_nodes ?? 16;

  const layerWiring = wiringInfo?.layers[selectedLayer] ?? null;

  const pIn = useMemo(() => columnPositions(IN, X_IN), [IN]);
  const pHid = useMemo(() => columnPositions(HID, X_HID), [HID]);
  const pOut = useMemo(() => columnPositions(IN, X_OUT), [IN]);

  const wiring = useMemo(
    () =>
      layerWiring ? buildWiring(layerWiring.w_in, layerWiring.w_out, pIn, pHid, pOut, IN) : null,
    [layerWiring, pIn, pHid, pOut, IN]
  );
  useEffect(() => {
    return () => {
      wiring?.geo.dispose();
    };
  }, [wiring]);

  const sphereGeo = useMemo(() => new THREE.SphereGeometry(0.3, 18, 18), []);
  const rimMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: scene.nodeRim, side: THREE.BackSide }),
    []
  );
  // Built per mount — the scene remounts on theme change, so these follow it.
  const colorSet = useMemo(() => {
    const full = {
      in: new THREE.Color(palette.promptToken),
      hid: new THREE.Color(palette.accent),
      out: new THREE.Color(palette.generatedToken),
    };
    return {
      full,
      pale: {
        in: new THREE.Color(palette.surface).lerp(full.in, 0.25),
        hid: new THREE.Color(palette.surface).lerp(full.hid, 0.25),
        out: new THREE.Color(palette.surface).lerp(full.out, 0.25),
      },
      edgeFaint: new THREE.Color(scene.edgeFaint),
      edgeStrong: new THREE.Color(scene.edge),
    };
  }, []);
  const nodeMats = useRef<(THREE.MeshBasicMaterial | null)[]>([]);
  const nodeMeshes = useRef<(THREE.Mesh | null)[]>([]);
  const current = useRef<Float32Array>(new Float32Array(IN + HID + IN));
  const tmpColor = useMemo(() => new THREE.Color(), []);
  const lastDataStep = useRef(-1);
  const sweepStart = useRef(-10);

  useFrame((state, dt) => {
    const { ffn, selectedLayer: sel } = useSimStore.getState();
    const data = ffn[sel] ?? null;
    const now = state.clock.elapsedTime;

    // A new token's data for this layer starts a fresh left→right sweep.
    if (data && data.step !== lastDataStep.current) {
      lastDataStep.current = data.step;
      sweepStart.current = now;
    }
    const tSince = now - sweepStart.current;

    const inVals = data?.inputPooled ?? null;
    const hidVals = data ? poolTo(data.activations, HID) : null;
    const outVals = data?.outputPooled ?? null;

    const norm = (arr: number[] | null) => {
      if (!arr) return null;
      let m = 1e-6;
      for (const v of arr) m = Math.max(m, Math.abs(v));
      return { arr, m };
    };
    const nIn = norm(inVals);
    const nHid = norm(hidVals);
    const nOut = norm(outVals);

    const total = IN + HID + IN;
    if (current.current.length !== total) current.current = new Float32Array(total);

    for (let idx = 0; idx < total; idx++) {
      const mat = nodeMats.current[idx];
      const mesh = nodeMeshes.current[idx];
      if (!mat || !mesh) continue;

      let t = 0;
      let col: "in" | "hid" | "out";
      if (idx < IN) {
        col = "in";
        t = nIn ? Math.abs(nIn.arr[idx] ?? 0) / nIn.m : 0;
      } else if (idx < IN + HID) {
        col = "hid";
        t = nHid ? Math.abs(nHid.arr[idx - IN] ?? 0) / nHid.m : 0;
      } else {
        col = "out";
        t = nOut ? Math.abs(nOut.arr[idx - IN - HID] ?? 0) / nOut.m : 0;
      }

      // Staggered easing: each column only starts moving toward the new
      // values after its delay, so the update reads left → right.
      const k = tSince < COLUMN_DELAY_S[col] ? 0 : 1 - Math.exp(-dt * 4.5);
      current.current[idx] += (t - current.current[idx]) * k;
      const v = current.current[idx];
      tmpColor.copy(colorSet.pale[col]).lerp(colorSet.full[col], v);
      mat.color.copy(tmpColor);
      mesh.scale.setScalar(0.78 + v * 0.55);
    }

    // Signal flow: each wire's brightness = its real |W| strength times the
    // live activation of the node feeding it.
    if (wiring) {
      const colors = wiring.geo.getAttribute("color") as THREE.BufferAttribute;
      for (let e = 0; e < wiring.src.length; e++) {
        const act = current.current[wiring.src[e]] ?? 0;
        const t = wiring.base[e] * (0.18 + 0.82 * act);
        tmpColor.copy(colorSet.edgeFaint).lerp(colorSet.edgeStrong, t);
        colors.setXYZ(e * 2, tmpColor.r, tmpColor.g, tmpColor.b);
        colors.setXYZ(e * 2 + 1, tmpColor.r, tmpColor.g, tmpColor.b);
      }
      colors.needsUpdate = true;
    }
  });

  const nInner = modelInfo?.n_inner ?? 0;
  const nEmbd = modelInfo?.n_embd ?? 0;
  const stats = ffnAll[selectedLayer];
  const allPositions = [...pIn, ...pHid, ...pOut];

  return (
    <group position={[origin[0], origin[1], origin[2]]}>
      <Html center position={[0, CY + 6.4, 0]} distanceFactor={26} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none whitespace-nowrap text-center">
          <div className="text-[14px] font-semibold uppercase tracking-[0.28em] text-ink">
            neurons — inside layer {selectedLayer}&apos;s mlp
          </div>
          <div className="mt-0.5 text-[11px] text-ink2">
            {lastToken
              ? `processing “${visualizeToken(lastToken)}” — signal flows input → neurons → output`
              : "wire brightness = real connection strength × live signal"}
          </div>
          <div className="font-mono text-[10px] text-ink3">
            each circle is a group of real neurons · pick a layer in the attention panel
            {stats && ` · ${Math.round(stats.activeFrac * 100)}% of ${nInner} firing`}
          </div>
        </div>
      </Html>

      {wiring && (
        <lineSegments geometry={wiring.geo} frustumCulled={false}>
          <lineBasicMaterial vertexColors transparent opacity={0.9} />
        </lineSegments>
      )}

      {allPositions.map((p, idx) => (
        <group key={idx} position={p}>
          <mesh geometry={sphereGeo} material={rimMat} scale={1.16} />
          <mesh
            ref={(m) => {
              nodeMeshes.current[idx] = m;
            }}
            geometry={sphereGeo}
          >
            <meshBasicMaterial
              ref={(m) => {
                nodeMats.current[idx] = m;
              }}
              color={palette.surface}
            />
          </mesh>
        </group>
      ))}

      <Html center position={[X_IN, CY - 6.3, 0]} distanceFactor={22} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none whitespace-nowrap text-center text-[10px]">
          <div className="font-semibold uppercase tracking-widest text-prompt-tok">input</div>
          <div className="font-mono text-ink3">token meaning · {nEmbd || "…"}d → {IN} groups</div>
        </div>
      </Html>
      <Html center position={[X_HID, CY - 6.3, 0]} distanceFactor={22} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none whitespace-nowrap text-center text-[10px]">
          <div className="font-semibold uppercase tracking-widest text-accent">neurons</div>
          <div className="font-mono text-ink3">{nInner || "…"} → {HID} groups</div>
        </div>
      </Html>
      <Html center position={[X_OUT, CY - 6.3, 0]} distanceFactor={22} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none whitespace-nowrap text-center text-[10px]">
          <div className="font-semibold uppercase tracking-widest text-gen-tok">output</div>
          <div className="font-mono text-ink3">what this layer adds → {IN} groups</div>
        </div>
      </Html>
    </group>
  );
}
