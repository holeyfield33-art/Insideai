"use client";

/**
 * Tokens in reading order on a straight line — the base of the attention arc
 * diagram. Green = input (your prompt), gold = output (generated), the same
 * semantic colors as the classic input→output network figure. The newest
 * token pulses while the model works on it.
 */
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { visualizeToken } from "@/lib/format";
import { rowPositions } from "@/lib/layout";
import { palette, scene } from "@/lib/palette";
import { useSimStore } from "@/lib/store";

export default function TokenVisualizer({ origin }: { origin: readonly [number, number, number] }) {
  const tokens = useSimStore((s) => s.tokens);
  const positions = useMemo(() => rowPositions(tokens.length), [tokens.length]);

  const groupRefs = useRef<(THREE.Group | null)[]>([]);
  const sphereGeo = useMemo(() => new THREE.SphereGeometry(0.26, 20, 20), []);
  const rimMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: scene.nodeRim, side: THREE.BackSide }),
    []
  );
  const promptMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: palette.promptToken }),
    []
  );
  const genMat = useMemo(
    () => new THREE.MeshBasicMaterial({ color: palette.generatedToken }),
    []
  );

  useFrame((state, dt) => {
    const { running, tokens: toks } = useSimStore.getState();
    const t = state.clock.elapsedTime;
    const last = toks.length - 1;
    for (let i = 0; i < groupRefs.current.length; i++) {
      const g = groupRefs.current[i];
      if (!g) continue;
      const grow = Math.min(1, (g.userData.s ?? 0.01) + dt * 3);
      g.userData.s = grow;
      const pulse = running && i === last ? 1 + 0.22 * Math.sin(t * 6) : 1;
      g.scale.setScalar(grow * pulse);
    }
  });

  // DOM-tracked labels are the priciest thing in the scene — thin them out
  // as the sequence grows (every label still reachable via the HUD strip).
  const labelEvery = tokens.length > 56 ? 3 : tokens.length > 24 ? 2 : 1;
  const firstGenerated = tokens.findIndex((t) => t.kind === "generated");

  return (
    <group position={[origin[0], origin[1], origin[2]]}>
      {tokens.map((tok, i) => {
        const p = positions[i];
        if (!p) return null;
        const color = tok.kind === "prompt" ? palette.promptToken : palette.generatedToken;
        return (
          <group
            key={`${tok.index}-${tok.id}`}
            ref={(g) => {
              groupRefs.current[i] = g;
            }}
            position={p}
            scale={0.01}
          >
            <mesh geometry={sphereGeo} material={rimMat} scale={1.16} />
            <mesh geometry={sphereGeo} material={tok.kind === "prompt" ? promptMat : genMat} />
            {(i % labelEvery === 0 || i === tokens.length - 1) && (
              <Html
                center
                position={[0, i % 2 === 0 ? -0.62 : -1.08, 0]}
                distanceFactor={15}
                zIndexRange={[0, 10]}
              >
                <div
                  className="pointer-events-none select-none whitespace-pre rounded border bg-surface/95 px-1 font-mono text-[10px] text-ink"
                  style={{ borderColor: `${color}88` }}
                >
                  {visualizeToken(tok.text) || "·"}
                </div>
              </Html>
            )}
          </group>
        );
      })}

      {tokens.length > 0 && positions[0] && (
        <Html
          center
          position={[positions[0][0], positions[0][1] + 1.15, 0]}
          distanceFactor={20}
          zIndexRange={[0, 10]}
        >
          <div className="pointer-events-none select-none whitespace-nowrap rounded-full border border-prompt-tok/40 bg-surface/95 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-widest text-prompt-tok">
            input →
          </div>
        </Html>
      )}
      {firstGenerated >= 0 && positions[firstGenerated] && (
        <Html
          center
          position={[positions[firstGenerated][0], positions[firstGenerated][1] + 1.15, 0]}
          distanceFactor={20}
          zIndexRange={[0, 10]}
        >
          <div className="pointer-events-none select-none whitespace-nowrap rounded-full border border-gen-tok/40 bg-surface/95 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-widest text-gen-tok">
            output →
          </div>
        </Html>
      )}
    </group>
  );
}
