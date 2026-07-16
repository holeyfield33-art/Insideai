"use client";

/**
 * 3D embedding scatter: each token's real embedding vector (896-d), projected
 * to 3 principal components by the backend. Classic scatter-plot aids for
 * readability: drop-lines to the floor grid, labeled PC axes, outlined nodes.
 * Green = input tokens, red = generated ones.
 */
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { visualizeToken } from "@/lib/format";
import { palette, scene } from "@/lib/palette";
import { useSimStore } from "@/lib/store";

const SCALE_XZ = 6;
const SCALE_Y = 4;
const BASE_Y = 3.4;
const FLOOR_Y = BASE_Y - SCALE_Y - 0.8;
const MAX_POINTS = 160;

export default function EmbeddingSpace({ origin }: { origin: readonly [number, number, number] }) {
  const tokens = useSimStore((s) => s.tokens);
  const pointCount = useSimStore((s) => s.points.length);
  const dim = useSimStore((s) => s.modelInfo?.n_embd ?? 0);

  const meshRefs = useRef<(THREE.Mesh | null)[]>([]);
  const dropRef = useRef<THREE.LineSegments>(null);
  const tmp = useMemo(() => new THREE.Vector3(), []);

  const sphereGeo = useMemo(() => new THREE.SphereGeometry(0.17, 18, 18), []);
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

  // Drop-lines buffer (updated every frame from the animated node positions).
  const dropGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(MAX_POINTS * 6), 3)
    );
    g.setDrawRange(0, 0);
    return g;
  }, []);

  const axesGeo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const s = SCALE_XZ + 1.2;
    const v = new Float32Array([
      -s, BASE_Y, 0, s, BASE_Y, 0,
      0, FLOOR_Y, 0, 0, BASE_Y + SCALE_Y + 1, 0,
      0, BASE_Y, -s, 0, BASE_Y, s,
    ]);
    g.setAttribute("position", new THREE.BufferAttribute(v, 3));
    return g;
  }, []);

  useFrame((_state, dt) => {
    const { points } = useSimStore.getState();
    const k = 1 - Math.exp(-dt * 3.6);
    const drop = dropRef.current?.geometry.getAttribute("position") as
      | THREE.BufferAttribute
      | undefined;
    let dropCount = 0;
    for (let i = 0; i < meshRefs.current.length; i++) {
      const mesh = meshRefs.current[i];
      const p = points[i];
      if (!mesh || !p) continue;
      tmp.set(p[0] * SCALE_XZ, p[1] * SCALE_Y + BASE_Y, p[2] * SCALE_XZ);
      mesh.position.lerp(tmp, k);
      mesh.scale.setScalar(Math.min(1, mesh.scale.x + dt * 2.5));
      if (drop && dropCount < MAX_POINTS) {
        drop.setXYZ(dropCount * 2, mesh.position.x, mesh.position.y, mesh.position.z);
        drop.setXYZ(dropCount * 2 + 1, mesh.position.x, FLOOR_Y, mesh.position.z);
        dropCount++;
      }
    }
    if (drop && dropRef.current) {
      drop.needsUpdate = true;
      dropRef.current.geometry.setDrawRange(0, dropCount * 2);
    }
  });

  const labelEvery = tokens.length > 56 ? 3 : tokens.length > 24 ? 2 : 1;

  return (
    <group position={[origin[0], origin[1], origin[2]]}>
      <gridHelper args={[15, 10, "#2a3450", "#1b2236"]} position={[0, FLOOR_Y, 0]} />

      <lineSegments geometry={axesGeo}>
        <lineBasicMaterial color={scene.axis} transparent opacity={0.8} />
      </lineSegments>
      <Html center position={[SCALE_XZ + 1.8, BASE_Y, 0]} distanceFactor={20} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none font-mono text-[10px] text-ink3">PC1</div>
      </Html>
      <Html center position={[0, BASE_Y + SCALE_Y + 1.4, 0]} distanceFactor={20} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none font-mono text-[10px] text-ink3">PC2</div>
      </Html>
      <Html center position={[0, BASE_Y, SCALE_XZ + 1.8]} distanceFactor={20} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none font-mono text-[10px] text-ink3">PC3</div>
      </Html>

      <lineSegments ref={dropRef} geometry={dropGeo} frustumCulled={false}>
        <lineBasicMaterial color="#3a4663" transparent opacity={0.55} />
      </lineSegments>

      <Html center position={[0, BASE_Y + SCALE_Y + 2.0, 0]} distanceFactor={26} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none whitespace-nowrap text-center">
          <div className="text-[14px] font-semibold uppercase tracking-[0.28em] text-ink">
            embeddings
          </div>
          <div className="mt-0.5 text-[11px] text-ink2">
            every token becomes a list of {dim || "…"} numbers — a point in space
          </div>
          <div className="font-mono text-[10px] text-ink3">
            similar meanings sit close together · 3-D view via PCA
          </div>
        </div>
      </Html>

      {tokens.slice(0, pointCount).map((tok, i) => (
        <mesh
          key={`${tok.index}-${tok.id}`}
          ref={(m) => {
            meshRefs.current[i] = m;
          }}
          position={[0, BASE_Y, 0]}
          scale={0.01}
          geometry={sphereGeo}
          material={tok.kind === "prompt" ? promptMat : genMat}
        >
          <mesh geometry={sphereGeo} material={rimMat} scale={1.16} />
          {(i % labelEvery === 0 || i === tokens.length - 1) && (
            <Html center position={[0, 0.45, 0]} distanceFactor={15} zIndexRange={[0, 10]}>
              <div
                className="pointer-events-none select-none whitespace-pre rounded border bg-surface/95 px-1 font-mono text-[10px] text-ink"
                style={{
                  borderColor:
                    tok.kind === "prompt" ? `${palette.promptToken}66` : `${palette.generatedToken}66`,
                }}
              >
                {visualizeToken(tok.text) || "·"}
              </div>
            </Html>
          )}
        </mesh>
      ))}
    </group>
  );
}
