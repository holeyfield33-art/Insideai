"use client";

/**
 * The transformer stack: one outlined slab per layer, like a cutaway diagram.
 * The layer computing right now turns amber; finished layers settle to a blue
 * whose depth shows how much that layer really changed the token's meaning
 * (‖Δresidual‖, normalized across layers). A pulse rides the central stream
 * upward as computation climbs.
 */
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";

import { palette, scene } from "@/lib/palette";
import { useSimStore } from "@/lib/store";

const BASE_Y = 0.45;

export const slabSpacing = (nLayer: number) => Math.min(1.15, 16 / Math.max(nLayer, 1));

const COLOR_IDLE = new THREE.Color(scene.slabFill);
const COLOR_BLUE = new THREE.Color("#5b89e6");
const COLOR_ACTIVE = new THREE.Color(palette.active);

export default function TransformerTower({ origin }: { origin: readonly [number, number, number] }) {
  const nLayer = useSimStore((s) => s.modelInfo?.n_layer ?? 0);
  const nEmbd = useSimStore((s) => s.modelInfo?.n_embd ?? 0);
  const layerMeta = useSimStore((s) => s.layerMeta);

  const slabMats = useRef<(THREE.MeshStandardMaterial | null)[]>([]);
  const slabMeshes = useRef<(THREE.Mesh | null)[]>([]);
  const beamMat = useRef<THREE.MeshBasicMaterial>(null);
  const orb = useRef<THREE.Mesh>(null);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  const slabGeo = useMemo(() => new THREE.BoxGeometry(5.4, 1, 5.4), []);
  const slabEdges = useMemo(() => new THREE.EdgesGeometry(slabGeo), [slabGeo]);

  const slabH = slabSpacing(nLayer);
  const thickness = Math.min(0.72, slabH * 0.62);
  const slabY = (i: number) => BASE_Y + i * slabH;
  const towerHeight = nLayer * slabH + 1.2;
  const labelEvery = nLayer > 12 ? 2 : 1;

  useFrame((state, dt) => {
    const { activeLayer, layerMeta: meta, running, stage } = useSimStore.getState();
    const t = state.clock.elapsedTime;
    const k = 1 - Math.exp(-dt * 6);

    let maxDelta = 0;
    for (const m of meta) if (m && m.residualDelta > maxDelta) maxDelta = m.residualDelta;

    for (let i = 0; i < slabMeshes.current.length; i++) {
      const mat = slabMats.current[i];
      const mesh = slabMeshes.current[i];
      if (!mat || !mesh) continue;
      const m = meta[i];
      const isActive = activeLayer === i;
      if (isActive) {
        tmpColor.copy(COLOR_ACTIVE);
      } else if (m && maxDelta > 0) {
        // sqrt spreads the perceptual range — otherwise one dominant layer
        // leaves the rest looking blank.
        tmpColor
          .copy(COLOR_IDLE)
          .lerp(COLOR_BLUE, 0.15 + 0.85 * Math.sqrt(m.residualDelta / maxDelta));
      } else {
        tmpColor.copy(COLOR_IDLE);
      }
      mat.color.lerp(tmpColor, k);
      const sy = thickness * (isActive ? 1 + 0.12 * Math.sin(t * 7) : 1);
      mesh.scale.y += (sy - mesh.scale.y) * k;
    }

    if (beamMat.current) {
      const target = running ? 0.45 : 0.15;
      beamMat.current.opacity += (target - beamMat.current.opacity) * k;
    }
    if (orb.current) {
      const targetY =
        activeLayer >= 0
          ? slabY(activeLayer)
          : stage === "logits" || stage === "sampled"
            ? towerHeight
            : BASE_Y - 0.4;
      orb.current.position.y += (targetY - orb.current.position.y) * (1 - Math.exp(-dt * 5));
      const s = running ? 1 + 0.2 * Math.sin(t * 9) : 0.0001;
      orb.current.scale.setScalar(s);
    }
  });

  if (nLayer === 0) return null;

  return (
    <group position={[origin[0], origin[1], origin[2]]}>
      <Html center position={[0, towerHeight + 1.7, 0]} distanceFactor={26} zIndexRange={[0, 10]}>
        <div className="pointer-events-none select-none whitespace-nowrap text-center">
          <div className="text-[14px] font-semibold uppercase tracking-[0.28em] text-ink">
            transformer stack
          </div>
          <div className="mt-0.5 text-[11px] text-ink2">
            the token's meaning passes through {nLayer} layers, bottom to top
          </div>
          <div className="font-mono text-[10px] text-ink3">
            amber = computing now · deeper blue = layer changed the meaning more · d={nEmbd}
          </div>
        </div>
      </Html>

      {/* residual stream */}
      <mesh position={[0, towerHeight / 2 - 0.4, 0]}>
        <cylinderGeometry args={[0.13, 0.13, towerHeight + 1.2, 12]} />
        <meshBasicMaterial ref={beamMat} color={scene.beam} transparent opacity={0.15} />
      </mesh>
      <mesh ref={orb} position={[0, BASE_Y - 0.4, 0]}>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshBasicMaterial color={palette.active} />
      </mesh>

      {Array.from({ length: nLayer }, (_, i) => (
        <group key={i} position={[0, slabY(i), 0]}>
          <mesh
            ref={(m) => {
              slabMeshes.current[i] = m;
              if (m) m.scale.y = thickness;
            }}
            geometry={slabGeo}
          >
            <meshStandardMaterial
              ref={(m) => {
                slabMats.current[i] = m;
              }}
              color={scene.slabFill}
              roughness={0.85}
              metalness={0}
            />
            <lineSegments geometry={slabEdges}>
              <lineBasicMaterial color={scene.slabEdge} transparent opacity={0.85} />
            </lineSegments>
          </mesh>
          {i % labelEvery === 0 && (
            <Html position={[3.4, 0, 0]} distanceFactor={18} zIndexRange={[0, 10]}>
              <div className="pointer-events-none select-none whitespace-nowrap font-mono text-[10px] text-ink3">
                L{i}
                {layerMeta[i] && (
                  <span className="ml-1.5 text-ink2">Δ{layerMeta[i]!.residualDelta}</span>
                )}
              </div>
            </Html>
          )}
        </group>
      ))}
    </group>
  );
}
