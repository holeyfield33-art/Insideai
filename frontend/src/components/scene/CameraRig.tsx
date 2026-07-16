"use client";

import { OrbitControls } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import * as THREE from "three";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import { useSimStore } from "@/lib/store";
import type { Zone } from "@/lib/types";

const ANCHORS: Record<Zone, { pos: [number, number, number]; target: [number, number, number] }> = {
  overview: { pos: [8, 16, 52], target: [8, 2.5, 0] },
  embedding: { pos: [-26, 6.5, 17], target: [-26, 3.2, 0] },
  attention: { pos: [0, 10, 27], target: [0, 3.8, 0] },
  tower: { pos: [19, 7.5, 20], target: [18, 4.5, 0] },
  neurons: { pos: [42, 5.0, 21], target: [42, 4.6, 0] },
};

/** Auto Tour: one deliberate pass through the pipeline per generation —
 *  tokens/attention → embeddings → layer stack → neurons → overview —
 *  dwelling on each stop instead of chasing every stage event. */
const TOUR: Zone[] = ["attention", "embedding", "tower", "neurons", "overview"];
const TOUR_DWELL_S = 7;

export default function CameraRig() {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const posTarget = useRef(new THREE.Vector3(...ANCHORS.overview.pos));
  const lookTarget = useRef(new THREE.Vector3(...ANCHORS.overview.target));
  const wasRunning = useRef(false);
  const tourTime = useRef(0);

  useFrame((state, dt) => {
    const { cinematic, focusZone, running, modelInfo } = useSimStore.getState();

    if (running && !wasRunning.current) tourTime.current = 0;
    wasRunning.current = running;
    if (running) tourTime.current += dt;

    let zone: Zone;
    if (!cinematic) {
      zone = focusZone;
    } else if (running) {
      const idx = Math.min(TOUR.length - 1, Math.floor(tourTime.current / TOUR_DWELL_S));
      zone = TOUR[idx];
    } else {
      zone = "overview";
    }

    if (zone === "tower") {
      // Frame the whole stack whatever the layer count (6 → 24+).
      const nLayer = modelInfo?.n_layer ?? 6;
      const h = Math.min(1.15, 16 / Math.max(nLayer, 1)) * nLayer;
      posTarget.current.set(19, h * 0.55 + 4.5, h * 1.15 + 11);
      lookTarget.current.set(18, h * 0.45 + 1, 0);
    } else {
      const anchor = ANCHORS[zone];
      posTarget.current.set(...anchor.pos);
      lookTarget.current.set(...anchor.target);
    }

    const controls = controlsRef.current;
    if (!controls) return;
    // Exponential damping toward the anchor — frame-rate independent.
    // Brisk enough that a zone click feels like direct navigation.
    const k = 1 - Math.exp(-dt * 3.2);
    state.camera.position.lerp(posTarget.current, k);
    controls.target.lerp(lookTarget.current, k);
    controls.update();
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping={false}
      maxDistance={120}
      minDistance={4}
      maxPolarAngle={Math.PI * 0.52}
      onStart={() => useSimStore.getState().setCinematic(false)}
    />
  );
}
