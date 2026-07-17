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
  attention: { pos: [5, 9.5, 24], target: [0, 3.6, -1.5] },
  tower: { pos: [19, 7.5, 20], target: [18, 4.5, 0] },
  neurons: { pos: [42, 5.0, 21], target: [42, 4.6, 0] },
};

/** Auto Tour: one deliberate pass through the pipeline — plays when the
 *  toggle is switched on and restarts with every generation. */
const TOUR: Zone[] = ["attention", "embedding", "tower", "neurons", "overview"];
const TOUR_DWELL_S = 7;

/** After a zone click the camera glides for this long, then the controls are
 *  fully yours — zoom, orbit and pan stay wherever you put them. */
const NAV_WINDOW_S = 3;

export default function CameraRig() {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const posTarget = useRef(new THREE.Vector3(...ANCHORS.overview.pos));
  const lookTarget = useRef(new THREE.Vector3(...ANCHORS.overview.target));
  const wasRunning = useRef(false);
  const wasCinematic = useRef(false);
  const tourTime = useRef(0);
  const lastFocusNonce = useRef(-1);
  const navRemaining = useRef(NAV_WINDOW_S); // initial glide to overview

  useFrame((state, dt) => {
    const { cinematic, focusZone, focusNonce, running, modelInfo } = useSimStore.getState();

    // Tour restarts on toggle-on and on generation start (while toggled on).
    if (cinematic && !wasCinematic.current) tourTime.current = 0;
    if (cinematic && running && !wasRunning.current) tourTime.current = 0;
    wasCinematic.current = cinematic;
    wasRunning.current = running;

    let zone: Zone | null = null;
    if (cinematic) {
      tourTime.current += dt;
      const idx = Math.min(TOUR.length - 1, Math.floor(tourTime.current / TOUR_DWELL_S));
      zone = TOUR[idx];
    } else {
      if (focusNonce !== lastFocusNonce.current) {
        lastFocusNonce.current = focusNonce;
        navRemaining.current = NAV_WINDOW_S;
      }
      if (navRemaining.current > 0) {
        navRemaining.current -= dt;
        zone = focusZone;
      }
    }

    const controls = controlsRef.current;
    if (!controls) return;

    if (zone !== null) {
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
      const k = 1 - Math.exp(-dt * 3.2);
      state.camera.position.lerp(posTarget.current, k);
      controls.target.lerp(lookTarget.current, k);
    }
    controls.update();

    // Test hook: lets the e2e suite assert camera behavior (free zoom, tour).
    (window as unknown as { __cam?: number[] }).__cam = state.camera.position
      .toArray()
      .map((v) => Math.round(v * 100) / 100);
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping={false}
      maxDistance={140}
      minDistance={3}
      maxPolarAngle={Math.PI * 0.52}
      onStart={() => {
        // The user grabbed the camera: cancel any glide and leave Auto Tour.
        navRemaining.current = 0;
        useSimStore.getState().setCinematic(false);
      }}
    />
  );
}
