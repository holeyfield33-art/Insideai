"use client";

import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
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
  const { camera, gl } = useThree();
  const reducedMotion = useRef(false);
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const posTarget = useRef(new THREE.Vector3(...ANCHORS.overview.pos));
  const lookTarget = useRef(new THREE.Vector3(...ANCHORS.overview.target));
  const wasRunning = useRef(false);
  const wasCinematic = useRef(false);
  const tourTime = useRef(0);
  const lastFocusNonce = useRef(-1);
  const navRemaining = useRef(NAV_WINDOW_S); // initial glide to overview

  useEffect(() => {
    const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => { reducedMotion.current = preference.matches; };
    sync(); preference.addEventListener("change", sync);
    const onKey = (event: KeyboardEvent) => {
      const controls = controlsRef.current;
      if (!controls || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "Home"].includes(event.key)) return;
      event.preventDefault();
      useSimStore.getState().setCinematic(false);
      navRemaining.current = 0;
      if (event.key === "Home") { useSimStore.getState().setFocusZone(useSimStore.getState().focusZone); return; }
      const spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
      if (event.key === "ArrowLeft") spherical.theta -= .12;
      if (event.key === "ArrowRight") spherical.theta += .12;
      if (event.key === "ArrowUp") spherical.phi -= .1;
      if (event.key === "ArrowDown") spherical.phi += .1;
      if (event.key === "+" || event.key === "=") spherical.radius *= .9;
      if (event.key === "-") spherical.radius *= 1.1;
      spherical.phi = THREE.MathUtils.clamp(spherical.phi, .1, Math.PI * .52);
      spherical.radius = THREE.MathUtils.clamp(spherical.radius, 3, 140);
      camera.position.copy(new THREE.Vector3().setFromSpherical(spherical).add(controls.target));
      camera.lookAt(controls.target); controls.update();
    };
    gl.domElement.addEventListener("keydown", onKey);
    return () => { preference.removeEventListener("change", sync); gl.domElement.removeEventListener("keydown", onKey); };
  }, [camera, gl]);

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
      const k = reducedMotion.current ? 1 : 1 - Math.exp(-dt * 5);
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
