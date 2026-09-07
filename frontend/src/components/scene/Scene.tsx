"use client";

import { Canvas } from "@react-three/fiber";

import { useSimStore } from "@/lib/store";
import { scene } from "@/lib/palette";

import AttentionVisualizer from "./AttentionVisualizer";
import CameraRig from "./CameraRig";
import EmbeddingSpace from "./EmbeddingSpace";
import NeuralNetworkView from "./NeuralNetworkView";
import TokenVisualizer from "./TokenVisualizer";
import TransformerTower from "./TransformerTower";

/** World layout: four zones along the x axis, camera flies between them.
    Spacing is generous so one zone doesn't photobomb another's close-up. */
export const ZONE_ORIGIN = {
  embedding: [-26, 0, 0],
  attention: [0, 0, 0],
  tower: [18, 0, 0],
  neurons: [42, 0, 0],
} as const;

export default function Scene() {
  const zone = useSimStore((s) => s.focusZone);
  const cinematic = useSimStore((s) => s.cinematic);
  const all = zone === "overview" || cinematic;
  return (
    <Canvas
      camera={{ position: [6, 16, 46], fov: 50, near: 0.1, far: 400 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      onCreated={({ gl }) => { gl.domElement.tabIndex = 0; gl.domElement.setAttribute("aria-label", "3D model. Arrow keys orbit, plus and minus zoom, Home resets the view."); }}
    >
      <color attach="background" args={[scene.background]} />
      <fog attach="fog" args={[scene.background, 90, 240]} />

      <ambientLight intensity={scene.ambient} />
      <directionalLight position={[12, 30, 24]} intensity={scene.directional} color="#ffffff" />

      <gridHelper args={[220, 56, scene.gridMajor, scene.gridMinor]} position={[9, -3, 0]} />

      {(all || zone === "embedding") && <EmbeddingSpace origin={ZONE_ORIGIN.embedding} />}
      {(all || zone === "attention") && <TokenVisualizer origin={ZONE_ORIGIN.attention} />}
      {(all || zone === "attention") && <AttentionVisualizer origin={ZONE_ORIGIN.attention} />}
      {(all || zone === "tower") && <TransformerTower origin={ZONE_ORIGIN.tower} />}
      {(all || zone === "neurons") && <NeuralNetworkView origin={ZONE_ORIGIN.neurons} />}

      <CameraRig />
    </Canvas>
  );
}
