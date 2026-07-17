"use client";

import { Canvas } from "@react-three/fiber";

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
  return (
    <Canvas
      camera={{ position: [6, 16, 46], fov: 50, near: 0.1, far: 400 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
    >
      <color attach="background" args={[scene.background]} />
      <fog attach="fog" args={[scene.background, 90, 240]} />

      <ambientLight intensity={scene.ambient} />
      <directionalLight position={[12, 30, 24]} intensity={scene.directional} color="#ffffff" />

      <gridHelper args={[220, 56, scene.gridMajor, scene.gridMinor]} position={[9, -3, 0]} />

      <EmbeddingSpace origin={ZONE_ORIGIN.embedding} />
      <TokenVisualizer origin={ZONE_ORIGIN.attention} />
      <AttentionVisualizer origin={ZONE_ORIGIN.attention} />
      <TransformerTower origin={ZONE_ORIGIN.tower} />
      <NeuralNetworkView origin={ZONE_ORIGIN.neurons} />

      <CameraRig />
    </Canvas>
  );
}
