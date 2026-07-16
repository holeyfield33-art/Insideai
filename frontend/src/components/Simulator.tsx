"use client";

import { useEffect } from "react";

import AttentionPanel from "@/components/hud/AttentionPanel";
import ErrorToast from "@/components/hud/ErrorToast";
import PipelineStatus from "@/components/hud/PipelineStatus";
import ProbabilityPanel from "@/components/hud/ProbabilityPanel";
import PromptInput from "@/components/hud/PromptInput";
import TokenGenerationAnimation from "@/components/hud/TokenGenerationAnimation";
import TopBar from "@/components/hud/TopBar";
import Scene from "@/components/scene/Scene";
import { simSocket } from "@/lib/ws";

export default function Simulator() {
  useEffect(() => {
    simSocket.connect();
    return () => simSocket.disconnect();
  }, []);

  return (
    <div className="relative h-dvh w-full overflow-hidden bg-page">
      <div className="absolute inset-0">
        <Scene />
      </div>

      {/* HUD overlay — panels opt back into pointer events individually.
          z-20 keeps panels above drei <Html> labels (zIndexRange caps at 10). */}
      <div className="pointer-events-none absolute inset-0 z-20 flex flex-col">
        <TopBar />

        <div className="flex min-h-0 flex-1 items-start justify-between gap-3 px-3 pt-2">
          <div className="hidden max-h-full w-72 flex-col gap-3 overflow-y-auto pb-2 md:flex">
            <PipelineStatus />
            <AttentionPanel />
          </div>
          <div className="hidden max-h-full w-80 flex-col gap-3 overflow-y-auto pb-2 md:flex">
            <ProbabilityPanel />
          </div>
        </div>

        <div className="flex flex-col gap-2 px-3 pb-3">
          <TokenGenerationAnimation />
          <PromptInput />
        </div>
      </div>

      <ErrorToast />
    </div>
  );
}
