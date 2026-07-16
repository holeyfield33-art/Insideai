"use client";

import dynamic from "next/dynamic";

// The simulator owns a WebGL canvas and a live websocket — client-only.
const Simulator = dynamic(() => import("@/components/Simulator"), {
  ssr: false,
  loading: () => (
    <div className="flex h-dvh w-full items-center justify-center bg-page">
      <div className="text-center">
        <div className="glow-text text-2xl font-semibold tracking-[0.35em] text-ink">
          INSIDE<span className="text-accent">AI</span>
        </div>
        <div className="mt-3 text-sm text-ink3">booting visualization engine…</div>
      </div>
    </div>
  ),
});

export default function Page() {
  return <Simulator />;
}
