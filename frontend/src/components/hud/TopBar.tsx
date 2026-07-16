"use client";

import { useSimStore } from "@/lib/store";
import type { Zone } from "@/lib/types";

const ZONES: { id: Zone; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "embedding", label: "Embeddings" },
  { id: "attention", label: "Attention" },
  { id: "tower", label: "Layers" },
  { id: "neurons", label: "Neurons" },
];

const CONN = {
  open: { dot: "bg-emerald-500", label: "live" },
  connecting: { dot: "bg-amber-400", label: "connecting" },
  closed: { dot: "bg-danger", label: "offline" },
} as const;

export default function TopBar() {
  const connection = useSimStore((s) => s.connection);
  const modelInfo = useSimStore((s) => s.modelInfo);
  const focusZone = useSimStore((s) => s.focusZone);
  const cinematic = useSimStore((s) => s.cinematic);
  const setFocusZone = useSimStore((s) => s.setFocusZone);
  const setCinematic = useSimStore((s) => s.setCinematic);
  const stats = useSimStore((s) => s.stats);

  const conn = CONN[connection];

  return (
    <div className="pointer-events-auto flex items-center justify-between gap-3 px-4 py-2.5">
      <div className="flex items-baseline gap-3">
        <div className="glow-text select-none text-lg font-semibold tracking-[0.3em]">
          INSIDE<span className="text-accent">AI</span>
        </div>
        <div className="hidden text-[11px] uppercase tracking-widest text-ink3 lg:block">
          real-time transformer visualization
        </div>
      </div>

      <div className="glass flex items-center gap-1 rounded-full p-1">
        {ZONES.map((z) => (
          <button
            key={z.id}
            onClick={() => setFocusZone(z.id)}
            className={`rounded-full px-3 py-1 text-xs transition-colors ${
              focusZone === z.id && !cinematic
                ? "bg-accent text-white"
                : "text-ink2 hover:text-ink"
            }`}
          >
            {z.label}
          </button>
        ))}
        <button
          onClick={() => setCinematic(!cinematic)}
          title="Camera automatically follows each pipeline stage while generating"
          className={`ml-1 rounded-full px-3 py-1 text-xs transition-colors ${
            cinematic ? "bg-accent text-white" : "text-ink2 hover:text-ink"
          }`}
        >
          ✦ Auto Tour
        </button>
      </div>

      <div className="flex items-center gap-2">
        {stats.computeMs !== null && (
          <div className="glass hidden rounded-full px-3 py-1 font-mono text-[11px] text-ink2 xl:block">
            {stats.computeMs}ms/step
            {stats.tokensPerS !== null && ` · ${stats.tokensPerS} tok/s`}
          </div>
        )}
        {modelInfo && (
          <div className="glass hidden rounded-full px-3 py-1 font-mono text-[11px] text-ink2 sm:block">
            {modelInfo.model} · {modelInfo.n_layer}L · {modelInfo.n_head}H ·{" "}
            {modelInfo.param_count_h}
          </div>
        )}
        <div className="glass flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] text-ink2">
          <span className={`h-2 w-2 rounded-full ${conn.dot}`} />
          {conn.label}
        </div>
      </div>
    </div>
  );
}
