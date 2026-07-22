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
  const theme = useSimStore((s) => s.theme);
  const setTheme = useSimStore((s) => s.setTheme);

  const conn = CONN[connection];
  const shortModel = modelInfo?.model.split("/").pop() ?? "";

  return (
    <div className="pointer-events-auto relative z-10 flex items-center justify-between gap-3 bg-page/85 px-4 py-2.5 backdrop-blur-md">
      <div className="flex items-baseline gap-3">
        <div className="glow-text relative z-10 select-none text-lg font-semibold tracking-[0.3em]">
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
          <div
            className="glass hidden cursor-help rounded-full px-3 py-1 font-mono text-[11px] text-ink2 sm:block"
            title={`The real language model running locally behind this visualization: ${modelInfo.model} — ${modelInfo.n_layer} transformer layers, ${modelInfo.n_head} attention heads, ${modelInfo.param_count_h} parameters. Everything on screen comes from its live computation.`}
          >
            {shortModel} · {modelInfo.n_layer} layers · {modelInfo.param_count_h}
          </div>
        )}
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title="Switch between dark and light theme"
          className="glass rounded-full px-3 py-1 text-[11px] text-ink2 transition-colors hover:text-ink"
        >
          {theme === "dark" ? "☾ Dark" : "☀ Light"}
        </button>
        <div className="glass flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] text-ink2">
          <span className={`h-2 w-2 rounded-full ${conn.dot}`} />
          {conn.label}
        </div>
      </div>
    </div>
  );
}
