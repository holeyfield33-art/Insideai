"use client";

/**
 * Live "flight recorder" trace: one line for zeta_raw ticking in as
 * `anomaly` events arrive (one per generation step). The chip row below
 * shows which layers are part of that step's forward pass — it is NOT a
 * per-layer reading, since zeta_raw is a single cross-layer coherence
 * signal shared by every layer in that step.
 *
 * zeta_proxy (the wire field name, kept as-is) is unitarity-lab's real
 * passive-mode zeta_raw (one reading per generation step, shared across
 * that step's layers — the hook reports a single cross-layer coherence
 * signal per forward pass, not a distinct value per layer). `flagged`
 * comes from VAR's calibrated SpectralRuptureDetector on the model's
 * spectral_gap, not a hardcoded value.
 */
import { useMemo } from "react";

import { useSimStore } from "@/lib/store";

const TRACE_POINTS = 80; // how many recent ticks the sparkline shows
const CHART_W = 260;
const CHART_H = 56;

/** Deterministic per-layer hue so the same layer always reads as the same
    color across the chip row and the trace line. */
function layerColor(layer: number, nLayer: number): string {
  const hue = nLayer > 1 ? (layer / (nLayer - 1)) * 260 : 200;
  return `hsl(${hue.toFixed(0)}, 70%, 55%)`;
}

export default function AnomalyEkgPanel() {
  const modelInfo = useSimStore((s) => s.modelInfo);
  const anomalyLatest = useSimStore((s) => s.anomalyLatest);
  const anomalyHistory = useSimStore((s) => s.anomalyHistory);

  const nLayer = modelInfo?.n_layer ?? 0;
  const recent = anomalyHistory.slice(-TRACE_POINTS);
  const latestPoint = recent[recent.length - 1] ?? null;

  const { path, lastX, lastY } = useMemo(() => {
    if (recent.length < 2) return { path: "", lastX: 0, lastY: CHART_H / 2 };
    const values = recent.map((p) => p.zetaProxy);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const stepX = CHART_W / Math.max(1, TRACE_POINTS - 1);
    const offset = TRACE_POINTS - recent.length; // right-align when not full yet
    const coords = recent.map((p, i) => {
      const x = (offset + i) * stepX;
      const norm = (p.zetaProxy - min) / span;
      const y = CHART_H - norm * (CHART_H - 8) - 4;
      return { x, y };
    });
    const d = coords.map((c, i) => `${i === 0 ? "M" : "L"}${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
    const last = coords[coords.length - 1];
    return { path: d, lastX: last.x, lastY: last.y };
  }, [recent]);

  const hasData = anomalyHistory.length > 0;

  return (
    <div className="glass pointer-events-auto rounded-2xl p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-ink2">
          Anomaly pipe
        </h2>
        <span className="font-mono text-[10px] text-ink3">
          {latestPoint ? `step ${latestPoint.step}` : "waiting"}
        </span>
      </div>

      <svg
        viewBox={`0 0 ${CHART_W} ${CHART_H}`}
        className="w-full overflow-visible"
        preserveAspectRatio="none"
      >
        {/* zero/mid reference line */}
        <line
          x1={0}
          x2={CHART_W}
          y1={CHART_H / 2}
          y2={CHART_H / 2}
          className="stroke-grid"
          strokeWidth={1}
        />
        {hasData && (
          <path
            d={path}
            fill="none"
            stroke="var(--color-accent, #4d84ea)"
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
        {latestPoint && recent.length > 0 && (
          <circle cx={lastX} cy={lastY} r={2.5} fill={layerColor(latestPoint.layer, nLayer)} />
        )}
      </svg>
      <p className="mt-1 text-[10px] text-ink3">
        {hasData
          ? "cross-layer coherence · one reading per step"
          : "zeta_raw ticks will appear here once a generation is running"}
      </p>

      {nLayer > 0 && (
        <div className="mt-2">
          <div className="mb-1 text-[10px] text-ink3">layers covered this step</div>
          <div className="flex flex-wrap gap-[3px]">
            {Array.from({ length: nLayer }, (_, i) => {
              const point = anomalyLatest[i];
              const isLatestLayer = latestPoint?.layer === i;
              return (
                <span
                  key={i}
                  title={
                    point
                      ? `layer ${i} was part of step ${point.step}'s forward pass · zeta_raw ${point.zetaProxy} (step-level, shared across all layers)`
                      : `layer ${i} · no reading yet`
                  }
                  className={`h-3.5 w-4 rounded-[3px] text-center font-mono text-[8px] leading-[14px] transition-colors ${
                    isLatestLayer ? "text-white" : "text-ink2"
                  }`}
                  style={{
                    backgroundColor: point
                      ? layerColor(i, nLayer)
                      : "var(--color-raised, #1c2333)",
                    opacity: point ? (isLatestLayer ? 1 : 0.55) : 0.4,
                  }}
                >
                  {i}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
