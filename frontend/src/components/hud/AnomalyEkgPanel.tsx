"use client";

import { useEffect, useRef, useState } from "react";
import { useSimStore } from "@/lib/store";

const H = 110, LEFT = 48, RIGHT = 14, TOP = 12, BOTTOM = 22;

export default function AnomalyEkgPanel() {
  const chart = useRef<SVGSVGElement>(null);
  const [W, setWidth] = useState(800);
  useEffect(() => {
    if (!chart.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)));
    observer.observe(chart.current);
    return () => observer.disconnect();
  }, []);
  const [metric, setMetric] = useState<"coherence" | "gap">("coherence");
  const history = useSimStore((s) => s.anomalyHistory);
  const recent = history.slice(-80);
  const latest = recent.at(-1);
  const values = recent.map((p) => metric === "coherence" ? p.zetaProxy : p.spectralGap);
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v));
  const threshold = metric === "gap" ? latest?.threshold ?? null : null;
  const bound = [...finite, ...(threshold === null ? [] : [threshold])];
  const min = metric === "coherence" ? -1 : Math.min(0, ...bound);
  const max = metric === "coherence" ? 1 : Math.max(1e-6, ...bound) * 1.15;
  const y = (v: number) => TOP + (1 - (v - min) / (max - min)) * (H - TOP - BOTTOM);
  const x = (i: number) => LEFT + i / Math.max(1, recent.length - 1) * (W - LEFT - RIGHT);
  let path = "", segment = false;
  values.forEach((v, i) => { if (v === null) { segment = false; return; } path += `${segment ? "L" : "M"}${x(i).toFixed(2)},${y(v).toFixed(2)} `; segment = true; });
  const value = values.at(-1);
  const format = (v: number) => Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < .001) ? v.toExponential(1) : v.toFixed(3);
  const state = !latest ? "Awaiting readings" : latest.calibrated === false ? "Calibrating" : latest.calibrated === null ? "Calibration unknown" : latest.flagged ? "Change flagged" : "No change flagged";

  return (
    <section className="telemetry-panel" aria-label="Step telemetry">
      <div className="trace-heading">
        <div><span className="eyebrow">STEP TELEMETRY</span><span className="trace-value">{value == null ? "—" : format(value)}</span></div>
        <div className="trace-switch" role="group" aria-label="Trace measurement">
          <button aria-pressed={metric === "coherence"} onClick={() => setMetric("coherence")}>Coherence ζ</button>
          <button aria-pressed={metric === "gap"} onClick={() => setMetric("gap")}>Spectral gap</button>
        </div>
        <span className={`trace-status ${latest?.flagged ? "text-danger" : ""}`}>{state}</span>
      </div>
      <svg ref={chart} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${metric === "coherence" ? "Cross-layer coherence, fixed scale minus one to one" : "Spectral gap, automatic scale"}. ${finite.length} measured steps. Current reading ${value == null ? "unavailable" : format(value)}.`} className="trace-chart">
        {[min, (max + min) / 2, max].map((v, i) => <g key={i}><line x1={LEFT} x2={W - RIGHT} y1={y(v)} y2={y(v)} stroke="var(--t-grid)" strokeDasharray={i === 1 ? "2 5" : undefined} /><text x={LEFT - 8} y={y(v) + 4} textAnchor="end" fill="var(--t-ink3)" fontSize="12" fontFamily="monospace">{metric === "coherence" ? v.toFixed(1) : v === 0 ? "0" : v.toPrecision(2)}</text></g>)}
        {threshold !== null && <line x1={LEFT} x2={W - RIGHT} y1={y(threshold)} y2={y(threshold)} stroke="var(--t-danger)" strokeDasharray="6 4"><title>Detector threshold {format(threshold)}</title></line>}
        <path d={path} fill="none" stroke="var(--t-accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        {value != null && <circle cx={x(values.length - 1)} cy={y(value)} r="3" fill="var(--t-accent)" />}
        {finite.length === 0 && <text x={W / 2} y={H / 2} textAnchor="middle" fill="var(--t-ink3)" fontSize="13">{recent.length ? "This signal is unavailable in this run" : "No measurements yet"}</text>}
        <text x={LEFT} y={H - 2} fill="var(--t-ink3)" fontSize="12" fontFamily="monospace">{recent.length ? `step ${recent[0].step}` : "step —"}</text>
        <text x={W - RIGHT} y={H - 2} textAnchor="end" fill="var(--t-ink3)" fontSize="12" fontFamily="monospace">{latest ? `step ${latest.step}` : "—"}</text>
      </svg>
      <div className="trace-footnote"><span>One reading per generation step · {metric === "coherence" ? "fixed scale −1 to +1" : `auto scale${threshold === null ? "" : ` · dashed threshold ${format(threshold)}`}`}</span><span>{latest?.calibrationScope === "backend_session" ? "Baseline: backend session" : "Baseline scope unavailable"}</span></div>
      <p className="trace-disclosure">Flags use spectral gap. Coherence is a separate observation; neither establishes model health.</p>
    </section>
  );
}
