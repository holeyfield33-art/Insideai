"use client";

import { useEffect, useState } from "react";
import AnomalyEkgPanel from "@/components/hud/AnomalyEkgPanel";
import AttentionPanel from "@/components/hud/AttentionPanel";
import ErrorToast from "@/components/hud/ErrorToast";
import PipelineStatus from "@/components/hud/PipelineStatus";
import ProbabilityPanel from "@/components/hud/ProbabilityPanel";
import PromptInput from "@/components/hud/PromptInput";
import ReplayBanner from "@/components/hud/ReplayBanner";
import RunPicker from "@/components/hud/RunPicker";
import TokenGenerationAnimation from "@/components/hud/TokenGenerationAnimation";
import TopBar from "@/components/hud/TopBar";
import Scene from "@/components/scene/Scene";
import { useSimStore } from "@/lib/store";
import { simSocket } from "@/lib/ws";
import type { Zone } from "@/lib/types";

const VIEWS: { id: Zone; label: string; detail: string }[] = [
  { id: "overview", label: "Overview", detail: "Complete transformer pathway" },
  { id: "embedding", label: "Embeddings", detail: "Token vectors · PCA projection" },
  { id: "attention", label: "Attention", detail: "Query–key relationships" },
  { id: "tower", label: "Layers", detail: "Residual stream · transformer stack" },
  { id: "neurons", label: "Neurons", detail: "Pooled MLP activations" },
];
const PANELS = ["Inspect", "Output", "Runs"] as const;
type Panel = typeof PANELS[number];

export default function Simulator() {
  const [panel, setPanel] = useState<Panel>("Inspect");
  const [expanded, setExpanded] = useState(false);
  const theme = useSimStore((s) => s.theme);
  const model = useSimStore((s) => s.modelInfo);
  const zone = useSimStore((s) => s.focusZone);
  const setZone = useSimStore((s) => s.setFocusZone);
  const cinematic = useSimStore((s) => s.cinematic);
  const setCinematic = useSimStore((s) => s.setCinematic);
  const stats = useSimStore((s) => s.stats);
  const step = useSimStore((s) => s.step);
  const tokens = useSimStore((s) => s.tokens.length);
  const running = useSimStore((s) => s.running);
  const selected = VIEWS.find((v) => v.id === zone)!;

  useEffect(() => { simSocket.connect(); return () => simSocket.disconnect(); }, []);
  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);

  return (
    <div className={`observatory ${expanded ? "stage-expanded" : ""}`}>
      <a className="skip-link" href="#main">Skip to model workspace</a>
      <TopBar />
      <main id="main" className="workbench" tabIndex={-1}>
        <section className="stage-column" aria-label="Interactive model workspace">
          <nav className="view-navigation" aria-label="3D views">
            {VIEWS.map((v, i) => <button key={v.id} onClick={() => setZone(v.id)} aria-pressed={zone === v.id && !cinematic} className={zone === v.id && !cinematic ? "is-selected" : ""}><span>{String(i + 1).padStart(2, "0")}</span>{v.label}</button>)}
          </nav>
          <PromptInput />
          <div className="stage-toolbar">
            <div><span className="eyebrow">SPATIAL VIEW</span><h1>{selected.label}</h1></div>
            <div className="stage-actions">
              <button className="instrument-button" aria-pressed={cinematic} onClick={() => setCinematic(!cinematic)}>Auto tour {cinematic ? "on" : "off"}</button>
              <button className="instrument-button" onClick={() => setZone(zone)} title="Return to the selected view’s camera position">Reset view</button>
              <button className="instrument-button" onClick={() => setExpanded(!expanded)} aria-expanded={expanded}>{expanded ? "Show inspector" : "Expand stage"}</button>
            </div>
          </div>
          <div className="spatial-stage" role="region" aria-label={`${selected.label} 3D visualization. Drag to orbit, scroll to zoom, right-drag to pan.`}>
            <Scene key={theme} />
            <ReplayBanner />
            <div className="stage-caption"><span className="eyebrow">{selected.detail}</span><span className="stage-index">{String(VIEWS.findIndex((v) => v.id === zone) + 1).padStart(2, "0")} / 05</span></div>
            {!model && <div className="stage-empty"><span className="eyebrow">INSTRUMENT STANDBY</span><p>Connect a model.<br />See the computation.</p><span>Live tensors and recorded runs appear here.</span></div>}
            <div className="stage-key"><span><i className="key-input" />Input</span><span><i className="key-compute" />Processing</span><span><i className="key-output" />Output</span></div>
            <div className="stage-gestures">Drag to orbit <b>·</b> Scroll to zoom <b>·</b> Right-drag to pan</div>
          </div>
          <div className="readout-strip" aria-label="Run measurements">
            <div><span>GENERATION STEP</span><strong>{tokens ? String(step).padStart(3, "0") : "—"}</strong></div>
            <div><span>FORWARD PASS</span><strong>{stats.computeMs === null ? "—" : stats.computeMs.toFixed(0)}<small>{stats.computeMs === null ? "" : " ms"}</small></strong></div>
            <div><span>MODEL LAYERS</span><strong>{model?.n_layer ?? "—"}</strong></div>
            <div><span>EXECUTION</span><strong className="readout-text">{model ? `${model.device}${model.dtype ? ` / ${model.dtype}` : ""}` : "Not connected"}</strong></div>
          </div>
          <AnomalyEkgPanel />
          <TokenGenerationAnimation />
        </section>
        <aside className="inspection-dock" aria-label="Model inspector" hidden={expanded}>
          <div className="dock-heading"><span className="eyebrow">EXPERIMENT DESK</span><span className="eyebrow">{running ? "RUNNING" : "PASSIVE VIEW"}</span></div>
          <div className="dock-tabs" role="tablist" aria-label="Inspection panels">{PANELS.map((name) => <button key={name} id={`tab-${name}`} role="tab" aria-selected={panel === name} aria-controls={`panel-${name}`} tabIndex={panel === name ? 0 : -1} onClick={() => setPanel(name)} onKeyDown={(e) => { const direction = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (direction) { e.preventDefault(); const next = PANELS[(PANELS.indexOf(name) + direction + PANELS.length) % PANELS.length]; setPanel(next); document.getElementById(`tab-${next}`)?.focus(); } }}>{name}</button>)}</div>
          <div className="dock-content">
            <div id="panel-Inspect" role="tabpanel" aria-labelledby="tab-Inspect" hidden={panel !== "Inspect"}><AttentionPanel /><PipelineStatus /></div>
            <div id="panel-Output" role="tabpanel" aria-labelledby="tab-Output" hidden={panel !== "Output"}><ProbabilityPanel /></div>
            <div id="panel-Runs" role="tabpanel" aria-labelledby="tab-Runs" hidden={panel !== "Runs"}><RunPicker /><div className="desk-note"><span className="eyebrow">REPRODUCIBLE OBSERVATION</span><p>Replay preserves the original readings. A recorded run stays labeled throughout playback.</p></div></div>
          </div>
          <div className="dock-footer"><span>OBSERVE / MEASURE / REPLAY</span><span>InsideAI</span></div>
        </aside>
      </main>
      <ErrorToast />
    </div>
  );
}
