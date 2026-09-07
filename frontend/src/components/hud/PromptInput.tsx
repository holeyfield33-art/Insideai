"use client";

import { useState } from "react";
import { useSimStore } from "@/lib/store";
import { simSocket } from "@/lib/ws";

export default function PromptInput() {
  const [prompt, setPrompt] = useState("");
  const connection = useSimStore((s) => s.connection);
  const running = useSimStore((s) => s.running);
  const liveModel = useSimStore((s) => s.liveModel);
  const params = useSimStore((s) => s.params);
  const setParams = useSimStore((s) => s.setParams);
  const model = useSimStore((s) => s.modelInfo);
  const online = connection === "open";
  const canRun = online && liveModel !== false && !!model && !!prompt.trim() && !running;
  const run = () => { if (canRun) simSocket.generate(prompt, params); };

  return (
    <div className="prompt-console">
      <div className="prompt-row">
        <label className="eyebrow" htmlFor="experiment-prompt">INPUT<br /><span>01</span></label>
        <textarea id="experiment-prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); run(); } }} rows={2} placeholder="Give the model a prompt to inspect…" />
        {running ? <button className="run-button stop-button" onClick={() => simSocket.stop()}>Stop run</button> : <button className="run-button" onClick={run} disabled={!canRun}>{!online ? "Disconnected" : liveModel === false ? "Replay only" : "Generate"}<span aria-hidden="true">↗</span></button>}
      </div>
      <details className="generation-settings">
        <summary><span>Generation settings</span><span className="settings-summary">{params.max_new_tokens} tokens · T {params.temperature.toFixed(2)} · {params.chat === "auto" ? "Chat" : "Raw"}</span></summary>
        <div className="settings-grid">
          <label>New tokens <output>{params.max_new_tokens}</output><input type="range" min="1" max={model?.max_new_tokens_cap ?? 96} value={params.max_new_tokens} onChange={(e) => setParams({ max_new_tokens: Number(e.target.value) })} /></label>
          <label>Temperature <output>{params.temperature.toFixed(2)}</output><input type="range" min="0" max="2" step="0.05" value={params.temperature} onChange={(e) => setParams({ temperature: Number(e.target.value) })} /></label>
          <label>Top-k <output>{params.top_k || "off"}</output><input type="range" min="0" max="200" value={params.top_k} onChange={(e) => setParams({ top_k: Number(e.target.value) })} /></label>
          <label>Top-p <output>{params.top_p.toFixed(2)}</output><input type="range" min="0.05" max="1" step="0.05" value={params.top_p} onChange={(e) => setParams({ top_p: Number(e.target.value) })} /></label>
          <label>Repetition penalty <output>{params.repetition_penalty.toFixed(2)}</output><input type="range" min="1" max="2" step="0.05" value={params.repetition_penalty} onChange={(e) => setParams({ repetition_penalty: Number(e.target.value) })} /></label>
          <label>Prompt mode<select value={params.chat} onChange={(e) => setParams({ chat: e.target.value as "auto" | "raw" })}><option value="auto">Chat template</option><option value="raw">Raw continuation</option></select></label>
          <label>Playback pacing<select value={params.speed} onChange={(e) => setParams({ speed: Number(e.target.value) })}><option value={1}>Cinematic</option><option value={0.4}>Fast</option><option value={0}>Instant</option></select></label>
          <label>Seed<input type="text" inputMode="numeric" value={params.seed} onChange={(e) => setParams({ seed: e.target.value })} placeholder="Random" /></label>
        </div>
        <p>Enter to generate · Shift + Enter for a new line. A fixed seed helps reproduce a run in the same environment.</p>
      </details>
    </div>
  );
}
