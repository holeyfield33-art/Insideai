"use client";

import { useSimStore } from "@/lib/store";

export default function TopBar() {
  const connection = useSimStore((s) => s.connection);
  const model = useSimStore((s) => s.modelInfo);
  const theme = useSimStore((s) => s.theme);
  const setTheme = useSimStore((s) => s.setTheme);
  const replaying = useSimStore((s) => s.replaying);
  const running = useSimStore((s) => s.running);
  const liveModel = useSimStore((s) => s.liveModel);
  const status = connection !== "open" ? (connection === "connecting" ? "Connecting" : "Disconnected")
    : replaying ? "Recorded run" : liveModel === false ? "Replay only" : running ? "Acquiring" : "Ready";

  return (
    <header className="instrument-header">
      <a className="wordmark" href="#main" aria-label="InsideAI workbench">inside<span>ai</span><i aria-hidden="true">/</i></a>
      <div className="header-descriptor"><span>UNITARITY LABS</span><strong>Model observatory</strong></div>
      <div className="header-model"><span className="eyebrow">MODEL</span><strong title={model?.model}>{model?.model.split("/").pop() ?? "No model connected"}</strong></div>
      <div className="header-actions">
        <span className={`connection-state ${running ? "is-acquiring" : ""}`} role="status">{status}</span>
        <button className="instrument-button theme-button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>{theme === "dark" ? "Light" : "Dark"}</button>
      </div>
    </header>
  );
}
