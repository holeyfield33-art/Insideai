"use client";

/**
 * Replay run picker. Lists runs the backend recorded (GET /runs) by prompt +
 * timestamp; selecting one replays it over the same WebSocket. Replay is
 * driven entirely by recorded events — no model is loaded server-side — so the
 * REPLAY banner (see ReplayBanner) is what keeps a recording from ever being
 * mistaken for a live run.
 */
import { useCallback, useEffect, useState } from "react";

import { useSimStore } from "@/lib/store";
import { fetchRuns, simSocket } from "@/lib/ws";

function shortTime(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function RunPicker() {
  const connection = useSimStore((s) => s.connection);
  const runs = useSimStore((s) => s.runs);
  const setRuns = useSimStore((s) => s.setRuns);
  const running = useSimStore((s) => s.running);
  const replaying = useSimStore((s) => s.replaying);
  const liveModel = useSimStore((s) => s.liveModel);
  const speed = useSimStore((s) => s.params.speed);

  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<string>("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setRuns(await fetchRuns());
    setLoading(false);
  }, [setRuns]);

  // Load once the socket is up, and whenever it reconnects.
  useEffect(() => {
    if (connection === "open") void refresh();
  }, [connection, refresh]);

  const replay = (id: string) => {
    if (!id) return;
    setSelected(id);
    simSocket.replay(id, speed);
  };

  return (
    <div className="glass pointer-events-auto rounded-2xl p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-ink2">
          Replay recorded run
        </h2>
        <button
          onClick={() => void refresh()}
          title="Reload the list of recorded runs"
          className="font-mono text-[10px] text-ink3 transition-colors hover:text-ink"
        >
          {loading ? "…" : "↻"}
        </button>
      </div>

      {liveModel === false && (
        <p className="mb-2 text-[10px] text-amber-400">
          Replay-only backend — no live model loaded.
        </p>
      )}

      {runs.length === 0 ? (
        <p className="text-[10px] text-ink3">
          {loading
            ? "loading runs…"
            : "No recorded runs yet. Set INSIDEAI_RECORD=1 and generate to capture one."}
        </p>
      ) : (
        <div className="flex max-h-48 flex-col gap-1 overflow-y-auto pr-1">
          {runs.map((r) => {
            const active = replaying && selected === r.id;
            return (
              <button
                key={r.id}
                onClick={() => replay(r.id)}
                disabled={running && !active}
                title={`${r.prompt}\n${r.model ?? ""} · ${r.steps} steps · ${r.id}`}
                className={`rounded-lg border px-2 py-1.5 text-left transition-colors disabled:opacity-40 ${
                  active
                    ? "border-amber-400/70 bg-amber-400/15"
                    : "border-grid hover:border-accent/60 hover:bg-accent/5"
                }`}
              >
                <div className="truncate text-xs text-ink">{r.prompt || "(empty prompt)"}</div>
                <div className="mt-0.5 flex items-center gap-2 font-mono text-[9px] text-ink3">
                  <span>{shortTime(r.timestamp)}</span>
                  <span>· {r.steps} steps</span>
                  {r.model && <span className="truncate">· {r.model.split("/").pop()}</span>}
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
