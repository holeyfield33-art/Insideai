"use client";

import { motion } from "framer-motion";
import { useState } from "react";

import { useSimStore } from "@/lib/store";
import { simSocket } from "@/lib/ws";

const SUGGESTIONS = [
  "Explain quantum computing",
  "The meaning of life is",
  "Once upon a time, a robot",
];

const PACING = [
  { label: "Cinematic", value: 1 },
  { label: "Fast", value: 0.4 }, // default — realtime feel with visible stages
  { label: "Instant", value: 0 },
];

export default function PromptInput() {
  const [prompt, setPrompt] = useState("");
  const [showSettings, setShowSettings] = useState(false);

  const connection = useSimStore((s) => s.connection);
  const running = useSimStore((s) => s.running);
  const params = useSimStore((s) => s.params);
  const setParams = useSimStore((s) => s.setParams);
  const modelInfo = useSimStore((s) => s.modelInfo);
  const hasTokens = useSimStore((s) => s.tokens.length > 0);

  const online = connection === "open";
  const canRun = online && prompt.trim().length > 0 && !running;

  // Generating never steals the camera: the Auto Tour toggle alone decides
  // whether the view follows the pipeline.
  const run = () => {
    if (!canRun) return;
    simSocket.generate(prompt, params);
  };

  return (
    <motion.div
      initial={{ y: 24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: "spring", stiffness: 160, damping: 20 }}
      className="pointer-events-auto mx-auto w-full max-w-3xl"
    >
      {!hasTokens && !running && (
        <div className="mb-2 flex flex-wrap items-center justify-center gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setPrompt(s)}
              className="glass rounded-full px-3 py-1 text-xs text-ink2 transition-colors hover:text-ink"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="glass rounded-2xl p-3">
        <div className="flex items-end gap-2">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                run();
              }
            }}
            rows={1}
            placeholder={
              online ? "Enter a prompt to step inside the model…" : "waiting for backend…"
            }
            className="max-h-28 min-h-[2.5rem] flex-1 resize-none bg-transparent px-2 py-2 text-sm text-ink outline-none placeholder:text-ink3"
          />
          <button
            onClick={() => setShowSettings(!showSettings)}
            title="Generation settings"
            className={`rounded-xl px-3 py-2 text-sm transition-colors ${
              showSettings ? "bg-accent/20 text-ink" : "text-ink2 hover:text-ink"
            }`}
          >
            ⚙
          </button>
          {running ? (
            <motion.button
              whileTap={{ scale: 0.96 }}
              onClick={() => simSocket.stop()}
              className="rounded-xl border border-danger/60 px-4 py-2 text-sm font-medium text-danger transition-colors hover:bg-danger/10"
            >
              ■ Stop
            </motion.button>
          ) : (
            <motion.button
              whileTap={{ scale: 0.96 }}
              onClick={run}
              disabled={!canRun}
              className="rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-white shadow-md shadow-accent/30 transition-opacity disabled:opacity-40"
            >
              {online ? "Generate ⏎" : "Offline"}
            </motion.button>
          )}
        </div>

        {showSettings && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            className="overflow-hidden"
          >
            <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-3 border-t border-grid px-2 pt-3 text-xs text-ink2 sm:grid-cols-4">
              <label className="flex flex-col gap-1">
                <span>
                  new tokens <b className="text-ink">{params.max_new_tokens}</b>
                </span>
                <input
                  type="range"
                  min={1}
                  max={modelInfo?.max_new_tokens_cap ?? 64}
                  value={params.max_new_tokens}
                  onChange={(e) => setParams({ max_new_tokens: Number(e.target.value) })}
                  className="accent-accent"
                />
              </label>
              <label className="flex flex-col gap-1">
                <span>
                  temperature <b className="text-ink">{params.temperature.toFixed(2)}</b>
                </span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={params.temperature}
                  onChange={(e) => setParams({ temperature: Number(e.target.value) })}
                  className="accent-accent"
                />
              </label>
              <label className="flex flex-col gap-1">
                <span>
                  top-k <b className="text-ink">{params.top_k === 0 ? "off" : params.top_k}</b>
                </span>
                <input
                  type="range"
                  min={0}
                  max={200}
                  value={params.top_k}
                  onChange={(e) => setParams({ top_k: Number(e.target.value) })}
                  className="accent-accent"
                />
              </label>
              <label className="flex flex-col gap-1">
                <span>
                  top-p <b className="text-ink">{params.top_p.toFixed(2)}</b>
                </span>
                <input
                  type="range"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={params.top_p}
                  onChange={(e) => setParams({ top_p: Number(e.target.value) })}
                  className="accent-accent"
                />
              </label>

              <label className="flex flex-col gap-1">
                <span>
                  repetition <b className="text-ink">{params.repetition_penalty.toFixed(2)}</b>
                </span>
                <input
                  type="range"
                  min={1}
                  max={2}
                  step={0.05}
                  value={params.repetition_penalty}
                  onChange={(e) => setParams({ repetition_penalty: Number(e.target.value) })}
                  className="accent-accent"
                />
              </label>
              <div className="flex items-center gap-2">
                <span>mode</span>
                <div className="flex rounded-lg border border-grid p-0.5">
                  {(
                    [
                      { label: "Chat", value: "auto" },
                      { label: "Raw", value: "raw" },
                    ] as const
                  ).map((m) => (
                    <button
                      key={m.value}
                      title={
                        m.value === "auto"
                          ? "Instruct models answer via their chat template"
                          : "Plain continuation of your text"
                      }
                      onClick={() => setParams({ chat: m.value })}
                      className={`rounded-md px-2.5 py-1 transition-colors ${
                        params.chat === m.value
                          ? "bg-accent/25 text-ink"
                          : "text-ink3 hover:text-ink2"
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="col-span-2 flex items-center gap-2">
                <span>pacing</span>
                <div className="flex rounded-lg border border-grid p-0.5">
                  {PACING.map((p) => (
                    <button
                      key={p.label}
                      onClick={() => setParams({ speed: p.value })}
                      className={`rounded-md px-2.5 py-1 transition-colors ${
                        params.speed === p.value
                          ? "bg-accent/25 text-ink"
                          : "text-ink3 hover:text-ink2"
                      }`}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
              <label className="col-span-2 flex items-center gap-2 sm:col-span-4">
                <span>seed</span>
                <input
                  type="text"
                  value={params.seed}
                  onChange={(e) => setParams({ seed: e.target.value })}
                  placeholder="random"
                  className="w-24 rounded-lg border border-grid bg-transparent px-2 py-1 font-mono text-ink outline-none placeholder:text-ink3"
                />
                <span className="text-ink3">fixed seed → reproducible run</span>
              </label>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}
