"use client";

/**
 * The pipeline as a living flow diagram: every stage lights up as its real
 * event arrives, and shows a live example from the actual run — words → ids,
 * vectors, the current top candidate, the sampled token, the streaming text.
 */
import { visualizeToken } from "@/lib/format";
import { useSimStore } from "@/lib/store";
import type { Stage } from "@/lib/types";

const ORDER: Stage[] = [
  "tokenize",
  "embeddings",
  "positional",
  "layers",
  "logits",
  "sampled",
  "streaming",
];

/** Rows can cover several protocol stages (embeddings row includes position). */
function rowState(
  covers: Stage[],
  current: Stage,
  running: boolean
): "done" | "active" | "idle" {
  if (!running && (current === "done" || current === "streaming")) return "done";
  const ci = ORDER.indexOf(current);
  if (ci < 0) return "idle";
  const indices = covers.map((s) => ORDER.indexOf(s));
  if (indices.includes(ci)) return "active";
  if (Math.max(...indices) < ci) return "done";
  return "idle";
}

function Dot({ state }: { state: "done" | "active" | "idle" }) {
  return (
    <span
      className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full border transition-colors ${
        state === "active"
          ? "pulse-soft border-active bg-active"
          : state === "done"
            ? "border-prompt-tok bg-prompt-tok"
            : "border-grid bg-raised"
      }`}
    />
  );
}

function Row({
  state,
  label,
  children,
  last = false,
}: {
  state: "done" | "active" | "idle";
  label: string;
  children?: React.ReactNode;
  last?: boolean;
}) {
  return (
    <div className="flex gap-2.5">
      <div className="flex flex-col items-center">
        <Dot state={state} />
        {!last && (
          <span
            className={`w-px flex-1 transition-colors ${
              state === "done" ? "bg-prompt-tok/50" : "bg-grid"
            }`}
          />
        )}
      </div>
      <div className={`min-w-0 flex-1 pb-2 ${last ? "pb-0" : ""}`}>
        <div className={`text-xs font-medium ${state === "idle" ? "text-ink3" : "text-ink"}`}>
          {label}
        </div>
        {children && <div className="mt-0.5 text-[10px] leading-relaxed">{children}</div>}
      </div>
    </div>
  );
}

export default function PipelineStatus() {
  const stage = useSimStore((s) => s.stage);
  const running = useSimStore((s) => s.running);
  const step = useSimStore((s) => s.step);
  const activeLayer = useSimStore((s) => s.activeLayer);
  const modelInfo = useSimStore((s) => s.modelInfo);
  const maxSteps = useSimStore((s) => s.params.max_new_tokens);
  const tokens = useSimStore((s) => s.tokens);
  const prompt = useSimStore((s) => s.prompt);
  const logits = useSimStore((s) => s.logits);
  const sampled = useSimStore((s) => s.sampled);
  const text = useSimStore((s) => s.text);

  const nLayer = modelInfo?.n_layer ?? 0;
  const nEmbd = modelInfo?.n_embd ?? 0;

  const hasRun = tokens.length > 0;
  const promptTokens = tokens.filter((t) => t.kind === "prompt");
  const sampleTokens = promptTokens.slice(0, 3);
  const layersState = rowState(["layers"], stage, running);
  const top = logits?.topk[0] ?? null;

  return (
    <div className="glass pointer-events-auto rounded-2xl p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-ink2">
          Pipeline
        </h2>
        <span className="font-mono text-[10px] text-ink3">
          {running || stage === "done"
            ? `token ${Math.min(step + (running ? 1 : 0), maxSteps)}/${maxSteps} · seq ${tokens.length}`
            : "waiting for a prompt"}
        </span>
      </div>

      <Row state={hasRun ? "done" : "idle"} label="Prompt">
        {prompt ? (
          <span className="italic text-ink2">&ldquo;{prompt.slice(0, 42)}{prompt.length > 42 ? "…" : ""}&rdquo;</span>
        ) : (
          <span className="text-ink3">your words go in</span>
        )}
      </Row>

      <Row state={rowState(["tokenize"], stage, running)} label="Tokenizer">
        {sampleTokens.length > 0 ? (
          <span className="font-mono text-ink2">
            {sampleTokens.map((t, i) => (
              <span key={i}>
                {i > 0 && " · "}
                {visualizeToken(t.text)}
                <span className="text-ink3">→{t.id}</span>
              </span>
            ))}
            {promptTokens.length > 3 && <span className="text-ink3"> +{promptTokens.length - 3}</span>}
          </span>
        ) : (
          <span className="text-ink3">words → numbers (token ids)</span>
        )}
      </Row>

      <Row state={rowState(["embeddings", "positional"], stage, running)} label="Embeddings">
        <span className={hasRun ? "font-mono text-ink2" : "text-ink3"}>
          {hasRun
            ? `${tokens.length} tokens → ${nEmbd} numbers each · + position`
            : "each token becomes a vector"}
        </span>
      </Row>

      <Row state={layersState} label={`Transformer ×${nLayer || "…"}`}>
        <span className="text-ink3">attention + neurons, layer by layer</span>
        {nLayer > 0 && (
          <span className="mt-1 flex flex-wrap gap-[3px]">
            {Array.from({ length: nLayer }, (_, i) => {
              const active = activeLayer === i;
              const done =
                layersState === "done" || (layersState === "active" && activeLayer > i);
              return (
                <span
                  key={i}
                  title={`layer ${i}`}
                  className={`h-3.5 w-4 rounded-[3px] text-center font-mono text-[8px] leading-[14px] transition-colors ${
                    active
                      ? "bg-active text-white"
                      : done
                        ? "bg-accent/25 text-ink2"
                        : "bg-raised text-ink3"
                  }`}
                >
                  {i}
                </span>
              );
            })}
          </span>
        )}
      </Row>

      <Row state={rowState(["logits"], stage, running)} label="Logits">
        {top ? (
          <span className="font-mono text-ink2">
            best guess: “{visualizeToken(top.token)}” {(top.prob * 100).toFixed(1)}%
          </span>
        ) : (
          <span className="text-ink3">a score for every word it knows</span>
        )}
      </Row>

      <Row state={rowState(["sampled"], stage, running)} label="Sampling">
        {sampled ? (
          <span className="font-mono text-ink2">
            picked “{visualizeToken(sampled.token)}” (rank {sampled.rank})
          </span>
        ) : (
          <span className="text-ink3">one word is chosen</span>
        )}
      </Row>

      <Row state={rowState(["streaming"], stage, running)} label="Streaming" last>
        {text ? (
          <span className="font-mono text-ink2">
            …{text.slice(-26)}
            {running && <span className="pulse-soft text-active">▍</span>}
          </span>
        ) : (
          <span className="text-ink3">the answer appears, token by token</span>
        )}
      </Row>
    </div>
  );
}
