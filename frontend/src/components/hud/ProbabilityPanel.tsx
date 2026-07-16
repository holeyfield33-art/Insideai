"use client";

/**
 * Next-token probability panel.
 *
 * Chart rules (dataviz method): nominal candidates → one series, one hue
 * (accent, validated vs surface); bar = thin mark, 4px rounded data-end,
 * square at the left baseline; track = darker step of the same ramp; token
 * text and values wear ink tokens, never the series color. The sampled row
 * is a selection state (ring + dot + label), not a second series.
 */
import { AnimatePresence, motion } from "framer-motion";

import { pct, visualizeToken } from "@/lib/format";
import { useSimStore } from "@/lib/store";

export default function ProbabilityPanel() {
  const logits = useSimStore((s) => s.logits);
  const sampled = useSimStore((s) => s.sampled);

  return (
    <div className="glass pointer-events-auto rounded-2xl p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-ink2">
          Next-token probabilities
        </h2>
        {logits && (
          <span className="font-mono text-[10px] text-ink3">
            top-{logits.topk.length} of {logits.vocab_size.toLocaleString()}
          </span>
        )}
      </div>

      {!logits ? (
        <p className="py-6 text-center text-xs text-ink3">
          run a prompt — the model&apos;s live distribution appears here
        </p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {/* popLayout: exiting rows leave the flow immediately, so burst
              updates (instant pacing) never leave ghost gaps in the list */}
          <AnimatePresence initial={false} mode="popLayout">
            {logits.topk.map((entry) => {
              const isSampled = sampled !== null && sampled.id === entry.id;
              return (
                <motion.div
                  key={entry.id}
                  layout
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  title={`id ${entry.id} · logit ${entry.logit} · raw ${entry.raw}`}
                  className={`group rounded-lg px-1.5 py-1 ${
                    isSampled ? "ring-1 ring-accent/80 bg-accent/[0.07]" : ""
                  }`}
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="max-w-[11rem] truncate font-mono text-xs text-ink">
                      {visualizeToken(entry.token) || "·"}
                    </span>
                    <span className="flex items-center gap-1.5 font-mono text-[11px] text-ink2">
                      {isSampled && (
                        <span className="flex items-center gap-1 text-[9px] uppercase tracking-wider text-ink2">
                          <span className="h-1.5 w-1.5 rounded-full bg-accent" />
                          sampled
                        </span>
                      )}
                      {pct(entry.prob)}
                    </span>
                  </div>
                  <div className="mt-1 h-[8px] w-full rounded-r-[4px] bg-accent-track">
                    <motion.div
                      className="h-full rounded-r-[4px] bg-accent"
                      initial={false}
                      animate={{ width: `${Math.max(entry.prob * 100, 0.8)}%` }}
                      transition={{ type: "spring", stiffness: 170, damping: 26 }}
                    />
                  </div>
                  <div className="hidden pt-0.5 font-mono text-[9px] text-ink3 group-hover:block">
                    logit {entry.logit} · id {entry.id}
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>

          <div className="mt-1 flex items-baseline justify-between border-t border-grid pt-2 font-mono text-[10px] text-ink3">
            <span>entropy {logits.entropy_bits} bits</span>
            {sampled && <span>{sampled.strategy}</span>}
          </div>
          {sampled && sampled.rank > logits.topk.length && (
            <div className="font-mono text-[10px] text-ink3">
              sampled “{visualizeToken(sampled.token)}” came from rank #{sampled.rank} (
              {pct(sampled.prob)}) — outside the top-{logits.topk.length}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
