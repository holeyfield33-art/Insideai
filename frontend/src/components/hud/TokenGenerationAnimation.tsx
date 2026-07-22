"use client";

/**
 * Live token stream: prompt tokens (blue) flow into generated tokens
 * (magenta) as the model emits them. The two colors are the validated
 * identity pair; the PROMPT/OUTPUT section labels are the secondary encoding.
 */
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

import { visualizeToken } from "@/lib/format";
import { useSimStore } from "@/lib/store";

function Chip({
  text,
  kind,
  title,
}: {
  text: string;
  kind: "prompt" | "generated" | "pending";
  title?: string;
}) {
  const styles =
    kind === "prompt"
      ? "border-prompt-tok/50 bg-prompt-tok/[0.12]"
      : kind === "generated"
        ? "border-gen-tok/50 bg-gen-tok/[0.12]"
        : "border-accent/60 bg-accent/[0.14] pulse-soft";
  return (
    <motion.span
      initial={{ opacity: 0, y: 8, scale: 0.85 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.85 }}
      transition={{ type: "spring", stiffness: 300, damping: 22 }}
      title={title}
      className={`inline-flex shrink-0 items-center whitespace-pre rounded-md border px-1.5 py-0.5 font-mono text-xs text-ink ${styles}`}
    >
      {visualizeToken(text) || "·"}
    </motion.span>
  );
}

export default function TokenGenerationAnimation() {
  const tokens = useSimStore((s) => s.tokens);
  const generated = useSimStore((s) => s.generated);
  const pending = useSimStore((s) => s.pendingSample);
  const endReason = useSimStore((s) => s.endReason);
  const text = useSimStore((s) => s.text);
  const running = useSimStore((s) => s.running);
  const scroller = useRef<HTMLDivElement>(null);
  const textScroller = useRef<HTMLDivElement>(null);

  // Typewriter: the displayed answer catches up to the real streamed text a
  // couple of characters per frame — the text itself is never altered.
  const [shown, setShown] = useState("");
  useEffect(() => {
    const id = setInterval(() => {
      setShown((s) => {
        const target = useSimStore.getState().text;
        if (!target) return s === "" ? s : "";
        if (!target.startsWith(s)) return ""; // new run — restart the reveal
        if (s.length >= target.length) return s;
        return target.slice(0, s.length + 3);
      });
    }, 50);
    return () => clearInterval(id);
  }, []);

  const promptTokens = tokens.filter((t) => t.kind === "prompt");

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollLeft = el.scrollWidth;
    const tl = textScroller.current;
    if (tl) tl.scrollTop = tl.scrollHeight;
  }, [tokens.length, generated.length, pending, shown]);

  if (promptTokens.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="glass pointer-events-auto mx-auto w-full max-w-5xl rounded-2xl px-3 py-2"
    >
      <div ref={scroller} className="flex items-center gap-1.5 overflow-x-auto pb-1">
        <span className="sticky left-0 z-10 shrink-0 bg-surface pr-1.5 text-[9px] font-semibold uppercase tracking-widest text-ink3">
          prompt
        </span>
        <AnimatePresence initial={false}>
          {promptTokens.map((t) => (
            <Chip key={`p${t.index}`} text={t.text} kind="prompt" title={`id ${t.id} · ${t.raw}`} />
          ))}
        </AnimatePresence>

        <span className="shrink-0 px-1 text-ink3">→</span>
        <span className="shrink-0 bg-surface pr-1.5 text-[9px] font-semibold uppercase tracking-widest text-ink3">
          output
        </span>
        <AnimatePresence initial={false}>
          {generated.map((t, i) => (
            <Chip
              key={`g${i}`}
              text={t.token}
              kind="generated"
              title={`id ${t.id} · p=${(t.prob * 100).toFixed(1)}% · rank ${t.rank}`}
            />
          ))}
          {pending && <Chip key="pending" text={pending.token} kind="pending" title="sampling…" />}
        </AnimatePresence>

        {endReason && endReason !== "max_tokens" && (
          <span className="shrink-0 rounded-full border border-grid px-2 py-0.5 text-[9px] uppercase tracking-widest text-ink3">
            {endReason}
          </span>
        )}
      </div>

      {text && (
        <div
          ref={textScroller}
          className="mt-1 max-h-16 overflow-y-auto whitespace-pre-wrap border-t border-grid px-1 pt-1 text-xs leading-relaxed text-ink2"
        >
          <span className="mr-1.5 select-none text-[9px] font-semibold uppercase tracking-widest text-ink3">
            answer
          </span>
          {shown}
          {(running || shown.length < text.length) && (
            <span className="pulse-soft text-active">▍</span>
          )}
        </div>
      )}
    </motion.div>
  );
}
