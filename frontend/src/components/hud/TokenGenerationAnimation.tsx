"use client";

import { useEffect, useRef } from "react";
import { visualizeToken } from "@/lib/format";
import { useSimStore } from "@/lib/store";

export default function TokenGenerationAnimation() {
  const tokens = useSimStore((s) => s.tokens);
  const generated = useSimStore((s) => s.generated);
  const text = useSimStore((s) => s.text);
  const reason = useSimStore((s) => s.endReason);
  const running = useSimStore((s) => s.running);
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => { if (scroller.current) scroller.current.scrollLeft = scroller.current.scrollWidth; }, [tokens.length, generated.length]);
  if (!tokens.length) return null;
  return (
    <section className="token-output" aria-label="Model output">
      <div className="output-heading"><span className="eyebrow">TOKEN STREAM</span><span className="eyebrow">{running ? "GENERATING" : reason ?? ""}</span></div>
      <div className="token-ribbon" ref={scroller}>
        <span className="eyebrow">INPUT</span>
        {tokens.filter((t) => t.kind === "prompt").map((t) => <span className="token-chip input-chip" key={`p${t.index}`} title={`Token ${t.id} · ${t.raw}`}>{visualizeToken(t.text) || "·"}</span>)}
        <span className="eyebrow">OUTPUT</span>
        {generated.map((t, i) => <span className="token-chip output-chip" key={`g${i}`} title={`Token ${t.id} · ${(t.prob * 100).toFixed(2)}% · rank ${t.rank}`}>{visualizeToken(t.token) || "·"}</span>)}
      </div>
      {text && <div className="generated-answer"><span className="eyebrow">RESPONSE</span><p>{text}</p></div>}
    </section>
  );
}
