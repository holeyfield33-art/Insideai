"use client";

/**
 * 2D attention heatmap for one layer (head mean or a single head).
 *
 * Sequential encoding per the dataviz method: one hue (blue), monotone
 * lightness, dark-anchored — near-zero attention recedes into the surface,
 * bright = 1. A scale legend is always shown; hover reads out the exact
 * query→key pair and weight.
 */
import { useEffect, useMemo, useRef, useState } from "react";

import { visualizeToken } from "@/lib/format";
import { ATTN_RAMP, attnLUT } from "@/lib/palette";
import { useSimStore } from "@/lib/store";

const CANVAS_PX = 512;

export default function AttentionPanel() {
  const attention = useSimStore((s) => s.attention);
  const tokens = useSimStore((s) => s.tokens);
  const layerMeta = useSimStore((s) => s.layerMeta);
  const modelInfo = useSimStore((s) => s.modelInfo);
  const selectedLayer = useSimStore((s) => s.selectedLayer);
  const selectedHead = useSimStore((s) => s.selectedHead);
  const setSelectedLayer = useSimStore((s) => s.setSelectedLayer);
  const setSelectedHead = useSimStore((s) => s.setSelectedHead);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);

  const att = attention[selectedLayer] ?? null;
  const headsAvailable = att?.heads != null;
  const matrix = useMemo(() => {
    if (!att) return null;
    if (selectedHead >= 0 && att.heads) return att.heads[selectedHead] ?? att.mean;
    return att.mean;
  }, [att, selectedHead]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, CANVAS_PX, CANVAS_PX);
    if (!matrix || matrix.length === 0) return;

    const n = matrix.length;
    const off = document.createElement("canvas");
    off.width = n;
    off.height = n;
    const octx = off.getContext("2d");
    if (!octx) return;
    const img = octx.createImageData(n, n);
    for (let i = 0; i < n; i++) {
      const row = matrix[i];
      for (let j = 0; j < n; j++) {
        // Accumulated decode rows are ragged (row i has i+1 entries); the
        // missing upper-triangle cells are true zeros by causality.
        const v = row[j] ?? 0;
        const k = (i * n + j) * 4;
        img.data[k] = attnLUT[v * 3];
        img.data[k + 1] = attnLUT[v * 3 + 1];
        img.data[k + 2] = attnLUT[v * 3 + 2];
        img.data[k + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, CANVAS_PX, CANVAS_PX);
  }, [matrix]);

  const n = matrix?.length ?? 0;
  const hoverInfo =
    hover && matrix && hover.i < n && hover.j < n
      ? {
          from: tokens[hover.i]?.text ?? `#${hover.i}`,
          to: tokens[hover.j]?.text ?? `#${hover.j}`,
          w: (matrix[hover.i][hover.j] ?? 0) / 255,
        }
      : null;

  const meta = layerMeta[selectedLayer];
  const nLayer = modelInfo?.n_layer ?? 0;
  const nHead = modelInfo?.n_head ?? 0;

  return (
    <div className="glass pointer-events-auto rounded-2xl p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-ink2">
          Attention map
        </h2>
        <span className="font-mono text-[10px] text-ink3">
          L{selectedLayer} · {selectedHead === -1 ? "head μ" : `head ${selectedHead}`}
        </span>
      </div>

      {nLayer > 0 && (
        <div className="mb-1.5 flex flex-wrap items-center gap-1">
          <span className="w-10 text-[9px] uppercase tracking-wider text-ink3">layer</span>
          {Array.from({ length: nLayer }, (_, l) => (
            <button
              key={l}
              onClick={() => setSelectedLayer(l)}
              className={`h-5 w-6 rounded font-mono text-[9px] transition-colors ${
                selectedLayer === l
                  ? "bg-accent text-white"
                  : "bg-raised text-ink3 hover:text-ink"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
      )}
      {nHead > 0 && (
        <div className="mb-2 flex flex-wrap items-center gap-1">
          <span className="w-10 text-[9px] uppercase tracking-wider text-ink3">head</span>
          <button
            onClick={() => setSelectedHead(-1)}
            title="mean over heads"
            className={`h-5 w-6 rounded font-mono text-[9px] transition-colors ${
              selectedHead === -1
                ? "bg-accent text-white"
                : "bg-raised text-ink3 hover:text-ink"
            }`}
          >
            μ
          </button>
          {Array.from({ length: nHead }, (_, h) => (
            <button
              key={h}
              onClick={() => setSelectedHead(h)}
              disabled={!headsAvailable}
              title={
                headsAvailable
                  ? `head ${h} · entropy ${att?.headEntropy[h] ?? "–"} bits`
                  : "per-head maps unavailable (sequence over cap) — showing head mean"
              }
              className={`h-5 w-6 rounded font-mono text-[9px] transition-colors disabled:opacity-30 ${
                selectedHead === h
                  ? "bg-accent text-white"
                  : "bg-raised text-ink3 hover:text-ink"
              }`}
            >
              {h}
            </button>
          ))}
        </div>
      )}

      <div className="relative">
        <canvas
          ref={canvasRef}
          width={CANVAS_PX}
          height={CANVAS_PX}
          onMouseMove={(e) => {
            if (!n) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const j = Math.floor(((e.clientX - rect.left) / rect.width) * n);
            const i = Math.floor(((e.clientY - rect.top) / rect.height) * n);
            setHover({ i: Math.min(i, n - 1), j: Math.min(j, n - 1) });
          }}
          onMouseLeave={() => setHover(null)}
          className="aspect-square w-full rounded-lg border border-grid"
        />
        {matrix === null && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-ink3">
            attention arrives with the first layer
          </div>
        )}
        {hover && n > 0 && (
          <div
            className="pointer-events-none absolute ring-1 ring-ink/70"
            style={{
              left: `${(hover.j / n) * 100}%`,
              top: `${(hover.i / n) * 100}%`,
              width: `${100 / n}%`,
              height: `${100 / n}%`,
            }}
          />
        )}
      </div>

      <div className="mt-1.5 flex h-4 items-center gap-2">
        <span className="font-mono text-[9px] text-ink3">0</span>
        <div
          className="h-1.5 flex-1 rounded-full"
          style={{ background: `linear-gradient(to right, ${ATTN_RAMP.join(",")})` }}
        />
        <span className="font-mono text-[9px] text-ink3">1</span>
        <span className="text-[9px] text-ink3">attention</span>
      </div>

      <div className="mt-1 min-h-[1rem] font-mono text-[10px] text-ink3">
        {hoverInfo ? (
          <>
            “{visualizeToken(hoverInfo.from)}” → “{visualizeToken(hoverInfo.to)}” ·{" "}
            <span className="text-ink2">{hoverInfo.w.toFixed(3)}</span>
          </>
        ) : (
          "rows = query token · cols = key token"
        )}
      </div>

      {meta && (
        <div className="mt-1 border-t border-grid pt-1.5 font-mono text-[10px] text-ink3">
          ‖h‖ {meta.hiddenNorm} · Δresidual {meta.residualDelta}
          {att && selectedHead >= 0 && ` · H ${att.headEntropy[selectedHead]} bits`}
        </div>
      )}
    </div>
  );
}
