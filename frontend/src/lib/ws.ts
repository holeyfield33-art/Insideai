"use client";

/** Reconnecting websocket client. Parses server events into the store. */

import { useSimStore } from "./store";
import type { GenerationParamsUI } from "./store";
import type { RunSummary, ServerEvent } from "./types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://127.0.0.1:8000/ws";

/** Derive the HTTP base (for GET /runs) from the WebSocket URL: ws->http,
 *  wss->https, dropping the trailing /ws path. */
function httpBase(): string {
  try {
    const u = new URL(WS_URL);
    u.protocol = u.protocol === "wss:" ? "https:" : "http:";
    u.pathname = u.pathname.replace(/\/ws\/?$/, "");
    return u.origin + u.pathname;
  } catch {
    return "http://127.0.0.1:8000";
  }
}

/** Fetch the list of recorded runs available for replay. Returns [] on error
 *  (e.g. an older backend without the /runs endpoint) so the UI degrades
 *  gracefully rather than throwing. */
export async function fetchRuns(): Promise<RunSummary[]> {
  try {
    const res = await fetch(`${httpBase()}/runs`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = (await res.json()) as RunSummary[];
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

class SimSocket {
  private ws: WebSocket | null = null;
  private retryMs = 1000;
  private closedByUser = false;
  private queue: ServerEvent[] = [];
  private flushScheduled = false;

  /** Batch incoming events per animation frame: decode steps arrive in
   *  bursts (one event per layer), and applying them one render at a time
   *  is what makes the UI stutter. One flush per frame = one React render.
   *  rAF races a timeout fallback — browsers throttle rAF for hidden or
   *  occluded tabs, and the stream must keep flowing there too. */
  private scheduleFlush(): void {
    if (this.flushScheduled) return;
    this.flushScheduled = true;
    let ran = false;
    const run = () => {
      if (ran) return;
      ran = true;
      this.flushScheduled = false;
      const events = this.queue;
      this.queue = [];
      const apply = useSimStore.getState().applyEvent;
      for (const ev of events) apply(ev);
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(run);
    setTimeout(run, 50);
  }

  connect(): void {
    if (typeof window === "undefined") return;
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    this.closedByUser = false;
    useSimStore.getState().setConnection("connecting");

    const ws = new WebSocket(WS_URL);
    this.ws = ws;

    ws.onopen = () => {
      this.retryMs = 1000;
      useSimStore.getState().setConnection("open");
    };

    ws.onmessage = (msg) => {
      try {
        const ev = JSON.parse(msg.data as string) as ServerEvent;
        this.queue.push(ev);
        this.scheduleFlush();
      } catch {
        // ignore malformed frames
      }
    };

    ws.onclose = () => {
      useSimStore.getState().setConnection("closed");
      this.ws = null;
      if (!this.closedByUser) {
        const delay = this.retryMs;
        this.retryMs = Math.min(this.retryMs * 1.6, 10_000);
        window.setTimeout(() => this.connect(), delay);
      }
    };

    ws.onerror = () => ws.close();
  }

  disconnect(): void {
    this.closedByUser = true;
    this.ws?.close();
  }

  private send(payload: object): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
      return true;
    }
    return false;
  }

  generate(prompt: string, params: GenerationParamsUI): boolean {
    const seed = params.seed.trim() === "" ? null : Number(params.seed);
    if (useSimStore.getState().liveModel === false) return false;
    const ok = this.send({
      type: "generate",
      prompt,
      max_new_tokens: params.max_new_tokens,
      temperature: params.temperature,
      top_k: params.top_k,
      top_p: params.top_p,
      repetition_penalty: params.repetition_penalty,
      chat_mode: params.chat,
      seed: Number.isFinite(seed as number) ? seed : null,
      speed: params.speed,
    });
    if (ok) useSimStore.getState().setReplaying(false);
    return ok;
  }

  /** Replay a recorded run over the same protocol. The events are identical
   *  to a live run, so the client flags the stream as a replay here — that
   *  flag is the only thing distinguishing a recording from the real thing. */
  replay(id: string, speed: number): boolean {
    const ok = this.send({ type: "replay", id, speed });
    if (ok) useSimStore.getState().setReplaying(true);
    return ok;
  }

  stop(): boolean {
    return this.send({ type: "stop" });
  }
}

export const simSocket = new SimSocket();
