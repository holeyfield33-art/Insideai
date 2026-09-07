"use client";
import { useSimStore } from "@/lib/store";
export default function ReplayBanner() {
  const replaying = useSimStore((s) => s.replaying);
  const running = useSimStore((s) => s.running);
  if (!replaying) return null;
  return <div className="replay-label" role="status">REPLAY / {running ? "Recorded playback" : "Recorded run finished"}</div>;
}
