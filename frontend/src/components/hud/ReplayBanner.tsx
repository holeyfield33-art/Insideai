"use client";

/**
 * Unmistakable REPLAY indicator. A recording streams the exact same events as
 * a live run — that's the whole point of replay — so nothing in the data tells
 * a viewer this isn't the model thinking right now. This banner is the honesty
 * guarantee: whenever a recorded run is on screen, it is impossible to miss.
 */
import { AnimatePresence, motion } from "framer-motion";

import { useSimStore } from "@/lib/store";

export default function ReplayBanner() {
  const replaying = useSimStore((s) => s.replaying);
  const running = useSimStore((s) => s.running);

  return (
    <AnimatePresence>
      {replaying && (
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -20, opacity: 0 }}
          className="pointer-events-none absolute left-1/2 top-14 z-40 -translate-x-1/2"
        >
          <div className="flex items-center gap-2 rounded-full border border-amber-400 bg-amber-500/95 px-4 py-1.5 text-xs font-bold uppercase tracking-widest text-black shadow-lg shadow-amber-500/40">
            <motion.span
              aria-hidden
              animate={{ opacity: [1, 0.25, 1] }}
              transition={{ duration: 1.1, repeat: Infinity }}
              className="text-sm leading-none"
            >
              ●
            </motion.span>
            <span>Replay — recorded run{running ? ", not live" : " (finished)"}</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
