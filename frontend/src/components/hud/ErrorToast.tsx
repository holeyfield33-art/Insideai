"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

import { useSimStore } from "@/lib/store";

export default function ErrorToast() {
  const error = useSimStore((s) => s.error);
  const [dismissed, setDismissed] = useState<string | null>(null);

  useEffect(() => {
    if (error) setDismissed(null);
  }, [error]);

  const visible = error !== null && dismissed !== error;

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ y: 40, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: 40, opacity: 0 }}
          className="glass pointer-events-auto absolute bottom-24 left-1/2 z-20 flex -translate-x-1/2 items-center gap-2 rounded-xl border-danger/40 px-4 py-2.5"
        >
          <span aria-hidden className="text-danger">
            ⚠
          </span>
          <span className="text-xs text-ink">
            <b className="text-danger">error</b> · {error}
          </span>
          <button
            onClick={() => setDismissed(error)}
            className="ml-2 text-ink3 transition-colors hover:text-ink"
            aria-label="dismiss"
          >
            ✕
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
