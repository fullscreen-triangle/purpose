import React from "react";
import { motion } from "framer-motion";

/**
 * Subtle indicator shown when one or more knowledge packs are active in
 * the current synthesis. Lets the user see, without effort, that the
 * model is drawing on specialist material rather than generic literature.
 */
export default function PackBadge({ packs = [] }) {
  if (!packs || packs.length === 0) return null;
  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full
                 bg-primary/10 dark:bg-primaryDark/15
                 border border-primary/30 dark:border-primaryDark/30
                 text-xs text-primary dark:text-primaryDark"
    >
      <span
        className="inline-block w-1.5 h-1.5 rounded-full
                   bg-primary dark:bg-primaryDark animate-pulse"
        aria-hidden
      />
      drawing on{" "}
      {packs.map((p, i) => (
        <span key={p.id} className="font-medium">
          {p.label}
          {i < packs.length - 1 ? ", " : ""}
        </span>
      ))}
    </motion.div>
  );
}
