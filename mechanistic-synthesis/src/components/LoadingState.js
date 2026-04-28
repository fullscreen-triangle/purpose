import React from "react";
import { motion } from "framer-motion";

const MESSAGES = {
  triaging: "Reading your description…",
  synthesizing: "Composing the synthesis. This usually takes 30–90 seconds.",
};

export default function LoadingState({ phase = "triaging" }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="w-full max-w-3xl mx-auto py-16 flex flex-col items-center"
    >
      <div className="relative w-16 h-16 mb-6">
        <span className="absolute inset-0 rounded-full border-2 border-primary/20 dark:border-primaryDark/20" />
        <span
          className="absolute inset-0 rounded-full border-2 border-transparent
                     border-t-primary dark:border-t-primaryDark animate-spin"
        />
      </div>
      <p className="text-dark/70 dark:text-light/70 text-sm">
        {MESSAGES[phase] || "Working…"}
      </p>
    </motion.div>
  );
}
