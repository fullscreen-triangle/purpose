import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

const PLACEHOLDER = `Describe your experiment in your own words.

You might write about:
  • what you're studying and why
  • the question you're trying to answer
  • how you plan to measure or observe
  • what you expect, and what you're worried about
  • anything else that feels relevant

There is no required structure. Write as much as is useful — the system will ask follow-up questions if it needs more.`;

export default function ExperimentInput({ initial = "", disabled = false, onSubmit }) {
  const [text, setText] = useState(initial);
  const [error, setError] = useState("");

  useEffect(() => {
    if (initial) setText(initial);
  }, [initial]);

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = text.trim();
    if (trimmed.length < 20) {
      setError("Please describe your experiment in at least a sentence or two.");
      return;
    }
    setError("");
    onSubmit(trimmed);
  }

  return (
    <motion.form
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      onSubmit={handleSubmit}
      className="w-full max-w-3xl mx-auto"
    >
      <label
        htmlFor="experiment-input"
        className="block text-sm uppercase tracking-wider text-dark/60 dark:text-light/60 mb-3 font-medium"
      >
        Your experiment
      </label>
      <textarea
        id="experiment-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={PLACEHOLDER}
        rows={14}
        disabled={disabled}
        className="w-full p-5 rounded-lg border border-dark/15 dark:border-light/15
                   bg-light dark:bg-dark text-dark dark:text-light
                   placeholder:text-dark/30 dark:placeholder:text-light/30
                   focus:outline-none focus:ring-2 focus:ring-primary/40
                   font-mont text-base leading-relaxed
                   resize-y min-h-[280px] disabled:opacity-50"
      />
      {error && (
        <p className="mt-2 text-sm text-primary dark:text-primaryDark">{error}</p>
      )}
      <div className="flex items-center justify-between mt-4">
        <p className="text-xs text-dark/50 dark:text-light/40">
          {text.length} characters
        </p>
        <button
          type="submit"
          disabled={disabled}
          className="px-6 py-2.5 rounded-md bg-dark text-light dark:bg-light dark:text-dark
                     font-medium hover:opacity-90 transition disabled:opacity-50
                     disabled:cursor-not-allowed"
        >
          Synthesize
        </button>
      </div>
    </motion.form>
  );
}
