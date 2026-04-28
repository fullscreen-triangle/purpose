import React, { useState } from "react";
import { motion } from "framer-motion";

export default function FollowupPanel({ summary, questions, onSubmit, onSkip }) {
  const [answers, setAnswers] = useState(() => questions.map(() => ""));
  const [skips, setSkips] = useState(() => questions.map(() => false));

  function setAnswer(i, value) {
    const next = [...answers];
    next[i] = value;
    setAnswers(next);
  }

  function setSkip(i, value) {
    const next = [...skips];
    next[i] = value;
    setSkips(next);
  }

  function handleSubmit(e) {
    e.preventDefault();
    const followups = questions.map((q, i) => ({
      question: q,
      answer: skips[i] ? "" : answers[i],
    }));
    onSubmit(followups);
  }

  return (
    <motion.form
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      onSubmit={handleSubmit}
      className="w-full max-w-3xl mx-auto"
    >
      {summary && (
        <div className="mb-6 p-4 rounded-md bg-primary/5 dark:bg-primaryDark/5 border border-primary/20 dark:border-primaryDark/20">
          <p className="text-sm text-dark/70 dark:text-light/70">
            <span className="font-medium text-dark dark:text-light">
              I read this as:
            </span>{" "}
            {summary}
          </p>
        </div>
      )}

      <p className="text-sm uppercase tracking-wider text-dark/60 dark:text-light/60 mb-3 font-medium">
        A few clarifications
      </p>

      <div className="space-y-5">
        {questions.map((q, i) => (
          <div
            key={i}
            className="p-4 rounded-lg border border-dark/15 dark:border-light/15"
          >
            <p className="text-dark dark:text-light mb-2">{q}</p>
            <textarea
              value={answers[i]}
              onChange={(e) => setAnswer(i, e.target.value)}
              disabled={skips[i]}
              rows={2}
              className="w-full p-3 rounded-md border border-dark/10 dark:border-light/10
                         bg-light dark:bg-dark text-dark dark:text-light
                         placeholder:text-dark/30 dark:placeholder:text-light/30
                         focus:outline-none focus:ring-2 focus:ring-primary/40
                         font-mont text-sm resize-y disabled:opacity-40"
              placeholder="Your answer…"
            />
            <label className="mt-2 inline-flex items-center text-xs text-dark/50 dark:text-light/50 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={skips[i]}
                onChange={(e) => setSkip(i, e.target.checked)}
                className="mr-2 accent-primary dark:accent-primaryDark"
              />
              Skip — not relevant or unknown
            </label>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between mt-6">
        <button
          type="button"
          onClick={onSkip}
          className="text-sm text-dark/60 dark:text-light/60 hover:text-dark dark:hover:text-light transition"
        >
          Skip all and synthesize anyway
        </button>
        <button
          type="submit"
          className="px-6 py-2.5 rounded-md bg-dark text-light dark:bg-light dark:text-dark
                     font-medium hover:opacity-90 transition"
        >
          Continue
        </button>
      </div>
    </motion.form>
  );
}
