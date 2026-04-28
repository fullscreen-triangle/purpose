import React from "react";

export default function Footer() {
  return (
    <footer
      className="w-full border-t border-dark/10 dark:border-light/10 py-6 px-8 sm:px-6
                 text-xs text-dark/50 dark:text-light/50 flex items-center justify-between"
    >
      <span>mechanistic-synthesis · {new Date().getFullYear()}</span>
      <span className="font-mono opacity-70">
        procedural learning for experimental research
      </span>
    </footer>
  );
}
