import React from "react";
import Logo from "./Logo";
import { MoonIcon, SunIcon } from "./Icons";
import { motion } from "framer-motion";
import { useThemeSwitch } from "./Hooks/useThemeSwitch";

export default function Navbar() {
  const [mode, setMode] = useThemeSwitch();

  return (
    <header className="w-full px-8 sm:px-6 py-5 flex items-center justify-between
                       border-b border-dark/10 dark:border-light/10
                       bg-light/80 dark:bg-dark/80 backdrop-blur z-30 sticky top-0">
      <Logo />

      <nav className="flex items-center gap-3">
        <button
          aria-label="Toggle theme"
          onClick={() => setMode(mode === "light" ? "dark" : "light")}
          className="flex items-center justify-center rounded-full p-2
                     bg-dark/5 dark:bg-light/10 hover:bg-dark/10 dark:hover:bg-light/20
                     transition"
        >
          {mode === "dark" ? (
            <SunIcon className={"fill-dark"} />
          ) : (
            <MoonIcon className={"fill-light"} />
          )}
        </button>
      </nav>
    </header>
  );
}
