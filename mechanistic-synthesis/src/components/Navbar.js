import React from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import Logo from "./Logo";
import { MoonIcon, SunIcon } from "./Icons";
import { useThemeSwitch } from "./Hooks/useThemeSwitch";

const LINKS = [
  { href: "/", label: "Synthesis" },
  { href: "/platform", label: "Platform" },
];

export default function Navbar() {
  const [mode, setMode] = useThemeSwitch();
  const router = useRouter();
  const loggedOut = router.pathname === "/login";

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
    window.location.assign("/login");
  }

  return (
    <header className="w-full px-8 sm:px-6 py-5 flex items-center justify-between
                       border-b border-dark/10 dark:border-light/10
                       bg-light/80 dark:bg-dark/80 backdrop-blur z-30 sticky top-0">
      <Logo />

      <nav className="flex items-center gap-3">
        {!loggedOut &&
          LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              aria-current={router.pathname === l.href ? "page" : undefined}
              className={`text-sm px-2 py-1 rounded transition ${
                router.pathname === l.href
                  ? "text-dark dark:text-light font-medium"
                  : "text-dark/50 dark:text-light/50 hover:text-dark dark:hover:text-light"
              }`}
            >
              {l.label}
            </Link>
          ))}
        {!loggedOut && (
          <button
            onClick={logout}
            className="text-sm px-2 py-1 rounded text-dark/50 dark:text-light/50 hover:text-dark dark:hover:text-light transition"
          >
            Log out
          </button>
        )}
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
