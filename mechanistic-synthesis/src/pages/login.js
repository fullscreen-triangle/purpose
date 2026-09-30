import Head from "next/head";
import { useRouter } from "next/router";
import { useState } from "react";

export default function Login() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error || `login failed (${res.status})`);
      const next = typeof router.query.next === "string" && router.query.next.startsWith("/")
        ? router.query.next
        : "/platform";
      window.location.assign(next);
    } catch (err) {
      setError(err.message);
      setPassword("");
      setBusy(false);
    }
  }

  return (
    <>
      <Head>
        <title>Log in · mechanistic-synthesis</title>
      </Head>
      <div className="w-full min-h-[calc(100vh-180px)] px-8 sm:px-6 py-24 flex justify-center">
        <form onSubmit={submit} className="w-full max-w-sm">
          <h1 className="text-2xl font-semibold tracking-tight text-dark dark:text-light mb-2">Log in</h1>
          <p className="text-sm text-dark/60 dark:text-light/60 mb-8">
            This site and its model platform are private.
          </p>
          <label htmlFor="password" className="block text-sm font-medium text-dark/80 dark:text-light/80 mb-2">
            Password
          </label>
          <input
            id="password"
            type="password"
            autoComplete="current-password"
            autoFocus
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-3 py-2 rounded-md bg-dark/5 dark:bg-light/5 border border-dark/10
                       dark:border-light/10 text-dark dark:text-light focus:outline-none
                       focus:border-primary dark:focus:border-primaryDark"
          />
          {error && (
            <p role="alert" className="mt-3 text-sm text-primary dark:text-primaryDark">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={busy || !password}
            className="mt-6 w-full px-5 py-2 rounded-md bg-dark text-light dark:bg-light dark:text-dark
                       font-medium hover:opacity-90 transition disabled:opacity-40"
          >
            {busy ? "Checking…" : "Log in"}
          </button>
        </form>
      </div>
    </>
  );
}
