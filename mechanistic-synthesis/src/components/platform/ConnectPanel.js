import { useState } from "react";
import { DEFAULT_URL, forgetConnection, saveConnection } from "@/lib/purpose-client";
import { Button, Card, ErrorLine, Field, inputClass } from "./ui";

/** Where the local purpose server is, and its token. Shows connection state. */
export default function ConnectPanel({ conn, setConn, status }) {
  const [url, setUrl] = useState(conn.url);
  const [token, setToken] = useState(conn.token);
  const [open, setOpen] = useState(!conn.token);

  function connect(e) {
    e.preventDefault();
    const next = { url: url.trim() || DEFAULT_URL, token: token.trim() };
    saveConnection(next);
    setConn(next);
    setOpen(false);
  }

  function forget() {
    forgetConnection();
    setToken("");
    setConn({ url: DEFAULT_URL, token: "" });
    setOpen(true);
  }

  const dot = status.ok
    ? "bg-primaryDark"
    : status.checking
      ? "bg-dark/30 dark:bg-light/30"
      : "bg-primary";

  return (
    <Card>
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3 min-w-0">
          <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${dot}`} aria-hidden="true" />
          <p className="text-sm text-dark dark:text-light truncate">
            {status.ok
              ? `Connected to purpose ${status.version} at ${conn.url}${status.jobRunning ? " · training" : ""}`
              : status.checking
                ? "Checking purpose…"
                : conn.token
                  ? "Not connected"
                  : "Connect to your local purpose"}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setOpen((o) => !o)}>
            {open ? "Close" : "Connection"}
          </Button>
          {conn.token && (
            <Button variant="secondary" onClick={forget}>
              Forget token
            </Button>
          )}
        </div>
      </div>
      <ErrorLine error={!status.ok && !status.checking && conn.token ? status.error : ""} />

      {open && (
        <form onSubmit={connect} className="mt-5 grid grid-cols-2 sm:grid-cols-1 gap-4">
          <Field label="purpose address" hint="Where `purpose serve` listens on this laptop.">
            <input className={inputClass} value={url} onChange={(e) => setUrl(e.target.value)} />
          </Field>
          <Field label="Token" hint="Printed by `purpose serve` on start. Stored only in this browser.">
            <input
              className={inputClass}
              type="password"
              autoComplete="off"
              value={token}
              onChange={(e) => setToken(e.target.value)}
            />
          </Field>
          <div className="col-span-2 sm:col-span-1 text-xs text-dark/50 dark:text-light/50 leading-relaxed">
            Start it with:{" "}
            <code className="font-mono">
              purpose serve --origins {typeof window !== "undefined" ? window.location.origin : "https://your-site"}
            </code>
          </div>
          <div className="col-span-2 sm:col-span-1">
            <Button type="submit" disabled={!token.trim()}>
              Connect
            </Button>
          </div>
        </form>
      )}
    </Card>
  );
}
