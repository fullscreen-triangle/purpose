// Browser client for the local `purpose serve` (purpose-factory/src/server.rs).
//
// Called from the page itself, never from a Vercel function: the server runs
// on this laptop at 127.0.0.1, which only a browser on the laptop can reach.
// Mirrors purpose-factory-ts/src/http-client.ts, so the shapes match.
// The URL and token are kept in this browser's localStorage only.

const KEY = "ms-purpose-v1";
export const DEFAULT_URL = "http://127.0.0.1:8420";

export function loadConnection() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) || "{}");
    return { url: raw.url || DEFAULT_URL, token: raw.token || "" };
  } catch {
    return { url: DEFAULT_URL, token: "" };
  }
}

export function saveConnection(conn) {
  try {
    localStorage.setItem(KEY, JSON.stringify({ url: conn.url, token: conn.token }));
  } catch {
    /* storage blocked: the connection lasts for this page only */
  }
}

export function forgetConnection() {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* nothing stored */
  }
}

export class PurposeError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function call(conn, path, { method = "GET", json, form, raw } = {}) {
  const headers = { Authorization: `Bearer ${conn.token}` };
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form;
  }

  let res;
  try {
    res = await fetch(`${conn.url.replace(/\/+$/, "")}${path}`, { method, headers, body });
  } catch {
    throw new PurposeError(
      "Cannot reach purpose. Is `purpose serve` running, is this site's address in its " +
        "--origins, and did the browser allow local-network access?",
      0,
    );
  }
  if (res.status === 401) throw new PurposeError("purpose rejected the token", 401);
  if (!res.ok) {
    let msg = `purpose answered ${res.status}`;
    try {
      const j = await res.json();
      if (j.error) msg = j.error;
    } catch {
      /* not JSON */
    }
    throw new PurposeError(msg, res.status);
  }
  if (raw) return res;
  return res.json();
}

export const health = (conn) => call(conn, "/health");
export const planRun = (conn, req) => call(conn, "/plan", { method: "POST", json: req });
export const listModels = (conn) => call(conn, "/themes");
export const listSources = (conn, theme) => call(conn, `/themes/${encodeURIComponent(theme)}/sources`);
export const listJobs = (conn) => call(conn, "/jobs");
export const getJob = (conn, id) => call(conn, `/jobs/${encodeURIComponent(id)}`);
export const jobLog = (conn, id, tail = 200) => call(conn, `/jobs/${encodeURIComponent(id)}/log?tail=${tail}`);
export const cancelJob = (conn, id) => call(conn, `/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });

/** `request`: { theme, model: {kind:"scratch"} | {kind:"pretrained", repo, block_size?}, training: {...} } */
export const submitJob = (conn, request) => call(conn, "/jobs", { method: "POST", json: request });

export function uploadSources(conn, theme, files) {
  const form = new FormData();
  for (const f of files) form.append("files", f, f.name);
  return call(conn, `/themes/${encodeURIComponent(theme)}/sources`, { method: "POST", form });
}

/** Saves one exported model file through the browser's download. */
export async function downloadModelFile(conn, theme, file) {
  const res = await call(conn, `/themes/${encodeURIComponent(theme)}/model/${file}`, { raw: true });
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = `${theme}-${file}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
