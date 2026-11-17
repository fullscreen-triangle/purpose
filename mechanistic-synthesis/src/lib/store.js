// The board's notes store: an Upstash Redis database reached over its REST
// API (no client library). Add it to the Vercel project from the Marketplace
// (Storage -> Upstash for Redis); the integration sets KV_REST_API_URL and
// KV_REST_API_TOKEN. UPSTASH_REDIS_REST_URL / _TOKEN work too.

export class NotConfigured extends Error {}

function config() {
  const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;
  return url && token ? { url: url.replace(/\/+$/, ""), token } : null;
}

export const configured = () => config() !== null;

/** Runs one Redis command, e.g. ["RPUSH", "key", "value"], and returns its result. */
export async function redis(command) {
  const c = config();
  if (!c) throw new NotConfigured("no notes store");
  const res = await fetch(c.url, {
    method: "POST",
    headers: { Authorization: `Bearer ${c.token}`, "Content-Type": "application/json" },
    body: JSON.stringify(command),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || body.error) throw new Error(body.error || `notes store answered ${res.status}`);
  return body.result;
}
