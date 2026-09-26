import { test } from "node:test";
import assert from "node:assert/strict";

import { ChigutiroClient, ChigutiroError, measurement } from "../index.js";

function fakeFetch(handler: (url: string, init: RequestInit) => { status: number; body: unknown }) {
  const calls: { url: string; init: RequestInit }[] = [];
  const f = (async (url: string | URL, init?: RequestInit) => {
    calls.push({ url: String(url), init: init ?? {} });
    const { status, body } = handler(String(url), init ?? {});
    return new Response(JSON.stringify(body), { status });
  }) as typeof fetch;
  return { f, calls };
}

test("ingest batches and re-bases rejected indices", async () => {
  const { f, calls } = fakeFetch((_url, init) => {
    const n = (JSON.parse(String(init.body)) as { records: unknown[] }).records.length;
    return { status: 200, body: { accepted: n - 1, duplicates: 0, rejected: [{ index: 0, reason: "bad" }], committed: 7 } };
  });
  const client = new ChigutiroClient({ baseUrl: "http://x/", token: "t0k", batchSize: 2, fetch: f });
  const recs = [1, 2, 3].map((v) => measurement("s", "2026-09-01T00:00:00Z", "m", v));
  const report = await client.ingest(recs);
  assert.equal(calls.length, 2);
  assert.equal(calls[0]?.url, "http://x/ingest");
  assert.equal((calls[0]?.init.headers as Record<string, string>).authorization, "Bearer t0k");
  assert.deepEqual(report.rejected.map((r) => r.index), [0, 2]);
  assert.equal(report.accepted, 1);
});

test("errors carry status and server message", async () => {
  const { f } = fakeFetch(() => ({ status: 401, body: { error: "missing or wrong bearer token" } }));
  const client = new ChigutiroClient({ baseUrl: "http://x", token: "bad", fetch: f });
  await assert.rejects(client.status(), (e: unknown) => e instanceof ChigutiroError && e.status === 401 && /bearer token/.test(e.message));
});

test("consolidate treats 409 as a declined start, not an error", async () => {
  const { f } = fakeFetch(() => ({ status: 409, body: { started: false, reason: "not due" } }));
  const client = new ChigutiroClient({ baseUrl: "http://x", token: "t", fetch: f });
  assert.deepEqual(await client.consolidate(), { started: false, reason: "not due" });
});
