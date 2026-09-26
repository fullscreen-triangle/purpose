// Drives the real Rust server through the bridge. Skipped when the binary
// has not been built (`cargo build` at the chigutiro root) and CHIGUTIRO_BIN
// is unset.

import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { ChigutiroClient, contact, event, fromBankTransactions, fromGarminSummary, fromGmailMessages, measurements } from "../index.js";

const here = fileURLToPath(new URL(".", import.meta.url));
const exe = process.platform === "win32" ? "chigutiro.exe" : "chigutiro";
const bin = process.env.CHIGUTIRO_BIN ?? resolve(here, "../../../target/debug", exe);

async function freePort(): Promise<number> {
  return new Promise((ok, fail) => {
    const s = createServer();
    s.once("error", fail);
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as { port: number }).port;
      s.close(() => ok(port));
    });
  });
}

test("bridge ↔ server round trip", { skip: !existsSync(bin) && `no binary at ${bin}` }, async () => {
  const data = mkdtempSync(join(tmpdir(), "chigutiro-e2e-"));
  const port = await freePort();
  const token = "e2e-token-0123456789abcdef";
  const key = "11".repeat(32);
  const child = spawn(bin, ["serve", "--data", data, "--port", String(port)], {
    env: { ...process.env, CHIGUTIRO_TOKEN: token, CHIGUTIRO_KEY: key, CHIGUTIRO_OLLAMA_MODEL: "", RUST_LOG: "warn" },
    stdio: ["ignore", "ignore", "pipe"],
  });
  let stderr = "";
  child.stderr.on("data", (d) => (stderr += d));
  const client = new ChigutiroClient({ baseUrl: `http://127.0.0.1:${port}`, token, timeoutMs: 5000 });
  try {
    for (let i = 0; ; i++) {
      try {
        await client.health();
        break;
      } catch {
        if (i > 100) throw new Error(`server did not start: ${stderr}`);
        await new Promise((r) => setTimeout(r, 50));
      }
    }

    const today = new Date();
    const yesterday = new Date(today.getTime() - 86_400_000);
    const report = await client.ingest([
      ...fromGarminSummary({ hrv: 58, sleep_hours: 7.1 }, yesterday),
      ...measurements("track-log", yesterday, { hrv: [60, "ms"], "100m time": [11.42, "s"] }, "s1"),
      ...measurements("polar", yesterday, { hrv: [59, "ms"] }, "p1"),
      ...fromBankTransactions([{ date: yesterday.toISOString().slice(0, 10), payee: "REWE", amount: -42.1, bank: "dkb" }]).records,
      ...fromGmailMessages(
        [{ uid: "g1", subject: "Cluster", from: "m@uni.de", fromAddress: "m@uni.de", date: yesterday.toISOString(), body: "The cluster allocation for the enzyme screen is approved." }],
        ["me@example.org"],
      ),
      contact("hr", yesterday, "Mark Dörr", { subject: "m@uni.de", org: "Universität Greifswald", role: "group leader" }),
      event("plans", new Date(today.getTime() + 30 * 86_400_000), "Zanzibar holiday", { place: "Stone Town" }),
    ]);
    assert.equal(report.rejected.length, 0, JSON.stringify(report.rejected));
    assert.equal(report.accepted, 9);

    const hrv = await client.ask("hrv", { generate: false });
    assert.equal(hrv.grade, "grounded", hrv.answer);
    assert.match(hrv.answer, /hrv: latest/);

    const trip = await client.ask("upcoming holiday", { generate: false });
    assert.ok(trip.claims.some((c) => c.receiver === "calendar" && /Zanzibar/.test(c.text)), trip.answer);

    const rewe = await client.ask("how much at rewe", { generate: false });
    assert.match(rewe.answer, /-42\.10 EUR/);

    const erased = await client.erase({ subject: "m@uni.de" });
    assert.equal(erased.removed, 2);
    const after = await client.ask("enzyme screen dörr", { generate: false });
    assert.equal(after.grade, "declined");

    const status = await client.status();
    assert.equal(status.encrypted, true);
    assert.equal(status.stats.committed, 9);
    assert.equal(status.stats.held, 7);
    assert.equal(status.consolidation.available, false);

    const onDisk = readFileSync(join(data, "records.log"), "utf8");
    assert.ok(!onDisk.includes("REWE") && !onDisk.includes("Zanzibar"), "records must be encrypted at rest");
  } finally {
    child.kill();
    await new Promise((r) => child.once("exit", r));
    rmSync(data, { recursive: true, force: true });
  }
});
