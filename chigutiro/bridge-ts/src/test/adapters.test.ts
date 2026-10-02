import { test } from "node:test";
import assert from "node:assert/strict";

import { academic, chatSession, fromBankTransactions, fromGarminSummary, fromGmailMessages, measurements, parseBankDate, upload } from "../index.js";

test("garmin summary: one measurement per present metric, ids per day", () => {
  const recs = fromGarminSummary({ sleep_hours: 7.2, hrv: 58, body_battery: null, steps: 10412 }, "2026-09-25T18:00:00Z");
  assert.deepEqual(recs.map((r) => r.metric), ["sleep_hours", "hrv", "steps"]);
  assert.equal(recs[1]?.id, "hrv:2026-09-25");
  assert.equal(recs[1]?.ts, "2026-09-25T00:00:00.000Z");
  assert.equal(recs[1]?.unit, "ms");
});

test("bank: German dates, skipped rows, identical bookings stay distinct", () => {
  const { records, skipped } = fromBankTransactions([
    { date: "24.09.2026", payee: "Bäckerei", purpose: "Kaffee", amount: -3.2, bank: "dkb" },
    { date: "24.09.2026", payee: "Bäckerei", purpose: "Kaffee", amount: -3.2, bank: "dkb" },
    { date: "31.02.2026", payee: "x", amount: -1, bank: "dkb" },
    { date: "2026-09-01", payee: "Arbeitgeber", amount: null, bank: "dkb" },
  ]);
  assert.equal(records.length, 2);
  assert.notEqual(records[0]?.id, records[1]?.id);
  assert.equal(records[0]?.ts, "2026-09-24T00:00:00.000Z");
  assert.equal(records[0]?.source, "bank:dkb");
  assert.equal(skipped.length, 2);
  assert.equal(parseBankDate("01.10.26"), "2026-10-01T00:00:00.000Z");
});

test("gmail: owner mail is voice material, other mail carries its sender as erasure subject", () => {
  const [mine, theirs] = fromGmailMessages(
    [
      { uid: "a", subject: "Draft", from: "Kundai <K@Example.org>", fromAddress: "K@Example.org", date: "2026-09-20T10:00:00Z", body: "Here is the draft." },
      { uid: "b", subject: "Re: Draft", from: "Mark <m@uni.de>", fromAddress: "m@uni.de", date: "2026-09-21T10:00:00Z", body: "Looks good." },
    ],
    ["k@example.org"],
  );
  assert.equal(mine?.authored_by_owner, true);
  assert.equal(mine?.subject, undefined);
  assert.equal(theirs?.authored_by_owner, false);
  assert.equal(theirs?.subject, "m@uni.de");
});

test("measurements: athletics session with units, non-finite skipped, stable ids", () => {
  const recs = measurements("track-log", "2026-09-24T17:00:00Z", { "100m time": [11.42, "s"], "top speed": [9.8, "m/s"], rpe: 8, broken: Number.NaN }, "session-42");
  assert.equal(recs.length, 3);
  assert.equal(recs[0]?.id, "session-42:100m time");
  assert.equal(recs[2]?.unit, undefined);
});

test("work adapters: channels put records in the work scope, authorship follows who wrote them", () => {
  const turns = chatSession("assistant", "s1", [
    { role: "user", text: "Why does run 14 stall at 40%?", ts: "2026-10-02T09:00:00Z" },
    { role: "assistant", text: "Likely PLP depletion; add 0.1 mM.", ts: "2026-10-02T09:00:05Z" },
    { role: "user", text: "  ", ts: "2026-10-02T09:01:00Z" },
  ]);
  assert.equal(turns.length, 2);
  assert.deepEqual(turns.map((t) => [t.source, t.authored_by_owner]), [["chat:assistant", true], ["chat:assistant", false]]);
  assert.equal(academic("semantic-scholar", "2026-10-02", "transaminase PLP kinetics").source, "academic:semantic-scholar");
  assert.equal(upload("paper", "2026-10-02", "Kinetics", "...").authored_by_owner, false);
  assert.equal(upload("lab-report", "2026-10-02", "Run 14", "...").source, "upload:lab-report");
  assert.equal(upload("lab-report", "2026-10-02", "Run 14", "...").authored_by_owner, true);
});
