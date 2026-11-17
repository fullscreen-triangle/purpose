// What the board's readers add: questions they submit (with the answer they
// know), their own answer to a question, a mark and note on any claim, and a
// comment on an answer. Logged-in readers only (the middleware sees to it).
//
// Every note is an event appended to one Redis list and never changed; the
// page reads them all and keeps each reader's latest per item. So nothing a
// reader wrote is ever lost, and two readers saving at once cannot collide.
//
//   GET  /api/notes              {configured, events}
//   GET  /api/notes?download=1   the same, as a file
//   POST /api/notes              one event; answers it with id and time

import { randomBytes } from "node:crypto";
import { z } from "zod";
import { NotConfigured, configured, redis } from "@/lib/store";

const LIST = "board:events";
const MAX_EVENTS = 50000;

const name = z.string().trim().min(1).max(60);
const q = z.string().min(1).max(80);
const Event = z.discriminatedUnion("kind", [
  z.object({ kind: z.literal("question"), author: name, text: z.string().trim().min(5).max(2000), expected: z.string().max(8000).default("") }),
  z.object({ kind: z.literal("answer"), author: name, q, text: z.string().max(8000) }),
  z.object({ kind: z.literal("comment"), author: name, q, text: z.string().max(8000) }),
  z.object({
    kind: z.literal("claim"),
    author: name,
    q,
    claim: z.string().min(1).max(40),
    verdict: z.enum(["right", "partly", "wrong", "unsure", ""]),
    note: z.string().max(4000).default(""),
  }),
]);

export default async function handler(req, res) {
  res.setHeader("Cache-Control", "private, no-store");
  try {
    if (req.method === "GET") {
      if (!configured()) return res.status(200).json({ configured: false, events: [] });
      const raw = (await redis(["LRANGE", LIST, "0", "-1"])) || [];
      const events = raw.flatMap((s) => {
        try {
          return [JSON.parse(s)];
        } catch {
          return [];
        }
      });
      if (req.query.download) {
        res.setHeader("Content-Disposition", `attachment; filename="board-notes-${new Date().toISOString().slice(0, 10)}.json"`);
      }
      return res.status(200).json({ configured: true, events });
    }
    if (req.method === "POST") {
      const parsed = Event.safeParse(req.body);
      if (!parsed.success) return res.status(400).json({ error: "That note could not be read." });
      if ((await redis(["LLEN", LIST])) >= MAX_EVENTS) return res.status(507).json({ error: "The notes store is full." });
      const event = { ...parsed.data, id: randomBytes(6).toString("hex"), at: new Date().toISOString() };
      await redis(["RPUSH", LIST, JSON.stringify(event)]);
      return res.status(200).json(event);
    }
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "method not allowed" });
  } catch (e) {
    if (e instanceof NotConfigured) return res.status(503).json({ error: "Notes cannot be saved yet: the notes store is not set up." });
    return res.status(502).json({ error: "The notes store did not answer; try again." });
  }
}
