// The board's answers, for logged-in readers (the middleware turns everyone
// else away with 401).
//
// The answers quote papers and this repository is public, so they are kept
// encrypted (board/answers.enc, written by scripts/board-publish.mjs) and
// decrypted here with BOARD_KEY, a Vercel environment variable that never
// enters the repository.

import { createDecipheriv } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const FILE = path.join(process.cwd(), "board", "answers.enc");

export default function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ error: "method not allowed" });
  }
  // Pasted values often carry spaces, a newline or quotes; the key is 43 base64url characters.
  const given = (process.env.BOARD_KEY || "").trim().replace(/^["']|["']$/g, "").replace(/^BOARD_KEY=/, "");
  const key = Buffer.from(given, "base64url");
  if (key.length !== 32) {
    return res.status(503).json({ error: process.env.BOARD_KEY ? "BOARD_KEY is set but is not the board key (43 characters from `node scripts/board-publish.mjs key`)." : "The board is not configured: set BOARD_KEY." });
  }
  let box;
  try {
    box = JSON.parse(fs.readFileSync(FILE, "utf8"));
  } catch {
    return res.status(200).json({ v: 1, entries: [] });
  }
  try {
    const data = Buffer.from(box.data, "base64url");
    const d = createDecipheriv("aes-256-gcm", key, Buffer.from(box.iv, "base64url"));
    d.setAuthTag(data.subarray(data.length - 16));
    const plain = Buffer.concat([d.update(data.subarray(0, data.length - 16)), d.final()]);
    res.setHeader("Cache-Control", "private, no-store");
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    return res.status(200).send(plain);
  } catch {
    return res.status(500).json({ error: "The answers could not be opened: BOARD_KEY does not match them." });
  }
}
