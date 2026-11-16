#!/usr/bin/env node
// Publishes coalition's answered questions to the site's board (/board).
//
//   node scripts/board-publish.mjs add <run-dir>...   add runs and rebuild
//   node scripts/board-publish.mjs remove <run-dir>... drop runs and rebuild
//   node scripts/board-publish.mjs                    rebuild from the list
//   node scripts/board-publish.mjs list               the runs on the board
//   node scripts/board-publish.mjs link               the link to send
//
// A run is a directory coalition filed for one question (answer.json and
// question.txt; the chat files them in %LOCALAPPDATA%\coalition-chat\runs).
//
// The answers quote the papers, and this repository is public, so the board
// is written encrypted (AES-256-GCM) to public/board/answers.enc. The key
// never enters the repository or the server: it lives beside the run list in
// %LOCALAPPDATA%\coalition-chat\board\ and travels only in the link's
// fragment (#k=...), which browsers do not send to the server. Whoever has the
// link can read the board; to withdraw it, delete board.key and rebuild, which
// makes a new key and so a new link.
//
// Commit public/board/answers.enc and push; Vercel deploys the site from main.

import { createCipheriv, randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SITE = "https://mechanistic-synthesis.vercel.app";
const HOME = path.join(
  process.env.LOCALAPPDATA || path.join(process.env.HOME || ".", ".local", "share"),
  "coalition-chat",
  "board",
);
const KEY_FILE = path.join(HOME, "board.key");
const RUNS_FILE = path.join(HOME, "runs.txt");
const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "public", "board", "answers.enc");

const b64url = (buf) => Buffer.from(buf).toString("base64url");

function key() {
  fs.mkdirSync(HOME, { recursive: true });
  if (!fs.existsSync(KEY_FILE)) fs.writeFileSync(KEY_FILE, b64url(randomBytes(32)) + "\n", { mode: 0o600 });
  const k = Buffer.from(fs.readFileSync(KEY_FILE, "utf8").trim(), "base64url");
  if (k.length !== 32) throw new Error(`${KEY_FILE} does not hold a 32-byte key`);
  return k;
}

function runs() {
  if (!fs.existsSync(RUNS_FILE)) return [];
  return fs.readFileSync(RUNS_FILE, "utf8").split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
}

function saveRuns(list) {
  fs.mkdirSync(HOME, { recursive: true });
  fs.writeFileSync(RUNS_FILE, [...new Set(list)].join("\n") + "\n");
}

/** Where a paper can be opened: its DOI, else its Europe PMC page. */
function href(p) {
  if (p.doi) return `https://doi.org/${p.doi}`;
  const id = p.paper || "";
  if (id.startsWith("doi:")) return `https://doi.org/${id.slice(4)}`;
  const m = id.match(/^(MED|PMC)-?(.+)$/);
  if (m) return `https://europepmc.org/article/${m[1]}/${m[2]}`;
  return null;
}

/** One run as the board shows it: what was asked, the answer, and what supports it. */
function entry(dir) {
  const a = JSON.parse(fs.readFileSync(path.join(dir, "answer.json"), "utf8"));
  const v = a.view || {};
  const id = path.basename(dir);
  const secs = Number(id.split("-")[0]);
  const sections = [];
  for (const c of v.claims || []) {
    let s = sections.find((x) => x.title === c.section);
    if (!s) sections.push((s = { title: c.section, claims: [] }));
    s.claims.push({
      text: c.text,
      status: c.status,
      proofs: (c.proofs || []).map((p) => ({
        relation: p.relation,
        quote: p.quote,
        passage: p.passage,
        title: p.title,
        href: href(p),
        origin: p.origin,
      })),
    });
  }
  const count = (st) => (v.claims || []).filter((c) => c.status === st).length;
  return {
    id,
    asked: Number.isFinite(secs) ? new Date(secs * 1000).toISOString() : null,
    question: (v.how && v.how.question) || fs.readFileSync(path.join(dir, "question.txt"), "utf8").trim(),
    outcome: a.outcome,
    headline: v.headline || "",
    sections,
    counts: {
      grounded: count("grounded"),
      partly: count("partly"),
      contradicted: count("contradicted"),
      disputed: count("disputed"),
      unfound: count("unfound"),
      paper: count("paper"),
    },
    made: v.made || [],
  };
}

function build() {
  const list = runs();
  const entries = [];
  for (const dir of list) {
    try {
      entries.push(entry(dir));
    } catch (e) {
      console.error(`skipped ${dir}: ${e.message}`);
    }
  }
  entries.sort((x, y) => (y.asked || "").localeCompare(x.asked || ""));
  const plain = Buffer.from(JSON.stringify({ v: 1, built: new Date().toISOString(), entries }), "utf8");
  const iv = randomBytes(12);
  const c = createCipheriv("aes-256-gcm", key(), iv);
  const data = Buffer.concat([c.update(plain), c.final(), c.getAuthTag()]);
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify({ v: 1, alg: "AES-256-GCM", iv: b64url(iv), data: b64url(data) }));
  console.log(`${entries.length} questions on the board -> ${path.relative(process.cwd(), OUT)}`);
}

const [cmd, ...args] = process.argv.slice(2);
const abs = (d) => path.resolve(d);
switch (cmd) {
  case "add":
    for (const d of args) if (!fs.existsSync(path.join(d, "answer.json"))) throw new Error(`${d} has no answer.json`);
    saveRuns([...runs(), ...args.map(abs)]);
    build();
    break;
  case "remove":
    saveRuns(runs().filter((r) => !args.map(abs).includes(r)));
    build();
    break;
  case "list":
    for (const r of runs()) console.log(r);
    break;
  case "link":
    console.log(`${SITE}/board#k=${b64url(key())}`);
    break;
  case undefined:
    build();
    break;
  default:
    console.error("usage: board-publish.mjs [add|remove <run-dir>... | list | link]");
    process.exit(2);
}
