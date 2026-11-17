// The board: coalition's answers to the questions the group submitted, for
// reading, checking and correcting. Readers log in with the site's password;
// /api/board decrypts the answers (made on Kundai's machine and published with
// scripts/board-publish.mjs).
//
// Readers add to it through /api/notes: a question they want answered
// (with the answer they know), their own answer to any question, a mark
// (right, partly right, wrong, can't tell) and a note on any claim, and a
// comment on the whole answer. Notes are signed with the name a reader gives,
// since the login is shared. Each note is an event that is never changed; the
// page keeps each reader's latest per item.

import Head from "next/head";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

const STATUS = {
  grounded: {
    label: "Found in the papers",
    pill: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-200",
    edge: "border-l-2 border-solid border-emerald-600 dark:border-emerald-400",
  },
  partly: {
    label: "Found in part",
    pill: "bg-emerald-50 text-emerald-800 ring-1 ring-inset ring-emerald-300 dark:bg-emerald-950/40 dark:text-emerald-300 dark:ring-emerald-700",
    edge: "border-l-2 border-dotted border-emerald-600 dark:border-emerald-400",
  },
  contradicted: {
    label: "Contradicted by the papers",
    pill: "bg-rose-100 text-rose-900 dark:bg-rose-900/40 dark:text-rose-200",
    edge: "border-l-2 border-solid border-rose-600 dark:border-rose-400",
  },
  disputed: {
    label: "The papers differ",
    pill: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-200",
    edge: "border-l-2 border-solid border-amber-600 dark:border-amber-400",
  },
  unfound: {
    label: "No mention in the material",
    pill: "bg-neutral-200/70 text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400",
    edge: "border-l-2 border-solid border-neutral-300 dark:border-neutral-700",
  },
  paper: {
    label: "From the group's papers",
    pill: "bg-fuchsia-100 text-fuchsia-900 dark:bg-teal-900/40 dark:text-teal-200",
    edge: "border-l-2 border-solid border-primary dark:border-primaryDark",
  },
};
const ORDER = ["grounded", "partly", "contradicted", "disputed", "unfound", "paper"];

const VERB = {
  supports: "states this",
  partly: "states this part",
  contradicts: "states otherwise",
  answers: "says",
};

// A reader's mark on a claim.
const MARKS = [
  { id: "right", label: "Right", on: "bg-emerald-600 text-white dark:bg-emerald-500 dark:text-dark" },
  { id: "partly", label: "Partly right", on: "bg-emerald-200 text-emerald-950 dark:bg-emerald-800 dark:text-emerald-50" },
  { id: "wrong", label: "Wrong", on: "bg-rose-600 text-white dark:bg-rose-500 dark:text-dark" },
  { id: "unsure", label: "Can't tell", on: "bg-neutral-500 text-white dark:bg-neutral-400 dark:text-dark" },
];
const MARK = Object.fromEntries(MARKS.map((m) => [m.id, m]));

const NAME_STORE = "board-name";

// ---------------------------------------------------------------- data

async function getJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (res.status === 401) {
    window.location.assign("/login?next=/board");
    return null;
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `the board could not be loaded (${res.status})`);
  return body;
}

async function postNote(event) {
  const res = await fetch("/api/notes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(event),
  });
  if (res.status === 401) {
    window.location.assign("/login?next=/board");
    throw new Error("Your login has run out; log in again.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `the note was not saved (${res.status})`);
  return body;
}

async function logOut() {
  await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
  window.location.assign("/login");
}

/** A short, stable key for a claim, from its words (FNV-1a). */
function claimKey(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return `c${(h >>> 0).toString(16)}-${text.length}`;
}

const norm = (s) => (s || "").toLowerCase().replace(/\s+/g, " ").replace(/[\s?.!]+$/, "").trim();

/**
 * Folds the events into what the page shows. A submitted question whose words
 * match a published answer becomes that answer's question, and the answer its
 * submitter gave counts as their answer to it.
 */
function fold(events, entries) {
  const byText = new Map(entries.map((e) => [norm(e.question), e.id]));
  const alias = new Map();
  const submitted = [];
  const notes = new Map();
  const at = (q) => {
    if (!notes.has(q)) notes.set(q, { answers: {}, comments: {}, claims: {}, count: 0 });
    return notes.get(q);
  };
  const later = (old, ev) => !old || old.at <= ev.at;

  for (const ev of [...events].sort((a, b) => (a.at || "").localeCompare(b.at || ""))) {
    if (ev.kind === "question") {
      const run = byText.get(norm(ev.text));
      if (run) alias.set(ev.id, run);
      else submitted.push({ id: ev.id, question: ev.text, author: ev.author, asked: ev.at });
      if (ev.expected && ev.expected.trim()) {
        const n = at(run || ev.id);
        n.answers[ev.author] = { text: ev.expected, at: ev.at };
      }
      continue;
    }
    const q = alias.get(ev.q) || ev.q;
    const n = at(q);
    if (ev.kind === "answer" && later(n.answers[ev.author], ev)) n.answers[ev.author] = { text: ev.text, at: ev.at };
    if (ev.kind === "comment" && later(n.comments[ev.author], ev)) n.comments[ev.author] = { text: ev.text, at: ev.at };
    if (ev.kind === "claim") {
      const c = (n.claims[ev.claim] = n.claims[ev.claim] || {});
      if (later(c[ev.author], ev)) c[ev.author] = { verdict: ev.verdict, note: ev.note, at: ev.at };
    }
  }
  for (const n of notes.values()) {
    const filled = (o) => Object.values(o).filter((x) => (x.text || "").trim() || x.verdict || (x.note || "").trim()).length;
    n.count = filled(n.answers) + filled(n.comments) + Object.values(n.claims).reduce((k, c) => k + filled(c), 0);
  }
  return { submitted: submitted.reverse(), notes };
}

const Notes = createContext(null);
const useNotes = () => useContext(Notes);
const EMPTY = { answers: {}, comments: {}, claims: {}, count: 0 };

// ---------------------------------------------------------------- small parts

function date(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

function Pill({ status, className = "" }) {
  const s = STATUS[status] || STATUS.unfound;
  return (
    <span className={`inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium ${s.pill} ${className}`}>
      {s.label}
    </span>
  );
}

/** The counts as one thin bar, so the list shows at a glance how much was found. */
function Bar({ counts }) {
  const total = ORDER.reduce((n, k) => n + (counts[k] || 0), 0) || 1;
  const fill = {
    grounded: "bg-emerald-600 dark:bg-emerald-400",
    partly: "bg-emerald-300 dark:bg-emerald-700",
    contradicted: "bg-rose-500",
    disputed: "bg-amber-500",
    unfound: "bg-neutral-300 dark:bg-neutral-700",
    paper: "bg-primary dark:bg-primaryDark",
  };
  return (
    <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
      {ORDER.map((k) =>
        counts[k] ? <div key={k} className={fill[k]} style={{ width: `${(100 * counts[k]) / total}%` }} /> : null,
      )}
    </div>
  );
}

function Passage({ passage, quote }) {
  const i = passage.indexOf(quote);
  if (i < 0) return <p className="italic">“{quote}”</p>;
  return (
    <p className="text-neutral-600 dark:text-neutral-400">
      {passage.slice(0, i)}
      <mark className="rounded-sm bg-amber-100 px-0.5 text-dark dark:bg-amber-500/25 dark:text-light">{quote}</mark>
      {passage.slice(i + quote.length)}
    </p>
  );
}

function Proof({ p }) {
  const [whole, setWhole] = useState(false);
  return (
    <li className="rounded-md border border-neutral-200 bg-white/70 p-3 dark:border-neutral-800 dark:bg-white/[0.03]">
      <div className="mb-1.5 flex flex-wrap items-baseline gap-x-2 gap-y-1 text-xs">
        <span className={`font-semibold ${p.relation === "contradicts" ? "text-rose-700 dark:text-rose-300" : ""}`}>
          {p.origin === "group" ? "A group paper" : "A paper in the literature"} {VERB[p.relation] || p.relation}:
        </span>
      </div>
      <div className="text-[13px] leading-relaxed">
        {whole ? <Passage passage={p.passage} quote={p.quote} /> : <p>“{p.quote}”</p>}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-neutral-500 dark:text-neutral-400">
        {p.href ? (
          <a href={p.href} target="_blank" rel="noreferrer" className="underline decoration-neutral-300 underline-offset-2 hover:text-primary dark:hover:text-primaryDark">
            {p.title || p.href}
          </a>
        ) : (
          <span>{p.title}</span>
        )}
        {p.passage && p.passage !== p.quote ? (
          <button type="button" onClick={() => setWhole(!whole)} className="font-medium text-primary dark:text-primaryDark">
            {whole ? "Only the sentence" : "The sentence in its passage"}
          </button>
        ) : null}
      </div>
    </li>
  );
}

const link = "text-xs font-medium text-primary dark:text-primaryDark disabled:opacity-40";
const field =
  "w-full rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm leading-relaxed text-dark focus:border-primary focus:outline-none " +
  "dark:border-neutral-700 dark:bg-white/[0.04] dark:text-light dark:focus:border-primaryDark";
const button =
  "rounded-md bg-dark px-3 py-1.5 text-xs font-medium text-light hover:opacity-90 disabled:opacity-40 dark:bg-light dark:text-dark";

/** A reader's own text (answer or comment), with what the others wrote below it. */
function OwnText({ q, kind, title, hint, placeholder, mine }) {
  const { me, save, enabled } = useNotes();
  const all = mine[kind === "answer" ? "answers" : "comments"];
  const own = me ? all[me]?.text || "" : "";
  const [text, setText] = useState(own);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => setText(own), [own]);
  const others = Object.entries(all).filter(([a, x]) => a !== me && x.text.trim());

  async function submit(ev) {
    ev.preventDefault();
    setBusy(true);
    setError("");
    try {
      await save({ kind, q, text });
    } catch (e) {
      setError(e.message);
    }
    setBusy(false);
  }

  return (
    <section className="mt-8 rounded-lg border border-neutral-200 p-4 dark:border-neutral-800">
      <h3 className="text-sm font-semibold">{title}</h3>
      {hint ? <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">{hint}</p> : null}
      {others.length ? (
        <ul className="mt-3 space-y-2">
          {others.map(([a, x]) => (
            <li key={a} className="text-sm leading-relaxed">
              <span className="font-semibold">{a}:</span> <span className="whitespace-pre-wrap">{x.text}</span>
            </li>
          ))}
        </ul>
      ) : null}
      <form onSubmit={submit} className="mt-3">
        <textarea
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={placeholder}
          disabled={!enabled}
          className={field}
        />
        <div className="mt-2 flex items-center gap-3">
          <button type="submit" disabled={!enabled || busy || text === own} className={button}>
            {busy ? "Saving…" : own ? "Save the change" : "Save"}
          </button>
          {error ? <span className="text-xs text-rose-700 dark:text-rose-300">{error}</span> : null}
        </div>
      </form>
    </section>
  );
}

/** The marks and notes on one claim: the reader's own, editable, and the others'. */
function ClaimNotes({ q, ck, marks }) {
  const { me, save, enabled } = useNotes();
  const own = (me && marks[me]) || { verdict: "", note: "" };
  const [writing, setWriting] = useState(false);
  const [note, setNote] = useState(own.note || "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => setNote(own.note || ""), [own.note]);
  const others = Object.entries(marks).filter(([a, x]) => a !== me && (x.verdict || (x.note || "").trim()));

  async function store(verdict, text) {
    setBusy(true);
    setError("");
    try {
      await save({ kind: "claim", q, claim: ck, verdict, note: text });
      return true;
    } catch (e) {
      setError(e.message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-2">
      <div className="flex flex-wrap items-center gap-1.5">
        {MARKS.map((m) => {
          const on = own.verdict === m.id;
          return (
            <button
              key={m.id}
              type="button"
              disabled={!enabled || busy}
              aria-pressed={on}
              onClick={() => store(on ? "" : m.id, own.note || "")}
              className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors disabled:opacity-40 ${
                on ? m.on : "bg-neutral-200/60 text-neutral-700 hover:bg-neutral-300/70 dark:bg-white/[0.06] dark:text-neutral-300 dark:hover:bg-white/[0.1]"
              }`}
            >
              {m.label}
            </button>
          );
        })}
        <button type="button" disabled={!enabled} onClick={() => setWriting(!writing)} className={`${link} ml-1`}>
          {own.note ? "Your note" : "Add a note"}
        </button>
        {error ? <span className="text-xs text-rose-700 dark:text-rose-300">{error}</span> : null}
      </div>
      {own.note && !writing ? (
        <p className="mt-1.5 whitespace-pre-wrap text-[13px] leading-relaxed text-neutral-600 dark:text-neutral-400">
          <span className="font-semibold">You:</span> {own.note}
        </p>
      ) : null}
      {writing ? (
        <form
          className="mt-2"
          onSubmit={async (ev) => {
            ev.preventDefault();
            if (await store(own.verdict || "", note)) setWriting(false);
          }}
        >
          <textarea
            rows={3}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="What is right or wrong about this claim, or what it leaves out"
            className={field}
          />
          <div className="mt-1.5 flex items-center gap-3">
            <button type="submit" disabled={busy || note === (own.note || "")} className={button}>
              {busy ? "Saving…" : "Save the note"}
            </button>
            <button type="button" onClick={() => setWriting(false)} className={link}>
              Cancel
            </button>
          </div>
        </form>
      ) : null}
      {others.length ? (
        <ul className="mt-2 space-y-1">
          {others.map(([a, x]) => (
            <li key={a} className="text-[13px] leading-relaxed text-neutral-600 dark:text-neutral-400">
              <span className="font-semibold text-dark dark:text-light">{a}</span>
              {x.verdict ? (
                <span className={`ml-1.5 rounded-full px-1.5 py-px text-[10px] font-medium ${MARK[x.verdict]?.on || ""}`}>
                  {MARK[x.verdict]?.label || x.verdict}
                </span>
              ) : null}
              {(x.note || "").trim() ? <span className="ml-1.5 whitespace-pre-wrap">{x.note}</span> : null}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function Claim({ q, c, mine }) {
  const [shown, setShown] = useState(false);
  const s = STATUS[c.status] || STATUS.unfound;
  const n = c.proofs.length;
  const ck = claimKey(c.text);
  return (
    <li className={`pl-3 ${s.edge}`}>
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1">
        <p className={`min-w-0 flex-1 sm:w-full sm:flex-none text-[15px] leading-relaxed ${c.status === "unfound" ? "text-neutral-500 dark:text-neutral-400" : ""}`}>
          {c.text}
        </p>
        <Pill status={c.status} className="mt-0.5" />
      </div>
      {n > 0 ? (
        <>
          <button type="button" onClick={() => setShown(!shown)} aria-expanded={shown} className={`${link} mt-1`}>
            {shown ? "Hide" : "Show"} {n === 1 ? "the source" : `the ${n} sources`}
          </button>
          {shown ? <ul className="mt-2 space-y-2">{c.proofs.map((p, i) => <Proof key={i} p={p} />)}</ul> : null}
        </>
      ) : null}
      <ClaimNotes q={q} ck={ck} marks={mine.claims[ck] || {}} />
    </li>
  );
}

// ---------------------------------------------------------------- views

function Answer({ e }) {
  const { notes } = useNotes();
  const mine = notes.get(e.id) || EMPTY;
  const [onlyFound, setOnlyFound] = useState(false);
  const [how, setHow] = useState(false);
  const sections = e.sections
    .map((s) => ({ ...s, claims: onlyFound ? s.claims.filter((c) => c.proofs.length) : s.claims }))
    .filter((s) => s.claims.length);
  return (
    <article>
      <p className="text-xs uppercase tracking-wider text-neutral-500 dark:text-neutral-400">Asked {date(e.asked)}</p>
      <h2 className="mt-1 text-2xl font-bold leading-snug md:text-xl">{e.question}</h2>
      {e.headline ? (
        <p className="mt-4 rounded-lg bg-white p-4 text-[15px] leading-relaxed shadow-sm dark:bg-white/[0.04]">{e.headline}</p>
      ) : null}

      <OwnText
        q={e.id}
        kind="answer"
        mine={mine}
        title="The answer you know"
        hint="If you know the right answer to this question, write it here. It is what the answer above is judged against."
        placeholder="The correct answer, as you would give it"
      />

      <div className="mt-8 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {ORDER.filter((k) => e.counts[k]).map((k) => (
            <span key={k} className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${STATUS[k].pill}`}>
              {e.counts[k]} {STATUS[k].label.toLowerCase()}
            </span>
          ))}
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-xs text-neutral-600 dark:text-neutral-400">
          <input type="checkbox" checked={onlyFound} onChange={(ev) => setOnlyFound(ev.target.checked)} className="accent-primary" />
          Only claims with a source
        </label>
      </div>
      <p className="mt-3 text-xs text-neutral-500 dark:text-neutral-400">
        Mark each claim right, partly right, wrong, or that you can&apos;t tell, and add a note where it helps.
      </p>

      {sections.map((s) => (
        <section key={s.title} className="mt-7">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-neutral-700 dark:text-neutral-300">{s.title}</h3>
          <ul className="space-y-6">{s.claims.map((c, i) => <Claim key={i} q={e.id} c={c} mine={mine} />)}</ul>
        </section>
      ))}

      <OwnText
        q={e.id}
        kind="comment"
        mine={mine}
        title="Comments on this answer"
        hint="Anything about the answer as a whole: what it misses, what is misleading, what would make it useful."
        placeholder="Your comment"
      />

      {e.made.length ? (
        <div className="mt-10 border-t border-neutral-200 pt-4 dark:border-neutral-800">
          <button type="button" onClick={() => setHow(!how)} aria-expanded={how} className={link}>
            {how ? "Hide" : "How this answer was made"}
          </button>
          {how ? (
            <ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-relaxed text-neutral-600 dark:text-neutral-400">
              {e.made.map((m, i) => <li key={i}>{m}</li>)}
            </ul>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

function Waiting({ s }) {
  const { notes } = useNotes();
  const mine = notes.get(s.id) || EMPTY;
  return (
    <article>
      <p className="text-xs uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
        Submitted by {s.author} · {date(s.asked)}
      </p>
      <h2 className="mt-1 text-2xl font-bold leading-snug md:text-xl">{s.question}</h2>
      <p className="mt-4 rounded-lg bg-white p-4 text-sm leading-relaxed text-neutral-600 shadow-sm dark:bg-white/[0.04] dark:text-neutral-400">
        Not answered yet. The answer will appear here once it has been run.
      </p>
      <OwnText
        q={s.id}
        kind="answer"
        mine={mine}
        title="The answer you know"
        hint="The correct answer, if you know it. The answer the system gives will be judged against it."
        placeholder="The correct answer, as you would give it"
      />
    </article>
  );
}

function Submit({ onDone }) {
  const { save, enabled } = useNotes();
  const [text, setText] = useState("");
  const [expected, setExpected] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(ev) {
    ev.preventDefault();
    setBusy(true);
    setError("");
    try {
      const saved = await save({ kind: "question", text: text.trim(), expected });
      onDone(saved.id);
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit}>
      <h2 className="text-2xl font-bold md:text-xl">Submit a question</h2>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
        Ask something you already know the answer to, so the system&apos;s answer can be checked against yours. It is
        answered within a few days and then appears in the list.
      </p>
      <label htmlFor="sq" className="mt-6 block text-sm font-semibold">
        Your question
      </label>
      <textarea id="sq" rows={3} value={text} onChange={(e) => setText(e.target.value)} disabled={!enabled} className={`${field} mt-2`} />
      <label htmlFor="se" className="mt-5 block text-sm font-semibold">
        The answer you know
      </label>
      <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">Optional, and shown only to the board&apos;s readers.</p>
      <textarea id="se" rows={6} value={expected} onChange={(e) => setExpected(e.target.value)} disabled={!enabled} className={`${field} mt-2`} />
      <div className="mt-3 flex items-center gap-3">
        <button type="submit" disabled={!enabled || busy || text.trim().length < 5} className={button}>
          {busy ? "Submitting…" : "Submit the question"}
        </button>
        {error ? <span className="text-xs text-rose-700 dark:text-rose-300">{error}</span> : null}
      </div>
    </form>
  );
}

// ---------------------------------------------------------------- page

const NEW = "__new";

export default function Board() {
  const [state, setState] = useState({ phase: "loading" });
  const [events, setEvents] = useState([]);
  const [store, setStore] = useState({ configured: true, error: "" });
  const [pick, setPick] = useState(null);
  const [me, setMe] = useState("");
  const [nameAsked, setNameAsked] = useState(false);
  const nameRef = useRef(null);

  useEffect(() => {
    try {
      setMe(window.localStorage.getItem(NAME_STORE) || "");
    } catch {}
    getJson("/api/board")
      .then((board) => {
        if (!board) return;
        setState({ phase: "ok", board });
        setPick((p) => p || board.entries[0]?.id || null);
      })
      .catch((e) => setState({ phase: "failed", error: e.message }));
    getJson("/api/notes")
      .then((n) => {
        if (!n) return;
        setEvents(n.events || []);
        setStore({ configured: n.configured !== false, error: "" });
      })
      .catch((e) => setStore({ configured: true, error: e.message }));
  }, []);

  function rename(v) {
    setMe(v);
    setNameAsked(false);
    try {
      window.localStorage.setItem(NAME_STORE, v);
    } catch {}
  }

  const entries = useMemo(() => (state.board ? state.board.entries : []), [state.board]);
  const { submitted, notes } = useMemo(() => fold(events, entries), [events, entries]);

  const save = useCallback(
    async (event) => {
      const author = me.trim();
      if (!author) {
        setNameAsked(true);
        nameRef.current?.focus();
        nameRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
        throw new Error("Write your name at the top first.");
      }
      const saved = await postNote({ ...event, author });
      setEvents((es) => [...es, saved]);
      return saved;
    },
    [me],
  );

  const enabled = store.configured && !store.error;
  const ctx = useMemo(() => ({ me: me.trim(), save, notes, enabled }), [me, save, notes, enabled]);
  const current = entries.find((e) => e.id === pick);
  const waiting = submitted.find((s) => s.id === pick);

  function choose(id) {
    setPick(id);
    if (window.matchMedia("(max-width: 1023px)").matches) {
      setTimeout(() => document.getElementById("answer")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
    }
  }

  const item = (id, active) =>
    `w-full rounded-lg p-3 text-left transition-colors ${
      active ? "bg-white shadow-sm ring-1 ring-primary/40 dark:bg-white/[0.06] dark:ring-primaryDark/40" : "hover:bg-white/70 dark:hover:bg-white/[0.03]"
    }`;

  return (
    <Notes.Provider value={ctx}>
      <Head>
        <title>Answers board</title>
        <meta name="robots" content="noindex, nofollow" />
      </Head>
      <div className="mx-auto w-full max-w-6xl px-8 pb-24 pt-8 text-dark dark:text-light sm:px-4">
        <div className="mb-6 flex flex-wrap items-center justify-end gap-x-4 gap-y-2">
          <label className={`flex items-center gap-2 text-xs ${nameAsked ? "text-rose-700 dark:text-rose-300" : "text-neutral-500 dark:text-neutral-400"}`}>
            Your name, shown with your notes
            <input
              ref={nameRef}
              value={me}
              onChange={(e) => rename(e.target.value)}
              maxLength={60}
              placeholder="Name"
              className={`w-40 rounded-md border bg-white px-2 py-1 text-xs text-dark focus:outline-none dark:bg-white/[0.04] dark:text-light ${
                nameAsked ? "border-rose-500" : "border-neutral-300 focus:border-primary dark:border-neutral-700 dark:focus:border-primaryDark"
              }`}
            />
          </label>
          <button type="button" onClick={logOut} className="text-xs text-neutral-500 hover:text-primary dark:text-neutral-400 dark:hover:text-primaryDark">
            Log out
          </button>
        </div>
        <header className="max-w-3xl">
          <h1 className="text-3xl font-bold md:text-2xl">Answers to the submitted questions</h1>
          <p className="mt-3 text-[15px] leading-relaxed text-neutral-600 dark:text-neutral-400">
            Three models answered each question from their own knowledge. Every claim in the answer was then looked
            for in the group&apos;s papers and in the literature (Europe PMC and OpenAlex). Where a passage states it,
            the claim shows the sentence and the paper. &ldquo;No mention in the material&rdquo; means only that no
            passage was found; the claim may still be right or wrong. Please mark each claim, and write the answer you
            know.
          </p>
        </header>

        {!store.configured ? (
          <p className="mt-6 max-w-3xl rounded-lg bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-900/30 dark:text-amber-200">
            Notes cannot be saved yet: the board&apos;s notes store is not set up.
          </p>
        ) : null}
        {store.error ? (
          <p className="mt-6 max-w-3xl rounded-lg bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-900/30 dark:text-amber-200">
            The notes could not be loaded ({store.error}). Reload the page to try again.
          </p>
        ) : null}

        {state.phase === "loading" ? <p className="mt-12 text-sm text-neutral-500">Opening the answers…</p> : null}
        {state.phase === "failed" ? (
          <p className="mt-12 max-w-xl rounded-lg bg-white p-4 text-sm leading-relaxed shadow-sm dark:bg-white/[0.04]">{state.error}</p>
        ) : null}

        {state.phase === "ok" ? (
          <div className="mt-10 grid grid-cols-[18rem_1fr] gap-10 lg:grid-cols-1 lg:gap-8">
            <nav aria-label="Questions">
              <div className="sticky top-6 lg:static">
                <button type="button" onClick={() => choose(NEW)} className={`${item(NEW, pick === NEW)} mb-4 border border-dashed border-neutral-300 dark:border-neutral-700`}>
                  <span className="text-sm font-medium">+ Submit a question</span>
                </button>
                <p className="mb-2 text-xs uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
                  {entries.length} answered{submitted.length ? ` · ${submitted.length} waiting` : ""}
                </p>
                <ul className="space-y-2">
                  {submitted.map((s) => (
                    <li key={s.id}>
                      <button type="button" onClick={() => choose(s.id)} aria-current={s.id === pick} className={item(s.id, s.id === pick)}>
                        <span className="line-clamp-3 text-sm font-medium leading-snug">{s.question}</span>
                        <span className="mt-1 block text-[11px] text-neutral-500 dark:text-neutral-400">
                          Waiting for an answer · {s.author}
                        </span>
                      </button>
                    </li>
                  ))}
                  {entries.map((e) => {
                    const n = notes.get(e.id)?.count || 0;
                    const claims = e.counts.grounded + e.counts.partly + e.counts.contradicted + e.counts.disputed + e.counts.unfound;
                    return (
                      <li key={e.id}>
                        <button type="button" onClick={() => choose(e.id)} aria-current={e.id === pick} className={item(e.id, e.id === pick)}>
                          <span className="line-clamp-3 text-sm font-medium leading-snug">{e.question}</span>
                          <span className="mt-1 block text-[11px] text-neutral-500 dark:text-neutral-400">
                            {date(e.asked)} · {e.counts.grounded + e.counts.partly} of {claims} claims found
                            {n ? ` · ${n} ${n === 1 ? "note" : "notes"}` : ""}
                          </span>
                          <span className="mt-2 block">
                            <Bar counts={e.counts} />
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
                {store.configured && !store.error ? (
                  // A file download from the API, not a page to navigate to.
                  // eslint-disable-next-line @next/next/no-html-link-for-pages
                  <a href="/api/notes?download=1" download className={`${link} mt-6 inline-block`}>
                    Download all notes (JSON)
                  </a>
                ) : null}
              </div>
            </nav>
            <div id="answer" className="min-w-0 scroll-mt-6">
              {pick === NEW ? <Submit onDone={(id) => setPick(id)} /> : null}
              {current ? <Answer key={current.id} e={current} /> : null}
              {waiting ? <Waiting key={waiting.id} s={waiting} /> : null}
              {!current && !waiting && pick !== NEW ? (
                <p className="text-sm text-neutral-500">No answers have been published yet.</p>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>
    </Notes.Provider>
  );
}
