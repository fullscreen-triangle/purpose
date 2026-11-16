// The board: coalition's answers to the questions the group submitted, for
// reading and checking. Nobody types a question here; the answers are made on
// Kundai's machine and published with scripts/board-publish.mjs. Readers log
// in with the site's password; /api/board decrypts the answers for them.

import Head from "next/head";
import { useEffect, useMemo, useState } from "react";

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

async function load() {
  const res = await fetch("/api/board", { cache: "no-store" });
  if (res.status === 401) {
    window.location.assign("/login?next=/board");
    return null;
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `the answers could not be loaded (${res.status})`);
  return body;
}

async function logOut() {
  await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
  window.location.assign("/login");
}

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

function Claim({ c }) {
  const [shown, setShown] = useState(false);
  const s = STATUS[c.status] || STATUS.unfound;
  const n = c.proofs.length;
  return (
    <li className={`pl-3 ${s.edge}`}>
      <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1">
        <p className={`min-w-0 flex-1 text-[15px] leading-relaxed ${c.status === "unfound" ? "text-neutral-500 dark:text-neutral-400" : ""}`}>
          {c.text}
        </p>
        <Pill status={c.status} className="mt-0.5" />
      </div>
      {n > 0 ? (
        <>
          <button
            type="button"
            onClick={() => setShown(!shown)}
            aria-expanded={shown}
            className="mt-1 text-xs font-medium text-primary dark:text-primaryDark"
          >
            {shown ? "Hide" : "Show"} {n === 1 ? "the source" : `the ${n} sources`}
          </button>
          {shown ? <ul className="mt-2 space-y-2">{c.proofs.map((p, i) => <Proof key={i} p={p} />)}</ul> : null}
        </>
      ) : null}
    </li>
  );
}

function Answer({ e }) {
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

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
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

      {sections.map((s) => (
        <section key={s.title} className="mt-7">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-neutral-700 dark:text-neutral-300">{s.title}</h3>
          <ul className="space-y-4">{s.claims.map((c, i) => <Claim key={i} c={c} />)}</ul>
        </section>
      ))}

      {e.made.length ? (
        <div className="mt-10 border-t border-neutral-200 pt-4 dark:border-neutral-800">
          <button type="button" onClick={() => setHow(!how)} aria-expanded={how} className="text-xs font-medium text-primary dark:text-primaryDark">
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

export default function Board() {
  const [state, setState] = useState({ phase: "loading" });
  const [pick, setPick] = useState(null);

  useEffect(() => {
    load()
      .then((board) => {
        if (!board) return;
        setState({ phase: "ok", board });
        setPick(board.entries[0]?.id || null);
      })
      .catch((e) => setState({ phase: "failed", error: e.message }));
  }, []);

  const entries = useMemo(() => (state.board ? state.board.entries : []), [state.board]);
  const current = entries.find((e) => e.id === pick);

  function choose(id) {
    setPick(id);
    if (window.matchMedia("(max-width: 1023px)").matches) {
      document.getElementById("answer")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  return (
    <>
      <Head>
        <title>Answers board</title>
        <meta name="robots" content="noindex, nofollow" />
      </Head>
      <div className="mx-auto w-full max-w-6xl px-8 pb-24 pt-8 text-dark dark:text-light sm:px-4 sm:pt-8">
        <div className="mb-6 flex justify-end">
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
            passage was found; the claim may still be right or wrong. Open a claim&apos;s sources to judge them.
          </p>
        </header>

        {state.phase === "loading" ? <p className="mt-12 text-sm text-neutral-500">Opening the answers…</p> : null}
        {state.phase === "failed" ? (
          <p className="mt-12 max-w-xl rounded-lg bg-white p-4 text-sm leading-relaxed shadow-sm dark:bg-white/[0.04]">
            {state.error}
          </p>
        ) : null}

        {state.phase === "ok" ? (
          entries.length ? (
            <div className="mt-10 grid grid-cols-[18rem_1fr] gap-10 lg:grid-cols-1 lg:gap-8">
              <nav aria-label="Questions">
                <p className="mb-2 text-xs uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
                  {entries.length} {entries.length === 1 ? "question" : "questions"}
                </p>
                <ul className="space-y-2 sticky top-6 lg:static">
                  {entries.map((e) => (
                    <li key={e.id}>
                      <button
                        type="button"
                        onClick={() => choose(e.id)}
                        aria-current={e.id === pick}
                        className={`w-full rounded-lg p-3 text-left transition-colors ${
                          e.id === pick
                            ? "bg-white shadow-sm ring-1 ring-primary/40 dark:bg-white/[0.06] dark:ring-primaryDark/40"
                            : "hover:bg-white/70 dark:hover:bg-white/[0.03]"
                        }`}
                      >
                        <span className="line-clamp-3 text-sm font-medium leading-snug">{e.question}</span>
                        <span className="mt-1 block text-[11px] text-neutral-500 dark:text-neutral-400">
                          {date(e.asked)} · {e.counts.grounded + e.counts.partly} of{" "}
                          {e.counts.grounded + e.counts.partly + e.counts.contradicted + e.counts.disputed + e.counts.unfound} claims
                          found
                        </span>
                        <span className="mt-2 block">
                          <Bar counts={e.counts} />
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </nav>
              <div id="answer" className="min-w-0 scroll-mt-6">
                {current ? <Answer key={current.id} e={current} /> : null}
              </div>
            </div>
          ) : (
            <p className="mt-12 text-sm text-neutral-500">No answers have been published yet.</p>
          )
        ) : null}
      </div>
    </>
  );
}
