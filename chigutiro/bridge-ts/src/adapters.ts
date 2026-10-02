// Adapters from the host framework's existing data shapes to chigutiro
// records. Each is pure: it maps, it never sends. Shapes are taken from
// kwisatz-haderach — backend/routes/health.py (/health/summary),
// backend/routes/bank.py (/bank/transactions), web/lib/gmail.js.

import type {
  BalanceRecord,
  ContactRecord,
  EventRecord,
  MeasurementRecord,
  PositionRecord,
  ProseRecord,
  TransactionRecord,
  UploadKind,
} from "./types.js";

type When = Date | string;

function iso(when: When): string {
  const d = typeof when === "string" ? new Date(when) : when;
  if (Number.isNaN(d.getTime())) throw new Error(`chigutiro: not a date: ${String(when)}`);
  return d.toISOString();
}

function dayOf(when: When): string {
  return iso(when).slice(0, 10);
}

// ---------------------------------------------------------------- work scope
//
// The only records chigutiro treats as work: their `source` channel puts them
// there (chat:*, academic:*, upload:<kind>). Work is what an absicht
// federation may consult and what a work model is trained on off this machine.

export interface ChatTurn {
  role: "user" | "assistant";
  text: string;
  ts: When;
}

/** One chat session as one record per turn; the user's own turns count as their writing. */
export function chatSession(app: string, sessionId: string, turns: ChatTurn[]): ProseRecord[] {
  return turns
    .filter((t) => t.text.trim())
    .map((t, i) => ({
      kind: "prose",
      source: `chat:${app}`,
      id: `${sessionId}:${i}`,
      ts: iso(t.ts),
      text: t.text,
      title: `${t.role} turn ${i + 1}`,
      authored_by_owner: t.role === "user",
      tags: [`session:${sessionId}`],
    }));
}

/** An academic search (query plus the results read) or a conversation about it. */
export function academic(engine: string, ts: When, text: string, opts: { title?: string; id?: string } = {}): ProseRecord {
  return {
    kind: "prose",
    source: `academic:${engine}`,
    ts: iso(ts),
    text,
    ...(opts.title ? { title: opts.title } : {}),
    ...(opts.id ? { id: opts.id } : {}),
  };
}

/**
 * An uploaded lab report, paper or presentation, as extracted text. `ownWork`
 * marks a document the user wrote (their lab report, their slides); a paper
 * they read is not.
 */
export function upload(kind: UploadKind, ts: When, title: string, text: string, opts: { id?: string; ownWork?: boolean } = {}): ProseRecord {
  return {
    kind: "prose",
    source: `upload:${kind}`,
    ts: iso(ts),
    title,
    text,
    authored_by_owner: opts.ownWork ?? kind !== "paper",
    ...(opts.id ? { id: opts.id } : {}),
  };
}

// ---------------------------------------------------------------- generic

export function measurement(source: string, ts: When, metric: string, value: number, unit?: string, id?: string): MeasurementRecord {
  return { kind: "measurement", source, ts: iso(ts), metric, value, ...(unit ? { unit } : {}), ...(id ? { id } : {}) };
}

/**
 * Several metrics observed together — an athletics session, a sensor sample.
 * `values` maps metric → number or [number, unit]; non-finite values are skipped.
 * `sessionId` makes re-sending the same session a no-op.
 */
export function measurements(
  source: string,
  ts: When,
  values: Record<string, number | [number, string] | null | undefined>,
  sessionId?: string,
): MeasurementRecord[] {
  const out: MeasurementRecord[] = [];
  for (const [metric, v] of Object.entries(values)) {
    const [value, unit] = Array.isArray(v) ? v : [v, undefined];
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    out.push(measurement(source, ts, metric, value, unit, sessionId ? `${sessionId}:${metric}` : undefined));
  }
  return out;
}

export function event(source: string, start: When, title: string, opts: { end?: When; place?: string; notes?: string; id?: string } = {}): EventRecord {
  return {
    kind: "event",
    source,
    ts: iso(start),
    title,
    ...(opts.end ? { end: iso(opts.end) } : {}),
    ...(opts.place ? { place: opts.place } : {}),
    ...(opts.notes ? { notes: opts.notes } : {}),
    ...(opts.id ? { id: opts.id } : {}),
  };
}

/**
 * One person in the owner's graph. `subject` (an email, say) is the erasure
 * key, and links the person to every other record carrying it.
 */
export function contact(
  source: string,
  ts: When,
  person: string,
  opts: { subject?: string; org?: string; role?: string; relation?: string } = {},
): ContactRecord {
  return {
    kind: "contact",
    source,
    ts: iso(ts),
    person,
    subject: (opts.subject ?? person).toLowerCase(),
    ...(opts.org ? { org: opts.org } : {}),
    ...(opts.role ? { role: opts.role } : {}),
    ...(opts.relation ? { relation: opts.relation } : {}),
  };
}

export function position(source: string, ts: When, lat: number, lon: number, opts: { accuracy_m?: number; label?: string } = {}): PositionRecord {
  return { kind: "position", source, ts: iso(ts), lat, lon, ...opts };
}

/** A statement of what an account holds. Debt as a negative amount. */
export function balance(source: string, ts: When, account: string, amount: number, opts: { currency?: string; class?: string } = {}): BalanceRecord {
  return { kind: "balance", source, ts: iso(ts), account, amount, subject: account, ...opts };
}

// ---------------------------------------------------------------- Garmin

/** Shape of kwisatz-haderach GET /health/summary. */
export interface GarminSummary {
  sleep_hours?: number | null;
  hrv?: number | null;
  body_battery?: number | null;
  steps?: number | null;
  stress?: number | null;
  spo2?: number | null;
}

const GARMIN_UNITS: Record<keyof GarminSummary, string> = {
  sleep_hours: "h",
  hrv: "ms",
  body_battery: "score",
  steps: "steps",
  stress: "score",
  spo2: "%",
};

/**
 * One day's summary. Ids are per metric per day, so send a day once it is
 * complete (e.g. yesterday's, each morning): a later send of the same day is
 * treated as a duplicate, not an update.
 */
export function fromGarminSummary(summary: GarminSummary, day: When, source = "garmin"): MeasurementRecord[] {
  const d = dayOf(day);
  const out: MeasurementRecord[] = [];
  for (const key of Object.keys(GARMIN_UNITS) as (keyof GarminSummary)[]) {
    const v = summary[key];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    out.push(measurement(source, `${d}T00:00:00Z`, key, v, GARMIN_UNITS[key], `${key}:${d}`));
  }
  return out;
}

// ---------------------------------------------------------------- bank

/** Shape of one kwisatz-haderach /bank/transactions entry. */
export interface BankTransaction {
  date: string;
  payee?: string;
  purpose?: string;
  amount: number | null;
  bank: string;
  category?: string;
}

/** Accepts dd.mm.yyyy, dd.mm.yy (German exports) and ISO dates. */
export function parseBankDate(raw: string): string | null {
  const s = raw.trim();
  const de = /^(\d{1,2})\.(\d{1,2})\.(\d{2}|\d{4})$/.exec(s);
  if (de) {
    const [, dd, mm, yy] = de as unknown as [string, string, string, string];
    const year = yy.length === 2 ? 2000 + Number(yy) : Number(yy);
    const d = new Date(Date.UTC(year, Number(mm) - 1, Number(dd)));
    return d.getUTCDate() === Number(dd) ? d.toISOString() : null;
  }
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d.toISOString();
}

/**
 * Transactions from a bank export. Identical bookings on one day (two
 * coffees) stay distinct through an occurrence counter, and ids are stable
 * across re-imports of an export in the same order, so overlapping exports
 * deduplicate. Entries with no amount or an unreadable date are returned in
 * `skipped` rather than guessed at.
 */
export function fromBankTransactions(txs: BankTransaction[]): { records: TransactionRecord[]; skipped: BankTransaction[] } {
  const records: TransactionRecord[] = [];
  const skipped: BankTransaction[] = [];
  const seen = new Map<string, number>();
  for (const t of txs) {
    const ts = parseBankDate(t.date ?? "");
    if (ts === null || typeof t.amount !== "number" || !Number.isFinite(t.amount)) {
      skipped.push(t);
      continue;
    }
    const key = [ts.slice(0, 10), t.amount.toFixed(2), t.payee ?? "", t.purpose ?? ""].join("|");
    const n = (seen.get(key) ?? 0) + 1;
    seen.set(key, n);
    records.push({
      kind: "transaction",
      id: `${key}|#${n}`,
      source: `bank:${t.bank}`,
      ts,
      account: t.bank,
      amount: t.amount,
      currency: "EUR",
      ...(t.payee ? { counterparty: t.payee } : {}),
      ...(t.purpose ? { memo: t.purpose } : {}),
      ...(t.category ? { category: t.category } : {}),
    });
  }
  return { records, skipped };
}

// ---------------------------------------------------------------- Gmail

/** Shape returned by kwisatz-haderach web/lib/gmail.js. */
export interface GmailMessage {
  uid: string;
  subject: string;
  from: string;
  fromAddress: string;
  date: string;
  body: string;
}

/**
 * Mail you sent becomes voice material; mail from anyone else is recall-only
 * and carries its sender as the erasure subject, so `erase({ subject })`
 * removes a correspondent completely. Note gmail.js fetches the inbox — to
 * teach your voice, also fetch `in:sent`.
 */
export function fromGmailMessages(msgs: GmailMessage[], ownerAddresses: string[]): ProseRecord[] {
  const owners = new Set(ownerAddresses.map((a) => a.trim().toLowerCase()));
  return msgs
    .filter((m) => m.body?.trim())
    .map((m) => {
      const from = (m.fromAddress || m.from).trim().toLowerCase();
      const mine = owners.has(from);
      return {
        kind: "prose" as const,
        id: m.uid,
        source: "gmail",
        ts: iso(m.date),
        title: m.subject,
        text: m.body,
        authored_by_owner: mine,
        ...(mine ? {} : { subject: from }),
      };
    });
}
