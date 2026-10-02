// Mirrors the JSON of chigutiro-core / chigutiro-server. Field names are the
// wire names (snake_case), so values pass through without mapping.

interface RecordBase {
  /** Caller-side id, unique within `source`. Omit to identify by content. */
  id?: string;
  /** garmin, gmail, bank:dkb, brut, ... — distinct sources count as independent support. */
  source: string;
  /** RFC 3339 timestamp. */
  ts: string;
  /** Erasure key: the person or account this record is about. */
  subject?: string;
  tags?: string[];
}

export interface ProseRecord extends RecordBase {
  kind: "prose";
  text: string;
  title?: string;
  /** Only owner-authored prose may ever reach model weights. */
  authored_by_owner?: boolean;
}

export interface MeasurementRecord extends RecordBase {
  kind: "measurement";
  metric: string;
  value: number;
  unit?: string;
}

export interface TransactionRecord extends RecordBase {
  kind: "transaction";
  account: string;
  /** Outflows negative. */
  amount: number;
  currency?: string;
  counterparty?: string;
  memo?: string;
  category?: string;
}

export interface BalanceRecord extends RecordBase {
  kind: "balance";
  account: string;
  /** Debt as a negative amount. */
  amount: number;
  currency?: string;
  /** cash | investment | debt | ... */
  class?: string;
}

export interface PositionRecord extends RecordBase {
  kind: "position";
  lat: number;
  lon: number;
  accuracy_m?: number;
  label?: string;
}

export interface ContactRecord extends RecordBase {
  kind: "contact";
  person: string;
  org?: string;
  role?: string;
  relation?: string;
}

export interface EventRecord extends RecordBase {
  kind: "event";
  title: string;
  end?: string;
  place?: string;
  notes?: string;
}

export type ChigutiroRecord =
  | ProseRecord
  | MeasurementRecord
  | TransactionRecord
  | BalanceRecord
  | PositionRecord
  | ContactRecord
  | EventRecord;

export type Grade = "declined" | "contested" | "single_sourced" | "two_sourced" | "grounded";

export interface Claim {
  receiver: "text" | "series" | "ledger" | "places" | "people" | "calendar";
  text: string;
  grade: Grade;
  sources: string[];
  record_ids: string[];
  contested?: string;
  gain: number;
}

export interface RouteShape {
  at: string;
  query_terms: number;
  budget: number;
  price: number;
  grade: Grade;
  receivers: { receiver: string; offered: number; admitted: number }[];
}

export interface Answer {
  answer: string;
  /** False when no generator is configured or it failed: `answer` is then the claims themselves. */
  generated: boolean;
  model?: string;
  generation_error?: string;
  /** The weakest grade among admitted claims. */
  grade: Grade;
  claims: Claim[];
  route: RouteShape;
  model_version: number | null;
  model_tainted: boolean;
}

export interface IngestReport {
  accepted: number;
  duplicates: number;
  rejected: { index: number; reason: string }[];
  committed: number;
}

export interface EraseCriteria {
  ids?: string[];
  subject?: string;
  source?: string;
  /** RFC 3339; only records strictly before it. */
  before?: string;
}

export interface EraseReport {
  removed: number;
  voice_material_removed: boolean;
  model_tainted: boolean;
}

export interface Consolidation {
  version: number;
  started: string;
  finished: string | null;
  status: "running" | "succeeded" | "failed" | "superseded";
  reason: string;
  voice_docs: number;
  highest_seq: number;
  voice_erasure_epoch: number;
  base_model: string;
  model_dir: string;
  error?: string;
}

export interface Status {
  encrypted: boolean;
  stats: {
    committed: number;
    held: number;
    by_kind: Record<string, number>;
    passages: number;
    metrics: number;
    measurements: number;
    accounts: number;
    positions: number;
    people: number;
    events: number;
    voice_docs: number;
    /** Records in the work scope (see `work` adapters), their passages, and exportable docs. */
    work_records: number;
    work_passages: number;
    work_docs: number;
  };
  consolidation: {
    available: boolean;
    base_model: string;
    due: string | null;
    running: number | null;
    current: Consolidation | null;
    tainted: boolean;
  };
}

/**
 * `work` asks the work scope only (chat sessions, academic searches and
 * conversations, uploaded lab reports, papers and presentations): the view an
 * absicht federation gets. Omit it to ask over the whole profile.
 */
export type Scope = "work";

/** Documents a user can upload into the work scope. */
export type UploadKind = "lab-report" | "paper" | "presentation";
