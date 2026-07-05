// Shared types for @buhera/purpose.
//
// These reconcile the two consumer contracts:
//  - buhera-specifications.md uses `Step` / `Goal` (the Session layer).
//  - graffiti.md uses `ContextGraphView` / `TaggedItem` (the core layer).
// Both are exported; they describe the same graph from different heights.

/** Opaque caller-chosen identifier. Purpose treats it as a string only. */
export type StepId = string;

/** Alias kept for graffiti.md compatibility (its `ItemId`). */
export type ItemId = StepId;

/** A term is a string; purpose does not care what it means. */
export type Term = string;

/** The distinctions a step draws — the paper's τ(u). Caller-supplied. */
export type TermSet = ReadonlySet<Term>;

/** The distinguished vertex standing for "everything not yet individuated". */
export type MediumId = string;

/** The default medium identifier if a caller does not choose one. */
export const DEFAULT_MEDIUM: MediumId = "__medium__";

/** One committed step in the history (buhera-specifications.md §4.1). */
export interface Step {
  /** Caller-chosen unique id within the session. */
  id: StepId;
  /** The distinctions this step draws — the paper's τ(u). */
  terms: TermSet;
  /** Token cost of this step's content, computed by the caller. */
  cost: number;
  /** Monotonic timestamp (ms since epoch or a logical clock). */
  timestamp: number;
  /** Opaque caller data. Purpose never inspects this. */
  payload?: unknown;
}

/** A step with just id + terms — graffiti.md's `TaggedItem`. */
export interface TaggedItem {
  id: ItemId;
  terms: TermSet;
}

/** A goal is a set of terms the next act of reasoning must resolve. */
export interface Goal {
  terms: TermSet;
}

/** An undirected weighted edge of the context graph. */
export interface WeightedEdge {
  a: ItemId;
  b: ItemId;
  weight: number; // > 0
}

/**
 * The plain graph shape the pure core reasons over (graffiti.md §1).
 * A snapshot with no behaviour: the caller (or the Session) builds it,
 * the core reads it and never mutates or retains it.
 */
export interface ContextGraphView {
  medium: MediumId;
  items: TaggedItem[]; // excludes the medium
  edges: WeightedEdge[]; // undirected; endpoints in items or the medium
  floor: number; // the derived positive floor β
}
