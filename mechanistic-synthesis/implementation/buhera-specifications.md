# Purpose Library: Contract Specification

**Status:** design contract from the long-grass side
**Author:** long-grass integration team
**Date:** 2026-07-04
**Depends on:** `docs/sources/purpose-propagation.tex`

This document specifies what long-grass expects from the `purpose` library
so that long-grass can wire it in and grow with it. Every decision below
is fixed for v1. Anything not specified here is purpose's choice.

The library implements the tandem carry of `purpose-propagation.tex`:
given a history of steps and a goal, return the minimum-sufficient carry
under a budget. Long-grass supplies the steps and the term map; purpose
does the graph, seek, necessity, knapsack, floor, and (optionally) the
cascade.

---

## 1. Package identity

- **Name:** `@buhera/purpose`
- **Language:** TypeScript, published with `.d.ts` types
- **Distribution:** published as an npm package. long-grass imports it as
  a normal dependency (`bun add @buhera/purpose`). No vendoring, no
  submodules, no monorepo coupling.
- **Runtime targets:** Node ≥18 and modern browsers (Vercel edge + client
  bundle). No Node-only APIs in the core library. If purpose ships a
  Node-only helper (persistence, worker adapter), it must live in a
  separate subpath export (`@buhera/purpose/node`) so the core bundle
  stays universal.
- **Dependencies:** ideally zero. If any are strictly necessary, they
  must be small, tree-shakeable, and browser-safe.
- **License:** whatever the rest of the Buhera stack uses.

---

## 2. Versioning contract

- Semver. Long-grass will pin to `^0.x` during pre-1.0.
- **Before 1.0:** minor version bumps may break shapes; patch versions
  are non-breaking. A CHANGELOG note flagging each breaking minor bump
  is expected.
- **At 1.0.0:** the surface described in §4 is locked. Additions are
  minor bumps. Any change to the shapes in §4 is a major bump.
- The paper's cascade (§7) may land as a minor bump within 1.x — see §6.

---

## 3. Design principles (non-negotiable)

These are the constraints that make long-grass able to grow with purpose
without ever forking or patching it.

1. **Purpose is a graph library, not a content store.** It never reads
   or produces content. It works on step handles, terms, and residues.
2. **Purpose is stateful at the session level and pure underneath.** A
   `Session` accumulates history; the operators that act on it (seek,
   necessity, knapsack, floor, residue) are pure and independently
   exposed.
3. **Term extraction (`τ`) is caller policy, not library policy.**
   Purpose accepts terms; it does not compute them. Under no
   circumstances does purpose ship a default tokenizer, embedder, or
   normalizer. Callers hand purpose a `Set<string>` per step and are
   responsible for what's in it.
4. **Cost is caller policy.** Purpose accepts a number for each step's
   token cost. It does not tokenize.
5. **Payloads are opaque.** Steps may carry arbitrary caller-side data
   in a `payload: unknown` slot. Purpose never inspects it.
6. **Everything in the core is synchronous.** Async is a distraction
   the core doesn't need. Callers handle any I/O before calling `carry`.
7. **Zero side effects at the library level.** No global state, no
   singletons. Every session is an instance the caller holds.
8. **Errors are typed, not thrown.** Invalid inputs produce a
   `CarryResult` with `ok: false` and a specific error kind. Only
   programmer errors (calling the wrong method on the wrong shape)
   should throw.

---

## 4. The public API (locked at 1.0)

### 4.1 Types

```typescript
/** Opaque caller-chosen identifier. Purpose treats it as a string only. */
export type StepId = string;

/** A term is a lowercase string; purpose does not care what it means. */
export type Term = string;

/** One committed step in the history. */
export interface Step {
  /** Caller-chosen unique id. Must be unique within the session. */
  id: StepId;

  /** The distinctions this step draws. This is the paper's τ(u). */
  terms: ReadonlySet<Term>;

  /** Token cost of this step's content, computed by the caller. */
  cost: number;

  /** Monotonic timestamp (ms since epoch or a logical clock). */
  timestamp: number;

  /** Opaque caller data. Purpose does not inspect this. */
  payload?: unknown;
}

/** A goal is a set of terms the next act of reasoning must resolve. */
export interface Goal {
  terms: ReadonlySet<Term>;
}

/** Optional configuration for a session. All fields have sensible defaults. */
export interface SessionConfig {
  /**
   * Branching factor for the cascade (paper §7). In v1 the cascade is a
   * single flat frame; this knob reserves shape for future k-ary
   * routing. Default: 1 (flat).
   */
  cascadeArity?: number;

  /**
   * Per-frame residue budget for the cascade. Ignored when
   * cascadeArity === 1. Default: Infinity.
   */
  frameBudget?: number;

  /**
   * Weight function for shared-term edges. Given the size of the term
   * intersection, return an edge weight > 0. Default: identity.
   */
  edgeWeight?: (sharedTermCount: number) => number;
}

/** The result of a carry request. */
export type CarryResult =
  | {
      ok: true;
      /** Step ids to include in the working context. */
      keep: StepId[];
      /**
       * Steps that are reachable from the goal but not load-bearing.
       * Callers may re-fetch these lazily if a downstream step needs
       * them (paper §5.2).
       */
      regenerable: StepId[];
      /** Steps that the goal cannot reach; free to drop (paper §5.3). */
      dropped: StepId[];
      /** The ambient floor β of the current context graph (paper §3). */
      ambientFloor: number;
      /** Residue per kept step, for downstream inspection. */
      residueMap: ReadonlyMap<StepId, number>;
      /**
       * Diagnostic info: the sum of costs of kept steps, budget used,
       * knapsack gap. Callers can log these but should not depend on
       * their exact numeric values.
       */
      diagnostics: {
        totalKeptCost: number;
        budgetRemaining: number;
        knapsackRelaxationGap: number;
      };
    }
  | {
      ok: false;
      error:
        | { kind: 'unknown-step'; stepId: StepId }
        | { kind: 'empty-goal' }
        | { kind: 'infeasible'; message: string };
    };

/** A serialized session snapshot, for persistence and hot reload. */
export interface SessionSnapshot {
  version: 1;
  config: SessionConfig;
  steps: Step[];
  /**
   * Purpose may include additional graph-derived state here (cached
   * cascade tree, cached residues). Long-grass treats this as opaque.
   */
  internal: unknown;
}
```

### 4.2 The Session class

```typescript
export class Session {
  constructor(config?: SessionConfig);

  /**
   * Register a new step in the history. Idempotent for the same id.
   * Adding a step with an id that already exists but with different
   * terms/cost is an error.
   */
  addStep(step: Step): void;

  /**
   * Remove a step by id. Returns true if the step existed. This is a
   * caller-initiated eviction; the paper's Free-Drop Theorem covers
   * only automatic drops of purposeless steps.
   */
  removeStep(id: StepId): boolean;

  /**
   * Compute the tandem carry for the given goal under the given
   * budget. Never throws for graph-related reasons; returns a
   * CarryResult with ok=false when the request cannot be satisfied.
   */
  carry(args: { goal: Goal; budget: number }): CarryResult;

  /**
   * The number of steps currently in the session.
   */
  stepCount(): number;

  /**
   * The ambient floor β of the current context graph. Constant-time
   * cached; recomputed on structural change.
   */
  floor(): number;

  /**
   * Serialize the session for persistence.
   */
  snapshot(): SessionSnapshot;

  /**
   * Reconstruct a session from a snapshot. Static factory.
   */
  static fromSnapshot(snapshot: SessionSnapshot): Session;
}
```

### 4.3 Pure operators (exported for testing and lightweight callers)

Callers who don't want a stateful Session can drive the pipeline manually
against a `ReadonlyArray<Step>`. Purpose exposes each stage:

```typescript
/**
 * Paper §5.1: reachable set from goal terms, BFS in linear time over
 * shared-term adjacency. Returns the reachable step ids and, for each,
 * its BFS distance from the goal.
 */
export function seek(
  steps: ReadonlyArray<Step>,
  goal: Goal,
): ReadonlyMap<StepId, number>;

/**
 * Paper §5.2: filter the reachable set to load-bearing steps. Under the
 * identification "necessary iff reachable" (Theorem 5.1), this is
 * equivalent to seek() in v1; the separate function reserves shape for
 * future explicit ablation-based necessity.
 */
export function necessary(
  steps: ReadonlyArray<Step>,
  reached: ReadonlyMap<StepId, number>,
  goal: Goal,
): ReadonlySet<StepId>;

/**
 * Paper §6: value-density greedy knapsack over the necessary set.
 * Value = residue / (1 + distance from goal); cost is Step.cost.
 * Returns the chosen ids in admission order.
 */
export function knapsack(
  steps: ReadonlyArray<Step>,
  necessary: ReadonlySet<StepId>,
  distances: ReadonlyMap<StepId, number>,
  residues: ReadonlyMap<StepId, number>,
  budget: number,
): {
  keep: StepId[];
  totalCost: number;
  relaxationGap: number;
};

/**
 * Paper §3: ambient floor β of the graph induced by a step set.
 * Returns 0 if fewer than two steps or no shared-term edges exist.
 */
export function floor(steps: ReadonlyArray<Step>): number;

/**
 * Paper §3: residue of one step relative to the graph induced by a step
 * set. v1 implementation may approximate the minimum cut; the returned
 * value MUST be ≥ floor(steps) for every non-isolated step.
 */
export function residue(
  steps: ReadonlyArray<Step>,
  stepId: StepId,
): number;
```

### 4.4 Exports summary

The package's `index.ts` (or `mod.ts`) exports exactly:

- Types: `Step`, `StepId`, `Term`, `Goal`, `SessionConfig`,
  `CarryResult`, `SessionSnapshot`.
- Class: `Session`.
- Pure operators: `seek`, `necessary`, `knapsack`, `floor`, `residue`.

Nothing else is public. Internal graph structures (adjacency lists,
cascade nodes, min-cut solvers) are not part of the public surface.

---

## 5. Behavioural guarantees

These are the correctness constraints long-grass will assume in
integration tests. Purpose's implementation is free to change as long as
these hold.

1. **Idempotent addStep.** Adding a step with an id already present and
   identical `terms`/`cost` is a no-op. Adding with the same id but
   different fields throws (this is a programmer error).
2. **Deterministic carry.** For a fixed session state and fixed goal /
   budget, `carry()` returns the same result across calls. No hidden
   randomness.
3. **Order independence.** Given the same set of steps, the result of
   `carry()` does not depend on the order they were added.
4. **Floor positivity when non-trivial.** If the graph has at least one
   shared-term edge, `floor()` returns a strictly positive number.
   Otherwise it returns 0.
5. **Necessity subset of reachability.** For any goal, the necessary set
   returned is a subset of the reachable set.
6. **Budget respected.** For any successful `carry()`, the sum of
   `Step.cost` over `keep` is ≤ `budget`.
7. **Free drop preserved.** `dropped` is disjoint from both `keep` and
   `regenerable`, and their union covers every step in the session.
8. **Snapshot round-trip.** For any session `s`,
   `Session.fromSnapshot(s.snapshot())` produces a session whose future
   `carry()` results are identical to `s`'s. This is the persistence
   contract that lets long-grass hot-reload or hydrate from server.

---

## 6. What's deferred (deliberately)

Long-grass explicitly does NOT depend on any of the following in v1.
Purpose is free to add them as minor releases; long-grass will consume
them incrementally.

- **The k-ary cascade (§7).** Long-grass will run with
  `cascadeArity: 1` and a single flat frame. The config knob reserves
  the shape.
- **Exact minimum-cut residue via max-flow.** Long-grass accepts an
  approximation as long as the "residue ≥ floor" invariant holds.
- **Streaming carries.** `carry()` returns synchronously in one shot.
- **Multi-goal simultaneous carries.** One goal per call.
- **Automatic step expiry / TTL.** Removal is caller-initiated.
- **Cross-session merging.** No merge / diff / rebase operations.
- **Persistence backend beyond `SessionSnapshot`.** Callers wire their
  own storage against `snapshot()` / `fromSnapshot()`.
- **A default term extractor τ.** This is stated as non-negotiable in
  §3 and repeated here for emphasis.

---

## 7. Non-goals

Explicitly outside the scope of purpose, even as future work:

- LLM calls of any kind.
- Token counting.
- Retrieval, embedding, similarity search.
- Content storage.
- Prompt assembly.
- The paper's synthesis / federation model (that's a different tool).

If any of these creep into the library, long-grass will refuse the
upgrade. The whole point of this contract is that purpose stays a
graph-and-residue library.

---

## 8. Integration surface in long-grass

Once purpose ships, long-grass will add:

- `long-grass/src/lib/purpose-terms.ts` — the τ implementation. Default:
  lowercase alphanumeric tokens, minimum length 3, drop a small
  stopword list, dedupe. This is long-grass policy, not purpose's.
- `long-grass/src/lib/purpose-cost.ts` — a token estimator. Default: a
  cheap heuristic (chars / 4). Users can swap in `js-tiktoken` later
  without touching purpose.
- `long-grass/src/lib/modules/purpose-carry-module.js` — the Buhera
  module adapter. Instruction shape:
  ```
  { kind: "carry", goal: string[], budget: number }
  ```
  Reads from a per-page `Session` singleton, calls `carry()`, returns
  an ActResult with `output_delta.kind = "purpose_carry"`.
- **Audit-log feeder.** After every registry dispatch, extract terms
  from the instruction via `purpose-terms.ts` and call
  `session.addStep(...)`. This is the load-bearing wire-up: it makes
  every module dispatch a Step in the purpose session automatically.
- **Renderer + `:context` meta command.** Show kept/dropped step counts,
  ambient floor, and the residue histogram in the terminal.

That's the shape. Nothing more, nothing less.

---

## 9. Testing expectations

Purpose should ship with:

- Unit tests for each pure operator.
- Round-trip tests for `snapshot()` / `fromSnapshot()`.
- Property tests: order-independence, budget-respected, necessity ⊆
  reachability, dropped ∩ keep = ∅.
- One end-to-end scenario: build a session with, say, 5 steps drawing
  on overlapping and disjoint term sets, run a carry for two different
  goals, assert the expected keep / drop partitions.

Long-grass will run its own integration test in the terminal (the
smoke test described in the closing example).

---

## 10. Long-grass smoke test (the "sufficient test")

The passing criterion for the integration is:

```
memory store "note1" = "TUM was founded in 1868"
memory store "note2" = "AIMe Registry is at TU Munich"
memory store "note3" = "buy milk from the corner shop"

item c = dispatch("purpose", {
  kind: "carry",
  goal: ["TUM", "AIMe"],
  budget: 500
})
```

- `c.ok === true`
- `c.output_delta.keep` contains the ids of note1 and note2
- `c.output_delta.dropped` contains the id of note3
- `c.output_delta.ambientFloor > 0`
- `c.output_delta.residueMap` has an entry for each id in keep

If that runs green, the integration is done.

---

## 11. Open questions purpose owns

These are decisions long-grass explicitly hands to purpose to make.
Nothing in this spec depends on the answer; whatever purpose picks,
long-grass will consume.

- The internal graph representation (adjacency list vs. adjacency
  matrix vs. incidence structure).
- How residue is approximated in v1 (min-incident-edge-weight is fine,
  but purpose may prefer something else).
- Whether snapshots are JSON-serializable or something binary — as
  long as `snapshot()` returns a value `fromSnapshot()` accepts.
- Internal caching policy for the ambient floor and per-step residues.
- Whether the pure operators use recursion or iteration.

---

## 12. Change management

Any proposed change to §4 (the public API) or §5 (behavioural
guarantees) is a coordination point. Purpose posts a proposal; long-grass
reviews and either accepts (with a version bump and migration note) or
rejects. Everything outside those sections is purpose's call alone.
