# The Graffiti <-> Purpose interface

Status: specification, not yet implemented. Nothing under `src/graffiti`
currently imports a `purpose` package; this document is the contract both
sides build to, so each can be implemented (and re-implemented) without
reopening the other.

## Why this interface exists

`docs/sources/purpose-propagation.tex` ("Carry the Uncertainty, Not the
Knowledge") proves a calculus for what a finite reasoning agent should
carry forward across steps of a growing history: not the accumulated
content, but a bounded, reachability-pruned, budget-fitted slice of
*residue*. Graffiti's own manuscript
(`docs/publications/semantic-causal-propagation/semantic-causal-propagation.tex`)
proves the calculus of *search* -- individuating one claim from a medium.
The two share their foundational machinery (a contact/context graph with
a medium vertex, a derived positive floor, minimum-cut residue) but prove
theorems about different questions: Graffiti's `seek` answers "what claim
does this query individuate"; Purpose's tandem (`seek` + `nec`) answers
"of everything already individuated, what is still load-bearing, and
what fits in the budget."

Today, `Interpreter` (`lang/interpreter.ts`) accumulates a `ContactGraph`
that only ever grows for the life of one `runProject`/`runSource` call,
then is discarded (`orchestration/orchestrator.ts:92` constructs a fresh
`Interpreter` per call). There is no notion of a session spanning many
calls, no pruning, no budget. Purpose is the module responsible for that:
it takes Graffiti's committed history, decides what remains necessary for
a goal, fits it to a budget, and hands back what to keep. Graffiti never
implements necessity, carry, or cascade logic itself -- it only calls
Purpose through the boundary specified below and applies what comes back.

## Division of responsibility

| Question | Owned by | Existing code |
|---|---|---|
| What claim does a query individuate? | Graffiti (`seek`) | `lang/interpreter.ts` |
| Is a claim reachable from a goal? | Purpose (`seek`/`reach`, paper's operator, not Graffiti's) | none yet -- Purpose |
| Does dropping a claim change what the goal can resolve? | Purpose (`nec`) | none yet -- Purpose |
| Which necessary claims fit a token budget? | Purpose (knapsack carry) | none yet -- Purpose |
| How is a long session's context organised into bounded frames? | Purpose (cascade) | none yet -- Purpose |
| How is a claim committed, and what is its residue? | Graffiti (`ContactGraph`) | `core/graph.ts`, `core/maxflow.ts` |
| What does a session actually do with a carry decision? | Graffiti (`Interpreter`/session wrapper) | to be added, see below |

Purpose never invokes a catalyst, never parses `.grf` source, and never
decides search outcomes (convergence/decline). Graffiti never decides
necessity, never runs a knapsack, and never owns cascade routing. The
interface below is the only channel between them.

## Package boundary

Purpose ships as its own package (e.g. `@graffiti/purpose` or a sibling
directory imported by path, the user's choice at implementation time) and
is imported by Graffiti, never the reverse. Purpose's public surface must
depend on nothing Graffiti-specific: every type it exports is expressed
in terms of a generic *item* (a claim, in Graffiti's vocabulary, but
Purpose does not know that word) and a generic weighted graph. This is
what lets Purpose be implemented, tested, and versioned independently,
and lets Graffiti swap Purpose implementations (e.g. a pure-TS one now, a
WASM-backed one later) without changing call sites.

```
web/src/graffiti/     -- imports purpose, never the reverse
web/src/purpose/      -- (or an external package) -- imports nothing from graffiti
```

## 1. The shared substrate: `ContextGraph`

Purpose's every theorem operates on a finite weighted graph with a medium
vertex -- structurally identical to Graffiti's `ContactGraph`
(`core/graph.ts`). Rather than Purpose depending on Graffiti's class, or
Graffiti depending on Purpose's, the interface is a plain data shape both
sides can construct and read, with no imported class on either side of
the boundary:

```ts
/** A step/claim identifier -- opaque to Purpose, meaningful to Graffiti. */
export type ItemId = string;

/** The distinguished vertex standing for "everything not yet individuated". */
export type MediumId = string; // Graffiti passes its own MEDIUM constant's value

export interface WeightedEdge {
  a: ItemId;
  b: ItemId;
  weight: number; // > 0
}

/**
 * The graph Purpose reasons over. Graffiti constructs one from its live
 * ContactGraph on every Purpose call (see ContextGraphAdapter below);
 * Purpose never mutates it and never holds a reference to it after a
 * call returns.
 */
export interface ContextGraphView {
  medium: MediumId;
  items: ItemId[]; // excludes medium
  edges: WeightedEdge[]; // undirected; both endpoints in items or medium
  floor: number; // the derived positive floor beta, shared with Graffiti's
}
```

`ContextGraphView` is intentionally a plain snapshot, not a class with
behavior. Purpose is free to build its own internal graph/flow
representation from it (mirroring `core/maxflow.ts`'s `FlowNetwork`, or
reusing it via a thin adapter -- see below) but the interface itself
carries no computation.

### `ContextGraphAdapter` (lives in Graffiti, not Purpose)

Graffiti owns the adapter that turns a live `ContactGraph` into a
`ContextGraphView`, since only Graffiti knows its internal edge storage:

```ts
// web/src/graffiti/purpose-adapter.ts (new file)
import type { ContactGraph } from "./core/graph";
import type { ContextGraphView } from "purpose";

export function toContextGraphView(graph: ContactGraph): ContextGraphView;
```

This keeps `ContactGraph`'s internal `Map<string, number>` edge storage
private; Purpose only ever sees the exported view shape.

## 2. Term tagging: `τ(u)`

Purpose's reachability (`reach(goal)`) is defined over *term* sharing
(Definition: Reachability from the Goal, `purpose-propagation.tex`
`sec:necessity`), not over the contact graph's edges directly. Graffiti's
committed claims are currently untagged strings (e.g.
`"web_search:unresolved:budget_overrun:seed:corroborated"`) -- there is
no explicit term set per claim today. This is the one piece of new data
Graffiti must start recording for Purpose to be usable at all:

```ts
/** The distinctions a claim/step draws -- Purpose's tau(u). Graffiti-owned data. */
export type TermSet = ReadonlySet<string>;

export interface TaggedItem {
  id: ItemId;
  terms: TermSet;
}
```

`ContextGraphView` gains one more field so Purpose can compute
reachability without re-deriving terms itself:

```ts
export interface ContextGraphView {
  medium: MediumId;
  items: TaggedItem[]; // replaces items: ItemId[] above
  edges: WeightedEdge[];
  floor: number;
}
```

**Where Graffiti gets terms from (concrete, not hypothetical):** every
committed claim already passes through `evalArgsToRecord` in
`lang/interpreter.ts:67`, which flattens a catalyst call's arguments to a
`Record<string, unknown>`. The term set for a claim is the union of:

- the catalyst name that produced it (`step.qname.join(".")` in
  `runBranch`, `lang/interpreter.ts:197`),
- the stringified values of `evalArgsToRecord`'s result,
- the seek name it was committed under (`seek.name`).

This is a mechanical, low-risk addition: `commitContact` and `ensureClaim`
in `lang/interpreter.ts` need to accept and store a `TermSet` alongside
each claim they already track, threaded through from `runBranch` and
`executeSeek`. No existing behavior changes; this is additive state.

## 3. Goal: `Goal`

Purpose's goal is a term set, not a Graffiti-specific type:

```ts
export interface Goal {
  terms: TermSet;
}
```

**Where Graffiti constructs a `Goal` from:** a `seek`'s `toward{}` region
(already resolved to a string by `regionTargetName`,
`lang/interpreter.ts:81`) plus its `not{}` boundary terms (already parsed
into `BoundaryExpr`, `lang/ast.ts:69`) plus, if present, the enclosing
`goal{}` block's claim list (`GoalBlock.claims`, `lang/ast.ts:48`).
Graffiti is responsible for this construction; Purpose only ever consumes
a `Goal` value, never parses `.grf` syntax to get one.

## 4. Purpose's public surface

Everything below is what Purpose exports; Graffiti only ever calls these
functions, never reimplements them.

```ts
// ---- Necessity (purpose-propagation.tex, sec:necessity) --------------

/** Reach(goal): items whose terms are transitively goal-connected (Def: Reachability). */
export function reach(graph: ContextGraphView, goal: Goal): Set<ItemId>;

/** contrib(u, goal): >0 iff dropping u changes the goal's reachable resolution (Def: Contribution). */
export function contribution(graph: ContextGraphView, item: ItemId, goal: Goal, retained: ReadonlySet<ItemId>): number;

/** nec(W, goal): the necessary subset of a retained set W (Thm: Necessity Equals Reachability). */
export function necessary(graph: ContextGraphView, retained: ReadonlySet<ItemId>, goal: Goal): Set<ItemId>;

// ---- Knapsack carry (sec:knapsack) ------------------------------------

export interface CarryItem {
  id: ItemId;
  value: number;   // caller-supplied, or use defaultValue() below
  cost: number;    // token/size cost, caller-supplied (Graffiti: serialized claim length, or similar)
}

export interface CarryResult {
  keep: Set<ItemId>;
  totalValue: number;
  totalCost: number;
}

/** Value-density greedy carry (Thm: Value-Density Greedy Is Optimal Under the Relaxation). */
export function carryGreedy(items: readonly CarryItem[], budget: number): CarryResult;

/** Exact 0/1 knapsack DP (Thm: The Optimal Carry Is a 0/1 Knapsack). O(items * budget). */
export function carryExact(items: readonly CarryItem[], budget: number): CarryResult;

/** Canonical value assignment: residue / (1 + distance-from-goal) (sec:knapsack). */
export function defaultValue(residue: number, distanceFromGoal: number): number;

// ---- Cascade (sec:cascade) -- phase 2, see "What Graffiti does NOT need yet" ----

export interface CascadeFrame {
  id: string;
  items: ItemId[]; // resident residues in this frame
  children: CascadeFrame[];
}

export interface RouteResult {
  framesVisited: string[];
  resolvedIn: string | null; // frame id that resolved the goal, or null if it descended to a leaf and declined
}

export function route(root: CascadeFrame, goal: Goal): RouteResult;
export function fileInto(root: CascadeFrame, item: TaggedItem, branchingFactor: number): CascadeFrame; // returns updated tree

// ---- The tandem entry point (sec:tandem) -- the one call Graffiti actually makes ----

export interface TandemCarryOptions {
  budget: number;
  valueOf?: (item: ItemId, graph: ContextGraphView, goal: Goal) => number; // defaults to defaultValue()
  costOf: (item: ItemId, graph: ContextGraphView) => number; // Graffiti must supply this (token cost is Graffiti's data)
}

export interface TandemCarryResult {
  keep: Set<ItemId>;
  drop: Set<ItemId>;
  necessary: Set<ItemId>; // pre-budget necessary set, for diagnostics/logging
  carry: CarryResult;
}

/** Construction: The Tandem Carry. Runs seek -> prune -> fit in one call. */
export function tandemCarry(
  graph: ContextGraphView,
  retained: ReadonlySet<ItemId>,
  goal: Goal,
  options: TandemCarryOptions,
): TandemCarryResult;
```

`tandemCarry` is the one function Graffiti actually needs to call in the
common case; `reach`/`necessary`/`carryGreedy`/`carryExact` are exposed
separately for diagnostics, testing, and for callers who want to log or
inspect the intermediate necessity set (`rem:above` in the paper is
explicit that necessity is a layer-above judgment, not something a step
decides for itself -- logging it is expected, not incidental).

## 5. What Graffiti must add on its side

1. **Term tagging** (`§2` above): thread a `TermSet` through
   `commitContact`/`ensureClaim` in `lang/interpreter.ts`. Additive; no
   existing field or method signature is removed, only extended.
2. **`purpose-adapter.ts`**: `toContextGraphView(graph: ContactGraph): ContextGraphView`,
   converting the interpreter's private edge map into the plain view
   shape. One function, no state.
3. **Session persistence.** This is the one real structural change.
   Today `runProject` (`orchestration/orchestrator.ts:90`) constructs a
   fresh `Interpreter` every call and throws it away. For Purpose's
   pruning/carry to have any effect, something must persist an
   `Interpreter` (or its `ContactGraph` + term tags) across multiple
   `runProject`/`runSource` calls within one session, calling
   `tandemCarry` between calls and applying `TandemCarryResult.drop` by
   removing those claims (and their edges) from the retained graph.
   This is new code (a `Session` wrapper, not yet designed in this
   document -- proposed name `GraffitiSession` in
   `web/src/graffiti/orchestration/session.ts`), not a change to
   `Interpreter`'s existing per-call behavior, which remains correct and
   usable standalone (a `Session` wraps it, does not replace it).
4. **`costOf` implementation.** Purpose's `TandemCarryOptions.costOf` is
   deliberately left to Graffiti: Purpose has no opinion on what a
   "token" is. Graffiti's natural cost function is the serialized length
   of a claim's string identity, or a fixed per-claim cost if a coarser
   measure is preferred. This is a small function, not a module.

## 6. What Graffiti does NOT need yet

The cascade (`route`/`fileInto`/`CascadeFrame`) is specified above for
interface completeness -- so a future addition doesn't reshape the
already-agreed contract -- but Graffiti should not call it until a
single, `tandemCarry`-pruned `ContextGraphView` is demonstrated
insufficient in practice (i.e. one session's necessary-and-budgeted slice
still exceeds what's workable). Building the cascade before that is
speculative; the paper itself presents it as the answer to a scale
problem (`§sec:cascade`: "across many goals in a long session, the
reachable slices themselves accumulate"), not a precondition for
necessity or the knapsack carry to be useful standalone.

## 7. Testing contract

Purpose is responsible for proving its own theorems hold (mirroring
Graffiti's `test/t01`-`t13` discipline: one test module per theorem --
`reach`-equals-`necessary` on constructed instances, free-drop leaves
`Res(goal | *)` unchanged, greedy carry within the stated
`cost_max/budget` bound of the exact DP, and so on). Graffiti is
responsible for one additional integration test once `session.ts`
exists: a session across multiple `runSource` calls where an
unnecessary claim from an earlier call is confirmed pruned (not present
in the graph handed to a later call) and a necessary one survives. That
test lives in `web/src/graffiti/test/` once written, following the
existing `t01`-style module convention, numbered after `t13`.

## 8. Versioning and stability

Every type in `§1`-`§4` above is the frozen contract. Purpose's *internal*
graph representation, its choice of max-flow algorithm, and its cascade
branching strategy are free to change without notice to Graffiti, as
long as the exported function signatures in `§4` are preserved. If a
signature in `§4` must change, that is a breaking change to this
document and should be flagged as such, not silently absorbed.
