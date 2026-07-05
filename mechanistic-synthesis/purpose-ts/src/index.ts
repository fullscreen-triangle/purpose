// @buhera/purpose — public surface.
//
// Two entry heights over one implementation (reconciliation Decision 1):
//   - the pure core (over a ContextGraphView) for callers that own their
//     own graph and session (Graffiti);
//   - the Session class (over Step[]) for callers that want purpose to
//     hold the history (long-grass / Buhera).
//
// buhera-specifications.md §4.4 asks the top level to export the Session,
// the shared types, and the pure operators `seek`/`necessary`/`knapsack`/
// `floor`/`residue` in a Step[]-flavoured form. Those wrappers are here;
// the graph-view-flavoured operators live under the `./core` subpath.

import type { Goal, Step, StepId, ContextGraphView } from "./core/index.js";
import {
  buildGraph,
  seek as seekView,
  necessary as necessaryView,
  reach as reachView,
  floor as floorSteps,
  residue as residueSteps,
  defaultValue,
  carryGreedy,
  type CarryItem,
  type EdgeWeight,
} from "./core/index.js";

// ---- Shared types ----
export type {
  StepId,
  ItemId,
  Term,
  TermSet,
  MediumId,
  Step,
  TaggedItem,
  Goal,
  WeightedEdge,
  ContextGraphView,
} from "./core/index.js";

// ---- The Session class (stateful layer) ----
export { Session } from "./session.js";
export type { SessionConfig, CarryResult, SessionSnapshot } from "./session.js";

// ---- Pure operators (Step[]-flavoured, buhera §4.3) ----

/** Paper §5.1: reachable set from goal terms, with BFS distance. */
export function seek(
  steps: ReadonlyArray<Step>,
  goal: Goal,
  opts?: { edgeWeight?: EdgeWeight },
): ReadonlyMap<StepId, number> {
  return seekView(buildGraph(steps, opts), goal);
}

/** Paper §5.2: load-bearing subset of the reachable set (v1: = reachable). */
export function necessary(
  steps: ReadonlyArray<Step>,
  reached: ReadonlyMap<StepId, number>,
  goal: Goal,
  opts?: { edgeWeight?: EdgeWeight },
): ReadonlySet<StepId> {
  const view = buildGraph(steps, opts);
  return necessaryView(view, new Set(reached.keys()), goal);
}

/** Paper §6: value-density greedy knapsack over the necessary set. */
export function knapsack(
  _steps: ReadonlyArray<Step>,
  necessarySet: ReadonlySet<StepId>,
  distances: ReadonlyMap<StepId, number>,
  residues: ReadonlyMap<StepId, number>,
  budget: number,
  costOf: (id: StepId) => number,
): { keep: StepId[]; totalCost: number; relaxationGap: number } {
  const items: CarryItem[] = [];
  for (const id of necessarySet) {
    const r = residues.get(id) ?? 0;
    const d = distances.get(id) ?? 0;
    items.push({ id, value: defaultValue(r, d), cost: costOf(id) });
  }
  const out = carryGreedy(items, budget);
  return { keep: out.keep, totalCost: out.totalCost, relaxationGap: out.relaxationGap };
}

/** Paper §3: ambient floor β of the graph induced by a step set. */
export function floor(
  steps: ReadonlyArray<Step>,
  opts?: { edgeWeight?: EdgeWeight },
): number {
  return floorSteps(steps, opts);
}

/** Paper §3: residue of one step relative to the induced graph. */
export function residue(
  steps: ReadonlyArray<Step>,
  stepId: StepId,
  opts?: { edgeWeight?: EdgeWeight },
): number {
  return residueSteps(steps, stepId, opts);
}

// ---- Re-exports for graph-view-flavoured callers (Graffiti) ----
export { reachView as reachInView, buildGraph, defaultValue };
export type { CarryItem, EdgeWeight };
