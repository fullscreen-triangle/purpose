// Reachability (seek) and necessity (paper §5).
//
// `seek` is the forward reachability from the goal terms: a breadth-first
// growth over shared-term adjacency, returning each reached step's
// distance from the goal.
//
// `necessary` is, per the reconciliation record (Decision 3), the
// load-bearing subset of the reachable set. v1 ships the honest interim
// `necessary := seek` — exact on tree-like contexts, over-retaining on
// redundant (diamond) ones, with the knapsack mitigating by ranking
// redundant steps last. The dominator-based exact test is the v1.x
// target; the function boundary is already the right one, so upgrading
// it never changes call sites.

import type { ContextGraphView, Goal, ItemId } from "./types.js";
import { goalSeeds, termAdjacency } from "./graph.js";

/**
 * Reach(goal): items whose terms are transitively goal-connected, with
 * BFS distance from the goal (paper Def: Reachability from the Goal).
 * Linear in the graph size.
 */
export function seek(
  view: ContextGraphView,
  goal: Goal,
): Map<ItemId, number> {
  const dist = new Map<ItemId, number>();
  const adj = termAdjacency(view);
  const seeds = goalSeeds(view, goal);

  const queue: ItemId[] = [];
  for (const s of seeds) {
    dist.set(s, 0);
    queue.push(s);
  }

  let head = 0;
  while (head < queue.length) {
    const u = queue[head++]!;
    const d = dist.get(u)!;
    for (const v of adj.get(u) ?? []) {
      if (!dist.has(v)) {
        dist.set(v, d + 1);
        queue.push(v);
      }
    }
  }
  return dist;
}

/**
 * The reachable set as ids (drops the distances).
 */
export function reach(view: ContextGraphView, goal: Goal): Set<ItemId> {
  return new Set(seek(view, goal).keys());
}

/**
 * nec(W, goal): the necessary (load-bearing) subset of a retained set W
 * (paper §5, Necessity). v1 interim: the members of W that are reachable
 * from the goal — `necessary ⊆ reachable`, with equality on tree-like
 * contexts (see reconciliation Decision 3). The exact dominator test
 * will narrow this on redundant contexts without changing the signature.
 */
export function necessary(
  view: ContextGraphView,
  retained: ReadonlySet<ItemId>,
  goal: Goal,
): Set<ItemId> {
  const reached = reach(view, goal);
  const out = new Set<ItemId>();
  for (const id of retained) if (reached.has(id)) out.add(id);
  return out;
}

/**
 * contrib(u, goal): does dropping u change the goal's reachable set?
 * v1 interim proxy consistent with `necessary := seek`: 1 if u is
 * reachable and removing it disconnects at least one otherwise-reachable
 * step from the goal, else 0. This is the cheap dominator-flavoured test;
 * it already narrows the pure `necessary := seek` on simple diamonds and
 * is the seed of the exact dominator upgrade.
 */
export function contribution(
  view: ContextGraphView,
  item: ItemId,
  goal: Goal,
  _retained: ReadonlySet<ItemId>,
): number {
  const withU = reach(view, goal);
  if (!withU.has(item)) return 0; // unreachable: purposeless

  // Reachable set with `item` removed from the adjacency.
  const pruned: ContextGraphView = {
    ...view,
    items: view.items.filter((it) => it.id !== item),
    edges: view.edges.filter((e) => e.a !== item && e.b !== item),
  };
  const withoutU = reach(pruned, goal);

  // Did any step (other than u itself) lose reachability?
  for (const id of withU) {
    if (id === item) continue;
    if (!withoutU.has(id)) return 1; // u dominated a route to `id`
  }
  return 0; // u is reachable but redundant — a diamond leg
}
