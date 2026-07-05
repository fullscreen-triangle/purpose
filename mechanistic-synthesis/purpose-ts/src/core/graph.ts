// Graph construction, floor, and residue (paper §2–§3).
//
// The context graph joins two steps when their term sets intersect; the
// edge weight is a function of how many terms they share. The medium is
// adjacent to every step. The floor β is the minimum positive edge
// weight; every genuine step's residue is ≥ β (Floor Theorem).

import type {
  ContextGraphView,
  Goal,
  ItemId,
  Step,
  TaggedItem,
  Term,
  TermSet,
  WeightedEdge,
} from "./types.js";
import { DEFAULT_MEDIUM } from "./types.js";

/** How an edge weight is derived from the number of shared terms. */
export type EdgeWeight = (sharedTermCount: number) => number;

/** Default edge weight: identity on the shared-term count. */
export const identityWeight: EdgeWeight = (n) => n;

function intersectionSize(a: TermSet, b: TermSet): number {
  // Iterate the smaller set for cost.
  const [small, big] = a.size <= b.size ? [a, b] : [b, a];
  let n = 0;
  for (const t of small) if (big.has(t)) n++;
  return n;
}

/**
 * Build a ContextGraphView from a set of steps (the Session's job, and
 * the reference construction for graffiti's adapter to match).
 *
 * - one item per step (terms carried through),
 * - an undirected edge between two steps sharing ≥1 term, weighted by
 *   edgeWeight(sharedCount),
 * - an edge from every step to the medium, weighted by the step's own
 *   term count (its cost of being told apart from "everything else"),
 * - floor = minimum positive edge weight.
 */
export function buildGraph(
  steps: ReadonlyArray<Step>,
  opts?: { medium?: string; edgeWeight?: EdgeWeight },
): ContextGraphView {
  const medium = opts?.medium ?? DEFAULT_MEDIUM;
  const edgeWeight = opts?.edgeWeight ?? identityWeight;

  const items: TaggedItem[] = steps.map((s) => ({ id: s.id, terms: s.terms }));
  const edges: WeightedEdge[] = [];

  // Medium edges: every step is individuated against the rest at a cost
  // proportional to how many distinctions it draws (min 1, so the floor
  // is well-defined even for a single-term step).
  for (const s of steps) {
    const w = edgeWeight(Math.max(1, s.terms.size));
    if (w > 0) edges.push({ a: s.id, b: medium, weight: w });
  }

  // Shared-term edges between steps.
  for (let i = 0; i < steps.length; i++) {
    for (let j = i + 1; j < steps.length; j++) {
      const si = steps[i]!;
      const sj = steps[j]!;
      const shared = intersectionSize(si.terms, sj.terms);
      if (shared > 0) {
        const w = edgeWeight(shared);
        if (w > 0) edges.push({ a: si.id, b: sj.id, weight: w });
      }
    }
  }

  const floor = floorOfEdges(edges);
  return { medium, items, edges, floor };
}

/** The floor β: minimum positive edge weight; 0 if no positive edges. */
export function floorOfEdges(edges: ReadonlyArray<WeightedEdge>): number {
  let min = Infinity;
  for (const e of edges) if (e.weight > 0 && e.weight < min) min = e.weight;
  return min === Infinity ? 0 : min;
}

/**
 * Ambient floor of the graph induced by a step set (paper §3).
 * 0 if fewer than two steps or no shared-term edges exist
 * (buhera-specifications.md §4.3 `floor`).
 */
export function floor(
  steps: ReadonlyArray<Step>,
  opts?: { medium?: string; edgeWeight?: EdgeWeight },
): number {
  if (steps.length < 2) return 0;
  // Floor is defined by shared-term edges between steps, not medium edges.
  const edgeWeight = opts?.edgeWeight ?? identityWeight;
  let min = Infinity;
  for (let i = 0; i < steps.length; i++) {
    for (let j = i + 1; j < steps.length; j++) {
      const shared = intersectionSize(steps[i]!.terms, steps[j]!.terms);
      if (shared > 0) {
        const w = edgeWeight(shared);
        if (w > 0 && w < min) min = w;
      }
    }
  }
  return min === Infinity ? 0 : min;
}

/**
 * Residue of one step relative to the graph induced by a step set
 * (paper §3). v1 approximation: the minimum positive weight of an edge
 * incident to the step (its cheapest separator). Guaranteed ≥ floor for
 * every non-isolated step (buhera-specifications.md §4.3 `residue`).
 * Returns the medium-edge weight for an otherwise-isolated step, which is
 * that step's cost of being told apart from everything else.
 */
export function residue(
  steps: ReadonlyArray<Step>,
  stepId: ItemId,
  opts?: { edgeWeight?: EdgeWeight },
): number {
  const edgeWeight = opts?.edgeWeight ?? identityWeight;
  const self = steps.find((s) => s.id === stepId);
  if (!self) return 0;

  let min = Infinity;
  for (const other of steps) {
    if (other.id === stepId) continue;
    const shared = intersectionSize(self.terms, other.terms);
    if (shared > 0) {
      const w = edgeWeight(shared);
      if (w > 0 && w < min) min = w;
    }
  }
  if (min !== Infinity) return min;
  // Isolated among steps: fall back to its medium separator cost.
  return edgeWeight(Math.max(1, self.terms.size));
}

/** Adjacency over shared-term edges only (excludes the medium). */
export function termAdjacency(
  view: ContextGraphView,
): Map<ItemId, Set<ItemId>> {
  const adj = new Map<ItemId, Set<ItemId>>();
  for (const it of view.items) adj.set(it.id, new Set());
  for (const e of view.edges) {
    if (e.a === view.medium || e.b === view.medium) continue;
    adj.get(e.a)?.add(e.b);
    adj.get(e.b)?.add(e.a);
  }
  return adj;
}

/** The set of items whose terms intersect the goal (distance-0 seeds). */
export function goalSeeds(view: ContextGraphView, goal: Goal): Set<ItemId> {
  const seeds = new Set<ItemId>();
  for (const it of view.items) {
    for (const t of it.terms) {
      if (goal.terms.has(t)) {
        seeds.add(it.id);
        break;
      }
    }
  }
  return seeds;
}
