// The knapsack carry (paper §6).
//
// Given the necessary steps, each with a value (worth to the goal) and a
// cost (token size), and a budget, choose the subset of maximum value
// within budget. Value-density greedy is optimal under the fractional
// relaxation; the exact 0/1 DP attains the integral optimum.

import type { ItemId } from "./types.js";

/** One candidate for the carry. */
export interface CarryItem {
  id: ItemId;
  value: number; // ≥ 0
  cost: number; // > 0 (token/size)
}

/** The result of a carry: which items to keep, and the totals. */
export interface CarryOutcome {
  keep: ItemId[]; // in admission order
  totalValue: number;
  totalCost: number;
  /** Upper bound on the relative optimality gap (cost_max / budget). */
  relaxationGap: number;
}

/** Canonical value: residue discounted by distance from the goal (§6). */
export function defaultValue(residue: number, distanceFromGoal: number): number {
  return residue / (1 + distanceFromGoal);
}

/**
 * Value-density greedy carry (Thm: Value-Density Greedy Is Optimal Under
 * the Relaxation). Admits items in decreasing value/cost while the budget
 * allows. Optimal under the fractional relaxation; within cost_max/budget
 * of the integral optimum.
 */
export function carryGreedy(
  items: ReadonlyArray<CarryItem>,
  budget: number,
): CarryOutcome {
  const ranked = [...items]
    .filter((it) => it.cost > 0 && it.value > 0)
    .sort((a, b) => b.value / b.cost - a.value / a.cost);

  const keep: ItemId[] = [];
  let totalValue = 0;
  let totalCost = 0;
  let costMax = 0;
  for (const it of ranked) {
    if (it.cost > costMax) costMax = it.cost;
    if (totalCost + it.cost <= budget) {
      keep.push(it.id);
      totalValue += it.value;
      totalCost += it.cost;
    }
  }
  const relaxationGap = budget > 0 ? costMax / budget : 0;
  return { keep, totalValue, totalCost, relaxationGap };
}

/**
 * Exact 0/1 knapsack DP (Thm: The Optimal Carry Is a 0/1 Knapsack).
 * O(items * budget). Requires integer costs and budget; for non-integer
 * costs, callers should round costs up to integers (a conservative carry)
 * or use carryGreedy. Falls back to greedy if any cost is non-integer.
 */
export function carryExact(
  items: ReadonlyArray<CarryItem>,
  budget: number,
): CarryOutcome {
  const usable = items.filter((it) => it.cost > 0 && it.value > 0);
  const allInt =
    Number.isInteger(budget) && usable.every((it) => Number.isInteger(it.cost));
  if (!allInt) {
    // DP needs an integer table; greedy is the honest fallback.
    return carryGreedy(items, budget);
  }

  const n = usable.length;
  const B = budget;
  // T[i][b] = best value using first i items within budget b.
  const T: number[][] = Array.from({ length: n + 1 }, () =>
    new Array<number>(B + 1).fill(0),
  );
  for (let i = 1; i <= n; i++) {
    const it = usable[i - 1]!;
    for (let b = 0; b <= B; b++) {
      const without = T[i - 1]![b]!;
      const with_ =
        it.cost <= b ? it.value + T[i - 1]![b - it.cost]! : -Infinity;
      T[i]![b] = Math.max(without, with_);
    }
  }

  // Backtrack to recover the chosen set.
  const keep: ItemId[] = [];
  let b = B;
  let totalValue = 0;
  let totalCost = 0;
  let costMax = 0;
  for (let i = n; i >= 1; i--) {
    if (usable[i - 1]!.cost > costMax) costMax = usable[i - 1]!.cost;
    if (T[i]![b]! !== T[i - 1]![b]!) {
      const it = usable[i - 1]!;
      keep.push(it.id);
      totalValue += it.value;
      totalCost += it.cost;
      b -= it.cost;
    }
  }
  keep.reverse();
  const relaxationGap = budget > 0 ? costMax / budget : 0;
  return { keep, totalValue, totalCost, relaxationGap };
}
