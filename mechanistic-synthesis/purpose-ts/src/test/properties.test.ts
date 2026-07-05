// Property and invariant tests (buhera-specifications.md §5, §9).
// Run with: node --test dist/test/
//
// One assertion cluster per behavioural guarantee. These are the
// correctness constraints long-grass assumes in integration.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  Session,
  seek,
  necessary,
  floor,
  residue,
  carryGreedy,
  carryExact,
  type Step,
  type CarryItem,
} from "../index.js";

function mk(id: string, terms: string[], cost: number): Step {
  return { id, terms: new Set(terms), cost, timestamp: terms.length };
}

const notes: Step[] = [
  mk("n1", ["tum", "founded", "1868"], 24),
  mk("n2", ["aime", "registry", "tum", "munich"], 30),
  mk("n3", ["buy", "milk", "corner", "shop"], 28),
  mk("n4", ["munich", "weather", "cold"], 20),
];
const goal = { terms: new Set(["tum", "aime"]) };

test("guarantee 2: carry is deterministic", () => {
  const s = new Session();
  notes.forEach((n) => s.addStep(n));
  const a = s.carry({ goal, budget: 500 });
  const b = s.carry({ goal, budget: 500 });
  assert.deepEqual(a, b);
});

test("guarantee 3: order independence", () => {
  const s1 = new Session();
  notes.forEach((n) => s1.addStep(n));
  const s2 = new Session();
  [...notes].reverse().forEach((n) => s2.addStep(n));
  const r1 = s1.carry({ goal, budget: 500 });
  const r2 = s2.carry({ goal, budget: 500 });
  assert.ok(r1.ok && r2.ok);
  if (r1.ok && r2.ok) {
    assert.deepEqual([...r1.keep].sort(), [...r2.keep].sort());
    assert.deepEqual([...r1.dropped].sort(), [...r2.dropped].sort());
  }
});

test("guarantee 4: floor positive when a shared-term edge exists, 0 otherwise", () => {
  assert.ok(floor(notes) > 0); // n1,n2 share "tum"; n2,n4 share "munich"
  const disjoint = [mk("a", ["x"], 1), mk("b", ["y"], 1)];
  assert.equal(floor(disjoint), 0);
  assert.equal(floor([mk("solo", ["z"], 1)]), 0); // < 2 steps
});

test("guarantee 5: necessary is a subset of reachable", () => {
  const s = new Session();
  notes.forEach((n) => s.addStep(n));
  const reached = seek(notes, goal);
  const nec = necessary(notes, reached, goal);
  for (const id of nec) assert.ok(reached.has(id));
});

test("guarantee 6: budget respected", () => {
  const s = new Session();
  notes.forEach((n) => s.addStep(n));
  const r = s.carry({ goal, budget: 40 }); // tight: only one 24/30 note fits
  assert.ok(r.ok);
  if (r.ok) {
    const cost = r.keep.reduce(
      (acc, id) => acc + (notes.find((n) => n.id === id)?.cost ?? 0),
      0,
    );
    assert.ok(cost <= 40, `kept cost ${cost} exceeds budget 40`);
  }
});

test("guarantee 7: keep / regenerable / dropped partition the history", () => {
  const s = new Session();
  notes.forEach((n) => s.addStep(n));
  const r = s.carry({ goal, budget: 40 });
  assert.ok(r.ok);
  if (r.ok) {
    const keep = new Set(r.keep);
    const regen = new Set(r.regenerable);
    const drop = new Set(r.dropped);
    // pairwise disjoint
    for (const id of keep) assert.ok(!regen.has(id) && !drop.has(id));
    for (const id of regen) assert.ok(!drop.has(id));
    // cover every step exactly once
    const union = new Set([...keep, ...regen, ...drop]);
    assert.equal(union.size, notes.length);
  }
});

test("residue >= floor for every non-isolated step", () => {
  const f = floor(notes);
  for (const n of notes.filter((x) => x.id !== "n3")) {
    // n3 is disjoint from the rest; others share terms
    assert.ok(residue(notes, n.id) >= f, `residue(${n.id}) < floor`);
  }
});

test("guarantee 8: snapshot round-trip preserves carry", () => {
  const s = new Session();
  notes.forEach((n) => s.addStep(n));
  const restored = Session.fromSnapshot(s.snapshot());
  assert.deepEqual(
    s.carry({ goal, budget: 500 }),
    restored.carry({ goal, budget: 500 }),
  );
});

test("carry errors are returned, not thrown", () => {
  const s = new Session();
  notes.forEach((n) => s.addStep(n));
  const empty = s.carry({ goal: { terms: new Set() }, budget: 500 });
  assert.equal(empty.ok, false);
  if (!empty.ok) assert.equal(empty.error.kind, "empty-goal");
  const badBudget = s.carry({ goal, budget: 0 });
  assert.equal(badBudget.ok, false);
  if (!badBudget.ok) assert.equal(badBudget.error.kind, "infeasible");
});

test("addStep idempotent for identical, throws on conflict", () => {
  const s = new Session();
  s.addStep(mk("x", ["a"], 5));
  s.addStep(mk("x", ["a"], 5)); // idempotent no-op
  assert.equal(s.stepCount(), 1);
  assert.throws(() => s.addStep(mk("x", ["a"], 9))); // conflicting cost
});

test("knapsack: greedy within cost_max/budget of exact", () => {
  const items: CarryItem[] = [
    { id: "a", value: 60, cost: 10 },
    { id: "b", value: 100, cost: 20 },
    { id: "c", value: 120, cost: 30 },
  ];
  const budget = 50;
  const g = carryGreedy(items, budget);
  const e = carryExact(items, budget);
  const costMax = Math.max(...items.map((i) => i.cost));
  // greedy value >= exact * (1 - cost_max/budget)
  assert.ok(g.totalValue >= e.totalValue * (1 - costMax / budget) - 1e-9);
  assert.ok(e.totalValue >= g.totalValue - 1e-9); // exact is optimal
});
