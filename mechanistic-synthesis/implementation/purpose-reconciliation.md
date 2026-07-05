# Purpose: reconciliation of the two consumer contracts

**Status:** decision record. Supersedes conflicting clauses in
`graffiti.md` and `buhera-specifications.md` where they disagree; both
should link here. Everything not touched below stands as written in the
originating doc.

**Author:** synthesis pass, 2026-07-04
**Depends on:** `docs/sources/purpose-propagation.tex`
("Carry the Uncertainty, Not the Knowledge")

`purpose` has two consumers who each wrote a contract for it:
`graffiti.md` (Graffiti side) and `buhera-specifications.md` (long-grass /
Buhera side). They agree on the substance -- terms are caller policy, cost
is caller policy, no LLM/embedder/tokenizer in the library, the operators
are floor/residue/seek/necessary/knapsack, the cascade is deferred. They
disagree on three points of shape and inherited one defect from the paper.
This document resolves all four so that a single `purpose` implementation
satisfies both, and the paper is corrected to match.

---

## Decision 1 — one library, two entry heights (layered)

There is no ownership conflict; there is a vocabulary collision. The
resolution is the layered shape that `buhera-specifications.md` §3.2
already names ("stateful at the session level and pure underneath"):

```
@buhera/purpose
  core/     -- pure operators over a plain graph view. No state.
            -- seek, necessary, knapsack, floor, residue.
            -- This is the layer Graffiti drives directly.
  session/  -- Session class: thin stateful wrapper over core.
            -- addStep / removeStep / carry / snapshot / fromSnapshot.
            -- Builds the graph from Step[] internally, calls core.
            -- This is the layer long-grass uses.
```

- **Package name:** `@buhera/purpose` (single package, npm-published,
  per `buhera-specifications.md` §1). Graffiti imports the same package;
  it simply imports from the `core` subpath and ignores `Session`.
- **Graffiti** owns its `ContactGraph` and session persistence. It calls
  the **pure core** with a `ContextGraphView` it builds via its own
  `toContextGraphView` adapter (`graffiti.md` §1). It never touches
  `Session`.
- **long-grass** has no pre-existing graph. It uses **`Session`**, hands
  it `Step[]`, and lets `purpose` build the graph
  (`buhera-specifications.md` §4.2). It never touches `ContextGraphView`.
- The `Session` wrapper is small (accumulate steps, build a
  `ContextGraphView` from them, delegate to core, cache the floor). It
  introduces no capability the core lacks; it is convenience state.

**One open verification for Graffiti (flagged, not blocking):** confirm
that Graffiti's `ContactGraph` holds nothing that a `{id, terms, cost}`
per step cannot regenerate (e.g. cached max-flow residues that are not a
pure function of the term graph). If it does, that state must be either
(a) recomputable by the core from the view, or (b) passed explicitly in
the view. If `ContactGraph` is a pure function of its committed
`(id, terms)` set -- which the term-tagging plan in `graffiti.md` §2
implies -- there is nothing to reconcile and both layers see the same
graph. Resolve this before Graffiti wires in; it does not affect the
core's design.

---

## Decision 2 — the graph input: core takes a view, Session takes steps

Both input shapes are kept, at their respective layers:

- **core** operators take a `ContextGraphView` (medium, tagged items,
  edges, floor). Graffiti supplies this directly.
- **`Session`** takes `Step[]` and builds the `ContextGraphView`
  internally (shared-term edges, weight = a function of shared-term
  count, floor derived). long-grass supplies steps.

The `Session`'s graph-building is the reference implementation of the
term-graph construction in the paper (`sec:model`): two steps are joined
when their term sets intersect, edge weight is `edgeWeight(|shared|)`
(default identity), the medium is adjacent to every step, and the floor
is the minimum positive edge weight. Graffiti's adapter must produce a
view consistent with this construction so that both layers agree on the
same graph for the same `(id, terms)` set.

---

## Decision 3 — `necessary` is the dominator test, not `seek`, and not max-flow ablation

This is the substantive correction. Both consumer docs wrote
`necessary(...) ≈ seek(...)` for v1, citing the paper's Necessity
Theorem. But the paper's own Separation Theorem proves reachable-but-%
redundant steps exist, so `necessary = seek` over-retains. The two
theorems are both true; the paper conflated two distinct quantities under
one word. The corrected hierarchy:

> **necessary ⊆ reachable.** Equality holds exactly on redundancy-free
> (tree-like) contexts. Inclusion is strict whenever a redundant route
> exists.

### Why they split (the diamond)

```
        u1
       /    \
goal--t      r      u1, u2 both share term t with the goal;
       \    /       both connect to a resolving step r.
        u2
```

`u1` and `u2` are both **reachable** (both touch `t`). But `r` is
reachable through `u1` alone, so **dropping `u2` changes nothing** -- `u2`
is reachable yet not load-bearing. Reachability says "on-topic";
necessity says "load-bearing"; in a diamond these differ. Conversation
and research histories are diamond-shaped (two turns naming the same
entity, both feeding a later conclusion is the *common* case), so
`necessary = seek` over-retains precisely where the tool exists to help.

### The right operator: dominators, not ablation

The exact necessity test is: *a reachable step is unnecessary iff every
goal-route through it has a parallel route that survives its deletion.*
This is a **dominator** relation on the goal-rooted reachability DAG:

> A step `u` is **necessary** iff it **dominates** at least one step that
> the goal needs -- i.e. `u` lies on *every* route from the goal to some
> load-bearing step. A step that is bypassed by a parallel route
> dominates nothing new and is unnecessary.

Dominators are computable in near-linear time (`O(V + E)` with
Lengauer--Tarjan, or a simple iterative dataflow for the small graphs
here). This gives **exact necessity at essentially `seek`'s cost** -- no
max-flow, no per-step ablation, no over-retention. It sits strictly
between the two rejected options:

| Option | Cost | Correctness |
|---|---|---|
| `nec := seek` (both docs, v1) | linear | over-retains redundant reachable |
| per-step max-flow ablation | `O(reach · maxflow)` | exact, expensive |
| **dominator test (adopt)** | near-linear | **exact, cheap** |

### v1 latitude (explicit)

If a v1 ships `nec := seek` before the dominator test lands, it MUST be
documented as *"necessity approximated by reachability; exact on
tree-like contexts, over-retains on redundant ones; the knapsack
mitigates by ranking redundant steps last."* It must not be documented as
`necessary = seek`, which the Separation Theorem falsifies. The knapsack
backstop is real (a redundant step buys little marginal resolution, so it
is admitted last and falls off the budget first), so `nec := seek` is a
tolerable *interim*, not a correct *definition*. The dominator test is
the v1.x target and the definition of record.

---

## Decision 4 — the paper is corrected, not softened

`purpose-propagation.tex` must restate its Necessity Theorem so it no
longer contradicts its Separation Theorem. The correction strengthens the
paper. Concretely:

1. **Necessity Theorem (`thm:necessity`)** is re-scoped to
   redundancy-free contexts: *on a context graph in which each reachable
   step lies on a unique goal-route (tree-like), necessity equals
   reachability.* Its proof already secretly assumes this (the phrase
   "lies on a shortest reaching path" smuggles uniqueness of route); the
   correction makes the hypothesis explicit.

2. **A new theorem, Necessity-by-Domination**, states the general case:
   *in an arbitrary context graph, a reachable step is necessary iff it
   dominates a load-bearing step in the goal-rooted reachability DAG;
   necessity is a strict subset of reachability exactly when a redundant
   route exists; the necessary set is computable in near-linear time by
   dominator analysis.*

3. **The Separation Theorem (`thm:separation`)** is unchanged; it now
   reads as the witness that motivates Necessity-by-Domination rather than
   a fact in tension with `thm:necessity`.

4. **The Free-Drop Theorem** widens: a *purposeless* step (unnecessary,
   whether unreachable OR reachable-but-dominated-away) may be dropped
   freely against the invariant. Currently it covers only the unreachable
   case; the diamond's redundant reachable step is equally free to drop.

This is a paper edit for the "we will get back to the paper later" pass;
it does not block the library, which builds to Decision 3 directly.

---

## What both consumers change

**`graffiti.md`:** replace "`nec` / necessary" language that implies
`= seek` with a pointer to Decision 3. The `necessary()` signature in §4
is unchanged; only its *semantics* sharpen (dominator test, not
reachability copy). Package name references become `@buhera/purpose`.

**`buhera-specifications.md`:** §4.3's note that `necessary()` is
"equivalent to seek() in v1" becomes "approximated by seek() in v1,
exact via dominators in v1.x (see reconciliation Decision 3)". Behavioural
guarantee §5.5 ("necessity subset of reachability") is *already correct*
and now becomes strict, not equality -- no change needed, it was right.

**`purpose-propagation.tex`:** the four edits of Decision 4, deferred to
the paper pass.

---

## Net effect

- One library, `@buhera/purpose`, layered core + Session. Both consumers
  satisfied, neither bent.
- `necessary` is the dominator test: exact, near-linear, no
  over-retention. `nec := seek` allowed only as a labelled interim.
- The paper's self-contradiction is resolved by strengthening: tree-case
  theorem + general dominator theorem + widened free-drop.
- Nothing in the deferred set (cascade, exact max-flow residue,
  persistence backend) is disturbed.
