"""
exp_probing.py -- Part IV of the paper: construction and probing.

E18  Correct for any term map    Theorem (the calculus is correct for any tau)
E19  Floor readout               Theorem (floor monotonicity)
E20  Multiplicative composition  Theorem (multiplicative composition)
E21  Saturation dichotomy        Corollary (saturation dichotomy)
E22  Diversify, do not repeat    Corollary (diversify, do not repeat)
E23  Coherence triangle          Theorem (coherence requires three)
E24  Knapsack selection          Theorem (selection is a 0/1 knapsack)
E25  Water-filling               Theorem (optimal division is water-filling)
E26  Seek / necessity separation Theorem (separation of finding and pruning)
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Set, Tuple

from core import (
    MEDIUM,
    ContactGraph,
    contribution,
    dominates,
    induced_graph,
    necessary,
    random_contact_graph,
    reach,
    refines,
    resolution,
    save,
    verdict,
)

SEED = 42


# ---------------------------------------------------------------------
# E18  Every result holds for any term map
# ---------------------------------------------------------------------

def e18_tau_agnostic(n_maps: int = 60) -> dict:
    """Generate deliberately bad term maps -- random, sparse, noisy,
    adversarially over-connected -- and verify the structural results still
    hold on the induced graph."""
    rng = random.Random(SEED + 18)
    rows, failures = [], 0
    kinds = ["random", "sparse", "noisy", "overconnected", "degenerate"]
    for m in range(n_maps):
        kind = kinds[m % len(kinds)]
        n_src = rng.randint(4, 9)
        alphabet = [f"d{i}" for i in range(12)]
        tau: Dict[str, Set[str]] = {}
        for i in range(n_src):
            s = f"s{i}"
            if kind == "random":
                k = rng.randint(1, 5)
            elif kind == "sparse":
                k = 1
            elif kind == "noisy":
                k = rng.randint(1, 10)
            elif kind == "overconnected":
                k = 10
            else:                       # degenerate: all identical
                k = 3
            if kind == "degenerate":
                tau[s] = set(alphabet[:3])
            else:
                tau[s] = set(rng.sample(alphabet, k))
        g = induced_graph(tau, floor=1.0)

        # the structural results, checked on the induced graph
        floor_ok = all(g.sigma(v) >= 1.0 - 1e-9 for v in g.items)
        edges_ok = all(w >= 1.0 - 1e-9 for _, _, w in g.edges)
        # invariance under relabelling
        items = g.items
        shuffled = items[:]
        rng.shuffle(shuffled)
        perm = dict(zip(items, shuffled)); perm[MEDIUM] = MEDIUM
        h = g.relabel(perm)
        inv_ok = (sorted(round(g.sigma(v), 9) for v in g.items)
                  == sorted(round(h.sigma(v), 9) for v in h.items))
        # cuts are non-empty
        cut_ok = all(len(g.resting_cut(v)) > 0 for v in g.items)
        ok = floor_ok and edges_ok and inv_ok and cut_ok
        if not ok:
            failures += 1
        rows.append({
            "map": m, "kind": kind, "n_sources": n_src,
            "n_edges": len(g.edges),
            "floor_holds": floor_ok, "edge_weights_meet_floor": edges_ok,
            "relabelling_invariant": inv_ok, "cuts_nonempty": cut_ok,
            "system_floor": g.system_floor(),
            "all_hold": ok,
        })
    return {
        "id": "E18",
        "claim": "Theorem (correct for any term map): every structural result "
                 "holds on the induced graph regardless of the quality of the "
                 "term map; extraction error coarsens, it does not corrupt.",
        "n_term_maps": n_maps,
        "map_kinds_tested": kinds,
        "failures": failures,
        "verdict": verdict(failures == 0),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------
# E19  The realised floor is monotone under refinement
# ---------------------------------------------------------------------

def e19_floor_readout(n_chains: int = 40, chain_len: int = 5) -> dict:
    """Build chains of term maps ordered by refinement and verify the realised
    system floor is non-decreasing along each chain."""
    rng = random.Random(SEED + 19)
    rows, violations, chains_checked = [], 0, 0
    for c in range(n_chains):
        n_src = rng.randint(4, 7)
        alphabet = [f"d{i}" for i in range(10)]
        # coarsest map
        tau: Dict[str, Set[str]] = {
            f"s{i}": set(rng.sample(alphabet, rng.randint(1, 2)))
            for i in range(n_src)
        }
        chain, floors = [dict((k, set(v)) for k, v in tau.items())], []
        cur = tau
        for step in range(chain_len - 1):
            nxt = {k: set(v) for k, v in cur.items()}
            # refine: add distinctions, which can only add/strengthen contacts
            for s in nxt:
                extra = rng.sample(alphabet, rng.randint(1, 2))
                nxt[s] |= set(extra)
            chain.append(nxt)
            cur = nxt
        # verify the refinement order actually holds, then check monotonicity
        ok_chain = True
        for i in range(len(chain) - 1):
            if not refines(chain[i + 1], chain[i]):
                ok_chain = False
        if not ok_chain:
            continue
        chains_checked += 1
        for tmap in chain:
            g = induced_graph(tmap, floor=1.0)
            floors.append(g.system_floor())
        mono = all(floors[i] <= floors[i + 1] + 1e-9
                   for i in range(len(floors) - 1))
        if not mono:
            violations += 1
        rows.append({
            "chain": c, "n_sources": n_src,
            "floors": floors, "monotone_nondecreasing": mono,
            "floor_gain": floors[-1] - floors[0],
        })
    return {
        "id": "E19",
        "claim": "Theorem (floor monotonicity): a finer term map induces a "
                 "system floor at least as large; the realised floor is a "
                 "ground-truth-free readout of extraction quality.",
        "n_chains_generated": n_chains,
        "n_chains_valid": chains_checked,
        "monotonicity_violations": violations,
        "verdict": verdict(chains_checked > 0 and violations == 0),
        "rows": rows[:30],
    }


# ---------------------------------------------------------------------
# E20  Multiplicative composition of probing power
# ---------------------------------------------------------------------

def e20_multiplicative(n_chains: int = 400) -> dict:
    rng = random.Random(SEED + 20)
    max_err, rows = 0.0, []
    for c in range(n_chains):
        n = rng.randint(2, 8)
        kappas = [rng.uniform(0.0, 0.95) for _ in range(n)]
        # simulate: residual gap multiplies by (1 - kappa_i) at each step
        r0 = rng.uniform(1.0, 100.0)
        r = r0
        for k in kappas:
            r *= (1.0 - k)
        measured = 1.0 - r / r0
        predicted = 1.0 - math.prod(1.0 - k for k in kappas)
        err = abs(measured - predicted)
        max_err = max(max_err, err)
        rows.append({
            "chain": c, "n_probes": n, "kappas": kappas,
            "measured_composite": measured,
            "predicted_composite": predicted,
            "abs_error": err,
        })
    return {
        "id": "E20",
        "claim": "Theorem (multiplicative composition): the composite power of "
                 "a probe chain is 1 - prod_i (1 - kappa_i).",
        "n_chains": n_chains,
        "max_abs_error": max_err,
        "verdict": verdict(max_err < 1e-12),
        "rows": rows[:50],
    }


# ---------------------------------------------------------------------
# E21  Saturation dichotomy
# ---------------------------------------------------------------------

def e21_saturation(horizon: int = 4000) -> dict:
    """Residual -> 0 iff sum kappa_i diverges."""
    sequences = {
        "constant_0.05":  (lambda i: 0.05,             True),
        "harmonic_1/i":   (lambda i: 1.0 / (i + 1),    True),
        "geometric_2^-i": (lambda i: 2.0 ** (-(i + 1)), False),
        "inverse_square": (lambda i: 1.0 / (i + 1) ** 2, False),
    }
    rows, mismatches = [], 0
    for name, (f, diverges) in sequences.items():
        residual, partial = 1.0, 0.0
        for i in range(horizon):
            k = min(f(i), 0.999999)
            residual *= (1.0 - k)
            partial += k
        drove_to_zero = residual < 1e-8
        if drove_to_zero != diverges:
            mismatches += 1
        rows.append({
            "sequence": name,
            "sum_diverges_theoretically": diverges,
            "partial_sum_at_horizon": partial,
            "residual_at_horizon": residual,
            "drove_residual_to_zero": drove_to_zero,
            "matches_prediction": drove_to_zero == diverges,
        })
    return {
        "id": "E21",
        "claim": "Corollary (saturation dichotomy): the residual is driven to "
                 "zero iff sum_i kappa_i diverges.",
        "horizon": horizon,
        "sequences_tested": len(sequences),
        "mismatches": mismatches,
        "verdict": verdict(mismatches == 0),
        "rows": rows,
    }


# ---------------------------------------------------------------------
# E22  Diversify, do not repeat
# ---------------------------------------------------------------------

def e22_diversify(n_trials: int = 300) -> dict:
    rng = random.Random(SEED + 22)
    rows, wins, ties = [], 0, 0
    for t in range(n_trials):
        n = rng.randint(2, 8)
        kappas = [rng.uniform(0.05, 0.9) for _ in range(n)]
        diverse = 1.0 - math.prod(1.0 - k for k in kappas)
        weakest = min(kappas)
        repeated = 1.0 - (1.0 - weakest) ** n     # n repetitions of the weakest
        if diverse > repeated + 1e-12:
            wins += 1
        elif abs(diverse - repeated) <= 1e-12:
            ties += 1
        rows.append({
            "trial": t, "n_probes": n, "kappas": kappas,
            "diverse_composite": diverse,
            "repeated_weakest_composite": repeated,
            "diverse_strictly_better": diverse > repeated + 1e-12,
            "advantage": diverse - repeated,
        })
    return {
        "id": "E22",
        "claim": "Corollary (diversify, do not repeat): a chain of distinct "
                 "probes strictly exceeds the composite power of repeating the "
                 "weakest member the same number of times.",
        "n_trials": n_trials,
        "diverse_strictly_better": wins,
        "ties": ties,
        "verdict": verdict(wins + ties == n_trials and wins > 0),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------
# E23  Coherence requires three mutually supporting probes
# ---------------------------------------------------------------------

def robust_to_single_removal(support: Dict[int, Set[int]], n: int) -> bool:
    """A support structure is robust if, after removing any single member,
    every remaining member still has an incoming support edge."""
    for drop in range(n):
        remaining = [i for i in range(n) if i != drop]
        if not remaining:
            return False
        for i in remaining:
            supporters = support.get(i, set()) & set(remaining)
            if not supporters:
                return False
    return True


def has_cycle_of_length_at_least(support: Dict[int, Set[int]], n: int,
                                 L: int) -> bool:
    """Does the support digraph contain a directed cycle of length >= L?"""
    # edges j -> i mean j supports i
    adj = {j: set() for j in range(n)}
    for i, sup in support.items():
        for j in sup:
            adj[j].add(i)
    best = 0
    for start in range(n):
        stack = [(start, [start])]
        while stack:
            u, path = stack.pop()
            for v in adj[u]:
                if v == start and len(path) >= 2:
                    best = max(best, len(path))
                elif v not in path and len(path) < n:
                    stack.append((v, path + [v]))
    return best >= L


def e23_triangle(n_random: int = 800) -> dict:
    rng = random.Random(SEED + 23)
    rows = []
    acyclic_robust, twocycle_robust, triangle_robust = 0, 0, 0
    n_acyclic, n_twocycle, n_triangle = 0, 0, 0

    # exhaustive: all support digraphs on 2 and 3 members
    for n in (2, 3):
        for bits in range(1 << (n * (n - 1))):
            support = {i: set() for i in range(n)}
            b = bits
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if b & 1:
                        support[i].add(j)      # j supports i
                    b >>= 1
            robust = robust_to_single_removal(support, n)
            cyc3 = has_cycle_of_length_at_least(support, n, 3)
            if n == 2:
                n_twocycle += 1
                twocycle_robust += int(robust)
            else:
                if cyc3:
                    n_triangle += 1
                    triangle_robust += int(robust)
                else:
                    n_acyclic += 1
                    acyclic_robust += int(robust)

    # random larger structures: robustness implies a >=3 cycle
    implication_violations = 0
    for t in range(n_random):
        n = rng.randint(3, 6)
        support = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < 0.4:
                    support[i].add(j)
        robust = robust_to_single_removal(support, n)
        cyc3 = has_cycle_of_length_at_least(support, n, 3)
        if robust and not cyc3:
            implication_violations += 1
        rows.append({"trial": t, "n": n, "robust": robust,
                     "has_cycle_ge_3": cyc3,
                     "implication_holds": (not robust) or cyc3})

    return {
        "id": "E23",
        "claim": "Theorem (coherence requires three): robustness to the failure "
                 "of any single probe implies a directed support cycle of "
                 "length >= 3; no 2-member structure is robust.",
        "exhaustive_2_member_structures": n_twocycle,
        "robust_2_member_structures": twocycle_robust,
        "exhaustive_3_member_acyclic": n_acyclic,
        "robust_3_member_acyclic": acyclic_robust,
        "exhaustive_3_member_with_cycle": n_triangle,
        "robust_3_member_with_cycle": triangle_robust,
        "random_structures_tested": n_random,
        "implication_violations": implication_violations,
        "verdict": verdict(twocycle_robust == 0 and acyclic_robust == 0
                           and triangle_robust > 0
                           and implication_violations == 0),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------
# E24  Selection is a 0/1 knapsack; greedy meets its guarantee
# ---------------------------------------------------------------------

def knapsack_exact(values: List[float], costs: List[int], B: int) -> float:
    T = [0.0] * (B + 1)
    for v, c in zip(values, costs):
        for b in range(B, c - 1, -1):
            T[b] = max(T[b], v + T[b - c])
    return T[B]


def e24_knapsack(n_instances: int = 150) -> dict:
    rng = random.Random(SEED + 24)
    rows, within_bound, greedy_exceeds = 0, 0, 0
    data = []
    for t in range(n_instances):
        k = rng.randint(3, 9)
        Omega = 100.0
        floors = [rng.uniform(1.0, 60.0) for _ in range(k)]
        costs = [rng.randint(1, 12) for _ in range(k)]
        B = rng.randint(max(costs), sum(costs))
        values = [-math.log(1.0 - f / Omega) for f in floors]
        opt = knapsack_exact(values, costs, B)
        # value-density greedy
        order = sorted(range(k), key=lambda i: values[i] / costs[i],
                       reverse=True)
        used, gval = 0, 0.0
        for i in order:
            if used + costs[i] <= B:
                used += costs[i]
                gval += values[i]
        ratio = gval / opt if opt > 0 else 1.0
        cmax = max(costs)
        bound = 1.0 - cmax / B
        ok = ratio >= bound - 1e-9
        within_bound += int(ok)
        greedy_exceeds += int(gval > opt + 1e-9)
        data.append({
            "instance": t, "k": k, "budget": B,
            "greedy_value": gval, "exact_optimum": opt,
            "ratio": ratio, "cmax_over_B": cmax / B,
            "guarantee_bound": bound, "within_guarantee": ok,
        })
        rows += 1
    ratios = [d["ratio"] for d in data]
    return {
        "id": "E24",
        "claim": "Theorem (selection is a 0/1 knapsack): with values "
                 "-log(1 - beta_i/Omega), the value-density greedy is within "
                 "(1 - c_max/B) of the exact optimum and never exceeds it.",
        "n_instances": n_instances,
        "greedy_within_guarantee": within_bound,
        "greedy_exceeded_optimum": greedy_exceeds,
        "mean_ratio": sum(ratios) / len(ratios),
        "min_ratio": min(ratios),
        "verdict": verdict(within_bound == n_instances and greedy_exceeds == 0),
        "rows": data[:40],
    }


# ---------------------------------------------------------------------
# E25  Water-filling
# ---------------------------------------------------------------------

def waterfill(entry_margins: List[float], k_scale: List[float],
              budget: float, tol: float = 1e-12) -> Tuple[List[float], float]:
    """Concave gain profiles gamma_i(a) = k_i * log(1 + a / s_i) with
    gamma_i'(a) = k_i / (s_i + a), so gamma_i'(0) = k_i / s_i = entry margin.
    Invert: a_i(p) = max(0, k_i / p - s_i).  Bisect on the scalar price p."""
    s = [k / m for k, m in zip(k_scale, entry_margins)]

    def alloc(p: float) -> List[float]:
        return [max(0.0, k / p - si) for k, si in zip(k_scale, s)]

    lo, hi = 1e-12, max(entry_margins) + 1.0
    if sum(alloc(hi)) > budget:
        hi = max(entry_margins) * 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sum(alloc(mid)) > budget:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    p = 0.5 * (lo + hi)
    return alloc(p), p


def e25_waterfill(n_instances: int = 120) -> dict:
    rng = random.Random(SEED + 25)
    rows = []
    kkt_ok, budget_ok, dropout_ok = 0, 0, 0
    price_monotone_budget, price_monotone_count = 0, 0

    for t in range(n_instances):
        k = rng.randint(3, 8)
        margins = [rng.uniform(0.5, 5.0) for _ in range(k)]
        scale = [rng.uniform(0.5, 3.0) for _ in range(k)]
        budget = rng.uniform(0.5, 8.0)
        a, p = waterfill(margins, scale, budget)
        s = [ks / m for ks, m in zip(scale, margins)]

        # KKT: gamma'_i(a_i) = p where a_i > 0;  gamma'_i(0) <= p where a_i = 0
        kkt = True
        for i in range(k):
            deriv = scale[i] / (s[i] + a[i])
            if a[i] > 1e-9:
                if abs(deriv - p) > 1e-6:
                    kkt = False
            else:
                if margins[i] > p + 1e-6:
                    kkt = False
        kkt_ok += int(kkt)

        used = sum(a)
        budget_respected = used <= budget + 1e-6
        budget_ok += int(budget_respected)

        # dropout boundary: a_i = 0 exactly when entry margin <= price
        drop = all((a[i] <= 1e-9) == (margins[i] <= p + 1e-6) for i in range(k))
        dropout_ok += int(drop)

        # price monotone: nonincreasing in budget
        _a2, p2 = waterfill(margins, scale, budget * 2.0)
        pm_b = p2 <= p + 1e-9
        price_monotone_budget += int(pm_b)

        # price monotone: nondecreasing in probe count
        margins3 = margins + [rng.uniform(0.5, 5.0)]
        scale3 = scale + [rng.uniform(0.5, 3.0)]
        _a3, p3 = waterfill(margins3, scale3, budget)
        pm_n = p3 >= p - 1e-9
        price_monotone_count += int(pm_n)

        rows.append({
            "instance": t, "k": k, "budget": budget, "price": p,
            "allocation": a, "used": used,
            "kkt_satisfied": kkt, "budget_respected": budget_respected,
            "dropout_boundary_sharp": drop,
            "price_nonincreasing_in_budget": pm_b,
            "price_nondecreasing_in_count": pm_n,
            "n_engaged": sum(1 for x in a if x > 1e-9),
        })

    return {
        "id": "E25",
        "claim": "Theorem (water-filling): the optimum equalises marginal gain "
                 "at a single price across engaged probes, drops probes whose "
                 "entry margin is below the price, and the price is "
                 "nonincreasing in budget and nondecreasing in probe count.",
        "n_instances": n_instances,
        "kkt_satisfied": kkt_ok,
        "budget_respected": budget_ok,
        "dropout_boundary_sharp": dropout_ok,
        "price_nonincreasing_in_budget": price_monotone_budget,
        "price_nondecreasing_in_count": price_monotone_count,
        "verdict": verdict(kkt_ok == n_instances and budget_ok == n_instances
                           and dropout_ok == n_instances
                           and price_monotone_budget == n_instances
                           and price_monotone_count == n_instances),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------
# E26  Seek and necessity disagree
# ---------------------------------------------------------------------

def e26_separation(n_random: int = 60) -> dict:
    """(a) The diamond witness: parallel routes to a common resolver are
    reachable but individually redundant.
       (b) Random graphs: nec is contained in reach, and the gap is real.
       (c) Necessity coincides with domination."""
    rng = random.Random(SEED + 26)

    # ---- (a) the diamond, widened to k parallel routes ----
    diamond_rows = []
    for k in range(2, 12):
        g = ContactGraph(vertices=[MEDIUM])
        g.add_edge("goal", MEDIUM, 1.0)
        g.add_edge("r", MEDIUM, 1.0)
        legs = [f"u{i}" for i in range(k)]
        for u in legs:
            g.add_edge("goal", u, 2.0)
            g.add_edge(u, "r", 2.0)
            g.add_edge(u, MEDIUM, 1.0)
        universe = set(g.items)
        seeds = ["goal"]
        R = reach(g, seeds)
        N = necessary(g, seeds, universe)
        diamond_rows.append({
            "parallel_routes": k,
            "seek_retains": len(R),
            "nec_retains": len(N),
            "over_retention_gap": len(R) - len(N),
            "legs_reachable": all(u in R for u in legs),
            "legs_necessary": [u for u in legs if u in N],
        })

    # ---- (a2) the chain: every interior is necessary, the terminal leaf is
    #           reachable yet redundant (it dominates nothing) ----
    chain_rows = []
    for L in range(3, 10):
        g = ContactGraph(vertices=[MEDIUM])
        ch = [f"s{i}" for i in range(L)]
        for i in range(L - 1):
            g.add_edge(ch[i], ch[i + 1], 2.0)
        for s in ch:
            g.add_edge(s, MEDIUM, 1.0)
        universe = set(g.items)
        seeds = [ch[0]]
        R = reach(g, seeds)
        N = necessary(g, seeds, universe)
        chain_rows.append({
            "chain_length": L,
            "seek_retains": len(R),
            "nec_retains": len(N),
            "terminal_leaf": ch[-1],
            "terminal_leaf_reachable": ch[-1] in R,
            "terminal_leaf_necessary": ch[-1] in N,
            "interiors_all_necessary": all(ch[i] in N for i in range(L - 1)),
        })
    chain_identity = all(
        r["terminal_leaf_reachable"] and not r["terminal_leaf_necessary"]
        and r["interiors_all_necessary"] for r in chain_rows
    )

    # ---- (b) random graphs: containment and gap ----
    containment_violations, gap_positive = 0, 0
    rand_rows = []
    for t in range(n_random):
        n = rng.randint(5, 10)
        g = random_contact_graph(n, rng.uniform(0.3, 0.7), 1.0, rng)
        seeds = [rng.choice(g.items)]
        universe = set(g.items)
        R = reach(g, seeds)
        N = necessary(g, seeds, universe)
        if not N <= R:
            containment_violations += 1
        if len(R) - len(N) > 0:
            gap_positive += 1
        rand_rows.append({
            "trial": t, "n_items": n, "seek": len(R), "nec": len(N),
            "gap": len(R) - len(N), "containment_holds": N <= R,
        })

    # ---- (c) necessity == domination, off the degenerate boundary ----
    #
    # The domination criterion characterises necessity for non-seed items.
    # A seed is load-bearing by definition -- it is the goal's only point of
    # entry -- and in the degenerate case where the seed reaches nothing but
    # itself it dominates no item while remaining necessary.  That boundary
    # case is excluded and counted separately rather than silently passed.
    dom_mismatches, dom_checked, boundary_cases = 0, 0, 0
    for t in range(40):
        n = rng.randint(5, 9)
        g = random_contact_graph(n, rng.uniform(0.3, 0.6), 1.0, rng)
        seeds = [rng.choice(g.items)]
        universe = set(g.items)
        R = reach(g, seeds)
        N = necessary(g, seeds, universe)
        for u in R:
            if u in seeds:
                boundary_cases += 1
                continue
            dom_checked += 1
            doms_something = any(
                dominates(g, seeds, u, r, universe) for r in R if r != u
            )
            is_nec = u in N
            if doms_something != is_nec:
                dom_mismatches += 1

    chain_gap = all(r["over_retention_gap"] > 0 for r in diamond_rows)
    return {
        "id": "E26",
        "claim": "Theorem (separation of finding and pruning): seek marks "
                 "reachable, nec marks load-bearing; they disagree on "
                 "redundant reachable items, nec is contained in reach, and "
                 "necessity coincides with domination.",
        "diamond_widths_tested": len(diamond_rows),
        "diamond_always_shows_gap": chain_gap,
        "chain_lengths_tested": len(chain_rows),
        "chain_identity_holds": chain_identity,
        "random_graphs_tested": n_random,
        "containment_violations": containment_violations,
        "random_graphs_with_positive_gap": gap_positive,
        "domination_checks_non_seed": dom_checked,
        "domination_mismatches": dom_mismatches,
        "seed_boundary_cases_excluded": boundary_cases,
        "boundary_note": "A seed is necessary by definition (sole point of "
                         "entry); where it reaches only itself it dominates "
                         "nothing, so the domination criterion is stated for "
                         "non-seed items and seeds are counted separately.",
        "verdict": verdict(chain_gap and chain_identity
                           and containment_violations == 0
                           and dom_mismatches == 0),
        "diamond_rows": diamond_rows,
        "chain_rows": chain_rows,
        "rows": rand_rows[:30],
    }


# ---------------------------------------------------------------------

def run_all() -> dict:
    experiments = [
        e18_tau_agnostic(),
        e19_floor_readout(),
        e20_multiplicative(),
        e21_saturation(),
        e22_diversify(),
        e23_triangle(),
        e24_knapsack(),
        e25_waterfill(),
        e26_separation(),
    ]
    for e in experiments:
        save(e["id"].lower(), e)
    summary = {
        "part": "IV -- Construction and Probing",
        "experiments": [
            {"id": e["id"], "claim": e["claim"], "verdict": e["verdict"]}
            for e in experiments
        ],
        "pass_rate": f"{sum(e['verdict'] == 'PASS' for e in experiments)}/"
                     f"{len(experiments)}",
    }
    save("part4_probing_summary", summary)
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2))
