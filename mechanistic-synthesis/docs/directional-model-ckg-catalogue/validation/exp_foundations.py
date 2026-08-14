"""
exp_foundations.py -- Part I of the paper.

E01  Floor                       Theorem (positive floor)
E02  Invariance                  Theorem (the minimum cut is the conserved invariant)
E03  Region-valuedness           Theorem (identity is a region, never a point)
E04  Monotone non-return         Theorem (monotone non-return)
E05  Self-blunting               Theorem (self-blunting)
E06  No perfect repeatable cut   Theorem (no perfect repeatable instrument)
"""

from __future__ import annotations

import random

from core import (
    MEDIUM,
    ContactGraph,
    Record,
    random_contact_graph,
    random_permutation,
    save,
    verdict,
)

SEED = 42


# ---------------------------------------------------------------------
# E01  The floor is positive and no separation crosses it
# ---------------------------------------------------------------------

def e01_floor(n_graphs: int = 60) -> dict:
    rng = random.Random(SEED)
    rows, violations = [], 0
    for gi in range(n_graphs):
        n = rng.randint(4, 12)
        floor = rng.choice([0.5, 1.0, 2.0])
        g = random_contact_graph(n, rng.uniform(0.2, 0.6), floor, rng)
        for v in g.items:
            s = g.sigma(v)
            if s < floor - 1e-9:
                violations += 1
            rows.append(
                {"graph": gi, "item": v, "sigma": s, "floor": floor,
                 "ratio": s / floor, "n_items": n}
            )
    ratios = [r["ratio"] for r in rows]
    return {
        "id": "E01",
        "claim": "Theorem (positive floor): sigma(v) >= beta > 0 for every item.",
        "n_graphs": n_graphs,
        "n_separations": len(rows),
        "violations": violations,
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "mean_ratio": sum(ratios) / len(ratios),
        "min_sigma": min(r["sigma"] for r in rows),
        "verdict": verdict(violations == 0),
        "rows": rows[:200],
    }


# ---------------------------------------------------------------------
# E02  Residue is invariant under relabelling; labels are not
# ---------------------------------------------------------------------

def e02_invariance(n_graphs: int = 60) -> dict:
    rng = random.Random(SEED + 1)
    max_disc, rows, label_moved = 0.0, [], 0
    for gi in range(n_graphs):
        n = rng.randint(4, 10)
        g = random_contact_graph(n, rng.uniform(0.2, 0.6), 1.0, rng)
        perm = random_permutation(g, rng)
        h = g.relabel(perm)
        for v in g.items:
            before = g.sigma(v)
            after = h.sigma(perm[v])
            disc = abs(before - after)
            max_disc = max(max_disc, disc)
            if perm[v] != v:
                label_moved += 1
            rows.append(
                {"graph": gi, "item": v, "image": perm[v],
                 "sigma_before": before, "sigma_after": after,
                 "discrepancy": disc}
            )
    return {
        "id": "E02",
        "claim": "Theorem (conserved invariant): sigma is preserved by every "
                 "weighted isomorphism; the label is not.",
        "n_graphs": n_graphs,
        "n_items_checked": len(rows),
        "max_discrepancy": max_disc,
        "labels_permuted": label_moved,
        "verdict": verdict(max_disc < 1e-9 and label_moved > 0),
        "rows": rows[:200],
    }


# ---------------------------------------------------------------------
# E03  Identity is a region: the minimising side is not a singleton
# ---------------------------------------------------------------------

def e03_region(k_values=(2, 3, 4, 5, 6, 7, 8)) -> dict:
    """Two dense clusters joined by one floor-weight contact, each thinly
    joined to the medium.  The minimum cut against the medium separates a
    whole cluster, so the minimiser is not {v}."""
    rows, non_singleton = [], 0
    for k in k_values:
        g = ContactGraph(vertices=[MEDIUM])
        A = [f"a{i}" for i in range(k)]
        B = [f"b{i}" for i in range(k)]
        W = 20.0          # dense intra-cluster
        floor = 1.0
        for i in range(k):
            for j in range(i + 1, k):
                g.add_edge(A[i], A[j], W)
                g.add_edge(B[i], B[j], W)
        g.add_edge(A[0], B[0], floor)      # single thin bridge
        for x in A + B:
            g.add_edge(x, MEDIUM, floor)   # thin to medium
        side = g.min_cut_side(A[0])
        size = len([s for s in side if s != MEDIUM])
        if size > 1:
            non_singleton += 1
        rows.append(
            {"k": k, "minimiser_side_size": size, "cluster_size": k,
             "sigma": g.sigma(A[0]), "is_region": size > 1}
        )
    return {
        "id": "E03",
        "claim": "Theorem (identity is a region): the minimising side is in "
                 "general not the singleton {v}.",
        "n_cases": len(rows),
        "non_singleton_cases": non_singleton,
        "verdict": verdict(non_singleton == len(rows)),
        "rows": rows,
    }


# ---------------------------------------------------------------------
# E04  The record is monotone; revisiting a configuration is not return
# ---------------------------------------------------------------------

def e04_record(n_walks: int = 40, steps: int = 30) -> dict:
    rng = random.Random(SEED + 3)
    rows, decrements, revisits_distinguished = [], 0, 0
    for wi in range(n_walks):
        g = random_contact_graph(6, 0.5, 1.0, rng)
        rec = Record()
        seen_configs = {}
        prev = 0
        for _ in range(steps):
            u, v, _w = rng.choice(g.edges)
            rec.commit(u, v)
            if rec.count <= prev:
                decrements += 1
            prev = rec.count
            config = frozenset(rec.committed)
            if config in seen_configs:
                # configuration recurs; the record must differ
                if seen_configs[config] != rec.count:
                    revisits_distinguished += 1
            seen_configs[config] = rec.count
        # demonstrate that "uncommit" advances rather than decrements
        before = rec.count
        u, v, _w = g.edges[0]
        after = rec.uncommit(u, v)
        rows.append(
            {"walk": wi, "final_record": rec.count,
             "uncommit_before": before, "uncommit_after": after,
             "uncommit_advanced": after > before}
        )
    all_advanced = all(r["uncommit_advanced"] for r in rows)
    return {
        "id": "E04",
        "claim": "Theorem (monotone non-return): the record never decreases; "
                 "un-committing is a further committing act; a recurring "
                 "configuration is a distinct state.",
        "n_walks": n_walks,
        "steps_per_walk": steps,
        "record_decrements": decrements,
        "recurrences_distinguished_by_record": revisits_distinguished,
        "uncommit_always_advanced": all_advanced,
        "verdict": verdict(decrements == 0 and all_advanced),
        "rows": rows[:50],
    }


# ---------------------------------------------------------------------
# E05  Self-blunting: capacity falls while the record rises
# ---------------------------------------------------------------------

def e05_blunting(n_instruments: int = 30) -> dict:
    rng = random.Random(SEED + 4)
    rows, monotone_ok, crossing_found = [], 0, 0
    for ii in range(n_instruments):
        g = random_contact_graph(rng.randint(6, 10), 0.6, 1.0, rng)
        u = rng.choice(g.items)
        rec = Record()
        incident = [e for e in g.weights if u in e]
        rng.shuffle(incident)
        traj = []
        cap0 = g.capacity(u, rec.committed)
        ok = True
        crossed = False
        for e in incident:
            a, b = tuple(e)
            cap_before = g.capacity(u, rec.committed)
            rec_before = rec.count
            rec.commit(a, b)
            cap_after = g.capacity(u, rec.committed)
            if not (cap_after < cap_before and rec.count > rec_before):
                ok = False
            traj.append({"record": rec.count, "capacity": cap_after})
            if rec.count >= cap_after and not crossed:
                crossed = True
        monotone_ok += int(ok)
        crossing_found += int(crossed)
        rows.append(
            {"instrument": ii, "vertex": u, "initial_capacity": cap0,
             "final_capacity": g.capacity(u, rec.committed),
             "final_record": rec.count, "strictly_blunted": ok,
             "curves_cross": crossed, "trajectory": traj}
        )
    return {
        "id": "E05",
        "claim": "Theorem (self-blunting): every contact operation strictly "
                 "reduces the instrument's uncommitted capacity and strictly "
                 "increases its record.",
        "n_instruments": n_instruments,
        "instruments_strictly_blunted": monotone_ok,
        "instruments_with_curve_crossing": crossing_found,
        "verdict": verdict(monotone_ok == n_instruments),
        "rows": rows[:20],
    }


# ---------------------------------------------------------------------
# E06  No perfect repeatable traceless instrument
# ---------------------------------------------------------------------

def e06_no_perfect(n_trials: int = 200) -> dict:
    """Search for a contact operation that is traceless (residue 0) and
    leaves the instrument unchanged (record unchanged).  Both are separately
    forbidden, so the search must come up empty."""
    rng = random.Random(SEED + 5)
    traceless_found, unchanged_found, both_found = 0, 0, 0
    rows = []
    for t in range(n_trials):
        floor = rng.choice([0.25, 0.5, 1.0, 2.0])
        g = random_contact_graph(rng.randint(4, 9), 0.5, floor, rng)
        rec = Record()
        u, v, w = rng.choice(g.edges)
        cap_before = g.capacity(u, rec.committed)
        rec_before = rec.count
        rec.commit(u, v)
        traceless = w <= 1e-12
        unchanged = (rec.count == rec_before) and (
            g.capacity(u, rec.committed) == cap_before
        )
        traceless_found += int(traceless)
        unchanged_found += int(unchanged)
        both_found += int(traceless and unchanged)
        rows.append(
            {"trial": t, "edge_weight": w, "floor": floor,
             "traceless": traceless, "instrument_unchanged": unchanged}
        )
    # The degenerate object: what it would require
    return {
        "id": "E06",
        "claim": "Theorem (no perfect repeatable instrument): no operation is "
                 "both traceless (requires beta = 0) and leaves the instrument "
                 "unchanged (requires a static record).",
        "n_trials": n_trials,
        "traceless_operations_found": traceless_found,
        "instrument_unchanged_found": unchanged_found,
        "both_properties_found": both_found,
        "note": "A stateless determiner asserts both properties of every "
                "application; the search confirms neither is realisable.",
        "verdict": verdict(both_found == 0 and traceless_found == 0
                           and unchanged_found == 0),
        "rows": rows[:50],
    }


# ---------------------------------------------------------------------

def run_all() -> dict:
    experiments = [
        e01_floor(),
        e02_invariance(),
        e03_region(),
        e04_record(),
        e05_blunting(),
        e06_no_perfect(),
    ]
    for e in experiments:
        save(e["id"].lower(), e)
    summary = {
        "part": "I -- Foundations",
        "experiments": [
            {"id": e["id"], "claim": e["claim"], "verdict": e["verdict"]}
            for e in experiments
        ],
        "pass_rate": f"{sum(e['verdict'] == 'PASS' for e in experiments)}/"
                     f"{len(experiments)}",
    }
    save("part1_foundations_summary", summary)
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2))
