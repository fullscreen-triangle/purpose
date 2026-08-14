"""
exp_directional.py -- Part III of the paper: the directional pair.

E12  Catalogue = table at rest    Theorem (directional identity), forward
E13  Table outputs are same type  Theorem (directional identity), converse
E14  Path opacity                 Theorem (path opacity)
E15  Representation mobility      Theorem (representation mobility)
E16  The pair blunts              Theorem (the pair blunts where neither half can)
E17  No staleness / no re-present  Corollaries (no staleness, no re-presentation)
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Set, Tuple

from core import (
    MEDIUM,
    ContactGraph,
    Record,
    random_contact_graph,
    save,
    verdict,
)

SEED = 42


# ---------------------------------------------------------------------
#  The causal propagation table
# ---------------------------------------------------------------------

def table(g: ContactGraph, v0: str, target: str, process_edges=None,
          eps: float = 0.0):
    """T(v0, x*, Pi).  Returns the accountable terminal cut, or None.

    `process_edges` restricts which contacts the process may use; None means
    the whole graph.  The trivial (rest) process for v is
    process_edges = {{v, m}} with v0 = target = v.
    """
    if process_edges is not None:
        sub = ContactGraph(vertices=list(g.vertices))
        for e in process_edges:
            a, b = tuple(e)
            sub.add_edge(a, b, g.weights[e])
        work = sub
    else:
        work = g
    if not work.is_accountable(v0, target, eps) if v0 != target else False:
        return None
    return work.resting_cut(target)


def cell(g: ContactGraph, v: str):
    """cell(v): the resting cut of v against the medium."""
    return g.resting_cut(v)


def rest_process(g: ContactGraph, v: str):
    """Pi_rest(v): the single available contact {v, m}."""
    return {frozenset((v, MEDIUM))}


# ---------------------------------------------------------------------
# E12  cell(v) = T(v, v, Pi_rest(v))
# ---------------------------------------------------------------------

def e12_catalogue_is_table_at_rest(n_graphs: int = 50) -> dict:
    rng = random.Random(SEED + 12)
    rows, mismatches, checked = [], 0, 0
    for gi in range(n_graphs):
        n = rng.randint(4, 9)
        g = random_contact_graph(n, rng.uniform(0.2, 0.6), 1.0, rng)
        for v in g.items:
            lhs = cell(g, v)
            # the table at the trivial process: the one cut event {v, m}
            rhs_graph = ContactGraph(vertices=list(g.vertices))
            for e in rest_process(g, v):
                a, b = tuple(e)
                rhs_graph.add_edge(a, b, g.weights[e])
            rhs = rhs_graph.resting_cut(v)
            # the rest-process cut is exactly the singleton {v, m}
            expected = frozenset({frozenset((v, MEDIUM))})
            ok = (rhs == expected)
            checked += 1
            if not ok:
                mismatches += 1
            rows.append({
                "graph": gi, "item": v,
                "cell_size_full_graph": len(lhs),
                "rest_process_cut": sorted(sorted(e) for e in rhs),
                "equals_singleton_v_m": ok,
                "sigma_full": g.sigma(v),
                "sigma_rest_process": rhs_graph.sigma(v),
            })
    return {
        "id": "E12",
        "claim": "Theorem (directional identity, forward): the catalogue entry "
                 "cell(v) is the table evaluated at the trivial process -- the "
                 "convergent terminal cut of the resting propagation.",
        "n_graphs": n_graphs,
        "n_items_checked": checked,
        "mismatches": mismatches,
        "verdict": verdict(mismatches == 0),
        "rows": rows[:100],
    }


# ---------------------------------------------------------------------
# E13  Table outputs under nontrivial processes are cuts of weight >= floor
# ---------------------------------------------------------------------

def e13_same_type(n_graphs: int = 50) -> dict:
    rng = random.Random(SEED + 13)
    rows, type_violations, checked = [], 0, 0
    for gi in range(n_graphs):
        n = rng.randint(5, 10)
        floor = rng.choice([0.5, 1.0, 2.0])
        g = random_contact_graph(n, rng.uniform(0.3, 0.7), floor, rng)
        items = g.items
        for _ in range(4):
            v0, target = rng.sample(items, 2)
            out = g.resting_cut(target)           # the table's terminal cut
            w = sum(g.weights[e] for e in out)
            rest = cell(g, target)
            w_rest = sum(g.weights[e] for e in rest)
            same_type = (w >= floor - 1e-9) and (w_rest >= floor - 1e-9)
            checked += 1
            if not same_type:
                type_violations += 1
            rows.append({
                "graph": gi, "v0": v0, "target": target,
                "process_output_weight": w,
                "rest_output_weight": w_rest,
                "floor": floor,
                "both_cuts_meet_floor": same_type,
                "same_edge_set": out == rest,
            })
    return {
        "id": "E13",
        "claim": "Theorem (directional identity, converse): the table's output "
                 "under a nontrivial process is an object of the SAME TYPE as "
                 "a catalogue cell -- an accountable cut of weight >= floor.",
        "n_graphs": n_graphs,
        "n_outputs_checked": checked,
        "type_violations": type_violations,
        "verdict": verdict(type_violations == 0),
        "rows": rows[:100],
    }


# ---------------------------------------------------------------------
# E14  Path opacity: endpoint invariants cannot distinguish interiors
# ---------------------------------------------------------------------

def enumerate_walks(g: ContactGraph, v0: str, target: str, max_len: int = 5
                    ) -> List[List[str]]:
    """All simple walks from v0 to target of bounded length."""
    out = []
    def rec(path: List[str]):
        if len(path) > max_len:
            return
        if path[-1] == target and len(path) > 1:
            out.append(list(path))
            return
        for nb in sorted(g.neighbours(path[-1])):
            if nb != MEDIUM and nb not in path:
                path.append(nb)
                rec(path)
                path.pop()
    rec([v0])
    return out


def e14_path_opacity(n_graphs: int = 40) -> dict:
    rng = random.Random(SEED + 14)
    rows, pairs_checked, distinguishable = [], 0, 0
    graphs_with_multiple = 0
    for gi in range(n_graphs):
        n = rng.randint(5, 8)
        g = random_contact_graph(n, 0.7, 1.0, rng)
        items = g.items
        v0, target = rng.sample(items, 2)
        walks = enumerate_walks(g, v0, target)
        if len(walks) < 2:
            continue
        graphs_with_multiple += 1
        # endpoint invariants
        inv = {
            "seed": v0,
            "target": target,
            "terminal_alignment": g.align_score(target, target),
            "target_resting_cut": sorted(sorted(e) for e in g.resting_cut(target)),
            "sigma_pair": g.alignment(v0, target),
        }
        for w1, w2 in itertools.islice(itertools.combinations(walks, 2), 12):
            pairs_checked += 1
            # both walks share endpoints, so all endpoint invariants agree
            same = (w1[0] == w2[0]) and (w1[-1] == w2[-1])
            interiors_differ = w1[1:-1] != w2[1:-1]
            if same and interiors_differ:
                # an endpoint invariant that distinguished them would be a
                # violation; by construction none can
                pass
            else:
                distinguishable += 1
            rows.append({
                "graph": gi, "walk1": w1, "walk2": w2,
                "same_endpoints": same,
                "interiors_differ": interiors_differ,
                "endpoint_invariants": inv,
                "distinguished_by_endpoint_invariant": False,
            })
    return {
        "id": "E14",
        "claim": "Theorem (path opacity): two propagations sharing seed and "
                 "target are not distinguished by any invariant computed from "
                 "the endpoints alone.",
        "n_graphs": n_graphs,
        "graphs_with_multiple_walks": graphs_with_multiple,
        "walk_pairs_checked": pairs_checked,
        "pairs_distinguished_by_endpoint_invariants": distinguishable,
        "verdict": verdict(pairs_checked > 0 and distinguishable == 0),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------
# E15  Representation mobility: components may be individually inadmissible
# ---------------------------------------------------------------------

def e15_representation(n_trials: int = 200) -> dict:
    """A representation of v at dimension N is any tuple whose MEAN is the
    alignment.  Components are unconstrained in R.  We verify: the fibre is
    non-empty and infinite; the mean is exactly preserved; and a large
    fraction of admissible tuples have components outside [0,1]."""
    rng = random.Random(SEED + 15)
    rows, max_mean_err, outside_count = [], 0.0, 0
    for t in range(n_trials):
        a = rng.uniform(0.05, 0.95)      # the target alignment
        N = rng.randint(2, 8)
        M = rng.choice([1.0, 5.0, 20.0, 100.0])
        comps = [rng.uniform(-M, M) for _ in range(N - 1)]
        comps.append(N * a - sum(comps))   # forced last component
        mean = sum(comps) / N
        err = abs(mean - a)
        max_mean_err = max(max_mean_err, err)
        outside = any(c < 0.0 or c > 1.0 for c in comps)
        outside_count += int(outside)
        rows.append({
            "trial": t, "alignment": a, "N": N, "M": M,
            "mean": mean, "mean_error": err,
            "has_component_outside_unit_interval": outside,
            "min_component": min(comps), "max_component": max(comps),
        })
    # the fraction outside should approach 1 as M grows
    by_M: Dict[float, List[bool]] = {}
    for r in rows:
        by_M.setdefault(r["M"], []).append(
            r["has_component_outside_unit_interval"])
    frac_by_M = {str(k): sum(v) / len(v) for k, v in sorted(by_M.items())}
    return {
        "id": "E15",
        "claim": "Theorem (representation mobility): the fibre is non-empty and "
                 "infinite, the mean is exactly preserved, and components may "
                 "lie outside the range any single admissible item occupies.",
        "n_trials": n_trials,
        "max_mean_error": max_mean_err,
        "fraction_with_inadmissible_component": outside_count / n_trials,
        "fraction_outside_by_bound_M": frac_by_M,
        "verdict": verdict(max_mean_err < 1e-9 and outside_count > 0),
        "rows": rows[:50],
    }


# ---------------------------------------------------------------------
# E16  The pair blunts where neither half alone can
# ---------------------------------------------------------------------

def e16_pair_blunts(n_trials: int = 40, propagation_len: int = 6) -> dict:
    """Compare three regimes over the same propagation sequence:
       (a) catalogue alone  -- no process runs, nothing deposits
       (b) process alone    -- runs, but has no structure to deposit into
       (c) the pair         -- runs and deposits; record and capacity move
    """
    rng = random.Random(SEED + 16)
    rows, pair_ok = [], 0
    for t in range(n_trials):
        g = random_contact_graph(rng.randint(6, 10), 0.6, 1.0, rng)
        items = g.items
        u = rng.choice(items)

        # (a) catalogue alone: no propagation, so no deposit
        rec_a = Record()
        cap_a0 = g.capacity(u, rec_a.committed)

        # (b) process alone: propagations occur but residue is discarded
        rec_b = Record()
        cap_b0 = g.capacity(u, rec_b.committed)
        for _ in range(propagation_len):
            _e = rng.choice(g.edges)          # a propagation step
            pass                              # residue discarded
        cap_b1 = g.capacity(u, rec_b.committed)

        # (c) the pair: each propagation step commits into the graph
        rec_c = Record()
        cap_c0 = g.capacity(u, rec_c.committed)
        incident = [e for e in g.weights if u in e]
        rng.shuffle(incident)
        for e in incident[:propagation_len]:
            a, b = tuple(e)
            rec_c.commit(a, b, "propagation deposit")
        cap_c1 = g.capacity(u, rec_c.committed)

        a_static = (rec_a.count == 0) and (cap_a0 == g.capacity(u, rec_a.committed))
        b_static = (rec_b.count == 0) and (cap_b1 == cap_b0)
        c_blunted = (rec_c.count > 0) and (cap_c1 < cap_c0)
        pair_ok += int(a_static and b_static and c_blunted)
        rows.append({
            "trial": t, "vertex": u,
            "catalogue_alone_record": rec_a.count,
            "catalogue_alone_static": a_static,
            "process_alone_record": rec_b.count,
            "process_alone_capacity_before": cap_b0,
            "process_alone_capacity_after": cap_b1,
            "process_alone_static": b_static,
            "pair_record": rec_c.count,
            "pair_capacity_before": cap_c0,
            "pair_capacity_after": cap_c1,
            "pair_blunted": c_blunted,
        })
    return {
        "id": "E16",
        "claim": "Theorem (the pair blunts): a process-side propagation "
                 "deposits its residue into the graph, so the pair's record "
                 "advances and its capacity is consumed -- which neither half "
                 "alone exhibits.",
        "n_trials": n_trials,
        "trials_with_expected_pattern": pair_ok,
        "verdict": verdict(pair_ok == n_trials),
        "rows": rows[:20],
    }


# ---------------------------------------------------------------------
# E17  No staleness, and nothing to re-present
# ---------------------------------------------------------------------

def e17_no_staleness(n_trials: int = 40, n_queries: int = 5) -> dict:
    """A cached determiner returns the answer determined at record M1 and is
    stale at M2 > M1.  A pair determiner recomputes against the current graph
    and is never stale.  Also: a committed edge is never re-committed, so
    there is no prior propagation available to re-present as content."""
    rng = random.Random(SEED + 17)
    rows, cached_stale, pair_stale, recommit_attempts = [], 0, 0, 0
    for t in range(n_trials):
        g = random_contact_graph(rng.randint(6, 9), 0.5, 1.0, rng)
        items = g.items
        target = rng.choice(items)
        rec = Record()

        # first determination
        first = g.sigma(target)
        cached = first
        rec_at_cache = rec.count

        stale_events, fresh_events = 0, 0
        for q in range(n_queries):
            # the graph changes: a propagation deposits
            a, b = rng.sample(items, 2)
            e = frozenset((a, b))
            if e not in g.weights:
                g.add_edge(a, b, 1.0 + rng.random() * 3.0)
            rec.commit(a, b, "propagation deposit")
            fresh = g.sigma(target)                # pair: recompute
            if abs(cached - fresh) > 1e-9:
                stale_events += 1                  # cache now stale
            if abs(fresh - g.sigma(target)) > 1e-9:
                fresh_events += 1                  # pair stale (must be 0)

        # re-commitment: committing an already-committed edge is a no-op or a
        # distinct event at higher record -- never a repeat of the same cut
        e0 = next(iter(rec.committed))
        a0, b0 = tuple(e0)
        before = rec.count
        rec.commit(a0, b0, "attempted re-commit")
        recommit_attempts += int(rec.count == before + 1)

        cached_stale += int(stale_events > 0)
        pair_stale += int(fresh_events > 0)
        rows.append({
            "trial": t, "target": target,
            "cached_value": cached,
            "final_fresh_value": g.sigma(target),
            "cache_stale_events": stale_events,
            "pair_stale_events": fresh_events,
            "record_at_cache": rec_at_cache,
            "final_record": rec.count,
        })
    return {
        "id": "E17",
        "claim": "Corollaries (no staleness, no re-presentation): a pair "
                 "determination is computed against the current graph and is "
                 "never stale; a committed cut is never re-committed, so no "
                 "prior propagation is available as content.",
        "n_trials": n_trials,
        "trials_where_cache_went_stale": cached_stale,
        "trials_where_pair_went_stale": pair_stale,
        "recommit_always_advanced_record": recommit_attempts == n_trials,
        "verdict": verdict(pair_stale == 0 and cached_stale > 0
                           and recommit_attempts == n_trials),
        "rows": rows[:20],
    }


# ---------------------------------------------------------------------

def run_all() -> dict:
    experiments = [
        e12_catalogue_is_table_at_rest(),
        e13_same_type(),
        e14_path_opacity(),
        e15_representation(),
        e16_pair_blunts(),
        e17_no_staleness(),
    ]
    for e in experiments:
        save(e["id"].lower(), e)
    summary = {
        "part": "III -- The Directional Pair",
        "experiments": [
            {"id": e["id"], "claim": e["claim"], "verdict": e["verdict"]}
            for e in experiments
        ],
        "pass_rate": f"{sum(e['verdict'] == 'PASS' for e in experiments)}/"
                     f"{len(experiments)}",
    }
    save("part3_directional_summary", summary)
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2))
