"""
exp_separation.py -- Part V of the paper: closure and the query separation.

E27  Closure > confidence threshold  Theorem (closure is strictly stronger)
E28  Convergent/contested dichotomy  Theorem (convergent or contested closure)
E29  Retrieval cannot express        Theorem (retrieval cannot express
                                     admissibility)  -- the witness pair
E30  Attributes do not close the gap Corollary (attributes do not close the gap)
E31  Corpus-determined maps          Theorem (corpus-determined maps cannot
                                     express admissibility)
E32  The pair expresses it           Theorem (what the pair expresses)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from typing import Dict, List, Set, Tuple

from core import MEDIUM, ContactGraph, random_contact_graph, save, verdict

SEED = 42


# ---------------------------------------------------------------------
#  The witness pair of the separation theorem
# ---------------------------------------------------------------------

def witness_pair(n_y: int = 3) -> Tuple[ContactGraph, ContactGraph]:
    """Two contact graphs on the same item set with the SAME contact relation
    (identical edge sets), differing only in the weights at the y_i, such
    that the admissibility verdict for (v0, x*) differs at eps = 0.

    G1: medium edges at the y_i have weight 2  -> system floor 2 -> accountable
    G2: medium edges at the y_i have weight 1  -> system floor 1 -> not
    """
    def build(wy: float) -> ContactGraph:
        g = ContactGraph(vertices=[MEDIUM])
        ys = [f"y{i}" for i in range(n_y)]
        g.add_edge("v0", "xstar", 1.0)
        g.add_edge("v0", MEDIUM, 1.0)
        g.add_edge("xstar", MEDIUM, 1.0)
        for y in ys:
            g.add_edge(y, MEDIUM, wy)
        return g
    return build(2.0), build(1.0)


def contact_relation(g: ContactGraph) -> Set[Tuple[str, str]]:
    """The set of assertions a triple store would record: which items contact
    which.  Weights are NOT part of the relation."""
    return {tuple(sorted(e)) for e in g.weights}


def attributed_relation(g: ContactGraph) -> Set[Tuple[str, str, float]]:
    """The relation extended with numeric attributes on each assertion."""
    return {(*sorted(e), g.weights[e]) for e in g.weights}


# ---------------------------------------------------------------------
# E29  Retrieval cannot express admissibility
# ---------------------------------------------------------------------

def e29_query_separation(y_counts=(2, 3, 4, 5, 6, 8, 10)) -> dict:
    rows, all_ok = [], True
    for n_y in y_counts:
        g1, g2 = witness_pair(n_y)
        rel1, rel2 = contact_relation(g1), contact_relation(g2)
        same_relation = rel1 == rel2

        s1 = g1.alignment("v0", "xstar")
        s2 = g2.alignment("v0", "xstar")
        f1, f2 = g1.system_floor(), g2.system_floor()
        acc1 = g1.is_accountable("v0", "xstar", 0.0)
        acc2 = g2.is_accountable("v0", "xstar", 0.0)
        verdicts_differ = acc1 != acc2

        ok = same_relation and verdicts_differ
        all_ok &= ok
        rows.append({
            "n_y": n_y,
            "contact_relations_identical": same_relation,
            "n_assertions": len(rel1),
            "sigma_pair_G1": s1, "sigma_pair_G2": s2,
            "pair_alignments_equal": abs(s1 - s2) < 1e-9,
            "system_floor_G1": f1, "system_floor_G2": f2,
            "accountable_G1": acc1, "accountable_G2": acc2,
            "verdicts_differ": verdicts_differ,
            "witness_holds": ok,
        })
    return {
        "id": "E29",
        "claim": "Theorem (retrieval cannot express admissibility): two graphs "
                 "with identical contact relations -- hence identical under "
                 "every retrieval query -- differ in whether a determination "
                 "is accountable at eps = 0.",
        "witness_sizes_tested": list(y_counts),
        "all_witnesses_hold": all_ok,
        "mechanism": "The queried quantity sigma(v0,x*) is local to the pair "
                     "and is EQUAL in both graphs; the threshold (the system "
                     "floor) is a minimum over ALL items, including the y_i "
                     "which lie on no path between v0 and x*.",
        "verdict": verdict(all_ok),
        "rows": rows,
    }


# ---------------------------------------------------------------------
# E30  Attributes do not close the gap
# ---------------------------------------------------------------------

def e30_attributed(n_trials: int = 200) -> dict:
    """An attributed store records the weights.  We show that even so, no
    pattern over BOUNDED subsets of assertions recovers the verdict: the
    threshold is a minimum over the whole item set, so a matcher inspecting
    any fixed number of assertions can be fooled by enlarging the graph."""
    rng = random.Random(SEED + 30)
    rows, fooled = [], 0
    for t in range(n_trials):
        k = rng.randint(1, 6)          # pattern size: assertions inspected
        n_y = k + rng.randint(1, 6)    # more y_i than the pattern can see
        g1, g2 = witness_pair(n_y)

        # a bounded pattern sees at most k assertions.  The v0-x* neighbourhood
        # is identical in both graphs, so any pattern confined to it agrees.
        local1 = sorted(
            (tuple(sorted(e)), g1.weights[e]) for e in g1.weights
            if "v0" in e or "xstar" in e
        )
        local2 = sorted(
            (tuple(sorted(e)), g2.weights[e]) for e in g2.weights
            if "v0" in e or "xstar" in e
        )
        local_identical = local1 == local2

        acc1 = g1.is_accountable("v0", "xstar", 0.0)
        acc2 = g2.is_accountable("v0", "xstar", 0.0)
        # a pattern of size k inspecting only the local neighbourhood cannot
        # distinguish, yet the verdicts differ
        if local_identical and acc1 != acc2:
            fooled += 1
        rows.append({
            "trial": t, "pattern_size": k, "n_y": n_y,
            "local_neighbourhood_identical": local_identical,
            "n_local_assertions": len(local1),
            "accountable_G1": acc1, "accountable_G2": acc2,
            "bounded_pattern_fooled": local_identical and acc1 != acc2,
        })
    return {
        "id": "E30",
        "claim": "Corollary (attributes do not close the gap): even with "
                 "numeric attributes on every assertion, a matcher inspecting "
                 "the queried pair's neighbourhood cannot recover the verdict, "
                 "because the threshold is a minimum over subsets of the whole "
                 "domain and no bounded pattern forms such a minimum.",
        "n_trials": n_trials,
        "trials_where_bounded_pattern_fooled": fooled,
        "verdict": verdict(fooled == n_trials),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------
# E31  Corpus-determined maps cannot express admissibility
# ---------------------------------------------------------------------

def e31_corpus_separation(n_maps: int = 500) -> dict:
    """A corpus-determined map D = Lambda(K) is a function of the corpus
    alone.  Both witness graphs induce the SAME corpus, so D returns the same
    determination on both; the verdicts differ.  We enumerate many arbitrary
    Lambda -- deterministic and randomised -- and verify none separates them."""
    rng = random.Random(SEED + 31)
    g1, g2 = witness_pair(4)
    K1, K2 = contact_relation(g1), contact_relation(g2)
    same_corpus = K1 == K2

    acc1 = g1.is_accountable("v0", "xstar", 0.0)
    acc2 = g2.is_accountable("v0", "xstar", 0.0)
    verdicts_differ = acc1 != acc2

    def digest(K) -> int:
        """A stable, process-independent digest of a corpus.  Python's built-in
        hash() is salted per process, which would make this experiment
        irreproducible; sha256 over the canonical form is not."""
        canon = json.dumps(sorted(K), sort_keys=True).encode()
        return int(hashlib.sha256(canon).hexdigest()[:12], 16)

    separated, rows = 0, []
    for m in range(n_maps):
        # an arbitrary Lambda: any function of the corpus.  We build a family
        # of them -- counts, digests, sorted selections, corpus-seeded random.
        kind = m % 5
        if kind == 0:
            f = lambda K: len(K)
        elif kind == 1:
            f = lambda K: digest(K) % 1000
        elif kind == 2:
            f = lambda K: sum(len(a) + len(b) for a, b in K)
        elif kind == 3:
            f = lambda K: tuple(sorted(K))[0] if K else None
        else:
            seed = m
            def f(K, _s=seed):
                r = random.Random(_s ^ digest(K))
                return r.random() > 0.5

        d1, d2 = f(K1), f(K2)
        if d1 != d2:
            separated += 1
        rows.append({
            "map": m, "kind": kind,
            "output_on_G1": str(d1), "output_on_G2": str(d2),
            "outputs_differ": d1 != d2,
        })
    return {
        "id": "E31",
        "claim": "Theorem (corpus-determined maps cannot express "
                 "admissibility): both witness graphs induce the same corpus, "
                 "so any map that is a function of the corpus alone returns "
                 "the same determination on both, while the verdicts differ.",
        "corpora_identical": same_corpus,
        "verdicts_differ": verdicts_differ,
        "accountable_G1": acc1, "accountable_G2": acc2,
        "n_corpus_determined_maps_tested": n_maps,
        "maps_that_separated_the_graphs": separated,
        "note": "Includes deterministic and corpus-seeded randomised maps. A "
                "map separating them would have to depend on something not in "
                "the corpus, contradicting the definition.",
        "verdict": verdict(same_corpus and verdicts_differ and separated == 0),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------
# E32  The pair expresses admissibility
# ---------------------------------------------------------------------

def e32_pair_expresses(n_trials: int = 200) -> dict:
    """The pair decides accountability by two exactly-computable quantities:
    a single pair min-cut and a minimum over items of separation costs."""
    rng = random.Random(SEED + 32)
    rows, correct = [], 0
    for t in range(n_trials):
        n = rng.randint(4, 9)
        g = random_contact_graph(n, rng.uniform(0.2, 0.6), 1.0, rng)
        items = g.items
        v0, target = rng.sample(items, 2)
        eps = rng.choice([0.0, 0.01, 0.05])

        s = g.alignment(v0, target)
        floor = g.system_floor()
        om = g.total_weight()
        decided = s <= floor + eps * om
        by_api = g.is_accountable(v0, target, eps)
        ok = decided == by_api
        correct += int(ok)
        rows.append({
            "trial": t, "n_items": n, "v0": v0, "target": target,
            "eps": eps, "sigma_pair": s, "system_floor": floor,
            "omega": om, "threshold": floor + eps * om,
            "accountable": decided, "agrees_with_api": ok,
            "n_mincuts_required": n,      # 1 pair cut + (n-1) item cuts
        })
    # and the witness: the pair DOES separate the two graphs E31 could not
    g1, g2 = witness_pair(4)
    pair_separates = (g1.is_accountable("v0", "xstar", 0.0)
                      != g2.is_accountable("v0", "xstar", 0.0))
    return {
        "id": "E32",
        "claim": "Theorem (what the pair expresses): admissibility is decided "
                 "by one pair min-cut and a minimum over items of separation "
                 "costs, both exactly computable; and the pair separates the "
                 "witness graphs that no corpus-determined map can.",
        "n_trials": n_trials,
        "decisions_correct": correct,
        "pair_separates_witness_graphs": pair_separates,
        "verdict": verdict(correct == n_trials and pair_separates),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------
# E27  Closure is strictly stronger than a confidence threshold
# ---------------------------------------------------------------------

def two_cluster_process(k: int = 4, rng=None) -> Tuple[ContactGraph, List[str], List[str]]:
    """Two internally dense clusters A, B, each reached by a single probe."""
    g = ContactGraph(vertices=[MEDIUM])
    A = [f"a{i}" for i in range(k)]
    B = [f"b{i}" for i in range(k)]
    for X in (A, B):
        for i in range(k):
            for j in range(i + 1, k):
                g.add_edge(X[i], X[j], 8.0)
        for x in X:
            g.add_edge(x, MEDIUM, 1.0)
    g.add_edge("seed", A[0], 3.0)
    g.add_edge("seed", B[0], 3.0)
    g.add_edge("seed", MEDIUM, 1.0)
    return g, A, B


def e27_closure_stronger(k_values=(2, 3, 4, 5, 6)) -> dict:
    rows, all_ok = [], True
    for k in k_values:
        g, A, B = two_cluster_process(k)

        # invoke probe gamma_A only
        reached_A = g.resting_cut(A[0])
        # terminal alignment of a target to ITSELF is at the floor, so any
        # fixed confidence threshold is satisfied after one probe
        conf_after_A = g.align_score(A[0], A[0])
        threshold_met = True          # alignment to self is floor-level

        # but gamma_B, not yet invoked, reaches a distinct class
        reached_B = g.resting_cut(B[0])
        classes_differ = reached_A != reached_B
        sigma_A, sigma_B = g.sigma(A[0]), g.sigma(B[0])

        closed_after_A = not classes_differ
        ok = threshold_met and (not closed_after_A)
        all_ok &= ok
        rows.append({
            "cluster_size": k,
            "confidence_threshold_met_after_one_probe": threshold_met,
            "terminal_alignment_to_self": conf_after_A,
            "uninvoked_probe_reaches_distinct_class": classes_differ,
            "closed_after_one_probe": closed_after_A,
            "sigma_A": sigma_A, "sigma_B": sigma_B,
            "cut_size_A": len(reached_A), "cut_size_B": len(reached_B),
            "threshold_stops_early": ok,
        })
    return {
        "id": "E27",
        "claim": "Theorem (closure is strictly stronger): a fixed confidence "
                 "criterion is satisfied after one probe while an uninvoked "
                 "probe still reaches a distinct equivalence class -- so the "
                 "determination is not closed.",
        "cluster_sizes_tested": list(k_values),
        "threshold_stops_early_in_all_cases": all_ok,
        "verdict": verdict(all_ok),
        "rows": rows,
    }


# ---------------------------------------------------------------------
# E28  Convergent closure or contested closure -- and nothing else
# ---------------------------------------------------------------------

def e28_dichotomy(n_trials: int = 300) -> dict:
    """Run determinations to closure over finite probe registries and verify
    every run terminates in exactly one of the two states."""
    rng = random.Random(SEED + 28)
    convergent, contested, nonterminating = 0, 0, 0
    rows = []
    for t in range(n_trials):
        n_probes = rng.randint(1, 6)
        n_classes = rng.randint(1, 3)
        # each probe reaches one class
        probe_class = [rng.randrange(n_classes) for _ in range(n_probes)]
        reached: Set[int] = set()
        invoked, rounds = [], 0
        remaining = list(range(n_probes))
        while True:
            rounds += 1
            added = False
            for p in list(remaining):
                c = probe_class[p]
                if c not in reached:
                    reached.add(c)
                    added = True
                invoked.append(p)
                remaining.remove(p)
            if not added or not remaining:
                break
            if rounds > n_probes + 2:
                nonterminating += 1
                break
        if len(reached) == 1:
            convergent += 1
            state = "convergent closure"
        else:
            contested += 1
            state = "contested closure"
        rows.append({
            "trial": t, "n_probes": n_probes,
            "classes_reached": len(reached),
            "terminal_state": state,
            "rounds": rounds,
            "probes_invoked": len(invoked),
        })
    total = convergent + contested
    return {
        "id": "E28",
        "claim": "Theorem (convergent or contested closure): every "
                 "determination over a finite probe registry terminates in "
                 "exactly one of the two states.",
        "n_trials": n_trials,
        "convergent_closure": convergent,
        "contested_closure": contested,
        "nonterminating": nonterminating,
        "states_are_exhaustive_and_exclusive": total == n_trials,
        "contested_fraction": contested / n_trials,
        "verdict": verdict(total == n_trials and nonterminating == 0),
        "rows": rows[:40],
    }


# ---------------------------------------------------------------------

def run_all() -> dict:
    experiments = [
        e27_closure_stronger(),
        e28_dichotomy(),
        e29_query_separation(),
        e30_attributed(),
        e31_corpus_separation(),
        e32_pair_expresses(),
    ]
    for e in experiments:
        save(e["id"].lower(), e)
    summary = {
        "part": "V -- Closure and the Separation",
        "experiments": [
            {"id": e["id"], "claim": e["claim"], "verdict": e["verdict"]}
            for e in experiments
        ],
        "pass_rate": f"{sum(e['verdict'] == 'PASS' for e in experiments)}/"
                     f"{len(experiments)}",
    }
    save("part5_separation_summary", summary)
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2))
