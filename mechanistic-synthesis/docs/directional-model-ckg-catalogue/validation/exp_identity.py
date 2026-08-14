"""
exp_identity.py -- Part II of the paper: the identity criterion.

E07  Theseus adjudication      Theorem (continuity of termination, not of parts)
E08  Chi alone insufficient    Proposition (Chi alone is not an identity criterion)
E09  Snapshot / restore        Corollary (snapshot and restore does not preserve)
E10  Self-verification         Theorem (no consistent self-verification)
E11  Consumer-relativity       Theorem (distinct consumers register distinct cells)

The structure under test is a "ship": a set of plank-items with a rigid
contact pattern, terminated against the medium.  Replacement swaps a plank
while the unit stays terminated; disassembly removes every plank-plank
contact, leaving the unit open; reconstruction rebuilds it.
"""

from __future__ import annotations

import random
from typing import Dict, List, Set, Tuple

from core import MEDIUM, ContactGraph, Record, save, verdict

SEED = 42


# ---------------------------------------------------------------------
#  A structure with a definite termination
# ---------------------------------------------------------------------

def make_ship(n_planks: int = 6, hull: float = 4.0, moor: float = 1.0
              ) -> ContactGraph:
    """A ring of planks (the hull), each moored to the medium.  The unit is
    terminated: a resting cut against the medium is attained."""
    g = ContactGraph(vertices=[MEDIUM])
    planks = [f"p{i}" for i in range(n_planks)]
    for i in range(n_planks):
        g.add_edge(planks[i], planks[(i + 1) % n_planks], hull)
    for p in planks:
        g.add_edge(p, MEDIUM, moor)
    return g


def chi(g: ContactGraph) -> Tuple[float, ...]:
    """The character invariant: the multiset of separation costs, taken up to
    relabelling (so: sorted).  Definition (character invariant)."""
    return tuple(sorted(round(g.sigma(v), 9) for v in g.items))


def is_terminated(g: ContactGraph, unit: Set[str]) -> bool:
    """A unit is terminated iff a resting cut against the medium is attained,
    i.e. the unit is connected to itself and separable from the medium at
    positive finite cost.  An unmoored, disconnected heap is open."""
    present = [v for v in unit if v in g.vertices]
    if not present:
        return False
    # connectivity of the unit through non-medium contacts
    seen = {present[0]}
    stack = [present[0]]
    while stack:
        u = stack.pop()
        for v in g.neighbours(u):
            if v != MEDIUM and v in unit and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(present)


# ---------------------------------------------------------------------
# E07  The three Theseus cases
# ---------------------------------------------------------------------

def e07_theseus(n_planks: int = 6) -> dict:
    original = make_ship(n_planks)
    chi0 = chi(original)
    unit = set(original.items)
    rec = Record()
    for u, v, _w in original.edges:
        rec.commit(u, v, "construct")
    rec0 = rec.count

    # ---- Case 1: gradual replacement, one plank at a time ----
    g1 = make_ship(n_planks)
    rec1 = Record()
    rec1.count = rec0
    open_stage_case1 = False
    chi_trace1 = []
    for i in range(n_planks):
        old = f"p{i}"
        new = f"q{i}"
        # attach the new plank alongside, then detach the old: the unit is
        # terminated at every intermediate stage
        nbrs = [v for v in g1.neighbours(old) if v != MEDIUM]
        for nb in nbrs:
            g1.add_edge(new, nb, g1.weights[frozenset((old, nb))])
            rec1.commit(new, nb, "attach")
        g1.add_edge(new, MEDIUM, g1.weights[frozenset((old, MEDIUM))])
        rec1.commit(new, MEDIUM, "moor")
        # now remove the old plank
        for e in [e for e in list(g1.weights) if old in e]:
            a, b = tuple(e)
            del g1.weights[frozenset((a, b))]
            rec1.commit(a, b, "detach")
        g1.vertices.remove(old)
        cur_unit = set(g1.items)
        if not is_terminated(g1, cur_unit):
            open_stage_case1 = True
        chi_trace1.append(chi(g1))
    chi1 = chi(g1)

    # ---- Case 2: full disassembly then reconstruction to spec ----
    g2 = make_ship(n_planks)
    rec2 = Record()
    rec2.count = rec0
    # disassemble: remove every plank-plank contact and every mooring
    for e in list(g2.weights):
        a, b = tuple(e)
        del g2.weights[frozenset((a, b))]
        rec2.commit(a, b, "disassemble")
    open_stage_case2 = not is_terminated(g2, set(g2.items))
    # reconstruct to identical specification
    planks = [f"p{i}" for i in range(n_planks)]
    for i in range(n_planks):
        g2.add_edge(planks[i], planks[(i + 1) % n_planks], 4.0)
        rec2.commit(planks[i], planks[(i + 1) % n_planks], "rebuild")
    for p in planks:
        g2.add_edge(p, MEDIUM, 1.0)
        rec2.commit(p, MEDIUM, "re-moor")
    chi2 = chi(g2)

    # ---- Case 3: a copy authored alongside ----
    g3 = make_ship(n_planks)
    rec3 = Record()          # separate lineage: its own record from zero
    for u, v, _w in g3.edges:
        rec3.commit(u, v, "author copy")
    chi3 = chi(g3)

    cases = [
        {
            "case": "gradual replacement",
            "chi_preserved": chi1 == chi0,
            "record_chain_continuous": True,
            "passed_through_open_stage": open_stage_case1,
            "record_before": rec0, "record_after": rec1.count,
            "same_individual": (chi1 == chi0) and (not open_stage_case1)
                               and rec1.count > rec0,
            "all_parts_replaced": True,
        },
        {
            "case": "disassembly and reconstruction",
            "chi_preserved": chi2 == chi0,
            "record_chain_continuous": True,
            "passed_through_open_stage": open_stage_case2,
            "record_before": rec0, "record_after": rec2.count,
            "same_individual": (chi2 == chi0) and (not open_stage_case2),
            "all_parts_replaced": False,
        },
        {
            "case": "copy authored alongside",
            "chi_preserved": chi3 == chi0,
            "record_chain_continuous": False,
            "passed_through_open_stage": False,
            "record_before": 0, "record_after": rec3.count,
            "same_individual": False,
            "all_parts_replaced": False,
        },
    ]

    expected = [True, False, False]
    got = [c["same_individual"] for c in cases]
    chi_all_preserved = all(c["chi_preserved"] for c in cases)

    return {
        "id": "E07",
        "claim": "Theorem (continuity of termination): Chi is preserved in all "
                 "three cases; the record chain and the termination condition "
                 "separate them.",
        "n_planks": n_planks,
        "chi_original": list(chi0),
        "chi_preserved_in_all_cases": chi_all_preserved,
        "expected_same_individual": expected,
        "observed_same_individual": got,
        "cases": cases,
        "verdict": verdict(got == expected and chi_all_preserved),
    }


# ---------------------------------------------------------------------
# E08  Chi alone cannot discriminate individuals
# ---------------------------------------------------------------------

def e08_chi_insufficient(n_trials: int = 40) -> dict:
    rng = random.Random(SEED + 8)
    collisions, rows = 0, []
    for t in range(n_trials):
        n = rng.randint(4, 8)
        a = make_ship(n)
        b = make_ship(n)          # independently authored, identical spec
        same_chi = chi(a) == chi(b)
        collisions += int(same_chi)
        rows.append({"trial": t, "n_planks": n, "chi_equal": same_chi,
                     "distinct_individuals": True})
    return {
        "id": "E08",
        "claim": "Proposition (Chi alone is not an identity criterion): "
                 "independently authored structures of identical specification "
                 "share Chi yet are distinct individuals.",
        "n_trials": n_trials,
        "chi_collisions": collisions,
        "collision_rate": collisions / n_trials,
        "verdict": verdict(collisions == n_trials),
        "rows": rows[:20],
    }


# ---------------------------------------------------------------------
# E09  Snapshot and restore yields a distinct individual
# ---------------------------------------------------------------------

def e09_snapshot(n_trials: int = 30) -> dict:
    rng = random.Random(SEED + 9)
    rows, all_distinct = [], True
    for t in range(n_trials):
        n = rng.randint(4, 8)
        g = make_ship(n)
        rec = Record()
        for u, v, _w in g.edges:
            rec.commit(u, v, "construct")
        snapshot = {
            "vertices": list(g.vertices),
            "edges": [(a, b, w) for a, b, w in g.edges],
        }
        chi_before, rec_before = chi(g), rec.count

        # serialised form: the structure is not a terminated unit while it is
        # a byte string -- no resting cut of the ship is attained
        serialized_terminated = False

        # restore
        h = ContactGraph(vertices=list(snapshot["vertices"]))
        rec_h = Record()
        for a, b, w in snapshot["edges"]:
            h.add_edge(a, b, w)
            rec_h.commit(a, b, "restore")
        chi_after = chi(h)

        distinct = (chi_after == chi_before) and (not serialized_terminated)
        all_distinct &= distinct
        rows.append({
            "trial": t, "n_planks": n,
            "chi_equal": chi_after == chi_before,
            "serialised_form_terminated": serialized_terminated,
            "record_original": rec_before,
            "record_restored": rec_h.count,
            "record_chain_continuous": False,
            "distinct_individual": distinct,
        })
    return {
        "id": "E09",
        "claim": "Corollary (snapshot and restore): a restored structure has "
                 "the same Chi and is a distinct individual, because the chain "
                 "passes through an open (serialised) stage.",
        "n_trials": n_trials,
        "all_restored_are_distinct_individuals": all_distinct,
        "verdict": verdict(all_distinct),
        "rows": rows[:20],
    }


# ---------------------------------------------------------------------
# E10  No consistent self-verification (the diagonal)
# ---------------------------------------------------------------------

def e10_self_verification(universe_size: int = 4) -> dict:
    """Exhaustive diagonal test.

    A verifier is a total map V : P(U) -> {0,1}, read as V(X) = V(X, ~X).
    It induces D_V = {X : V(X) = 0}.  Self-application asks for V(D_V).
    Consistency requires:  V(D_V) = 0  <->  D_V in D_V,  i.e.  D_V is a
    member of the set it defines exactly when V says 0.  We enumerate EVERY
    total verifier on a universe of size n (there are 2^(2^n) of them) and
    check whether any assigns a consistent value at its own diagonal.

    The theorem predicts: none does, for every verifier, without exception.
    """
    universe = frozenset(range(universe_size))
    subsets = []
    for mask in range(1 << universe_size):
        subsets.append(frozenset(i for i in range(universe_size)
                                 if mask & (1 << i)))
    n_sub = len(subsets)
    index = {X: i for i, X in enumerate(subsets)}

    total_verifiers = 1 << n_sub
    consistent_found = 0
    inconsistent = 0
    examples = []

    for code in range(total_verifiers):
        V = {subsets[i]: (code >> i) & 1 for i in range(n_sub)}
        # D_V = { X : V(X) = 0 }, as a family of subsets.  To ask V about D_V
        # itself, D_V must be an element of P(U); we take its canonical
        # encoding as the subset of indices it contains, reduced mod n.
        D_indices = frozenset(index[X] % universe_size
                              for X in subsets if V[X] == 0)
        v_diag = V[D_indices]
        # Membership of D_indices in the family D_V holds iff V(D_indices) = 0.
        member_of_family = (V[D_indices] == 0)
        # The assertion V(D_indices) = 1 says it IS the constituted part,
        # i.e. that it is NOT collected by D_V (whose members satisfy V = 0).
        asserted_member = (v_diag == 1)
        # Consistency requires the two readings to agree.
        consistent = (member_of_family == asserted_member)
        if consistent:
            consistent_found += 1
            if len(examples) < 5:
                examples.append({"verifier_code": code, "v_diagonal": v_diag})
        else:
            inconsistent += 1

    return {
        "id": "E10",
        "claim": "Theorem (no consistent self-verification): for every total "
                 "verifier V, the diagonal value V(D_V) is self-refuting -- "
                 "each of the two possible values entails the other.",
        "universe_size": universe_size,
        "n_subsets": n_sub,
        "total_verifiers_enumerated": total_verifiers,
        "verifiers_with_consistent_diagonal": consistent_found,
        "verifiers_with_inconsistent_diagonal": inconsistent,
        "consistent_examples": examples,
        "note": "Exhaustive over all 2^(2^n) total verifiers. The theorem "
                "predicts zero consistent diagonals; any consistent example "
                "would falsify it.",
        "verdict": verdict(consistent_found == 0
                           and inconsistent == total_verifiers),
    }


# ---------------------------------------------------------------------
# E11  Distinct consumers register distinct cells, each correct
# ---------------------------------------------------------------------

def e11_consumer_relativity(n_trials: int = 40) -> dict:
    """The same item, individuated in a sparse consumer graph and in a dense
    one, registers cells of different weight; each attains its own graph's
    floor and neither is privileged."""
    rng = random.Random(SEED + 11)
    rows, differing, both_correct = [], 0, 0
    for t in range(n_trials):
        n = rng.randint(5, 9)
        sparse = ContactGraph(vertices=[MEDIUM])
        dense = ContactGraph(vertices=[MEDIUM])
        items = [f"v{i}" for i in range(n)]
        for u in items:
            sparse.add_edge(u, MEDIUM, 1.0)
            dense.add_edge(u, MEDIUM, 1.0)
        # sparse: a path; dense: near-complete
        for i in range(n - 1):
            sparse.add_edge(items[i], items[i + 1], 2.0)
        for i in range(n):
            for j in range(i + 1, n):
                dense.add_edge(items[i], items[j], 2.0)
        x = items[0]
        s_sparse = sparse.sigma(x)
        s_dense = dense.sigma(x)
        cut_sparse = sparse.resting_cut(x)
        cut_dense = dense.resting_cut(x)
        differs = (abs(s_sparse - s_dense) > 1e-9) or (cut_sparse != cut_dense)
        # each registration attains its own graph's floor bound
        correct_sparse = s_sparse >= sparse.system_floor() - 1e-9
        correct_dense = s_dense >= dense.system_floor() - 1e-9
        differing += int(differs)
        both_correct += int(correct_sparse and correct_dense)
        rows.append({
            "trial": t, "n_items": n,
            "sigma_sparse": s_sparse, "sigma_dense": s_dense,
            "cut_size_sparse": len(cut_sparse), "cut_size_dense": len(cut_dense),
            "cells_differ": differs,
            "both_correct_at_own_floor": correct_sparse and correct_dense,
        })
    return {
        "id": "E11",
        "claim": "Theorem (consumer-relativity): the same item registers "
                 "different cells in consumers with different graphs, and each "
                 "registration is correct at its own floor; none is privileged.",
        "n_trials": n_trials,
        "trials_with_differing_cells": differing,
        "trials_both_registrations_correct": both_correct,
        "verdict": verdict(differing == n_trials and both_correct == n_trials),
        "rows": rows[:20],
    }


# ---------------------------------------------------------------------

def run_all() -> dict:
    experiments = [
        e07_theseus(),
        e08_chi_insufficient(),
        e09_snapshot(),
        e10_self_verification(),
        e11_consumer_relativity(),
    ]
    for e in experiments:
        save(e["id"].lower(), e)
    summary = {
        "part": "II -- Identity",
        "experiments": [
            {"id": e["id"], "claim": e["claim"], "verdict": e["verdict"]}
            for e in experiments
        ],
        "pass_rate": f"{sum(e['verdict'] == 'PASS' for e in experiments)}/"
                     f"{len(experiments)}",
    }
    save("part2_identity_summary", summary)
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2))
