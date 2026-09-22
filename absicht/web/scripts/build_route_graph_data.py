"""
Repackages absicht/docs/distributed-domain-route-graph/validation/results/*.json
into the flat exp1..exp8 shape route-graph-charts.js expects, mirroring how
this site's other page consumes validation.json. Run this after re-running
that paper's validation suite to refresh the interactive page's charts.

public/data/ is gitignored (matching the rest of this site), so this script
is the reproducible source of truth for route-graph-validation.json, not
the JSON file itself.

Usage:  python scripts/build_route_graph_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent.parent / "docs" / "distributed-domain-route-graph" / "validation" / "results"
OUT = HERE.parent / "public" / "data" / "route-graph-validation.json"


def main() -> None:
    exp01 = json.loads((RESULTS / "exp01_route_receiver.json").read_text())
    exp02 = json.loads((RESULTS / "exp02_waterfill_limit.json").read_text())
    exp03 = json.loads((RESULTS / "exp03_phaselock_floor.json").read_text())
    exp04 = json.loads((RESULTS / "exp04_generate_test.json").read_text())
    exp05 = json.loads((RESULTS / "exp05_seam_end_to_end.json").read_text())
    exp06 = json.loads((RESULTS / "exp06_crowd_routing.json").read_text())
    exp07 = json.loads((RESULTS / "exp07_binary_lock.json").read_text())
    exp08 = json.loads((RESULTS / "exp08_operation_exhaustion.json").read_text())

    out = {
        "exp1": {
            "Ns": [4, 8, 16],
            "know_trace": {n: exp01["example_traces"][str(n)]["know_size"] for n in [4, 8, 16]},
            "cands_trace": {n: exp01["example_traces"][str(n)]["cands_size"] for n in [4, 8, 16]},
            "floor_mean_by_N": {n: exp01["trials_by_N"][str(n)]["floor_stats"]["mean"] for n in [4, 8, 16]},
            "floor_raw_by_N": {n: exp01["trials_by_N"][str(n)]["final_floors_raw"] for n in [4, 8, 16]},
            "bounded_satisfaction_rate": exp01["bounded_satisfaction_rate"],
        },
        "exp2": {
            "ks": [t["k"] for t in exp02["trials"]],
            "ratio_mean": [t["ratio_stats"]["mean"] for t in exp02["trials"]],
            "ratio_lo": [t["ratio_stats"]["ci95_lo"] for t in exp02["trials"]],
            "ratio_hi": [t["ratio_stats"]["ci95_hi"] for t in exp02["trials"]],
            "relative_cost": [t["max_relative_item_cost"] for t in exp02["trials"]],
            "ratio_at_largest_k": exp02["ratio_at_largest_k"],
        },
        "exp3": {
            "n_locked": [int(k) for k in exp03["q_joint_by_n_locked_above_Kc"].keys()],
            "q_joint_locked": list(exp03["q_joint_by_n_locked_above_Kc"].values()),
            "mean_q_locked": exp03["mean_q_locked_when_corrupted_present"],
            "mean_q_naive": exp03["mean_q_naive_when_corrupted_present"],
            "mean_R_above_Kc": exp03["mean_R_final_above_Kc"],
            "mean_R_below_Kc": exp03["mean_R_final_below_Kc"],
            "locked_beats_naive_rate": exp03["locked_beats_naive_rate"],
        },
        "exp4": {
            "resolved_rate": exp04["resolved_rate_phase2"],
            "exhaustion_rate": exp04["exhaustion_rate_phase1"],
            "dichotomy_violations": exp04["dichotomy_violations"],
            "resolved_trace": exp04["example_traces"]["resolved"],
            "declined_trace": exp04["example_traces"]["declined"],
        },
        "exp5": {
            "Ns": [4, 8, 16, 24],
            "achieved_floor_by_N": {n: exp05["trials_by_N"][str(n)]["mean_achieved_floor"] for n in [4, 8, 16, 24]},
            "bound_satisfaction_rate": exp05["bound_satisfaction_rate"],
            "decline_rate": exp05["decline_rate"],
            "tightness_gap_mean": exp05["tightness_gap_at_N24_stats"]["mean"],
            "tightness_gap_ci": [
                exp05["tightness_gap_at_N24_stats"]["ci95_lo"],
                exp05["tightness_gap_at_N24_stats"]["ci95_hi"],
            ],
        },
        "exp6": {
            "n_consult": [2, 4, 6, 8, 10],
            "q_routed": list(exp06["q_routed_by_n_consult"].values()),
            "q_uniform": list(exp06["q_uniform_by_n_consult"].values()),
            "ratio_mean": exp06["ratio_stats"]["mean"],
            "ratio_ci": [exp06["ratio_stats"]["ci95_lo"], exp06["ratio_stats"]["ci95_hi"]],
            "never_worse_rate": exp06["never_worse_rate"],
        },
        "exp7": {
            "delta": exp07["delta"],
            "fraction_extinct": exp07["fraction_extinct"],
            "fraction_distinguishable": exp07["fraction_distinguishable"],
            "min_positive_lag": exp07["min_positive_lag_observed"],
            "theoretical_min_positive_lag": exp07["theoretical_min_positive_lag"],
            "lag_sample": exp07["lag_distribution_sample"],
            "n_locked": [int(k) for k in exp07["q_joint_by_n_locked_above_Kc"].keys()],
            "q_joint_locked": list(exp07["q_joint_by_n_locked_above_Kc"].values()),
            "locked_beats_naive_rate": exp07["locked_beats_naive_rate"],
        },
        "exp8": {
            "n_operations_per_receiver": exp08["n_operations_per_receiver"],
            "false_exhaustion_rate": exp08["false_exhaustion_rate"],
            "correct_exhaustion_rate": exp08["correct_exhaustion_rate"],
            "dichotomy_violations": exp08["dichotomy_violations"],
            "resolved_by_generation_rate": exp08["resolved_by_generation_rate_among_exhausted"],
            "outcome_counts": exp08["outcome_counts"],
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
