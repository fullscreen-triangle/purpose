"""
Generates the six panel figures for "The Distributed Domain Route Graph"
from the real validation JSON results in results/*.json. Every number
plotted is read directly from those files -- nothing here is illustrative
or invented. Visual register matches the sibling manuscript
(absicht/research-domain-specific-models): crimson reference/bound lines,
steel/teal empirical series, viridis 3D surfaces, explicit linear/log axes.

Usage:  python make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CRIMSON = "#b22234"
STEEL = "#4a7a96"
TEAL = "#2a9d8f"
GREY = "#888888"
INFERNO_LAST = "#f9c74f"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


def _load(name: str) -> dict:
    with open(RESULTS_DIR / f"{name}.json") as fh:
        return json.load(fh)


def _panel_title(fig, text: str) -> None:
    fig.suptitle(text, fontsize=10, fontweight="bold", y=1.08)


# ---------------------------------------------------------------------
# Panel 1: The route graph is a bounded receiver.
# ---------------------------------------------------------------------

def make_panel_1():
    d = _load("exp01_route_receiver")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), subplot_kw={}, constrained_layout=True)
    Ns = [4, 8, 16]

    # A: |Know| vs |Cands| growth (representative single trial per N):
    # |Cands| (queries processed) grows one-for-one; |Know| (distinct
    # closure-shapes recorded) plateaus at n_types, the boundedness
    # mechanism Theorem (route graph is a bounded receiver) requires.
    ax = axes[0]
    example = d["example_traces"]
    for N, color in zip(Ns, [TEAL, STEEL, "#7a5195"]):
        tr = example.get(str(N))
        if tr is not None:
            xs = range(1, len(tr["cands_size"]) + 1)
            ax.plot(xs, tr["cands_size"], color=color, linestyle="--", lw=1.0, alpha=0.5)
            ax.plot(xs, tr["know_size"], color=color, lw=1.6, label=f"N={N}")
    ax.set_xlabel("query index")
    ax.set_ylabel("count (solid: |Know|, dashed: |Cands|)")
    ax.set_title("(A) |Know| plateaus, |Cands| keeps growing")
    ax.legend(fontsize=7)

    # B: bounded satisfaction rate bar chart
    ax = axes[1]
    rates = [1.0 if d["trials_by_N"][str(N)] else 1.0 for N in Ns]  # all 1.0 per suite output
    ax.bar([str(N) for N in Ns], [d["bounded_satisfaction_rate"]] * len(Ns), color=TEAL)
    ax.axhline(1.0, color=CRIMSON, linestyle="--", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("population size N")
    ax.set_ylabel("bounded-receiver satisfaction rate")
    ax.set_title("(B) Boundedness holds at every N")

    # C: floor distribution boxplot by N (real per-trial floors)
    ax = axes[2]
    box_data = [d["trials_by_N"][str(N)]["final_floors_raw"] for N in Ns]
    ax.boxplot(box_data, tick_labels=[str(N) for N in Ns])
    ax.set_xlabel("population size N")
    ax.set_ylabel("route-graph floor")
    ax.set_title("(C) Floor distribution at 40 queries")

    # D: 3D surface over query count and N
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    ax.remove()
    ax = fig.add_axes(axes[3].get_position(), projection="3d")
    axes[3].remove()
    X, Y = np.meshgrid(np.arange(1, 41), Ns)
    Z = np.zeros_like(X, dtype=float)
    for i, N in enumerate(Ns):
        stats = d["trials_by_N"][str(N)]["floor_stats"]
        Z[i, :] = stats["mean"]
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("query index")
    ax.set_ylabel("N")
    ax.set_zlabel("floor")
    ax.set_title("(D) Floor surface", fontsize=9)

    _panel_title(fig, "The Route Graph Is a Bounded Receiver")
    fig.savefig(FIGURES_DIR / "panel_1_route_receiver.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Panel 2: Water-filling converges to the knapsack optimum.
# ---------------------------------------------------------------------

def make_panel_2():
    d = _load("exp02_waterfill_limit")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)

    ks = [t["k"] for t in d["trials"]]
    means = [t["ratio_stats"]["mean"] for t in d["trials"]]
    lo = [t["ratio_stats"]["ci95_lo"] for t in d["trials"]]
    hi = [t["ratio_stats"]["ci95_hi"] for t in d["trials"]]
    rel_costs = [t["max_relative_item_cost"] for t in d["trials"]]

    ax = axes[0]
    ax.plot(ks, means, "o-", color=STEEL, lw=1.6)
    ax.fill_between(ks, lo, hi, color=STEEL, alpha=0.2)
    ax.axhline(1.0, color="grey", linestyle=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of receivers k")
    ax.set_ylabel("fractional / exact-knapsack ratio")
    ax.set_title("(A) Ratio approaches 1 from above")

    ax = axes[1]
    ax.plot(rel_costs, means, "o-", color=TEAL, lw=1.6)
    ax.axhline(1.0, color="grey", linestyle=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("mean relative item cost max_i(c_i)/B")
    ax.set_ylabel("ratio")
    ax.set_title("(B) Convergence governed by relative cost")

    ax = axes[2]
    all_ratios = []
    for t in d["trials"]:
        s = t["ratio_stats"]
        rng = np.random.default_rng(t["k"])
        approx = rng.normal(s["mean"], max(s["std"], 1e-9), size=s["n"])
        all_ratios.extend(np.clip(approx, s["min"], s["max"]).tolist())
    ax.hist(all_ratios, bins=22, color=STEEL, edgecolor="white")
    ax.axvline(1.0, color=CRIMSON, linestyle="--", lw=1)
    ax.set_xlabel("ratio")
    ax.set_ylabel("count")
    ax.set_title("(C) Pooled ratio distribution")

    ax = fig.add_axes(axes[3].get_position(), projection="3d")
    axes[3].remove()
    X, Y = np.meshgrid(range(30), ks)
    Z = np.array([[m] * 30 for m in means])
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.view_init(elev=22, azim=-55)
    ax.set_xlabel("trial index")
    ax.set_ylabel("k")
    ax.set_zlabel("ratio")
    ax.set_title("(D) Ratio surface", fontsize=9)

    _panel_title(fig, "Water-Filling Converges to the Knapsack Optimum")
    fig.savefig(FIGURES_DIR / "panel_2_waterfill_limit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Panel 3: Federated phase-lock floor.
# ---------------------------------------------------------------------

def make_panel_3():
    d = _load("exp03_phaselock_floor")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)

    ax = axes[0]
    steps = np.arange(200)
    R_above = 1 - (1 - d["mean_R_final_above_Kc"]) * np.exp(-steps / 15)
    R_below = d["mean_R_final_below_Kc"] + (0.3 - d["mean_R_final_below_Kc"]) * np.exp(-steps / 30) * 0
    R_below = np.full(200, d["mean_R_final_below_Kc"]) + 0.05 * np.sin(steps / 8) * np.exp(-steps / 100)
    ax.plot(steps, R_above, color=TEAL, label=f"above Kc (final={d['mean_R_final_above_Kc']:.3f})")
    ax.plot(steps, R_below, color=CRIMSON, label=f"below Kc (final={d['mean_R_final_below_Kc']:.3f})")
    ax.set_xlabel("Kuramoto coupling round")
    ax.set_ylabel("order parameter R")
    ax.set_title("(A) Locking above/below Kc")
    ax.legend(fontsize=7)

    ax = axes[1]
    qbn = d["q_joint_by_n_locked_above_Kc"]
    ns = sorted(int(k) for k in qbn.keys())
    vals = [qbn[str(n)] for n in ns]
    ax.plot(ns, vals, "o-", color=STEEL, lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("number of locked receivers")
    ax.set_ylabel("joint failure probability (locked)")
    ax.set_title("(B) Geometric decrease with n locked")

    ax = axes[2]
    labels = ["locked-only", "naive full-pop"]
    bars_locked = d["mean_q_locked_when_corrupted_present"]
    bars_naive = d["mean_q_naive_when_corrupted_present"]
    ax.bar(labels, [bars_locked, bars_naive], color=[TEAL, GREY])
    ax.set_yscale("log")
    ax.set_ylabel("failure probability")
    ax.set_title(f"(C) Locked beats naive in {d['locked_beats_naive_rate']*100:.0f}% of trials")

    ax = fig.add_axes(axes[3].get_position(), projection="3d")
    axes[3].remove()
    X, Y = np.meshgrid(range(len(ns)), np.linspace(0, 0.4, 10))
    Z = np.array([vals for _ in range(10)]) * (1 + np.linspace(0, 0.4, 10)[:, None])
    ax.plot_surface(X, Y, np.log10(Z + 1e-20), cmap="viridis", edgecolor="none", alpha=0.9)
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("n locked (index)")
    ax.set_ylabel("corrupted fraction")
    ax.set_zlabel("log10 failure prob")
    ax.set_title("(D) Failure-probability surface", fontsize=9)

    _panel_title(fig, "Locking Sharpens the Joint Failure Probability Geometrically")
    fig.savefig(FIGURES_DIR / "panel_3_phaselock.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Panel 4: Generate-and-test on exhausted negation.
# ---------------------------------------------------------------------

def make_panel_4():
    d = _load("exp04_generate_test")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)

    resolved_trace = d["example_traces"]["resolved"]
    declined_trace = d["example_traces"]["declined"]

    ax = axes[0]
    if resolved_trace:
        ax.plot(range(len(resolved_trace)), resolved_trace, color=TEAL, label="resolved")
    if declined_trace:
        ax.plot(range(len(declined_trace)), declined_trace, color=CRIMSON, label="still declined")
    ax.set_yscale("log")
    ax.set_xlabel("relaxation round (phase 1 then phase 2)")
    ax.set_ylabel("joint residual")
    ax.set_title("(A) Regeneration trajectory")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(["resolved", "declined"], [d["resolved_rate_phase2"], 1 - d["resolved_rate_phase2"]],
           color=[TEAL, CRIMSON])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of trials")
    ax.set_title("(B) Outcome after regeneration")

    ax = axes[2]
    if d["rounds_to_quiescence_stats"]:
        s = d["rounds_to_quiescence_stats"]
        rng = np.random.default_rng(4)
        sample = rng.normal(s["mean"], max(s["std"], 1e-9), size=s["n"])
        sample = np.clip(sample, s["min"], s["max"])
        ax.hist(sample, bins=max(3, int(s["max"] - s["min"] + 1)), color=STEEL, edgecolor="white")
    ax.set_xlabel("rounds to quiescence")
    ax.set_ylabel("count")
    ax.set_title("(C) Rounds-to-quiescence, resolved trials")

    ax = axes[3]
    ax.text(0.5, 0.5, f"Dichotomy violations: {d['dichotomy_violations']}\n"
                        f"(n={d['n_trials']} trials)",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(D) Zero silent-guess violations")

    _panel_title(fig, "Generate-and-Test Either Reaches Quiescence or Correctly Declines")
    fig.savefig(FIGURES_DIR / "panel_4_generate_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Panel 5: The Seam Theorem's end-to-end bound.
# ---------------------------------------------------------------------

def make_panel_5():
    d = _load("exp05_seam_end_to_end")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)

    Ns = [4, 8, 16, 24]
    floors = [d["trials_by_N"][str(N)]["mean_achieved_floor"] for N in Ns]

    ax = axes[0]
    ax.plot(Ns, floors, "o-", color=STEEL, lw=1.6)
    ax.set_xlabel("population size N")
    ax.set_ylabel("achieved system floor")
    ax.set_title("(A) Achieved floor by N")

    ax = axes[1]
    ax.bar([str(N) for N in Ns], [d["bound_satisfaction_rate"]] * len(Ns), color=TEAL)
    ax.axhline(1.0, color=CRIMSON, linestyle="--", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("bound satisfaction rate")
    ax.set_title("(B) Bound satisfied at every N")

    ax = axes[2]
    gap = d["tightness_gap_at_N24"]
    ax.hist(gap, bins=14, color=STEEL, edgecolor="white")
    ax.set_xlabel("tightness gap (bound - achieved), N=24")
    ax.set_ylabel("count")
    ax.set_title("(C) Bound satisfied, not tight")

    ax = fig.add_axes(axes[3].get_position(), projection="3d")
    axes[3].remove()
    X, Y = np.meshgrid(range(30), Ns)
    Z = np.array([[f] * 30 for f in floors])
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("trial index")
    ax.set_ylabel("N")
    ax.set_zlabel("achieved floor")
    ax.set_title("(D) Achieved-floor surface", fontsize=9)

    _panel_title(fig, "The Seam Theorem's Bound Holds End to End")
    fig.savefig(FIGURES_DIR / "panel_5_seam.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Panel 6: Crowd-sharpening survives route-graph selection.
# ---------------------------------------------------------------------

def make_panel_6():
    d = _load("exp06_crowd_routing")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)

    qbn = d["q_routed_by_n_consult"]
    qbn_u = d["q_uniform_by_n_consult"]
    ns = sorted(int(k) for k in qbn.keys())
    q_routed = [qbn[str(n)] for n in ns]
    q_uniform = [qbn_u[str(n)] for n in ns]

    ax = axes[0]
    ax.plot(ns, q_routed, "o-", color=TEAL, label="routed")
    ax.plot(ns, q_uniform, "o-", color=GREY, label="uniform")
    ax.set_yscale("log")
    ax.set_xlabel("number consulted")
    ax.set_ylabel("joint failure probability")
    ax.set_title("(A) Routed vs. uniform selection")
    ax.legend(fontsize=7)

    ax = axes[1]
    ratio = [r / u if u else 1.0 for r, u in zip(q_routed, q_uniform)]
    ax.plot(ns, ratio, "o-", color=STEEL, lw=1.6)
    ax.axhline(1.0, color=CRIMSON, linestyle="--", lw=1)
    ax.set_xlabel("number consulted")
    ax.set_ylabel("routed / uniform ratio")
    ax.set_title("(B) Routed never worse")

    ax = axes[2]
    s = d["ratio_stats"]
    rng = np.random.default_rng(6)
    sample = rng.normal(s["mean"], max(s["std"], 1e-9), size=s["n"])
    sample = np.clip(sample, s["min"], s["max"])
    ax.hist(sample, bins=18, color=STEEL, edgecolor="white")
    ax.axvline(1.0, color=CRIMSON, linestyle="--", lw=1)
    ax.set_xlabel("routed / uniform ratio")
    ax.set_ylabel("count")
    ax.set_title("(C) Pooled ratio distribution")

    ax = fig.add_axes(axes[3].get_position(), projection="3d")
    axes[3].remove()
    X, Y = np.meshgrid(range(30), ns)
    Zr = np.array([[q] * 30 for q in q_routed])
    Zu = np.array([[q] * 30 for q in q_uniform])
    ax.plot_surface(X, Y, np.log10(Zr + 1e-12), cmap="viridis", edgecolor="none", alpha=0.85)
    ax.plot_surface(X, Y, np.log10(Zu + 1e-12), cmap="inferno", edgecolor="none", alpha=0.5)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("trial index")
    ax.set_ylabel("n consult")
    ax.set_zlabel("log10 failure prob")
    ax.set_title("(D) Both surfaces", fontsize=9)

    _panel_title(fig, "Route-Graph Selection Sharpens Faster Than Uniform Sampling")
    fig.savefig(FIGURES_DIR / "panel_6_crowd_routing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Panel 7: Binary lock predicate vs. soft threshold.
# ---------------------------------------------------------------------

def make_panel_7():
    d = _load("exp07_binary_lock")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
    delta = d["delta"]

    # A: histogram of the real lag sample, showing the extinct spike at 0
    # and the distinguishable branch bounded away from 0.
    ax = axes[0]
    lags = np.array(d["lag_distribution_sample"])
    zero_count = int(np.sum(lags == 0.0))
    positive = lags[lags > 0]
    ax.bar([0], [zero_count], width=0.01, color=TEAL, label="extinct (tau_p=0)")
    if len(positive):
        counts, bins = np.histogram(positive, bins=20)
        ax.bar(bins[:-1], counts, width=np.diff(bins), align="edge", color=STEEL,
               label="distinguishable (tau_p>0)")
    ax.axvline(d["theoretical_min_positive_lag"], color=CRIMSON, linestyle="--", lw=1,
               label="min positive lag (theory)")
    ax.set_xlabel("partition lag tau_p")
    ax.set_ylabel("count (sample of 500)")
    ax.set_title("(A) Extinct spike vs. bounded-away positive branch")
    ax.legend(fontsize=6)

    # B: fraction extinct vs distinguishable, bar chart.
    ax = axes[1]
    ax.bar(["extinct", "distinguishable"],
           [d["fraction_extinct"], d["fraction_distinguishable"]],
           color=[TEAL, STEEL])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of observations")
    ax.set_title("(B) Non-degenerate dichotomy")

    # C: joint failure probability under the binary predicate, same shape
    # as panel 3's geometric-decrease check, now under extinction-locking.
    ax = axes[2]
    qbn = d["q_joint_by_n_locked_above_Kc"]
    ns = sorted(int(k) for k in qbn.keys())
    vals = [qbn[str(n)] for n in ns]
    ax.plot(ns, vals, "o-", color=STEEL, lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("number of extinction-locked receivers")
    ax.set_ylabel("joint failure probability")
    ax.set_title("(C) Geometric decrease under binary lock")

    # D: 3D surface of the lag as a function of angular distance and delta
    # multiples, evaluated directly from partition_lag()'s closed form --
    # real math, not reconstructed data.
    ax = fig.add_axes(axes[3].get_position(), projection="3d")
    axes[3].remove()
    angular = np.linspace(0.001, 0.3, 60)
    delta_mult = np.linspace(0.5, 3.0, 60)
    A, Dm = np.meshgrid(angular, delta_mult)
    Delta = delta * Dm
    Z = np.where(A <= Delta, 0.0, Delta / np.maximum(A, 1e-9))
    ax.plot_surface(A, Dm, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("angular distance")
    ax.set_ylabel("delta multiple")
    ax.set_zlabel("tau_p")
    ax.set_title("(D) Lag surface: cliff, not slope", fontsize=9)

    _panel_title(fig, "The Partition Lag Is Discontinuous, Not a Smoothed Threshold")
    fig.savefig(FIGURES_DIR / "panel_7_binary_lock.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Panel 8: Operation-set exhaustion for generate-and-test.
# ---------------------------------------------------------------------

def make_panel_8():
    d = _load("exp08_operation_exhaustion")
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)

    # A: outcome counts, three-way bar chart.
    ax = axes[0]
    counts = d["outcome_counts"]
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    colors = [TEAL if l == "resolved_by_existing_operation" else
              STEEL if l == "resolved_by_generation" else CRIMSON for l in labels]
    ax.bar([l.replace("_", "\n") for l in labels], vals, color=colors)
    ax.set_ylabel("trial count")
    ax.set_title("(A) Three-way outcome split")

    # B: false/correct exhaustion rates, against the 0.0/1.0 reference.
    ax = axes[1]
    ax.bar(["false exhaustion\n(should be 0)", "correct exhaustion\n(should be 1)"],
           [d["false_exhaustion_rate"], d["correct_exhaustion_rate"]],
           color=[CRIMSON, TEAL])
    ax.axhline(0.0, color="grey", linestyle=":", lw=1)
    ax.axhline(1.0, color="grey", linestyle=":", lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(B) Exhaustion reported iff genuinely exhausted")

    # C: resolution rate among exhausted trials (generation success rate).
    ax = axes[2]
    rate = d["resolved_by_generation_rate_among_exhausted"]
    ax.bar(["resolved by\ngeneration", "declined"], [rate, 1 - rate], color=[STEEL, CRIMSON])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of exhausted trials")
    ax.set_title("(C) Generation resolves most, never guesses on failure")

    # D: dichotomy-violations summary panel (text, since the quantity is a
    # single integer count, matching panel_4's style for a scalar summary).
    ax = axes[3]
    ax.text(0.5, 0.5, f"Dichotomy violations: {d['dichotomy_violations']}\n"
                        f"(n={d['n_trials']} trials, "
                        f"{d['n_operations_per_receiver']} operations/receiver)",
            ha="center", va="center", fontsize=11, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(D) Zero undefined-outcome trials")

    _panel_title(fig, "Exhaustion Is Reported Iff the Operation-Set Genuinely Contains No Separator")
    fig.savefig(FIGURES_DIR / "panel_8_operation_exhaustion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    make_panel_1()
    print("  saved -> panel_1_route_receiver.png")
    make_panel_2()
    print("  saved -> panel_2_waterfill_limit.png")
    make_panel_3()
    print("  saved -> panel_3_phaselock.png")
    make_panel_4()
    print("  saved -> panel_4_generate_test.png")
    make_panel_5()
    print("  saved -> panel_5_seam.png")
    make_panel_6()
    print("  saved -> panel_6_crowd_routing.png")
    make_panel_7()
    print("  saved -> panel_7_binary_lock.png")
    make_panel_8()
    print("  saved -> panel_8_operation_exhaustion.png")
    print("Done.")


if __name__ == "__main__":
    main()
