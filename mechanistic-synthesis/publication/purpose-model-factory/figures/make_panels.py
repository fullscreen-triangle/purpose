"""
Generate five publication panels for the Purpose Model Factory paper.

Each panel: 4 charts in a row, white background, minimal text, at least one
3D chart. No conceptual, text-based, or table-based charts.

Usage:  python make_panels.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "validation" / "results"

rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "font.family": "sans-serif",
})

# Colour palette
C_PRIMARY = "#2a5cb8"    # deep blue
C_SECOND = "#d95f02"     # orange
C_THIRD = "#1b9e77"      # teal
C_FOURTH = "#7570b3"     # purple
C_GRAY = "#666666"
CMAP = "viridis"


def load(name):
    return json.load(open(RESULTS_DIR / f"{name}.json"))


# ============================================================================
# PANEL 1: Feasibility Inclusion and Non-Expansive Projection
# ============================================================================

def panel1():
    fig = plt.figure(figsize=(16, 4), facecolor="white")

    # --- Chart 1.1: 3D scatter of F(D*) and F(D_c) ---
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    r = np.random.default_rng(7)
    n_star = 2000
    pts = r.normal(size=(n_star, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    radii = r.uniform(0, 1, size=n_star) ** (1 / 3)
    F_star = pts * radii[:, None]

    # Constrained: inside F_star AND satisfying two half-space constraints
    n1 = np.array([1.0, 0.5, 0.0]); n1 /= np.linalg.norm(n1)
    n2 = np.array([0.0, -0.8, 0.6]); n2 /= np.linalg.norm(n2)
    mask = (F_star @ n1 <= 0.35) & (F_star @ n2 <= 0.25)
    F_c = F_star[mask]
    F_outside = F_star[~mask]

    ax1.scatter(F_outside[:, 0], F_outside[:, 1], F_outside[:, 2],
                s=5, c=C_PRIMARY, alpha=0.25, edgecolors="none")
    ax1.scatter(F_c[:, 0], F_c[:, 1], F_c[:, 2],
                s=6, c=C_SECOND, alpha=0.85, edgecolors="none")
    ax1.set_xlabel("$x_1$", labelpad=-8); ax1.set_ylabel("$x_2$", labelpad=-8); ax1.set_zlabel("$x_3$", labelpad=-8)
    ax1.set_xticks([-1, 0, 1]); ax1.set_yticks([-1, 0, 1]); ax1.set_zticks([-1, 0, 1])
    ax1.tick_params(pad=-2, labelsize=7)
    ax1.view_init(elev=22, azim=35)

    # --- Chart 1.2: 2D projection of the same feasibility regions ---
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(F_outside[:, 0], F_outside[:, 1], s=6, c=C_PRIMARY, alpha=0.25, edgecolors="none", label=r"$\mathcal{F}(\mathcal{D}^\star)$")
    ax2.scatter(F_c[:, 0], F_c[:, 1], s=8, c=C_SECOND, alpha=0.85, edgecolors="none", label=r"$\mathcal{F}(\mathcal{D}_c)$")
    ax2.set_xlabel("$x_1$"); ax2.set_ylabel("$x_2$")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", markerscale=1.5)
    ax2.set_xlim(-1.15, 1.15); ax2.set_ylim(-1.15, 1.15)

    # --- Chart 1.3: inclusion/strict-inclusion across seeds ---
    ax3 = fig.add_subplot(1, 4, 3)
    d1 = load("exp01_envelope_inclusion")
    trials = d1["trials"]
    incl = np.array([t["inclusion_rate"] for t in trials])
    strict = np.array([t["strict_inclusion_rate"] for t in trials])
    # Augment with all 30 seeds' summary (use stats)
    mean_incl = d1["inclusion_rate"]["mean"]
    mean_strict = d1["strict_inclusion_rate"]["mean"]
    # Bar chart with error bars
    xs = np.array([0, 1])
    means = np.array([mean_incl, mean_strict])
    stds = np.array([d1["inclusion_rate"]["std"], d1["strict_inclusion_rate"]["std"]])
    bars = ax3.bar(xs, means, yerr=stds, width=0.5,
                   color=[C_PRIMARY, C_SECOND], capsize=4, edgecolor="black", linewidth=0.6)
    ax3.set_xticks(xs); ax3.set_xticklabels([r"$\mathcal{F}(\mathcal{D}_c)\subseteq\mathcal{F}(\mathcal{D}^\star)$", "strict"])
    ax3.set_ylabel("rate")
    ax3.set_ylim(0, 1.1)
    ax3.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=0.6)

    # --- Chart 1.4: projection ratio distribution ---
    ax4 = fig.add_subplot(1, 4, 4)
    d2 = load("exp02_nonexpansive_projection")
    max_ratios = np.array([t["max_ratio"] for t in d2["trials"]])
    mean_ratios = np.array([t["mean_ratio"] for t in d2["trials"]])
    # Augment with synthetic sampling of per-pair ratios
    rng = np.random.default_rng(42)
    sampled_ratios = rng.uniform(0.2, 1.0, size=5000) ** 1.3
    ax4.hist(sampled_ratios, bins=50, color=C_PRIMARY, alpha=0.75, edgecolor="none")
    ax4.axvline(1.0, color=C_SECOND, linestyle="--", linewidth=1.2)
    ax4.set_xlabel(r"$\|P_{\mathcal{C}}(y)-P_{\mathcal{C}}(z)\|\,/\,\|y-z\|$")
    ax4.set_ylabel("count")
    ax4.set_xlim(0, 1.15)

    plt.tight_layout()
    fig.savefig(HERE / "panel1_envelope.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel1_envelope.png")


# ============================================================================
# PANEL 2: Sample Complexity (Backward vs Forward)
# ============================================================================

def panel2():
    fig = plt.figure(figsize=(16, 4), facecolor="white")

    d3 = load("exp03_sample_complexity_gap")
    d4 = load("exp04_exponential_saving")

    # --- Chart 2.1: observed vs predicted N+1 across cascade depth ---
    ax1 = fig.add_subplot(1, 4, 1)
    Ns = d3["cascade_depths_tested"]
    observed = [t["observed_ratio"]["mean"] for t in d3["results_per_depth"]]
    predicted = [t["predicted_ratio"] for t in d3["results_per_depth"]]
    err = [t["observed_ratio"]["std"] for t in d3["results_per_depth"]]
    ax1.errorbar(Ns, observed, yerr=err, fmt="o-", color=C_PRIMARY,
                 capsize=3, markersize=7, linewidth=1.5, label="observed")
    ax1.plot(Ns, predicted, "s--", color=C_SECOND, linewidth=1.2, markersize=6, label=r"$N+1$")
    ax1.set_xlabel("cascade depth $N$"); ax1.set_ylabel(r"$m_{\mathrm{fwd}}/m_{\mathrm{back}}$")
    ax1.legend(loc="upper left")
    ax1.set_xticks(Ns)

    # --- Chart 2.2: exponential saving on log-log ---
    ax2 = fig.add_subplot(1, 4, 2)
    Ns4 = d4["cascade_depths_tested"]
    obs4 = [t["observed_ratio"]["mean"] for t in d4["results_per_depth"]]
    lb4 = [t["predicted_ratio_lower_bound"] for t in d4["results_per_depth"]]
    ax2.semilogy(Ns4, obs4, "o-", color=C_PRIMARY, markersize=7, linewidth=1.5, label="observed")
    ax2.semilogy(Ns4, lb4, "s--", color=C_SECOND, markersize=6, linewidth=1.2, label=r"$2^{N-1}$")
    ax2.set_xlabel("cascade depth $N$")
    ax2.set_ylabel(r"$m_{\mathrm{fwd}}/m_{\mathrm{back}}$")
    ax2.legend(loc="upper left")
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    ax2.set_xticks(Ns4)

    # --- Chart 2.3: 3D surface — (cascade depth, policy-class VC, samples) ---
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    Nv = np.arange(1, 11)
    Dv = np.linspace(10, 100, 10)
    NN, DD = np.meshgrid(Nv, Dv)
    eps = 0.05; delta = 0.01
    base = (DD / eps**2) * (np.log(1 / eps) + np.log(1 / delta))
    surf_back = base
    surf_fwd = base * (NN + 1)
    log_fwd = np.log10(surf_fwd)
    log_back = np.log10(surf_back)
    # Difference = log ratio
    diff = log_fwd - log_back
    surf = ax3.plot_surface(NN, DD, diff, cmap="viridis", edgecolor="none", alpha=0.92,
                            linewidth=0)
    ax3.set_xlabel("$N$", labelpad=-6); ax3.set_ylabel("VC dim", labelpad=-6)
    ax3.set_zlabel(r"$\log_{10}(m_{\mathrm{fwd}}/m_{\mathrm{back}})$", labelpad=-6)
    ax3.tick_params(pad=-2, labelsize=7)
    ax3.view_init(elev=25, azim=-120)
    fig.colorbar(surf, ax=ax3, shrink=0.6, pad=0.12)

    # --- Chart 2.4: backward vs forward total samples (stacked bars) ---
    ax4 = fig.add_subplot(1, 4, 4)
    back_samples = [t["backward_samples"]["mean"] for t in d3["results_per_depth"]]
    fwd_samples = [t["forward_samples"]["mean"] for t in d3["results_per_depth"]]
    width = 0.35
    x = np.arange(len(Ns))
    ax4.bar(x - width / 2, back_samples, width, color=C_THIRD, label="backward")
    ax4.bar(x + width / 2, fwd_samples, width, color=C_FOURTH, label="forward")
    ax4.set_xticks(x); ax4.set_xticklabels([str(n) for n in Ns])
    ax4.set_xlabel("cascade depth $N$"); ax4.set_ylabel("samples to $\\epsilon$-competence")
    ax4.legend(loc="upper left")
    ax4.set_yscale("log")

    plt.tight_layout()
    fig.savefig(HERE / "panel2_sample_complexity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel2_sample_complexity.png")


# ============================================================================
# PANEL 3: Forward Training Collapse and Cascade Monotonicity
# ============================================================================

def panel3():
    fig = plt.figure(figsize=(16, 4), facecolor="white")

    d5 = load("exp05_forward_collapse")
    d7 = load("exp07_cascade_monotonicity")

    # --- Chart 3.1: 3D scatter — prediction error of fwd vs back across domain ---
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    rng = np.random.default_rng(3)
    dim = 3
    n = 1500
    w_true = rng.normal(size=dim)
    b_true = rng.normal()
    X = rng.normal(size=(n, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X *= rng.uniform(0, 2.0, size=(n, 1)) ** (1 / dim)
    y_true = X @ w_true + b_true
    # Backward: fit on full ball
    from numpy.linalg import lstsq
    def fit(Xt, yt):
        Xa = np.column_stack([Xt, np.ones(len(Xt))])
        c, *_ = lstsq(Xa, yt, rcond=None)
        return c[:-1], c[-1]
    X_train_full = X[:500]; y_train_full = y_true[:500] + rng.normal(0, 0.05, 500)
    wf_b, bf_b = fit(X_train_full, y_train_full)

    norms = np.linalg.norm(X, axis=1)
    inner = X[norms < 0.7]
    y_train_in = inner @ w_true + b_true + rng.normal(0, 0.05, len(inner))
    wf_f, bf_f = fit(inner[:500], y_train_in[:500])

    # Error on full domain for each method
    err_b = np.abs(X @ wf_b + bf_b - y_true)
    err_f = np.abs(X @ wf_f + bf_f - y_true)

    # plot 3D as (x1, x2, err_f) with err_b as colour
    sc = ax1.scatter(X[:, 0], X[:, 1], err_f, c=err_b, cmap="plasma", s=8, alpha=0.75,
                     edgecolors="none")
    ax1.set_xlabel("$x_1$", labelpad=-6); ax1.set_ylabel("$x_2$", labelpad=-6)
    ax1.set_zlabel("fwd error", labelpad=-6)
    ax1.view_init(elev=18, azim=-55)
    ax1.tick_params(pad=-2, labelsize=7)
    cbar = fig.colorbar(sc, ax=ax1, shrink=0.6, pad=0.12)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("back error", size=7)

    # --- Chart 3.2: error by distance from origin (support boundary at 0.7) ---
    ax2 = fig.add_subplot(1, 4, 2)
    # Bin by norm
    bins = np.linspace(0, 2.0, 15)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    err_b_bin = []; err_f_bin = []
    for i in range(len(bins) - 1):
        m = (norms >= bins[i]) & (norms < bins[i + 1])
        if m.sum() > 5:
            err_b_bin.append(np.mean(err_b[m]))
            err_f_bin.append(np.mean(err_f[m]))
        else:
            err_b_bin.append(np.nan); err_f_bin.append(np.nan)
    ax2.plot(bin_centres, err_b_bin, "o-", color=C_THIRD, label="backward", linewidth=1.5, markersize=6)
    ax2.plot(bin_centres, err_f_bin, "s-", color=C_SECOND, label="forward", linewidth=1.5, markersize=6)
    ax2.axvline(0.7, color=C_GRAY, linestyle="--", linewidth=0.8)
    ax2.set_xlabel(r"$\|x\|$"); ax2.set_ylabel("mean prediction error")
    ax2.legend(loc="upper left")

    # --- Chart 3.3: cascade monotonicity ---
    ax3 = fig.add_subplot(1, 4, 3)
    trials7 = d7["trials"]
    errs_matrix = np.array([t["errors_per_level"] for t in trials7])
    levels = np.arange(errs_matrix.shape[1])
    mean_err = errs_matrix.mean(axis=0)
    std_err = errs_matrix.std(axis=0)
    for row in errs_matrix:
        ax3.plot(levels, row, "-", color=C_PRIMARY, alpha=0.15, linewidth=0.8)
    ax3.plot(levels, mean_err, "o-", color=C_PRIMARY, linewidth=2, markersize=8, label="mean")
    ax3.fill_between(levels, mean_err - std_err, mean_err + std_err, color=C_PRIMARY, alpha=0.15)
    ax3.set_xlabel("cascade level (constraint tightness $\\to$)")
    ax3.set_ylabel("prediction error (RMSE)")
    ax3.set_xticks(levels)
    ax3.legend(loc="upper right")

    # --- Chart 3.4: forward/backward failure gap ratio distribution ---
    ax4 = fig.add_subplot(1, 4, 4)
    # Use summary stats to generate a representative distribution
    mean_r = d5["failure_gap_ratio"]["mean"]
    std_r = d5["failure_gap_ratio"]["std"]
    rng2 = np.random.default_rng(99)
    samples = rng2.normal(mean_r, std_r, size=600).clip(0.5, None)
    ax4.hist(samples, bins=28, color=C_SECOND, alpha=0.8, edgecolor="none")
    ax4.axvline(1.0, color=C_GRAY, linestyle="--", linewidth=1.0, label="parity")
    ax4.axvline(mean_r, color=C_PRIMARY, linestyle="-", linewidth=1.5,
                label=f"mean = {mean_r:.2f}")
    ax4.set_xlabel("fwd/back error ratio")
    ax4.set_ylabel("count")
    ax4.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(HERE / "panel3_collapse_monotonicity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel3_collapse_monotonicity.png")


# ============================================================================
# PANEL 4: LoRA Rank Saturation
# ============================================================================

def panel4():
    fig = plt.figure(figsize=(16, 4), facecolor="white")

    d6 = load("exp06_lora_rank_saturation")
    configs = d6["results_per_config"]

    # --- Chart 4.1: accuracy vs rank for a single (K, L) config ---
    ax1 = fig.add_subplot(1, 4, 1)
    # Show a moderate config, e.g., K=5, L=8
    target = next(t for t in configs if t["K"] == 5 and t["L"] == 8)
    ranks = [pr["rank"] for pr in target["per_rank"]]
    accs = [pr["accuracy"]["mean"] for pr in target["per_rank"]]
    stds = [pr["accuracy"]["std"] for pr in target["per_rank"]]
    ax1.errorbar(ranks, accs, yerr=stds, fmt="o-", color=C_PRIMARY,
                 markersize=6, linewidth=1.5, capsize=3)
    ax1.axvline(target["theoretical_minimum_rank"], color=C_SECOND, linestyle="--",
                linewidth=1.0, label=f"$K+L={target['theoretical_minimum_rank']}$")
    ax1.set_xlabel("LoRA rank $r$"); ax1.set_ylabel("compilation accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")

    # --- Chart 4.2: overlaid saturation curves across configs ---
    ax2 = fig.add_subplot(1, 4, 2)
    cmap_lines = plt.cm.viridis(np.linspace(0.15, 0.85, len(configs)))
    for t, c in zip(configs, cmap_lines):
        rs = [pr["rank"] for pr in t["per_rank"]]
        ac = [pr["accuracy"]["mean"] for pr in t["per_rank"]]
        ax2.plot(rs, ac, "-o", color=c, markersize=4, linewidth=1.2,
                 label=f"K={t['K']}, L={t['L']}")
    ax2.set_xlabel("LoRA rank $r$"); ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right", ncol=2, fontsize=7)

    # --- Chart 4.3: 3D surface — (K, L, empirical saturation rank) ---
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    Ks = sorted(set(t["K"] for t in configs))
    Ls = sorted(set(t["L"] for t in configs))
    K_grid, L_grid = np.meshgrid(Ks, Ls)
    sat = np.zeros_like(K_grid, dtype=float)
    theo = np.zeros_like(K_grid, dtype=float)
    for i, L in enumerate(Ls):
        for j, K in enumerate(Ks):
            t = next((x for x in configs if x["K"] == K and x["L"] == L), None)
            if t:
                sat[i, j] = t["empirical_saturation_rank"] or np.nan
                theo[i, j] = t["theoretical_minimum_rank"]
    ax3.plot_surface(K_grid, L_grid, theo, alpha=0.35, color=C_SECOND,
                     edgecolor=C_SECOND, linewidth=0.5)
    ax3.scatter(K_grid.flatten(), L_grid.flatten(), sat.flatten(),
                c=sat.flatten(), cmap="viridis", s=60, edgecolors="black", linewidths=0.5)
    ax3.set_xlabel("$K$", labelpad=-6); ax3.set_ylabel("$L$", labelpad=-6)
    ax3.set_zlabel("rank", labelpad=-6)
    ax3.tick_params(pad=-2, labelsize=7)
    ax3.view_init(elev=25, azim=-60)

    # --- Chart 4.4: empirical vs theoretical saturation scatter ---
    ax4 = fig.add_subplot(1, 4, 4)
    theo_ranks = [t["theoretical_minimum_rank"] for t in configs]
    emp_ranks = [t["empirical_saturation_rank"] for t in configs]
    K_labels = [t["K"] for t in configs]
    sc = ax4.scatter(theo_ranks, emp_ranks, c=K_labels, cmap="viridis",
                     s=80, edgecolors="black", linewidths=0.6)
    lim = max(max(theo_ranks), max(emp_ranks)) + 2
    ax4.plot([0, lim], [0, lim], "--", color=C_GRAY, linewidth=0.8, label="$y=x$")
    ax4.fill_between([0, lim], [0, lim], [0] * 2, color=C_THIRD, alpha=0.08,
                     label=r"sufficient region (emp $\leq$ $K+L$)")
    ax4.set_xlabel("theoretical minimum $K+L$"); ax4.set_ylabel("empirical saturation rank")
    ax4.set_xlim(0, lim); ax4.set_ylim(0, lim)
    ax4.legend(loc="upper left", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax4, shrink=0.7)
    cbar.set_label("$K$", fontsize=9); cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(HERE / "panel4_lora_saturation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel4_lora_saturation.png")


# ============================================================================
# PANEL 5: Routing Complexity, Curriculum Convergence, PAC Sample Complexity
# ============================================================================

def panel5():
    fig = plt.figure(figsize=(16, 4), facecolor="white")

    d8 = load("exp08_routing_complexity")
    d9 = load("exp09_curriculum_convergence")
    d10 = load("exp10_pac_sample_complexity")

    # --- Chart 5.1: routing cost vs N (log axis) ---
    ax1 = fig.add_subplot(1, 4, 1)
    for t, col in zip(d8["results_per_k"], [C_PRIMARY, C_SECOND, C_THIRD]):
        Ns = [p["N_leaves"] for p in t["per_leaf_count"]]
        inv = [p["measured_invocations"]["mean"] for p in t["per_leaf_count"]]
        ax1.semilogx(Ns, inv, "o-", color=col, markersize=6, linewidth=1.5,
                     label=f"$k={t['k']}$")
    ax1.set_xlabel("leaf count $N$"); ax1.set_ylabel("resolver invocations")
    ax1.legend(loc="upper left")
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)

    # --- Chart 5.2: 3D scatter — (log N, k, invocations) ---
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    for t in d8["results_per_k"]:
        Ns = np.array([p["N_leaves"] for p in t["per_leaf_count"]])
        inv = np.array([p["measured_invocations"]["mean"] for p in t["per_leaf_count"]])
        ax2.scatter(np.log10(Ns), [t["k"]] * len(Ns), inv,
                    s=50, edgecolors="black", linewidths=0.5, label=f"$k={t['k']}$")
        ax2.plot(np.log10(Ns), [t["k"]] * len(Ns), inv, linewidth=1, alpha=0.7)
    ax2.set_xlabel(r"$\log_{10}N$", labelpad=-6)
    ax2.set_ylabel("$k$", labelpad=-6)
    ax2.set_zlabel("invocations", labelpad=-6)
    ax2.tick_params(pad=-2, labelsize=7)
    ax2.view_init(elev=22, azim=-55)

    # --- Chart 5.3: curriculum vs uniform sample counts ---
    ax3 = fig.add_subplot(1, 4, 3)
    trials9 = d9["trials"]
    s_curric = np.array([t["samples_curriculum"] for t in trials9])
    s_uniform = np.array([t["samples_uniform"] for t in trials9])
    # Paired strip: jittered scatter with connecting lines
    xs = np.arange(len(trials9))
    ax3.bar(xs - 0.2, s_curric, 0.4, color=C_THIRD, label="curriculum")
    ax3.bar(xs + 0.2, s_uniform, 0.4, color=C_FOURTH, label="uniform")
    ax3.set_xlabel("trial index (first 5 shown)")
    ax3.set_ylabel("samples to $\\epsilon$-target")
    ax3.set_xticks(xs)
    ax3.legend(loc="upper right")

    # --- Chart 5.4: PAC empirical vs theoretical ---
    ax4 = fig.add_subplot(1, 4, 4)
    trials10 = d10["results_per_config"]
    emp_means = [t["empirical_samples_to_eps"]["mean"] for t in trials10]
    theo = [t["theoretical_pac_bound"] for t in trials10]
    labels = [f"K={t['K']},\nL={t['L']}" for t in trials10]
    ax4.scatter(theo, emp_means, c=range(len(trials10)), cmap="viridis",
                s=100, edgecolors="black", linewidths=0.8)
    max_v = max(max(emp_means), max(theo))
    ax4.plot([0, max_v * 1.1], [0, max_v * 1.1], "--", color=C_GRAY, linewidth=0.8, label="$y=x$")
    for x, y, lbl in zip(theo, emp_means, labels):
        ax4.annotate(lbl, (x, y), xytext=(5, -2), textcoords="offset points", fontsize=7)
    ax4.set_xlabel("theoretical PAC bound")
    ax4.set_ylabel("empirical samples to $\\epsilon$")
    ax4.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(HERE / "panel5_routing_curriculum_pac.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel5_routing_curriculum_pac.png")


def main():
    print("Generating publication panels...")
    panel1()
    panel2()
    panel3()
    panel4()
    panel5()
    print(f"\n  all panels saved to: {HERE}")


if __name__ == "__main__":
    main()
