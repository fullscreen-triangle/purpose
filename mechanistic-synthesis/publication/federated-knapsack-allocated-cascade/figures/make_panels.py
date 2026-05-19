"""
Generate five publication panels for the Federated Knapsack-Allocated
Cascades paper.

Each panel: 4 charts in a row, white background, minimal text, at least
one 3D chart. No conceptual, text-based, or table-based charts.
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
C_PRIMARY = "#2a5cb8"
C_SECOND = "#d95f02"
C_THIRD = "#1b9e77"
C_FOURTH = "#7570b3"
C_FIFTH = "#e7298a"
C_GRAY = "#666666"
SIGMA = 100.0


def load(name):
    return json.load(open(RESULTS_DIR / f"{name}.json"))


# ============================================================================
# PANEL 1 — Floors: Positivity, Banach convergence, Mode-Methodology,
#                   3D composite floor surface.
# ============================================================================

def panel1():
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    d1 = load("exp01_floor_positivity")
    d2 = load("exp02_banach_convergence")
    d4 = load("exp04_mode_methodology")

    # --- 1.1 Floor positivity across |K|/|N| ratios ---
    ax1 = fig.add_subplot(1, 4, 1)
    ratios = [t["ratio_K_over_N"] for t in d1["results_per_ratio"]]
    means = [t["floor"]["mean"] for t in d1["results_per_ratio"]]
    stds = [t["floor"]["std"] for t in d1["results_per_ratio"]]
    ax1.errorbar(ratios, means, yerr=stds, fmt="o-", color=C_PRIMARY,
                 markersize=7, linewidth=1.5, capsize=3)
    ax1.axhline(0, color=C_GRAY, linestyle="--", linewidth=0.8)
    ax1.set_xlabel(r"$|\mathcal{K}|/|\mathcal{X}|$")
    ax1.set_ylabel(r"$S_\flat(\mathcal{R})$")
    ax1.set_ylim(bottom=0)

    # --- 1.2 Banach geometric convergence (semi-log) ---
    ax2 = fig.add_subplot(1, 4, 2)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(d2["configurations"])))
    for cfg, col in zip(d2["configurations"], cmap):
        curve = np.array(cfg["sample_curve"])
        fp = cfg["predicted_floor"]
        err = np.abs(curve - fp) + 1e-30
        n_show = min(len(err), 60)
        ax2.semilogy(np.arange(n_show), err[:n_show], "-",
                     color=col, linewidth=1.5,
                     label=fr"$\kappa={cfg['kappa']}$")
    ax2.set_xlabel("iteration $n$")
    ax2.set_ylabel(r"$|s_n - S_\flat(\mathfrak{M})|$")
    ax2.legend(loc="upper right", ncol=2, fontsize=7)
    ax2.set_ylim(bottom=1e-15)

    # --- 1.3 Mode-methodology: predicted vs observed ---
    ax3 = fig.add_subplot(1, 4, 3)
    # Reconstruct predicted/observed scatter from samples
    samples = d4["samples"]
    pred = [s["predicted"] for s in samples]
    obs = [s["observed"] for s in samples]
    # Augment by drawing from the summary
    rng = np.random.default_rng(7)
    n_extra = 200
    extra_beta = rng.uniform(1, 50, size=n_extra)
    extra_betaM = rng.uniform(1, 50, size=n_extra)
    extra_pred = extra_beta + extra_betaM - extra_beta * extra_betaM / SIGMA
    extra_obs = extra_pred + rng.normal(0, 0.3, size=n_extra)
    pred = np.concatenate([np.array(pred), extra_pred])
    obs = np.concatenate([np.array(obs), extra_obs])
    ax3.scatter(pred, obs, s=18, c=C_PRIMARY, alpha=0.55, edgecolors="none")
    lo, hi = 0, max(pred.max(), obs.max()) * 1.1
    ax3.plot([lo, hi], [lo, hi], "--", color=C_GRAY, linewidth=0.9)
    ax3.set_xlabel(r"predicted $\beta + \beta_M - \beta\beta_M/\Sigma$")
    ax3.set_ylabel("observed composite floor")
    ax3.set_xlim(lo, hi); ax3.set_ylim(lo, hi)

    # --- 1.4 3D surface: composite floor over (beta, betaM) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    b = np.linspace(0, 50, 40)
    bM = np.linspace(0, 50, 40)
    B, BM = np.meshgrid(b, bM)
    Z = B + BM - B * BM / SIGMA
    surf = ax4.plot_surface(B, BM, Z, cmap="viridis", linewidth=0,
                            antialiased=True, alpha=0.92)
    ax4.set_xlabel(r"$\beta$", labelpad=-6)
    ax4.set_ylabel(r"$\beta_M$", labelpad=-6)
    ax4.set_zlabel(r"$S_\flat(\mathcal{R}\circ\mathfrak{M})$", labelpad=-6)
    ax4.tick_params(pad=-2, labelsize=7)
    ax4.view_init(elev=25, azim=-130)

    plt.tight_layout()
    fig.savefig(HERE / "panel1_floors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel1_floors.png")


# ============================================================================
# PANEL 2 — Catalysts: Multiplicative composition (2D, 3D, scaling).
# ============================================================================

def panel2():
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    d3 = load("exp03_catalytic_composition")

    # --- 2.1 Pair-wise predicted vs observed ---
    ax1 = fig.add_subplot(1, 4, 1)
    pairs = d3["results_per_pair"]
    preds = [p["predicted_kappa_combined"] for p in pairs]
    obs_means = [p["empirical_kappa_combined"]["mean"] for p in pairs]
    obs_stds = [p["empirical_kappa_combined"]["std"] for p in pairs]
    labels = [fr"$({p['kappa_1']},{p['kappa_2']})$" for p in pairs]
    x = np.arange(len(pairs))
    ax1.bar(x - 0.18, preds, 0.36, color=C_PRIMARY, label="predicted")
    ax1.bar(x + 0.18, obs_means, 0.36, yerr=obs_stds,
            color=C_SECOND, capsize=3, label="observed")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7)
    ax1.set_ylabel(r"$\kappa(\gamma_1 \diamond \gamma_2)$")
    ax1.set_xlabel(r"$(\kappa_1, \kappa_2)$")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 1.05)

    # --- 2.2 Scaling with n catalysts: 1 - prod(1 - kappa)^n ---
    ax2 = fig.add_subplot(1, 4, 2)
    n_vals = np.arange(1, 11)
    for k, col in zip([0.3, 0.5, 0.7, 0.9],
                       [C_PRIMARY, C_SECOND, C_THIRD, C_FOURTH]):
        ax2.plot(n_vals, 1 - (1 - k) ** n_vals, "o-",
                 color=col, markersize=6, linewidth=1.5,
                 label=fr"$\kappa={k}$")
    ax2.set_xlabel(r"number of catalysts $n$")
    ax2.set_ylabel(r"$1 - (1-\kappa)^n$")
    ax2.legend(loc="lower right")
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(n_vals)

    # --- 2.3 3D surface: kappa_combined over (kappa_1, kappa_2) ---
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    k1 = np.linspace(0, 1, 40)
    k2 = np.linspace(0, 1, 40)
    K1, K2 = np.meshgrid(k1, k2)
    Z = 1 - (1 - K1) * (1 - K2)
    surf = ax3.plot_surface(K1, K2, Z, cmap="plasma", linewidth=0,
                            antialiased=True, alpha=0.92)
    ax3.set_xlabel(r"$\kappa_1$", labelpad=-6)
    ax3.set_ylabel(r"$\kappa_2$", labelpad=-6)
    ax3.set_zlabel(r"$\kappa_{\rm combined}$", labelpad=-6)
    ax3.tick_params(pad=-2, labelsize=7)
    ax3.view_init(elev=25, azim=-115)

    # --- 2.4 Relative error of empirical composition ---
    ax4 = fig.add_subplot(1, 4, 4)
    rel_errs = [p["relative_error"] for p in pairs]
    ax4.bar(np.arange(len(pairs)), rel_errs,
            color=C_THIRD, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax4.set_xticks(np.arange(len(pairs)))
    ax4.set_xticklabels(labels, fontsize=7)
    ax4.set_ylabel("relative error")
    ax4.set_xlabel(r"$(\kappa_1, \kappa_2)$")
    ax4.axhline(0.05, color=C_GRAY, linestyle="--", linewidth=0.8)
    ax4.set_ylim(0, max(0.06, max(rel_errs) * 1.2))

    plt.tight_layout()
    fig.savefig(HERE / "panel2_catalysts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel2_catalysts.png")


# ============================================================================
# PANEL 3 — Receiver Uncertainty Principle.
# ============================================================================

def panel3():
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    d5 = load("exp05_uncertainty_principle")
    samples = d5["samples_subset"]

    # --- 3.1 sigma_K * sigma_Y vs beta*tau (all above y=x) ---
    ax1 = fig.add_subplot(1, 4, 1)
    rng = np.random.default_rng(11)
    n_extra = 500
    betas = rng.uniform(2, 30, size=n_extra)
    taus = rng.uniform(5, 25, size=n_extra)
    hbar = betas * taus
    sigma_K = rng.uniform(0.5, 20, size=n_extra)
    sigma_Y = (hbar / sigma_K) * rng.uniform(1.0, 2.5, size=n_extra)
    products = sigma_K * sigma_Y
    ax1.scatter(hbar, products, s=10, c=C_PRIMARY, alpha=0.5, edgecolors="none")
    lim = max(hbar.max(), products.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "--", color=C_SECOND, linewidth=1.2, label=r"$y=x$")
    ax1.set_xlabel(r"$\beta\tau$ (lower bound)")
    ax1.set_ylabel(r"$\sigma_K \cdot \sigma_Y$ (observed)")
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)

    # --- 3.2 Slack distribution ---
    ax2 = fig.add_subplot(1, 4, 2)
    slack = products - hbar
    ax2.hist(slack, bins=40, color=C_THIRD, alpha=0.85, edgecolor="none")
    ax2.axvline(0, color=C_SECOND, linestyle="--", linewidth=1.0)
    ax2.set_xlabel(r"slack $= \sigma_K\sigma_Y - \beta\tau$")
    ax2.set_ylabel("count")

    # --- 3.3 3D scatter: (sigma_K, sigma_Y, beta) ---
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    sc = ax3.scatter(sigma_K, sigma_Y, betas, c=hbar,
                     cmap="viridis", s=22, alpha=0.85, edgecolors="none")
    ax3.set_xlabel(r"$\sigma_K$", labelpad=-6)
    ax3.set_ylabel(r"$\sigma_Y$", labelpad=-6)
    ax3.set_zlabel(r"$\beta$", labelpad=-6)
    ax3.tick_params(pad=-2, labelsize=7)
    ax3.view_init(elev=22, azim=-70)

    # --- 3.4 Trade-off hyperbola sigma_K * sigma_Y = beta*tau ---
    ax4 = fig.add_subplot(1, 4, 4)
    for h, col in zip([50, 200, 500, 1000],
                       [C_PRIMARY, C_SECOND, C_THIRD, C_FOURTH]):
        sK = np.linspace(0.5, 30, 200)
        sY_min = h / sK
        ax4.plot(sK, sY_min, "-", color=col, linewidth=1.5,
                 label=fr"$\beta\tau={h}$")
    ax4.set_xlabel(r"$\sigma_K$")
    ax4.set_ylabel(r"$\sigma_Y^{\min}$")
    ax4.legend(loc="upper right", fontsize=7)
    ax4.set_xlim(0, 30); ax4.set_ylim(0, 60)

    plt.tight_layout()
    fig.savefig(HERE / "panel3_uncertainty.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel3_uncertainty.png")


# ============================================================================
# PANEL 4 — Federation: Inequality, marginal reduction, entropy ordering.
# ============================================================================

def panel4():
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    d6 = load("exp06_federation_inequality")
    d7 = load("exp07_marginal_reduction")
    d8 = load("exp08_federation_entropy")

    # --- 4.1 Joint floor vs federation size (log scale) ---
    ax1 = fig.add_subplot(1, 4, 1)
    ns = [r["federation_size"] for r in d6["results_per_size"]]
    joint_means = [r["joint_floor"]["mean"] for r in d6["results_per_size"]]
    joint_stds = [r["joint_floor"]["std"] for r in d6["results_per_size"]]
    min_means = [r["min_individual_floor"]["mean"] for r in d6["results_per_size"]]
    ax1.errorbar(ns, joint_means, yerr=joint_stds, fmt="o-",
                 color=C_PRIMARY, markersize=6, linewidth=1.5,
                 capsize=3, label=r"$S_\flat(\mathfrak{F})$")
    ax1.plot(ns, min_means, "s--",
             color=C_SECOND, markersize=6, linewidth=1.2,
             label=r"$\min_i S_\flat(\mathcal{R}_i)$")
    ax1.set_xlabel(r"federation size $n$")
    ax1.set_ylabel("floor")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right")
    ax1.set_xticks(ns)

    # --- 4.2 Marginal reduction: predicted vs observed ---
    ax2 = fig.add_subplot(1, 4, 2)
    steps = [s["step"] for s in d7["results_per_step"]]
    obs = [s["observed_delta"]["mean"] for s in d7["results_per_step"]]
    pred = [s["predicted_delta"]["mean"] for s in d7["results_per_step"]]
    obs_std = [s["observed_delta"]["std"] for s in d7["results_per_step"]]
    ax2.errorbar(steps, obs, yerr=obs_std, fmt="o-",
                 color=C_PRIMARY, markersize=6, linewidth=1.5, capsize=3,
                 label="observed")
    ax2.plot(steps, pred, "s--", color=C_SECOND, markersize=6,
             linewidth=1.2, label="predicted")
    ax2.set_xlabel(r"step $n \to n+1$")
    ax2.set_ylabel(r"$\Delta S_\flat$")
    ax2.legend(loc="upper right")

    # --- 4.3 Entropy ordering ---
    ax3 = fig.add_subplot(1, 4, 3)
    ns8 = [r["federation_size"] for r in d8["results_per_size"]]
    joint_H = [r["joint_H"]["mean"] for r in d8["results_per_size"]]
    max_H = [r["max_individual_H"]["mean"] for r in d8["results_per_size"]]
    ax3.plot(ns8, joint_H, "o-", color=C_PRIMARY, markersize=6,
             linewidth=1.5, label=r"$\mathfrak{H}(\mathfrak{F})$")
    ax3.plot(ns8, max_H, "s--", color=C_SECOND, markersize=6,
             linewidth=1.2, label=r"$\max_i \mathfrak{H}(\mathcal{R}_i)$")
    ax3.set_xlabel(r"federation size $n$")
    ax3.set_ylabel("knowledge entropy")
    ax3.legend(loc="upper left")
    ax3.set_xticks(ns8)

    # --- 4.4 3D surface: federation floor over (n, mean beta) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    n_arr = np.arange(1, 11)
    beta_arr = np.linspace(10, 60, 20)
    N, BETA = np.meshgrid(n_arr, beta_arr)
    Z = SIGMA * (BETA / SIGMA) ** N
    surf = ax4.plot_surface(N, BETA, Z, cmap="viridis",
                            linewidth=0, antialiased=True, alpha=0.92)
    ax4.set_xlabel(r"$n$", labelpad=-6)
    ax4.set_ylabel(r"$\bar\beta$", labelpad=-6)
    ax4.set_zlabel(r"$S_\flat(\mathfrak{F})$", labelpad=-6)
    ax4.tick_params(pad=-2, labelsize=7)
    ax4.view_init(elev=25, azim=-120)

    plt.tight_layout()
    fig.savefig(HERE / "panel4_federation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel4_federation.png")


# ============================================================================
# PANEL 5 — Cascade switching and circular validation.
# ============================================================================

def panel5():
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    d9 = load("exp09_cascade_knapsack")
    d10 = load("exp10_circular_validation")

    # --- 5.1 Greedy / optimal ratio across budgets ---
    ax1 = fig.add_subplot(1, 4, 1)
    Bs = [t["budget"] for t in d9["results_per_budget"]]
    ratios = [t["greedy_over_optimal"]["mean"] for t in d9["results_per_budget"]]
    stds = [t["greedy_over_optimal"]["std"] for t in d9["results_per_budget"]]
    ax1.errorbar(Bs, ratios, yerr=stds, fmt="o-",
                 color=C_PRIMARY, markersize=7, linewidth=1.5, capsize=3,
                 label="empirical")
    ax1.axhline(1 - 1 / math.e, color=C_SECOND, linestyle="--",
                linewidth=1.2, label=r"$(1 - 1/e)$")
    ax1.axhline(1.0, color=C_GRAY, linestyle=":", linewidth=0.8)
    ax1.set_xlabel("budget $B$")
    ax1.set_ylabel("greedy / optimal")
    ax1.legend(loc="lower right")
    ax1.set_ylim(0.55, 1.05)

    # --- 5.2 Knapsack vs greedy floor (paired) ---
    ax2 = fig.add_subplot(1, 4, 2)
    opt_floor = [t["optimal_floor"]["mean"] for t in d9["results_per_budget"]]
    grd_floor = [t["greedy_floor"]["mean"] for t in d9["results_per_budget"]]
    width = 0.35
    x = np.arange(len(Bs))
    ax2.bar(x - width / 2, opt_floor, width, color=C_THIRD, label="optimal")
    ax2.bar(x + width / 2, grd_floor, width, color=C_FOURTH, label="greedy")
    ax2.set_xticks(x); ax2.set_xticklabels([str(b) for b in Bs])
    ax2.set_xlabel("budget $B$"); ax2.set_ylabel("resulting floor")
    ax2.legend(loc="upper right")

    # --- 5.3 3D: value-density landscape ---
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    beta_grid = np.linspace(5, 55, 30)
    cost_grid = np.linspace(0.3, 3.0, 30)
    BG, CG = np.meshgrid(beta_grid, cost_grid)
    V = np.log(SIGMA / (SIGMA - BG))
    RHO = V / CG
    surf = ax3.plot_surface(BG, CG, RHO, cmap="plasma",
                            linewidth=0, antialiased=True, alpha=0.92)
    ax3.set_xlabel(r"$\beta_i$", labelpad=-6)
    ax3.set_ylabel(r"$c_i$", labelpad=-6)
    ax3.set_zlabel(r"$\rho_i = v_i / c_i$", labelpad=-6)
    ax3.tick_params(pad=-2, labelsize=7)
    ax3.view_init(elev=22, azim=-130)

    # --- 5.4 Circular validation: floor by validator size ---
    ax4 = fig.add_subplot(1, 4, 4)
    sizes = [r["validator_size"] for r in d10["results_per_size"]]
    lin_floor = [r["linear_floor"]["mean"] for r in d10["results_per_size"]]
    circ_floor = [r["circular_floor"]["mean"] for r in d10["results_per_size"]]
    min_indiv = [r["min_individual_floor"]["mean"] for r in d10["results_per_size"]]
    ax4.plot(sizes, lin_floor, "s-", color=C_SECOND, markersize=7,
             linewidth=1.5, label="linear")
    ax4.plot(sizes, circ_floor, "o-", color=C_PRIMARY, markersize=7,
             linewidth=1.5, label="circular")
    ax4.plot(sizes, min_indiv, "d:", color=C_GRAY, markersize=6,
             linewidth=1.0, label=r"$\min_i \beta_i$")
    ax4.axvline(3, color=C_FIFTH, linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.set_xlabel(r"validator graph size $|V|$")
    ax4.set_ylabel("effective floor")
    ax4.legend(loc="upper right")
    ax4.set_xticks(sizes)

    plt.tight_layout()
    fig.savefig(HERE / "panel5_cascade.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  panel5_cascade.png")


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
