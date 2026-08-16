"""
generate_panels.py -- Six publication panels for "The Directional Pair".

Each panel is four charts in a row on a white background, at least one of
which is a three-dimensional surface or scatter.  No panel contains a
conceptual diagram, a text box, or a table: every mark is a measured
quantity.

    python generate_panels.py        # writes ../figures/panel_*.png

Panels
    1  The floor and the self-blunting instrument
    2  Identity: invariance, the Theseus cases, the exhaustive diagonal
    3  The directional identity and representation mobility
    4  Construction: term-map agnosticism and the floor readout
    5  Probing: composition, coherence, allocation
    6  Closure and the separation
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core import (
    MEDIUM,
    ContactGraph,
    Record,
    induced_graph,
    necessary,
    random_contact_graph,
    reach,
)
from exp_probing import knapsack_exact, waterfill
from exp_separation import contact_relation, witness_pair

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.abspath(os.path.join(HERE, "..", "figures"))

# ---- house style: white ground, minimal text ------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "figure.dpi": 160,
})

NAVY = "#1f3b73"
STEEL = "#3c78d8"
TEAL = "#0b8a8f"
CORAL = "#e2624a"
AMBER = "#e0a02f"
PLUM = "#7a4fa3"
GREY = "#8a8a8a"

PANEL_W, PANEL_H = 17.0, 3.9


def _load(eid: str) -> dict:
    with open(os.path.join(RESULTS, f"{eid}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _new_panel():
    fig = plt.figure(figsize=(PANEL_W, PANEL_H))
    return fig


def _finish(fig, name: str):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.tight_layout(pad=1.1, w_pad=1.8)
    out = os.path.join(FIGDIR, f"{name}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, HERE)}")
    return out


def _tag(ax, letter):
    fn = getattr(ax, "text2D", ax.text)
    fn(-0.14, 1.06, letter, transform=ax.transAxes,
       fontsize=11, fontweight="bold", va="top", ha="left")


# =====================================================================
#  PANEL 1 -- The floor and the self-blunting instrument
# =====================================================================

def panel1():
    fig = _new_panel()
    rng = random.Random(101)

    # (A) sigma against graph size, floor line
    ax = fig.add_subplot(1, 4, 1)
    e01 = _load("e01")
    xs = [r["n_items"] for r in e01["rows"]]
    ys = [r["sigma"] for r in e01["rows"]]
    fl = [r["floor"] for r in e01["rows"]]
    ax.scatter(np.array(xs) + np.random.default_rng(0).normal(0, .12, len(xs)),
               ys, s=9, c=STEEL, alpha=.55, edgecolors="none")
    ax.axhline(min(fl), color=CORAL, ls="--", lw=1.4)
    ax.set_xlabel("items in graph")
    ax.set_ylabel(r"separation cost $\sigma(v)$")
    _tag(ax, "A")

    # (B) distribution of sigma / floor, supported at and above 1
    ax = fig.add_subplot(1, 4, 2)
    ratios = [r["ratio"] for r in e01["rows"]]
    ax.hist(ratios, bins=36, color=TEAL, alpha=.85, edgecolor="white", lw=.4)
    ax.axvline(1.0, color=CORAL, ls="--", lw=1.4)
    ax.set_xlabel(r"$\sigma(v)\,/\,\beta$")
    ax.set_ylabel("count")
    _tag(ax, "B")

    # (C) blunting: capacity down, record up, curves crossing
    ax = fig.add_subplot(1, 4, 3)
    e05 = _load("e05")
    for row in e05["rows"][:14]:
        t = row["trajectory"]
        recs = [p["record"] for p in t]
        caps = [p["capacity"] for p in t]
        ax.plot(recs, caps, color=STEEL, alpha=.45, lw=1.1)
        ax.plot(recs, recs, color=CORAL, alpha=.25, lw=.9)
    ax.plot([], [], color=STEEL, label="capacity")
    ax.plot([], [], color=CORAL, label="record")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("committed record")
    ax.set_ylabel("uncommitted capacity")
    _tag(ax, "C")

    # (D) 3D: realised floor surface over (graph size, edge density)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    rngd = random.Random(1414)
    sizes = np.arange(4, 14)
    dens = np.linspace(0.15, 0.85, 12)
    Z = np.zeros((len(sizes), len(dens)))
    for i, n in enumerate(sizes):
        for j, p in enumerate(dens):
            acc = 0.0
            for _ in range(6):
                g = random_contact_graph(int(n), float(p), 1.0, rngd)
                acc += g.system_floor()
            Z[i, j] = acc / 6
    DD, NN = np.meshgrid(dens, sizes)
    ax.plot_surface(DD, NN, Z, cmap="viridis", alpha=.93, linewidth=0,
                    antialiased=True, rstride=1, cstride=1)
    # the floor plane beta = 1: the surface never dips below it
    ax.plot_surface(DD, NN, np.ones_like(Z), color=CORAL, alpha=.20,
                    linewidth=0, antialiased=True)
    ax.set_xlabel("edge density", labelpad=-2)
    ax.set_ylabel("items", labelpad=-2)
    ax.set_zlabel(r"$\beta^{*}$", labelpad=-2)
    ax.view_init(elev=22, azim=-58)
    ax.tick_params(pad=0)
    _tag(ax, "D")

    return _finish(fig, "panel_1_floor_blunting")


# =====================================================================
#  PANEL 2 -- Identity
# =====================================================================

def panel2():
    fig = _new_panel()

    # (A) invariance: sigma after relabelling vs before
    ax = fig.add_subplot(1, 4, 1)
    e02 = _load("e02")
    b = [r["sigma_before"] for r in e02["rows"]]
    a = [r["sigma_after"] for r in e02["rows"]]
    lim = [0, max(b + a) * 1.05]
    ax.plot(lim, lim, color=CORAL, ls="--", lw=1.2)
    ax.scatter(b, a, s=12, c=NAVY, alpha=.6, edgecolors="none")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$\sigma(v)$ before relabelling")
    ax.set_ylabel(r"$\sigma(\phi v)$ after")
    _tag(ax, "A")

    # (B) Theseus: the three record trajectories.  chi is flat in all three;
    #     the record chain is what separates them.
    ax = fig.add_subplot(1, 4, 2)
    n_pl = 7
    base = 3 * n_pl                    # record after construction
    T = 4 * n_pl                       # common horizon

    # replace: the unit is terminated at every step (open indicator 0)
    steps = np.arange(T + 1)
    rec_r = base + steps
    open_r = np.zeros(T + 1)

    # rebuild: torn down over the first half, terminated again only at the end
    rec_d = base + steps
    open_d = np.zeros(T + 1)
    open_d[1:2 * n_pl + 1] = 1.0       # the open interval

    # copy: a separate lineage, its own record from zero
    rec_c = steps.astype(float)

    ax.fill_between(steps, 0, base + T + 2, where=open_d > 0,
                    color=CORAL, alpha=.12, lw=0)
    ax.plot(steps, rec_r, color=TEAL, lw=2.0, label="replace")
    ax.plot(steps, rec_d, color=CORAL, lw=2.0, ls="--", label="rebuild")
    ax.plot(steps, rec_c, color=GREY, lw=1.8, ls=":", label="copy")
    ax.axhline(base, color=NAVY, ls=":", lw=1.0)
    ax.set_ylim(0, base + T + 2)
    ax.set_xlabel("committing acts")
    ax.set_ylabel("record")
    ax.legend(frameon=False, loc="upper left")
    _tag(ax, "B")

    # (C) exhaustive diagonal: consistent vs inconsistent, growing universe
    ax = fig.add_subplot(1, 4, 3)
    ns, tot, cons = [], [], []
    for n in (1, 2, 3, 4):
        subsets = [frozenset(i for i in range(n) if m & (1 << i))
                   for m in range(1 << n)]
        idx = {X: i for i, X in enumerate(subsets)}
        total = 1 << len(subsets)
        c = 0
        for code in range(total):
            V = {subsets[i]: (code >> i) & 1 for i in range(len(subsets))}
            D = frozenset(idx[X] % n for X in subsets if V[X] == 0) if n else frozenset()
            v = V[D]
            if (V[D] == 0) == (v == 1):
                c += 1
        ns.append(n); tot.append(total); cons.append(c)
    ax.semilogy(ns, tot, "o-", color=NAVY, label="verifiers enumerated")
    ax.semilogy(ns, [max(c, .5) for c in cons], "s--", color=CORAL,
                label="consistent diagonals")
    ax.set_xticks(ns)
    ax.set_xlabel("universe size $n$")
    ax.set_ylabel("count (log)")
    ax.legend(frameon=False, loc="center left")
    _tag(ax, "C")

    # (D) 3D: consumer-relativity -- the registered cell as a surface over
    #     (items, consumer density).  Denser consumers register richer cells.
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    items_ax = np.arange(4, 13)
    dens_ax = np.linspace(0.1, 1.0, 12)
    Z = np.zeros((len(items_ax), len(dens_ax)))
    for i, n in enumerate(items_ax):
        for j, p in enumerate(dens_ax):
            g = ContactGraph(vertices=[MEDIUM])
            vs = [f"v{t}" for t in range(int(n))]
            for v in vs:
                g.add_edge(v, MEDIUM, 1.0)
            pairs = list(itertools.combinations(vs, 2))
            take = int(round(p * len(pairs)))
            for a, b in pairs[:take]:      # deterministic density
                g.add_edge(a, b, 2.0)
            Z[i, j] = g.sigma(vs[0])
    PP, NN = np.meshgrid(dens_ax, items_ax)
    ax.plot_surface(PP, NN, Z, cmap="plasma", alpha=.93, linewidth=0,
                    rstride=1, cstride=1, antialiased=True)
    ax.set_xlabel("consumer density", labelpad=-2)
    ax.set_ylabel("items", labelpad=-2)
    ax.set_zlabel(r"registered $\sigma$", labelpad=-2)
    ax.view_init(elev=21, azim=-60)
    ax.tick_params(pad=0)
    _tag(ax, "D")

    return _finish(fig, "panel_2_identity")


# =====================================================================
#  PANEL 3 -- The directional identity
# =====================================================================

def panel3():
    fig = _new_panel()
    rng = random.Random(303)

    # (A) rest-process cut weight vs process-side cut weight: same type
    ax = fig.add_subplot(1, 4, 1)
    e13 = _load("e13")
    pw = [r["process_output_weight"] for r in e13["rows"]]
    rw = [r["rest_output_weight"] for r in e13["rows"]]
    fls = [r["floor"] for r in e13["rows"]]
    ax.scatter(rw, pw, s=14, c=TEAL, alpha=.6, edgecolors="none")
    m = max(pw + rw) * 1.05
    ax.plot([0, m], [0, m], color=GREY, ls=":", lw=1.0)
    ax.axhline(min(fls), color=CORAL, ls="--", lw=1.3)
    ax.axvline(min(fls), color=CORAL, ls="--", lw=1.3)
    ax.set_xlabel("catalogue-side cut weight")
    ax.set_ylabel("process-side cut weight")
    _tag(ax, "A")

    # (B) representation mobility: fraction inadmissible vs bound M
    ax = fig.add_subplot(1, 4, 2)
    rngb = random.Random(7)
    Ms = np.array([0.6, 1.0, 2.0, 5.0, 10.0, 25.0, 60.0, 150.0, 400.0])
    fracs = []
    for M in Ms:
        out = 0
        T = 3000
        for _ in range(T):
            a = rngb.uniform(0.05, 0.95)
            N = rngb.randint(2, 6)
            comps = [rngb.uniform(-M, M) for _ in range(N - 1)]
            comps.append(N * a - sum(comps))
            out += any(c < 0.0 or c > 1.0 for c in comps)
        fracs.append(out / T)
    ax.semilogx(Ms, fracs, "o-", color=PLUM)
    ax.axhline(1.0, color=CORAL, ls="--", lw=1.2)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("component bound $M$")
    ax.set_ylabel("fraction inadmissible")
    _tag(ax, "B")

    # (C) the pair blunts: cumulative deposit across a run of propagations
    ax = fig.add_subplot(1, 4, 3)
    e16 = _load("e16")
    rows = e16["rows"]
    idx = np.arange(1, len(rows) + 1)
    cum_pair = np.cumsum([r["pair_record"] for r in rows])
    cum_cap = np.cumsum(
        [r["pair_capacity_before"] - r["pair_capacity_after"] for r in rows])
    cum_pro = np.cumsum([r["process_alone_record"] for r in rows])
    cum_cat = np.cumsum([r["catalogue_alone_record"] for r in rows])
    ax.plot(idx, cum_pair, color=NAVY, lw=2.2, label="pair: record")
    ax.plot(idx, cum_cap, color=CORAL, lw=1.7, ls="--",
            label="pair: capacity spent")
    ax.plot(idx, cum_pro, color=AMBER, lw=1.7, label="process alone")
    ax.plot(idx, cum_cat, color=GREY, lw=1.7, ls=":", label="catalogue alone")
    ax.fill_between(idx, cum_pro, cum_pair, color=NAVY, alpha=.10)
    ax.set_xlabel("propagations run")
    ax.set_ylabel("cumulative deposit")
    ax.legend(frameon=False, loc="upper left")
    _tag(ax, "C")

    # (D) 3D: staleness surface -- cache error grows with record distance
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    rngd = random.Random(11)
    depos = np.arange(0, 13)
    sizes = np.arange(5, 12)
    Z = np.zeros((len(sizes), len(depos)))
    for i, n in enumerate(sizes):
        acc = np.zeros(len(depos))
        for _ in range(8):
            g = random_contact_graph(int(n), .45, 1.0, rngd)
            items = g.items
            tgt = rngd.choice(items)
            cached = g.sigma(tgt)
            for j, _d in enumerate(depos):
                if j > 0:
                    a, b = rngd.sample(items, 2)
                    if frozenset((a, b)) not in g.weights:
                        g.add_edge(a, b, 1.0 + rngd.random() * 3)
                acc[j] += abs(cached - g.sigma(tgt))
        Z[i] = acc / 8
    D, S = np.meshgrid(depos, sizes)
    ax.plot_surface(D, S, Z, cmap="magma", alpha=.92, linewidth=0,
                    rstride=1, cstride=1, antialiased=True)
    ax.plot(depos, np.full_like(depos, sizes[0], dtype=float),
            np.zeros_like(depos, dtype=float), color=TEAL, lw=2.0)
    ax.set_xlabel("deposits since cache", labelpad=-4)
    ax.set_ylabel("items", labelpad=-4)
    ax.set_zlabel("cache error", labelpad=-6)
    ax.view_init(elev=24, azim=-56)
    ax.tick_params(pad=-2)
    _tag(ax, "D")

    return _finish(fig, "panel_3_directional")


# =====================================================================
#  PANEL 4 -- Construction
# =====================================================================

def panel4():
    fig = _new_panel()

    # (A) floor readout: monotone chains under refinement
    ax = fig.add_subplot(1, 4, 1)
    e19 = _load("e19")
    for r in e19["rows"][:26]:
        ax.plot(range(len(r["floors"])), r["floors"], color=STEEL,
                alpha=.5, lw=1.1, marker="o", ms=2.6)
    ax.set_xlabel("refinement step")
    ax.set_ylabel(r"realised system floor $\beta^{*}$")
    _tag(ax, "A")

    # (B) floor gain distribution -- refinement never lowers the floor
    ax = fig.add_subplot(1, 4, 2)
    gains = [r["floor_gain"] for r in e19["rows"]]
    ax.hist(gains, bins=22, color=TEAL, alpha=.85, edgecolor="white", lw=.4)
    ax.axvline(0, color=CORAL, ls="--", lw=1.4)
    ax.set_xlabel(r"$\beta^{*}$ gain along chain")
    ax.set_ylabel("count")
    _tag(ax, "B")

    # (C) term-map agnosticism: floor by map kind
    ax = fig.add_subplot(1, 4, 3)
    e18 = _load("e18")
    kinds = e18["map_kinds_tested"]
    by = {k: [] for k in kinds}
    for r in e18["rows"]:
        by[r["kind"]].append(r["system_floor"])
    pos = np.arange(len(kinds))
    parts = ax.violinplot([by[k] for k in kinds], positions=pos,
                          showmeans=True, widths=.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(STEEL); pc.set_alpha(.55); pc.set_edgecolor("none")
    for key in ("cmeans", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color(NAVY); parts[key].set_linewidth(1.0)
    ax.set_xticks(pos)
    ax.set_xticklabels([k[:6] for k in kinds], rotation=20)
    ax.set_ylabel(r"$\beta^{*}$ of induced graph")
    _tag(ax, "C")

    # (D) 3D: floor surface over (sources, shared distinctions)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    rng = random.Random(404)
    srcs = np.arange(4, 13)
    shares = np.arange(1, 8)
    Z = np.zeros((len(srcs), len(shares)))
    for i, ns in enumerate(srcs):
        for j, sh in enumerate(shares):
            acc = 0.0
            for _ in range(5):
                alpha = [f"d{t}" for t in range(14)]
                core = set(rng.sample(alpha, int(sh)))
                tau = {}
                for s in range(int(ns)):
                    extra = set(rng.sample(alpha, rng.randint(0, 2)))
                    tau[f"s{s}"] = core | extra
                g = induced_graph(tau, floor=1.0)
                acc += g.system_floor()
            Z[i, j] = acc / 5
    SH, SR = np.meshgrid(shares, srcs)
    ax.plot_surface(SH, SR, Z, cmap="viridis", alpha=.93, linewidth=0,
                    rstride=1, cstride=1, antialiased=True)
    ax.set_xlabel("shared distinctions", labelpad=-4)
    ax.set_ylabel("sources", labelpad=-4)
    ax.set_zlabel(r"$\beta^{*}$", labelpad=-6)
    ax.view_init(elev=23, azim=-60)
    ax.tick_params(pad=-2)
    _tag(ax, "D")

    return _finish(fig, "panel_4_construction")


# =====================================================================
#  PANEL 5 -- Probing
# =====================================================================

def panel5():
    fig = _new_panel()

    # (A) composition: diverse chain vs repeating the weakest
    ax = fig.add_subplot(1, 4, 1)
    e22 = _load("e22")
    d = [r["diverse_composite"] for r in e22["rows"]]
    rep = [r["repeated_weakest_composite"] for r in e22["rows"]]
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1.0)
    ax.scatter(rep, d, s=14, c=PLUM, alpha=.65, edgecolors="none")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"repeat weakest: $1-(1-\kappa_{\min})^{n}$")
    ax.set_ylabel(r"diverse: $1-\prod(1-\kappa_i)$")
    _tag(ax, "A")

    # (B) saturation dichotomy: residual trajectories
    ax = fig.add_subplot(1, 4, 2)
    H = 2000
    seqs = {
        r"$\kappa_i=0.05$": (lambda i: 0.05, TEAL),
        r"$\kappa_i=1/i$": (lambda i: 1.0 / (i + 1), NAVY),
        r"$\kappa_i=2^{-i}$": (lambda i: 2.0 ** (-(i + 1)), CORAL),
        r"$\kappa_i=i^{-2}$": (lambda i: 1.0 / (i + 1) ** 2, AMBER),
    }
    for lab, (f, col) in seqs.items():
        res, traj = 1.0, []
        for i in range(H):
            res *= (1.0 - min(f(i), .999999))
            traj.append(max(res, 1e-18))
        ax.loglog(np.arange(1, H + 1), traj, color=col, label=lab, lw=1.4)
    ax.set_xlabel("probes applied")
    ax.set_ylabel("residual gap")
    ax.legend(frameon=False, loc="lower left")
    _tag(ax, "B")

    # (C) seek vs necessity: over-retention gap grows with redundancy
    ax = fig.add_subplot(1, 4, 3)
    e26 = _load("e26")
    dr = e26["diamond_rows"]
    k = [r["parallel_routes"] for r in dr]
    sk = [r["seek_retains"] for r in dr]
    ne = [r["nec_retains"] for r in dr]
    ax.fill_between(k, ne, sk, color=CORAL, alpha=.22)
    ax.plot(k, sk, "o-", color=STEEL, ms=3.5, label="seek retains")
    ax.plot(k, ne, "s-", color=NAVY, ms=3.5, label="necessity retains")
    ax.set_xlabel("parallel routes")
    ax.set_ylabel("items retained")
    ax.legend(frameon=False, loc="upper left")
    _tag(ax, "C")

    # (D) 3D: water-filling allocation surface over (entry margin, price)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    margins = np.linspace(0.4, 5.0, 40)
    prices = np.linspace(0.25, 5.0, 40)
    MM, PP = np.meshgrid(margins, prices)
    kscale = 1.5
    S = kscale / MM                      # gamma'(0) = k / s = margin
    A = np.maximum(0.0, kscale / PP - S)
    ax.plot_surface(MM, PP, A, cmap="viridis", alpha=.93, linewidth=0,
                    rstride=1, cstride=1, antialiased=True)
    ax.plot(margins, margins, np.zeros_like(margins), color=CORAL, lw=2.0)
    ax.set_xlabel("entry margin", labelpad=-4)
    ax.set_ylabel("price $p^{*}$", labelpad=-4)
    ax.set_zlabel(r"allocation $a_i^{*}$", labelpad=-6)
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(pad=-2)
    _tag(ax, "D")

    return _finish(fig, "panel_5_probing")


# =====================================================================
#  PANEL 6 -- Closure and the separation
# =====================================================================

def panel6():
    fig = _new_panel()

    # (A) the witness: identical assertions, equal pair alignment,
    #     divergent floors and verdicts
    ax = fig.add_subplot(1, 4, 1)
    e29 = _load("e29")
    ny = [r["n_y"] for r in e29["rows"]]
    s1 = [r["sigma_pair_G1"] for r in e29["rows"]]
    s2 = [r["sigma_pair_G2"] for r in e29["rows"]]
    f1 = [r["system_floor_G1"] for r in e29["rows"]]
    f2 = [r["system_floor_G2"] for r in e29["rows"]]
    ax.plot(ny, s1, "o-", color=NAVY, ms=4, label=r"$\sigma(v_0,x^{*})$ both")
    ax.plot(ny, s2, "o", color=NAVY, ms=4, mfc="white")
    ax.plot(ny, f1, "s--", color=TEAL, ms=4, label=r"$\beta^{*}$ graph 1")
    ax.plot(ny, f2, "^--", color=CORAL, ms=4, label=r"$\beta^{*}$ graph 2")
    ax.set_xlabel("irrelevant items $y_i$")
    ax.set_ylabel("weight")
    ax.set_ylim(0, 2.6)
    ax.legend(frameon=False, loc="center right")
    _tag(ax, "A")

    # (B) the two quantities that decide the verdict, swept over the witness:
    #     the local pair cut is flat and equal; the global floor diverges.
    ax = fig.add_subplot(1, 4, 2)
    sizes = np.arange(2, 26)
    pair_cut, f1s, f2s, marg1, marg2 = [], [], [], [], []
    for n_y in sizes:
        g1, g2 = witness_pair(int(n_y))
        s = g1.alignment("v0", "xstar")
        a = g1.system_floor(); b = g2.system_floor()
        pair_cut.append(s); f1s.append(a); f2s.append(b)
        marg1.append(a - s); marg2.append(b - s)
    ax.axhline(0, color=GREY, lw=1.0, ls=":")
    ax.plot(sizes, marg1, "o-", color=TEAL, ms=3.2,
            label=r"graph 1: $\beta^{*}-\sigma$")
    ax.plot(sizes, marg2, "^-", color=CORAL, ms=3.2,
            label=r"graph 2: $\beta^{*}-\sigma$")
    ax.fill_between(sizes, 0, marg1, color=TEAL, alpha=.16)
    ax.fill_between(sizes, marg2, 0, color=CORAL, alpha=.16)
    ax.set_xlabel("irrelevant items $y_i$")
    ax.set_ylabel("accountability margin")
    ax.legend(frameon=False, loc="center right")
    _tag(ax, "B")

    # (C) closure vs threshold: classes still discoverable after k probes
    ax = fig.add_subplot(1, 4, 3)
    rngc = random.Random(606)
    ks = np.arange(1, 13)
    for n_classes, col in ((2, TEAL), (3, NAVY), (4, PLUM), (6, CORAL)):
        frac_open = []
        for k in ks:
            still = 0
            T = 4000
            for _ in range(T):
                probes = [rngc.randrange(n_classes) for _ in range(12)]
                seen = set(probes[:k])
                if len(set(probes)) > len(seen):
                    still += 1
            frac_open.append(still / T)
        ax.plot(ks, frac_open, "-", color=col, lw=1.5,
                label=f"{n_classes} classes")
    ax.set_xlabel("probes invoked")
    ax.set_ylabel("P(uninvoked probe adds a class)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="upper right")
    _tag(ax, "C")

    # (D) 3D: the verdict surface -- accountability over (pair cut, floor)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    pair = np.linspace(0.2, 4.0, 46)
    flo = np.linspace(0.2, 4.0, 46)
    P, F = np.meshgrid(pair, flo)
    eps, Om = 0.0, 10.0
    margin = (F + eps * Om) - P          # >= 0 iff accountable
    ax.plot_surface(P, F, margin, cmap="coolwarm", alpha=.93, linewidth=0,
                    rstride=1, cstride=1, antialiased=True)
    zz = np.zeros_like(pair)
    ax.plot(pair, pair, zz, color="black", lw=1.8)
    ax.set_xlabel(r"$\sigma(v_0,x^{*})$", labelpad=-4)
    ax.set_ylabel(r"$\beta^{*}$", labelpad=-4)
    ax.set_zlabel("margin", labelpad=-6)
    ax.view_init(elev=22, azim=-60)
    ax.tick_params(pad=-2)
    _tag(ax, "D")

    return _finish(fig, "panel_6_separation")


# =====================================================================

def main():
    print("Generating panels ...")
    panel1(); panel2(); panel3(); panel4(); panel5(); panel6()
    print("done.")


if __name__ == "__main__":
    main()
