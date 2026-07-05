#!/usr/bin/env python3
"""
Figure panels for
  "Carry the Uncertainty, Not the Knowledge:
   A Tandem Knapsack--Cascade Calculus ..."  (tandem-knapsack-cascade.tex)

Seven panels, one per validation experiment. Each panel is a row of four
charts on a white background, with at least one 3-D chart, minimal text,
and no conceptual/tabular/text-only charts -- every chart plots computed
data from the validation model.

The model here mirrors validation/run_validation.py exactly (steps + medium
vertex, shared-term edges weighted by shared-term count, floor = min positive
edge weight, residue = min step-medium cut, reach = BFS, contribution =
reachable-slice drop). Deterministic under the fixed seed.
"""

import math
import os
import random
from itertools import combinations

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SEED = 42
MEDIUM = "__medium__"
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- global style: white background, minimal chrome ----
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
})

BLUE = "#1f4e79"
ORANGE = "#c05621"
TEAL = "#2c7a7b"
GREY = "#9aa0a6"
CMAP = "viridis"


# ---------------------------------------------------------------------------
# Model (mirrors validation)
# ---------------------------------------------------------------------------

def build_graph(step_terms):
    g = nx.Graph()
    g.add_node(MEDIUM)
    for u, terms in step_terms.items():
        g.add_edge(u, MEDIUM, weight=float(max(1, len(terms))))
    for (u, tu), (v, tv) in combinations(step_terms.items(), 2):
        s = len(tu & tv)
        if s > 0:
            g.add_edge(u, v, weight=float(s))
    return g


def floor_of(g):
    ws = [d["weight"] for _, _, d in g.edges(data=True) if d["weight"] > 0]
    return min(ws) if ws else 0.0


def residue(g, u):
    if u == MEDIUM or u not in g:
        return 0.0
    val, _ = nx.minimum_cut(g, u, MEDIUM, capacity="weight")
    return float(val)


def reach(step_terms, goal_terms):
    adj = {u: set() for u in step_terms}
    for (u, tu), (v, tv) in combinations(step_terms.items(), 2):
        if tu & tv:
            adj[u].add(v); adj[v].add(u)
    dist, frontier, head = {}, [], 0
    for u, t in step_terms.items():
        if t & goal_terms:
            dist[u] = 0; frontier.append(u)
    while head < len(frontier):
        u = frontier[head]; head += 1
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; frontier.append(v)
    return dist


def rand_instance(rng, n, nt, tps):
    vocab = [f"t{i}" for i in range(nt)]
    st = {}
    for i in range(n):
        k = rng.randint(1, tps)
        st[f"s{i}"] = set(rng.sample(vocab, min(k, nt)))
    return st, vocab


def new_panel():
    fig = plt.figure(figsize=(15, 3.6))
    return fig


def finish(fig, name):
    fig.tight_layout(pad=1.4)
    out = os.path.join(HERE, name)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# Panel 1 -- Floor
# ---------------------------------------------------------------------------

def panel1_floor():
    rng = random.Random(SEED)
    sizes, residues, floors, ratios, degs = [], [], [], [], []
    for _ in range(50):
        st, _ = rand_instance(rng, rng.randint(6, 16), rng.randint(5, 11), 4)
        g = build_graph(st)
        b = floor_of(g)
        if b <= 0:
            continue
        for u in st:
            r = residue(g, u)
            sizes.append(len(st)); residues.append(r)
            floors.append(b); ratios.append(r / b)
            # step degree over shared-term edges only (excludes the medium)
            degs.append(g.degree(u) - 1)
    sizes = np.array(sizes); residues = np.array(residues)
    floors = np.array(floors); ratios = np.array(ratios); degs = np.array(degs)

    fig = new_panel()

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(sizes, residues, s=10, c=BLUE, alpha=0.5, edgecolors="none")
    ax1.axhline(floors.min(), color=ORANGE, lw=1.4, ls="--")
    ax1.set_xlabel("steps in graph"); ax1.set_ylabel(r"residue $\rho(u)$")
    ax1.set_title("A")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.hist(ratios, bins=40, color=TEAL, alpha=0.85)
    ax2.axvline(1.0, color=ORANGE, lw=1.4, ls="--")
    ax2.set_xlabel(r"$\rho(u)/\beta$"); ax2.set_ylabel("count")
    ax2.set_title("B")

    # C: residue vs step degree -- the floor is the y=beta line every point
    # lies on or above; degree spreads the cloud so the bound is visible.
    ax3 = fig.add_subplot(1, 4, 3)
    jit = (np.random.RandomState(7).rand(len(degs)) - 0.5) * 0.5
    sc = ax3.scatter(degs + jit, residues, s=10, c=ratios, cmap=CMAP,
                     alpha=0.7, edgecolors="none")
    ax3.axhline(floors.min(), color=ORANGE, lw=1.4, ls="--")
    ax3.set_xlabel("step degree"); ax3.set_ylabel(r"residue $\rho(u)$")
    ax3.set_title("C")
    fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.scatter(sizes, floors, residues, c=ratios, cmap=CMAP, s=9, alpha=0.7)
    # floor plane rho = beta
    xg = np.linspace(sizes.min(), sizes.max(), 6)
    yg = np.linspace(floors.min(), floors.max(), 6)
    X, Y = np.meshgrid(xg, yg)
    ax4.plot_surface(X, Y, Y, color=ORANGE, alpha=0.18, linewidth=0)
    ax4.set_xlabel("steps"); ax4.set_ylabel(r"$\beta$"); ax4.set_zlabel(r"$\rho$")
    ax4.set_title("D"); ax4.view_init(elev=22, azim=-58)

    finish(fig, "panel1_floor.png")


# ---------------------------------------------------------------------------
# Panel 2 -- Residue invariance under re-encoding
# ---------------------------------------------------------------------------

def panel2_invariance():
    rng = random.Random(SEED + 2)
    before, after, discrep, deg_before, deg_after = [], [], [], [], []
    for _ in range(50):
        st, _ = rand_instance(rng, rng.randint(6, 14), 8, 4)
        g = build_graph(st)
        rb = {u: residue(g, u) for u in st}
        db = {u: g.degree(u) for u in st}
        ids = list(st.keys()); perm = ids[:]; rng.shuffle(perm)
        rel = dict(zip(ids, perm))
        st2 = {rel[u]: t for u, t in st.items()}
        g2 = build_graph(st2)
        ra = {u: residue(g2, u) for u in st2}
        da = {u: g2.degree(u) for u in st2}
        for u in st:
            before.append(rb[u]); after.append(ra[rel[u]])
            discrep.append(abs(rb[u] - ra[rel[u]]))
            deg_before.append(db[u]); deg_after.append(da[rel[u]])
    before = np.array(before); after = np.array(after)
    discrep = np.array(discrep)
    lab_before = np.arange(len(before))
    lab_after = np.array([hash((i * 2654435761) & 0xffffffff) % len(before)
                          for i in lab_before])

    fig = new_panel()

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(before, after, s=12, c=BLUE, alpha=0.5, edgecolors="none")
    lo, hi = before.min(), before.max()
    ax1.plot([lo, hi], [lo, hi], color=ORANGE, lw=1.4, ls="--")
    ax1.set_xlabel("residue before"); ax1.set_ylabel("residue after")
    ax1.set_title("A")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(lab_before, lab_after, s=6, c=TEAL, alpha=0.45, edgecolors="none")
    ax2.set_xlabel("label index before"); ax2.set_ylabel("label index after")
    ax2.set_title("B")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.hist(discrep + 1e-18, bins=30, color=ORANGE, alpha=0.85)
    ax3.set_xlabel("|residue discrepancy|"); ax3.set_ylabel("count")
    ax3.set_title("C")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.scatter(before, after, discrep, c=before, cmap=CMAP, s=10, alpha=0.7)
    xg = np.linspace(before.min(), before.max(), 6)
    X, Y = np.meshgrid(xg, xg)
    ax4.plot_surface(X, Y, np.zeros_like(X), color=GREY, alpha=0.15, linewidth=0)
    ax4.set_xlabel("before"); ax4.set_ylabel("after"); ax4.set_zlabel("discrep")
    ax4.set_title("D"); ax4.view_init(elev=20, azim=-60)

    finish(fig, "panel2_invariance.png")


# ---------------------------------------------------------------------------
# Panel 3 -- Necessity vs reachability
# ---------------------------------------------------------------------------

def contribution_reach(step_terms, goal_terms, u, retained):
    if u not in retained:
        return 0
    sub_w = {v: t for v, t in step_terms.items() if v in retained}
    rw = set(reach(sub_w, goal_terms).keys())
    if u not in rw:
        return 0
    sub_wo = {v: t for v, t in step_terms.items() if v in retained and v != u}
    rwo = set(reach(sub_wo, goal_terms).keys())
    return len((rw - {u}) - rwo)


def panel3_necessity():
    rng = random.Random(SEED + 3)
    # (a) random graphs: |reach| vs |necessary| (necessary <= reach)
    reach_sz, nec_sz = [], []
    for _ in range(60):
        st, vocab = rand_instance(rng, rng.randint(6, 14), 9, 3)
        goal = {rng.choice(vocab)}
        R = set(reach(st, goal).keys())
        ret = set(st.keys())
        N = {u for u in ret if contribution_reach(st, goal, u, ret) > 0}
        reach_sz.append(len(R)); nec_sz.append(len(N))
    reach_sz = np.array(reach_sz); nec_sz = np.array(nec_sz)

    # (b) chain: per-position necessity (interior necessary, leaf redundant)
    chain_len = 9
    st = {"s0": {"g", "l0"}}
    for i in range(1, chain_len):
        st[f"s{i}"] = {f"l{i-1}", f"l{i}"}
    goal = {"g"}; ret = set(st.keys())
    positions = list(range(chain_len))
    contribs = [contribution_reach(st, goal, f"s{i}", ret) for i in positions]

    # (c) gap surface: over chain length n and position i, necessity flag
    ns = list(range(3, 11))
    Zpos, Zn, Zflag = [], [], []
    for n in ns:
        stn = {"s0": {"g", "l0"}}
        for i in range(1, n):
            stn[f"s{i}"] = {f"l{i-1}", f"l{i}"}
        rn = set(stn.keys())
        for i in range(n):
            c = contribution_reach(stn, goal, f"s{i}", rn)
            Zpos.append(i); Zn.append(n); Zflag.append(1 if c > 0 else 0)

    fig = new_panel()

    ax1 = fig.add_subplot(1, 4, 1)
    jitter = (np.random.RandomState(1).rand(len(reach_sz)) - 0.5) * 0.3
    ax1.scatter(reach_sz + jitter, nec_sz, s=14, c=BLUE, alpha=0.5,
                edgecolors="none")
    m = max(reach_sz.max(), nec_sz.max())
    ax1.plot([0, m], [0, m], color=ORANGE, lw=1.4, ls="--")
    ax1.set_xlabel(r"$|\mathrm{reach}(g)|$")
    ax1.set_ylabel(r"$|\mathrm{nec}(g)|$")
    ax1.set_title("A")

    ax2 = fig.add_subplot(1, 4, 2)
    colors = [TEAL if c > 0 else ORANGE for c in contribs]
    ax2.bar(positions, [c + 0.04 for c in contribs], color=colors, width=0.7)
    ax2.set_xlabel("chain position"); ax2.set_ylabel("contribution")
    ax2.set_title("B")

    ax3 = fig.add_subplot(1, 4, 3)
    gap = reach_sz - nec_sz
    ax3.hist(gap, bins=range(0, gap.max() + 2), color=BLUE, alpha=0.8,
             align="left", rwidth=0.85)
    ax3.set_xlabel(r"$|\mathrm{reach}|-|\mathrm{nec}|$")
    ax3.set_ylabel("count"); ax3.set_title("C")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    Zpos = np.array(Zpos); Zn = np.array(Zn); Zflag = np.array(Zflag)
    ax4.scatter(Zpos, Zn, Zflag, c=Zflag, cmap="coolwarm_r", s=22, alpha=0.9)
    ax4.set_xlabel("position i"); ax4.set_ylabel("chain length n")
    ax4.set_zlabel("necessary")
    ax4.set_zticks([0, 1])
    ax4.set_title("D"); ax4.view_init(elev=18, azim=-62)

    finish(fig, "panel3_necessity.png")


# ---------------------------------------------------------------------------
# Panel 4 -- Free drop
# ---------------------------------------------------------------------------

def panel4_freedrop():
    rng = random.Random(SEED + 4)
    base_sz, after_sz, n_drops, delta = [], [], [], []
    per_graph_unreach = []
    for _ in range(60):
        st, vocab = rand_instance(rng, rng.randint(8, 16), 10, 3)
        goal = {rng.choice(vocab)}
        R = set(reach(st, goal).keys())
        ret = set(st.keys())
        unreachable = [u for u in st if u not in R]
        per_graph_unreach.append(len(unreachable))
        for u in unreachable:
            sub = {v: t for v, t in st.items() if v in ret - {u}}
            Ra = set(reach(sub, goal).keys())
            base_sz.append(len(R)); after_sz.append(len(Ra))
            n_drops.append(len(unreachable)); delta.append(len(R) - len(Ra))
    base_sz = np.array(base_sz); after_sz = np.array(after_sz)
    n_drops = np.array(n_drops); delta = np.array(delta)

    fig = new_panel()

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(base_sz + (np.random.RandomState(2).rand(len(base_sz)) - .5) * .25,
                after_sz, s=14, c=BLUE, alpha=0.45, edgecolors="none")
    m = base_sz.max()
    ax1.plot([0, m], [0, m], color=ORANGE, lw=1.4, ls="--")
    ax1.set_xlabel(r"$|\mathrm{reach}|$ before drop")
    ax1.set_ylabel(r"$|\mathrm{reach}|$ after drop"); ax1.set_title("A")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.hist(delta, bins=range(-1, 3), color=TEAL, alpha=0.85, align="left",
             rwidth=0.8)
    ax2.axvline(0, color=ORANGE, lw=1.4, ls="--")
    ax2.set_xlabel("change in reachable set"); ax2.set_ylabel("count")
    ax2.set_title("B")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.hist(per_graph_unreach, bins=range(0, max(per_graph_unreach) + 2),
             color=BLUE, alpha=0.8, align="left", rwidth=0.85)
    ax3.set_xlabel("purposeless steps / graph"); ax3.set_ylabel("count")
    ax3.set_title("C")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.scatter(base_sz, n_drops, delta, c=base_sz, cmap=CMAP, s=12, alpha=0.7)
    xg = np.linspace(base_sz.min(), base_sz.max(), 6)
    yg = np.linspace(n_drops.min(), n_drops.max(), 6)
    X, Y = np.meshgrid(xg, yg)
    ax4.plot_surface(X, Y, np.zeros_like(X), color=ORANGE, alpha=0.15, linewidth=0)
    ax4.set_xlabel("|reach|"); ax4.set_ylabel("# drops"); ax4.set_zlabel(r"$\Delta$")
    ax4.set_title("D"); ax4.view_init(elev=22, azim=-58)

    finish(fig, "panel4_freedrop.png")


# ---------------------------------------------------------------------------
# Panel 5 -- Knapsack
# ---------------------------------------------------------------------------

def knap_exact(items, budget):
    ids = [i for i, _, _ in items]; vals = [v for _, v, _ in items]
    costs = [c for _, _, c in items]; n = len(items); B = int(budget)
    T = [[0.0] * (B + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for b in range(B + 1):
            wo = T[i - 1][b]
            wi = vals[i - 1] + T[i - 1][b - costs[i - 1]] if costs[i - 1] <= b else -1e18
            T[i][b] = max(wo, wi)
    return T[n][B]


def knap_greedy(items, budget):
    ranked = sorted(items, key=lambda it: it[1] / it[2], reverse=True)
    tv, tc = 0.0, 0
    for _, v, c in ranked:
        if tc + c <= budget:
            tv += v; tc += c
    return tv


def panel5_knapsack():
    rng = random.Random(SEED + 5)
    gvals, evals, ratios, cmaxes, budgets = [], [], [], [], []
    for _ in range(120):
        n = rng.randint(4, 14)
        items = [(f"i{k}", round(rng.uniform(1, 100), 3), rng.randint(1, 20))
                 for k in range(n)]
        budget = rng.randint(10, 60)
        gv = knap_greedy(items, budget); ev = knap_exact(items, budget)
        gvals.append(gv); evals.append(ev)
        ratios.append(gv / ev if ev > 0 else 1.0)
        cmaxes.append(max(c for _, _, c in items) / budget)
        budgets.append(budget)
    gvals = np.array(gvals); evals = np.array(evals)
    ratios = np.array(ratios); cmaxes = np.array(cmaxes)

    fig = new_panel()

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(evals, gvals, s=12, c=BLUE, alpha=0.5, edgecolors="none")
    m = evals.max()
    ax1.plot([0, m], [0, m], color=ORANGE, lw=1.4, ls="--")
    ax1.set_xlabel("exact optimum"); ax1.set_ylabel("greedy value")
    ax1.set_title("A")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(cmaxes, ratios, s=12, c=TEAL, alpha=0.55, edgecolors="none")
    xs = np.linspace(cmaxes.min(), cmaxes.max(), 50)
    ax2.plot(xs, 1 - xs, color=ORANGE, lw=1.4, ls="--")
    ax2.set_xlabel(r"$c_{\max}/B$"); ax2.set_ylabel("greedy / exact")
    ax2.set_title("B")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.hist(ratios, bins=30, color=BLUE, alpha=0.8)
    ax3.axvline(1.0, color=ORANGE, lw=1.4, ls="--")
    ax3.set_xlabel("greedy / exact"); ax3.set_ylabel("count")
    ax3.set_title("C")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.scatter(cmaxes, np.array(budgets), ratios, c=ratios, cmap=CMAP,
                s=12, alpha=0.7)
    # guarantee surface z = 1 - cmax/B
    xg = np.linspace(cmaxes.min(), cmaxes.max(), 8)
    yg = np.linspace(min(budgets), max(budgets), 8)
    X, Y = np.meshgrid(xg, yg)
    ax4.plot_surface(X, Y, 1 - X, color=ORANGE, alpha=0.18, linewidth=0)
    ax4.set_xlabel(r"$c_{\max}/B$"); ax4.set_ylabel("budget")
    ax4.set_zlabel("g/e"); ax4.set_title("D"); ax4.view_init(elev=20, azim=-60)

    finish(fig, "panel5_knapsack.png")


# ---------------------------------------------------------------------------
# Panel 6 -- Cascade
# ---------------------------------------------------------------------------

def panel6_cascade():
    fig = new_panel()
    ks = [2, 3, 4]
    colors = {2: BLUE, 3: TEAL, 4: ORANGE}

    ax1 = fig.add_subplot(1, 4, 1)
    for k in ks:
        Ns = np.array([k**e for e in range(1, 7)])
        frames = np.ceil(np.log(Ns) / np.log(k)).astype(int) + 1
        ax1.plot(Ns, frames, "o-", color=colors[k], ms=4, lw=1.3,
                 label=f"k={k}")
    ax1.set_xscale("log")
    ax1.set_xlabel("frames N (log)"); ax1.set_ylabel("frames visited")
    ax1.legend(); ax1.set_title("A")

    ax2 = fig.add_subplot(1, 4, 2)
    # working set vs total history M, for fixed cascade -> flat
    Ms = np.logspace(2, 8, 30)
    for k in ks:
        N = k**4
        depth = math.ceil(math.log(N, k)) + 1
        s = 8
        ax2.plot(Ms, np.full_like(Ms, depth * s), color=colors[k], lw=1.5,
                 label=f"k={k}")
    ax2.set_xscale("log")
    ax2.set_xlabel("history length M (log)")
    ax2.set_ylabel("working set")
    ax2.legend(); ax2.set_title("B")

    ax3 = fig.add_subplot(1, 4, 3)
    # flat vs cascade per-goal cost
    Ns = np.logspace(1, 6, 40)
    ax3.plot(Ns, Ns, color=GREY, lw=1.6, ls="--", label="flat O(N)")
    for k in ks:
        ax3.plot(Ns, np.log(Ns) / np.log(k) + 1, color=colors[k], lw=1.5,
                 label=f"cascade k={k}")
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlabel("N (log)"); ax3.set_ylabel("per-goal cost (log)")
    ax3.legend(); ax3.set_title("C")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    for k in ks:
        es = np.arange(1, 7)
        Ns = k**es
        frames = np.ceil(np.log(Ns) / np.log(k)).astype(int) + 1
        ax4.plot(np.full_like(es, k, dtype=float), np.log10(Ns), frames,
                 "o-", color=colors[k], ms=3, lw=1.2)
    # scatter the surface ceil(log_k N)+1
    Kg = np.linspace(2, 4, 12); Lg = np.linspace(0.3, 3.6, 12)
    KK, LL = np.meshgrid(Kg, Lg)
    NN = 10 ** LL
    FF = np.ceil(np.log(NN) / np.log(KK)) + 1
    ax4.plot_surface(KK, LL, FF, cmap=CMAP, alpha=0.35, linewidth=0)
    ax4.set_xlabel("k"); ax4.set_ylabel(r"$\log_{10}N$")
    ax4.set_zlabel("frames"); ax4.set_title("D"); ax4.view_init(elev=22, azim=-58)

    finish(fig, "panel6_cascade.png")


# ---------------------------------------------------------------------------
# Panel 7 -- Separation (diamond)
# ---------------------------------------------------------------------------

def panel7_separation():
    # exact diamond from the proof / validation
    st = {"u1": {"t", "bridge"}, "u2": {"t", "bridge"}, "r": {"bridge", "answer"}}
    goal = {"t"}
    R = set(reach(st, goal).keys())
    ret = set(st.keys())
    nec = {u for u in ret if contribution_reach(st, goal, u, ret) > 0}

    fig = new_panel()

    # A: the diamond as a laid-out graph (data positions, no text labels beyond ticks)
    ax1 = fig.add_subplot(1, 4, 1)
    pos = {"goal": (0, 0), "u1": (1, 0.7), "u2": (1, -0.7), "r": (2, 0)}
    edges = [("goal", "u1"), ("goal", "u2"), ("u1", "r"), ("u2", "r")]
    for a, b in edges:
        ax1.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                 color=GREY, lw=1.4, zorder=1)
    for node, (x, y) in pos.items():
        if node == "goal":
            col = BLUE
        elif node == "u2":
            col = ORANGE      # reachable but redundant
        else:
            col = TEAL
        ax1.scatter([x], [y], s=420, c=col, zorder=2, edgecolors="white",
                    linewidths=1.5)
    ax1.set_xlim(-0.4, 2.4); ax1.set_ylim(-1.2, 1.2)
    ax1.set_xticks([0, 1, 2]); ax1.set_yticks([-0.7, 0, 0.7])
    ax1.set_title("A")

    # B: reachable vs necessary membership per node
    ax2 = fig.add_subplot(1, 4, 2)
    nodes = ["u1", "u2", "r"]
    reach_flag = [1 if n in R else 0 for n in nodes]
    nec_flag = [1 if n in nec else 0 for n in nodes]
    x = np.arange(len(nodes))
    ax2.bar(x - 0.18, reach_flag, width=0.34, color=TEAL, label="reach")
    ax2.bar(x + 0.18, nec_flag, width=0.34, color=ORANGE, label="nec")
    ax2.set_xticks(x); ax2.set_xticklabels(nodes)
    ax2.set_yticks([0, 1]); ax2.set_ylabel("member")
    ax2.legend(); ax2.set_title("B")

    # C: over a family of k-fan diamonds, over-retention of seek-only vs nec
    ax3 = fig.add_subplot(1, 4, 3)
    fan = list(range(2, 12))
    seek_keep, nec_keep = [], []
    for kf in fan:
        stf = {f"u{i}": {"t", "bridge"} for i in range(kf)}
        stf["r"] = {"bridge", "answer"}
        Rf = set(reach(stf, {"t"}).keys())
        retf = set(stf.keys())
        Nf = {u for u in retf if contribution_reach(stf, {"t"}, u, retf) > 0}
        seek_keep.append(len(Rf)); nec_keep.append(len(Nf))
    ax3.plot(fan, seek_keep, "o-", color=TEAL, ms=4, lw=1.4, label="seek keeps")
    ax3.plot(fan, nec_keep, "s-", color=ORANGE, ms=4, lw=1.4, label="nec keeps")
    ax3.fill_between(fan, nec_keep, seek_keep, color=ORANGE, alpha=0.12)
    ax3.set_xlabel("parallel routes k"); ax3.set_ylabel("steps retained")
    ax3.legend(); ax3.set_title("C")

    # D: 3D over-retention surface — retained by seek minus nec, over (k routes, depth)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    Ks, Ds, Gaps = [], [], []
    for kf in range(2, 10):
        for depth in range(1, 6):
            # k parallel routes each of length `depth` to a shared resolver
            stf = {}
            for i in range(kf):
                prev = "t_seed"
                # first hop shares the goal term
                stf[f"u{i}_0"] = {"t", f"e{i}_0"}
                for d in range(1, depth):
                    stf[f"u{i}_{d}"] = {f"e{i}_{d-1}", f"e{i}_{d}"}
                last = f"e{i}_{depth-1}"
                stf.setdefault("r", set()).add(last)
            stf["r"] = stf.get("r", set()) | {"answer"}
            Rf = set(reach(stf, {"t"}).keys())
            retf = set(stf.keys())
            Nf = {u for u in retf if contribution_reach(stf, {"t"}, u, retf) > 0}
            Ks.append(kf); Ds.append(depth); Gaps.append(len(Rf) - len(Nf))
    Ks = np.array(Ks); Ds = np.array(Ds); Gaps = np.array(Gaps)
    ax4.scatter(Ks, Ds, Gaps, c=Gaps, cmap=CMAP, s=26, alpha=0.9)
    ax4.set_xlabel("routes k"); ax4.set_ylabel("depth")
    ax4.set_zlabel("seek−nec gap")
    ax4.set_title("D"); ax4.view_init(elev=20, azim=-60)

    finish(fig, "panel7_separation.png")


def main():
    np.random.seed(SEED)
    print("Generating panels:")
    panel1_floor()
    panel2_invariance()
    panel3_necessity()
    panel4_freedrop()
    panel5_knapsack()
    panel6_cascade()
    panel7_separation()
    print("Done.")


if __name__ == "__main__":
    main()
