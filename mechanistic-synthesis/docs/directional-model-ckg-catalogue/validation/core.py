"""
core.py -- Contact graphs, exact minimum cuts, and the primitives of
"The Directional Pair".

Self-contained: an exact Edmonds-Karp max-flow / min-cut backend with no
external dependency beyond the standard library.  Every quantity in the
paper is a minimum cut, a reachability, a domination, or a convex optimum,
so this module supplies the first three and `waterfill.py` the fourth.

Correspondence to the paper:
    Definition  (finite weighted graph, cut)      -> Graph, cut_weight
    Definition  (contact graph, medium)           -> ContactGraph
    Definition  (separation cost, resting cut)    -> sigma(), resting_cut()
    Definition  (system floor)                    -> system_floor()
    Definition  (alignment, alignment score)      -> alignment(), align_score()
    Definition  (committed record)                -> Record
    Definition  (edge capacity of an instrument)  -> capacity()
    Definition  (accountability)                  -> is_accountable()
    Definition  (term map, induced contact graph) -> induced_graph()
    Definition  (reachability)                    -> reach()
    Definition  (contribution, necessity)         -> contribution(), necessary()
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

MEDIUM = "m"  # the distinguished medium vertex


# =====================================================================
#  Exact max-flow / min-cut  (Edmonds-Karp)
# =====================================================================

class FlowNetwork:
    """Integer/float capacity max-flow with residual graph, Edmonds-Karp.

    Undirected edge {u,v} of weight w is modelled as two directed arcs each
    of capacity w, which is the standard reduction for undirected min cut.
    """

    def __init__(self) -> None:
        self.cap: Dict[str, Dict[str, float]] = {}

    def add_undirected(self, u: str, v: str, w: float) -> None:
        self.cap.setdefault(u, {})
        self.cap.setdefault(v, {})
        self.cap[u][v] = self.cap[u].get(v, 0.0) + w
        self.cap[v][u] = self.cap[v].get(u, 0.0) + w

    def _bfs(self, s: str, t: str) -> Optional[List[str]]:
        parent: Dict[str, Optional[str]] = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, c in self.cap.get(u, {}).items():
                if c > 1e-12 and v not in parent:
                    parent[v] = u
                    if v == t:
                        path = [t]
                        while parent[path[-1]] is not None:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    q.append(v)
        return None

    def max_flow(self, s: str, t: str) -> float:
        """Destructive: consumes residual capacity.  Use on a fresh copy."""
        total = 0.0
        while True:
            path = self._bfs(s, t)
            if path is None:
                return total
            bottleneck = min(
                self.cap[path[i]][path[i + 1]] for i in range(len(path) - 1)
            )
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                self.cap[u][v] -= bottleneck
                self.cap[v][u] = self.cap[v].get(u, 0.0) + bottleneck
            total += bottleneck

    def min_cut_side(self, s: str) -> Set[str]:
        """After max_flow, the residual-reachable set from s is the s-side."""
        seen = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            for v, c in self.cap.get(u, {}).items():
                if c > 1e-12 and v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen


# =====================================================================
#  Contact graphs
# =====================================================================

@dataclass
class ContactGraph:
    """A finite weighted graph with a distinguished medium vertex adjacent
    to every item.  Items are the vertices other than the medium."""

    vertices: List[str]
    weights: Dict[FrozenSet[str], float] = field(default_factory=dict)

    # ---------- construction ----------

    def add_edge(self, u: str, v: str, w: float) -> None:
        if u == v:
            raise ValueError("no loops")
        if w <= 0:
            raise ValueError("weights must be strictly positive")
        self.weights[frozenset((u, v))] = w
        for x in (u, v):
            if x not in self.vertices:
                self.vertices.append(x)

    @property
    def items(self) -> List[str]:
        return [v for v in self.vertices if v != MEDIUM]

    @property
    def edges(self) -> List[Tuple[str, str, float]]:
        out = []
        for e, w in self.weights.items():
            u, v = tuple(e)
            out.append((u, v, w))
        return out

    def total_weight(self) -> float:
        """Omega = w(E)."""
        return sum(self.weights.values())

    def min_edge_weight(self) -> float:
        return min(self.weights.values()) if self.weights else 0.0

    def neighbours(self, u: str) -> Set[str]:
        out = set()
        for e in self.weights:
            if u in e:
                a, b = tuple(e)
                out.add(b if a == u else a)
        return out

    def _network(self) -> FlowNetwork:
        net = FlowNetwork()
        for u, v, w in self.edges:
            net.add_undirected(u, v, w)
        for x in self.vertices:
            net.cap.setdefault(x, {})
        return net

    # ---------- the primitives of the paper ----------

    def sigma(self, u: str, target: str = MEDIUM) -> float:
        """Separation cost: minimum weight of a cut placing u on one side and
        `target` on the other.  Definition (separation cost)."""
        if u == target:
            return 0.0
        net = self._network()
        return net.max_flow(u, target)

    def resting_cut(self, u: str) -> FrozenSet[FrozenSet[str]]:
        """The minimising cut of u against the medium, as a set of edges.
        Definition (resting cut)."""
        net = self._network()
        net.max_flow(u, MEDIUM)
        side = net.min_cut_side(u)
        cut = set()
        for e in self.weights:
            a, b = tuple(e)
            if (a in side) != (b in side):
                cut.add(e)
        return frozenset(cut)

    def min_cut_side(self, u: str, target: str = MEDIUM) -> Set[str]:
        net = self._network()
        net.max_flow(u, target)
        return net.min_cut_side(u)

    def system_floor(self) -> float:
        """beta* = min over items of sigma(v).  Definition (system floor)."""
        return min(self.sigma(v) for v in self.items)

    def alignment(self, x: str, target: str) -> float:
        """sigma(x, x*).  Definition (alignment)."""
        return self.sigma(x, target)

    def align_score(self, x: str, target: str) -> float:
        """a(x, x*) = sigma(x,x*) / Omega.  Definition (alignment score)."""
        om = self.total_weight()
        return self.alignment(x, target) / om if om > 0 else 0.0

    def capacity(self, u: str, committed: Set[FrozenSet[str]]) -> int:
        """Edge capacity of u: contacts incident to u not yet committed.
        Definition (edge capacity of an instrument)."""
        inc = {e for e in self.weights if u in e}
        return len(inc - committed)

    def is_accountable(self, v0: str, target: str, eps: float = 0.0) -> bool:
        """sigma(v0,x*) <= beta* + eps * Omega.  Definition (accountability)."""
        return self.alignment(v0, target) <= (
            self.system_floor() + eps * self.total_weight()
        )

    def relabel(self, perm: Dict[str, str]) -> "ContactGraph":
        """Apply a weighted isomorphism fixing the medium."""
        g = ContactGraph(vertices=[perm.get(v, v) for v in self.vertices])
        for u, v, w in self.edges:
            g.add_edge(perm.get(u, u), perm.get(v, v), w)
        return g


# =====================================================================
#  The committed record
# =====================================================================

@dataclass
class Record:
    """Monotone committed count.  Theorem (monotone non-return): strictly
    increasing, never decremented; un-committing is a further commit."""

    count: int = 0
    committed: Set[FrozenSet[str]] = field(default_factory=set)
    log: List[Tuple[int, FrozenSet[str], str]] = field(default_factory=list)

    def commit(self, u: str, v: str, note: str = "") -> int:
        e = frozenset((u, v))
        self.count += 1
        self.committed.add(e)
        self.log.append((self.count, e, note))
        return self.count

    def uncommit(self, u: str, v: str) -> int:
        """NOT a decrement.  Un-committing is itself a committing act, so the
        record advances.  This method exists to demonstrate the theorem."""
        e = frozenset((u, v))
        self.count += 1
        self.committed.discard(e)
        self.log.append((self.count, e, "uncommit (a further commit)"))
        return self.count


# =====================================================================
#  Term maps and induced graphs
# =====================================================================

def induced_graph(
    tau: Dict[str, Set[str]],
    floor: float = 1.0,
    f=lambda k: float(k),
) -> ContactGraph:
    """Induced contact graph of a term map.  Definition (term map).

    Contact {u,v} whenever tau(u) & tau(v) != empty, weight f(|shared|);
    the medium is adjacent to every source at weight `floor`.
    """
    sources = sorted(tau)
    g = ContactGraph(vertices=list(sources) + [MEDIUM])
    for u, v in itertools.combinations(sources, 2):
        shared = tau[u] & tau[v]
        if shared:
            g.add_edge(u, v, max(floor, f(len(shared))))
    for u in sources:
        g.add_edge(u, MEDIUM, floor)
    return g


def refines(tau_fine: Dict[str, Set[str]], tau_coarse: Dict[str, Set[str]]) -> bool:
    """tau_fine <= tau_coarse iff every shared distinction of the coarse map
    is also shared by the fine map.  Definition (refinement of term maps)."""
    for u, v in itertools.combinations(sorted(tau_coarse), 2):
        if not (tau_coarse[u] & tau_coarse[v]) <= (
            tau_fine.get(u, set()) & tau_fine.get(v, set())
        ):
            return False
    return True


# =====================================================================
#  Reachability, contribution, necessity, domination
# =====================================================================

def reach(g: ContactGraph, seeds: Iterable[str]) -> Set[str]:
    """Items reachable from the seeds through contacts, excluding the medium.
    Definition (reachability)."""
    seen: Set[str] = set()
    q = deque(s for s in seeds if s != MEDIUM)
    seen.update(q)
    while q:
        u = q.popleft()
        for v in g.neighbours(u):
            if v != MEDIUM and v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def resolution(g: ContactGraph, target_seeds: Iterable[str], retained: Set[str]) -> int:
    """R(x* | W) = |reach_W(x*)|.  The resolution functional; it GROWS with
    resolving power, per the Remark on orientation."""
    sub = ContactGraph(vertices=[v for v in g.vertices if v in retained or v == MEDIUM])
    for u, v, w in g.edges:
        if (u in retained or u == MEDIUM) and (v in retained or v == MEDIUM):
            sub.add_edge(u, v, w)
    seeds = [s for s in target_seeds if s in retained]
    return len(reach(sub, seeds)) if seeds else 0


def contribution(
    g: ContactGraph, target_seeds: Iterable[str], retained: Set[str], u: str
) -> int:
    """R(x*|W) - R(x*|W \\ {u}), measured over the items OTHER than u.

    Definition (contribution).  The dropped item must be excluded from both
    counts: otherwise u's own disappearance from the reachable set makes every
    reachable item score at least 1, and `necessary` degenerates to `reach`.
    What the contribution measures is what the goal can still resolve among
    the remaining items, which is exactly the domination criterion of the
    Proposition (necessity is domination).
    """
    if u in target_seeds:
        # a seed is load-bearing by definition: without it the goal has no
        # point of entry into the graph
        return max(0, resolution(g, target_seeds, retained))
    others = retained - {u}
    full = len(reach_within(g, target_seeds, retained) - {u})
    minus = len(reach_within(g, target_seeds, others))
    return full - minus


def reach_within(
    g: ContactGraph, seeds: Iterable[str], retained: Set[str]
) -> Set[str]:
    """Reachability from the seeds using only the retained items."""
    sub = ContactGraph(
        vertices=[v for v in g.vertices if v in retained or v == MEDIUM]
    )
    for u, v, w in g.edges:
        if (u in retained or u == MEDIUM) and (v in retained or v == MEDIUM):
            sub.add_edge(u, v, w)
    live = [s for s in seeds if s in retained]
    return reach(sub, live) if live else set()


def necessary(
    g: ContactGraph, target_seeds: Iterable[str], retained: Set[str]
) -> Set[str]:
    """nec(W, x*) = {u : contribution > 0}."""
    return {
        u for u in retained if contribution(g, target_seeds, retained, u) > 0
    }


def dominates(
    g: ContactGraph, target_seeds: Iterable[str], u: str, r: str, universe: Set[str]
) -> bool:
    """u dominates r iff every path from the seeds to r passes through u,
    i.e. removing u makes r unreachable.  Proposition (necessity is
    domination)."""
    if u == r:
        return False
    without = universe - {u}
    sub = ContactGraph(
        vertices=[v for v in g.vertices if v in without or v == MEDIUM]
    )
    for a, b, w in g.edges:
        if (a in without or a == MEDIUM) and (b in without or b == MEDIUM):
            sub.add_edge(a, b, w)
    seeds = [s for s in target_seeds if s in without]
    return r not in reach(sub, seeds) if seeds else True


# =====================================================================
#  Random instance generation
# =====================================================================

def random_contact_graph(
    n_items: int, p_edge: float, floor: float, rng: random.Random, wmax: float = 5.0
) -> ContactGraph:
    """Random contact graph: every item joined to the medium at weight >= floor,
    item-item contacts present with probability p_edge."""
    items = [f"v{i}" for i in range(n_items)]
    g = ContactGraph(vertices=items + [MEDIUM])
    for u in items:
        g.add_edge(u, MEDIUM, floor + rng.random() * wmax)
    for u, v in itertools.combinations(items, 2):
        if rng.random() < p_edge:
            g.add_edge(u, v, floor + rng.random() * wmax)
    return g


def random_permutation(g: ContactGraph, rng: random.Random) -> Dict[str, str]:
    """A relabelling fixing the medium."""
    items = g.items
    shuffled = items[:]
    rng.shuffle(shuffled)
    perm = dict(zip(items, shuffled))
    perm[MEDIUM] = MEDIUM
    return perm


# =====================================================================
#  Result persistence
# =====================================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def save(name: str, payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_default)
    return path


def _default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(str(x) for x in o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"
