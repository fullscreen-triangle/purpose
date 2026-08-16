//! The primitives of the calculus: separation cost, resting cut, system floor,
//! alignment, capacity, accountability.
//!
//! Each is a minimum cut or a simple function of one. Nothing here is a
//! heuristic score, and nothing here is calibrated against a reference corpus:
//! the floor is *derived* from the graph by taking `|V|-1` exact cuts.
//!
//! Port of `validation/core.py::ContactGraph` methods.

use std::collections::BTreeSet;

use crate::graph::{ContactGraph, EdgeKey, MEDIUM};

/// Separation cost `σ(u)` against `target` — the minimum weight of a cut
/// putting `u` on one side and `target` on the other.
///
/// `σ(u, u) = 0` by convention; nothing separates a thing from itself.
pub fn sigma(g: &ContactGraph, u: &str, target: &str) -> f64 {
    if u == target {
        return 0.0;
    }
    g.network().max_flow(u, target)
}

/// Separation cost against the medium — the quantity the floor is a minimum of.
pub fn sigma_medium(g: &ContactGraph, u: &str) -> f64 {
    sigma(g, u, MEDIUM)
}

/// The **resting cut** of `u`: the edges realising `σ(u)`, as unordered pairs.
///
/// This is the cut the item rests at when nothing is pushing on it, and it is
/// what `ckg why` reports — the concrete set of contacts that hold a module
/// apart from the medium, rather than a bare number.
pub fn resting_cut(g: &ContactGraph, u: &str) -> BTreeSet<EdgeKey> {
    let mut net = g.network();
    net.max_flow(u, MEDIUM);
    let side = net.min_cut_side(u);
    g.edges()
        .filter(|(a, b, _)| side.contains(*a) != side.contains(*b))
        .map(|(a, b, _)| EdgeKey::new(a, b))
        .collect()
}

/// The `u`-side of a minimum cut separating `u` from the medium.
pub fn min_cut_side(g: &ContactGraph, u: &str) -> BTreeSet<String> {
    let mut net = g.network();
    net.max_flow(u, MEDIUM);
    net.min_cut_side(u)
}

/// The **system floor** `β* = min over items of σ(v)`.
///
/// Costs `|V|-1` exact minimum cuts. It is deliberately not cached anywhere:
/// a stored floor is a stored answer, and a stored answer goes stale the
/// moment the graph moves.
///
/// Note what this number is and is not. It rises under refinement of the term
/// map, so it is an honest *monotonicity* signal — but a degenerate map in
/// which every module draws identical distinctions induces among the highest
/// floors while discriminating worst. It is not a quantity to maximise.
pub fn system_floor(g: &ContactGraph) -> Option<f64> {
    g.items()
        .iter()
        .map(|v| sigma_medium(g, v))
        .fold(None, |acc, s| {
            Some(match acc {
                None => s,
                Some(a) => a.min(s),
            })
        })
}

/// The item attaining the floor, with its cost. Ties break on vertex order.
pub fn floor_witness(g: &ContactGraph) -> Option<(String, f64)> {
    g.items()
        .into_iter()
        .map(|v| {
            let s = sigma_medium(g, &v);
            (v, s)
        })
        .fold(None, |acc, (v, s)| match acc {
            None => Some((v, s)),
            Some((_, bs)) if s < bs => Some((v, s)),
            keep => keep,
        })
}

/// Alignment `σ(x, x*)` — the separation cost between a pair of items.
pub fn alignment(g: &ContactGraph, x: &str, target: &str) -> f64 {
    sigma(g, x, target)
}

/// Alignment score `a = σ(x,x*)/Ω`, the alignment read against total weight.
pub fn align_score(g: &ContactGraph, x: &str, target: &str) -> f64 {
    let omega = g.total_weight();
    if omega > 0.0 {
        alignment(g, x, target) / omega
    } else {
        0.0
    }
}

/// Uncommitted capacity at `u`: incident contacts not yet in the record.
///
/// Every contact operation spends one. Capacity falls by exactly one as the
/// record rises by exactly one — the instrument blunts itself by being used,
/// and there is no operation that restores it.
pub fn capacity(g: &ContactGraph, u: &str, committed: &BTreeSet<EdgeKey>) -> usize {
    g.incident(u).into_iter().filter(|e| !committed.contains(e)).count()
}

/// **Accountability**: `σ(v₀, x*) ≤ β* + ε·Ω`.
///
/// The left side is local to the queried pair; the right side is a global
/// minimum over the whole graph. That asymmetry is the point — two graphs can
/// agree on every local quantity and disagree here, which is why a store of
/// local assertions cannot express this predicate however it is scored.
pub fn is_accountable(g: &ContactGraph, v0: &str, target: &str, eps: f64) -> bool {
    let Some(floor) = system_floor(g) else {
        return false;
    };
    alignment(g, v0, target) <= floor + eps * g.total_weight()
}

/// The accountability margin `β* + ε·Ω − σ(v₀,x*)`: non-negative iff accountable.
pub fn accountability_margin(g: &ContactGraph, v0: &str, target: &str, eps: f64) -> Option<f64> {
    let floor = system_floor(g)?;
    Some(floor + eps * g.total_weight() - alignment(g, v0, target))
}

/// The **character invariant** `χ`: the family of separation costs, sorted.
///
/// Conserved under relabelling. Two structures sharing a `χ` are not thereby
/// the same individual — the record and the termination condition are what
/// discriminate — but a `χ` that moved under a mere renaming would mean the
/// quantity was never structural to begin with.
pub fn character(g: &ContactGraph) -> Vec<f64> {
    let mut chi: Vec<f64> = g.items().iter().map(|v| sigma_medium(g, v)).collect();
    chi.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    chi
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Star: `n` items each attached to the medium at weight `w`.
    fn star(n: usize, w: f64) -> ContactGraph {
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        for i in 0..n {
            g.add_edge(&format!("v{i}"), MEDIUM, w).unwrap();
        }
        g
    }

    #[test]
    fn sigma_against_self_is_zero() {
        let g = star(3, 1.0);
        assert_eq!(sigma(&g, "v0", "v0"), 0.0);
    }

    #[test]
    fn floor_is_positive_and_never_below_beta() {
        let g = star(5, 1.5);
        let beta = 1.5;
        let floor = system_floor(&g).unwrap();
        assert!(floor >= beta, "floor {floor} fell beneath β {beta}");
        assert_eq!(floor, 1.5);
    }

    #[test]
    fn floor_witness_names_the_cheapest_item() {
        let mut g = star(3, 2.0);
        // v3 is individuated by a single weak contact.
        g.add_edge("v3", MEDIUM, 0.5).unwrap();
        let (who, cost) = floor_witness(&g).unwrap();
        assert_eq!(who, "v3");
        assert_eq!(cost, 0.5);
        assert_eq!(system_floor(&g).unwrap(), 0.5);
    }

    #[test]
    fn resting_cut_is_never_empty_for_a_contacted_item() {
        let g = star(3, 1.0);
        let cut = resting_cut(&g, "v0");
        assert!(!cut.is_empty());
        assert!(cut.contains(&EdgeKey::new("v0", MEDIUM)));
    }

    #[test]
    fn character_is_conserved_under_relabelling() {
        // binv:invariant — the multiset of separation costs is structural.
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        g.add_edge("a", MEDIUM, 1.0).unwrap();
        g.add_edge("b", MEDIUM, 2.0).unwrap();
        g.add_edge("c", MEDIUM, 1.0).unwrap();
        g.add_edge("a", "b", 3.0).unwrap();

        let mut perm = BTreeMap::new();
        perm.insert("a".to_string(), "zeta".to_string());
        perm.insert("b".to_string(), "alpha".to_string());
        perm.insert("c".to_string(), "mu".to_string());

        let h = g.relabel(&perm).unwrap();
        assert_eq!(character(&g), character(&h), "χ moved under a renaming");
        // And the labels genuinely moved.
        assert_ne!(g.items(), h.items());
    }

    #[test]
    fn capacity_falls_as_the_record_takes_contacts() {
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        g.add_edge("a", MEDIUM, 1.0).unwrap();
        g.add_edge("a", "b", 1.0).unwrap();
        g.add_edge("a", "c", 1.0).unwrap();

        let mut committed: BTreeSet<EdgeKey> = BTreeSet::new();
        assert_eq!(capacity(&g, "a", &committed), 3);
        committed.insert(EdgeKey::new("a", "b"));
        assert_eq!(capacity(&g, "a", &committed), 2);
        committed.insert(EdgeKey::new("a", MEDIUM));
        assert_eq!(capacity(&g, "a", &committed), 1);
    }

    #[test]
    fn alignment_score_is_bounded_by_total_weight() {
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        g.add_edge("x", MEDIUM, 1.0).unwrap();
        g.add_edge("y", MEDIUM, 1.0).unwrap();
        g.add_edge("x", "y", 2.0).unwrap();
        let a = align_score(&g, "x", "y");
        assert!((0.0..=1.0).contains(&a), "score {a} out of range");
    }

    #[test]
    fn empty_graph_has_no_floor() {
        let g = ContactGraph::with_vertices([MEDIUM]);
        assert!(system_floor(&g).is_none());
        assert!(floor_witness(&g).is_none());
        assert!(!is_accountable(&g, "a", "b", 0.0));
    }
}
