//! The calculus of the directional pair.
//!
//! A contact graph, exact minimum cuts over it, a derived system floor, a
//! monotone record, and the two separation operators. Everything here is exact
//! set arithmetic and exact max-flow — there is no scoring, no threshold anyone
//! chose, and no cached answer.
//!
//! The crate is pure: no filesystem, no network, no clock. Binding it to a
//! particular universe of items is the job of a domain crate, and the calculus
//! is correct for *any* term map, so a crude binding coarsens the cells without
//! corrupting them.
//!
//! ```
//! use purpose_ckg::{induced_graph, system_floor, MEDIUM};
//! use std::collections::{BTreeMap, BTreeSet};
//!
//! let mut tau: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
//! tau.insert("a".into(), ["parse", "token"].iter().map(|s| s.to_string()).collect());
//! tau.insert("b".into(), ["token", "emit"].iter().map(|s| s.to_string()).collect());
//! tau.insert("c".into(), ["render"].iter().map(|s| s.to_string()).collect());
//!
//! let g = induced_graph(&tau, 1.0).unwrap();
//! assert_eq!(g.weight("a", "b"), Some(1.0)); // one shared distinction
//! assert_eq!(g.weight("a", "c"), None);      // no contact
//! assert!(system_floor(&g).unwrap() >= 1.0); // the floor is never beneath β
//! ```

pub mod flow;
pub mod graph;
pub mod primitives;
pub mod reach;
pub mod record;

pub use flow::{FlowNetwork, EPS};
pub use graph::{ContactGraph, EdgeKey, GraphError, MEDIUM};
pub use primitives::{
    accountability_margin, align_score, alignment, capacity, character, floor_witness,
    is_accountable, min_cut_side, resting_cut, sigma, sigma_medium, system_floor,
};
pub use reach::{
    contribution, dominated_by, dominates, necessary, reach, reach_within, resolution,
};
pub use record::{Entry, Record};

use std::collections::{BTreeMap, BTreeSet};

/// The default floor. Every contact with the medium carries at least this.
pub const FLOOR: f64 = 1.0;

/// A term map: what distinctions each source draws.
pub type TermMap = BTreeMap<String, BTreeSet<String>>;

/// Build the contact graph induced by a term map.
///
/// Two sources are in contact when they draw a distinction in common, weighted
/// by how many they share; every source is in contact with the medium at the
/// floor. Sharing more structure raises the weight, and so raises the cost of
/// telling the two apart — which is what makes the induced floor a signal of
/// how finely the map cuts.
///
/// Weights use `max(β, |shared|)` so no contact ever falls beneath the floor.
pub fn induced_graph(tau: &TermMap, floor: f64) -> Result<ContactGraph, GraphError> {
    induced_graph_with(tau, floor, |k| k as f64)
}

/// `induced_graph` with an explicit weight function of the shared count.
pub fn induced_graph_with<F>(
    tau: &TermMap,
    floor: f64,
    f: F,
) -> Result<ContactGraph, GraphError>
where
    F: Fn(usize) -> f64,
{
    let sources: Vec<&String> = tau.keys().collect();
    let mut g = ContactGraph::new();
    for s in &sources {
        g.add_vertex(s);
    }
    g.add_vertex(MEDIUM);

    for (i, u) in sources.iter().enumerate() {
        for v in sources.iter().skip(i + 1) {
            let shared = tau[*u].intersection(&tau[*v]).count();
            if shared > 0 {
                g.add_edge(u, v, floor.max(f(shared)))?;
            }
        }
    }
    for s in &sources {
        g.add_edge(s, MEDIUM, floor)?;
    }
    Ok(g)
}

/// Is `fine` a refinement of `coarse` — does it draw at least every distinction
/// the coarser map draws, for every source?
///
/// The floor is non-decreasing along a refinement, which is the one honest
/// reading of that number: it reports *movement* toward a finer map. It is not
/// a quality score. A degenerate map in which every source draws identical
/// distinctions induces among the highest floors while discriminating worst.
pub fn refines(fine: &TermMap, coarse: &TermMap) -> bool {
    coarse
        .iter()
        .all(|(k, cs)| fine.get(k).is_some_and(|fs| cs.is_subset(fs)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tm(pairs: &[(&str, &[&str])]) -> TermMap {
        pairs
            .iter()
            .map(|(k, ts)| {
                (
                    k.to_string(),
                    ts.iter().map(|t| t.to_string()).collect::<BTreeSet<String>>(),
                )
            })
            .collect()
    }

    /// The separation witness. Two graphs with **identical contact relations**:
    /// the same vertices, the same pairs in contact, at every size. Only the
    /// weight on the irrelevant items differs.
    ///
    /// A store of local assertions records the same thing about both. Yet the
    /// verdicts differ, because the threshold is a global minimum.
    fn witness_pair(n_y: usize) -> (ContactGraph, ContactGraph) {
        let build = |wy: f64| {
            let mut g = ContactGraph::with_vertices([MEDIUM]);
            g.add_edge("v0", "xstar", 1.0).unwrap();
            g.add_edge("v0", MEDIUM, 1.0).unwrap();
            g.add_edge("xstar", MEDIUM, 1.0).unwrap();
            for i in 0..n_y {
                g.add_edge(&format!("y{i}"), MEDIUM, wy).unwrap();
            }
            g
        };
        (build(2.0), build(1.0))
    }

    #[test]
    fn the_witness_pair_agrees_locally_and_differs_in_verdict() {
        // thm:query-separation, at n_y = 4 — the sharpest instance in the suite.
        let (g1, g2) = witness_pair(4);

        // Identical contact relations: same vertices, same pairs.
        let e1: BTreeSet<(String, String)> = g1
            .edges()
            .map(|(a, b, _)| (a.to_string(), b.to_string()))
            .collect();
        let e2: BTreeSet<(String, String)> = g2
            .edges()
            .map(|(a, b, _)| (a.to_string(), b.to_string()))
            .collect();
        assert_eq!(e1, e2, "the two graphs must be locally indistinguishable");
        assert_eq!(g1.items(), g2.items());

        // The queried quantity is local, and equal.
        assert_eq!(alignment(&g1, "v0", "xstar"), 2.0);
        assert_eq!(alignment(&g2, "v0", "xstar"), 2.0);

        // The threshold is global, and different.
        assert_eq!(system_floor(&g1).unwrap(), 2.0);
        assert_eq!(system_floor(&g2).unwrap(), 1.0);

        // Hence the verdicts differ at ε = 0.
        assert!(is_accountable(&g1, "v0", "xstar", 0.0));
        assert!(!is_accountable(&g2, "v0", "xstar", 0.0));
    }

    #[test]
    fn the_accountability_margin_is_flat_as_irrelevant_items_accumulate() {
        // Adding items on no path between the queried pair moves the verdict
        // without moving anything local to it. Margins: exactly 0 and −1.
        for n_y in 2..=10 {
            let (g1, g2) = witness_pair(n_y);
            assert_eq!(accountability_margin(&g1, "v0", "xstar", 0.0), Some(0.0));
            assert_eq!(accountability_margin(&g2, "v0", "xstar", 0.0), Some(-1.0));
        }
    }

    #[test]
    fn induced_contacts_follow_shared_distinctions() {
        let tau = tm(&[
            ("a", &["parse", "token", "emit"]),
            ("b", &["token", "emit"]),
            ("c", &["render"]),
        ]);
        let g = induced_graph(&tau, 1.0).unwrap();
        assert_eq!(g.weight("a", "b"), Some(2.0), "two shared distinctions");
        assert_eq!(g.weight("a", "c"), None, "no shared distinction, no contact");
        for s in ["a", "b", "c"] {
            assert_eq!(g.weight(s, MEDIUM), Some(1.0));
        }
    }

    #[test]
    fn no_induced_contact_falls_beneath_the_floor() {
        // thm:floor / thm:tau-agnostic — holds for adversarial maps too.
        let families = [
            tm(&[("a", &["x"]), ("b", &["y"]), ("c", &["z"])]), // sparse
            tm(&[("a", &["x", "y"]), ("b", &["x", "y"]), ("c", &["x", "y"])]), // degenerate
            tm(&[("a", &[]), ("b", &["q"])]),                   // empty source
        ];
        for tau in families {
            for beta in [0.5, 1.0, 2.0] {
                let g = induced_graph(&tau, beta).unwrap();
                for (_, _, w) in g.edges() {
                    assert!(w >= beta, "contact of weight {w} fell beneath β {beta}");
                }
                assert!(system_floor(&g).unwrap() >= beta);
            }
        }
    }

    #[test]
    fn the_floor_does_not_fall_under_refinement() {
        // thm:floor-readout. The finer map draws every distinction the coarser
        // one draws, and more.
        let coarse = tm(&[("a", &["x"]), ("b", &["x"]), ("c", &["y"])]);
        let fine = tm(&[
            ("a", &["x", "p", "q"]),
            ("b", &["x", "p", "q"]),
            ("c", &["y", "p"]),
        ]);
        assert!(refines(&fine, &coarse));

        let bc = system_floor(&induced_graph(&coarse, 1.0).unwrap()).unwrap();
        let bf = system_floor(&induced_graph(&fine, 1.0).unwrap()).unwrap();
        assert!(bf >= bc, "floor fell under refinement: {bc} → {bf}");
    }

    #[test]
    fn a_degenerate_map_induces_a_high_floor_while_discriminating_worst() {
        // rem:quality-honest — the floor is a monotonicity signal, not a score
        // to maximise. Every source drawing identical distinctions is the worst
        // possible map and yields among the highest floors.
        let degenerate = tm(&[
            ("a", &["x", "y", "z"]),
            ("b", &["x", "y", "z"]),
            ("c", &["x", "y", "z"]),
        ]);
        let discriminating = tm(&[("a", &["x"]), ("b", &["y"]), ("c", &["z"])]);

        let bd = system_floor(&induced_graph(&degenerate, 1.0).unwrap()).unwrap();
        let bs = system_floor(&induced_graph(&discriminating, 1.0).unwrap()).unwrap();
        assert!(
            bd > bs,
            "the degenerate map should induce the higher floor ({bd} vs {bs})"
        );
    }

    #[test]
    fn an_empty_term_map_yields_a_graph_with_no_items() {
        let g = induced_graph(&TermMap::new(), 1.0).unwrap();
        assert!(g.items().is_empty());
        assert!(system_floor(&g).is_none());
    }

    #[test]
    fn cuts_are_reproducible_across_runs() {
        // The suite requires byte-identical reruns; ordered maps give that.
        let tau = tm(&[
            ("m1", &["a", "b", "c"]),
            ("m2", &["b", "c"]),
            ("m3", &["c", "d"]),
            ("m4", &["e"]),
        ]);
        let g = induced_graph(&tau, 1.0).unwrap();
        let first = character(&g);
        for _ in 0..8 {
            assert_eq!(character(&g), first);
        }
    }
}
