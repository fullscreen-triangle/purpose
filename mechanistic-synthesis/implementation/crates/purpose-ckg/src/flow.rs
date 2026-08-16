//! Exact max-flow / min-cut (Edmonds–Karp).
//!
//! Every quantity in the directional pair is a minimum cut, a reachability, a
//! domination, or a convex optimum. This module supplies the first of those
//! exactly, with no external numerical dependency.
//!
//! Port of `validation/core.py::FlowNetwork`, which the executed suite checks
//! against 60 random graphs and the separation witness.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Residual capacities below this are treated as saturated. Matches the
/// reference implementation exactly; changing it changes which cuts are found.
pub const EPS: f64 = 1e-12;

/// Max-flow over a residual graph, Edmonds–Karp (BFS augmenting paths).
///
/// An undirected edge `{u,v}` of weight `w` is modelled as two directed arcs
/// each of capacity `w` — the standard reduction for undirected minimum cut.
///
/// `BTreeMap` rather than `HashMap` throughout: augmenting-path selection then
/// depends only on the vertex ordering, so two runs on the same graph return
/// byte-identical results. The validation suite requires that.
#[derive(Debug, Clone, Default)]
pub struct FlowNetwork {
    cap: BTreeMap<String, BTreeMap<String, f64>>,
}

impl FlowNetwork {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an undirected edge of weight `w` as a symmetric pair of arcs.
    /// Repeated edges accumulate, mirroring the reference.
    pub fn add_undirected(&mut self, u: &str, v: &str, w: f64) {
        *self
            .cap
            .entry(u.to_string())
            .or_default()
            .entry(v.to_string())
            .or_insert(0.0) += w;
        *self
            .cap
            .entry(v.to_string())
            .or_default()
            .entry(u.to_string())
            .or_insert(0.0) += w;
    }

    /// Ensure a vertex exists even with no incident capacity.
    pub fn touch(&mut self, v: &str) {
        self.cap.entry(v.to_string()).or_default();
    }

    /// Shortest augmenting path from `s` to `t` in the residual graph.
    fn bfs(&self, s: &str, t: &str) -> Option<Vec<String>> {
        let mut parent: BTreeMap<&str, Option<&str>> = BTreeMap::new();
        parent.insert(s, None);
        let mut q: VecDeque<&str> = VecDeque::new();
        q.push_back(s);

        while let Some(u) = q.pop_front() {
            let Some(adj) = self.cap.get(u) else { continue };
            for (v, c) in adj {
                if *c > EPS && !parent.contains_key(v.as_str()) {
                    parent.insert(v.as_str(), Some(u));
                    if v == t {
                        // Walk the parent chain back to the source.
                        let mut path = vec![t.to_string()];
                        let mut cur = t;
                        while let Some(Some(p)) = parent.get(cur) {
                            path.push(p.to_string());
                            cur = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    q.push_back(v.as_str());
                }
            }
        }
        None
    }

    /// Maximum flow from `s` to `t`, equal by max-flow/min-cut to the minimum
    /// weight of a cut separating them.
    ///
    /// **Destructive**: consumes residual capacity. Build a fresh network per
    /// query, exactly as `ContactGraph::network` does.
    pub fn max_flow(&mut self, s: &str, t: &str) -> f64 {
        let mut total = 0.0;
        while let Some(path) = self.bfs(s, t) {
            let mut bottleneck = f64::INFINITY;
            for w in path.windows(2) {
                let c = self.cap[&w[0]][&w[1]];
                if c < bottleneck {
                    bottleneck = c;
                }
            }
            for w in path.windows(2) {
                *self.cap.get_mut(&w[0]).unwrap().get_mut(&w[1]).unwrap() -= bottleneck;
                *self
                    .cap
                    .entry(w[1].clone())
                    .or_default()
                    .entry(w[0].clone())
                    .or_insert(0.0) += bottleneck;
            }
            total += bottleneck;
        }
        total
    }

    /// After `max_flow`, the residual-reachable set from `s` is the `s`-side of
    /// a minimum cut.
    pub fn min_cut_side(&self, s: &str) -> BTreeSet<String> {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        seen.insert(s.to_string());
        let mut q: VecDeque<String> = VecDeque::new();
        q.push_back(s.to_string());

        while let Some(u) = q.pop_front() {
            let Some(adj) = self.cap.get(&u) else { continue };
            for (v, c) in adj {
                if *c > EPS && !seen.contains(v) {
                    seen.insert(v.clone());
                    q.push_back(v.clone());
                }
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_edge_flow_is_its_weight() {
        let mut net = FlowNetwork::new();
        net.add_undirected("a", "b", 3.5);
        assert_eq!(net.max_flow("a", "b"), 3.5);
    }

    #[test]
    fn bottleneck_governs_a_chain() {
        // a =5= b =2= c : the cut through the weaker link costs 2.
        let mut net = FlowNetwork::new();
        net.add_undirected("a", "b", 5.0);
        net.add_undirected("b", "c", 2.0);
        assert_eq!(net.max_flow("a", "c"), 2.0);
    }

    #[test]
    fn parallel_routes_add() {
        // Two disjoint routes of capacity 1 and 2 give a cut of 3.
        let mut net = FlowNetwork::new();
        net.add_undirected("s", "x", 1.0);
        net.add_undirected("x", "t", 1.0);
        net.add_undirected("s", "y", 2.0);
        net.add_undirected("y", "t", 2.0);
        assert_eq!(net.max_flow("s", "t"), 3.0);
    }

    #[test]
    fn disconnected_pair_has_zero_flow_and_a_bounded_side() {
        let mut net = FlowNetwork::new();
        net.add_undirected("a", "b", 1.0);
        net.touch("z");
        assert_eq!(net.max_flow("a", "z"), 0.0);
        let side = net.min_cut_side("a");
        assert!(side.contains("a") && side.contains("b"));
        assert!(!side.contains("z"));
    }

    #[test]
    fn saturated_side_excludes_the_sink() {
        let mut net = FlowNetwork::new();
        net.add_undirected("a", "b", 5.0);
        net.add_undirected("b", "c", 2.0);
        net.max_flow("a", "c");
        let side = net.min_cut_side("a");
        assert!(side.contains("a"));
        assert!(!side.contains("c"), "sink must lie across the cut");
    }
}
