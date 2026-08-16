//! The contact graph.
//!
//! A finite weighted graph together with a distinguished *medium* vertex `m`
//! adjacent to every item. Contacts carry strictly positive weight, and every
//! contact with the medium carries at least the floor. Separating an item from
//! the medium therefore always costs something: this is where the positive
//! floor comes from, and it is a fact about the medium rather than a parameter
//! anyone chose.
//!
//! Port of `validation/core.py::ContactGraph`.

use std::collections::{BTreeMap, BTreeSet};

use crate::flow::FlowNetwork;

/// The distinguished medium vertex. Every item is adjacent to it.
pub const MEDIUM: &str = "m";

/// An unordered pair, stored ordered so that `{u,v}` and `{v,u}` are one key.
/// Stands in for the reference implementation's `frozenset`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EdgeKey(String, String);

impl EdgeKey {
    pub fn new(u: &str, v: &str) -> Self {
        if u <= v {
            EdgeKey(u.to_string(), v.to_string())
        } else {
            EdgeKey(v.to_string(), u.to_string())
        }
    }

    pub fn left(&self) -> &str {
        &self.0
    }

    pub fn right(&self) -> &str {
        &self.1
    }

    pub fn contains(&self, x: &str) -> bool {
        self.0 == x || self.1 == x
    }

    /// The endpoint that is not `x`, if `x` is an endpoint at all.
    pub fn other(&self, x: &str) -> Option<&str> {
        if self.0 == x {
            Some(&self.1)
        } else if self.1 == x {
            Some(&self.0)
        } else {
            None
        }
    }
}

/// Rejected constructions. Weights and loops are checked at the edge, not
/// deep inside a cut computation, so a malformed graph cannot be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    Loop(String),
    NonPositiveWeight(String, String),
    BadPermutation(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::Loop(v) => write!(f, "no loops: {v} to itself"),
            GraphError::NonPositiveWeight(u, v) => {
                write!(f, "weights must be strictly positive: {{{u}, {v}}}")
            }
            GraphError::BadPermutation(m) => write!(f, "bad relabelling: {m}"),
        }
    }
}

impl std::error::Error for GraphError {}

/// A finite weighted contact graph with a distinguished medium.
///
/// Weights are *set*, not accumulated, by `add_edge` — re-adding a pair
/// replaces its weight. (The residual network built for a cut accumulates;
/// the graph itself does not. This mirrors the reference exactly.)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ContactGraph {
    vertices: Vec<String>,
    weights: BTreeMap<EdgeKey, f64>,
}

impl ContactGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// A graph seeded with the given vertices, in order.
    pub fn with_vertices<I, S>(vertices: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            vertices: vertices.into_iter().map(Into::into).collect(),
            weights: BTreeMap::new(),
        }
    }

    /// Add a contact of weight `w`. Loops and non-positive weights are refused.
    /// Endpoints not yet present are appended, preserving insertion order.
    pub fn add_edge(&mut self, u: &str, v: &str, w: f64) -> Result<(), GraphError> {
        if u == v {
            return Err(GraphError::Loop(u.to_string()));
        }
        if w <= 0.0 {
            return Err(GraphError::NonPositiveWeight(u.to_string(), v.to_string()));
        }
        self.weights.insert(EdgeKey::new(u, v), w);
        for x in [u, v] {
            if !self.vertices.iter().any(|y| y == x) {
                self.vertices.push(x.to_string());
            }
        }
        Ok(())
    }

    /// Register a vertex with no incident contact.
    pub fn add_vertex(&mut self, v: &str) {
        if !self.vertices.iter().any(|y| y == v) {
            self.vertices.push(v.to_string());
        }
    }

    pub fn vertices(&self) -> &[String] {
        &self.vertices
    }

    /// The items: every vertex except the medium.
    pub fn items(&self) -> Vec<String> {
        self.vertices
            .iter()
            .filter(|v| v.as_str() != MEDIUM)
            .cloned()
            .collect()
    }

    /// Edges as `(u, v, w)` in deterministic key order.
    pub fn edges(&self) -> impl Iterator<Item = (&str, &str, f64)> {
        self.weights
            .iter()
            .map(|(k, w)| (k.left(), k.right(), *w))
    }

    pub fn edge_count(&self) -> usize {
        self.weights.len()
    }

    pub fn weight(&self, u: &str, v: &str) -> Option<f64> {
        self.weights.get(&EdgeKey::new(u, v)).copied()
    }

    /// Total contact weight `Ω = w(E)`, the scale against which `ε` is read.
    pub fn total_weight(&self) -> f64 {
        self.weights.values().sum()
    }

    /// Lightest contact in the graph — the empirical lower bound `β` must respect.
    pub fn min_edge_weight(&self) -> Option<f64> {
        self.weights.values().cloned().fold(None, |acc, w| {
            Some(match acc {
                None => w,
                Some(a) => a.min(w),
            })
        })
    }

    pub fn neighbours(&self, u: &str) -> BTreeSet<String> {
        self.weights
            .keys()
            .filter_map(|k| k.other(u))
            .map(|s| s.to_string())
            .collect()
    }

    /// Edges incident to `u`.
    pub fn incident(&self, u: &str) -> Vec<&EdgeKey> {
        self.weights.keys().filter(|k| k.contains(u)).collect()
    }

    /// A fresh residual network over the current weights.
    ///
    /// Rebuilt per query because `max_flow` consumes capacity. Cheap, and it
    /// removes any possibility of one determination contaminating the next —
    /// which is the same discipline the no-cached-answers invariant demands at
    /// the level above.
    pub(crate) fn network(&self) -> FlowNetwork {
        let mut net = FlowNetwork::new();
        for (u, v, w) in self.edges() {
            net.add_undirected(u, v, w);
        }
        for x in &self.vertices {
            net.touch(x);
        }
        net
    }

    /// Relabel every vertex through `perm`. The separation costs are carried
    /// along unchanged — that they are is the conserved-invariant theorem, and
    /// is asserted as a test rather than assumed here.
    pub fn relabel(&self, perm: &BTreeMap<String, String>) -> Result<Self, GraphError> {
        let map = |x: &str| -> String { perm.get(x).cloned().unwrap_or_else(|| x.to_string()) };

        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut vertices = Vec::with_capacity(self.vertices.len());
        for v in &self.vertices {
            let nv = map(v);
            if !seen.insert(nv.clone()) {
                return Err(GraphError::BadPermutation(format!(
                    "two vertices collide on {nv}"
                )));
            }
            vertices.push(nv);
        }

        let mut out = ContactGraph {
            vertices,
            weights: BTreeMap::new(),
        };
        for (u, v, w) in self.edges() {
            out.add_edge(&map(u), &map(v), w)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loops_and_nonpositive_weights_are_refused() {
        let mut g = ContactGraph::new();
        assert!(matches!(g.add_edge("a", "a", 1.0), Err(GraphError::Loop(_))));
        assert!(matches!(
            g.add_edge("a", "b", 0.0),
            Err(GraphError::NonPositiveWeight(_, _))
        ));
        assert!(matches!(
            g.add_edge("a", "b", -1.0),
            Err(GraphError::NonPositiveWeight(_, _))
        ));
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn edge_keys_are_unordered() {
        let mut g = ContactGraph::new();
        g.add_edge("b", "a", 2.0).unwrap();
        assert_eq!(g.weight("a", "b"), Some(2.0));
        // Re-adding sets rather than accumulates.
        g.add_edge("a", "b", 5.0).unwrap();
        assert_eq!(g.weight("b", "a"), Some(5.0));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn items_exclude_the_medium() {
        let mut g = ContactGraph::new();
        g.add_edge("a", MEDIUM, 1.0).unwrap();
        g.add_edge("b", MEDIUM, 1.0).unwrap();
        assert_eq!(g.items(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(g.total_weight(), 2.0);
        assert_eq!(g.min_edge_weight(), Some(1.0));
    }

    #[test]
    fn neighbours_are_symmetric() {
        let mut g = ContactGraph::new();
        g.add_edge("a", "b", 1.0).unwrap();
        g.add_edge("a", "c", 1.0).unwrap();
        assert_eq!(
            g.neighbours("a"),
            ["b", "c"].iter().map(|s| s.to_string()).collect()
        );
        assert_eq!(g.neighbours("b"), ["a"].iter().map(|s| s.to_string()).collect());
    }

    #[test]
    fn relabelling_rejects_collisions() {
        let mut g = ContactGraph::new();
        g.add_edge("a", "b", 1.0).unwrap();
        let mut perm = BTreeMap::new();
        perm.insert("a".to_string(), "b".to_string());
        assert!(matches!(
            g.relabel(&perm),
            Err(GraphError::BadPermutation(_))
        ));
    }
}
