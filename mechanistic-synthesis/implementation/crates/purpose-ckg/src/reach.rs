//! Reachability and necessity — the two separation operators.
//!
//! `seek` asks what a goal can *get to*; `nec` asks what it cannot *do without*.
//! They are not the same question and they do not return the same set.
//! `nec ⊆ reach` always, and the gap between them is exactly the over-retention
//! a reachability-only mechanism incurs: on the diamond witness `seek` climbs
//! from 4 to 13 items while `nec` stays flat at 1.
//!
//! Port of `validation/core.py::{reach, resolution, contribution, necessary,
//! dominates}`.

use std::collections::{BTreeSet, VecDeque};

use crate::graph::{ContactGraph, MEDIUM};

/// Items reachable from `seeds` through contacts, excluding the medium.
///
/// The medium is adjacent to everything, so leaving it in would make every
/// graph a single blob and the operator vacuous.
pub fn reach(g: &ContactGraph, seeds: &BTreeSet<String>) -> BTreeSet<String> {
    let items: BTreeSet<String> = g.items().into_iter().collect();
    let mut seen: BTreeSet<String> = seeds.iter().filter(|s| items.contains(*s)).cloned().collect();
    let mut q: VecDeque<String> = seen.iter().cloned().collect();

    while let Some(u) = q.pop_front() {
        for v in g.neighbours(&u) {
            if v != MEDIUM && items.contains(&v) && seen.insert(v.clone()) {
                q.push_back(v);
            }
        }
    }
    seen
}

/// Reachability confined to a retained subset of items.
pub fn reach_within(
    g: &ContactGraph,
    seeds: &BTreeSet<String>,
    retained: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut seen: BTreeSet<String> = seeds
        .iter()
        .filter(|s| retained.contains(*s))
        .cloned()
        .collect();
    let mut q: VecDeque<String> = seen.iter().cloned().collect();

    while let Some(u) = q.pop_front() {
        for v in g.neighbours(&u) {
            if v != MEDIUM && retained.contains(&v) && seen.insert(v.clone()) {
                q.push_back(v);
            }
        }
    }
    seen
}

/// The resolution functional: how much the retained set resolves for the goal.
///
/// It **grows** with resolving power. A functional that fell as more became
/// resolvable would invert every comparison built on top of it, so the
/// orientation is fixed here once and relied on everywhere below.
pub fn resolution(
    g: &ContactGraph,
    target_seeds: &BTreeSet<String>,
    retained: &BTreeSet<String>,
) -> usize {
    reach_within(g, target_seeds, retained).len()
}

/// The contribution of `u`: `R(x*|W) − R(x*|W\{u})`.
///
/// **`u` is excluded from both counts.** Counting `u`'s own disappearance would
/// give every reachable item a contribution of at least one and collapse `nec`
/// onto `reach` — the operators would stop being distinguishable at all. The
/// comparison must be over the items *other than* `u`.
///
/// A seed is load-bearing by definition: it is the entry point, and an isolated
/// seed dominates nothing while still being the only way in. So seeds are
/// answered directly rather than through the domination test.
pub fn contribution(
    g: &ContactGraph,
    target_seeds: &BTreeSet<String>,
    retained: &BTreeSet<String>,
    u: &str,
) -> i64 {
    if target_seeds.contains(u) {
        return resolution(g, target_seeds, retained).max(0) as i64;
    }
    let others: BTreeSet<String> = retained.iter().filter(|x| x.as_str() != u).cloned().collect();

    let full = reach_within(g, target_seeds, retained)
        .into_iter()
        .filter(|x| x.as_str() != u)
        .count() as i64;
    let minus = reach_within(g, target_seeds, &others).len() as i64;

    full - minus
}

/// `nec`: the retained items whose contribution is strictly positive.
pub fn necessary(
    g: &ContactGraph,
    target_seeds: &BTreeSet<String>,
    retained: &BTreeSet<String>,
) -> BTreeSet<String> {
    retained
        .iter()
        .filter(|u| contribution(g, target_seeds, retained, u) > 0)
        .cloned()
        .collect()
}

/// Does `u` dominate `r` — is every route from the seeds to `r` through `u`?
///
/// For a **non-seed** item, necessity and domination coincide. That equivalence
/// is what makes the necessity verdict explicable: `ckg why` can name the items
/// a module dominates rather than only reporting that dropping it hurt.
pub fn dominates(
    g: &ContactGraph,
    target_seeds: &BTreeSet<String>,
    u: &str,
    r: &str,
    universe: &BTreeSet<String>,
) -> bool {
    if u == r {
        return false;
    }
    let without: BTreeSet<String> = universe.iter().filter(|x| x.as_str() != u).cloned().collect();
    let seeds: BTreeSet<String> = target_seeds
        .iter()
        .filter(|s| without.contains(*s))
        .cloned()
        .collect();
    if seeds.is_empty() {
        return true;
    }
    !reach_within(g, &seeds, &without).contains(r)
}

/// Every reachable item that `u` dominates.
pub fn dominated_by(
    g: &ContactGraph,
    target_seeds: &BTreeSet<String>,
    u: &str,
    universe: &BTreeSet<String>,
) -> BTreeSet<String> {
    reach_within(g, target_seeds, universe)
        .into_iter()
        .filter(|r| dominates(g, target_seeds, u, r, universe))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeds(xs: &[&str]) -> BTreeSet<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    /// The diamond witness of the executed suite: a goal fanning into `k`
    /// parallel legs that **rejoin at a common resolver** `r`.
    ///
    /// The rejoining is the whole point. Each leg reaches `r`, so no leg is
    /// the only route to anything, so no leg is load-bearing — while every
    /// leg is reachable. Reachability grows with `k`; necessity does not.
    fn diamond(k: usize) -> ContactGraph {
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        g.add_edge("goal", MEDIUM, 1.0).unwrap();
        g.add_edge("r", MEDIUM, 1.0).unwrap();
        for i in 0..k {
            let u = format!("u{i}");
            g.add_edge("goal", &u, 2.0).unwrap();
            g.add_edge(&u, "r", 2.0).unwrap();
            g.add_edge(&u, MEDIUM, 1.0).unwrap();
        }
        g
    }

    /// A path `s0 — s1 — … — s_{n-1}`, each item also touching the medium.
    fn chain(n: usize) -> ContactGraph {
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        for i in 0..n.saturating_sub(1) {
            g.add_edge(&format!("s{i}"), &format!("s{}", i + 1), 2.0)
                .unwrap();
        }
        for i in 0..n {
            g.add_edge(&format!("s{i}"), MEDIUM, 1.0).unwrap();
        }
        g
    }

    #[test]
    fn reach_excludes_the_medium() {
        let g = diamond(2);
        let r = reach(&g, &seeds(&["goal"]));
        assert!(!r.contains(MEDIUM), "the medium must not be reachable");
        assert!(r.contains("goal") && r.contains("u0") && r.contains("r"));
    }

    #[test]
    fn seek_grows_while_nec_stays_flat_on_the_diamond() {
        // thm:separation-operators. seek climbs linearly with the number of
        // parallel legs; nec does not move at all, because no leg is the only
        // route to anything.
        let mut seek_sizes = Vec::new();
        let mut nec_sizes = Vec::new();

        for k in 2..=11 {
            let g = diamond(k);
            let s = seeds(&["goal"]);
            let universe: BTreeSet<String> = g.items().into_iter().collect();
            let r = reach(&g, &s);
            let n = necessary(&g, &s, &universe);
            assert!(n.is_subset(&r), "nec must be contained in reach");
            assert!(
                (0..k).all(|i| r.contains(&format!("u{i}"))),
                "every leg is reachable"
            );
            assert!(
                (0..k).all(|i| !n.contains(&format!("u{i}"))),
                "no leg is load-bearing once the legs rejoin"
            );
            seek_sizes.push(r.len());
            nec_sizes.push(n.len());
        }

        // k legs plus the goal and the resolver.
        assert_eq!(seek_sizes, vec![4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
        assert_eq!(nec_sizes, vec![1; 10], "nec is flat while seek climbs");
    }

    #[test]
    fn the_gap_between_seek_and_nec_is_the_over_retention() {
        // The shaded region of the panel: what a reachability-only mechanism
        // retains for nothing, growing from 3 items to 12.
        let gaps: Vec<usize> = (2..=11)
            .map(|k| {
                let g = diamond(k);
                let s = seeds(&["goal"]);
                let universe: BTreeSet<String> = g.items().into_iter().collect();
                reach(&g, &s).len() - necessary(&g, &s, &universe).len()
            })
            .collect();
        assert_eq!(gaps, vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn on_a_chain_interiors_are_necessary_and_the_leaf_is_redundant() {
        for n in 3..=9 {
            let g = chain(n);
            let s = seeds(&["s0"]);
            let universe: BTreeSet<String> = g.items().into_iter().collect();
            let r = reach(&g, &s);
            let nec = necessary(&g, &s, &universe);

            let leaf = format!("s{}", n - 1);
            assert!(r.contains(&leaf), "the leaf is reachable");
            assert!(
                !nec.contains(&leaf),
                "the terminal leaf is reachable yet redundant"
            );
            for i in 0..n - 1 {
                let interior = format!("s{i}");
                assert!(
                    nec.contains(&interior),
                    "interior {interior} carries the chain and must be necessary"
                );
            }
        }
    }

    #[test]
    fn contribution_excludes_the_dropped_item_from_both_sides() {
        // rem:contribution-care. Were u counted on the full side only, the
        // redundant leaf would score 1 and nec would collapse onto reach.
        let g = chain(3);
        let s = seeds(&["s0"]);
        let r = reach(&g, &s);
        assert_eq!(contribution(&g, &s, &r, "s2"), 0, "leaf contributes nothing");
        assert!(contribution(&g, &s, &r, "s0") > 0, "s0 carries s1 and s2");
        assert!(contribution(&g, &s, &r, "s1") > 0, "s1 is the only route to s2");
    }

    #[test]
    fn an_isolated_seed_is_necessary_though_it_dominates_nothing() {
        // prop:domination is stated for non-seed items; the seed case is
        // answered directly. An isolated seed is the sole way in, and so is
        // load-bearing, while dominating nothing at all.
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        g.add_edge("v0", MEDIUM, 1.0).unwrap();
        let s = seeds(&["v0"]);
        let r = reach(&g, &s);
        assert_eq!(r, seeds(&["v0"]));
        assert!(contribution(&g, &s, &r, "v0") > 0);
        assert_eq!(necessary(&g, &s, &r), seeds(&["v0"]));
        assert!(dominated_by(&g, &s, "v0", &r).is_empty());
    }

    #[test]
    fn necessity_coincides_with_domination_on_non_seed_items() {
        // cor:no-single, checked on both witnesses where the answer is known.
        for g in [chain(5), diamond(4)] {
            let seed = if g.items().iter().any(|i| i == "goal") {
                "goal"
            } else {
                "s0"
            };
            let s = seeds(&[seed]);
            let universe: BTreeSet<String> = g.items().into_iter().collect();
            let nec = necessary(&g, &s, &universe);
            for u in universe.iter().filter(|u| !s.contains(*u)) {
                let doms = dominated_by(&g, &s, u, &universe);
                assert_eq!(
                    nec.contains(u),
                    !doms.is_empty(),
                    "{u}: necessity and domination disagree"
                );
            }
        }
    }

    #[test]
    fn resolution_grows_with_the_retained_set() {
        // rem:orientation — the functional must grow with resolving power.
        let g = chain(5);
        let s = seeds(&["s0"]);
        let full = reach(&g, &s);
        let mut smaller = full.clone();
        smaller.remove("s4");
        assert!(resolution(&g, &s, &full) > resolution(&g, &s, &smaller));
    }
}
