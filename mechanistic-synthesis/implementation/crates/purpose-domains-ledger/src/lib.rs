//! Residue-ledger: carry the uncertainty, not the knowledge.
//!
//! This crate is the executable form of a single principle drawn from the
//! framework's epistemology papers: a finite agent should carry forward its
//! *residue* (the specific unknowability driving the next step) rather than
//! its *knowledge* (the disposable, re-individuable surface).
//!
//! ## Model
//!
//! A conversation is a monotone, append-only sequence of **turns** (the
//! trajectory count only grows — `computational-systems-structure`,
//! Thm. "Strict monotonicity"). Each turn deposits **residue**: a weight
//! `>= β` (the floor) and the set of *terms* it individuates — the
//! distinctions it drew.
//!
//! A **goal** is the current step's intent: a set of terms it must resolve.
//!
//! **Necessity** (orchestra paper, "Contribution decides necessity"): a turn
//! is *necessary* iff removing it changes what the goal can reach. Concretely,
//! a turn contributes iff its residue is reachable from the goal through the
//! term-overlap graph. Turns whose residue no longer connects to the current
//! goal have contribution 0 — they are *purposeless* (correct, but the goal
//! does not need them) and may be pruned without changing the reachable answer.
//!
//! **What you carry** is the necessary residue-slice, not the full history.
//! Dropping the rest costs nothing against the invariant (an *internal*
//! operation deposits zero graph-residue — `computational-systems-structure`,
//! Lemma "Type-specific residue bounds").
//!
//! The computation is deterministic — reachability plus term overlap on a
//! finite graph. No model reads meaning; the ledger holds the goal and decides
//! necessity, the turns merely report their residue (the "part reports,
//! layer-above decides" invariant, orchestra "Necessity is the layer above").

use std::collections::{BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

/// The floor β: the irreducible residue every genuine turn deposits.
/// A turn below this is not individuated at all (it drew no distinction).
/// Derived, not posited — the trace of the non-completable whole on its parts.
pub const FLOOR: f64 = 1.0;

/// One turn of accumulated context: a residue node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Monotone position in the conversation (the trajectory count).
    pub id: usize,
    /// Optional human label — the *knowledge* face. Relabellable, disposable;
    /// carried only for display, never used in the necessity computation.
    #[serde(default)]
    pub label: String,
    /// The terms this turn individuates — the distinctions it drew.
    /// This is the residue's *shape*; necessity is computed over these alone.
    pub terms: Vec<String>,
    /// Token cost of the turn's full text (what you pay to carry it verbatim).
    pub tokens: usize,
}

/// The accumulated context plus the current step's goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ledger {
    /// All turns so far, in monotone order.
    pub turns: Vec<Turn>,
    /// The current step's intent: the terms it must resolve.
    pub goal: Vec<String>,
}

/// A turn's necessity verdict for the current goal.
#[derive(Debug, Clone, Serialize)]
pub struct Verdict {
    pub id: usize,
    pub label: String,
    /// Residue weight of the turn (>= FLOOR for any genuine turn).
    pub residue: f64,
    /// Graph distance from the goal (0 = a turn touching a goal term directly).
    /// `None` means unreachable from the goal — contribution 0.
    pub reach: Option<usize>,
    /// Necessary iff reachable from the goal.
    pub necessary: bool,
    pub tokens: usize,
}

/// The result of clipping the ledger to the goal's necessary slice.
#[derive(Debug, Clone, Serialize)]
pub struct Clip {
    pub verdicts: Vec<Verdict>,
    /// Total tokens if the whole history were carried verbatim.
    pub tokens_full: usize,
    /// Tokens actually needed (the necessary slice).
    pub tokens_kept: usize,
    /// Tokens dropped as purposeless for this goal.
    pub tokens_dropped: usize,
}

impl Turn {
    /// The residue weight a turn deposits: the floor plus one unit per
    /// distinction drawn. A turn that draws no distinction sits at the floor
    /// and individuates nothing — it is not load-bearing on its own.
    pub fn residue(&self) -> f64 {
        FLOOR + self.terms.len() as f64
    }

    fn term_set(&self) -> BTreeSet<&str> {
        self.terms.iter().map(|s| s.as_str()).collect()
    }
}

impl Ledger {
    /// Compute, for the current goal, which turns are necessary to carry.
    ///
    /// A turn is necessary iff its residue is reachable from the goal through
    /// the term-overlap graph: turns that share a term with the goal are at
    /// distance 0, turns sharing a term with *those* are at distance 1, and so
    /// on. A turn unreachable from the goal contributes nothing to what the
    /// goal can resolve, so dropping it does not change the reachable answer.
    pub fn clip(&self) -> Clip {
        // Seed the frontier with goal terms.
        let goal_terms: BTreeSet<&str> = self.goal.iter().map(|s| s.as_str()).collect();

        // BFS over turns by shared-term adjacency, starting from the goal.
        // `reach[i]` = graph distance of turn i from the goal, if reachable.
        let n = self.turns.len();
        let mut reach: Vec<Option<usize>> = vec![None; n];
        let mut frontier_terms: BTreeSet<&str> = goal_terms.clone();
        let mut queue: VecDeque<usize> = VecDeque::new();

        // Distance-0 layer: any turn touching a goal term.
        for (i, t) in self.turns.iter().enumerate() {
            if !t.term_set().is_disjoint(&goal_terms) {
                reach[i] = Some(0);
                queue.push_back(i);
                for term in &t.terms {
                    frontier_terms.insert(term.as_str());
                }
            }
        }

        // Expand: a turn reachable at distance d+1 if it shares a term with any
        // turn already reached, and is not yet reached.
        while let Some(i) = queue.pop_front() {
            let d = reach[i].unwrap();
            let reached_terms: BTreeSet<&str> = self.turns[i].term_set();
            for (j, t) in self.turns.iter().enumerate() {
                if reach[j].is_some() {
                    continue;
                }
                if !t.term_set().is_disjoint(&reached_terms) {
                    reach[j] = Some(d + 1);
                    queue.push_back(j);
                }
            }
        }

        let mut verdicts = Vec::with_capacity(n);
        let mut tokens_full = 0usize;
        let mut tokens_kept = 0usize;
        for (i, t) in self.turns.iter().enumerate() {
            let necessary = reach[i].is_some();
            tokens_full += t.tokens;
            if necessary {
                tokens_kept += t.tokens;
            }
            verdicts.push(Verdict {
                id: t.id,
                label: t.label.clone(),
                residue: t.residue(),
                reach: reach[i],
                necessary,
                tokens: t.tokens,
            });
        }

        Clip {
            verdicts,
            tokens_full,
            tokens_kept,
            tokens_dropped: tokens_full - tokens_kept,
        }
    }
}

/// Parse a ledger from JSON.
pub fn parse(text: &str) -> Result<Ledger, purpose_core::Error> {
    serde_json::from_str(text)
        .map_err(|e| purpose_core::Error::Parse(format!("bad ledger json: {e}")))
}

/// Render a clip as a human-readable report.
pub fn render(clip: &Clip) -> String {
    let mut out = String::new();
    out.push_str("Residue ledger — necessity for the current goal\n");
    out.push_str("(a turn is carried only if its residue is reachable from the goal)\n\n");

    let mut kept: Vec<&Verdict> = clip.verdicts.iter().filter(|v| v.necessary).collect();
    kept.sort_by_key(|v| (v.reach.unwrap_or(usize::MAX), v.id));

    out.push_str("KEEP (necessary):\n");
    if kept.is_empty() {
        out.push_str("  (nothing — the goal shares no term with any turn)\n");
    }
    for v in &kept {
        out.push_str(&format!(
            "  #{:<3} reach={} residue={:.0}  {} [{} tok]\n",
            v.id,
            v.reach.unwrap(),
            v.residue,
            if v.label.is_empty() { "(unlabelled)" } else { &v.label },
            v.tokens,
        ));
    }

    let dropped: Vec<&Verdict> = clip.verdicts.iter().filter(|v| !v.necessary).collect();
    out.push_str("\nDROP (purposeless for this goal):\n");
    if dropped.is_empty() {
        out.push_str("  (none — every turn is load-bearing)\n");
    }
    for v in &dropped {
        out.push_str(&format!(
            "  #{:<3} residue={:.0}  {} [{} tok]\n",
            v.id,
            v.residue,
            if v.label.is_empty() { "(unlabelled)" } else { &v.label },
            v.tokens,
        ));
    }

    let pct = if clip.tokens_full > 0 {
        100.0 * clip.tokens_dropped as f64 / clip.tokens_full as f64
    } else {
        0.0
    };
    out.push_str(&format!(
        "\nTokens: carry {} of {} ({} dropped, {:.0}% saved)\n",
        clip.tokens_kept, clip.tokens_full, clip.tokens_dropped, pct,
    ));
    out
}
