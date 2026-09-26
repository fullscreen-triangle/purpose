//! What receivers return and how support is graded. A claim is never
//! reported as more reliable than the number of independent sources behind
//! it — the four-sided-triangle rule: one source is `single_sourced`, two is
//! `two_sourced`, and only three or more is `grounded`.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// A receiver's proposal, before the router decides whether it is admitted.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub receiver: &'static str,
    pub text: String,
    /// Fraction of distinct query terms this candidate matched — the one
    /// scale on which candidates from different receivers are compared.
    pub coverage: f64,
    /// Receiver-internal ordering (BM25, recency, ...). Never compared across
    /// receivers.
    pub rank_score: f64,
    pub sources: BTreeSet<String>,
    pub record_ids: Vec<String>,
    /// Set when independent sources report disagreeing values.
    pub contested: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grade {
    // Ordered weakest-first, so `min` over claims is the weakest link.
    Declined,
    Contested,
    SingleSourced,
    TwoSourced,
    Grounded,
}

impl Grade {
    pub fn from_support(sources: usize, contested: bool) -> Grade {
        if contested {
            return Grade::Contested;
        }
        match sources {
            0 => Grade::Declined,
            1 => Grade::SingleSourced,
            2 => Grade::TwoSourced,
            _ => Grade::Grounded,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Claim {
    pub receiver: &'static str,
    pub text: String,
    pub grade: Grade,
    pub sources: Vec<String>,
    pub record_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contested: Option<String>,
    /// The marginal gain this claim was admitted at.
    pub gain: f64,
}
