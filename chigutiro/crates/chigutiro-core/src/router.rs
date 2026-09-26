//! Splits a fixed claim budget across receivers.
//!
//! Each receiver's candidates, ordered by coverage, form a gain profile
//! `g_i = coverage_i · decay^i`: non-increasing, hence concave, so taking the
//! `budget` largest marginal gains over all receivers *is* the water-filling
//! allocation (the discrete form of the split-attention rule), and the gain
//! of the last admitted claim is the clearing price `p*`. A dense receiver
//! cannot crowd the others out, because its later candidates pay the decay.
//!
//! The route is logged as a `RouteShape` — which receivers were offered how
//! much and admitted how much, at what price — and never the query text or a
//! claim, so the log can be kept, shared, or inspected without disclosing
//! anything it routed.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::claim::{Candidate, Claim, Grade};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiverShape {
    pub receiver: String,
    pub offered: usize,
    pub admitted: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteShape {
    pub at: DateTime<Utc>,
    pub query_terms: usize,
    pub budget: usize,
    pub price: f64,
    pub grade: Grade,
    pub receivers: Vec<ReceiverShape>,
}

pub fn route(
    offers: Vec<(&'static str, Vec<Candidate>)>,
    query_terms: usize,
    budget: usize,
    decay: f64,
    at: DateTime<Utc>,
) -> (Vec<Claim>, RouteShape) {
    let mut gains: Vec<(f64, usize, usize)> = Vec::new();
    let mut ordered: Vec<(&'static str, Vec<Candidate>)> = Vec::with_capacity(offers.len());
    for (r, (name, mut cands)) in offers.into_iter().enumerate() {
        cands.retain(|c| c.coverage > 0.0);
        cands.sort_by(|a, b| b.coverage.total_cmp(&a.coverage).then(b.rank_score.total_cmp(&a.rank_score)));
        for (i, c) in cands.iter().enumerate() {
            gains.push((c.coverage * decay.powi(i as i32), r, i));
        }
        ordered.push((name, cands));
    }
    // Ties broken by receiver then position, so a route is reproducible.
    gains.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    gains.truncate(budget);

    let mut admitted_per = vec![0usize; ordered.len()];
    let mut claims: Vec<Claim> = gains
        .iter()
        .map(|&(gain, r, i)| {
            admitted_per[r] += 1;
            let c = &ordered[r].1[i];
            Claim {
                receiver: c.receiver,
                text: c.text.clone(),
                grade: Grade::from_support(c.sources.len(), c.contested.is_some()),
                sources: c.sources.iter().cloned().collect(),
                record_ids: c.record_ids.clone(),
                contested: c.contested.clone(),
                gain,
            }
        })
        .collect();
    claims.sort_by(|a, b| b.gain.total_cmp(&a.gain));

    // An answer is only as supported as its weakest admitted claim.
    let grade = claims.iter().map(|c| c.grade).min().unwrap_or(Grade::Declined);
    let shape = RouteShape {
        at,
        query_terms,
        budget,
        price: gains.last().map(|g| g.0).unwrap_or(0.0),
        grade,
        receivers: ordered
            .iter()
            .zip(&admitted_per)
            .map(|((name, cands), &admitted)| ReceiverShape { receiver: name.to_string(), offered: cands.len(), admitted })
            .collect(),
    };
    (claims, shape)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn cand(receiver: &'static str, coverage: f64, source: &str) -> Candidate {
        Candidate {
            receiver,
            text: format!("{receiver} {coverage}"),
            coverage,
            rank_score: 0.0,
            sources: BTreeSet::from([source.to_string()]),
            record_ids: vec![],
            contested: None,
        }
    }

    #[test]
    fn dense_receiver_cannot_crowd_out_others() {
        let text = (0..10).map(|_| cand("text", 1.0, "gmail")).collect();
        let series = vec![cand("series", 0.5, "garmin")];
        let (claims, shape) = route(vec![("text", text), ("series", series)], 2, 4, 0.5, Utc::now());
        // text gains 1, .5, .25, .125 ...; series .5 — series ties text's second slot.
        assert_eq!(claims.len(), 4);
        assert!(claims.iter().any(|c| c.receiver == "series"));
        assert_eq!(shape.receivers[0].admitted, 3);
        assert_eq!(shape.receivers[1].admitted, 1);
        assert_eq!(shape.price, 0.25);
        assert_eq!(shape.grade, Grade::SingleSourced);
    }

    #[test]
    fn nothing_relevant_is_declined() {
        let (claims, shape) = route(vec![("text", vec![cand("text", 0.0, "x")])], 1, 4, 0.5, Utc::now());
        assert!(claims.is_empty());
        assert_eq!(shape.grade, Grade::Declined);
    }
}
