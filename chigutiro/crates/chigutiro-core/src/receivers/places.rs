//! Geolocation receiver: the latest fix, and labelled places by visit count.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};

use super::{day, relative};
use crate::claim::Candidate;
use crate::record::{Body, Stored};
use crate::terms::{coverage, tokenize};

const VOCAB: &[&str] = &["where", "location", "position", "place", "places", "gps", "last", "seen", "wo", "ort"];

struct Fix {
    ts: DateTime<Utc>,
    lat: f64,
    lon: f64,
    accuracy_m: Option<f64>,
    label: Option<String>,
    source: String,
    record_id: String,
}

#[derive(Default)]
pub struct Places {
    fixes: Vec<Fix>,
}

impl Places {
    pub fn add(&mut self, stored: &Stored) {
        let Body::Position { lat, lon, accuracy_m, label } = &stored.record.body else { return };
        self.fixes.push(Fix {
            ts: stored.record.ts,
            lat: *lat,
            lon: *lon,
            accuracy_m: *accuracy_m,
            label: label.clone(),
            source: stored.record.source.clone(),
            record_id: stored.id.clone(),
        });
    }

    pub fn len(&self) -> usize {
        self.fixes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fixes.is_empty()
    }

    pub fn candidates(&self, query: &BTreeSet<String>, now: DateTime<Utc>) -> Vec<Candidate> {
        let mut out = Vec::new();
        let vocab: BTreeSet<String> = VOCAB.iter().map(|s| s.to_string()).collect();
        let past: Vec<&Fix> = self.fixes.iter().filter(|f| f.ts <= now).collect();

        if let Some(f) = past.iter().max_by_key(|f| f.ts) {
            let cov = coverage(query, &vocab);
            if cov > 0.0 {
                let acc = f.accuracy_m.map(|a| format!(" ±{a:.0} m")).unwrap_or_default();
                let at = f.label.as_deref().map(|l| format!(" ({l})")).unwrap_or_default();
                out.push(Candidate {
                    receiver: "places",
                    text: format!(
                        "last position {:.5}, {:.5}{acc}{at} at {} {} ({}) [{}]",
                        f.lat,
                        f.lon,
                        day(f.ts),
                        f.ts.format("%H:%M UTC"),
                        relative(f.ts, now),
                        f.source
                    ),
                    coverage: cov,
                    rank_score: 1.0,
                    sources: BTreeSet::from([f.source.clone()]),
                    record_ids: vec![f.record_id.clone()],
                    contested: None,
                });
            }
        }

        let mut labelled: BTreeMap<&str, Vec<&Fix>> = BTreeMap::new();
        for f in &past {
            if let Some(l) = f.label.as_deref() {
                labelled.entry(l).or_default().push(f);
            }
        }
        for (label, fixes) in labelled {
            let mut v = vocab.clone();
            v.extend(tokenize(label));
            let label_hit = tokenize(label).iter().any(|t| query.contains(t));
            if !label_hit {
                continue;
            }
            let last = fixes.iter().max_by_key(|f| f.ts).expect("non-empty");
            out.push(Candidate {
                receiver: "places",
                text: format!("place '{label}': {} fixes, last on {} ({})", fixes.len(), day(last.ts), relative(last.ts, now)),
                coverage: coverage(query, &v),
                rank_score: fixes.len() as f64,
                sources: fixes.iter().map(|f| f.source.clone()).collect(),
                record_ids: fixes.iter().rev().take(32).map(|f| f.record_id.clone()).collect(),
                contested: None,
            });
        }
        out
    }
}
