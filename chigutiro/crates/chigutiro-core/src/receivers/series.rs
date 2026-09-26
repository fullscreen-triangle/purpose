//! Numeric receiver for measurements: wearables, phone sensors, athletics,
//! weather. It answers from windows computed at query time, never from a
//! stored summary, so every ingest is reflected by the next question.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, Utc};

use super::{day, num, relative};
use crate::claim::Candidate;
use crate::record::{Body, Stored};
use crate::terms::{coverage, tokenize};

struct Point {
    ts: DateTime<Utc>,
    value: f64,
    source: String,
    record_id: String,
}

#[derive(Default)]
struct Metric {
    unit: Option<String>,
    /// Sorted by `ts`.
    points: Vec<Point>,
}

pub struct Series {
    metrics: BTreeMap<String, Metric>,
    /// Relative spread above which same-day readings from different sources
    /// are reported as contested. A declared tolerance, not a derived one —
    /// it is surfaced in the claim text so it is never mistaken for one.
    contest_tolerance: f64,
}

impl Series {
    pub fn new(contest_tolerance: f64) -> Self {
        Series { metrics: BTreeMap::new(), contest_tolerance }
    }

    pub fn add(&mut self, stored: &Stored) {
        let Body::Measurement { metric, value, unit } = &stored.record.body else { return };
        let m = self.metrics.entry(metric.clone()).or_default();
        if m.unit.is_none() {
            m.unit = unit.clone();
        }
        let ts = stored.record.ts;
        let at = m.points.partition_point(|p| p.ts <= ts);
        m.points.insert(
            at,
            Point { ts, value: *value, source: stored.record.source.clone(), record_id: stored.id.clone() },
        );
    }

    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }

    pub fn point_count(&self) -> usize {
        self.metrics.values().map(|m| m.points.len()).sum()
    }

    pub fn candidates(&self, query: &BTreeSet<String>, now: DateTime<Utc>) -> Vec<Candidate> {
        self.metrics
            .iter()
            .filter_map(|(name, m)| {
                let mut vocab: BTreeSet<String> = tokenize(&name.replace('_', " ")).into_iter().collect();
                if let Some(u) = &m.unit {
                    vocab.extend(tokenize(u));
                }
                let cov = coverage(query, &vocab);
                (cov > 0.0).then(|| self.summarize(name, m, now, cov)).flatten()
            })
            .collect()
    }

    fn summarize(&self, name: &str, m: &Metric, now: DateTime<Utc>, cov: f64) -> Option<Candidate> {
        // Readings stamped in the future are not yet observations.
        let past: Vec<&Point> = m.points.iter().filter(|p| p.ts <= now).collect();
        let latest = *past.last()?;
        let unit = m.unit.as_deref().map(|u| format!(" {u}")).unwrap_or_default();
        let window = |from_days: i64, to_days: i64| -> Vec<&Point> {
            let lo = now - Duration::days(from_days);
            let hi = now - Duration::days(to_days);
            past.iter().copied().filter(|p| p.ts > lo && p.ts <= hi).collect()
        };
        let mean = |pts: &[&Point]| pts.iter().map(|p| p.value).sum::<f64>() / pts.len() as f64;

        let mut parts = vec![format!(
            "{name}: latest {}{unit} on {} ({}) [{}]",
            num(latest.value),
            day(latest.ts),
            relative(latest.ts, now),
            latest.source
        )];
        let last7 = window(7, 0);
        let prior7 = window(14, 7);
        let last28 = window(28, 0);
        if last7.is_empty() {
            parts.push("no readings in the last 7 days".into());
        } else {
            let m7 = mean(&last7);
            let mut s = format!("7-day mean {}{unit} (n={})", num(m7), last7.len());
            if !prior7.is_empty() {
                let mp = mean(&prior7);
                s.push_str(&format!("; prior 7-day mean {} (change {:+})", num(mp), num(m7 - mp)));
            }
            parts.push(s);
        }
        if last28.len() > last7.len() {
            parts.push(format!("28-day mean {}{unit} (n={})", num(mean(&last28)), last28.len()));
        }

        let contested = self.contest(&past, latest.ts, &unit);
        let support: Vec<&Point> = if last28.is_empty() { vec![latest] } else { last28 };
        Some(Candidate {
            receiver: "series",
            text: parts.join("; "),
            coverage: cov,
            // Fresher series first among equally covered ones.
            rank_score: -((now - latest.ts).num_hours() as f64),
            sources: support.iter().map(|p| p.source.clone()).collect(),
            record_ids: support.iter().rev().take(32).map(|p| p.record_id.clone()).collect(),
            contested,
        })
    }

    /// On the latest day with readings, compares each source's last value.
    fn contest(&self, past: &[&Point], latest: DateTime<Utc>, unit: &str) -> Option<String> {
        let d = latest.date_naive();
        let mut by_source: BTreeMap<&str, f64> = BTreeMap::new();
        for p in past.iter().filter(|p| p.ts.date_naive() == d) {
            by_source.insert(&p.source, p.value);
        }
        if by_source.len() < 2 {
            return None;
        }
        let vals: Vec<f64> = by_source.values().copied().collect();
        let (lo, hi) = vals.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let scale = (vals.iter().sum::<f64>() / vals.len() as f64).abs().max(f64::EPSILON);
        let spread = (hi - lo) / scale;
        (spread > self.contest_tolerance).then(|| {
            let listed: Vec<String> = by_source.iter().map(|(s, v)| format!("{s} {}{unit}", num(*v))).collect();
            format!(
                "on {d}: {} (spread {:.0}% > declared tolerance {:.0}%)",
                listed.join(", "),
                spread * 100.0,
                self.contest_tolerance * 100.0
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;

    fn m(seq: u64, source: &str, ts: &str, metric: &str, value: f64) -> Stored {
        Stored {
            id: format!("r{seq}"),
            seq,
            record: Record {
                id: None,
                source: source.into(),
                ts: ts.parse().unwrap(),
                subject: None,
                tags: vec![],
                body: Body::Measurement { metric: metric.into(), value, unit: Some("ms".into()) },
            },
        }
    }

    #[test]
    fn windows_are_relative_to_now_and_contest_is_detected() {
        let mut s = Series::new(0.15);
        s.add(&m(1, "garmin", "2026-09-10T06:00:00Z", "hrv", 64.0));
        s.add(&m(2, "garmin", "2026-09-20T06:00:00Z", "hrv", 60.0));
        s.add(&m(3, "garmin", "2026-09-25T06:00:00Z", "hrv", 58.0));
        s.add(&m(4, "brut", "2026-09-25T07:00:00Z", "hrv", 75.0));
        let now: DateTime<Utc> = "2026-09-26T12:00:00Z".parse().unwrap();
        let c = s.candidates(&crate::terms::query_terms("hrv trend"), now);
        assert_eq!(c.len(), 1);
        let c = &c[0];
        assert_eq!(c.coverage, 0.5);
        assert!(c.text.contains("latest 75.0 ms on 2026-09-25 (yesterday) [brut]"), "{}", c.text);
        assert!(c.text.contains("7-day mean 64.3 ms (n=3)"), "{}", c.text);
        assert!(c.contested.as_deref().unwrap().contains("brut 75.0 ms, garmin 58.0 ms"));
        assert_eq!(c.sources.len(), 2);
    }

    #[test]
    fn stale_series_says_so() {
        let mut s = Series::new(0.15);
        s.add(&m(1, "garmin", "2026-08-01T06:00:00Z", "resting_heart_rate", 48.0));
        let now: DateTime<Utc> = "2026-09-26T12:00:00Z".parse().unwrap();
        let c = s.candidates(&crate::terms::query_terms("resting heart rate"), now);
        assert!(c[0].text.contains("no readings in the last 7 days"));
        assert!(c[0].text.contains("56 days ago"));
    }
}
