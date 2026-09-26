//! The fast timescale. Every accepted record is indexed by its receiver
//! before `ingest` returns, so the next question already sees it.

use std::collections::{BTreeMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::claim::{Claim, Grade};
use crate::error::Error;
use crate::receivers::{calendar::Calendar, ledger::Ledger, people::People, places::Places, series::Series, text::TextIndex};
use crate::record::{Record, Stored};
use crate::router::{route, RouteShape};
use crate::terms::query_terms;
use crate::voice::{voice_doc, VoiceDoc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Claims admitted per answer.
    pub budget: usize,
    /// Per-receiver diminishing return; see `router`.
    pub decay: f64,
    /// Relative spread at which same-day readings from different sources are
    /// reported as contested.
    pub contest_tolerance: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig { budget: 8, decay: 0.5, contest_tolerance: 0.15 }
    }
}

pub enum Ingested {
    Accepted(Stored),
    Duplicate(String),
}

/// Which records to erase. Every field given must match (AND); at least one
/// must be given.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Erase {
    #[serde(default)]
    pub ids: Vec<String>,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub before: Option<DateTime<Utc>>,
}

impl Erase {
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty() && self.subject.is_none() && self.source.is_none() && self.before.is_none()
    }

    fn matches(&self, s: &Stored) -> bool {
        (self.ids.is_empty() || self.ids.contains(&s.id))
            && self.subject.as_ref().is_none_or(|want| {
                s.record.subject.as_ref().is_some_and(|have| have.eq_ignore_ascii_case(want))
            })
            && self.source.as_ref().is_none_or(|want| &s.record.source == want)
            && self.before.is_none_or(|b| s.record.ts < b)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Retrieval {
    pub grade: Grade,
    pub claims: Vec<Claim>,
    pub route: RouteShape,
}

#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub committed: u64,
    pub held: usize,
    pub by_kind: BTreeMap<String, usize>,
    pub passages: usize,
    pub metrics: usize,
    pub measurements: usize,
    pub accounts: usize,
    pub positions: usize,
    pub people: usize,
    pub events: usize,
    pub voice_docs: usize,
}

pub struct Engine {
    cfg: EngineConfig,
    records: Vec<Stored>,
    ids: HashSet<String>,
    committed: u64,
    text: TextIndex,
    series: Series,
    ledger: Ledger,
    places: Places,
    people: People,
    calendar: Calendar,
}

impl Engine {
    pub fn new(cfg: EngineConfig) -> Engine {
        let series = Series::new(cfg.contest_tolerance);
        Engine {
            cfg,
            records: Vec::new(),
            ids: HashSet::new(),
            committed: 0,
            text: TextIndex::default(),
            series,
            ledger: Ledger::default(),
            places: Places::default(),
            people: People::default(),
            calendar: Calendar::default(),
        }
    }

    /// Rebuilds from a replayed log. `committed` comes from state, since
    /// erased records no longer appear in the log but still count.
    pub fn restore(cfg: EngineConfig, records: Vec<Stored>, committed: u64) -> Engine {
        let mut e = Engine::new(cfg);
        for s in records {
            e.index(&s);
            e.ids.insert(s.id.clone());
            e.committed = e.committed.max(s.seq);
            e.records.push(s);
        }
        e.committed = e.committed.max(committed);
        e
    }

    fn index(&mut self, s: &Stored) {
        self.text.add(s);
        self.series.add(s);
        self.ledger.add(s);
        self.places.add(s);
        self.people.add(s);
        self.calendar.add(s);
    }

    pub fn committed(&self) -> u64 {
        self.committed
    }

    pub fn records(&self) -> &[Stored] {
        &self.records
    }

    pub fn ingest(&mut self, record: Record) -> Result<Ingested, Error> {
        let record = record.validate()?;
        let id = record.canonical_id();
        if self.ids.contains(&id) {
            return Ok(Ingested::Duplicate(id));
        }
        self.committed += 1;
        let stored = Stored { id: id.clone(), seq: self.committed, record };
        self.index(&stored);
        self.ids.insert(id);
        self.records.push(stored.clone());
        Ok(Ingested::Accepted(stored))
    }

    pub fn ask(&self, query: &str, now: DateTime<Utc>, budget: Option<usize>) -> Retrieval {
        let q = query_terms(query);
        let offers = vec![
            ("text", self.text.candidates(&q, 32)),
            ("series", self.series.candidates(&q, now)),
            ("ledger", self.ledger.candidates(&q, now)),
            ("places", self.places.candidates(&q, now)),
            ("people", self.people.candidates(&q, now)),
            ("calendar", self.calendar.candidates(&q, now)),
        ];
        let (claims, route) = route(offers, q.len(), budget.unwrap_or(self.cfg.budget), self.cfg.decay, now);
        Retrieval { grade: route.grade, claims, route }
    }

    /// Removes matching records and rebuilds every receiver without them.
    /// Returns the removed records; the caller rewrites the log.
    pub fn erase(&mut self, criteria: &Erase) -> Vec<Stored> {
        if criteria.is_empty() {
            return Vec::new();
        }
        let (removed, kept): (Vec<Stored>, Vec<Stored>) =
            std::mem::take(&mut self.records).into_iter().partition(|s| criteria.matches(s));
        if removed.is_empty() {
            self.records = kept;
            return removed;
        }
        *self = Engine::restore(self.cfg.clone(), kept, self.committed);
        removed
    }

    pub fn voice_docs(&self) -> Vec<VoiceDoc> {
        self.records.iter().filter_map(voice_doc).collect()
    }

    pub fn stats(&self) -> Stats {
        let mut by_kind = BTreeMap::new();
        for s in &self.records {
            *by_kind.entry(s.record.body.kind().to_string()).or_insert(0) += 1;
        }
        Stats {
            committed: self.committed,
            held: self.records.len(),
            by_kind,
            passages: self.text.len(),
            metrics: self.series.metric_count(),
            measurements: self.series.point_count(),
            accounts: self.ledger.account_count(),
            positions: self.places.len(),
            people: self.people.len(),
            events: self.calendar.len(),
            voice_docs: self.voice_docs().len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Body;

    fn rec(source: &str, ts: &str, subject: Option<&str>, body: Body) -> Record {
        Record { id: None, source: source.into(), ts: ts.parse().unwrap(), subject: subject.map(String::from), tags: vec![], body }
    }

    fn hr(source: &str, ts: &str, v: f64) -> Record {
        rec(source, ts, None, Body::Measurement { metric: "resting heart rate".into(), value: v, unit: Some("bpm".into()) })
    }

    #[test]
    fn continuous_ingest_is_visible_to_the_next_question() {
        let now: DateTime<Utc> = "2026-09-26T12:00:00Z".parse().unwrap();
        let mut e = Engine::new(EngineConfig::default());
        assert_eq!(e.ask("resting heart rate", now, None).grade, Grade::Declined);

        e.ingest(hr("garmin", "2026-09-25T06:00:00Z", 48.0)).unwrap();
        let r = e.ask("resting heart rate", now, None);
        assert_eq!(r.grade, Grade::SingleSourced);

        e.ingest(hr("polar", "2026-09-24T06:00:00Z", 47.0)).unwrap();
        e.ingest(hr("manual-log", "2026-09-23T06:00:00Z", 49.0)).unwrap();
        let r = e.ask("resting heart rate", now, None);
        assert_eq!(r.grade, Grade::Grounded, "{:?}", r.claims);
        // Re-sending a record is a no-op, not a second reading.
        assert!(matches!(e.ingest(hr("garmin", "2026-09-25T06:00:00Z", 48.0)).unwrap(), Ingested::Duplicate(_)));
        assert_eq!(e.committed(), 3);
    }

    #[test]
    fn erasing_a_subject_removes_it_from_every_receiver_but_not_from_the_count() {
        let now: DateTime<Utc> = "2026-09-26T12:00:00Z".parse().unwrap();
        let mut e = Engine::new(EngineConfig::default());
        e.ingest(rec("hr", "2026-09-01T00:00:00Z", Some("m.doerr@uni.de"), Body::Contact {
            person: "Mark Dörr".into(), org: Some("Uni Greifswald".into()), role: Some("group leader".into()), relation: None,
        }))
        .unwrap();
        e.ingest(rec("gmail", "2026-09-20T00:00:00Z", Some("m.doerr@uni.de"), Body::Prose {
            text: "Kundai, the cluster allocation for the enzyme screen is approved.".into(), title: None, authored_by_owner: false,
        }))
        .unwrap();
        assert_eq!(e.ask("dörr greifswald", now, None).claims.len(), 1);
        assert!(!e.ask("cluster allocation", now, None).claims.is_empty());

        let removed = e.erase(&Erase { subject: Some("M.DOERR@uni.de".into()), ..Default::default() });
        assert_eq!(removed.len(), 2);
        assert_eq!(e.ask("dörr greifswald", now, None).grade, Grade::Declined);
        assert_eq!(e.ask("cluster allocation", now, None).grade, Grade::Declined);
        assert_eq!(e.committed(), 2);
        // Other people's prose never entered the voice corpus in the first place.
        assert!(e.voice_docs().is_empty());
    }
}
