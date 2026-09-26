//! Events: holidays, journeys, appointments, training blocks.

use std::collections::BTreeSet;

use chrono::{DateTime, Utc};

use super::{day, relative};
use crate::claim::Candidate;
use crate::record::{Body, Stored};
use crate::terms::{coverage, tokenize};

const VOCAB: &[&str] = &[
    "plan", "plans", "planned", "trip", "trips", "holiday", "holidays", "vacation", "upcoming", "next",
    "schedule", "calendar", "event", "events", "travel", "journey", "when", "urlaub", "reise", "termin",
];
const UPCOMING: &[&str] = &["upcoming", "next", "plans", "planned", "schedule", "calendar", "termin"];

struct Event {
    start: DateTime<Utc>,
    end: Option<DateTime<Utc>>,
    title: String,
    place: Option<String>,
    notes: Option<String>,
    tokens: BTreeSet<String>,
    source: String,
    record_id: String,
}

#[derive(Default)]
pub struct Calendar {
    events: Vec<Event>,
}

impl Calendar {
    pub fn add(&mut self, stored: &Stored) {
        let Body::Event { title, end, place, notes } = &stored.record.body else { return };
        let text = [Some(title.as_str()), place.as_deref(), notes.as_deref()].into_iter().flatten().collect::<Vec<_>>().join(" ");
        self.events.push(Event {
            start: stored.record.ts,
            end: *end,
            title: title.clone(),
            place: place.clone(),
            notes: notes.clone(),
            tokens: tokenize(&text).into_iter().collect(),
            source: stored.record.source.clone(),
            record_id: stored.id.clone(),
        });
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn candidates(&self, query: &BTreeSet<String>, now: DateTime<Utc>) -> Vec<Candidate> {
        let wants_upcoming = UPCOMING.iter().any(|t| query.contains(*t));
        let mut out = Vec::new();
        for e in &self.events {
            let own_hit = e.tokens.iter().any(|t| query.contains(t));
            let upcoming = e.start > now || e.end.is_some_and(|end| end > now);
            if !own_hit && !(wants_upcoming && upcoming) {
                continue;
            }
            let mut vocab = e.tokens.clone();
            vocab.extend(VOCAB.iter().map(|s| s.to_string()));
            let mut text = format!("event '{}' {}", e.title, day(e.start));
            if let Some(end) = e.end {
                text.push_str(&format!(" → {}", day(end)));
            }
            if let Some(place) = &e.place {
                text.push_str(&format!(" at {place}"));
            }
            text.push_str(&format!(" ({})", relative(e.start, now)));
            if let Some(notes) = &e.notes {
                text.push_str(&format!(" — {notes}"));
            }
            text.push_str(&format!(" [{}]", e.source));
            out.push(Candidate {
                receiver: "calendar",
                text,
                coverage: coverage(query, &vocab),
                // Soonest upcoming first; past events after all upcoming ones.
                rank_score: if upcoming { -((e.start - now).num_hours() as f64) } else { -1e9 - (now - e.start).num_hours() as f64 },
                sources: BTreeSet::from([e.source.clone()]),
                record_ids: vec![e.record_id.clone()],
                contested: None,
            });
        }
        out
    }
}
