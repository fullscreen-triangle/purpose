//! Contact-graph receiver: people, their organisation and role, and how
//! often other records reference them. A person is linked to their records
//! through the erasure `subject`, so erasing a subject removes both.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};

use super::{day, relative};
use crate::claim::Candidate;
use crate::record::{Body, Stored};
use crate::terms::{coverage, tokenize};

const VOCAB: &[&str] = &[
    "who", "contact", "contacts", "colleague", "colleagues", "person", "people", "senior", "boss", "manager",
    "team", "kollege", "kollegin", "chef",
];

struct Person {
    name: String,
    subject: Option<String>,
    org: Option<String>,
    role: Option<String>,
    relation: Option<String>,
    sources: BTreeSet<String>,
    record_ids: Vec<String>,
}

#[derive(Default)]
struct Mentions {
    count: usize,
    last: Option<DateTime<Utc>>,
    sources: BTreeSet<String>,
}

#[derive(Default)]
pub struct People {
    /// Keyed by lower-cased name; later contact records refine earlier ones.
    people: BTreeMap<String, Person>,
    /// Keyed by lower-cased subject, over every record kind.
    mentions: BTreeMap<String, Mentions>,
}

impl People {
    pub fn add(&mut self, stored: &Stored) {
        let r = &stored.record;
        if let Some(subject) = &r.subject {
            let m = self.mentions.entry(subject.to_lowercase()).or_default();
            m.count += 1;
            m.last = m.last.max(Some(r.ts));
            m.sources.insert(r.source.clone());
        }
        let Body::Contact { person, org, role, relation } = &r.body else { return };
        let p = self.people.entry(person.to_lowercase()).or_insert_with(|| Person {
            name: person.clone(),
            subject: None,
            org: None,
            role: None,
            relation: None,
            sources: BTreeSet::new(),
            record_ids: Vec::new(),
        });
        // Only fields a newer record actually states overwrite older ones.
        p.subject = r.subject.clone().or(p.subject.take());
        p.org = org.clone().or(p.org.take());
        p.role = role.clone().or(p.role.take());
        p.relation = relation.clone().or(p.relation.take());
        p.sources.insert(r.source.clone());
        p.record_ids.push(stored.id.clone());
    }

    pub fn len(&self) -> usize {
        self.people.len()
    }

    pub fn is_empty(&self) -> bool {
        self.people.is_empty()
    }

    pub fn candidates(&self, query: &BTreeSet<String>, now: DateTime<Utc>) -> Vec<Candidate> {
        self.people
            .values()
            .filter_map(|p| {
                let own: BTreeSet<String> = [Some(&p.name), p.org.as_ref(), p.role.as_ref(), p.relation.as_ref()]
                    .into_iter()
                    .flatten()
                    .flat_map(|s| tokenize(s))
                    .collect();
                // Generic words ("who", "colleague") alone would admit everyone;
                // a person is a candidate only when one of their own terms hits.
                if !own.iter().any(|t| query.contains(t)) {
                    return None;
                }
                let mut vocab = own;
                vocab.extend(VOCAB.iter().map(|s| s.to_string()));

                let mut text = p.name.clone();
                if let Some(role) = &p.role {
                    text.push_str(&format!(" — {role}"));
                }
                if let Some(org) = &p.org {
                    text.push_str(&format!(" at {org}"));
                }
                if let Some(rel) = &p.relation {
                    text.push_str(&format!(" ({rel})"));
                }
                let mut sources = p.sources.clone();
                let mut recency = 0.0;
                if let Some(m) = p.subject.as_ref().and_then(|s| self.mentions.get(&s.to_lowercase())) {
                    if let Some(last) = m.last {
                        text.push_str(&format!(
                            "; referenced by {} record(s), most recently {} ({})",
                            m.count,
                            day(last),
                            relative(last, now)
                        ));
                        recency = -((now - last).num_hours() as f64);
                    }
                    sources.extend(m.sources.iter().cloned());
                }
                Some(Candidate {
                    receiver: "people",
                    text,
                    coverage: coverage(query, &vocab),
                    rank_score: recency,
                    sources,
                    record_ids: p.record_ids.clone(),
                    contested: None,
                })
            })
            .collect()
    }
}
