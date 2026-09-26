//! Lexical receiver: BM25 over prose (email, documents, notes). Prose is cut
//! into passages so one long document cannot occupy the whole claim budget
//! with a single score.

use std::collections::{BTreeSet, HashMap};

use crate::claim::Candidate;
use crate::record::Stored;
use crate::terms::{coverage, tokenize};

const PASSAGE_CHARS: usize = 900;
const K1: f64 = 1.2;
const B: f64 = 0.75;

struct Passage {
    record_id: String,
    source: String,
    date: String,
    text: String,
    term_freq: HashMap<String, u32>,
    len: usize,
}

#[derive(Default)]
pub struct TextIndex {
    passages: Vec<Passage>,
    doc_freq: HashMap<String, u32>,
    total_len: usize,
}

impl TextIndex {
    pub fn add(&mut self, stored: &Stored) {
        let Some(surface) = stored.record.prose_surface() else { return };
        for chunk in passages(&surface, PASSAGE_CHARS) {
            let tokens = tokenize(chunk);
            if tokens.is_empty() {
                continue;
            }
            let mut term_freq = HashMap::new();
            for t in &tokens {
                *term_freq.entry(t.clone()).or_insert(0) += 1;
            }
            for t in term_freq.keys() {
                *self.doc_freq.entry(t.clone()).or_insert(0) += 1;
            }
            self.total_len += tokens.len();
            self.passages.push(Passage {
                record_id: stored.id.clone(),
                source: stored.record.source.clone(),
                date: stored.record.ts.format("%Y-%m-%d").to_string(),
                text: chunk.trim().to_string(),
                term_freq,
                len: tokens.len(),
            });
        }
    }

    pub fn len(&self) -> usize {
        self.passages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passages.is_empty()
    }

    /// Passages ranked by BM25. Cross-receiver comparability comes from the
    /// candidate's `coverage`; BM25 only orders passages within this receiver.
    pub fn candidates(&self, query: &BTreeSet<String>, limit: usize) -> Vec<Candidate> {
        if self.passages.is_empty() || query.is_empty() {
            return Vec::new();
        }
        let n = self.passages.len() as f64;
        let avg_len = self.total_len as f64 / n;
        let mut scored: Vec<(f64, &Passage)> = self
            .passages
            .iter()
            .filter_map(|p| {
                let mut score = 0.0;
                for term in query {
                    let Some(&tf) = p.term_freq.get(term) else { continue };
                    let df = f64::from(*self.doc_freq.get(term).unwrap_or(&0));
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    let tf = f64::from(tf);
                    score += idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * p.len as f64 / avg_len));
                }
                (score > 0.0).then_some((score, p))
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored
            .into_iter()
            .take(limit)
            .map(|(score, p)| {
                let terms: BTreeSet<String> = p.term_freq.keys().cloned().collect();
                Candidate {
                    receiver: "text",
                    text: format!("[{} · {}] {}", p.source, p.date, p.text),
                    coverage: coverage(query, &terms),
                    rank_score: score,
                    sources: BTreeSet::from([p.source.clone()]),
                    record_ids: vec![p.record_id.clone()],
                    contested: None,
                }
            })
            .collect()
    }
}

/// Cuts `text` into roughly `max_chars` windows, preferring paragraph then
/// sentence boundaries, never splitting inside a UTF-8 character.
fn passages(text: &str, max_chars: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = text.trim();
    while rest.len() > max_chars {
        let mut cut = max_chars;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        let window = &rest[..cut];
        let cut = window
            .rfind("\n\n")
            .or_else(|| window.rfind(". "))
            .filter(|&i| i > max_chars / 3)
            .map(|i| i + 1)
            .unwrap_or(cut);
        out.push(&rest[..cut]);
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Body, Record};

    fn prose(id: &str, text: &str) -> Stored {
        Stored {
            id: id.into(),
            seq: 1,
            record: Record {
                id: None,
                source: "notes".into(),
                ts: "2026-09-01T00:00:00Z".parse().unwrap(),
                subject: None,
                tags: vec![],
                body: Body::Prose { text: text.into(), title: None, authored_by_owner: true },
            },
        }
    }

    #[test]
    fn ranks_rarer_term_higher() {
        let mut idx = TextIndex::default();
        idx.add(&prose("a", "zanzibar trip in march with the team"));
        idx.add(&prose("b", "the team meeting in march"));
        idx.add(&prose("c", "the team lunch"));
        let q = crate::terms::query_terms("zanzibar march");
        let c = idx.candidates(&q, 5);
        assert_eq!(c[0].record_ids, vec!["a".to_string()]);
        assert_eq!(c[0].coverage, 1.0);
    }

    #[test]
    fn passages_respect_char_boundaries() {
        let text = "ü".repeat(2000);
        let parts = passages(&text, 901);
        assert!(parts.len() > 1);
        assert_eq!(parts.concat(), text);
    }
}
