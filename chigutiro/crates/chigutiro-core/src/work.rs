//! The work corpus: what a *work* model is trained on.
//!
//! This is a different path from the voice corpus. The voice corpus is the
//! owner's own writing and shapes how the personal model sounds. The work
//! corpus is everything in the work scope — chat sessions, academic searches
//! and conversations, uploaded lab reports, papers and presentations,
//! including text others wrote (a paper's authors, an assistant's replies) —
//! and teaches a model the owner's field. It is the only material that may
//! be shipped off this machine for training, so it is redacted the same way
//! as the voice corpus before it is written anywhere.

use serde::Serialize;

use crate::record::{Body, Stored};
use crate::scope::{Scope, ScopePolicy};
use crate::voice::redact;

/// Below this, a record carries too little text to teach anything.
const MIN_WORK_CHARS: usize = 40;

#[derive(Debug, Clone, Serialize)]
pub struct WorkDoc {
    pub id: String,
    pub seq: u64,
    pub source: String,
    pub text: String,
}

pub fn work_doc(stored: &Stored, policy: &ScopePolicy) -> Option<WorkDoc> {
    if policy.scope_of(&stored.record) != Scope::Work {
        return None;
    }
    let Body::Prose { text, title, .. } = &stored.record.body else { return None };
    let text = match title.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => format!("{t}\n\n{}", text.trim()),
        None => text.trim().to_string(),
    };
    let text = redact(&text);
    (text.chars().count() >= MIN_WORK_CHARS).then(|| WorkDoc {
        id: stored.id.clone(),
        seq: stored.seq,
        source: stored.record.source.clone(),
        text,
    })
}
