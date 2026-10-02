//! Scopes: which part of the owner's profile a record belongs to.
//!
//! A chigutiro profile is one person's whole life, but only a narrow slice
//! of it is *work*: the slice an absicht federation may consult and that may
//! be trained into a work model off this machine. Everything else is
//! *personal* and never leaves.
//!
//! The scope is decided by the record's **source channel**, never by what
//! the record says: a content classifier eventually lets a personal message
//! through, a channel rule cannot. It is also decided when a record is
//! read, not stored, so narrowing the policy takes effect at once for every
//! record already in the log. The host labels channels honestly
//! (`chat:assistant`, `upload:paper`); chigutiro decides what they mean.

use serde::{Deserialize, Serialize};

use crate::record::{Body, Record};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Personal,
    Work,
}

/// Source channels whose prose is work. A channel matches a source equal to
/// it or starting with it followed by `:`, so `chat` admits `chat:assistant`
/// but not `chatter`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopePolicy {
    pub work_channels: Vec<String>,
}

/// Work, for now, is exactly:
/// - `chat`: chat session history with an assistant;
/// - `academic`: academic searches and conversations about them;
/// - `upload:lab-report`, `upload:paper`, `upload:presentation`: documents
///   the owner uploaded.
///
/// University email, notes and everything else stay personal until added.
pub const DEFAULT_WORK_CHANNELS: &[&str] =
    &["chat", "academic", "upload:lab-report", "upload:paper", "upload:presentation"];

impl Default for ScopePolicy {
    fn default() -> Self {
        ScopePolicy { work_channels: DEFAULT_WORK_CHANNELS.iter().map(|s| s.to_string()).collect() }
    }
}

impl ScopePolicy {
    /// Work iff the record is prose from a work channel. Measurements, money,
    /// places, people and events are personal whatever their source.
    pub fn scope_of(&self, record: &Record) -> Scope {
        let prose = matches!(record.body, Body::Prose { .. });
        if prose && self.work_channels.iter().any(|c| channel_matches(c, &record.source)) {
            Scope::Work
        } else {
            Scope::Personal
        }
    }
}

fn channel_matches(channel: &str, source: &str) -> bool {
    let channel = channel.trim().to_ascii_lowercase();
    let source = source.trim().to_ascii_lowercase();
    !channel.is_empty()
        && (source == channel || source.strip_prefix(&channel).is_some_and(|rest| rest.starts_with(':')))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prose(source: &str) -> Record {
        Record {
            id: None,
            source: source.into(),
            ts: "2026-10-02T09:00:00Z".parse().unwrap(),
            subject: None,
            tags: vec![],
            body: Body::Prose { text: "text".into(), title: None, authored_by_owner: false },
        }
    }

    #[test]
    fn only_the_three_work_channels_are_work() {
        let p = ScopePolicy::default();
        for work in ["chat", "chat:assistant", "academic:semantic-scholar", "upload:paper", "Upload:Lab-Report", "upload:presentation:deck-7"] {
            assert_eq!(p.scope_of(&prose(work)), Scope::Work, "{work}");
        }
        for personal in ["gmail", "uni-mail", "upload", "upload:photo", "chatter", "notes", "academy", "upload:paperwork"] {
            assert_eq!(p.scope_of(&prose(personal)), Scope::Personal, "{personal}");
        }
    }

    #[test]
    fn non_prose_is_personal_even_from_a_work_channel() {
        let mut r = prose("upload:lab-report");
        r.body = Body::Measurement { metric: "yield".into(), value: 0.8, unit: None };
        assert_eq!(ScopePolicy::default().scope_of(&r), Scope::Personal);
    }

    #[test]
    fn an_empty_policy_makes_everything_personal() {
        let p = ScopePolicy { work_channels: vec![] };
        assert_eq!(p.scope_of(&prose("chat")), Scope::Personal);
    }
}
