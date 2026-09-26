//! The one input type. Everything the host framework knows about its owner
//! arrives as a `Record`; its `kind` decides which receiver holds it and
//! whether it may ever reach the voice corpus (only owner-authored prose may).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Caller-side id, unique within `source` (a Gmail message id, a Garmin
    /// activity id). Omit it and the record is identified by its content, so
    /// re-sending the same record is still a no-op.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Where the record came from: `garmin`, `gmail`, `bank:dkb`, `brut`, ...
    /// Distinct sources are what count as independent support for a claim.
    pub source: String,
    pub ts: DateTime<Utc>,
    /// The erasure key: the third party (or account) this record is about.
    /// `erase { subject }` removes every record carrying it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(flatten)]
    pub body: Body,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Body {
    /// Email, documents, notes, plans. Only prose with `authored_by_owner`
    /// ever enters the voice corpus; everything else is recall-only.
    Prose {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default)]
        authored_by_owner: bool,
    },
    /// Wearables, phone sensors, athletics, weather: one number at one time.
    Measurement {
        metric: String,
        value: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        unit: Option<String>,
    },
    /// A booked movement of money. Outflows are negative.
    Transaction {
        account: String,
        amount: f64,
        #[serde(default = "default_currency")]
        currency: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        counterparty: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        memo: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        category: Option<String>,
    },
    /// A statement of what an account holds at `ts` — cash, investments, or
    /// debt (debt as a negative amount).
    Balance {
        account: String,
        amount: f64,
        #[serde(default = "default_currency")]
        currency: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        class: Option<String>,
    },
    Position {
        lat: f64,
        lon: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        accuracy_m: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    /// One person in the owner's graph.
    Contact {
        person: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        org: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        role: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        relation: Option<String>,
    },
    /// Holidays, journeys, appointments: something that starts at `ts`.
    Event {
        title: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        end: Option<DateTime<Utc>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        place: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        notes: Option<String>,
    },
}

fn default_currency() -> String {
    "EUR".to_string()
}

impl Body {
    pub fn kind(&self) -> &'static str {
        match self {
            Body::Prose { .. } => "prose",
            Body::Measurement { .. } => "measurement",
            Body::Transaction { .. } => "transaction",
            Body::Balance { .. } => "balance",
            Body::Position { .. } => "position",
            Body::Contact { .. } => "contact",
            Body::Event { .. } => "event",
        }
    }
}

/// Lower-case, trimmed, runs of non-alphanumerics collapsed to `_`, so
/// `Resting Heart-Rate` and `resting_heart_rate` are one series.
pub fn normalize_metric(metric: &str) -> String {
    let mut out = String::with_capacity(metric.len());
    let mut pending_sep = false;
    for c in metric.trim().chars() {
        if c.is_alphanumeric() {
            if pending_sep && !out.is_empty() {
                out.push('_');
            }
            pending_sep = false;
            out.extend(c.to_lowercase());
        } else {
            pending_sep = true;
        }
    }
    out
}

impl Record {
    /// Rejects what would poison a receiver (non-finite numbers, empty text,
    /// impossible coordinates) and canonicalizes metric names.
    pub fn validate(mut self) -> Result<Record, Error> {
        if self.source.trim().is_empty() {
            return Err(Error::InvalidRecord("source is empty".into()));
        }
        self.source = self.source.trim().to_string();
        let finite = |name: &str, v: f64| {
            if v.is_finite() {
                Ok(())
            } else {
                Err(Error::InvalidRecord(format!("{name} is not a finite number")))
            }
        };
        match &mut self.body {
            Body::Prose { text, .. } => {
                if text.trim().is_empty() {
                    return Err(Error::InvalidRecord("prose text is empty".into()));
                }
            }
            Body::Measurement { metric, value, .. } => {
                finite("value", *value)?;
                *metric = normalize_metric(metric);
                if metric.is_empty() {
                    return Err(Error::InvalidRecord("metric is empty".into()));
                }
            }
            Body::Transaction { account, amount, .. } | Body::Balance { account, amount, .. } => {
                finite("amount", *amount)?;
                if account.trim().is_empty() {
                    return Err(Error::InvalidRecord("account is empty".into()));
                }
            }
            Body::Position { lat, lon, .. } => {
                finite("lat", *lat)?;
                finite("lon", *lon)?;
                if !(-90.0..=90.0).contains(lat) || !(-180.0..=180.0).contains(lon) {
                    return Err(Error::InvalidRecord("coordinates out of range".into()));
                }
            }
            Body::Contact { person, .. } => {
                if person.trim().is_empty() {
                    return Err(Error::InvalidRecord("person is empty".into()));
                }
            }
            Body::Event { title, end, .. } => {
                if title.trim().is_empty() {
                    return Err(Error::InvalidRecord("event title is empty".into()));
                }
                if matches!(end, Some(e) if *e < self.ts) {
                    return Err(Error::InvalidRecord("event ends before it starts".into()));
                }
            }
        }
        Ok(self)
    }

    /// Stable identity: `source:id` when the caller supplied an id, otherwise
    /// a hash of the content, so a replayed batch is idempotent either way.
    pub fn canonical_id(&self) -> String {
        match &self.id {
            Some(id) => format!("{}:{}", self.source, id),
            None => {
                let content = serde_json::to_vec(&(&self.source, &self.ts, &self.body))
                    .expect("record body serializes");
                let digest = Sha256::digest(&content);
                format!("{}:#{}", self.source, &hex::encode(digest)[..24])
            }
        }
    }

    /// The text the lexical receiver indexes. Only prose has one: every other
    /// kind is owned by a structured receiver that answers from its fields,
    /// and indexing it twice would let one fact take two claim slots.
    pub fn prose_surface(&self) -> Option<String> {
        match &self.body {
            Body::Prose { text, title: Some(title), .. } if !title.trim().is_empty() => {
                Some(format!("{title} — {text}"))
            }
            Body::Prose { text, .. } => Some(text.clone()),
            _ => None,
        }
    }
}

/// A record as held by the engine: its canonical id and the monotone
/// sequence number it was committed under.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stored {
    pub id: String,
    pub seq: u64,
    pub record: Record,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> DateTime<Utc> {
        s.parse().unwrap()
    }

    #[test]
    fn parses_flattened_kind() {
        let r: Record = serde_json::from_str(
            r#"{"source":"garmin","ts":"2026-09-25T06:00:00Z","kind":"measurement","metric":"HRV (ms)","value":58}"#,
        )
        .unwrap();
        let r = r.validate().unwrap();
        assert_eq!(r.body, Body::Measurement { metric: "hrv_ms".into(), value: 58.0, unit: None });
    }

    #[test]
    fn content_id_is_stable_and_caller_id_wins() {
        let a = Record {
            id: None,
            source: "notes".into(),
            ts: ts("2026-09-01T00:00:00Z"),
            subject: None,
            tags: vec![],
            body: Body::Prose { text: "hello".into(), title: None, authored_by_owner: true },
        };
        assert_eq!(a.canonical_id(), a.clone().canonical_id());
        let b = Record { id: Some("42".into()), ..a };
        assert_eq!(b.canonical_id(), "notes:42");
    }

    #[test]
    fn rejects_poison() {
        let bad = Record {
            id: None,
            source: "x".into(),
            ts: ts("2026-09-01T00:00:00Z"),
            subject: None,
            tags: vec![],
            body: Body::Measurement { metric: "hr".into(), value: f64::NAN, unit: None },
        };
        assert!(bad.validate().is_err());
        assert_eq!(normalize_metric("  Resting Heart-Rate "), "resting_heart_rate");
    }
}
