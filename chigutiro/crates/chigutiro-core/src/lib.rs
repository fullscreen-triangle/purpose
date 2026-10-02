//! Chigutiro core: a personal model that learns on two timescales.
//!
//! * **Fast** — every record is committed to an append-only log and indexed
//!   by a typed receiver (text, series, ledger, places, people, calendar)
//!   before `ingest` returns. Questions are answered from claims those
//!   receivers compute at query time, routed by water-filling and graded by
//!   independent support.
//! * **Slow** — the owner's own prose accumulates as a redacted voice corpus;
//!   `consolidate` retrains model weights on it in rounds.
//!
//! Only owner-authored prose reaches the *personal* model's weights. Numbers,
//! money, places and other people stay in receivers, where they can be
//! current, exact, and erased.
//!
//! A narrow *work* scope — chosen by source channel, see `scope` — is the
//! only part of the profile that may leave this machine: it can be asked on
//! its own (`Engine::ask_scoped`) and exported as a work corpus (`work`).

pub mod claim;
pub mod consolidate;
pub mod crypto;
pub mod engine;
pub mod error;
pub mod receivers;
pub mod record;
pub mod router;
pub mod scope;
pub mod store;
pub mod terms;
pub mod voice;
pub mod work;

pub use claim::{Claim, Grade};
pub use engine::{Engine, EngineConfig, Erase, Ingested, Retrieval, Stats};
pub use error::Error;
pub use record::{Body, Record, Stored};
pub use scope::{Scope, ScopePolicy};
pub use work::WorkDoc;
pub use store::{State, Store};
