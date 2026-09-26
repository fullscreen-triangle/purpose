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
//! Only owner-authored prose ever reaches weights. Numbers, money, places and
//! other people stay in receivers, where they can be current, exact, and
//! erased.

pub mod claim;
pub mod consolidate;
pub mod crypto;
pub mod engine;
pub mod error;
pub mod receivers;
pub mod record;
pub mod router;
pub mod store;
pub mod terms;
pub mod voice;

pub use claim::{Claim, Grade};
pub use engine::{Engine, EngineConfig, Erase, Ingested, Retrieval, Stats};
pub use error::Error;
pub use record::{Body, Record, Stored};
pub use store::{State, Store};
