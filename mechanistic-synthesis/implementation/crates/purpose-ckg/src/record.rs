//! The monotone record.
//!
//! The record counts committing acts. It never decrements — and in particular
//! **un-committing is itself a committing act**, so it *raises* the count. That
//! is not a quirk of the encoding: withdrawing a commitment is something you
//! do, and the doing is as much a deposit as the original act was.
//!
//! There is deliberately no method on this type that lowers `count`. Any future
//! change that adds one breaks the invariant the whole calculus rests on, and
//! `uncommit_still_increments` below is the test that will catch it.
//!
//! Port of `validation/core.py::Record`.

use std::collections::BTreeSet;

use crate::graph::EdgeKey;

/// One committing act: its position in the record, the contact it touched, and
/// a note. The log is append-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub position: u64,
    pub edge: EdgeKey,
    pub note: String,
}

/// A monotone record of committing acts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Record {
    count: u64,
    committed: BTreeSet<EdgeKey>,
    log: Vec<Entry>,
}

impl Record {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rehydrate a record from storage. Takes the count explicitly so a
    /// reloaded record continues its lineage rather than restarting it — a
    /// record that reset on reload would not be a record.
    pub fn resume(count: u64, committed: BTreeSet<EdgeKey>) -> Self {
        Self {
            count,
            committed,
            log: Vec::new(),
        }
    }

    /// Total committing acts. Only ever rises.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The contacts currently held as committed.
    pub fn committed(&self) -> &BTreeSet<EdgeKey> {
        &self.committed
    }

    /// Acts recorded in this process, oldest first.
    pub fn log(&self) -> &[Entry] {
        &self.log
    }

    pub fn is_committed(&self, u: &str, v: &str) -> bool {
        self.committed.contains(&EdgeKey::new(u, v))
    }

    /// Commit a contact. Returns the new record position.
    pub fn commit(&mut self, u: &str, v: &str, note: &str) -> u64 {
        let e = EdgeKey::new(u, v);
        self.count += 1;
        self.committed.insert(e.clone());
        self.log.push(Entry {
            position: self.count,
            edge: e,
            note: note.to_string(),
        });
        self.count
    }

    /// Withdraw a commitment — which **increments**, because withdrawing is
    /// itself a committing act. The contact leaves the committed set; the
    /// count does not fall.
    pub fn uncommit(&mut self, u: &str, v: &str) -> u64 {
        let e = EdgeKey::new(u, v);
        self.count += 1;
        self.committed.remove(&e);
        self.log.push(Entry {
            position: self.count,
            edge: e,
            note: "uncommit (a further commit)".to_string(),
        });
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_raises_the_count_and_holds_the_contact() {
        let mut r = Record::new();
        assert_eq!(r.count(), 0);
        assert_eq!(r.commit("a", "b", "first"), 1);
        assert!(r.is_committed("b", "a"));
        assert_eq!(r.committed().len(), 1);
    }

    #[test]
    fn uncommit_still_increments() {
        // binv:record / thm:monotone. Un-committing is a further committing act,
        // so the count rises even as the contact is released. If this test ever
        // fails, the record is no longer monotone and the calculus is unsound.
        let mut r = Record::new();
        r.commit("a", "b", "");
        assert_eq!(r.count(), 1);
        assert_eq!(r.uncommit("a", "b"), 2);
        assert_eq!(r.count(), 2, "uncommit must not decrement");
        assert!(!r.is_committed("a", "b"));
        assert_eq!(r.committed().len(), 0);
    }

    #[test]
    fn uncommitting_something_never_committed_still_increments() {
        let mut r = Record::new();
        assert_eq!(r.uncommit("x", "y"), 1);
        assert!(r.committed().is_empty());
    }

    #[test]
    fn recommitting_the_same_contact_advances_the_record() {
        // The committed set is idempotent; the record is not.
        let mut r = Record::new();
        r.commit("a", "b", "");
        r.commit("a", "b", "");
        assert_eq!(r.count(), 2);
        assert_eq!(r.committed().len(), 1);
    }

    #[test]
    fn the_log_is_append_only_and_positions_are_dense() {
        let mut r = Record::new();
        r.commit("a", "b", "one");
        r.uncommit("a", "b");
        r.commit("b", "c", "two");
        let positions: Vec<u64> = r.log().iter().map(|e| e.position).collect();
        assert_eq!(positions, vec![1, 2, 3]);
        assert_eq!(r.log()[1].note, "uncommit (a further commit)");
    }

    #[test]
    fn a_resumed_record_continues_its_lineage() {
        let mut r = Record::resume(41, BTreeSet::new());
        assert_eq!(r.commit("a", "b", ""), 42);
    }
}
