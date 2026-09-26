//! `chigutiro verify`: checks the data directory's invariants and exits
//! non-zero on any breach, so it can gate a deploy or a backup restore.

use std::collections::HashSet;

use chigutiro_core::consolidate::Status;
use chigutiro_core::crypto::Cipher;
use chigutiro_core::Store;

pub struct Report {
    pub checks: Vec<(bool, String)>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|(ok, _)| *ok)
    }
}

pub fn verify(data: &std::path::Path, cipher: Option<Cipher>) -> Report {
    let store = Store::open_read_only(data, cipher);
    let mut checks = Vec::new();
    let mut check = |ok: bool, what: String| checks.push((ok, what));

    let state = match store.load_state() {
        Ok(s) => s,
        Err(e) => {
            check(false, format!("state.json readable: {e}"));
            return Report { checks };
        }
    };
    let records = match store.load_records() {
        Ok(r) => {
            check(true, format!("every log line decodes ({} records)", r.len()));
            r
        }
        Err(e) => {
            check(false, format!("every log line decodes: {e}"));
            return Report { checks };
        }
    };

    let mut ids = HashSet::new();
    let dup: Vec<&str> = records.iter().filter(|r| !ids.insert(r.id.as_str())).map(|r| r.id.as_str()).collect();
    check(dup.is_empty(), format!("record ids unique{}", if dup.is_empty() { String::new() } else { format!(": duplicated {dup:?}") }));

    let increasing = records.windows(2).all(|w| w[0].seq < w[1].seq);
    check(increasing, "sequence numbers strictly increase".into());

    let max_seq = records.iter().map(|r| r.seq).max().unwrap_or(0);
    check(
        state.committed >= max_seq,
        format!("committed count ({}) covers every held record (max seq {max_seq})", state.committed),
    );

    let invalid: Vec<String> = records
        .iter()
        .filter_map(|r| r.record.clone().validate().err().map(|e| format!("{}: {e}", r.id)))
        .collect();
    check(invalid.is_empty(), format!("every record passes validation{}", if invalid.is_empty() { String::new() } else { format!(": {invalid:?}") }));

    let current: Vec<_> = state.consolidations.iter().filter(|c| c.status == Status::Succeeded).collect();
    check(current.len() <= 1, format!("at most one current model ({} marked succeeded)", current.len()));
    for c in &current {
        check(c.model_dir.exists(), format!("current model v{} present at {}", c.version, c.model_dir.display()));
    }
    for c in state.consolidations.iter().filter(|c| c.status == Status::Superseded) {
        let gone = !c.model_dir.exists();
        check(gone, format!("superseded model v{} deleted", c.version));
    }
    for c in &state.consolidations {
        let corpus = c.model_dir.with_file_name("corpus");
        if c.status != Status::Running {
            check(!corpus.exists(), format!("no plaintext corpus left behind by v{}", c.version));
        }
    }
    Report { checks }
}
