//! The data directory:
//!
//! ```text
//! records.log   one committed record per line, encrypted when a key is set
//! state.json    committed count, erasure epoch, consolidation history (no content)
//! routes.log    one RouteShape per ask (no content)
//! LOCK          held by the one writer
//! consolidations/  corpora (deleted after each build) and model versions
//! ```
//!
//! `records.log` is append-only except on erasure, when it is rewritten
//! without the erased records so they are physically gone, not tombstoned.
//! The committed count lives in `state.json` and never decreases, even then.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::consolidate::Consolidation;
use crate::crypto::{Cipher, PREFIX};
use crate::error::Error;
use crate::record::Stored;
use crate::router::RouteShape;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct State {
    /// Records ever committed. Monotone: erasure removes records, not history.
    pub committed: u64,
    /// Bumped whenever an erasure removed voice-corpus material; a model
    /// trained at an older epoch may still encode what was erased.
    #[serde(default)]
    pub voice_erasure_epoch: u64,
    #[serde(default)]
    pub consolidations: Vec<Consolidation>,
}

pub struct Store {
    root: PathBuf,
    cipher: Option<Cipher>,
    lock: Option<PathBuf>,
}

impl Store {
    /// Opens for writing, taking the directory lock.
    pub fn open(root: impl Into<PathBuf>, cipher: Option<Cipher>) -> Result<Store, Error> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|e| Error::io(&root, e))?;
        let lock = root.join("LOCK");
        match OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(mut f) => {
                let _ = writeln!(f, "{}", std::process::id());
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return Err(Error::Locked(lock)),
            Err(e) => return Err(Error::io(&lock, e)),
        }
        Ok(Store { root, cipher, lock: Some(lock) })
    }

    /// Opens without the lock, for inspection (`status`, `verify`, `ask`).
    pub fn open_read_only(root: impl Into<PathBuf>, cipher: Option<Cipher>) -> Store {
        Store { root: root.into(), cipher, lock: None }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn records_path(&self) -> PathBuf {
        self.root.join("records.log")
    }

    fn state_path(&self) -> PathBuf {
        self.root.join("state.json")
    }

    pub fn encrypted(&self) -> bool {
        self.cipher.is_some()
    }

    fn encode(&self, stored: &Stored) -> Result<String, Error> {
        let json = serde_json::to_vec(stored)?;
        match &self.cipher {
            Some(c) => c.seal(&json),
            None => Ok(String::from_utf8(json).expect("serde_json emits UTF-8")),
        }
    }

    fn decode(&self, line: &str, n: usize) -> Result<Stored, Error> {
        let corrupt = |reason: String| Error::CorruptLog { line: n, reason };
        let bytes = match (&self.cipher, line.starts_with(PREFIX)) {
            (Some(c), true) => c.open(line).map_err(|e| corrupt(e.to_string()))?,
            (Some(_), false) => return Err(corrupt("plaintext line in an encrypted log".into())),
            (None, true) => return Err(corrupt("encrypted line but no key configured (set CHIGUTIRO_KEY)".into())),
            (None, false) => line.as_bytes().to_vec(),
        };
        serde_json::from_slice(&bytes).map_err(|e| corrupt(e.to_string()))
    }

    pub fn load_records(&self) -> Result<Vec<Stored>, Error> {
        let path = self.records_path();
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(Error::io(&path, e)),
        };
        let mut out = Vec::new();
        for (i, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(|e| Error::io(&path, e))?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(self.decode(&line, i + 1)?);
        }
        Ok(out)
    }

    pub fn load_state(&self) -> Result<State, Error> {
        let path = self.state_path();
        match fs::read(&path) {
            Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(State::default()),
            Err(e) => Err(Error::io(&path, e)),
        }
    }

    pub fn save_state(&self, state: &State) -> Result<(), Error> {
        write_atomic(&self.state_path(), &serde_json::to_vec_pretty(state)?)
    }

    pub fn append(&self, records: &[Stored]) -> Result<(), Error> {
        if records.is_empty() {
            return Ok(());
        }
        let mut buf = String::new();
        for r in records {
            buf.push_str(&self.encode(r)?);
            buf.push('\n');
        }
        let path = self.records_path();
        let mut f = OpenOptions::new().create(true).append(true).open(&path).map_err(|e| Error::io(&path, e))?;
        f.write_all(buf.as_bytes()).map_err(|e| Error::io(&path, e))?;
        f.sync_data().map_err(|e| Error::io(&path, e))
    }

    /// Replaces the log with exactly `keep` — used by erasure.
    pub fn rewrite(&self, keep: &[Stored]) -> Result<(), Error> {
        let mut buf = String::new();
        for r in keep {
            buf.push_str(&self.encode(r)?);
            buf.push('\n');
        }
        write_atomic(&self.records_path(), buf.as_bytes())
    }

    pub fn append_route(&self, shape: &RouteShape) -> Result<(), Error> {
        let path = self.root.join("routes.log");
        let mut f = OpenOptions::new().create(true).append(true).open(&path).map_err(|e| Error::io(&path, e))?;
        writeln!(f, "{}", serde_json::to_string(shape)?).map_err(|e| Error::io(&path, e))
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        if let Some(lock) = &self.lock {
            let _ = fs::remove_file(lock);
        }
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = File::create(&tmp).map_err(|e| Error::io(&tmp, e))?;
        f.write_all(bytes).map_err(|e| Error::io(&tmp, e))?;
        f.sync_all().map_err(|e| Error::io(&tmp, e))?;
    }
    fs::rename(&tmp, path).map_err(|e| Error::io(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::random_hex;
    use crate::record::{Body, Record};

    fn stored(seq: u64) -> Stored {
        Stored {
            id: format!("n:{seq}"),
            seq,
            record: Record {
                id: Some(seq.to_string()),
                source: "n".into(),
                ts: "2026-09-01T00:00:00Z".parse().unwrap(),
                subject: None,
                tags: vec![],
                body: Body::Prose { text: format!("note {seq}"), title: None, authored_by_owner: true },
            },
        }
    }

    #[test]
    fn encrypted_round_trip_lock_and_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let key = random_hex(32);
        let s = Store::open(dir.path(), Some(Cipher::from_hex(&key).unwrap())).unwrap();
        assert!(matches!(Store::open(dir.path(), None), Err(Error::Locked(_))));
        s.append(&[stored(1), stored(2)]).unwrap();
        let raw = fs::read_to_string(dir.path().join("records.log")).unwrap();
        assert!(!raw.contains("note 1"), "plaintext on disk");
        assert_eq!(s.load_records().unwrap().len(), 2);
        s.rewrite(&[stored(2)]).unwrap();
        assert_eq!(s.load_records().unwrap(), vec![stored(2)]);
        // Reading without the key is refused, not misread.
        assert!(Store::open_read_only(dir.path(), None).load_records().is_err());
        drop(s);
        assert!(Store::open(dir.path(), None).is_ok(), "lock released on drop");
    }
}
