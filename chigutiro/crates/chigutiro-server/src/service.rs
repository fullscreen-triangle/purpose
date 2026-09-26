//! One owner of engine + store + state, shared by the HTTP server and the
//! CLI, so both paths commit, erase, and consolidate identically.

use std::path::{Path, PathBuf};

use chigutiro_core::consolidate::{self, Consolidation, ConsolidationConfig, Plan, Status};
use chigutiro_core::crypto::Cipher;
use chigutiro_core::voice::voice_doc;
use chigutiro_core::{Engine, EngineConfig, Erase, Error, Ingested, Record, Retrieval, State, Stats, Store};
use chrono::Utc;
use serde::Serialize;

pub struct Service {
    engine: Engine,
    store: Store,
    state: State,
    engine_cfg: EngineConfig,
    pub consolidation: ConsolidationConfig,
}

#[derive(Debug, Serialize)]
pub struct Rejected {
    pub index: usize,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct IngestReport {
    pub accepted: usize,
    pub duplicates: usize,
    pub rejected: Vec<Rejected>,
    pub committed: u64,
}

#[derive(Debug, Serialize)]
pub struct EraseReport {
    pub removed: usize,
    pub voice_material_removed: bool,
    /// True when the current model was trained on something just erased; it
    /// stays so until the next consolidation round supersedes and deletes it.
    pub model_tainted: bool,
}

#[derive(Debug, Serialize)]
pub struct ConsolidationStatus {
    pub available: bool,
    pub base_model: String,
    pub due: Option<String>,
    pub running: Option<u64>,
    pub current: Option<Consolidation>,
    pub tainted: bool,
}

#[derive(Debug, Serialize)]
pub struct StatusReport {
    pub encrypted: bool,
    pub stats: Stats,
    pub consolidation: ConsolidationStatus,
}

impl Service {
    pub fn open(
        data: &Path,
        cipher: Option<Cipher>,
        engine_cfg: EngineConfig,
        consolidation: ConsolidationConfig,
        read_only: bool,
    ) -> Result<Service, Error> {
        let store = if read_only { Store::open_read_only(data, cipher) } else { Store::open(data, cipher)? };
        let state = store.load_state()?;
        let records = store.load_records()?;
        let engine = Engine::restore(engine_cfg.clone(), records, state.committed);
        Ok(Service { engine, store, state, engine_cfg, consolidation })
    }

    /// Parses each value on its own, so one malformed record is reported
    /// with its index instead of rejecting the batch.
    pub fn ingest(&mut self, values: Vec<serde_json::Value>) -> Result<IngestReport, Error> {
        let mut accepted = Vec::new();
        let mut duplicates = 0;
        let mut rejected = Vec::new();
        for (index, value) in values.into_iter().enumerate() {
            let parsed = serde_json::from_value::<Record>(value).map_err(|e| e.to_string());
            match parsed.and_then(|r| self.engine.ingest(r).map_err(|e| e.to_string())) {
                Ok(Ingested::Accepted(s)) => accepted.push(s),
                Ok(Ingested::Duplicate(_)) => duplicates += 1,
                Err(reason) => rejected.push(Rejected { index, reason }),
            }
        }
        if let Err(e) = self.persist(&accepted) {
            // The engine indexed records the log does not hold; reload so
            // memory never runs ahead of disk.
            self.reload()?;
            return Err(e);
        }
        Ok(IngestReport { accepted: accepted.len(), duplicates, rejected, committed: self.engine.committed() })
    }

    fn persist(&mut self, accepted: &[chigutiro_core::Stored]) -> Result<(), Error> {
        if accepted.is_empty() {
            return Ok(());
        }
        self.store.append(accepted)?;
        self.state.committed = self.engine.committed();
        self.store.save_state(&self.state)
    }

    fn reload(&mut self) -> Result<(), Error> {
        self.state = self.store.load_state()?;
        self.engine = Engine::restore(self.engine_cfg.clone(), self.store.load_records()?, self.state.committed);
        Ok(())
    }

    pub fn ask(&self, query: &str, budget: Option<usize>) -> Retrieval {
        let r = self.engine.ask(query, Utc::now(), budget);
        if let Err(e) = self.store.append_route(&r.route) {
            tracing::warn!("route log: {e}");
        }
        r
    }

    pub fn erase(&mut self, criteria: &Erase) -> Result<EraseReport, Error> {
        if criteria.is_empty() {
            return Err(Error::InvalidRecord("erase needs at least one of ids, subject, source, before".into()));
        }
        let removed = self.engine.erase(criteria);
        let voice_material_removed = removed.iter().any(|s| voice_doc(s).is_some());
        if !removed.is_empty() {
            self.store.rewrite(self.engine.records())?;
            if voice_material_removed {
                self.state.voice_erasure_epoch += 1;
            }
            self.store.save_state(&self.state)?;
        }
        Ok(EraseReport { removed: removed.len(), voice_material_removed, model_tainted: consolidate::tainted(&self.state) })
    }

    pub fn status(&self) -> StatusReport {
        let docs = self.engine.voice_docs();
        StatusReport {
            encrypted: self.store.encrypted(),
            stats: self.engine.stats(),
            consolidation: ConsolidationStatus {
                available: self.purpose_bin().is_some(),
                base_model: self.consolidation.base_model.clone(),
                due: consolidate::due(&self.state, &docs, &self.consolidation),
                running: self.state.consolidations.iter().find(|c| c.status == Status::Running).map(|c| c.version),
                current: consolidate::latest_succeeded(&self.state).cloned(),
                tainted: consolidate::tainted(&self.state),
            },
        }
    }

    pub fn consolidations(&self) -> &[Consolidation] {
        &self.state.consolidations
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn purpose_bin(&self) -> Option<PathBuf> {
        self.consolidation.purpose_bin.clone().filter(|p| p.exists())
    }

    /// Starts a round if one is due (or `force`d). Returns the plan to run
    /// off-lock, then pass to `end_consolidation`.
    pub fn begin_consolidation(&mut self, force: bool) -> Result<Result<Plan, String>, Error> {
        if self.purpose_bin().is_none() {
            return Ok(Err(match &self.consolidation.purpose_bin {
                Some(p) => format!("purpose binary not found at {}", p.display()),
                None => "consolidation unavailable: set CHIGUTIRO_PURPOSE_BIN to a purpose binary with `factory`".into(),
            }));
        }
        if let Some(r) = self.state.consolidations.iter().find(|c| c.status == Status::Running) {
            return Ok(Err(format!("round v{} is already running", r.version)));
        }
        let docs = self.engine.voice_docs();
        let reason = match consolidate::due(&self.state, &docs, &self.consolidation) {
            Some(r) => r,
            None if force && !docs.is_empty() => format!("forced: {} voice documents", docs.len()),
            None if docs.is_empty() => return Ok(Err("no voice documents: nothing owner-authored has been ingested".into())),
            None => return Ok(Err("not due: too few new voice documents since the current model".into())),
        };
        let (plan, record) = consolidate::prepare(self.store.root(), &self.state, &docs, &self.consolidation, reason, Utc::now())?;
        self.state.consolidations.push(record);
        self.store.save_state(&self.state)?;
        Ok(Ok(plan))
    }

    pub fn end_consolidation(&mut self, version: u64, result: Result<(), String>) -> Result<(), Error> {
        let removed = consolidate::finish(&mut self.state, version, result, Utc::now());
        for dir in removed {
            tracing::info!("deleted superseded model {}", dir.display());
        }
        self.store.save_state(&self.state)
    }

    /// A round left `Running` by a crash can never finish; mark it failed so
    /// the next one may start.
    pub fn recover_interrupted(&mut self) -> Result<(), Error> {
        let mut changed = false;
        for c in self.state.consolidations.iter_mut().filter(|c| c.status == Status::Running) {
            c.status = Status::Failed;
            c.error = Some("interrupted: the process stopped before the build finished".into());
            c.finished = Some(Utc::now());
            let _ = std::fs::remove_dir_all(c.model_dir.with_file_name("corpus"));
            changed = true;
        }
        if changed {
            self.store.save_state(&self.state)?;
        }
        Ok(())
    }
}
