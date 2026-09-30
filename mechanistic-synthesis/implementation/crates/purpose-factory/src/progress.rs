//! Progress reporting and cancellation for a running build, so a caller that
//! is not the CLI (the job runner behind `purpose serve`) can show where a
//! build is and stop it between steps. The CLI passes `NoProgress`.

use serde::{Deserialize, Serialize};

/// One optimizer step. `loss` is the running mean over the current epoch.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StepProgress {
    pub epoch: usize,
    pub epochs: usize,
    pub step: usize,
    pub steps_per_epoch: usize,
    pub loss: f32,
}

pub trait ProgressSink: Send + Sync {
    /// A named phase of the build started (fetching, corpus, download, training, export).
    fn stage(&self, _stage: &str) {}

    fn step(&self, _progress: StepProgress) {}

    /// Checked between steps; returning true stops the build with `Error::Cancelled`.
    fn cancelled(&self) -> bool {
        false
    }
}

pub struct NoProgress;

impl ProgressSink for NoProgress {}
