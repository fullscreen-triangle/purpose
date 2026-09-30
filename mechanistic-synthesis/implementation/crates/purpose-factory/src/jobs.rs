//! Background build jobs for `purpose serve`: a build is submitted, runs on
//! one worker thread (the laptop trains one model at a time), and reports
//! its stage, step and loss as it goes, so a browser can poll it instead of
//! holding a request open for hours.
//!
//! Each build runs on its own current-thread tokio runtime inside the worker
//! thread, so CPU-bound training never blocks the server's runtime. Job
//! state is written to `<root>/jobs/<id>.json` and a human-readable log to
//! `<id>.log`; a job that was queued or running when the server stopped is
//! marked `interrupted` on the next start.

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::contract::{
    BaseModelSpec, HeuristicVerifier, PretrainedConfig, ScratchConfig, ThemeContract, TrainingSpec,
    DEFAULT_PRETRAINED_BLOCK_SIZE,
};
use crate::error::Error;
use crate::factory::{Factory, Registry, ThemeModel};
use crate::progress::{ProgressSink, StepProgress};
use crate::source::{LocalFileSource, SourceProvider, UrlSource};
use crate::workspace::{validate_theme_name, Workspace};

/// What to build: shared by the blocking `/themes/{name}/build` route and `/jobs`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildRequest {
    #[serde(default)]
    pub urls: Vec<String>,
    #[serde(default)]
    pub model: BuildModelChoice,
    #[serde(default)]
    pub training: Option<BuildTrainingSpec>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BuildModelChoice {
    #[default]
    Scratch,
    Pretrained {
        repo: String,
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        block_size: Option<usize>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildTrainingSpec {
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
    pub learning_rate: Option<f64>,
    pub lora_rank: Option<usize>,
    pub lora_alpha: Option<f64>,
}

impl BuildRequest {
    /// The contract for `theme`: its uploaded sources plus any URLs.
    pub fn contract(&self, ws: &Workspace, theme: &str) -> Result<ThemeContract, Error> {
        validate_theme_name(theme)?;
        let dir = ws.sources_dir(theme);
        let mut sources: Vec<Box<dyn SourceProvider>> = Vec::new();
        if dir.is_dir() {
            sources.push(Box::new(LocalFileSource::new(dir)));
        }
        if !self.urls.is_empty() {
            sources.push(Box::new(UrlSource::new(self.urls.clone())));
        }
        if sources.is_empty() {
            return Err(Error::Source(format!(
                "theme '{theme}' has no sources — upload files or pass urls before building"
            )));
        }

        let base_model = match &self.model {
            BuildModelChoice::Scratch => BaseModelSpec::Scratch(ScratchConfig::default()),
            BuildModelChoice::Pretrained { repo, revision, block_size } => {
                BaseModelSpec::Pretrained(PretrainedConfig {
                    repo: repo.clone(),
                    revision: revision.clone(),
                    block_size: block_size.unwrap_or(DEFAULT_PRETRAINED_BLOCK_SIZE),
                })
            }
        };

        let mut training = TrainingSpec::default();
        if let Some(t) = &self.training {
            training.epochs = t.epochs.unwrap_or(training.epochs);
            training.batch_size = t.batch_size.unwrap_or(training.batch_size);
            training.learning_rate = t.learning_rate.unwrap_or(training.learning_rate);
            training.lora_rank = t.lora_rank.unwrap_or(training.lora_rank);
            training.lora_alpha = t.lora_alpha.unwrap_or(training.lora_alpha);
        }

        Ok(ThemeContract {
            name: theme.to_string(),
            sources,
            base_model,
            verifier: Box::new(HeuristicVerifier::default()),
            training,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
}

impl JobStatus {
    pub fn is_finished(self) -> bool {
        !matches!(self, JobStatus::Queued | JobStatus::Running)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub theme: String,
    pub status: JobStatus,
    pub stage: Option<String>,
    pub progress: Option<StepProgress>,
    /// Mean loss so far, one point per logged step, for a loss curve.
    pub loss_history: Vec<f32>,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub result: Option<ThemeModel>,
    pub error: Option<String>,
    pub request: BuildRequest,
}

const MAX_LOSS_POINTS: usize = 400;
const STEP_LOG_INTERVAL: Duration = Duration::from_secs(2);

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

struct Inner {
    ws: Workspace,
    jobs: Mutex<BTreeMap<String, Job>>,
    cancel: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl Inner {
    fn update(&self, id: &str, f: impl FnOnce(&mut Job)) {
        let mut jobs = self.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(id) {
            f(job);
            self.persist(job);
        }
    }

    fn persist(&self, job: &Job) {
        let path = self.ws.jobs_dir().join(format!("{}.json", job.id));
        match serde_json::to_vec_pretty(job) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(&path, bytes) {
                    tracing::warn!(error = %e, path = %path.display(), "cannot persist job");
                }
            }
            Err(e) => tracing::warn!(error = %e, "cannot serialise job"),
        }
    }

    fn log(&self, id: &str, line: &str) {
        let path = self.ws.jobs_dir().join(format!("{id}.log"));
        let stamp = now();
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(f, "[{stamp}] {line}");
        }
    }
}

/// The job queue and its single worker thread.
#[derive(Clone)]
pub struct Jobs {
    inner: Arc<Inner>,
    tx: mpsc::Sender<String>,
}

impl Jobs {
    /// Loads existing jobs from `<root>/jobs/`, marks unfinished ones
    /// `interrupted`, and starts the worker.
    pub fn open(ws: Workspace) -> Result<Self, Error> {
        std::fs::create_dir_all(ws.jobs_dir())?;
        let mut jobs = BTreeMap::new();
        for entry in std::fs::read_dir(ws.jobs_dir())? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Ok(raw) = std::fs::read(&path) else { continue };
            let Ok(job) = serde_json::from_slice::<Job>(&raw) else {
                tracing::warn!(path = %path.display(), "skipping unreadable job file");
                continue;
            };
            jobs.insert(job.id.clone(), job);
        }

        let inner = Arc::new(Inner {
            ws,
            jobs: Mutex::new(jobs),
            cancel: Mutex::new(HashMap::new()),
        });
        {
            let mut jobs = inner.jobs.lock().unwrap();
            for job in jobs.values_mut().filter(|j| !j.status.is_finished()) {
                job.status = JobStatus::Interrupted;
                job.error = Some("the server stopped while this job was queued or running".into());
                job.finished_at = Some(now());
                inner.persist(job);
            }
        }

        let (tx, rx) = mpsc::channel::<String>();
        let worker = inner.clone();
        std::thread::Builder::new()
            .name("purpose-jobs".into())
            .spawn(move || {
                for id in rx {
                    run(&worker, &id);
                }
            })?;

        Ok(Self { inner, tx })
    }

    /// Queues a build of `theme`. Fails immediately (no job is created) if
    /// the theme name is invalid or the theme has nothing to train on.
    pub fn submit(&self, theme: &str, request: BuildRequest) -> Result<Job, Error> {
        request.contract(&self.inner.ws, theme)?;
        let id = format!("{:x}-{:04x}", now(), rand::random::<u16>());
        let job = Job {
            id: id.clone(),
            theme: theme.to_string(),
            status: JobStatus::Queued,
            stage: None,
            progress: None,
            loss_history: Vec::new(),
            created_at: now(),
            started_at: None,
            finished_at: None,
            result: None,
            error: None,
            request,
        };
        self.inner.persist(&job);
        self.inner.cancel.lock().unwrap().insert(id.clone(), Arc::new(AtomicBool::new(false)));
        self.inner.jobs.lock().unwrap().insert(id.clone(), job.clone());
        self.inner.log(&id, &format!("queued: theme '{theme}'"));
        self.tx
            .send(id)
            .map_err(|_| Error::Config("job worker has stopped".into()))?;
        Ok(job)
    }

    /// Newest first.
    pub fn list(&self) -> Vec<Job> {
        let mut jobs: Vec<Job> = self.inner.jobs.lock().unwrap().values().cloned().collect();
        jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.id.cmp(&a.id)));
        jobs
    }

    pub fn get(&self, id: &str) -> Option<Job> {
        self.inner.jobs.lock().unwrap().get(id).cloned()
    }

    pub fn is_running(&self) -> bool {
        self.inner.jobs.lock().unwrap().values().any(|j| j.status == JobStatus::Running)
    }

    /// The last `tail` lines of the job's log.
    pub fn log(&self, id: &str, tail: usize) -> Option<Vec<String>> {
        self.get(id)?;
        let raw = std::fs::read_to_string(self.inner.ws.jobs_dir().join(format!("{id}.log")))
            .unwrap_or_default();
        let lines: Vec<String> = raw.lines().map(str::to_string).collect();
        let start = lines.len().saturating_sub(tail);
        Some(lines[start..].to_vec())
    }

    /// A queued job is cancelled at once; a running one stops at its next
    /// training step. Finished jobs are left as they are.
    pub fn cancel(&self, id: &str) -> Option<Job> {
        let status = self.get(id)?.status;
        if let Some(flag) = self.inner.cancel.lock().unwrap().get(id) {
            flag.store(true, Ordering::SeqCst);
        }
        if status == JobStatus::Queued {
            self.inner.update(id, |j| {
                j.status = JobStatus::Cancelled;
                j.finished_at = Some(now());
            });
            self.inner.log(id, "cancelled before it started");
        } else if status == JobStatus::Running {
            self.inner.log(id, "cancel requested; stopping at the next step");
        }
        self.get(id)
    }
}

struct JobSink {
    inner: Arc<Inner>,
    id: String,
    cancel: Arc<AtomicBool>,
    last_logged: Mutex<Option<Instant>>,
}

impl ProgressSink for JobSink {
    fn stage(&self, stage: &str) {
        self.inner.update(&self.id, |j| j.stage = Some(stage.to_string()));
        self.inner.log(&self.id, stage);
    }

    fn step(&self, p: StepProgress) {
        let mut last = self.last_logged.lock().unwrap();
        let epoch_end = p.step == p.steps_per_epoch;
        let due = last.map(|t| t.elapsed() >= STEP_LOG_INTERVAL).unwrap_or(true);
        if !(due || epoch_end) {
            return;
        }
        *last = Some(Instant::now());
        drop(last);
        self.inner.update(&self.id, |j| {
            j.progress = Some(p);
            j.loss_history.push(p.loss);
            if j.loss_history.len() > MAX_LOSS_POINTS {
                // Keep the curve's shape at bounded size: drop every other point.
                j.loss_history = j.loss_history.iter().step_by(2).copied().collect();
            }
        });
        self.inner.log(
            &self.id,
            &format!(
                "epoch {}/{} step {}/{} loss {:.4}",
                p.epoch + 1,
                p.epochs,
                p.step,
                p.steps_per_epoch,
                p.loss
            ),
        );
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

fn run(inner: &Arc<Inner>, id: &str) {
    let Some(job) = inner.jobs.lock().unwrap().get(id).cloned() else { return };
    if job.status != JobStatus::Queued {
        return;
    }
    let cancel = inner
        .cancel
        .lock()
        .unwrap()
        .get(id)
        .cloned()
        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

    inner.update(id, |j| {
        j.status = JobStatus::Running;
        j.started_at = Some(now());
    });
    inner.log(id, "started");

    let sink = JobSink {
        inner: inner.clone(),
        id: id.to_string(),
        cancel,
        last_logged: Mutex::new(None),
    };
    let outcome = build(inner, &job, &sink);

    match outcome {
        Ok(model) => {
            inner.log(id, &format!("succeeded: {} examples -> {}", model.example_count, model.path.display()));
            inner.update(id, |j| {
                j.status = JobStatus::Succeeded;
                j.stage = Some("done".into());
                j.result = Some(model);
                j.finished_at = Some(now());
            });
        }
        Err(Error::Cancelled) => {
            inner.log(id, "cancelled");
            inner.update(id, |j| {
                j.status = JobStatus::Cancelled;
                j.finished_at = Some(now());
            });
        }
        Err(e) => {
            inner.log(id, &format!("failed: {e}"));
            inner.update(id, |j| {
                j.status = JobStatus::Failed;
                j.error = Some(e.to_string());
                j.finished_at = Some(now());
            });
        }
    }
    inner.cancel.lock().unwrap().remove(id);
}

fn build(inner: &Arc<Inner>, job: &Job, sink: &JobSink) -> Result<ThemeModel, Error> {
    let contract = job.request.contract(&inner.ws, &job.theme)?;
    let out_dir = inner.ws.model_dir(&job.theme);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(Error::Io)?;
    let model = runtime.block_on(Factory::build_with_progress(contract, &out_dir, sink))?;
    Registry::new(inner.ws.registry_path()).record(&model)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "Transaminases move an amino group from an amino acid donor onto a keto acid \
        acceptor. Aspartate transaminase converts L-aspartate and 2-oxoglutarate into oxaloacetate \
        and L-glutamate, using pyridoxal phosphate as its cofactor. Alanine transaminase does the \
        same with L-alanine, producing pyruvate. The knowledge graph infers the reaction type from \
        the roles of the participants rather than asserting it for each reaction.\n\n";

    fn workspace(tag: &str) -> Workspace {
        let root = std::env::temp_dir().join(format!("pf-jobs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let ws = Workspace::new(root);
        std::fs::create_dir_all(ws.sources_dir("t")).unwrap();
        std::fs::write(ws.sources_dir("t").join("a.md"), TEXT.repeat(20)).unwrap();
        ws
    }

    fn scratch(epochs: usize) -> BuildRequest {
        BuildRequest {
            training: Some(BuildTrainingSpec { epochs: Some(epochs), batch_size: Some(2), ..Default::default() }),
            ..Default::default()
        }
    }

    fn wait(jobs: &Jobs, id: &str, until: impl Fn(&Job) -> bool) -> Job {
        let deadline = Instant::now() + Duration::from_secs(300);
        loop {
            let job = jobs.get(id).unwrap();
            if until(&job) {
                return job;
            }
            assert!(Instant::now() < deadline, "timed out; last state {job:?}");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    #[test]
    fn a_job_runs_to_success_and_reports_progress() {
        let ws = workspace("ok");
        let jobs = Jobs::open(ws.clone()).unwrap();
        let job = jobs.submit("t", scratch(1)).unwrap();
        assert_eq!(job.status, JobStatus::Queued);
        let done = wait(&jobs, &job.id, |j| j.status.is_finished());
        assert_eq!(done.status, JobStatus::Succeeded, "{:?}", done.error);
        assert!(done.progress.is_some() && !done.loss_history.is_empty());
        assert!(ws.model_dir("t").join("model.safetensors").exists());
        let log = jobs.log(&job.id, 100).unwrap().join("\n");
        assert!(log.contains("training") && log.contains("succeeded"), "{log}");
        std::fs::remove_dir_all(ws.root()).ok();
    }

    #[test]
    fn jobs_can_be_cancelled_queued_or_running() {
        let ws = workspace("cancel");
        let jobs = Jobs::open(ws.clone()).unwrap();
        let long = jobs.submit("t", scratch(500)).unwrap();
        let queued = jobs.submit("t", scratch(1)).unwrap();
        assert_eq!(jobs.cancel(&queued.id).unwrap().status, JobStatus::Cancelled);

        wait(&jobs, &long.id, |j| j.progress.is_some());
        jobs.cancel(&long.id);
        let stopped = wait(&jobs, &long.id, |j| j.status.is_finished());
        assert_eq!(stopped.status, JobStatus::Cancelled);
        // The cancelled-while-queued job never started.
        assert!(jobs.get(&queued.id).unwrap().started_at.is_none());
        std::fs::remove_dir_all(ws.root()).ok();
    }

    #[test]
    fn unfinished_jobs_are_interrupted_on_restart_and_bad_themes_are_refused() {
        let ws = workspace("restart");
        std::fs::create_dir_all(ws.jobs_dir()).unwrap();
        let stale = Job {
            id: "stale".into(),
            theme: "t".into(),
            status: JobStatus::Running,
            stage: Some("training".into()),
            progress: None,
            loss_history: vec![],
            created_at: 1,
            started_at: Some(1),
            finished_at: None,
            result: None,
            error: None,
            request: BuildRequest::default(),
        };
        std::fs::write(ws.jobs_dir().join("stale.json"), serde_json::to_vec(&stale).unwrap()).unwrap();
        let jobs = Jobs::open(ws.clone()).unwrap();
        assert_eq!(jobs.get("stale").unwrap().status, JobStatus::Interrupted);

        assert!(jobs.submit("../escape", BuildRequest::default()).is_err());
        assert!(jobs.submit("empty", BuildRequest::default()).is_err(), "no sources, no job");
        assert_eq!(jobs.list().len(), 1);
        std::fs::remove_dir_all(ws.root()).ok();
    }
}
