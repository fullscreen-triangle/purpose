mod answer;
mod server;
mod service;
mod verify;

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use chigutiro_core::consolidate::{self, ConsolidationConfig};
use chigutiro_core::crypto::{random_hex, Cipher};
use chigutiro_core::{EngineConfig, Erase, Scope, ScopePolicy};
use clap::{Args, Parser, Subcommand};

use crate::answer::{extractive, Ollama};
use crate::server::{app, AppState};
use crate::service::Service;

#[derive(Parser)]
#[command(name = "chigutiro", version, about = "A personal model that learns from what its host framework feeds it")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct DataArgs {
    /// Data directory (records, state, models).
    #[arg(long, env = "CHIGUTIRO_DATA", default_value = ".chigutiro")]
    data: PathBuf,
    /// Store records unencrypted. Without this, CHIGUTIRO_KEY is required.
    #[arg(long)]
    plaintext: bool,
    /// Source channels that count as work, comma-separated. Default: chat,
    /// academic, upload:lab-report, upload:paper, upload:presentation.
    #[arg(long, env = "CHIGUTIRO_WORK_CHANNELS")]
    work_channels: Option<String>,
}

impl DataArgs {
    fn engine_config(&self) -> EngineConfig {
        let scope = match &self.work_channels {
            Some(list) => ScopePolicy {
                work_channels: list.split(',').map(str::trim).filter(|c| !c.is_empty()).map(String::from).collect(),
            },
            None => ScopePolicy::default(),
        };
        EngineConfig { scope, ..EngineConfig::default() }
    }
}

#[derive(Args, Clone)]
struct ConsolidationArgs {
    /// A `purpose` binary built with the `factory` subcommand.
    #[arg(long, env = "CHIGUTIRO_PURPOSE_BIN")]
    purpose_bin: Option<PathBuf>,
    /// Base checkpoint each round retrains from.
    #[arg(long, env = "CHIGUTIRO_BASE_MODEL", default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
    base_model: String,
    /// New owner-authored documents that make a round due.
    #[arg(long, env = "CHIGUTIRO_MIN_NEW_DOCS", default_value_t = 50)]
    min_new_docs: usize,
}

impl ConsolidationArgs {
    fn config(&self) -> ConsolidationConfig {
        ConsolidationConfig {
            purpose_bin: self.purpose_bin.clone(),
            base_model: self.base_model.clone(),
            min_new_docs: self.min_new_docs,
            ..ConsolidationConfig::default()
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Print a fresh encryption key and API token.
    Keygen,
    /// Serve the HTTP API.
    Serve {
        #[command(flatten)]
        data: DataArgs,
        #[command(flatten)]
        consolidation: ConsolidationArgs,
        /// Bind address. Keep loopback unless a TLS-terminating proxy fronts it.
        #[arg(long, env = "CHIGUTIRO_HOST", default_value = "127.0.0.1")]
        host: String,
        #[arg(long, env = "CHIGUTIRO_PORT", default_value_t = 8740)]
        port: u16,
        #[arg(long, env = "CHIGUTIRO_TOKEN", hide_env_values = true)]
        token: Option<String>,
        /// Ollama base URL used to phrase answers; without a model, answers are extractive.
        #[arg(long, env = "CHIGUTIRO_OLLAMA_URL", default_value = "http://localhost:11434")]
        ollama_url: String,
        #[arg(long, env = "CHIGUTIRO_OLLAMA_MODEL")]
        ollama_model: Option<String>,
        /// Name the model uses for the person it serves.
        #[arg(long, env = "CHIGUTIRO_OWNER", default_value = "the owner")]
        owner: String,
        /// Start a consolidation round automatically whenever one becomes due.
        #[arg(long, env = "CHIGUTIRO_AUTO_CONSOLIDATE")]
        auto_consolidate: bool,
    },
    /// Ingest records from a JSON array or JSON-lines file (`-` for stdin).
    Ingest {
        #[command(flatten)]
        data: DataArgs,
        file: PathBuf,
    },
    /// Answer a question from local data (extractive; no model).
    Ask {
        #[command(flatten)]
        data: DataArgs,
        query: String,
        #[arg(long)]
        budget: Option<usize>,
        #[arg(long)]
        json: bool,
        /// Answer from the work scope only.
        #[arg(long)]
        work: bool,
    },
    /// Write the redacted work corpus (`corpus.jsonl` + `manifest.json`) for
    /// training a work model elsewhere, e.g. on AppHub. Plaintext: ship it,
    /// then delete it.
    ExportWork {
        #[command(flatten)]
        data: DataArgs,
        /// Directory to write into (created if missing).
        #[arg(long)]
        out: PathBuf,
    },
    /// Counts, voice corpus, and consolidation state.
    Status {
        #[command(flatten)]
        data: DataArgs,
        #[command(flatten)]
        consolidation: ConsolidationArgs,
    },
    /// Check the data directory's invariants; exits 1 on any breach.
    Verify {
        #[command(flatten)]
        data: DataArgs,
    },
    /// Permanently remove records (all given filters must match).
    Erase {
        #[command(flatten)]
        data: DataArgs,
        #[arg(long)]
        id: Vec<String>,
        #[arg(long)]
        subject: Option<String>,
        #[arg(long)]
        source: Option<String>,
        /// RFC 3339 timestamp; only records strictly before it.
        #[arg(long)]
        before: Option<chrono::DateTime<chrono::Utc>>,
    },
    /// Run one consolidation round in the foreground.
    Consolidate {
        #[command(flatten)]
        data: DataArgs,
        #[command(flatten)]
        consolidation: ConsolidationArgs,
        /// Run even if not due.
        #[arg(long)]
        force: bool,
    },
}

fn cipher(data: &DataArgs) -> Result<Option<Cipher>, String> {
    match std::env::var("CHIGUTIRO_KEY").ok().filter(|k| !k.trim().is_empty()) {
        Some(key) => Cipher::from_hex(&key).map(Some).map_err(|e| e.to_string()),
        None if data.plaintext => Ok(None),
        None => Err("set CHIGUTIRO_KEY (see `chigutiro keygen`) or pass --plaintext".into()),
    }
}

fn open(data: &DataArgs, consolidation: ConsolidationConfig, read_only: bool) -> Result<Service, String> {
    Service::open(&data.data, cipher(data)?, data.engine_config(), consolidation, read_only).map_err(|e| e.to_string())
}

fn print_json(v: &impl serde::Serialize) {
    println!("{}", serde_json::to_string_pretty(v).expect("serializes"));
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .init();
    match run(Cli::parse()).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<ExitCode, String> {
    match cli.cmd {
        Cmd::Keygen => {
            println!("CHIGUTIRO_KEY={}", random_hex(32));
            println!("CHIGUTIRO_TOKEN={}", random_hex(32));
            eprintln!("Keep the key: without it the record log cannot be read. Store both as secrets, never in the repo.");
        }
        Cmd::Serve { data, consolidation, host, port, token, ollama_url, ollama_model, owner, auto_consolidate } => {
            let token = token.filter(|t| t.len() >= 16).ok_or("CHIGUTIRO_TOKEN must be set (16+ chars); see `chigutiro keygen`")?;
            let mut service = open(&data, consolidation.config(), false)?;
            service.recover_interrupted().map_err(|e| e.to_string())?;
            let generator = ollama_model.map(|m| Ollama::new(ollama_url, m, owner));
            let state = AppState {
                service: Arc::new(Mutex::new(service)),
                token: Arc::new(token),
                generator,
                auto_consolidate,
            };
            let addr = format!("{host}:{port}");
            let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| format!("bind {addr}: {e}"))?;
            tracing::info!("chigutiro listening on http://{addr} (data: {})", data.data.display());
            axum::serve(listener, app(state))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await
                .map_err(|e| e.to_string())?;
        }
        Cmd::Ingest { data, file } => {
            let mut raw = String::new();
            if file.as_os_str() == "-" {
                std::io::stdin().read_to_string(&mut raw).map_err(|e| e.to_string())?;
            } else {
                raw = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
            }
            let values: Vec<serde_json::Value> = if raw.trim_start().starts_with('[') {
                serde_json::from_str(&raw).map_err(|e| e.to_string())?
            } else {
                raw.lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(serde_json::from_str)
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("JSON-lines: {e}"))?
            };
            let mut service = open(&data, ConsolidationConfig::default(), false)?;
            let report = service.ingest(values).map_err(|e| e.to_string())?;
            print_json(&report);
        }
        Cmd::Ask { data, query, budget, json, work } => {
            let service = open(&data, ConsolidationConfig::default(), true)?;
            let r = service.ask(&query, budget, work.then_some(Scope::Work));
            if json {
                print_json(&r);
            } else {
                println!("grade: {:?}\n{}", r.grade, extractive(&r));
            }
        }
        Cmd::ExportWork { data, out } => {
            let service = open(&data, ConsolidationConfig::default(), true)?;
            let docs = service.work_docs();
            std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
            let mut lines = String::new();
            let mut by_channel = std::collections::BTreeMap::<String, usize>::new();
            for d in &docs {
                lines.push_str(&serde_json::json!({ "text": d.text, "source": d.source }).to_string());
                lines.push('\n');
                let channel = d.source.split(':').take(2).collect::<Vec<_>>().join(":");
                *by_channel.entry(channel).or_insert(0) += 1;
            }
            std::fs::write(out.join("corpus.jsonl"), lines).map_err(|e| e.to_string())?;
            let manifest = serde_json::json!({
                "created": chrono::Utc::now(),
                "work_channels": service.engine_config().scope.work_channels,
                "docs": docs.len(),
                "chars": docs.iter().map(|d| d.text.chars().count()).sum::<usize>(),
                "by_channel": by_channel,
                "highest_seq": docs.iter().map(|d| d.seq).max(),
                "ids": docs.iter().map(|d| &d.id).collect::<Vec<_>>(),
            });
            std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest).expect("serializes"))
                .map_err(|e| e.to_string())?;
            eprintln!(
                "wrote {} work documents to {} (plaintext, redacted): ship it, then delete it",
                docs.len(),
                out.display()
            );
        }
        Cmd::Status { data, consolidation } => {
            let service = open(&data, consolidation.config(), true)?;
            print_json(&service.status());
        }
        Cmd::Verify { data } => {
            let report = verify::verify(&data.data, cipher(&data)?);
            for (ok, what) in &report.checks {
                println!("{} {what}", if *ok { "ok  " } else { "FAIL" });
            }
            if !report.ok() {
                return Ok(ExitCode::FAILURE);
            }
        }
        Cmd::Erase { data, id, subject, source, before } => {
            let mut service = open(&data, ConsolidationConfig::default(), false)?;
            let report = service.erase(&Erase { ids: id, subject, source, before }).map_err(|e| e.to_string())?;
            print_json(&report);
        }
        Cmd::Consolidate { data, consolidation, force } => {
            let mut service = open(&data, consolidation.config(), false)?;
            service.recover_interrupted().map_err(|e| e.to_string())?;
            let plan = service.begin_consolidation(force).map_err(|e| e.to_string())??;
            let bin = service.purpose_bin().expect("checked by begin_consolidation");
            eprintln!("round v{}: training (this can take a long time on CPU)...", plan.version);
            let result = consolidate::run(&plan, &bin);
            let failed = result.as_ref().err().cloned();
            service.end_consolidation(plan.version, result).map_err(|e| e.to_string())?;
            if let Some(e) = failed {
                return Err(e);
            }
            print_json(&service.status().consolidation);
        }
    }
    Ok(ExitCode::SUCCESS)
}
