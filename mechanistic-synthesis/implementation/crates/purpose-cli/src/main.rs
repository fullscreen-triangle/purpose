//! Purpose CLI.
//!
//! Binary entry point for the MVP Purpose runtime.
//!
//! Examples:
//!     purpose query "Tell me about SOD1"
//!     purpose query "What is TP53?" --dry-run
//!     purpose operations
//!     purpose index
//!     purpose ask "where is the cascade router"

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use purpose_core::{typecheck::typecheck, Value};
use purpose_operations::{Executor, OperationRegistry};

#[derive(Parser, Debug)]
#[command(
    name = "purpose",
    version,
    about = "Purpose Model Factory CLI",
    long_about = "Compile natural-language queries to vaHera fragments and execute them against registered providers."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compile and execute a natural-language query.
    Query {
        /// Natural-language query (quote if it contains spaces).
        utterance: String,

        /// Print the compiled vaHera fragment as JSON and exit without executing.
        #[arg(long)]
        dry_run: bool,

        /// Print the raw JSON result rather than the formatted summary.
        #[arg(long)]
        raw: bool,
    },

    /// List all registered operations.
    Operations,

    /// Build a code index for the current project into `.purpose/index.json`.
    Index {
        /// Project root to index (defaults to the detected project root).
        #[arg(long)]
        root: Option<PathBuf>,
    },

    /// Ask a question about the current project; returns a context slice.
    Ask {
        /// The question (quote if it contains spaces).
        utterance: String,

        /// Print the compiled vaHera fragment as JSON and exit.
        #[arg(long)]
        dry_run: bool,

        /// Project root to query (defaults to the detected project root).
        #[arg(long)]
        root: Option<PathBuf>,
    },

    /// Clip an accumulated-context ledger to the slice necessary for its goal.
    ///
    /// Carry the uncertainty, not the knowledge: given a JSON ledger of turns
    /// (each with the terms it individuates and its token cost) and a goal,
    /// report which turns are load-bearing for the goal and how many tokens
    /// dropping the rest saves.
    Ledger {
        /// Path to a ledger JSON file (reads stdin if omitted).
        path: Option<PathBuf>,

        /// Print the raw clip result as JSON rather than the formatted report.
        #[arg(long)]
        raw: bool,
    },

    /// Determine which modules are load-bearing for a goal, and whether that
    /// determination is accountable.
    ///
    /// `ask` answers "where is X defined" from a scored index. This answers a
    /// question an index cannot express: given a goal, which modules cannot be
    /// dropped without changing what the goal resolves — and does the answer
    /// sit within the system's own floor, or is it contested?
    Ckg {
        #[command(subcommand)]
        cmd: CkgCommand,
    },
}

#[derive(Subcommand, Debug)]
enum CkgCommand {
    /// Induce the module contact graph from `.purpose/index.json`.
    Build {
        /// Project root (defaults to the detected project root).
        #[arg(long)]
        root: Option<PathBuf>,

        /// One module per source file, or per directory.
        #[arg(long, default_value = "file")]
        granularity: String,

        /// The floor β — the weight of every contact with the medium.
        #[arg(long, default_value_t = purpose_ckg::FLOOR)]
        floor: f64,

        /// Print the stored ckg as JSON rather than a summary.
        #[arg(long)]
        raw: bool,
    },

    /// Report the system floor β* and the module that realises it.
    Floor {
        #[arg(long)]
        root: Option<PathBuf>,

        #[arg(long)]
        raw: bool,
    },

    /// Determine the load-bearing modules for a goal.
    Ask {
        /// The goal (quote if it contains spaces).
        goal: String,

        /// Tolerance ε in the admissibility test σ ≤ β* + εΩ.
        #[arg(long, default_value_t = 0.0)]
        eps: f64,

        #[arg(long)]
        root: Option<PathBuf>,

        #[arg(long)]
        raw: bool,
    },

    /// Report one module's separation cost, resting cut, and what it dominates.
    Why {
        /// Module path as it appears in the ckg.
        module: String,

        /// Optional goal, to report reachability and necessity relative to it.
        #[arg(long)]
        goal: Option<String>,

        #[arg(long)]
        root: Option<PathBuf>,

        #[arg(long)]
        raw: bool,
    },
}

/// Walk up from `start` to find a project root (a dir containing `.git` or
/// `.purpose`); fall back to `start` itself.
fn detect_root(start: &Path) -> PathBuf {
    let mut cur = Some(start);
    while let Some(dir) = cur {
        if dir.join(".git").exists() || dir.join(".purpose").exists() {
            return dir.to_path_buf();
        }
        cur = dir.parent();
    }
    start.to_path_buf()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    // Build the registry and register the protein domain's providers.
    let mut registry = OperationRegistry::new();
    purpose_domains_protein::register_providers(&mut registry);

    let protein_domain = purpose_domains_protein::domain();
    let executor = Executor::new(registry.clone());

    match cli.cmd {
        Command::Query {
            utterance,
            dry_run,
            raw,
        } => {
            let fragment = protein_domain
                .resolver
                .compile(&utterance)
                .await
                .context("compilation failed")?;

            if !fragment.is_fully_resolved() {
                anyhow::bail!("compiled fragment contains unresolved holes");
            }

            // Type-check the fragment against the registered operations.
            let op_map: HashMap<String, purpose_core::Operation> = registry
                .operations()
                .cloned()
                .map(|op| (op.name.clone(), op))
                .collect();
            typecheck(&fragment, &op_map).context("type check failed")?;

            if dry_run {
                println!("{}", serde_json::to_string_pretty(&fragment)?);
                return Ok(());
            }

            let result = executor
                .execute(&fragment)
                .await
                .context("execution failed")?;

            if raw {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                match result {
                    Value::Str(s) => println!("{}", s),
                    other => println!("{}", serde_json::to_string_pretty(&other)?),
                }
            }
        }

        Command::Index { root } => {
            let cwd = std::env::current_dir().context("cannot read current directory")?;
            let root = root.unwrap_or_else(|| detect_root(&cwd));
            eprintln!("Indexing {} ...", root.display());
            let count = purpose_domains_codebase::build_index(&root)
                .context("indexing failed")?;
            println!(
                "Indexed {} symbol(s) into {}",
                count,
                purpose_domains_codebase::index_path(&root).display()
            );
        }

        Command::Ask {
            utterance,
            dry_run,
            root,
        } => {
            let cwd = std::env::current_dir().context("cannot read current directory")?;
            let root = root.unwrap_or_else(|| detect_root(&cwd));

            // Build a codebase registry/domain rooted at this project.
            let mut cb_registry = OperationRegistry::new();
            purpose_domains_codebase::register_providers(&mut cb_registry, root.clone());
            let cb_domain = purpose_domains_codebase::domain();
            let cb_executor = Executor::new(cb_registry.clone());

            let fragment = cb_domain
                .resolver
                .compile(&utterance)
                .await
                .context("compilation failed")?;

            if !fragment.is_fully_resolved() {
                anyhow::bail!("compiled fragment contains unresolved holes");
            }

            let op_map: HashMap<String, purpose_core::Operation> = cb_registry
                .operations()
                .cloned()
                .map(|op| (op.name.clone(), op))
                .collect();
            typecheck(&fragment, &op_map).context("type check failed")?;

            if dry_run {
                println!("{}", serde_json::to_string_pretty(&fragment)?);
                return Ok(());
            }

            let result = cb_executor
                .execute(&fragment)
                .await
                .context("execution failed")?;

            match result {
                Value::Str(s) => println!("{}", s),
                other => println!("{}", serde_json::to_string_pretty(&other)?),
            }
        }

        Command::Ledger { path, raw } => {
            let text = match path {
                Some(p) => std::fs::read_to_string(&p)
                    .with_context(|| format!("cannot read ledger {}", p.display()))?,
                None => {
                    use std::io::Read;
                    let mut buf = String::new();
                    std::io::stdin()
                        .read_to_string(&mut buf)
                        .context("cannot read ledger from stdin")?;
                    buf
                }
            };
            let ledger = purpose_domains_ledger::parse(&text).context("parse failed")?;
            let clip = ledger.clip();
            if raw {
                println!("{}", serde_json::to_string_pretty(&clip)?);
            } else {
                print!("{}", purpose_domains_ledger::render(&clip));
            }
        }

        Command::Ckg { cmd } => {
            let cwd = std::env::current_dir().context("cannot read current directory")?;
            match cmd {
                CkgCommand::Build {
                    root,
                    granularity,
                    floor,
                    raw,
                } => {
                    let root = root.unwrap_or_else(|| detect_root(&cwd));
                    let g = purpose_domains_ckg::Granularity::parse(&granularity)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    eprintln!("Inducing contact graph over {} ...", root.display());
                    let stored = purpose_domains_ckg::build(&root, g, floor)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;

                    if raw {
                        println!("{}", serde_json::to_string_pretty(&stored)?);
                    } else {
                        let graph = stored.graph().map_err(|e| anyhow::anyhow!("{e}"))?;
                        println!(
                            "{} module(s), {} contact(s) into {}",
                            stored.items.len(),
                            stored.edges.len(),
                            purpose_domains_ckg::ckg_path(&root).display()
                        );
                        print!("{}", purpose_domains_ckg::render_floor(&stored, &graph));
                    }
                }

                CkgCommand::Floor { root, raw } => {
                    let root = root.unwrap_or_else(|| detect_root(&cwd));
                    let stored = purpose_domains_ckg::load(&root)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let graph = stored.graph().map_err(|e| anyhow::anyhow!("{e}"))?;
                    if raw {
                        println!("{}", serde_json::to_string_pretty(&stored)?);
                    } else {
                        print!("{}", purpose_domains_ckg::render_floor(&stored, &graph));
                    }
                }

                CkgCommand::Ask {
                    goal,
                    eps,
                    root,
                    raw,
                } => {
                    let root = root.unwrap_or_else(|| detect_root(&cwd));
                    let d = purpose_domains_ckg::determine(&root, &goal, eps)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    if raw {
                        println!("{}", serde_json::to_string_pretty(&d)?);
                    } else {
                        print!("{}", purpose_domains_ckg::render_determination(&d));
                    }
                }

                CkgCommand::Why {
                    module,
                    goal,
                    root,
                    raw,
                } => {
                    let root = root.unwrap_or_else(|| detect_root(&cwd));
                    let w = purpose_domains_ckg::why(&root, &module, goal.as_deref())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    if raw {
                        println!("{}", serde_json::to_string_pretty(&w)?);
                    } else {
                        print!("{}", purpose_domains_ckg::render_why(&w));
                    }
                }
            }
        }

        Command::Operations => {
            let mut ops: Vec<_> = registry.operations().collect();
            ops.sort_by(|a, b| a.name.cmp(&b.name));
            for op in ops {
                println!("{}", op.name);
                println!("  {}", op.description);
                for (k, t) in &op.inputs {
                    println!("    input  {:<12} :: {:?}", k, t);
                }
                println!("    output             :: {:?}", op.output);
                println!();
            }
        }
    }

    Ok(())
}
