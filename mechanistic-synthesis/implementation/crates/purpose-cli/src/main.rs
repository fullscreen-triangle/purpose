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
    ///
    /// The graph is induced by a term map τ, and τ is yours to choose. Write
    /// `.purpose/lens.toml` (start with `purpose ckg lens --init`) to decide
    /// what counts as a distinction in this repository, and run
    /// `purpose ckg lens` to see what that choice did to the structure before
    /// committing it with `purpose ckg build`.
    Ckg {
        #[command(subcommand)]
        cmd: CkgCommand,
    },

    /// Build a theme-specific model: ingest sources (local files, email, web
    /// pages), form a verified training corpus, train and export a model.
    Factory {
        #[command(subcommand)]
        cmd: FactoryCommand,
    },
}

#[derive(Subcommand, Debug)]
enum FactoryCommand {
    /// Scaffold a starter `theme.toml` for a new theme.
    Init {
        /// Theme name.
        name: String,

        /// Where to write the scaffolded file (defaults to `<name>.theme.toml`).
        #[arg(long)]
        out: Option<PathBuf>,

        /// Overwrite an existing file.
        #[arg(long)]
        force: bool,
    },

    /// Fetch sources, train, and export a theme model.
    Build {
        /// Path to a `theme.toml`.
        config: PathBuf,

        /// Output directory for the exported model (defaults to
        /// `.purpose/factory/<name>`).
        #[arg(long)]
        out: Option<PathBuf>,

        /// Registry file to record the build in (defaults to
        /// `.purpose/factory/registry.json`).
        #[arg(long)]
        registry: Option<PathBuf>,

        /// Print the resulting `ThemeModel` as JSON rather than a summary.
        #[arg(long)]
        raw: bool,
    },

    /// List theme models recorded in the local registry.
    List {
        /// Registry file (defaults to `.purpose/factory/registry.json`).
        #[arg(long)]
        registry: Option<PathBuf>,

        #[arg(long)]
        raw: bool,
    },
}

#[derive(Subcommand, Debug)]
enum CkgCommand {
    /// Induce the module contact graph from `.purpose/index.json`.
    Build {
        /// Project root (defaults to the detected project root).
        #[arg(long)]
        root: Option<PathBuf>,

        /// Lens file (defaults to `.purpose/lens.toml`, or built-in defaults).
        #[arg(long)]
        lens: Option<PathBuf>,

        /// One module per source file, or per directory. Overrides the lens.
        #[arg(long)]
        granularity: Option<String>,

        /// The floor β — the weight of every contact with the medium.
        /// Overrides the lens.
        #[arg(long)]
        floor: Option<f64>,

        /// Print the stored ckg as JSON rather than a summary.
        #[arg(long)]
        raw: bool,
    },

    /// Report what a lens does to the structure of the graph.
    ///
    /// A dry run: it induces the graph in memory and never writes
    /// `.purpose/ckg.json`, so trying a lens costs nothing. There is
    /// deliberately no score to raise — read the components, the term spread,
    /// and the goal saturation.
    Lens {
        #[arg(long)]
        root: Option<PathBuf>,

        /// Lens file (defaults to `.purpose/lens.toml`, or built-in defaults).
        #[arg(long)]
        lens: Option<PathBuf>,

        /// Goals to report seed saturation for. Repeatable.
        #[arg(long)]
        goal: Vec<String>,

        /// Write a commented default lens to `.purpose/lens.toml` and exit.
        #[arg(long)]
        init: bool,

        /// Overwrite an existing lens file when used with `--init`.
        #[arg(long)]
        force: bool,

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
                    lens,
                    granularity,
                    floor,
                    raw,
                } => {
                    let root = root.unwrap_or_else(|| detect_root(&cwd));
                    let mut l = purpose_domains_ckg::load_lens(&root, lens.as_deref())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;

                    // CLI over lens over built-in — and say so. Silently
                    // overriding a checked-in file is how an afternoon is lost
                    // to wondering why an edit had no effect.
                    if let Some(g) = granularity {
                        let g = purpose_domains_ckg::Granularity::parse(&g)
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        if g != l.granularity {
                            eprintln!(
                                "note: --granularity {} overrides the lens ({})",
                                g.as_str(),
                                l.granularity.as_str()
                            );
                        }
                        l.granularity = g;
                    }
                    if let Some(f) = floor {
                        if f != l.floor {
                            eprintln!("note: --floor {f} overrides the lens ({})", l.floor);
                        }
                        l.floor = f;
                    }

                    eprintln!("Inducing contact graph over {} ...", root.display());
                    let stored = purpose_domains_ckg::build(&root, &l)
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
                        println!(
                            "lens: {} ({})",
                            stored.lens_source.as_deref().unwrap_or("built-in defaults"),
                            stored.lens_digest
                        );
                        print!("{}", purpose_domains_ckg::render_floor(&stored, &graph));
                    }
                }

                CkgCommand::Lens {
                    root,
                    lens,
                    goal,
                    init,
                    force,
                    raw,
                } => {
                    let root = root.unwrap_or_else(|| detect_root(&cwd));
                    if init {
                        let path = root.join(purpose_domains_ckg::lens::LENS_FILE);
                        if path.exists() && !force {
                            anyhow::bail!(
                                "{} already exists — pass --force to overwrite it",
                                path.display()
                            );
                        }
                        if let Some(dir) = path.parent() {
                            std::fs::create_dir_all(dir)?;
                        }
                        std::fs::write(&path, purpose_domains_ckg::lens::default_lens_toml())?;
                        println!("wrote {}", path.display());
                        println!(
                            "edit it, then run `purpose ckg lens` to see what it does before \
                             `purpose ckg build`"
                        );
                        return Ok(());
                    }

                    let l = purpose_domains_ckg::load_lens(&root, lens.as_deref())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let report = purpose_domains_ckg::lens_report(&root, &l, &goal)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    if raw {
                        println!("{}", serde_json::to_string_pretty(&report)?);
                    } else {
                        print!("{}", purpose_domains_ckg::render_lens_report(&report));
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

        Command::Factory { cmd } => {
            let cwd = std::env::current_dir().context("cannot read current directory")?;
            let root = detect_root(&cwd);

            match cmd {
                FactoryCommand::Init { name, out, force } => {
                    let out = out.unwrap_or_else(|| PathBuf::from(format!("{name}.theme.toml")));
                    if out.exists() && !force {
                        anyhow::bail!(
                            "{} already exists — pass --force to overwrite it",
                            out.display()
                        );
                    }
                    let scaffold = purpose_factory::theme_config::scaffold(&name);
                    std::fs::write(&out, scaffold.as_bytes())
                        .with_context(|| format!("cannot write {}", out.display()))?;
                    println!("wrote {}", out.display());
                    println!("edit it, then run `purpose factory build {}`", out.display());
                }

                FactoryCommand::Build {
                    config,
                    out,
                    registry,
                    raw,
                } => {
                    let theme_config = purpose_factory::ThemeConfig::from_file(&config)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let name = theme_config.name.clone();
                    let contract = theme_config
                        .into_contract()
                        .map_err(|e| anyhow::anyhow!("{e}"))?;

                    let out_dir = out.unwrap_or_else(|| {
                        root.join(".purpose").join("factory").join(&name)
                    });

                    eprintln!("Building theme '{name}' -> {} ...", out_dir.display());
                    let model = purpose_factory::Factory::build(contract, &out_dir)
                        .await
                        .map_err(|e| anyhow::anyhow!("{e}"))?;

                    let registry_path = registry.unwrap_or_else(|| {
                        root.join(".purpose").join("factory").join("registry.json")
                    });
                    purpose_factory::Registry::new(&registry_path)
                        .record(&model)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;

                    if raw {
                        println!("{}", serde_json::to_string_pretty(&model)?);
                    } else {
                        println!(
                            "Built '{}': {} document(s), {} example(s), vocab {} -> {}",
                            model.name,
                            model.document_count,
                            model.example_count,
                            model.vocab_size,
                            model.path.display()
                        );
                    }
                }

                FactoryCommand::List { registry, raw } => {
                    let registry_path = registry.unwrap_or_else(|| {
                        root.join(".purpose").join("factory").join("registry.json")
                    });
                    let models = purpose_factory::Registry::new(&registry_path)
                        .load()
                        .map_err(|e| anyhow::anyhow!("{e}"))?;

                    if raw {
                        println!("{}", serde_json::to_string_pretty(&models)?);
                    } else if models.is_empty() {
                        println!("no theme models recorded in {}", registry_path.display());
                    } else {
                        for m in models {
                            println!(
                                "{}  {} doc(s), {} example(s)  {}",
                                m.name,
                                m.document_count,
                                m.example_count,
                                m.path.display()
                            );
                        }
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
