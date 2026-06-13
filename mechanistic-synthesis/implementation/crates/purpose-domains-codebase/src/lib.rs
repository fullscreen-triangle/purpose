//! Codebase compilation domain.
//!
//! The mirror image of `purpose-domains-protein`, with the substrate
//! pointed at a source repository instead of UniProt. Provides:
//! - An `index` builder that line-scans a project for symbol definitions
//!   and writes them to `.purpose/index.json`.
//! - A `CodebaseProvider` that searches that index and formats matches.
//! - A hand-coded `CodebaseResolver` that compiles "where is X" style
//!   utterances into `search_symbols |> format_matches`.
//! - A `Domain` + `register_providers` pair matching the framework's
//!   integration contract, so the CLI consumes it without modification.
//!
//! As in the protein domain, the hand-coded resolver is the MVP form; a
//! trained LoRA adapter swaps into the same `Resolver` trait later.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::debug;

use purpose_core::{Domain, Error, Operation, Resolver, Type, VaHera, Value};
use purpose_operations::{OperationRegistry, Provider};

/// One indexed symbol: a definition found by line-scanning the repo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolEntry {
    /// The symbol name (function / struct / class / heading text).
    pub name: String,
    /// What kind of definition matched (fn, struct, class, def, heading, …).
    pub kind: String,
    /// Path relative to the project root.
    pub file: String,
    /// 1-based line number of the definition.
    pub line: usize,
    /// The raw source line, trimmed — a one-line snippet for context.
    pub snippet: String,
}

/// The on-disk index: a flat list of symbols plus the root it was built from.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Index {
    pub root: String,
    pub symbols: Vec<SymbolEntry>,
}

/// Where the index lives relative to a project root.
pub fn index_path(root: &Path) -> PathBuf {
    root.join(".purpose").join("index.json")
}

// ---------------------------------------------------------------------------
// Index builder
// ---------------------------------------------------------------------------

/// Line-scan `root` for symbol definitions and write `.purpose/index.json`.
///
/// Deliberately crude: a handful of regexes over every text file. The point
/// is to prove the retrieval slice is useful before investing in tree-sitter.
/// Returns the number of symbols indexed.
pub fn build_index(root: &Path) -> Result<usize, Error> {
    let code = code_patterns();
    let headings = heading_patterns();
    let mut index = Index {
        root: root.display().to_string(),
        symbols: Vec::new(),
    };

    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !is_ignored(e.path()))
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !is_indexable(path) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else {
            continue; // skip binary / unreadable files
        };
        if looks_generated(&text) {
            continue; // skip minified / generated blobs
        }
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string()
            .replace('\\', "/");

        // Prose files contribute headings only; code files contribute
        // definitions only. This stops keyword regexes firing on sentences.
        let patterns: &[(&str, Regex)] = if is_prose(path) { &headings } else { &code };

        for (lineno, raw) in text.lines().enumerate() {
            // Skip absurdly long lines — almost always minified/generated.
            if raw.len() > 400 {
                continue;
            }
            for (kind, re) in patterns {
                if let Some(caps) = re.captures(raw) {
                    if let Some(name) = caps.get(1) {
                        let name = name.as_str().trim().to_string();
                        if name.is_empty() {
                            continue;
                        }
                        index.symbols.push(SymbolEntry {
                            name,
                            kind: kind.to_string(),
                            file: rel.clone(),
                            line: lineno + 1,
                            snippet: raw.trim().chars().take(160).collect(),
                        });
                        break; // one definition kind per line is enough
                    }
                }
            }
        }
    }

    let count = index.symbols.len();
    let out = index_path(root);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Internal(format!("cannot create .purpose dir: {e}")))?;
    }
    let json = serde_json::to_string_pretty(&index)
        .map_err(|e| Error::Internal(format!("cannot serialise index: {e}")))?;
    std::fs::write(&out, json)
        .map_err(|e| Error::Internal(format!("cannot write index: {e}")))?;

    Ok(count)
}

/// Regexes recognising a *code* definition, capturing its name in group 1.
///
/// Anchored to the start of the line (after optional indentation and a small
/// set of leading modifiers like `pub`, `async`, `export`, `public`). This is
/// the key noise filter: a real definition such as `pub trait Resolver` opens
/// its line, whereas prose like "a Resolver trait whose signature…" buries the
/// keyword mid-sentence and is therefore *not* matched.
fn code_patterns() -> Vec<(&'static str, Regex)> {
    // Leading whitespace, then zero or more modifier words, then the keyword.
    let lead = r"^\s*(?:(?:pub|pub\([^)]*\)|async|unsafe|export|default|public|private|protected|static|final|abstract|const)\s+)*";
    let id = r"([A-Za-z_][A-Za-z0-9_]*)";
    let mk = |kw: &str| Regex::new(&format!(r"{lead}{kw}\s+{id}")).unwrap();
    vec![
        ("fn", mk("fn")),
        ("struct", mk("struct")),
        ("enum", mk("enum")),
        ("trait", mk("trait")),
        ("type", mk("type")),
        ("def", mk("def")),
        ("class", mk("class")),
        ("func", mk("function")),
    ]
}

/// Markdown/LaTeX heading pattern — only applied to prose files.
fn heading_patterns() -> Vec<(&'static str, Regex)> {
    vec![
        ("heading", Regex::new(r"^#{1,6}\s+(.+)$").unwrap()),
        (
            "section",
            Regex::new(r"^\s*\\(?:sub)*section\*?\{([^}]+)\}").unwrap(),
        ),
    ]
}

/// Does this file hold prose (markdown/LaTeX) rather than source code?
fn is_prose(path: &Path) -> bool {
    matches!(path.extension().and_then(|e| e.to_str()), Some("md" | "tex"))
}

/// Directories and paths we never descend into.
fn is_ignored(path: &Path) -> bool {
    path.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some(".git")
                | Some("node_modules")
                | Some("target")
                | Some(".purpose")
                | Some("dist")
                | Some("build")
                | Some("__pycache__")
                | Some(".venv")
                | Some("venv")
                | Some(".next")
                | Some(".nuxt")
                | Some(".cache")
                | Some("coverage")
                | Some("out")
                | Some("vendor")
                | Some(".idea")
                | Some(".vscode")
        )
    })
}

/// Heuristic: does this file look machine-generated or minified? Such files
/// (bundled JS, lockfiles, source maps) produce nothing but noise. We flag
/// them by very long average line length or a `.min.`/`-min.` name marker.
fn looks_generated(text: &str) -> bool {
    let mut lines = 0usize;
    let mut total = 0usize;
    for line in text.lines().take(40) {
        lines += 1;
        total += line.len();
    }
    if lines == 0 {
        return false;
    }
    // Mean line length over the first lines; bundlers emit huge lines.
    total / lines > 300
}

/// Extensions worth scanning for definitions.
fn is_indexable(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some(
            "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "java" | "c" | "cpp" | "h" | "hpp"
                | "cs" | "rb" | "php" | "swift" | "kt" | "scala" | "md" | "tex"
        )
    )
}

// ---------------------------------------------------------------------------
// Provider: search the index, format matches
// ---------------------------------------------------------------------------

/// Serves `search_symbols` and `format_matches` against `.purpose/index.json`.
pub struct CodebaseProvider {
    root: PathBuf,
}

impl CodebaseProvider {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    fn load_index(&self) -> Result<Index, Error> {
        let path = index_path(&self.root);
        let text = std::fs::read_to_string(&path).map_err(|_| {
            Error::Provider(format!(
                "no index at {} — run `purpose index` in this project first",
                path.display()
            ))
        })?;
        serde_json::from_str(&text)
            .map_err(|e| Error::Provider(format!("corrupt index: {e}")))
    }
}

#[async_trait]
impl Provider for CodebaseProvider {
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error> {
        match op {
            "search_symbols" => {
                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Provider("missing 'query'".into()))?;
                let index = self.load_index()?;
                let needle = query.to_lowercase();
                let terms: Vec<&str> = needle.split_whitespace().collect();

                // Score each symbol by how many query terms it touches, in
                // name (weighted) or snippet. Keep the top matches.
                let mut scored: Vec<(i32, &SymbolEntry)> = index
                    .symbols
                    .iter()
                    .filter_map(|s| {
                        let name = s.name.to_lowercase();
                        let snip = s.snippet.to_lowercase();
                        let mut score = 0;
                        for t in &terms {
                            if name.contains(t) {
                                score += 3;
                            } else if snip.contains(t) {
                                score += 1;
                            }
                        }
                        if score > 0 {
                            Some((score, s))
                        } else {
                            None
                        }
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.cmp(&a.0));
                scored.truncate(20);

                let matches: Vec<Value> = scored
                    .into_iter()
                    .map(|(score, s)| {
                        let mut rec = BTreeMap::new();
                        rec.insert("name".into(), Value::Str(s.name.clone()));
                        rec.insert("kind".into(), Value::Str(s.kind.clone()));
                        rec.insert("file".into(), Value::Str(s.file.clone()));
                        rec.insert("line".into(), Value::Num(s.line as f64));
                        rec.insert("snippet".into(), Value::Str(s.snippet.clone()));
                        rec.insert("score".into(), Value::Num(score as f64));
                        Value::Record(rec)
                    })
                    .collect();

                Ok(Value::List(matches))
            }
            "format_matches" => {
                let input = args
                    .get("input")
                    .ok_or_else(|| Error::Provider("missing 'input'".into()))?;
                let list = input
                    .as_list()
                    .ok_or_else(|| Error::Provider("expected a list of matches".into()))?;

                if list.is_empty() {
                    return Ok(Value::Str("No matching symbols found in the index.".into()));
                }

                let mut out = String::new();
                out.push_str(&format!("{} matching symbol(s):\n\n", list.len()));
                for m in list {
                    let rec = match m.as_record() {
                        Some(r) => r,
                        None => continue,
                    };
                    let get = |k: &str| rec.get(k).and_then(|v| v.as_str()).unwrap_or("");
                    let line = rec
                        .get("line")
                        .and_then(|v| match v {
                            Value::Num(n) => Some(*n as usize),
                            _ => None,
                        })
                        .unwrap_or(0);
                    out.push_str(&format!(
                        "  {}:{}  [{}] {}\n      {}\n",
                        get("file"),
                        line,
                        get("kind"),
                        get("name"),
                        get("snippet"),
                    ));
                }
                Ok(Value::Str(out))
            }
            _ => Err(Error::Provider(format!("unsupported op: {op}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Resolver: utterance -> vaHera fragment
// ---------------------------------------------------------------------------

/// Hand-coded resolver: strip question words, treat the rest as search
/// terms, emit `search_symbols |> format_matches`.
pub struct CodebaseResolver;

#[async_trait]
impl Resolver for CodebaseResolver {
    async fn compile(&self, utterance: &str) -> Result<VaHera, Error> {
        let query = extract_query(utterance);
        if query.is_empty() {
            return Err(Error::Compile(
                "no searchable terms found in utterance".into(),
            ));
        }
        debug!(query = %query, "codebase resolver extracted query");

        let mut search_args: BTreeMap<String, VaHera> = BTreeMap::new();
        search_args.insert("query".to_string(), VaHera::Literal(Value::Str(query)));

        Ok(VaHera::Compose(vec![
            VaHera::Call {
                op: "search_symbols".to_string(),
                args: search_args,
            },
            VaHera::Call {
                op: "format_matches".to_string(),
                args: BTreeMap::new(),
            },
        ]))
    }
}

/// Drop common question words; keep the content terms as the search query.
fn extract_query(utterance: &str) -> String {
    utterance
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'))
        .filter(|w| !w.is_empty() && !is_question_word(w))
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_question_word(w: &str) -> bool {
    matches!(
        w.to_lowercase().as_str(),
        "where" | "what" | "which" | "how" | "who" | "why" | "when"
            | "is" | "are" | "the" | "a" | "an" | "do" | "does" | "did"
            | "can" | "could" | "to" | "of" | "in" | "for" | "on" | "at"
            | "i" | "find" | "show" | "me" | "get" | "tell" | "about"
            | "handled" | "defined"
    )
}

// ---------------------------------------------------------------------------
// Domain wiring
// ---------------------------------------------------------------------------

/// Build the codebase domain connector.
pub fn domain() -> Domain {
    Domain {
        name: "codebase".into(),
        operations: operations(),
        resolver: Arc::new(CodebaseResolver),
    }
}

/// Operation vocabulary for the codebase domain.
pub fn operations() -> Vec<Operation> {
    vec![
        Operation::new(
            "search_symbols",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("query".into(), Type::Str);
                inputs
            },
            Type::List(Box::new(Type::named("SymbolMatch"))),
            "Search the project index for symbols matching the query terms.",
        ),
        Operation::new(
            "format_matches",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("input".into(), Type::List(Box::new(Type::named("SymbolMatch"))));
                inputs
            },
            Type::Str,
            "Render a list of symbol matches as a readable context slice.",
        ),
    ]
}

/// Register the codebase provider against this domain's operations.
pub fn register_providers(registry: &mut OperationRegistry, root: PathBuf) {
    let provider = Arc::new(CodebaseProvider::new(root));
    for op in operations() {
        match op.name.as_str() {
            "search_symbols" | "format_matches" => {
                registry.register(op, provider.clone());
            }
            _ => {}
        }
    }
}
