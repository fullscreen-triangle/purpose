use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::error::Error;

/// A single ingested unit of text, with provenance metadata.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub metadata: BTreeMap<String, String>,
}

/// A source of documents for a theme's training corpus. Distinct from
/// `purpose_operations::Provider` (single-operation invoke) because ingestion
/// is a batch pull, not a request/response call.
#[async_trait::async_trait]
pub trait SourceProvider: Send + Sync {
    async fn fetch(&self) -> Result<Vec<Document>, Error>;
}

pub async fn fetch_all(sources: &[Box<dyn SourceProvider>]) -> Result<Vec<Document>, Error> {
    let mut docs = Vec::new();
    for source in sources {
        docs.extend(source.fetch().await?);
    }
    Ok(docs)
}

// =====================================================================
// Local files
// =====================================================================

/// Walks a filesystem root, extracting text from `.tex`, `.pdf`, `.md`,
/// `.txt`, `.csv`, `.json`.
pub struct LocalFileSource {
    pub root: PathBuf,
    /// Optional glob-like suffix filter, e.g. only "*.tex". Empty = all supported extensions.
    pub extensions: Vec<String>,
}

impl LocalFileSource {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            extensions: vec![
                "tex".into(),
                "pdf".into(),
                "md".into(),
                "txt".into(),
                "csv".into(),
                "json".into(),
            ],
        }
    }
}

#[async_trait::async_trait]
impl SourceProvider for LocalFileSource {
    async fn fetch(&self) -> Result<Vec<Document>, Error> {
        let root = self.root.clone();
        let extensions = self.extensions.clone();
        tokio::task::spawn_blocking(move || walk_and_extract(&root, &extensions))
            .await
            .map_err(|e| Error::Source(e.to_string()))?
    }
}

fn walk_and_extract(root: &PathBuf, extensions: &[String]) -> Result<Vec<Document>, Error> {
    let mut docs = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if !extensions.iter().any(|e| e == &ext) {
            continue;
        }

        let text = match ext.as_str() {
            "tex" => std::fs::read_to_string(path).ok().map(|raw| strip_tex(&raw)),
            "pdf" => pdf_extract::extract_text(path).ok(),
            "md" | "txt" => std::fs::read_to_string(path).ok(),
            "csv" => std::fs::read_to_string(path).ok().map(|raw| flatten_csv(&raw)),
            "json" => std::fs::read_to_string(path).ok().and_then(|raw| flatten_json_str(&raw)),
            _ => None,
        };

        if let Some(text) = text {
            let text = text.trim().to_string();
            if text.is_empty() {
                continue;
            }
            let mut metadata = BTreeMap::new();
            metadata.insert("path".into(), path.display().to_string());
            metadata.insert("format".into(), ext);
            docs.push(Document {
                id: path.display().to_string(),
                text,
                metadata,
            });
        }
    }
    Ok(docs)
}

/// Minimal LaTeX-to-text reduction: drops comments, common macros/environments
/// that carry no prose content, and unwraps `\command{...}` to its argument.
/// Not a full LaTeX parser — good enough for corpus text, not for re-typesetting.
fn strip_tex(raw: &str) -> String {
    let no_comments: String = raw
        .lines()
        .map(|line| match line.find('%') {
            // Naive: doesn't handle escaped `\%`, acceptable for corpus extraction.
            Some(i) if !line[..i].ends_with('\\') => &line[..i],
            _ => line,
        })
        .collect::<Vec<_>>()
        .join("\n");

    // The `regex` crate does not support backreferences, so each dropped
    // environment is matched by its own fixed begin/end pair rather than a
    // single `\1`-backreferenced pattern.
    let mut stripped = no_comments;
    for env in ["figure\\*?", "table\\*?", "algorithm", "tikzpicture"] {
        let pattern = format!(r"(?s)\\begin\{{{env}\}}.*?\\end\{{{env}\}}");
        let drop_env = regex::Regex::new(&pattern).unwrap();
        stripped = drop_env.replace_all(&stripped, "").into_owned();
    }

    let drop_preamble =
        regex::Regex::new(r"(?s)\\documentclass.*?\\begin\{document\}").unwrap();
    let stripped = drop_preamble.replace(&stripped, "");

    let strip_command = regex::Regex::new(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?\{([^{}]*)\}").unwrap();
    let mut text = stripped.to_string();
    for _ in 0..3 {
        text = strip_command.replace_all(&text, "$2").to_string();
    }

    let drop_remaining = regex::Regex::new(r"\\[a-zA-Z]+\*?").unwrap();
    let text = drop_remaining.replace_all(&text, " ");

    let collapse_ws = regex::Regex::new(r"[ \t]+").unwrap();
    let text = collapse_ws.replace_all(&text, " ");
    let collapse_blank = regex::Regex::new(r"\n{3,}").unwrap();
    collapse_blank.replace_all(&text, "\n\n").trim().to_string()
}

/// Renders a CSV's rows as `col1: val1, col2: val2, ...` lines, one row per
/// line, using the header row for column names (falling back to positional
/// `col0`, `col1`, ... if the file has none `csv` can detect, i.e. is
/// malformed enough that even the header read fails). Deterministic, no
/// schema configuration required — good enough for corpus text without
/// modeling the CSV's actual structure.
fn flatten_csv(raw: &str) -> String {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(raw.as_bytes());

    let headers: Vec<String> = match reader.headers() {
        Ok(h) => h.iter().map(str::to_string).collect(),
        Err(_) => Vec::new(),
    };

    let mut lines = Vec::new();
    for result in reader.records() {
        let Ok(record) = result else { continue };
        let cells: Vec<String> = record
            .iter()
            .enumerate()
            .map(|(i, val)| {
                let col = headers.get(i).cloned().unwrap_or_else(|| format!("col{i}"));
                format!("{col}: {val}")
            })
            .collect();
        if !cells.is_empty() {
            lines.push(cells.join(", "));
        }
    }
    lines.join("\n")
}

/// Parses a JSON document and flattens it to `key: value` lines. Objects
/// recurse with dot-joined paths (`a.b.c: 1`); arrays of scalars render
/// inline (`tags: [x, y, z]`); arrays of objects render one flattened block
/// per element, blank-line separated, so each element reads like one
/// record. Falls back to the raw pretty-printed JSON if parsing fails
/// (malformed JSON is still text a model can learn some structure from)
/// or the value is empty.
fn flatten_json_str(raw: &str) -> Option<String> {
    match serde_json::from_str::<serde_json::Value>(raw) {
        Ok(value) => {
            let mut out = String::new();
            flatten_json(&value, "", &mut out);
            let out = out.trim().to_string();
            if out.is_empty() { None } else { Some(out) }
        }
        Err(_) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
        }
    }
}

fn flatten_json(value: &serde_json::Value, prefix: &str, out: &mut String) {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten_json(v, &path, out);
            }
        }
        serde_json::Value::Array(items) => {
            let all_scalar = items
                .iter()
                .all(|v| !matches!(v, serde_json::Value::Object(_) | serde_json::Value::Array(_)));
            if all_scalar {
                let rendered: Vec<String> = items.iter().map(scalar_to_string).collect();
                out.push_str(&format!("{prefix}: [{}]\n", rendered.join(", ")));
            } else {
                for item in items {
                    flatten_json(item, prefix, out);
                    out.push('\n');
                }
            }
        }
        other => {
            out.push_str(&format!("{prefix}: {}\n", scalar_to_string(other)));
        }
    }
}

fn scalar_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

// =====================================================================
// Email (IMAP)
// =====================================================================

pub struct ImapSource {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub mailbox: String,
    /// IMAP SEARCH criteria, e.g. "ALL" or "SINCE 01-Jan-2026".
    pub search: String,
}

#[async_trait::async_trait]
impl SourceProvider for ImapSource {
    async fn fetch(&self) -> Result<Vec<Document>, Error> {
        let cfg = ImapConfig {
            host: self.host.clone(),
            port: self.port,
            username: self.username.clone(),
            password: self.password.clone(),
            mailbox: self.mailbox.clone(),
            search: self.search.clone(),
        };
        tokio::task::spawn_blocking(move || fetch_imap(&cfg))
            .await
            .map_err(|e| Error::Source(e.to_string()))?
    }
}

struct ImapConfig {
    host: String,
    port: u16,
    username: String,
    password: String,
    mailbox: String,
    search: String,
}

fn fetch_imap(cfg: &ImapConfig) -> Result<Vec<Document>, Error> {
    let client = imap::ClientBuilder::new(cfg.host.as_str(), cfg.port)
        .connect()
        .map_err(|e| Error::Source(format!("imap connect: {e}")))?;
    let mut session = client
        .login(&cfg.username, &cfg.password)
        .map_err(|(e, _client)| Error::Source(format!("imap login: {e}")))?;

    session
        .select(&cfg.mailbox)
        .map_err(|e| Error::Source(format!("imap select {}: {e}", cfg.mailbox)))?;

    let ids = session
        .search(&cfg.search)
        .map_err(|e| Error::Source(format!("imap search: {e}")))?;

    let mut docs = Vec::new();
    for uid in ids {
        let messages = session
            .fetch(uid.to_string(), "RFC822")
            .map_err(|e| Error::Source(format!("imap fetch {uid}: {e}")))?;
        for msg in messages.iter() {
            let Some(body) = msg.body() else { continue };
            let Some(parsed) = mail_parser::MessageParser::default().parse(body) else {
                continue;
            };

            let subject = parsed.subject().unwrap_or_default().to_string();
            let text = parsed
                .body_text(0)
                .map(|c| c.to_string())
                .unwrap_or_default();
            if text.trim().is_empty() {
                continue;
            }

            let mut metadata = BTreeMap::new();
            metadata.insert("subject".into(), subject.clone());
            metadata.insert("mailbox".into(), cfg.mailbox.clone());
            if let Some(addr) = parsed.from().and_then(|f| f.first()).and_then(|a| a.address.as_ref()) {
                metadata.insert("from".into(), addr.to_string());
            }

            docs.push(Document {
                id: format!("{}:{}", cfg.mailbox, uid),
                text: format!("{subject}\n\n{text}"),
                metadata,
            });
        }
    }

    let _ = session.logout();
    Ok(docs)
}

// =====================================================================
// URL / web
// =====================================================================

pub struct UrlSource {
    pub urls: Vec<String>,
    pub client: reqwest::Client,
}

impl UrlSource {
    pub fn new(urls: Vec<String>) -> Self {
        Self {
            urls,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl SourceProvider for UrlSource {
    async fn fetch(&self) -> Result<Vec<Document>, Error> {
        let mut docs = Vec::new();
        for url in &self.urls {
            let resp = self
                .client
                .get(url)
                .send()
                .await
                .map_err(|e| Error::Source(format!("fetch {url}: {e}")))?;
            let html = resp
                .text()
                .await
                .map_err(|e| Error::Source(format!("read {url}: {e}")))?;
            let text = html2text::from_read(html.as_bytes(), 100);
            let text = text.trim().to_string();
            if text.is_empty() {
                continue;
            }

            let mut metadata = BTreeMap::new();
            metadata.insert("url".into(), url.clone());
            docs.push(Document {
                id: url.clone(),
                text,
                metadata,
            });
        }
        Ok(docs)
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;

    #[test]
    fn flattens_csv_rows_with_header() {
        let csv = "name,age\nAda,36\nGrace,85\n";
        let text = flatten_csv(csv);
        assert_eq!(text, "name: Ada, age: 36\nname: Grace, age: 85");
    }

    #[test]
    fn flattens_csv_with_quoted_commas() {
        let csv = "name,note\nAda,\"loves, algebra\"\n";
        let text = flatten_csv(csv);
        assert_eq!(text, "name: Ada, note: loves, algebra");
    }

    #[test]
    fn flattens_json_object_to_key_value_lines() {
        let json = r#"{"name": "Ada", "born": 1815}"#;
        let text = flatten_json_str(json).unwrap();
        assert!(text.contains("name: Ada"));
        assert!(text.contains("born: 1815"));
    }

    #[test]
    fn flattens_nested_json_with_dot_paths() {
        let json = r#"{"person": {"name": "Ada", "field": "math"}}"#;
        let text = flatten_json_str(json).unwrap();
        assert!(text.contains("person.name: Ada"));
        assert!(text.contains("person.field: math"));
    }

    #[test]
    fn flattens_array_of_objects_as_separate_blocks() {
        let json = r#"[{"name": "Ada"}, {"name": "Grace"}]"#;
        let text = flatten_json_str(json).unwrap();
        assert!(text.contains("name: Ada"));
        assert!(text.contains("name: Grace"));
    }

    #[test]
    fn flattens_scalar_array_inline() {
        let json = r#"{"tags": ["math", "computing"]}"#;
        let text = flatten_json_str(json).unwrap();
        assert!(text.contains("tags: [math, computing]"));
    }

    #[test]
    fn falls_back_to_raw_text_on_malformed_json() {
        let malformed = "{not valid json";
        let text = flatten_json_str(malformed).unwrap();
        assert_eq!(text, malformed);
    }

    #[test]
    fn empty_json_yields_none() {
        assert!(flatten_json_str("").is_none());
        assert!(flatten_json_str("{}").is_none());
    }
}
