//! The lens: how an AI decides what distinctions its repository draws.
//!
//! The term map τ was hard-coded once, so every repository induced the same
//! shape of graph and there was no way to sharpen it. A lens moves that choice
//! into a checked-in `.purpose/lens.toml` — which paths and kinds are admitted,
//! which tokens are noise, what a distinction is worth, how a shared
//! distinction becomes an edge weight.
//!
//! **What licenses this.** `thm:tau-agnostic` proves the calculus is correct
//! for *any* τ, because no theorem downstream inspects where an edge came from.
//! `cor:coarsen` bounds the damage of a poor lens: it coarsens cells, it cannot
//! make a determination unsound relative to the graph it induces. So τ is the
//! one component that can be handed over without re-deriving anything.
//!
//! **What is genuinely new.** The catalogue defines τ as a map to *sets*
//! (`def:term-map`). A lens produces a map to *weighted* sets, and an edge
//! weight becomes `Σ min(w_u(t), w_v(t))` over the shared terms rather than a
//! count. That is a strict generalisation, not a reinterpretation: at unit
//! weights the sum is the count and the paper's construction is recovered
//! exactly. The calculus is untouched; only the map from τ to the graph is
//! wider. `thm:tau-agnostic` still covers the result, since its premise is only
//! that the graph has a medium and no weight beneath the floor.
//!
//! **What a lens cannot do.** It cannot make the floor mean something else.
//! `rem:quality-honest` stands: a lens under which every module draws identical
//! distinctions induces among the highest floors while discriminating worst.
//! There is deliberately no score here to tune against.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde::{Deserialize, Serialize};

use purpose_ckg::{WeightedTermMap, FLOOR};
use purpose_domains_codebase::{Index, SymbolEntry};

use crate::Granularity;

/// Where a lens is looked for, relative to the repository root.
pub const LENS_FILE: &str = ".purpose/lens.toml";

/// The kinds the codebase indexer can emit. Naming an unknown kind is a
/// warning rather than an error: this list belongs to the indexer, and coupling
/// the lens schema to it would make every indexer change a breaking one.
pub const KNOWN_KINDS: &[&str] = &[
    "fn", "struct", "enum", "trait", "type", "def", "class", "func", "heading", "section",
];

/// Function words, which draw no distinction.
///
/// A heading token like `the` or `with` appears across most documents in most
/// repositories, so admitting it puts every prose module in contact with every
/// other. Excluding these does not make the map *correct* — no extraction is —
/// it stops one known artefact from dominating the contact structure.
pub const DEFAULT_STOPWORDS: &[&str] = &[
    "the", "and", "not", "for", "with", "from", "that", "this", "what", "when", "where",
    "which", "how", "why", "who", "are", "was", "were", "has", "have", "had", "can", "will",
    "its", "into", "out", "over", "than", "then", "there", "does", "did", "you", "your",
    "our", "all", "any", "but", "use", "using", "used", "via", "per", "about", "also",
];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// What went wrong reading a lens, and where.
#[derive(Debug, Clone)]
pub struct LensError {
    pub path: Option<PathBuf>,
    pub line: usize,
    pub kind: LensErrorKind,
}

#[derive(Debug, Clone)]
pub enum LensErrorKind {
    Syntax(String),
    UnknownTable { got: String },
    UnknownKey { table: String, got: String, suggestion: Option<String> },
    BadValue { key: String, expected: &'static str, got: String },
    Unsupported { what: &'static str },
    BadGlob { pattern: String, reason: String },
    Semantic(String),
}

impl fmt::Display for LensError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let where_ = match &self.path {
            Some(p) => p.display().to_string(),
            None => "<lens>".to_string(),
        };
        write!(f, "{where_}:{}: ", self.line)?;
        match &self.kind {
            LensErrorKind::Syntax(m) => write!(f, "{m}"),
            LensErrorKind::UnknownTable { got } => write!(
                f,
                "unknown table [{got}] — known tables are [lens], [include], [terms], \
                 [terms.alias], [terms.weight], [edges]"
            ),
            LensErrorKind::UnknownKey { table, got, suggestion } => {
                write!(f, "unknown key '{got}' in [{table}]")?;
                match suggestion {
                    Some(s) => write!(f, " — did you mean '{s}'?"),
                    None => Ok(()),
                }
            }
            LensErrorKind::BadValue { key, expected, got } => {
                write!(f, "'{key}' expects {expected}, got `{got}`")
            }
            LensErrorKind::Unsupported { what } => write!(
                f,
                "{what} is valid TOML but not accepted here — a lens uses only strings, \
                 numbers, booleans and arrays of strings"
            ),
            LensErrorKind::BadGlob { pattern, reason } => {
                write!(f, "cannot use path pattern '{pattern}': {reason}")
            }
            LensErrorKind::Semantic(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for LensError {}

impl From<LensError> for purpose_core::Error {
    fn from(e: LensError) -> Self {
        purpose_core::Error::Parse(e.to_string())
    }
}

fn err(line: usize, kind: LensErrorKind) -> LensError {
    LensError { path: None, line, kind }
}

// ---------------------------------------------------------------------------
// The lens
// ---------------------------------------------------------------------------

/// How a shared distinction becomes an edge weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeWeight {
    /// `Σ min(w_u(t), w_v(t))` over shared terms. At unit weights this is the
    /// shared count — the catalogue's own construction.
    Sum,
    /// `|shared|`, ignoring term weights entirely. The explicit way to say that
    /// `[terms.weight]` should affect diagnostics and nothing else.
    Count,
    /// Weighted Jaccard, in `(0, 1]`. Note that with the default floor of 1.0
    /// every contact clamps to the floor and the graph goes flat.
    Jaccard,
}

impl EdgeWeight {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sum" => Some(EdgeWeight::Sum),
            "count" => Some(EdgeWeight::Count),
            "jaccard" => Some(EdgeWeight::Jaccard),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeWeight::Sum => "sum",
            EdgeWeight::Count => "count",
            EdgeWeight::Jaccard => "jaccard",
        }
    }
}

/// A compiled path pattern. The source is kept so diagnostics and the stored
/// form can show what was written rather than what it compiled to.
#[derive(Clone, Debug)]
pub struct PathPattern {
    pub source: String,
    regex: Regex,
}

impl PathPattern {
    pub fn new(source: &str) -> Result<Self, LensError> {
        let regex = Regex::new(&glob_to_regex(source)).map_err(|e| {
            err(
                0,
                LensErrorKind::BadGlob { pattern: source.to_string(), reason: e.to_string() },
            )
        })?;
        Ok(PathPattern { source: source.to_string(), regex })
    }

    pub fn matches(&self, path: &str) -> bool {
        self.regex.is_match(path)
    }
}

/// Translate a glob into an anchored regex.
///
/// Paths in the index are relative and forward-slash normalised, so only `/`
/// needs handling. `*` stays within one segment; `**` crosses them.
fn glob_to_regex(glob: &str) -> String {
    let bytes: Vec<char> = glob.chars().collect();
    let mut out = String::from("^");
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            '*' if i + 1 < bytes.len() && bytes[i + 1] == '*' => {
                if i + 2 < bytes.len() && bytes[i + 2] == '/' {
                    // `**/` — zero or more whole segments.
                    out.push_str("(?:[^/]+/)*");
                    i += 3;
                } else if i + 2 == bytes.len() && i > 0 && bytes[i - 1] == '/' {
                    // trailing `/**` — this directory and everything beneath.
                    out.truncate(out.len() - 1); // drop the `/` just emitted
                    out.push_str("(?:/.*)?");
                    i += 2;
                } else {
                    out.push_str(".*");
                    i += 2;
                }
            }
            '*' => {
                out.push_str("[^/]*");
                i += 1;
            }
            '?' => {
                out.push_str("[^/]");
                i += 1;
            }
            c => {
                out.push_str(&regex::escape(&c.to_string()));
                i += 1;
            }
        }
    }
    out.push('$');
    out
}

/// Which index entries are admitted.
#[derive(Clone, Debug)]
pub struct Include {
    /// Empty means every path.
    pub paths: Vec<PathPattern>,
    /// `None` means every kind.
    pub kinds: Option<BTreeSet<String>>,
}

/// How an admitted entry becomes distinctions.
#[derive(Clone, Debug)]
pub struct Terms {
    pub stopwords: BTreeSet<String>,
    pub min_len: usize,
    pub split_camel_case: bool,
    /// Kinds treated as prose: their names are phrases to be split, not
    /// identifiers to be taken whole.
    pub prose_kinds: BTreeSet<String>,
    pub alias: BTreeMap<String, String>,
    /// Ordered, first match wins — so `"trait:*"` before `"*"` is meaningful.
    pub weight: Vec<(String, f64)>,
}

/// The instrument. Everything about how a repository is turned into a graph,
/// except the calculus itself.
#[derive(Clone, Debug)]
pub struct Lens {
    pub granularity: Granularity,
    pub floor: f64,
    pub include: Include,
    pub terms: Terms,
    pub edges: EdgeWeight,
    /// Where this lens was read from, if anywhere.
    pub source: Option<String>,
}

impl Default for Lens {
    fn default() -> Self {
        Lens {
            granularity: Granularity::File,
            floor: FLOOR,
            include: Include { paths: Vec::new(), kinds: None },
            terms: Terms {
                stopwords: DEFAULT_STOPWORDS.iter().map(|s| s.to_string()).collect(),
                min_len: 3,
                split_camel_case: false,
                // Both prose kinds the indexer emits. `section` was formerly
                // omitted, so a LaTeX `\section{The Directional Pair}` was taken
                // whole as an identifier instead of split into content words.
                prose_kinds: ["heading", "section"].iter().map(|s| s.to_string()).collect(),
                alias: BTreeMap::new(),
                weight: Vec::new(),
            },
            edges: EdgeWeight::Sum,
            source: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Term construction
// ---------------------------------------------------------------------------

/// Split a token on case and letter/digit boundaries.
fn camel_pieces(tok: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev: Option<char> = None;
    for c in tok.chars() {
        let boundary = match prev {
            Some(p) => {
                (p.is_lowercase() && c.is_uppercase())
                    || (p.is_alphabetic() && c.is_ascii_digit())
                    || (p.is_ascii_digit() && c.is_alphabetic())
            }
            None => false,
        };
        if boundary && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
        prev = Some(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.into_iter().map(|s| s.to_lowercase()).collect()
}

impl Lens {
    /// Is this entry admitted by the include rules?
    fn admits(&self, entry: &SymbolEntry) -> bool {
        self.admits_path(entry) && self.admits_kind(entry)
    }

    /// Is this entry admitted by `include.paths`?
    ///
    /// Separate from the kind test so the diagnostics can report the two stages
    /// as distinct numbers: a lens that drops most of the index by path is a
    /// different mistake from one that drops it by kind, and a single "admitted"
    /// count cannot tell the operator which they made.
    pub fn admits_path(&self, entry: &SymbolEntry) -> bool {
        if self.include.paths.is_empty() {
            return true;
        }
        let file = entry.file.replace('\\', "/");
        self.include.paths.iter().any(|p| p.matches(&file))
    }

    /// Is this entry admitted by `include.kinds`?
    pub fn admits_kind(&self, entry: &SymbolEntry) -> bool {
        match &self.include.kinds {
            Some(ks) => ks.contains(&entry.kind.trim().to_lowercase()),
            None => true,
        }
    }

    /// Apply an alias, if one is configured.
    fn alias(&self, t: &str) -> String {
        self.terms.alias.get(t).cloned().unwrap_or_else(|| t.to_string())
    }

    /// What a term is worth, given the kind of entry that contributed it.
    ///
    /// Patterns match on the *origin kind*, not on the text of the term.
    /// Prose terms are emitted bare — a heading contributes `directional`, never
    /// `heading:directional` — so a textual reading of `"heading:*"` would match
    /// nothing at all and the schema's own example would be a silent no-op.
    /// `"trait:*"` therefore means "any term contributed by a `trait` entry",
    /// which is also how anyone would read it.
    pub fn weight_of(&self, term: &str, origin_kind: &str) -> f64 {
        for (pat, w) in &self.terms.weight {
            if let Some(prefix) = pat.strip_suffix(":*") {
                if prefix == origin_kind || prefix == "*" {
                    return *w;
                }
            } else if pat == "*" {
                return *w;
            } else if let Some((k, t)) = pat.split_once(':') {
                if k == origin_kind && t == term {
                    return *w;
                }
            } else if pat == term {
                return *w;
            }
        }
        1.0
    }

    /// The distinctions one index entry draws, with what each is worth.
    pub fn terms_of(&self, entry: &SymbolEntry) -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        let name = entry.name.trim().to_lowercase();
        if name.is_empty() {
            return out;
        }
        let kind = entry.kind.trim().to_lowercase();

        // Merge by maximum, never by sum: a term contributed twice within a
        // module says nothing more than a term contributed once. Summing would
        // make a weight depend on how often the indexer's line scan happened to
        // fire, which is an artefact of extraction rather than a fact about the
        // module.
        let put = |t: String, w: f64, out: &mut BTreeMap<String, f64>| {
            let slot = out.entry(t).or_insert(w);
            if w > *slot {
                *slot = w;
            }
        };

        if self.terms.prose_kinds.contains(&kind) {
            // A heading is a phrase, not an identifier: split it and keep the
            // content words. Digits distinguish nothing, whatever the lens says.
            let raw = name.split(|c: char| !c.is_alphanumeric() && c != '_');
            let mut toks: Vec<String> = Vec::new();
            for tok in raw {
                if tok.is_empty() {
                    continue;
                }
                if self.terms.split_camel_case {
                    toks.extend(camel_pieces(tok));
                } else {
                    toks.push(tok.to_string());
                }
            }
            for tok in toks {
                if tok.len() < self.terms.min_len
                    || tok.chars().all(|c| c.is_ascii_digit())
                    || self.terms.stopwords.contains(&tok)
                {
                    continue;
                }
                let t = self.alias(&tok);
                let w = self.weight_of(&t, &kind);
                put(t, w, &mut out);
            }
            return out;
        }

        // An identifier is taken whole. Stopwords and the length floor are not
        // applied here: a `fn over` genuinely defines `over`, and dropping it
        // because the word is common in prose would lose a real distinction.
        let base = self.alias(&name);
        put(base.clone(), self.weight_of(&base, &kind), &mut out);
        if !kind.is_empty() {
            let kinded = format!("{kind}:{base}");
            let w = self.weight_of(&base, &kind);
            put(kinded, w, &mut out);
        }
        if self.terms.split_camel_case {
            for piece in camel_pieces(&name) {
                if piece.len() < self.terms.min_len || piece == name {
                    continue;
                }
                let t = self.alias(&piece);
                let w = self.weight_of(&t, &kind);
                put(t, w, &mut out);
            }
        }
        out
    }

    /// The weighted term map over modules at this lens's granularity.
    ///
    /// A module with no admitted symbols does not appear. It draws no
    /// distinctions this lens can see, so it has no contacts, and an item joined
    /// to nothing but the medium would sit at the floor and drag `β*` down for a
    /// reason that is an artefact of extraction rather than a fact about the
    /// repository.
    pub fn term_map(&self, index: &Index) -> WeightedTermMap {
        let mut tau: WeightedTermMap = BTreeMap::new();
        for entry in &index.symbols {
            if !self.admits(entry) {
                continue;
            }
            let terms = self.terms_of(entry);
            if terms.is_empty() {
                continue;
            }
            let module = self.granularity.module_of(&entry.file);
            let slot = tau.entry(module).or_default();
            for (t, w) in terms {
                let cur = slot.entry(t).or_insert(w);
                if w > *cur {
                    *cur = w;
                }
            }
        }
        tau.retain(|_, ts| !ts.is_empty());
        tau
    }

    /// The edge weight function this lens selects.
    ///
    /// A shared term is worth `min(w_u(t), w_v(t))`: the weight is the cost of
    /// telling two modules apart *on account of that term*, and the weaker party
    /// bounds what they genuinely hold in common. Taking the maximum would let
    /// one module's emphasis inflate a contact the other barely makes.
    pub fn edge_weight(
        &self,
    ) -> impl Fn(&BTreeMap<String, f64>, &BTreeMap<String, f64>) -> Option<f64> + '_ {
        let mode = self.edges;
        move |a, b| {
            let shared: Vec<f64> = a
                .iter()
                .filter_map(|(t, wa)| b.get(t).map(|wb| wa.min(*wb)))
                .collect();
            if shared.is_empty() {
                return None;
            }
            match mode {
                EdgeWeight::Count => Some(shared.len() as f64),
                EdgeWeight::Sum => Some(shared.iter().sum()),
                EdgeWeight::Jaccard => {
                    let inter: f64 = shared.iter().sum();
                    let wa: f64 = a.values().sum();
                    let wb: f64 = b.values().sum();
                    let union = wa + wb - inter;
                    Some(if union > 0.0 { inter / union } else { 0.0 })
                }
            }
        }
    }
}

/// Drop the weights, for the places that only ask whether a term is present.
///
/// Seeding is one of them: a weight says how much a shared distinction costs to
/// separate, not whether a goal meets it.
pub fn unweighted(tau: &WeightedTermMap) -> purpose_ckg::TermMap {
    tau.iter()
        .map(|(m, ts)| (m.clone(), ts.keys().cloned().collect()))
        .collect()
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

const KEYS_LENS: &[&str] = &["granularity", "floor"];
const KEYS_INCLUDE: &[&str] = &["paths", "kinds"];
const KEYS_TERMS: &[&str] =
    &["stopwords", "min_len", "split_camel_case", "prose_kinds"];
const KEYS_EDGES: &[&str] = &["weight"];

/// The nearest known key, by shared prefix then by length. Enough to catch a
/// plural dropped or a letter transposed, which is what typos in a short fixed
/// vocabulary actually look like.
fn suggest(got: &str, known: &[&str]) -> Option<String> {
    known
        .iter()
        .map(|k| {
            let shared = got
                .chars()
                .zip(k.chars())
                .take_while(|(a, b)| a == b)
                .count();
            (shared, *k)
        })
        .filter(|(shared, _)| *shared >= 2)
        .max_by_key(|(shared, k)| (*shared, usize::MAX - k.len()))
        .map(|(_, k)| k.to_string())
}

#[derive(Debug)]
enum Val {
    Str(String),
    Num(f64),
    Bool(bool),
    List(Vec<String>),
}

impl Val {
    fn type_name(&self) -> &'static str {
        match self {
            Val::Str(_) => "a string",
            Val::Num(_) => "a number",
            Val::Bool(_) => "a boolean",
            Val::List(_) => "a list",
        }
    }
}

/// Strip a trailing comment, respecting quotes.
fn strip_comment(s: &str) -> &str {
    let mut in_str = false;
    let mut esc = false;
    for (i, c) in s.char_indices() {
        if esc {
            esc = false;
            continue;
        }
        match c {
            '\\' if in_str => esc = true,
            '"' => in_str = !in_str,
            '#' if !in_str => return &s[..i],
            _ => {}
        }
    }
    s
}

fn parse_string(raw: &str, line: usize) -> Result<String, LensError> {
    let body = raw
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or_else(|| {
            err(line, LensErrorKind::Syntax(format!("unterminated string `{raw}`")))
        })?;
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => {
                    return Err(err(
                        line,
                        LensErrorKind::Syntax(format!("unknown escape `\\{other}`")),
                    ))
                }
                None => {
                    return Err(err(
                        line,
                        LensErrorKind::Syntax("string ends in a backslash".into()),
                    ))
                }
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

fn parse_value(raw: &str, line: usize) -> Result<Val, LensError> {
    let raw = raw.trim();
    if raw.starts_with('{') {
        return Err(err(line, LensErrorKind::Unsupported { what: "an inline table" }));
    }
    if raw.starts_with('"') {
        return Ok(Val::Str(parse_string(raw, line)?));
    }
    if raw == "true" {
        return Ok(Val::Bool(true));
    }
    if raw == "false" {
        return Ok(Val::Bool(false));
    }
    if raw.starts_with('[') {
        let inner = raw
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| {
                err(line, LensErrorKind::Syntax("unterminated array".into()))
            })?;
        let mut items = Vec::new();
        for part in split_top_level(inner) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if !part.starts_with('"') {
                return Err(err(
                    line,
                    LensErrorKind::Syntax(format!(
                        "a list may hold only quoted strings, found `{part}`"
                    )),
                ));
            }
            items.push(parse_string(part, line)?);
        }
        return Ok(Val::List(items));
    }
    raw.parse::<f64>().map(Val::Num).map_err(|_| {
        err(line, LensErrorKind::Syntax(format!("cannot read `{raw}` as a value")))
    })
}

/// Split on commas that are not inside a string.
fn split_top_level(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_str = false;
    let mut esc = false;
    for c in s.chars() {
        if esc {
            cur.push(c);
            esc = false;
            continue;
        }
        match c {
            '\\' if in_str => {
                cur.push(c);
                esc = true;
            }
            '"' => {
                in_str = !in_str;
                cur.push(c);
            }
            ',' if !in_str => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// Read a lens from TOML text.
///
/// The accepted grammar is deliberately narrower than TOML: strings, numbers,
/// booleans, and arrays of strings, in six named tables. Anything else is
/// refused *by name*, so a user who writes valid TOML we do not accept is told
/// which construct was the problem rather than given a syntax error.
pub fn parse_lens(text: &str) -> Result<Lens, LensError> {
    let mut lens = Lens::default();
    // Track what the file set, so defaults are only replaced when asked.
    let mut set_stopwords = false;
    let mut set_prose_kinds = false;

    let mut table = String::new();
    let mut buf = String::new();
    let mut buf_line = 0usize;

    for (no, raw_line) in text.lines().enumerate() {
        let line = no + 1;

        // Continue a multi-line array before anything else — a stopword list
        // gets long, and wrapping it should not be an error.
        if !buf.is_empty() {
            buf.push(' ');
            buf.push_str(strip_comment(raw_line).trim());
            if !buf.contains(']') {
                continue;
            }
            let (key, value) = buf.split_once('=').unwrap();
            let (key, value) = (key.trim().to_string(), value.trim().to_string());
            buf.clear();
            apply(
                &mut lens,
                &table,
                &key,
                parse_value(&value, buf_line)?,
                buf_line,
                &mut set_stopwords,
                &mut set_prose_kinds,
            )?;
            continue;
        }

        let s = strip_comment(raw_line).trim();
        if s.is_empty() {
            continue;
        }

        if s.starts_with("[[") {
            return Err(err(line, LensErrorKind::Unsupported { what: "an array of tables" }));
        }
        if let Some(name) = s.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
            let name = name.trim().to_string();
            match name.as_str() {
                "lens" | "include" | "terms" | "terms.alias" | "terms.weight" | "edges" => {
                    table = name;
                }
                _ => return Err(err(line, LensErrorKind::UnknownTable { got: name })),
            }
            continue;
        }

        let Some((key, value)) = s.split_once('=') else {
            return Err(err(
                line,
                LensErrorKind::Syntax(format!("expected `key = value`, found `{s}`")),
            ));
        };
        let key = key.trim();
        let value = value.trim();

        // An array that has not closed yet continues on the next line.
        if value.starts_with('[') && !value.contains(']') {
            buf = format!("{key} = {value}");
            buf_line = line;
            continue;
        }

        let key = if key.starts_with('"') {
            parse_string(key, line)?
        } else {
            key.to_string()
        };
        apply(
            &mut lens,
            &table,
            &key,
            parse_value(value, line)?,
            line,
            &mut set_stopwords,
            &mut set_prose_kinds,
        )?;
    }

    if !buf.is_empty() {
        return Err(err(buf_line, LensErrorKind::Syntax("unterminated array".into())));
    }

    // Semantic checks here rather than at build time, so a bad number is
    // reported against its line instead of surfacing from inside the graph.
    if !(lens.floor > 0.0) {
        return Err(err(
            0,
            LensErrorKind::Semantic(format!(
                "floor must be greater than zero, got {} — every contact carries at \
                 least the floor, so a floor of zero is not a graph",
                lens.floor
            )),
        ));
    }
    if lens.terms.min_len < 1 {
        return Err(err(
            0,
            LensErrorKind::Semantic("min_len must be at least 1".into()),
        ));
    }
    Ok(lens)
}

#[allow(clippy::too_many_arguments)]
fn apply(
    lens: &mut Lens,
    table: &str,
    key: &str,
    val: Val,
    line: usize,
    set_stopwords: &mut bool,
    set_prose_kinds: &mut bool,
) -> Result<(), LensError> {
    let bad = |expected: &'static str, v: &Val| {
        err(
            line,
            LensErrorKind::BadValue {
                key: key.to_string(),
                expected,
                got: v.type_name().to_string(),
            },
        )
    };
    let unknown = |known: &[&str]| {
        err(
            line,
            LensErrorKind::UnknownKey {
                table: table.to_string(),
                got: key.to_string(),
                suggestion: suggest(key, known),
            },
        )
    };

    match table {
        "" => Err(err(
            line,
            LensErrorKind::Syntax(format!(
                "`{key}` appears before any table — begin with [lens], [include], \
                 [terms] or [edges]"
            )),
        )),
        "lens" => match key {
            "granularity" => match &val {
                Val::Str(s) => {
                    lens.granularity = Granularity::parse(s).map_err(|e| {
                        err(line, LensErrorKind::Semantic(e.to_string()))
                    })?;
                    Ok(())
                }
                v => Err(bad("\"file\" or \"dir\"", v)),
            },
            "floor" => match val {
                Val::Num(n) => {
                    lens.floor = n;
                    Ok(())
                }
                v => Err(bad("a number", &v)),
            },
            _ => Err(unknown(KEYS_LENS)),
        },
        "include" => match key {
            "paths" => match val {
                Val::List(items) => {
                    let mut out = Vec::new();
                    for p in items {
                        out.push(PathPattern::new(&p).map_err(|mut e| {
                            e.line = line;
                            e
                        })?);
                    }
                    lens.include.paths = out;
                    Ok(())
                }
                v => Err(bad("a list of path patterns", &v)),
            },
            "kinds" => match val {
                Val::List(items) => {
                    for k in &items {
                        if !KNOWN_KINDS.contains(&k.as_str()) {
                            eprintln!(
                                "warning: {LENS_FILE}:{line}: kind '{k}' is not one the \
                                 indexer emits — known kinds are {}",
                                KNOWN_KINDS.join(", ")
                            );
                        }
                    }
                    lens.include.kinds = Some(items.into_iter().collect());
                    Ok(())
                }
                v => Err(bad("a list of kinds", &v)),
            },
            _ => Err(unknown(KEYS_INCLUDE)),
        },
        "terms" => match key {
            "stopwords" => match val {
                Val::List(items) => {
                    if !*set_stopwords {
                        lens.terms.stopwords.clear();
                        *set_stopwords = true;
                    }
                    lens.terms.stopwords.extend(items);
                    Ok(())
                }
                v => Err(bad("a list of words", &v)),
            },
            "prose_kinds" => match val {
                Val::List(items) => {
                    if !*set_prose_kinds {
                        lens.terms.prose_kinds.clear();
                        *set_prose_kinds = true;
                    }
                    lens.terms.prose_kinds.extend(items);
                    Ok(())
                }
                v => Err(bad("a list of kinds", &v)),
            },
            "min_len" => match val {
                Val::Num(n) if n >= 0.0 => {
                    lens.terms.min_len = n as usize;
                    Ok(())
                }
                v => Err(bad("a whole number", &v)),
            },
            "split_camel_case" => match val {
                Val::Bool(b) => {
                    lens.terms.split_camel_case = b;
                    Ok(())
                }
                v => Err(bad("true or false", &v)),
            },
            _ => Err(unknown(KEYS_TERMS)),
        },
        "terms.alias" => match val {
            Val::Str(s) => {
                lens.terms.alias.insert(key.to_string(), s);
                Ok(())
            }
            v => Err(bad("a string to rename the term to", &v)),
        },
        "terms.weight" => match val {
            Val::Num(n) if n > 0.0 => {
                lens.terms.weight.push((key.to_string(), n));
                Ok(())
            }
            Val::Num(n) => Err(err(
                line,
                LensErrorKind::Semantic(format!(
                    "weight for '{key}' must be greater than zero, got {n}"
                )),
            )),
            v => Err(bad("a positive number", &v)),
        },
        "edges" => match key {
            "weight" => match &val {
                Val::Str(s) => match EdgeWeight::parse(s) {
                    Some(e) => {
                        lens.edges = e;
                        Ok(())
                    }
                    None => Err(err(
                        line,
                        LensErrorKind::Semantic(format!(
                            "unknown edge weight '{s}' — expected 'sum', 'count' or \
                             'jaccard'"
                        )),
                    )),
                },
                v => Err(bad("\"sum\", \"count\" or \"jaccard\"", v)),
            },
            _ => Err(unknown(KEYS_EDGES)),
        },
        _ => unreachable!("table names are validated when the header is read"),
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Read the lens for a repository.
///
/// An explicit path must exist; the conventional `.purpose/lens.toml` is used
/// when present and the built-in defaults otherwise. Those defaults reproduce
/// the term map this tool had before lenses existed, with one deliberate
/// exception: `section` now counts as prose.
pub fn load_lens(root: &Path, explicit: Option<&Path>) -> Result<Lens, purpose_core::Error> {
    let (path, required) = match explicit {
        Some(p) => (p.to_path_buf(), true),
        None => (root.join(LENS_FILE), false),
    };
    if !path.exists() {
        if required {
            return Err(purpose_core::Error::Provider(format!(
                "no lens at {}",
                path.display()
            )));
        }
        return Ok(Lens::default());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| {
        purpose_core::Error::Provider(format!("cannot read {}: {e}", path.display()))
    })?;
    let mut lens = parse_lens(&text).map_err(|mut e| {
        e.path = Some(path.clone());
        purpose_core::Error::from(e)
    })?;
    lens.source = Some(
        path.strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/"),
    );
    Ok(lens)
}

/// A commented lens holding the built-in defaults, for `ckg lens --init`.
///
/// An AI arriving in a fresh repository should edit a working file rather than
/// author one blind against a schema it cannot see.
pub fn default_lens_toml() -> String {
    let stopwords = DEFAULT_STOPWORDS
        .chunks(8)
        .map(|c| {
            format!(
                "  {},",
                c.iter().map(|w| format!("\"{w}\"")).collect::<Vec<_>>().join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"# The term map is your instrument.
#
# The calculus is correct for ANY choice made here (thm:tau-agnostic), so
# nothing below can make a determination unsound. What it decides is which
# distinctions the graph can see — and therefore whether the graph can tell
# your modules apart at all.
#
# Edit, then run `purpose ckg lens` to see what the choice did to the
# structure, and `purpose ckg build` to induce the graph from it.
#
# There is deliberately no score to raise. A lens under which every module
# draws identical distinctions induces among the highest floors while
# discriminating worst. Read component sizes and term spread instead.

[lens]
granularity = "file"   # file | dir — one module per source file, or per directory
floor = {floor}            # β; every contact carries at least this

[include]
# Which files contribute. Empty means all of them.
# `*` stays within one path segment, `**` crosses them.
paths = []
# Which symbol kinds contribute. Comment out for all of them.
# Known kinds: {kinds}
# kinds = ["fn", "struct", "trait", "heading"]

[terms]
min_len = {min_len}                  # shortest prose token that counts as a distinction
split_camel_case = {split}      # also emit the pieces of parseRequest, not only the whole
prose_kinds = [{prose}]  # names here are phrases to split, not identifiers

# Tokens too common to distinguish anything. These apply to prose only: a
# `fn over` still defines `over`, because that is a real distinction.
# Add to this list from the TERM SPREAD section of `purpose ckg lens` — the
# terms at the top of it are the ones putting everything in contact.
stopwords = [
{stopwords}
]

# Rename a term wherever it appears, so two spellings become one distinction.
[terms.alias]
# "ckg" = "contact-knowledge-graph"

# What a distinction is worth when telling two modules apart. Patterns match on
# the KIND that contributed the term, first match wins, default 1.0.
# A shared term is worth min(what each module gives it).
[terms.weight]
# "trait:*" = 3.0        # anything a trait declares weighs triple
# "heading:*" = 0.5      # prose weighs half
# "fn:parse" = 2.0       # one exact term from one kind

[edges]
# sum     — Σ min(weights) over shared terms; at unit weights, the shared count
# count   — the shared count, ignoring weights entirely
# jaccard — weighted Jaccard in (0,1]; needs a floor well below 1 or every
#           contact clamps to β and the graph goes flat
weight = "sum"
"#,
        floor = FLOOR,
        kinds = KNOWN_KINDS.join(", "),
        min_len = 3,
        split = false,
        prose = ["heading", "section"]
            .iter()
            .map(|k| format!("\"{k}\""))
            .collect::<Vec<_>>()
            .join(", "),
        stopwords = stopwords,
    )
}

// ---------------------------------------------------------------------------
// The stored form: what produced a graph
// ---------------------------------------------------------------------------

/// A lens as recorded alongside the graph it induced.
///
/// Stored in full rather than as a hash alone, so a stale graph can say *which*
/// setting moved rather than merely that something did.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StoredLens {
    pub granularity: Granularity,
    pub floor: f64,
    pub paths: Vec<String>,
    pub kinds: Option<BTreeSet<String>>,
    pub stopwords: BTreeSet<String>,
    pub min_len: usize,
    pub split_camel_case: bool,
    pub prose_kinds: BTreeSet<String>,
    pub alias: BTreeMap<String, String>,
    pub weight: Vec<(String, f64)>,
    pub edges: EdgeWeight,
}

impl StoredLens {
    pub fn of(lens: &Lens) -> Self {
        StoredLens {
            granularity: lens.granularity,
            floor: lens.floor,
            paths: lens.include.paths.iter().map(|p| p.source.clone()).collect(),
            kinds: lens.include.kinds.clone(),
            stopwords: lens.terms.stopwords.clone(),
            min_len: lens.terms.min_len,
            split_camel_case: lens.terms.split_camel_case,
            prose_kinds: lens.terms.prose_kinds.clone(),
            alias: lens.terms.alias.clone(),
            weight: lens.terms.weight.clone(),
            edges: lens.edges,
        }
    }

    /// Rebuild a usable lens from the stored form.
    ///
    /// Determinations must run against the lens the graph was built with, not
    /// whatever `lens.toml` says today — otherwise seeds and cuts disagree and
    /// the determination is incoherent.
    pub fn to_lens(&self) -> Result<Lens, purpose_core::Error> {
        let mut paths = Vec::new();
        for p in &self.paths {
            paths.push(PathPattern::new(p).map_err(purpose_core::Error::from)?);
        }
        Ok(Lens {
            granularity: self.granularity,
            floor: self.floor,
            include: Include { paths, kinds: self.kinds.clone() },
            terms: Terms {
                stopwords: self.stopwords.clone(),
                min_len: self.min_len,
                split_camel_case: self.split_camel_case,
                prose_kinds: self.prose_kinds.clone(),
                alias: self.alias.clone(),
                weight: self.weight.clone(),
            },
            edges: self.edges,
            source: None,
        })
    }

    /// A stable digest of the *resolved* lens.
    ///
    /// Over the canonical serialisation rather than the file's bytes, so
    /// reformatting or reordering a stopword list does not invalidate a graph
    /// while changing a value does.
    ///
    /// FNV-1a rather than `DefaultHasher`, whose output is explicitly not stable
    /// across Rust releases — every toolchain bump would otherwise look like a
    /// stale graph.
    pub fn digest(&self) -> String {
        let canonical = serde_json::to_string(self).unwrap_or_default();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        // Fold in the schema version, so a binary whose defaults have moved
        // invalidates cleanly rather than silently reinterpreting a graph.
        for b in crate::CKG_VERSION.to_le_bytes().iter().chain(canonical.as_bytes()) {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{h:016x}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, kind: &str, file: &str) -> SymbolEntry {
        SymbolEntry {
            name: name.into(),
            kind: kind.into(),
            file: file.into(),
            line: 1,
            snippet: String::new(),
        }
    }

    fn lens_from(src: &str) -> Lens {
        parse_lens(src).expect("fixture lens must parse")
    }

    // -- Parsing ------------------------------------------------------------

    #[test]
    fn a_misspelled_key_is_refused_with_the_key_it_resembles() {
        // An unknown key must never be ignored: a silently dropped typo means
        // the author believes they tuned something and did not. The suggestion
        // is what makes the strictness affordable.
        let e = parse_lens("[terms]\nstopword = [\"new\"]\n").unwrap_err();
        match e.kind {
            LensErrorKind::UnknownKey { ref table, ref got, ref suggestion } => {
                assert_eq!(table, "terms");
                assert_eq!(got, "stopword");
                assert_eq!(suggestion.as_deref(), Some("stopwords"));
            }
            other => panic!("expected UnknownKey, got {other:?}"),
        }
    }

    #[test]
    fn a_quoted_key_carries_a_colon_and_a_star() {
        // `"trait:*"` is the schema's flagship weight pattern and is not a bare
        // TOML key, so quoted keys are load-bearing rather than a nicety.
        let l = lens_from("[terms.weight]\n\"trait:*\" = 3.0\n\"heading:*\" = 0.5\n");
        assert_eq!(
            l.terms.weight,
            vec![("trait:*".to_string(), 3.0), ("heading:*".to_string(), 0.5)],
            "order is preserved, because first match wins"
        );
    }

    #[test]
    fn a_stopword_list_may_span_several_lines() {
        // Stopword lists get long, and a list that must fit on one line is a
        // list nobody edits.
        let l = lens_from(
            "[terms]\nstopwords = [\n  \"new\",\n  \"invoke\",\n\n  \"operations\",\n]\n",
        );
        assert_eq!(
            l.terms.stopwords,
            ["new", "invoke", "operations"].iter().map(|s| s.to_string()).collect()
        );
    }

    #[test]
    fn a_string_containing_a_hash_keeps_it() {
        // The comment-stripping trap. Stripping from the first `#` regardless of
        // quoting would silently truncate a legitimate value.
        let l = lens_from("[terms.alias]\n\"c#\" = \"csharp\"  # a real comment\n");
        assert_eq!(l.terms.alias.get("c#").map(String::as_str), Some("csharp"));
    }

    #[test]
    fn a_floor_of_zero_is_refused_before_the_graph_is_built() {
        // Otherwise `GraphError::NonPositiveWeight` fires deep inside
        // `induced_graph`, far from the line that caused it.
        let e = parse_lens("[lens]\nfloor = 0.0\n").unwrap_err();
        assert!(
            matches!(e.kind, LensErrorKind::Semantic(_) | LensErrorKind::BadValue { .. }),
            "a non-positive floor must be refused at parse time, got {:?}",
            e.kind
        );
    }

    #[test]
    fn an_inline_table_is_refused_by_name_not_by_syntax_error() {
        // Valid TOML this grammar does not accept must say so, rather than
        // reporting a confusing syntax error about a character it met.
        let e = parse_lens("[terms]\nalias = { ckg = \"graph\" }\n").unwrap_err();
        assert!(
            matches!(e.kind, LensErrorKind::Unsupported { .. }),
            "expected Unsupported, got {:?}",
            e.kind
        );
    }

    // -- Terms --------------------------------------------------------------

    #[test]
    fn a_double_star_crosses_directories_and_a_single_star_does_not() {
        let deep = PathPattern::new("crates/**/*.rs").unwrap();
        assert!(deep.matches("crates/purpose-ckg/src/lib.rs"));
        assert!(deep.matches("crates/a.rs"), "zero intervening segments");
        assert!(!deep.matches("docs/x.rs"));

        let flat = PathPattern::new("crates/*.rs").unwrap();
        assert!(flat.matches("crates/a.rs"));
        assert!(!flat.matches("crates/x/a.rs"), "`*` stays within a segment");
    }

    #[test]
    fn an_alias_is_applied_before_the_kinded_term_is_formed() {
        // An alias is a rename, not an addition: if it applied after, a module
        // would carry `fn:ckg` and the rename would only half-take.
        let l = lens_from("[terms.alias]\n\"ckg\" = \"contact-graph\"\n");
        let t = l.terms_of(&entry("ckg", "fn", "a.rs"));
        let keys: BTreeSet<&str> = t.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["contact-graph", "fn:contact-graph"].into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn a_term_contributed_by_two_kinds_takes_the_greater_weight() {
        // Merged by max, never by sum. A sum would make the weight depend on how
        // many times the indexer's line scan happened to fire — an artefact of
        // extraction, not a fact about the module.
        let l = lens_from("[terms.weight]\n\"trait:*\" = 3.0\n\"fn:*\" = 1.0\n");
        let idx = Index {
            root: ".".into(),
            symbols: vec![entry("resolve", "fn", "a.rs"), entry("resolve", "trait", "a.rs")],
        };
        let tau = l.term_map(&idx);
        assert_eq!(tau["a.rs"]["resolve"], 3.0, "the greater of 1.0 and 3.0");
    }

    #[test]
    fn a_latex_section_draws_its_content_words() {
        // The `section` fix. Formerly only `heading` counted as prose, so a
        // LaTeX section title was taken whole as an identifier and every .tex
        // module came out contact-poor.
        let l = Lens::default();
        let t = l.terms_of(&entry("The Directional Pair", "section", "paper.tex"));
        let keys: BTreeSet<&str> = t.keys().map(String::as_str).collect();
        assert_eq!(keys, ["directional", "pair"].into_iter().collect::<BTreeSet<_>>());
    }

    // -- Edges --------------------------------------------------------------

    fn tmap(pairs: &[(&str, &[(&str, f64)])]) -> WeightedTermMap {
        pairs
            .iter()
            .map(|(m, ts)| {
                (m.to_string(), ts.iter().map(|(t, w)| (t.to_string(), *w)).collect())
            })
            .collect()
    }

    #[test]
    fn the_sum_of_unit_weights_is_the_shared_count() {
        // Pins backward compatibility: at unit weights `Σ min` is exactly the
        // pre-lens `f(k) = k as f64`.
        let l = Lens::default();
        let f = l.edge_weight();
        let tau = tmap(&[
            ("a", &[("x", 1.0), ("y", 1.0), ("z", 1.0)]),
            ("b", &[("x", 1.0), ("y", 1.0)]),
            ("c", &[("q", 1.0)]),
        ]);
        assert_eq!(f(&tau["a"], &tau["b"]), Some(2.0));
        assert_eq!(f(&tau["a"], &tau["c"]), None, "no shared term, no contact");
    }

    #[test]
    fn a_shared_term_is_worth_what_the_weaker_module_gives_it() {
        // `min`, not `max`. The edge weight is the cost of telling two modules
        // apart *on account of that term*, and one module's emphasis must not
        // inflate a contact the other barely makes.
        let l = Lens::default();
        let f = l.edge_weight();
        let tau = tmap(&[("a", &[("t", 3.0)]), ("b", &[("t", 0.5)])]);
        assert_eq!(f(&tau["a"], &tau["b"]), Some(0.5));
    }

    #[test]
    fn count_ignores_the_term_weights_entirely() {
        // The explicit way to say that `[terms.weight]` should move the
        // diagnostics and nothing else while tuning.
        let l = lens_from("[edges]\nweight = \"count\"\n");
        let f = l.edge_weight();
        let tau = tmap(&[("a", &[("t", 9.0), ("u", 9.0)]), ("b", &[("t", 0.1), ("u", 0.1)])]);
        assert_eq!(f(&tau["a"], &tau["b"]), Some(2.0));
    }

    #[test]
    fn every_weight_function_respects_the_floor() {
        // thm:tau-agnostic's premise, checked for all three modes including
        // jaccard, whose raw output lands in (0,1] and would otherwise sink
        // beneath β.
        let tau = tmap(&[
            ("a", &[("x", 0.25), ("y", 0.25)]),
            ("b", &[("x", 0.25)]),
            ("c", &[("z", 5.0)]),
        ]);
        for mode in ["sum", "count", "jaccard"] {
            for beta in [0.5, 1.0, 2.0] {
                let l = lens_from(&format!(
                    "[lens]\nfloor = {beta}\n[edges]\nweight = \"{mode}\"\n"
                ));
                let g =
                    purpose_ckg::induced_graph_weighted(&tau, l.floor, l.edge_weight()).unwrap();
                for (_, _, w) in g.edges() {
                    assert!(w >= beta, "{mode} at β={beta} produced a contact of {w}");
                }
            }
        }
    }

    // -- Provenance ---------------------------------------------------------

    #[test]
    fn reformatting_the_lens_file_does_not_change_its_digest() {
        // The digest is over the *resolved* lens, not the file's bytes, so
        // reordering a stopword list or adding a comment must not invalidate a
        // built graph.
        let a = lens_from("[terms]\nstopwords = [\"new\", \"invoke\"]\nmin_len = 4\n");
        let b = lens_from(
            "# a comment\n[terms]\n\nmin_len = 4\nstopwords = [\n  \"invoke\",\n  \"new\",\n]\n",
        );
        assert_eq!(StoredLens::of(&a).digest(), StoredLens::of(&b).digest());
    }

    #[test]
    fn changing_one_setting_changes_the_digest() {
        let a = lens_from("[terms]\nmin_len = 3\n");
        let b = lens_from("[terms]\nmin_len = 4\n");
        assert_ne!(StoredLens::of(&a).digest(), StoredLens::of(&b).digest());
    }

    #[test]
    fn the_stored_lens_round_trips() {
        // `determine` rebuilds τ from this form rather than from disk, so a
        // lossy round trip would silently change what a stored graph means.
        let l = lens_from(
            "[lens]\ngranularity = \"dir\"\nfloor = 0.5\n\
             [include]\npaths = [\"crates/**/*.rs\"]\nkinds = [\"fn\", \"trait\"]\n\
             [terms]\nmin_len = 4\nsplit_camel_case = true\n\
             [terms.alias]\n\"ckg\" = \"graph\"\n\
             [terms.weight]\n\"trait:*\" = 3.0\n\
             [edges]\nweight = \"count\"\n",
        );
        let back = StoredLens::of(&l).to_lens().unwrap();
        assert_eq!(StoredLens::of(&back).digest(), StoredLens::of(&l).digest());
    }
}
