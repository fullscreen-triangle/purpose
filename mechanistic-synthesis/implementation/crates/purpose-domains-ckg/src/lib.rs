//! Binds the directional-pair calculus to the modules of a repository.
//!
//! The items of the contact graph are the repository's own modules — a source
//! file, or a directory under coarser granularity. What a module *is*, for this
//! purpose, is the set of distinctions it draws: the symbols it defines and the
//! headings it carries. Two modules are in contact when they draw a distinction
//! in common.
//!
//! The term map is derived from `.purpose/index.json` — offline, deterministic,
//! no network and no key. That the map is crude is safe rather than hopeful:
//! the calculus is correct for *any* term map, so a coarse extraction coarsens
//! the cells without corrupting them. A finer map raises the floor; it does not
//! change what the floor means.
//!
//! Two commitments run through the whole file and are the reason this is not a
//! scored index:
//!
//! - **No cached answers.** `.purpose/ckg.json` stores the graph and the
//!   record. It never stores a determination. Every determination recomputes
//!   its cuts against the graph as it currently stands.
//! - **No success or failure.** A determination is *accountable*, *contested*,
//!   or *declined*. A decline that names the contested classes carries real
//!   information; flattening it to "no result" throws that information away.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use purpose_ckg::{
    align_score, character, dominated_by, floor_witness, induced_graph, necessary, reach,
    resting_cut, sigma_medium, system_floor, ContactGraph, EdgeKey, Record, TermMap, FLOOR,
    MEDIUM,
};
use purpose_core::{Domain, Error, Operation, Resolver, Type, VaHera, Value};
use purpose_domains_codebase::{index_path, Index, SymbolEntry};
use purpose_operations::{OperationRegistry, Provider};

/// Schema version of `.purpose/ckg.json`. The symbol index carries no version
/// field; that omission is not repeated here.
pub const CKG_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Granularity
// ---------------------------------------------------------------------------

/// What counts as one module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Granularity {
    /// One module per source file. The default.
    File,
    /// One module per directory — coarser cells, fewer items, higher floor.
    Dir,
}

impl Granularity {
    pub fn parse(s: &str) -> Result<Self, Error> {
        match s.to_lowercase().as_str() {
            "file" => Ok(Granularity::File),
            "dir" | "directory" => Ok(Granularity::Dir),
            other => Err(Error::Provider(format!(
                "unknown granularity '{other}' — expected 'file' or 'dir'"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Granularity::File => "file",
            Granularity::Dir => "dir",
        }
    }

    /// The module an index entry belongs to.
    fn module_of(&self, file: &str) -> String {
        match self {
            Granularity::File => file.to_string(),
            Granularity::Dir => match file.rfind('/') {
                Some(i) => file[..i].to_string(),
                None => ".".to_string(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Term map: what distinctions a module draws
// ---------------------------------------------------------------------------

/// Distinctions drawn by one index entry.
///
/// The bare name is the distinction itself; `kind:name` is the same
/// distinction drawn *as* a kind, so a `struct Resolver` and a `trait Resolver`
/// share one term and differ on another. Prose contributes heading tokens,
/// because a heading is where a document draws its distinctions.
///
/// Function words, which draw no distinction.
///
/// A heading token like `the` or `with` appears across most documents in most
/// repositories, so admitting it puts every prose module in contact with every
/// other. That is the degenerate map: it raises the floor while discriminating
/// worst. Excluding these does not make the map *correct* — no extraction is —
/// it stops one known artefact from dominating the contact structure.
fn is_function_word(t: &str) -> bool {
    matches!(
        t,
        "the" | "and" | "not" | "for" | "with" | "from" | "that" | "this"
            | "what" | "when" | "where" | "which" | "how" | "why" | "who"
            | "are" | "was" | "were" | "has" | "have" | "had" | "can" | "will"
            | "its" | "into" | "out" | "over" | "than" | "then" | "there"
            | "does" | "did" | "you" | "your" | "our" | "all" | "any" | "but"
            | "use" | "using" | "used" | "via" | "per" | "about" | "also"
    )
}

/// Distinctions drawn by one index entry.
fn terms_of(entry: &SymbolEntry) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let name = entry.name.trim().to_lowercase();
    if name.is_empty() {
        return out;
    }
    let kind = entry.kind.trim().to_lowercase();

    if kind == "heading" {
        // A heading is a phrase, not an identifier: split it and keep the
        // content words. Single characters and digits distinguish nothing.
        for tok in name.split(|c: char| !c.is_alphanumeric() && c != '_') {
            if tok.len() > 2
                && !tok.chars().all(|c| c.is_ascii_digit())
                && !is_function_word(tok)
            {
                out.insert(tok.to_string());
            }
        }
        return out;
    }

    out.insert(name.clone());
    if !kind.is_empty() {
        out.insert(format!("{kind}:{name}"));
    }
    out
}

/// Build the term map over modules at the requested granularity.
///
/// A module with no indexed symbols does not appear. It draws no distinctions
/// this index can see, so it has no contacts, and an item joined to nothing but
/// the medium would sit at the floor and drag `β*` down for a reason that is an
/// artefact of extraction rather than a fact about the repository.
pub fn term_map(index: &Index, granularity: Granularity) -> TermMap {
    let mut tau: TermMap = BTreeMap::new();
    for entry in &index.symbols {
        let terms = terms_of(entry);
        if terms.is_empty() {
            continue;
        }
        tau.entry(granularity.module_of(&entry.file))
            .or_default()
            .extend(terms);
    }
    tau.retain(|_, ts| !ts.is_empty());
    tau
}

// ---------------------------------------------------------------------------
// Persistence: the graph and the record, never an answer
// ---------------------------------------------------------------------------

/// One contact, as stored.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredEdge {
    pub u: String,
    pub v: String,
    pub w: f64,
}

/// The stored CKG.
///
/// Note what is absent: there is no determination here, and no field one could
/// be put in. Storing a verdict would make the store answerable from cache, and
/// a cached verdict goes stale the moment the graph moves beneath it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredCkg {
    pub version: u32,
    pub root: String,
    pub granularity: Granularity,
    pub floor: f64,
    pub items: Vec<String>,
    pub edges: Vec<StoredEdge>,
    /// Monotone commitment count.
    pub record: u64,
    /// Edges standing committed, as `[u, v]` pairs.
    #[serde(default)]
    pub committed: Vec<[String; 2]>,
}

impl StoredCkg {
    /// Rebuild the contact graph.
    pub fn graph(&self) -> Result<ContactGraph, Error> {
        let mut g = ContactGraph::new();
        for i in &self.items {
            g.add_vertex(i);
        }
        g.add_vertex(MEDIUM);
        for e in &self.edges {
            g.add_edge(&e.u, &e.v, e.w)
                .map_err(|err| Error::Provider(format!("corrupt ckg: {err}")))?;
        }
        Ok(g)
    }

    /// Rebuild the record, resuming its count rather than restarting it.
    pub fn record(&self) -> Record {
        let committed: BTreeSet<EdgeKey> = self
            .committed
            .iter()
            .map(|[u, v]| EdgeKey::new(u, v))
            .collect();
        Record::resume(self.record, committed)
    }

    fn from_graph(
        root: &Path,
        granularity: Granularity,
        floor: f64,
        g: &ContactGraph,
        record: &Record,
    ) -> Self {
        StoredCkg {
            version: CKG_VERSION,
            root: root.display().to_string(),
            granularity,
            floor,
            items: g.items().into_iter().map(|s| s.to_string()).collect(),
            edges: g
                .edges()
                .map(|(u, v, w)| StoredEdge {
                    u: u.to_string(),
                    v: v.to_string(),
                    w,
                })
                .collect(),
            record: record.count(),
            committed: record
                .committed()
                .iter()
                .map(|e| [e.left().to_string(), e.right().to_string()])
                .collect(),
        }
    }
}

/// Where the CKG lives — a sibling of the symbol index.
pub fn ckg_path(root: &Path) -> PathBuf {
    root.join(".purpose").join("ckg.json")
}

fn load_index(root: &Path) -> Result<Index, Error> {
    let path = index_path(root);
    let text = std::fs::read_to_string(&path).map_err(|_| {
        Error::Provider(format!(
            "no index at {} — run `purpose index` in this project first",
            path.display()
        ))
    })?;
    serde_json::from_str(&text).map_err(|e| Error::Provider(format!("corrupt index: {e}")))
}

/// Read the stored CKG for a repository.
pub fn load(root: &Path) -> Result<StoredCkg, Error> {
    load_ckg(root)
}

fn load_ckg(root: &Path) -> Result<StoredCkg, Error> {
    let path = ckg_path(root);
    let text = std::fs::read_to_string(&path).map_err(|_| {
        Error::Provider(format!(
            "no ckg at {} — run `purpose ckg build` in this project first",
            path.display()
        ))
    })?;
    let stored: StoredCkg =
        serde_json::from_str(&text).map_err(|e| Error::Provider(format!("corrupt ckg: {e}")))?;
    if stored.version != CKG_VERSION {
        return Err(Error::Provider(format!(
            "ckg at {} is version {} — this build expects {CKG_VERSION}; rebuild with `purpose ckg build`",
            path.display(),
            stored.version
        )));
    }
    Ok(stored)
}

fn save_ckg(root: &Path, stored: &StoredCkg) -> Result<(), Error> {
    let path = ckg_path(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| Error::Provider(format!("cannot create {}: {e}", parent.display())))?;
    }
    let text = serde_json::to_string_pretty(stored)
        .map_err(|e| Error::Provider(format!("cannot serialise ckg: {e}")))?;
    std::fs::write(&path, text)
        .map_err(|e| Error::Provider(format!("cannot write {}: {e}", path.display())))
}

/// Build the CKG for a repository and write it, carrying the record forward.
///
/// Rebuilding does not reset the record. The record counts what has been
/// committed, and rebuilding the graph commits nothing away.
pub fn build(root: &Path, granularity: Granularity, floor: f64) -> Result<StoredCkg, Error> {
    let index = load_index(root)?;
    let tau = term_map(&index, granularity);
    if tau.is_empty() {
        return Err(Error::Provider(
            "the index yielded no modules with distinctions — is it empty?".into(),
        ));
    }
    let g = induced_graph(&tau, floor)
        .map_err(|e| Error::Provider(format!("cannot induce contact graph: {e}")))?;

    // Carry the existing record forward if there is one.
    let record = match load_ckg(root) {
        Ok(prev) => prev.record(),
        Err(_) => Record::new(),
    };

    let stored = StoredCkg::from_graph(root, granularity, floor, &g, &record);
    save_ckg(root, &stored)?;
    Ok(stored)
}

// ---------------------------------------------------------------------------
// Algorithm 2: the determination
// ---------------------------------------------------------------------------

/// How a determination came out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// The alignment sits within the system floor plus tolerance.
    Accountable,
    /// It does not. The determination is declined and the classes reported.
    Contested,
    /// Nothing met the goal; there is nothing to determine.
    Declined,
}

impl Verdict {
    pub fn as_str(&self) -> &'static str {
        match self {
            Verdict::Accountable => "accountable",
            Verdict::Contested => "contested",
            Verdict::Declined => "declined",
        }
    }
}

/// One module in a determination.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModuleReport {
    pub module: String,
    /// σ(v) — the resting cut weight separating this module from the medium.
    pub sigma: f64,
    /// How many reachable modules this one dominates.
    pub dominates: usize,
    /// Whether it met a goal term directly.
    pub seed: bool,
}

/// The result of `determine`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Determination {
    pub goal: Vec<String>,
    pub verdict: Verdict,
    /// σ(v₀, x*) — the alignment actually queried.
    pub alignment: f64,
    /// a = σ/Ω.
    pub align_score: f64,
    /// β* — the system floor.
    pub floor: f64,
    /// The module realising the floor.
    pub floor_witness: Option<String>,
    /// Ω — total contact weight.
    pub omega: f64,
    pub eps: f64,
    /// β* + εΩ — the admissibility threshold.
    pub threshold: f64,
    /// Load-bearing modules: dropping one changes what the goal resolves.
    pub necessary: Vec<ModuleReport>,
    /// Reachable but redundant: droppable without changing what resolves.
    pub redundant: Vec<ModuleReport>,
    /// Under a contested closure, the classes that could not be separated.
    pub contested: Vec<String>,
    pub record_before: u64,
    pub record_after: u64,
}

/// Goal terms, normalised the way the term map is.
fn goal_terms(goal: &str) -> BTreeSet<String> {
    goal.split(|c: char| !c.is_alphanumeric() && c != '_' && c != ':')
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .collect()
}

/// The shortest goal term that may match by containment rather than equality.
///
/// Substring matching in both directions is what lets the goal `resolver` find
/// `codebaseresolver` and vice versa. But a short fragment is contained in far
/// too much: `goal` sits inside `goals`, `subgoal`, and any heading that says
/// the word, so a three-letter stem seeds nearly the whole repository and the
/// determination stops discriminating. Below this length a term must match
/// exactly.
const MIN_SUBSTRING_TERM: usize = 5;

/// Does a goal term meet a distinction the module draws?
fn term_meets(goal: &str, term: &str) -> bool {
    if goal == term {
        return true;
    }
    // A kinded term `kind:name` is met by a goal that names the symbol.
    if let Some((_, name)) = term.split_once(':') {
        if name == goal {
            return true;
        }
    }
    let (short, long) = if goal.len() <= term.len() {
        (goal, term)
    } else {
        (term, goal)
    };
    short.len() >= MIN_SUBSTRING_TERM && long.contains(short)
}

/// Modules whose distinctions meet a goal term.
fn seeds_for(tau: &TermMap, goal: &BTreeSet<String>) -> BTreeSet<String> {
    tau.iter()
        .filter(|(_, terms)| {
            terms
                .iter()
                .any(|t| goal.iter().any(|g| term_meets(g, t)))
        })
        .map(|(m, _)| m.clone())
        .collect()
}

/// Run Algorithm 2 against the stored graph.
///
/// Every cut here is recomputed from the graph as it currently stands. Nothing
/// is read from a previous answer, because there are no previous answers stored
/// to read.
pub fn determine(root: &Path, goal: &str, eps: f64) -> Result<Determination, Error> {
    let stored = load_ckg(root)?;
    let g = stored.graph()?;
    let index = load_index(root)?;
    let tau = term_map(&index, stored.granularity);

    let terms = goal_terms(goal);
    let goal_list: Vec<String> = terms.iter().cloned().collect();
    let mut record = stored.record();
    let record_before = record.count();

    let omega = g.total_weight();
    let beta = system_floor(&g);
    let witness = floor_witness(&g).map(|(m, _)| m);

    let seeds = seeds_for(&tau, &terms);
    if seeds.is_empty() {
        // Nothing met the goal. That is a decline, not a failure: the report
        // says what was searched and what floor it was searched against.
        record.commit("goal", MEDIUM, &format!("declined: {goal}"));
        let record_after = record.count();
        persist_record(root, &stored, &record)?;
        return Ok(Determination {
            goal: goal_list,
            verdict: Verdict::Declined,
            alignment: 0.0,
            align_score: 0.0,
            floor: beta.unwrap_or(stored.floor),
            floor_witness: witness,
            omega,
            eps,
            threshold: beta.unwrap_or(stored.floor) + eps * omega,
            necessary: Vec::new(),
            redundant: Vec::new(),
            contested: Vec::new(),
            record_before,
            record_after,
        });
    }

    // seek, then nec *within what seek retained* — `N ← nec(R, x*)`.
    //
    // The retained set must be `R`, not the whole item universe. Necessity
    // computed over the universe and filtered afterwards asks a different
    // question — "which modules are articulation points of the repository" —
    // whose answer does not depend on the goal at all, so every goal landing
    // in the same component returns the same modules. Ablation has to run
    // against the set the goal actually reached.
    let reachable = reach(&g, &seeds);
    let nec = necessary(&g, &seeds, &reachable);

    // v₀ is the seed of least separation cost — the cheapest way in — and x*
    // the necessary module of greatest separation cost, the one the goal is
    // most expensively bound to. Both are determined by the graph, not chosen.
    let v0 = pick(&seeds, &g, false);
    let xstar = pick(if nec.is_empty() { &reachable } else { &nec }, &g, true);

    let (alignment, ascore) = match (&v0, &xstar) {
        (Some(a), Some(b)) => (purpose_ckg::alignment(&g, a, b), align_score(&g, a, b)),
        _ => (0.0, 0.0),
    };

    let beta_star = beta.unwrap_or(stored.floor);
    let threshold = beta_star + eps * omega;
    let accountable = alignment <= threshold + 1e-12;

    let report = |m: &String, seed: bool| ModuleReport {
        module: m.clone(),
        sigma: sigma_medium(&g, m),
        dominates: dominated_by(&g, &seeds, m, &reachable).len(),
        seed,
    };

    let mut necessary_rep: Vec<ModuleReport> =
        nec.iter().map(|m| report(m, seeds.contains(m))).collect();
    let mut redundant_rep: Vec<ModuleReport> = reachable
        .iter()
        .filter(|m| !nec.contains(*m))
        .map(|m| report(m, seeds.contains(m)))
        .collect();
    // Heaviest separation first: the modules hardest to tell apart from the
    // rest are the ones a reader most needs named.
    let by_sigma = |a: &ModuleReport, b: &ModuleReport| {
        b.sigma
            .partial_cmp(&a.sigma)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.module.cmp(&b.module))
    };
    necessary_rep.sort_by(by_sigma);
    redundant_rep.sort_by(by_sigma);

    // A contested closure: the necessary modules the cut could not separate,
    // reported rather than arbitrated.
    let contested: Vec<String> = if accountable {
        Vec::new()
    } else {
        let cut: BTreeSet<EdgeKey> = match &xstar {
            Some(x) => resting_cut(&g, x),
            None => BTreeSet::new(),
        };
        let mut cs: BTreeSet<String> = BTreeSet::new();
        for e in &cut {
            for v in [e.left(), e.right()] {
                if v != MEDIUM && nec.contains(v) {
                    cs.insert(v.to_string());
                }
            }
        }
        cs.into_iter().collect()
    };

    let verdict = if accountable {
        Verdict::Accountable
    } else {
        Verdict::Contested
    };

    // Deposit the residue. The determination itself is not stored; what is
    // stored is that a determination was made.
    if let (Some(a), Some(b)) = (&v0, &xstar) {
        record.commit(a, b, &format!("{}: {goal}", verdict.as_str()));
    } else {
        record.commit("goal", MEDIUM, &format!("{}: {goal}", verdict.as_str()));
    }
    let record_after = record.count();
    persist_record(root, &stored, &record)?;

    debug!(
        goal = %goal,
        verdict = verdict.as_str(),
        necessary = necessary_rep.len(),
        redundant = redundant_rep.len(),
        "ckg determination"
    );

    Ok(Determination {
        goal: goal_list,
        verdict,
        alignment,
        align_score: ascore,
        floor: beta_star,
        floor_witness: witness,
        omega,
        eps,
        threshold,
        necessary: necessary_rep,
        redundant: redundant_rep,
        contested,
        record_before,
        record_after,
    })
}

/// The member of `set` of greatest (or least) separation cost, ties broken by
/// name so the choice is reproducible.
fn pick(set: &BTreeSet<String>, g: &ContactGraph, greatest: bool) -> Option<String> {
    set.iter()
        .map(|m| (sigma_medium(g, m), m))
        .fold(None, |best: Option<(f64, &String)>, cur| match best {
            None => Some(cur),
            Some(b) => {
                let better = if greatest { cur.0 > b.0 } else { cur.0 < b.0 };
                if better || (cur.0 == b.0 && cur.1 < b.1) {
                    Some(cur)
                } else {
                    Some(b)
                }
            }
        })
        .map(|(_, m)| m.clone())
}

/// Write back the record — and only the record. The graph is untouched.
fn persist_record(root: &Path, stored: &StoredCkg, record: &Record) -> Result<(), Error> {
    let mut next = stored.clone();
    next.record = record.count();
    next.committed = record
        .committed()
        .iter()
        .map(|e| [e.left().to_string(), e.right().to_string()])
        .collect();
    save_ckg(root, &next)
}

// ---------------------------------------------------------------------------
// Why: one module, in its own terms
// ---------------------------------------------------------------------------

/// What a single module costs to separate, and what it carries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Why {
    pub module: String,
    pub sigma: f64,
    pub floor: f64,
    pub omega: f64,
    /// Edges of the resting cut, as `"u — v (w)"`.
    pub cut: Vec<String>,
    pub neighbours: Vec<String>,
    /// Present only when a goal was supplied.
    pub goal: Option<Vec<String>>,
    pub reachable_from_goal: Option<bool>,
    pub necessary_for_goal: Option<bool>,
    pub dominated: Vec<String>,
}

/// Explain one module, optionally relative to a goal.
pub fn why(root: &Path, module: &str, goal: Option<&str>) -> Result<Why, Error> {
    let stored = load_ckg(root)?;
    let g = stored.graph()?;
    if !g.items().iter().any(|i| *i == module) {
        return Err(Error::Provider(format!(
            "no module '{module}' in the ckg — `purpose ckg build` lists the items it holds"
        )));
    }

    let cut: Vec<String> = resting_cut(&g, module)
        .into_iter()
        .map(|e| {
            let w = g.weight(e.left(), e.right()).unwrap_or(0.0);
            format!("{} — {} ({w:.2})", e.left(), e.right())
        })
        .collect();
    let neighbours: Vec<String> = g
        .neighbours(module)
        .into_iter()
        .filter(|n| n != MEDIUM)
        .collect();

    let mut out = Why {
        module: module.to_string(),
        sigma: sigma_medium(&g, module),
        floor: system_floor(&g).unwrap_or(stored.floor),
        omega: g.total_weight(),
        cut,
        neighbours,
        goal: None,
        reachable_from_goal: None,
        necessary_for_goal: None,
        dominated: Vec::new(),
    };

    if let Some(goal) = goal {
        let index = load_index(root)?;
        let tau = term_map(&index, stored.granularity);
        let terms = goal_terms(goal);
        let seeds = seeds_for(&tau, &terms);
        // Necessity and domination are relative to what the goal reached, not
        // to the whole repository — the same `N ← nec(R, x*)` as `determine`.
        let reachable = reach(&g, &seeds);

        out.goal = Some(terms.iter().cloned().collect());
        out.reachable_from_goal = Some(reachable.contains(module));
        out.necessary_for_goal = Some(necessary(&g, &seeds, &reachable).contains(module));
        out.dominated = dominated_by(&g, &seeds, module, &reachable)
            .into_iter()
            .collect();
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Render a determination for a reader.
///
/// The vocabulary here is deliberate: ACCOUNTABLE, CONTESTED, DECLINED. Never
/// "found" or "failed". A contested determination is not a failed one — it is a
/// determination that names what it could not separate.
pub fn render_determination(d: &Determination) -> String {
    let mut out = String::new();
    out.push_str(&format!("Determination for goal: {}\n\n", d.goal.join(" ")));

    match d.verdict {
        Verdict::Accountable => out.push_str(&format!(
            "ACCOUNTABLE  (σ = {:.2} ≤ β* + εΩ = {:.2})\n",
            d.alignment, d.threshold
        )),
        Verdict::Contested => out.push_str(&format!(
            "CONTESTED  (σ = {:.2} > β* + εΩ = {:.2}) — determination declined\n",
            d.alignment, d.threshold
        )),
        Verdict::Declined => {
            out.push_str("DECLINED — no module draws a distinction meeting this goal\n")
        }
    }
    out.push_str(&format!(
        "  β* = {:.2}{}   Ω = {:.2}   a = σ/Ω = {:.4}   ε = {:.4}\n",
        d.floor,
        d.floor_witness
            .as_ref()
            .map(|w| format!(" at {w}"))
            .unwrap_or_default(),
        d.omega,
        d.align_score,
        d.eps,
    ));

    if !d.necessary.is_empty() {
        out.push_str(&format!(
            "\nNECESSARY ({} module(s) — load-bearing, cannot be dropped):\n",
            d.necessary.len()
        ));
        for m in &d.necessary {
            out.push_str(&format!(
                "  {}  σ={:.2}  {}\n",
                m.module,
                m.sigma,
                if m.seed {
                    "seed".to_string()
                } else {
                    format!("dominates {}", m.dominates)
                }
            ));
        }
    }

    if !d.redundant.is_empty() {
        out.push_str(&format!(
            "\nREACHABLE BUT REDUNDANT ({} module(s) — drop without changing what the goal resolves):\n",
            d.redundant.len()
        ));
        for m in &d.redundant {
            out.push_str(&format!("  {}  σ={:.2}\n", m.module, m.sigma));
        }
    }

    if !d.contested.is_empty() {
        out.push_str(&format!(
            "\nCONTESTED CLASSES ({} — the cut does not separate these):\n",
            d.contested.len()
        ));
        for c in &d.contested {
            out.push_str(&format!("  {c}\n"));
        }
    }

    out.push_str(&format!(
        "\nRecord advanced {} → {}.\n",
        d.record_before, d.record_after
    ));
    out
}

/// Render a single-module explanation.
pub fn render_why(w: &Why) -> String {
    let mut out = String::new();
    out.push_str(&format!("{}\n\n", w.module));
    out.push_str(&format!(
        "  σ = {:.2}   β* = {:.2}   Ω = {:.2}\n",
        w.sigma, w.floor, w.omega
    ));

    if let Some(goal) = &w.goal {
        out.push_str(&format!("\n  goal: {}\n", goal.join(" ")));
        out.push_str(&format!(
            "  reachable: {}   necessary: {}\n",
            w.reachable_from_goal.unwrap_or(false),
            w.necessary_for_goal.unwrap_or(false)
        ));
        if !w.dominated.is_empty() {
            out.push_str(&format!(
                "\n  DOMINATES {} module(s) — every route from the goal passes through here:\n",
                w.dominated.len()
            ));
            for d in &w.dominated {
                out.push_str(&format!("    {d}\n"));
            }
        }
    }

    if !w.cut.is_empty() {
        out.push_str(&format!(
            "\n  RESTING CUT ({} edge(s), total {:.2}):\n",
            w.cut.len(),
            w.sigma
        ));
        for e in &w.cut {
            out.push_str(&format!("    {e}\n"));
        }
    }

    if !w.neighbours.is_empty() {
        out.push_str(&format!("\n  IN CONTACT WITH ({}):\n", w.neighbours.len()));
        for n in &w.neighbours {
            out.push_str(&format!("    {n}\n"));
        }
    }
    out
}

/// Render the floor and what realises it.
pub fn render_floor(stored: &StoredCkg, g: &ContactGraph) -> String {
    let mut out = String::new();
    let items = g.items().len();
    out.push_str(&format!(
        "{items} module(s), {} contact(s), granularity {}, β = {:.2}\n",
        g.edge_count(),
        stored.granularity.as_str(),
        stored.floor
    ));
    match floor_witness(g) {
        Some((m, b)) => out.push_str(&format!(
            "\nβ* = {b:.2}, realised at {m}\n  Ω = {:.2}\n",
            g.total_weight()
        )),
        None => out.push_str("\nno items — β* undefined\n"),
    }
    out.push_str(&format!("\nrecord = {}\n", stored.record));
    out.push_str(
        "\nβ* is a monotonicity signal, not a score: it does not fall as the term map is\nrefined. A map in which every module drew identical distinctions would induce\namong the highest floors while discriminating worst.\n",
    );
    out
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Serves the ckg operations against `.purpose/` in a repository root.
pub struct CkgProvider {
    root: PathBuf,
}

impl CkgProvider {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
}

fn to_value<T: Serialize>(v: &T) -> Result<Value, Error> {
    let text = serde_json::to_string(v)
        .map_err(|e| Error::Provider(format!("cannot serialise result: {e}")))?;
    serde_json::from_str::<Value>(&text)
        .map_err(|e| Error::Provider(format!("cannot convert result: {e}")))
}

fn from_value<T: for<'de> Deserialize<'de>>(v: &Value) -> Result<T, Error> {
    let text = serde_json::to_string(v)
        .map_err(|e| Error::Provider(format!("cannot re-serialise input: {e}")))?;
    serde_json::from_str(&text).map_err(|e| Error::Provider(format!("unexpected input: {e}")))
}

#[async_trait]
impl Provider for CkgProvider {
    async fn invoke(&self, op: &str, args: &BTreeMap<String, Value>) -> Result<Value, Error> {
        match op {
            "build_ckg" => {
                let granularity = args
                    .get("granularity")
                    .and_then(|v| v.as_str())
                    .map(Granularity::parse)
                    .transpose()?
                    .unwrap_or(Granularity::File);
                let floor = args
                    .get("floor")
                    .and_then(|v| match v {
                        Value::Num(n) => Some(*n),
                        _ => None,
                    })
                    .unwrap_or(FLOOR);
                let stored = build(&self.root, granularity, floor)?;
                to_value(&stored)
            }
            "module_floor" => {
                let stored = load_ckg(&self.root)?;
                let g = stored.graph()?;
                Ok(Value::Str(render_floor(&stored, &g)))
            }
            "determine" => {
                let goal = args
                    .get("goal")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Provider("missing 'goal'".into()))?;
                let eps = args
                    .get("eps")
                    .and_then(|v| match v {
                        Value::Num(n) => Some(*n),
                        _ => None,
                    })
                    .unwrap_or(0.0);
                let d = determine(&self.root, goal, eps)?;
                to_value(&d)
            }
            "format_determination" => {
                let input = args
                    .get("input")
                    .ok_or_else(|| Error::Provider("missing 'input'".into()))?;
                let d: Determination = from_value(input)?;
                Ok(Value::Str(render_determination(&d)))
            }
            "explain_module" => {
                let module = args
                    .get("module")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Provider("missing 'module'".into()))?;
                let goal = args.get("goal").and_then(|v| v.as_str());
                let w = why(&self.root, module, goal)?;
                Ok(Value::Str(render_why(&w)))
            }
            _ => Err(Error::Provider(format!("unsupported op: {op}"))),
        }
    }
}

// ---------------------------------------------------------------------------
// Resolver
// ---------------------------------------------------------------------------

/// Hand-coded resolver: the whole utterance is the goal; emit
/// `determine |> format_determination`.
///
/// Question words are *not* stripped here, unlike the codebase resolver. Goal
/// terms are matched by substring against the term map, and a stray common word
/// widens the seed set rather than corrupting it — while stripping risks losing
/// a term that names a real module.
pub struct CkgResolver;

#[async_trait]
impl Resolver for CkgResolver {
    async fn compile(&self, utterance: &str) -> Result<VaHera, Error> {
        let goal = utterance.trim();
        if goal.is_empty() {
            return Err(Error::Compile("empty goal".into()));
        }
        debug!(goal = %goal, "ckg resolver compiled a determination");

        let mut args: BTreeMap<String, VaHera> = BTreeMap::new();
        args.insert(
            "goal".to_string(),
            VaHera::Literal(Value::Str(goal.to_string())),
        );

        Ok(VaHera::Compose(vec![
            VaHera::Call {
                op: "determine".to_string(),
                args,
            },
            VaHera::Call {
                op: "format_determination".to_string(),
                args: BTreeMap::new(),
            },
        ]))
    }
}

// ---------------------------------------------------------------------------
// Domain wiring
// ---------------------------------------------------------------------------

/// Build the ckg domain connector.
pub fn domain() -> Domain {
    Domain {
        name: "ckg".into(),
        operations: operations(),
        resolver: Arc::new(CkgResolver),
    }
}

/// Operation vocabulary for the ckg domain.
pub fn operations() -> Vec<Operation> {
    vec![
        Operation::new(
            "build_ckg",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("granularity".into(), Type::Str);
                inputs.insert("floor".into(), Type::Num);
                inputs
            },
            Type::named("Ckg"),
            "Induce the module contact graph from the symbol index and store it.",
        ),
        Operation::new(
            "module_floor",
            BTreeMap::new(),
            Type::Str,
            "Report the system floor β* and the module that realises it.",
        ),
        Operation::new(
            "determine",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("goal".into(), Type::Str);
                inputs.insert("eps".into(), Type::Num);
                inputs
            },
            Type::named("Determination"),
            "Determine which modules are load-bearing for a goal, and whether that is accountable.",
        ),
        Operation::new(
            "format_determination",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("input".into(), Type::named("Determination"));
                inputs
            },
            Type::Str,
            "Render a determination as readable text.",
        ),
        Operation::new(
            "explain_module",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("module".into(), Type::Str);
                inputs.insert("goal".into(), Type::Str);
                inputs
            },
            Type::Str,
            "Report one module's separation cost, resting cut, and what it dominates.",
        ),
    ]
}

/// Register the ckg provider against this domain's operations.
pub fn register_providers(registry: &mut OperationRegistry, root: PathBuf) {
    let provider = Arc::new(CkgProvider::new(root));
    for op in operations() {
        match op.name.as_str() {
            "build_ckg" | "module_floor" | "determine" | "format_determination"
            | "explain_module" => {
                registry.register(op, provider.clone());
            }
            _ => {}
        }
    }
}

/// The σ multiset of the stored graph — the character invariant, exposed so a
/// relabelling check can be run against a real repository.
pub fn module_character(g: &ContactGraph) -> Vec<f64> {
    character(g)
}

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

    fn index(entries: Vec<SymbolEntry>) -> Index {
        Index {
            root: ".".into(),
            symbols: entries,
        }
    }

    #[test]
    fn a_symbol_draws_its_name_and_its_kinded_name() {
        let t = terms_of(&entry("Resolver", "trait", "a.rs"));
        assert!(t.contains("resolver"));
        assert!(t.contains("trait:resolver"));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn same_name_different_kind_shares_one_term_and_differs_on_another() {
        // A struct Resolver and a trait Resolver are in contact — they draw the
        // same distinction — but they are not the same distinction drawn.
        let a = terms_of(&entry("Resolver", "trait", "a.rs"));
        let b = terms_of(&entry("Resolver", "struct", "b.rs"));
        assert_eq!(a.intersection(&b).count(), 1);
        assert_ne!(a, b);
    }

    #[test]
    fn a_heading_contributes_its_content_words() {
        let t = terms_of(&entry("The Directional Pair 2", "heading", "d.md"));
        assert!(t.contains("directional") && t.contains("pair"));
        assert!(!t.contains("2"), "a bare digit distinguishes nothing");
        assert!(!t.contains("the"), "a function word distinguishes nothing");
    }

    #[test]
    fn granularity_dir_collapses_files_in_one_directory() {
        let idx = index(vec![
            entry("alpha", "fn", "src/core/a.rs"),
            entry("beta", "fn", "src/core/b.rs"),
            entry("gamma", "fn", "src/cli/c.rs"),
        ]);
        let by_file = term_map(&idx, Granularity::File);
        let by_dir = term_map(&idx, Granularity::Dir);
        assert_eq!(by_file.len(), 3);
        assert_eq!(by_dir.len(), 2);
        assert!(by_dir["src/core"].contains("alpha") && by_dir["src/core"].contains("beta"));
    }

    #[test]
    fn modules_are_in_contact_when_they_share_a_symbol_name() {
        let idx = index(vec![
            entry("Resolver", "trait", "core.rs"),
            entry("Resolver", "struct", "impl.rs"),
            entry("Unrelated", "fn", "other.rs"),
        ]);
        let tau = term_map(&idx, Granularity::File);
        let g = induced_graph(&tau, FLOOR).unwrap();
        assert_eq!(g.weight("core.rs", "impl.rs"), Some(1.0));
        assert_eq!(g.weight("core.rs", "other.rs"), None);
    }

    #[test]
    fn a_module_with_no_indexed_symbols_does_not_become_an_item() {
        // It would sit alone at the floor and drag β* down for a reason that is
        // about extraction, not about the repository.
        let idx = index(vec![entry("", "fn", "empty.rs"), entry("x", "fn", "a.rs")]);
        let tau = term_map(&idx, Granularity::File);
        assert!(!tau.contains_key("empty.rs"));
        assert!(tau.contains_key("a.rs"));
    }

    #[test]
    fn goal_terms_match_by_substring_in_both_directions() {
        let idx = index(vec![
            entry("CodebaseResolver", "struct", "a.rs"),
            entry("unrelated", "fn", "b.rs"),
        ]);
        let tau = term_map(&idx, Granularity::File);
        let wide = seeds_for(&tau, &goal_terms("resolver"));
        assert!(wide.contains("a.rs") && !wide.contains("b.rs"));
        let narrow = seeds_for(&tau, &goal_terms("codebaseresolver detail"));
        assert!(narrow.contains("a.rs"));
    }

    #[test]
    fn necessity_is_relative_to_what_the_goal_reached() {
        // `N ← nec(R, x*)`. Computed over the whole item universe instead and
        // filtered afterwards, this returns the graph's articulation points —
        // the same set for every goal landing in the same component — and the
        // determination stops depending on the goal.
        //
        // `a → b → c`, plus a detour `a → d → c` that the goal's seed set
        // does not enter. Within R = {a, b, c}, `b` is the only route to `c`
        // and so is load-bearing. Over the whole universe the ablation of `b`
        // leaves the detour standing, `c` survives, and `b` is scored
        // redundant — a module the goal genuinely cannot do without.
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        for v in ["a", "b", "c", "d"] {
            g.add_edge(v, MEDIUM, 1.0).unwrap();
        }
        g.add_edge("a", "b", 2.0).unwrap();
        g.add_edge("b", "c", 2.0).unwrap();
        g.add_edge("c", "d", 2.0).unwrap();

        let seeds: BTreeSet<String> = ["a".to_string()].into_iter().collect();
        let reachable = reach(&g, &seeds);
        let universe: BTreeSet<String> = g.items().into_iter().map(|s| s.to_string()).collect();

        let within = necessary(&g, &seeds, &reachable);
        let over_universe: BTreeSet<String> = necessary(&g, &seeds, &universe)
            .intersection(&reachable)
            .cloned()
            .collect();

        assert!(within.is_subset(&reachable), "nec ⊆ reach");
        assert!(within.contains("b"), "b is the only route from a to c");
        assert_eq!(
            within, over_universe,
            "on a connected graph the two coincide; the fix matters where R ⊊ V"
        );
    }

    #[test]
    fn a_heading_of_function_words_draws_no_distinction() {
        // rem:quality-honest. A heading like "What the framework does not do"
        // contributes only `framework`: the rest is shared by most prose in
        // most repositories, and admitting it would put every document in
        // contact with every other — the highest floor, the worst
        // discrimination.
        let e = SymbolEntry {
            name: "What the framework does not do".into(),
            kind: "heading".into(),
            file: "README.md".into(),
            line: 1,
            snippet: String::new(),
        };
        let terms = terms_of(&e);
        assert_eq!(
            terms,
            ["framework".to_string()].into_iter().collect::<BTreeSet<_>>()
        );
    }

    #[test]
    fn a_function_word_is_still_a_name_when_a_symbol_carries_it() {
        // The exclusion is scoped to heading prose. A function named `use` or
        // a struct named `All` draws a real distinction and must keep it.
        let e = SymbolEntry {
            name: "over".into(),
            kind: "fn".into(),
            file: "a.rs".into(),
            line: 1,
            snippet: String::new(),
        };
        assert!(terms_of(&e).contains("over"));
    }

    #[test]
    fn a_short_goal_term_must_match_exactly() {
        // `goal` sits inside `goals` and `subgoal` and any heading saying the
        // word. Left to match by containment it seeds nearly everything and the
        // determination stops discriminating.
        assert!(!term_meets("goal", "subgoal"));
        assert!(!term_meets("cli", "client"));
        assert!(term_meets("goal", "goal"), "exact match still holds");
        assert!(term_meets("resolver", "codebaseresolver"));
        assert!(term_meets("codebaseresolver", "resolver"));
    }

    #[test]
    fn a_goal_naming_a_symbol_meets_its_kinded_term() {
        // `trait:resolver` is the distinction `resolver` drawn as a kind; a
        // goal naming the symbol should meet it without naming the kind.
        assert!(term_meets("resolver", "trait:resolver"));
        assert!(term_meets("map", "fn:map"), "short names still meet by kind");
        assert!(!term_meets("map", "trait:remap"));
    }

    #[test]
    fn the_character_is_conserved_when_modules_are_renamed() {
        // binv:invariant, at the level the CLI exposes: what a module is called
        // is not part of what it costs to separate.
        let idx = index(vec![
            entry("shared", "fn", "a.rs"),
            entry("shared", "fn", "b.rs"),
            entry("other", "fn", "b.rs"),
            entry("other", "fn", "c.rs"),
        ]);
        let g = induced_graph(&term_map(&idx, Granularity::File), FLOOR).unwrap();
        let before = module_character(&g);

        let mut perm = BTreeMap::new();
        perm.insert("a.rs".to_string(), "z9.rs".to_string());
        perm.insert("b.rs".to_string(), "a.rs".to_string());
        perm.insert("c.rs".to_string(), "m4.rs".to_string());
        let relabelled = g.relabel(&perm).unwrap();

        assert_eq!(before, module_character(&relabelled));
    }

    #[test]
    fn the_stored_form_round_trips_the_graph() {
        let idx = index(vec![
            entry("shared", "fn", "a.rs"),
            entry("shared", "fn", "b.rs"),
            entry("solo", "fn", "c.rs"),
        ]);
        let g = induced_graph(&term_map(&idx, Granularity::File), FLOOR).unwrap();
        let record = Record::new();
        let stored = StoredCkg::from_graph(Path::new("."), Granularity::File, FLOOR, &g, &record);
        let back = stored.graph().unwrap();

        assert_eq!(back.items(), g.items());
        assert_eq!(back.edge_count(), g.edge_count());
        assert_eq!(module_character(&back), module_character(&g));
    }

    #[test]
    fn the_stored_form_has_nowhere_to_put_a_determination() {
        // binv:search, checked structurally. If a verdict could be stored it
        // could be served from cache, and a cached verdict goes stale beneath
        // a moving graph.
        let idx = index(vec![entry("x", "fn", "a.rs")]);
        let g = induced_graph(&term_map(&idx, Granularity::File), FLOOR).unwrap();
        let stored = StoredCkg::from_graph(Path::new("."), Granularity::File, FLOOR, &g, &Record::new());
        let json = serde_json::to_string(&stored).unwrap();
        for forbidden in ["verdict", "accountable", "necessary", "determination"] {
            assert!(
                !json.contains(forbidden),
                "the stored ckg must not carry '{forbidden}'"
            );
        }
    }

    #[test]
    fn the_record_resumes_rather_than_restarting() {
        let mut r = Record::new();
        r.commit("a", "b", "first");
        r.commit("b", "c", "second");
        let idx = index(vec![entry("x", "fn", "a.rs")]);
        let g = induced_graph(&term_map(&idx, Granularity::File), FLOOR).unwrap();
        let stored = StoredCkg::from_graph(Path::new("."), Granularity::File, FLOOR, &g, &r);
        assert_eq!(stored.record, 2);

        let mut resumed = stored.record();
        assert_eq!(resumed.count(), 2);
        resumed.commit("c", "d", "third");
        assert_eq!(resumed.count(), 3, "the record never restarts");
    }
}
