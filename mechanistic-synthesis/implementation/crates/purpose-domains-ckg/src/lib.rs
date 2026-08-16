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
    align_score, character, dominated_by, floor_witness, induced_graph_weighted, necessary,
    reach, resting_cut, sigma_medium, system_floor, ContactGraph, EdgeKey, Record, TermMap,
    WeightedTermMap, MEDIUM,
};
use purpose_core::{Domain, Error, Operation, Resolver, Type, VaHera, Value};
use purpose_domains_codebase::{index_path, Index, SymbolEntry};
use purpose_operations::{OperationRegistry, Provider};

pub mod lens;
pub use lens::{load_lens, Lens, StoredLens};

/// Schema version of `.purpose/ckg.json`.
///
/// Bumped to 2 when the lens arrived: edge weights are sums over weighted terms
/// rather than shared counts, so a v1 graph read by this binary would be
/// silently reinterpreted rather than merely be out of date.
pub const CKG_VERSION: u32 = 2;

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
    pub fn module_of(&self, file: &str) -> String {
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

/// Distinctions drawn by one index entry, under a lens.
///
/// The bare name is the distinction itself; `kind:name` is the same
/// distinction drawn *as* a kind, so a `struct Resolver` and a `trait Resolver`
/// share one term and differ on another. Prose contributes heading tokens,
/// because a heading is where a document draws its distinctions.
///
/// Which of those the lens admits, and what each is worth, is the lens's to
/// decide — see [`lens`]. The default lens reproduces the map this tool drew
/// before lenses existed, save that `section` now counts as prose.
pub fn terms_of(lens: &Lens, entry: &SymbolEntry) -> BTreeSet<String> {
    lens.terms_of(entry).into_keys().collect()
}

/// Build the weighted term map over modules under a lens.
pub fn weighted_term_map(index: &Index, lens: &Lens) -> WeightedTermMap {
    lens.term_map(index)
}

/// Build the term map over modules under a lens, discarding weights.
///
/// A module with no indexed symbols does not appear. It draws no distinctions
/// this index can see, so it has no contacts, and an item joined to nothing but
/// the medium would sit at the floor and drag `β*` down for a reason that is an
/// artefact of extraction rather than a fact about the repository.
pub fn term_map(index: &Index, lens: &Lens) -> TermMap {
    lens::unweighted(&lens.term_map(index))
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
    /// A stable digest of the lens that induced this graph.
    #[serde(default)]
    pub lens_digest: String,
    /// Where that lens was read from, if not the built-in defaults.
    #[serde(default)]
    pub lens_source: Option<String>,
    /// The lens itself.
    ///
    /// Kept in full rather than as a digest alone, for two reasons. A
    /// determination must run against the lens the graph was built with, so it
    /// needs the lens and not merely a way to notice it moved; and a stale
    /// graph can then say *which* setting changed rather than only that
    /// something did.
    #[serde(default)]
    pub lens: Option<StoredLens>,
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

    /// The lens this graph was built with.
    ///
    /// Determinations must reconstruct τ from here rather than from
    /// `lens.toml` on disk. If the file has been edited since the build, the
    /// two disagree, and seeding against one τ while cutting a graph induced by
    /// another is not a determination about anything.
    pub fn lens(&self) -> Result<Lens, Error> {
        match &self.lens {
            Some(s) => s.to_lens(),
            // A graph stored before lenses existed, or one written by a build
            // that could not record its lens: the defaults are what it used.
            None => Ok(Lens::default()),
        }
    }

    fn from_graph(
        root: &Path,
        lens: &Lens,
        g: &ContactGraph,
        record: &Record,
    ) -> Self {
        let stored_lens = StoredLens::of(lens);
        StoredCkg {
            version: CKG_VERSION,
            root: root.display().to_string(),
            granularity: lens.granularity,
            floor: lens.floor,
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
            lens_digest: stored_lens.digest(),
            lens_source: lens.source.clone(),
            lens: Some(stored_lens),
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
pub fn build(root: &Path, lens: &Lens) -> Result<StoredCkg, Error> {
    let index = load_index(root)?;
    let g = induce(&index, lens)?;

    // Carry the existing record forward if there is one.
    let record = match load_ckg(root) {
        Ok(prev) => prev.record(),
        Err(_) => Record::new(),
    };

    let stored = StoredCkg::from_graph(root, lens, &g, &record);
    save_ckg(root, &stored)?;
    Ok(stored)
}

/// Induce the contact graph from an index under a lens, without writing it.
///
/// Shared by `build` and the lens diagnostics, so what the diagnostics report
/// is what a build would produce and not an approximation of it.
pub fn induce(index: &Index, lens: &Lens) -> Result<ContactGraph, Error> {
    let tau = lens.term_map(index);
    if tau.is_empty() {
        return Err(Error::Provider(
            "the index yielded no modules with distinctions — is it empty?".into(),
        ));
    }
    // Jaccard lands in (0, 1]; a floor at or above 1 clamps every contact to β
    // and the graph goes flat, so every determination comes out accountable and
    // nothing has been learned. Warn rather than refuse: it is a coherent graph,
    // just an uninformative one, and the choice is the operator's.
    if lens.edges == lens::EdgeWeight::Jaccard && lens.floor >= 1.0 {
        eprintln!(
            "warning: edges = \"jaccard\" with floor = {} — Jaccard weights are at most 1, \
             so every contact will clamp to the floor and the graph will be uniform. \
             Try floor = 0.01.",
            lens.floor
        );
    }
    induced_graph_weighted(&tau, lens.floor, lens.edge_weight())
        .map_err(|e| Error::Provider(format!("cannot induce contact graph: {e}")))
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

/// The term map the stored graph was actually built with.
///
/// Deliberately *not* whatever `.purpose/lens.toml` says now. Seeding against
/// one τ while cutting a graph induced by another is not a determination about
/// anything: a goal could seed a module the graph has no edges for. If the file
/// on disk has moved, say so and carry on — the determination against the
/// stored graph is still sound (`thm:tau-agnostic`: it is *a* graph, correctly
/// determined), it is simply about a τ the operator has since revised.
fn tau_as_built(root: &Path, stored: &StoredCkg) -> Result<TermMap, Error> {
    let lens = stored.lens()?;
    if let Ok(on_disk) = load_lens(root, None) {
        let d = StoredLens::of(&on_disk).digest();
        if !stored.lens_digest.is_empty() && d != stored.lens_digest {
            eprintln!(
                "note: {} has changed since this graph was built ({} → {}); \
                 determining against the stored lens. Run `purpose ckg build` to adopt it.",
                lens::LENS_FILE,
                &stored.lens_digest,
                &d
            );
        }
    }
    let index = load_index(root)?;
    Ok(term_map(&index, &lens))
}

/// Run Algorithm 2 against the stored graph.
///
/// Every cut here is recomputed from the graph as it currently stands. Nothing
/// is read from a previous answer, because there are no previous answers stored
/// to read.
pub fn determine(root: &Path, goal: &str, eps: f64) -> Result<Determination, Error> {
    let stored = load_ckg(root)?;
    let g = stored.graph()?;
    let tau = tau_as_built(root, &stored)?;

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
        let tau = tau_as_built(root, &stored)?;
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
// Lens diagnostics
// ---------------------------------------------------------------------------

/// What the diagnostics must always say, whatever the numbers are.
///
/// Shared with `render_floor` in substance so the two never drift into
/// disagreeing about what the floor means.
pub const FLOOR_CAVEAT: &str = "\
β* is a monotonicity signal, not a score. Refining the term map cannot lower it, so a
floor that is not rising under attempted refinement means the map is not being refined.
It does not follow that a higher floor is better: a lens under which every module draws
identical distinctions induces among the highest floors while discriminating worst. Read
it alongside the component sizes and term spread above. There is deliberately no
aggregate score here, because a scalar to maximise would be optimised, and optimising
this one produces the degenerate lens.";

/// How much of the index a lens admits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Coverage {
    pub entries_indexed: usize,
    pub admitted_by_paths: usize,
    pub admitted_by_kinds: usize,
    pub yielded_no_terms: usize,
    pub modules: usize,
}

/// One term, and how far it spreads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TermSpread {
    pub term: String,
    pub modules: usize,
    pub fraction: f64,
    pub weight: f64,
}

/// One module, by degree.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hub {
    pub module: String,
    pub degree: usize,
    pub sigma: f64,
}

/// How far a goal seeds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalSaturation {
    pub goal: String,
    pub seeds: usize,
    pub fraction: f64,
}

/// What a lens does to the structure of a repository.
///
/// Note what is absent, and note it deliberately: there is no score, rating, or
/// overall figure. `rem:quality-honest` is the reason — a scalar to maximise
/// would be maximised, and the maximum of every scalar available here is the
/// degenerate lens under which no module can be told from any other.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LensReport {
    pub lens: StoredLens,
    pub lens_source: Option<String>,
    pub lens_digest: String,
    pub coverage: Coverage,
    /// Component sizes, largest first, **ignoring the medium** — with it every
    /// graph is one component and the number says nothing.
    pub components: Vec<usize>,
    pub singletons: usize,
    pub density: f64,
    pub edge_count: usize,
    pub mean_weight: f64,
    pub median_weight: f64,
    pub floor: f64,
    pub system_floor: Option<f64>,
    pub floor_witness: Option<String>,
    pub omega: f64,
    pub hubs: Vec<Hub>,
    pub distinct_terms: usize,
    pub terms_in_one_module: usize,
    pub mean_terms_per_module: f64,
    pub spread: Vec<TermSpread>,
    pub goals: Vec<GoalSaturation>,
    pub caveat: String,
}

/// Measure what a lens does, without writing anything.
///
/// A dry run by construction: it induces the graph in memory and never touches
/// `ckg.json`. "Let me see what this does" must not be destructive, or nobody
/// will try anything.
pub fn lens_report(root: &Path, lens: &Lens, goals: &[String]) -> Result<LensReport, Error> {
    let index = load_index(root)?;

    // Coverage, counted while walking rather than inferred, so a lens that
    // quietly drops most of the index shows it as a number.
    let mut admitted_by_paths = 0usize;
    let mut admitted_by_kinds = 0usize;
    let mut yielded_no_terms = 0usize;
    for e in &index.symbols {
        if !lens.admits_path(e) {
            continue;
        }
        admitted_by_paths += 1;
        if !lens.admits_kind(e) {
            continue;
        }
        admitted_by_kinds += 1;
        if lens.terms_of(e).is_empty() {
            yielded_no_terms += 1;
        }
    }

    let tau = lens.term_map(&index);
    let g = induce(&index, lens)?;
    let items: Vec<String> = g.items();
    let n = items.len();

    // Components over the item-induced subgraph. The medium is adjacent to
    // everything, so leaving it in would make every graph connected.
    let mut adj: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for i in &items {
        adj.entry(i).or_default();
    }
    let mut weights: Vec<f64> = Vec::new();
    let mut edge_count = 0usize;
    for (u, v, w) in g.edges() {
        if u == MEDIUM || v == MEDIUM {
            continue;
        }
        edge_count += 1;
        weights.push(w);
        adj.entry(u).or_default().insert(v);
        adj.entry(v).or_default().insert(u);
    }

    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut components: Vec<usize> = Vec::new();
    for i in &items {
        let start: &str = i;
        if seen.contains(start) {
            continue;
        }
        let mut stack = vec![start];
        let mut size = 0usize;
        seen.insert(start);
        while let Some(x) = stack.pop() {
            size += 1;
            for y in adj.get(x).into_iter().flatten() {
                if seen.insert(y) {
                    stack.push(y);
                }
            }
        }
        components.push(size);
    }
    components.sort_unstable_by(|a, b| b.cmp(a));
    let singletons = components.iter().filter(|c| **c == 1).count();

    weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_weight = if weights.is_empty() {
        0.0
    } else {
        weights.iter().sum::<f64>() / weights.len() as f64
    };
    let median_weight = if weights.is_empty() {
        0.0
    } else {
        weights[weights.len() / 2]
    };
    let density = if n > 1 {
        2.0 * edge_count as f64 / (n as f64 * (n as f64 - 1.0))
    } else {
        0.0
    };

    let mut hubs: Vec<Hub> = items
        .iter()
        .map(|m| Hub {
            module: m.clone(),
            degree: adj.get(m.as_str()).map(|s| s.len()).unwrap_or(0),
            sigma: sigma_medium(&g, m),
        })
        .collect();
    hubs.sort_by(|a, b| b.degree.cmp(&a.degree).then_with(|| a.module.cmp(&b.module)));
    hubs.truncate(10);

    // Term spread. The head of this list is the actionable output of the whole
    // command: a term carried by most modules is what put them all in contact.
    let mut carriers: BTreeMap<&str, usize> = BTreeMap::new();
    let mut weight_of: BTreeMap<&str, f64> = BTreeMap::new();
    let mut total_terms = 0usize;
    for ts in tau.values() {
        total_terms += ts.len();
        for (t, w) in ts {
            *carriers.entry(t).or_insert(0) += 1;
            let slot = weight_of.entry(t).or_insert(*w);
            if *w > *slot {
                *slot = *w;
            }
        }
    }
    let modules = tau.len();
    let terms_in_one_module = carriers.values().filter(|c| **c == 1).count();
    let mut spread: Vec<TermSpread> = carriers
        .iter()
        .map(|(t, c)| TermSpread {
            term: (*t).to_string(),
            modules: *c,
            fraction: if modules > 0 { *c as f64 / modules as f64 } else { 0.0 },
            weight: weight_of.get(*t).copied().unwrap_or(1.0),
        })
        .collect();
    spread.sort_by(|a, b| b.modules.cmp(&a.modules).then_with(|| a.term.cmp(&b.term)));
    spread.truncate(20);

    // Per goal, never averaged: a mean over goals would hide the one goal that
    // seeds the whole repository, which is the only one worth acting on.
    let unweighted = lens::unweighted(&tau);
    let goal_reports = goals
        .iter()
        .map(|goal| {
            let seeds = seeds_for(&unweighted, &goal_terms(goal));
            GoalSaturation {
                goal: goal.clone(),
                seeds: seeds.len(),
                fraction: if modules > 0 {
                    seeds.len() as f64 / modules as f64
                } else {
                    0.0
                },
            }
        })
        .collect();

    let stored_lens = StoredLens::of(lens);
    Ok(LensReport {
        lens_digest: stored_lens.digest(),
        lens: stored_lens,
        lens_source: lens.source.clone(),
        coverage: Coverage {
            entries_indexed: index.symbols.len(),
            admitted_by_paths,
            admitted_by_kinds,
            yielded_no_terms,
            modules,
        },
        components,
        singletons,
        density,
        edge_count,
        mean_weight,
        median_weight,
        floor: lens.floor,
        system_floor: system_floor(&g),
        floor_witness: floor_witness(&g).map(|(m, _)| m),
        omega: g.total_weight(),
        hubs,
        distinct_terms: carriers.len(),
        terms_in_one_module,
        mean_terms_per_module: if modules > 0 {
            total_terms as f64 / modules as f64
        } else {
            0.0
        },
        spread,
        goals: goal_reports,
        caveat: FLOOR_CAVEAT.to_string(),
    })
}

/// Render a lens report for a reader.
pub fn render_lens_report(r: &LensReport) -> String {
    let mut out = String::new();
    let l = &r.lens;

    out.push_str("LENS\n");
    out.push_str(&format!(
        "  source          {}\n  digest          {}\n",
        r.lens_source.as_deref().unwrap_or("(built-in defaults)"),
        r.lens_digest
    ));
    out.push_str(&format!(
        "  granularity     {}\n  floor (β)       {:.2}\n  edges           {}\n",
        l.granularity.as_str(),
        l.floor,
        l.edges.as_str()
    ));
    out.push_str(&format!(
        "  paths           {}\n  kinds           {}\n",
        if l.paths.is_empty() {
            "(all)".to_string()
        } else {
            l.paths.join(", ")
        },
        match &l.kinds {
            Some(k) => k.iter().cloned().collect::<Vec<_>>().join(", "),
            None => "(all)".to_string(),
        }
    ));
    out.push_str(&format!(
        "  min_len         {}\n  split_camel     {}\n  prose_kinds     {}\n  stopwords       {}\n  aliases         {}\n  weights         {}\n",
        l.min_len,
        l.split_camel_case,
        l.prose_kinds.iter().cloned().collect::<Vec<_>>().join(", "),
        l.stopwords.len(),
        l.alias.len(),
        l.weight.len()
    ));

    let c = &r.coverage;
    out.push_str("\nCOVERAGE\n");
    out.push_str(&format!(
        "  {} entries indexed\n  {} admitted by paths\n  {} admitted by kinds\n  {} yielded no terms\n  {} modules in τ\n",
        c.entries_indexed, c.admitted_by_paths, c.admitted_by_kinds, c.yielded_no_terms, c.modules
    ));

    out.push_str("\nCOMPONENTS (medium excluded)\n");
    let shown: Vec<String> = r.components.iter().take(12).map(|s| s.to_string()).collect();
    out.push_str(&format!(
        "  {} component(s): {}{}\n  {} singleton(s)\n",
        r.components.len(),
        shown.join(", "),
        if r.components.len() > 12 { ", …" } else { "" },
        r.singletons
    ));
    if let Some(largest) = r.components.first() {
        let n: usize = r.components.iter().sum();
        if n > 0 && *largest as f64 / n as f64 > 0.5 {
            out.push_str(&format!(
                "  {}% of modules are in one component — τ is not discriminating between them\n",
                (100.0 * *largest as f64 / n as f64).round() as u64
            ));
        }
    }

    out.push_str("\nDENSITY\n");
    let n: usize = r.components.iter().sum();
    out.push_str(&format!(
        "  n = {}, e = {}, density = {:.3}\n  weight mean {:.2}, median {:.2}\n  β = {:.2}",
        n, r.edge_count, r.density, r.mean_weight, r.median_weight, r.floor
    ));
    match (r.system_floor, &r.floor_witness) {
        (Some(b), Some(m)) => out.push_str(&format!(", β* = {b:.2} at {m}\n")),
        (Some(b), None) => out.push_str(&format!(", β* = {b:.2}\n")),
        _ => out.push_str(", β* undefined\n"),
    }
    out.push_str(&format!("  Ω = {:.2}\n", r.omega));

    out.push_str("\nDEGREE HUBS\n");
    for h in &r.hubs {
        out.push_str(&format!(
            "  {:>4}  σ {:>7.2}  {}\n",
            h.degree, h.sigma, h.module
        ));
    }

    out.push_str("\nTERM SPREAD\n");
    out.push_str(&format!(
        "  {} distinct terms, {} in exactly one module, {:.1} per module\n",
        r.distinct_terms, r.terms_in_one_module, r.mean_terms_per_module
    ));
    out.push_str("  the terms at the top of this list are the stopword candidates:\n");
    for s in &r.spread {
        out.push_str(&format!(
            "  {:>4} ({:>3.0}%)  w {:.2}  {}\n",
            s.modules,
            100.0 * s.fraction,
            s.weight,
            s.term
        ));
    }

    if !r.goals.is_empty() {
        out.push_str("\nGOAL SATURATION\n");
        for g in &r.goals {
            out.push_str(&format!(
                "  {:>4} module(s) ({:>3.0}%)  {}{}\n",
                g.seeds,
                100.0 * g.fraction,
                g.goal,
                if g.fraction > 0.5 {
                    "   ← seeds most of the repository; the graph cannot discriminate here"
                } else {
                    ""
                }
            ));
        }
    }

    out.push_str("\n");
    out.push_str(&r.caveat);
    out.push('\n');
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
            // `granularity` and `floor` remain accepted here, and still
            // override, so every vaHera script written against the pre-lens
            // surface keeps working. What changed is where they come from when
            // the caller says nothing: the lens, not a constant.
            "build_ckg" => {
                let mut lens = load_lens(&self.root, args.get("lens").and_then(|v| v.as_str()).map(Path::new))?;
                if let Some(g) = args.get("granularity").and_then(|v| v.as_str()) {
                    lens.granularity = Granularity::parse(g)?;
                }
                if let Some(Value::Num(f)) = args.get("floor") {
                    lens.floor = *f;
                }
                let stored = build(&self.root, &lens)?;
                to_value(&stored)
            }
            "lens_report" => {
                let lens = load_lens(&self.root, args.get("lens").and_then(|v| v.as_str()).map(Path::new))?;
                let goals: Vec<String> = match args.get("goal") {
                    Some(Value::Str(s)) => vec![s.clone()],
                    Some(Value::List(xs)) => {
                        xs.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
                    }
                    _ => Vec::new(),
                };
                let r = lens_report(&self.root, &lens, &goals)?;
                Ok(Value::Str(render_lens_report(&r)))
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
                inputs.insert("lens".into(), Type::Str);
                inputs
            },
            Type::named("Ckg"),
            "Induce the module contact graph from the symbol index and store it.",
        ),
        Operation::new(
            "lens_report",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("lens".into(), Type::Str);
                inputs.insert("goal".into(), Type::Str);
                inputs
            },
            Type::Str,
            "Report what a lens does to the structure of the graph, without building it.",
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
            "build_ckg" | "lens_report" | "module_floor" | "determine"
            | "format_determination" | "explain_module" => {
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

    /// The default lens, which by construction reproduces the map this tool
    /// drew before lenses existed. Every pre-lens test routes through it, with
    /// its assertions unchanged — that is the compatibility claim, checked once
    /// per behaviour rather than asserted once in prose.
    fn def() -> Lens {
        Lens::default()
    }

    /// A lens parsed from a fragment, for the tests that are about parsing.
    fn lens_from(src: &str) -> Lens {
        lens::parse_lens(src).expect("fixture lens must parse")
    }

    fn tmap(idx: &Index) -> TermMap {
        term_map(idx, &def())
    }

    fn graph_of(idx: &Index) -> ContactGraph {
        induce(idx, &def()).unwrap()
    }

    /// A scratch repository holding one index, removed when the test ends.
    ///
    /// `lens_report` reads `.purpose/index.json` from disk by design — it must
    /// measure what the tool will actually see — so the tests that exercise it
    /// need a real directory rather than an in-memory `Index`.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new(tag: &str, idx: &Index) -> Scratch {
            let dir = std::env::temp_dir().join(format!("purpose-ckg-test-{tag}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join(".purpose")).unwrap();
            std::fs::write(
                dir.join(".purpose").join("index.json"),
                serde_json::to_string(idx).unwrap(),
            )
            .unwrap();
            Scratch(dir)
        }

        fn root(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_symbol_draws_its_name_and_its_kinded_name() {
        let t = terms_of(&def(), &entry("Resolver", "trait", "a.rs"));
        assert!(t.contains("resolver"));
        assert!(t.contains("trait:resolver"));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn same_name_different_kind_shares_one_term_and_differs_on_another() {
        // A struct Resolver and a trait Resolver are in contact — they draw the
        // same distinction — but they are not the same distinction drawn.
        let a = terms_of(&def(), &entry("Resolver", "trait", "a.rs"));
        let b = terms_of(&def(), &entry("Resolver", "struct", "b.rs"));
        assert_eq!(a.intersection(&b).count(), 1);
        assert_ne!(a, b);
    }

    #[test]
    fn a_heading_contributes_its_content_words() {
        let t = terms_of(&def(), &entry("The Directional Pair 2", "heading", "d.md"));
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
        let by_file = tmap(&idx);
        let by_dir = term_map(&idx, &Lens { granularity: Granularity::Dir, ..def() });
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
        let g = graph_of(&idx);
        assert_eq!(g.weight("core.rs", "impl.rs"), Some(1.0));
        assert_eq!(g.weight("core.rs", "other.rs"), None);
    }

    #[test]
    fn a_module_with_no_indexed_symbols_does_not_become_an_item() {
        // It would sit alone at the floor and drag β* down for a reason that is
        // about extraction, not about the repository.
        let idx = index(vec![entry("", "fn", "empty.rs"), entry("x", "fn", "a.rs")]);
        let tau = tmap(&idx);
        assert!(!tau.contains_key("empty.rs"));
        assert!(tau.contains_key("a.rs"));
    }

    #[test]
    fn goal_terms_match_by_substring_in_both_directions() {
        let idx = index(vec![
            entry("CodebaseResolver", "struct", "a.rs"),
            entry("unrelated", "fn", "b.rs"),
        ]);
        let tau = tmap(&idx);
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
        // The two agree on any graph the goal reaches entirely, so a sharp
        // witness needs R ⊊ V — and needs the excluded part to *matter*. An
        // unreached module that merely sits there changes nothing: it is
        // unreachable under every retained set, contributes 0, and scores
        // redundant either way. What discriminates is an unreached module that
        // acts as a BRIDGE, offering a detour the goal does not actually have.
        //
        //           a ─── b ─── c
        //            \         /
        //             ╰── d ──╯          d admitted, but not seeded into
        //
        // R is what the goal's *own* seeds reach. Take the goal to seed at `a`
        // and `c` with `d` outside R — the case where the index admits `d` as a
        // module but no goal term meets it.
        //
        // Within R = {a, b, c}: dropping `b` severs `a` from `c`, so `b` is
        // load-bearing. Over V = {a, b, c, d}: dropping `b` leaves the detour
        // through `d` standing, `c` survives, and `b` is scored REDUNDANT —
        // a module the goal genuinely cannot do without, rescued by a route
        // through a module the determination never admitted.
        let mut g = ContactGraph::with_vertices([MEDIUM]);
        for v in ["a", "b", "c", "d"] {
            g.add_edge(v, MEDIUM, 1.0).unwrap();
        }
        g.add_edge("a", "b", 2.0).unwrap();
        g.add_edge("b", "c", 2.0).unwrap();
        g.add_edge("a", "d", 2.0).unwrap();
        g.add_edge("d", "c", 2.0).unwrap();

        let seeds: BTreeSet<String> = ["a".to_string()].into_iter().collect();
        // R is stated directly: the retained set is the determination's, and
        // here it excludes the detour.
        let reachable: BTreeSet<String> =
            ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let universe: BTreeSet<String> = g.items().into_iter().map(|s| s.to_string()).collect();
        assert!(
            reachable.is_subset(&universe) && reachable.len() < universe.len(),
            "the witness is only sharp where R ⊊ V"
        );

        let within = necessary(&g, &seeds, &reachable);
        let over_universe = necessary(&g, &seeds, &universe);

        assert!(within.is_subset(&reachable), "nec ⊆ reach");
        assert!(
            within.contains("b"),
            "within R, b is the only route from a to c"
        );

        // The defect this pins, stated as the disagreement itself.
        assert!(
            !over_universe.contains("b"),
            "over the universe the detour rescues b and it is scored redundant"
        );
        assert_ne!(
            within, over_universe,
            "the two computations must disagree, or the test proves nothing"
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
        let terms = terms_of(&def(), &e);
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
        assert!(terms_of(&def(), &e).contains("over"));
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
        let g = graph_of(&idx);
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
        let g = graph_of(&idx);
        let record = Record::new();
        let stored = StoredCkg::from_graph(Path::new("."), &def(), &g, &record);
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
        let g = graph_of(&idx);
        let stored = StoredCkg::from_graph(Path::new("."), &def(), &g, &Record::new());
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
        let g = graph_of(&idx);
        let stored = StoredCkg::from_graph(Path::new("."), &def(), &g, &r);
        assert_eq!(stored.record, 2);

        let mut resumed = stored.record();
        assert_eq!(resumed.count(), 2);
        resumed.commit("c", "d", "third");
        assert_eq!(resumed.count(), 3, "the record never restarts");
    }

    // -- Compatibility ------------------------------------------------------

    #[test]
    fn the_default_lens_reproduces_the_hard_coded_term_map() {
        // The whole compatibility requirement in one assertion: against a
        // frozen expectation, not against a second computation, so a change to
        // the lens machinery cannot move both sides at once.
        //
        // The `section` entry is the single deliberate departure. Before
        // lenses, only `heading` counted as prose, so this title was taken
        // whole as an identifier and yielded `the directional pair` plus
        // `section:the directional pair`.
        let idx = index(vec![
            entry("Resolver", "trait", "a.rs"),
            entry("compile", "fn", "a.rs"),
            entry("What the framework does not do", "heading", "README.md"),
            entry("The Directional Pair", "section", "paper.tex"),
        ]);

        let expected: TermMap = [
            (
                "a.rs",
                vec!["resolver", "trait:resolver", "compile", "fn:compile"],
            ),
            ("README.md", vec!["framework"]),
            ("paper.tex", vec!["directional", "pair"]),
        ]
        .into_iter()
        .map(|(m, ts)| {
            (m.to_string(), ts.into_iter().map(String::from).collect::<BTreeSet<_>>())
        })
        .collect();

        assert_eq!(tmap(&idx), expected);
    }

    // -- Diagnostics --------------------------------------------------------

    #[test]
    fn the_diagnostics_report_no_aggregate_score() {
        // rem:quality-honest, made mechanically checkable. A scalar to maximise
        // would be maximised, and the maximum of every scalar available here is
        // the degenerate lens under which no module can be told from any other.
        // In the spirit of `the_stored_form_has_nowhere_to_put_a_determination`:
        // the absence is structural, not a matter of what the renderer prints.
        let idx = index(vec![
            entry("shared", "fn", "a.rs"),
            entry("shared", "fn", "b.rs"),
            entry("solo", "fn", "c.rs"),
        ]);
        let s = Scratch::new("no-score", &idx);
        let r = lens_report(s.root(), &def(), &["shared".to_string()]).unwrap();
        let json = serde_json::to_value(&r).unwrap();

        fn keys(v: &serde_json::Value, out: &mut Vec<String>) {
            match v {
                serde_json::Value::Object(m) => {
                    for (k, x) in m {
                        out.push(k.clone());
                        keys(x, out);
                    }
                }
                serde_json::Value::Array(xs) => xs.iter().for_each(|x| keys(x, out)),
                _ => {}
            }
        }
        let mut ks = Vec::new();
        keys(&json, &mut ks);
        for k in &ks {
            let l = k.to_lowercase();
            for forbidden in ["score", "quality", "rating", "grade", "overall"] {
                assert!(
                    !l.contains(forbidden),
                    "the diagnostics must carry no aggregate figure, found key '{k}'"
                );
            }
        }
    }

    #[test]
    fn a_degenerate_lens_shows_one_component_and_the_highest_floor() {
        // The paper's argument in a single test. A lens admitting every token
        // puts every module in contact with every other: the floor rises, the
        // components collapse to one, and a term appears in every module. That
        // is the worst possible map and it maximises the one number available,
        // which is why there is no score to raise.
        let idx = index(vec![
            entry("What the parser does", "heading", "a.md"),
            entry("What the emitter does", "heading", "b.md"),
            entry("What the loader does", "heading", "c.md"),
        ]);
        let s = Scratch::new("degenerate", &idx);

        let sharp = lens_report(s.root(), &def(), &[]).unwrap();
        let blunt = lens_from("[terms]\nstopwords = []\nmin_len = 1\n");
        let blunt = lens_report(s.root(), &blunt, &[]).unwrap();

        assert_eq!(blunt.components, vec![3], "one component, everything in it");
        assert!(
            sharp.components.len() > blunt.components.len(),
            "the discriminating lens must leave the modules apart: {:?} vs {:?}",
            sharp.components,
            blunt.components
        );
        assert!(
            blunt.system_floor > sharp.system_floor,
            "the degenerate lens induces the higher floor ({:?} vs {:?})",
            blunt.system_floor,
            sharp.system_floor
        );
        assert!(
            blunt.spread.iter().any(|t| t.modules == 3),
            "a term carried by every module is exactly what went wrong"
        );
    }

    #[test]
    fn running_the_lens_command_does_not_write_the_ckg() {
        // "Let me see what this does" must not be destructive, or nobody will
        // try anything and the instrument goes unused.
        let idx = index(vec![entry("x", "fn", "a.rs")]);
        let s = Scratch::new("dry-run", &idx);
        lens_report(s.root(), &def(), &[]).unwrap();
        assert!(
            !ckg_path(s.root()).exists(),
            "`ckg lens` must never write {}",
            ckg_path(s.root()).display()
        );
    }

    #[test]
    fn a_determination_uses_the_lens_the_graph_was_built_with() {
        // Design risk 2, pinned. If `determine` rebuilt τ from `lens.toml` on
        // disk while the graph came from an older lens, seeds and cuts would
        // disagree and the determination would be incoherent. The stored lens
        // is the authority.
        let idx = index(vec![
            entry("resolver", "fn", "a.rs"),
            entry("resolver", "fn", "b.rs"),
            entry("emit", "fn", "c.rs"),
        ]);
        let s = Scratch::new("stored-lens", &idx);
        build(s.root(), &def()).unwrap();

        // Now move the on-disk lens somewhere that would seed nothing at all:
        // `resolver` becomes a stopword and only headings are admitted.
        std::fs::write(
            s.root().join(lens::LENS_FILE),
            "[include]\nkinds = [\"heading\"]\n[terms]\nstopwords = [\"resolver\"]\n",
        )
        .unwrap();

        // `Determination` carries no seed list, so read the reachable set —
        // which is exactly what seeding produced. Under the edited lens
        // `resolver` is a stopword and no `.rs` file is admitted, so seeds
        // would be empty and nothing would be reachable.
        let d = determine(s.root(), "resolver", 0.0).unwrap();
        assert!(
            !d.necessary.is_empty() || !d.redundant.is_empty(),
            "the determination must seed against the stored lens, not the edited file"
        );

        // And the graph it ran against still records the lens it was built with.
        let stored = load(s.root()).unwrap();
        assert_eq!(
            stored.lens_digest,
            StoredLens::of(&def()).digest(),
            "the stored graph must say which lens induced it"
        );
    }

    #[test]
    fn jaccard_at_the_default_floor_flattens_every_contact() {
        // Design risk 1, pinned rather than merely warned about. Jaccard lands
        // in (0, 1], so at β = 1 every contact clamps to the floor: the graph
        // is uniform and every determination comes out accountable for a reason
        // that is an artefact of the weight function.
        let idx = index(vec![
            entry("shared", "fn", "a.rs"),
            entry("shared", "fn", "b.rs"),
            entry("other", "fn", "b.rs"),
        ]);
        let l = lens_from("[edges]\nweight = \"jaccard\"\n");
        let g = induce(&idx, &l).unwrap();
        for (_, _, w) in g.edges() {
            assert_eq!(w, 1.0, "every contact clamped to β — the graph is flat");
        }

        // Below the floor the same lens discriminates again.
        let l = lens_from("[lens]\nfloor = 0.01\n[edges]\nweight = \"jaccard\"\n");
        let g = induce(&idx, &l).unwrap();
        assert!(
            g.edges().any(|(_, _, w)| w < 1.0 && w > 0.01),
            "at a floor beneath the Jaccard range the weights survive"
        );
    }
}
