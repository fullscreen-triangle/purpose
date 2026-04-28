# Integrating with the Purpose Framework

This document specifies exactly how other components — the Buhera operating system, the Blank Screen Interceptor, cascade routers, the MSI, individual scientific domains, external tools, HuggingFace models, databases, and arbitrary third-party services — integrate with the Rust workspace under `implementation/`.

It is a working reference, not a marketing document. If it says an interface is stable, that interface is frozen. If it says an interface evolves, expect changes. Every integration path is illustrated with real code from the crates that currently exist.

---

## 1. What the framework provides

The Rust workspace provides four things:

1. **A durable type system** for compilation: the `VaHera` AST, a typed `Operation` signature, a `Value` runtime type, a `Type` enum, and a `typecheck` function. These types are the interchange format between every component.
2. **Two integration points**: the `Resolver` trait (natural language → vaHera fragment) and the `Provider` trait (named operation → value). Everything plugs in through one or the other.
3. **A runtime executor** that walks vaHera fragments and dispatches their calls to registered providers, threading the output of each composition step into the next step's `input` slot.
4. **A CLI shell** that wraps the above into a usable command, mostly for debugging and development.

Everything else — the kernel, the interceptor, cascade routing, the factory, the Aperture Base — sits *on top of* these primitives. The framework does not impose a particular orchestration layer; it provides the pieces out of which orchestration layers are constructed.

### 1.1 What Purpose deliberately does not provide

- A training loop (belongs in `purpose-factory`, future work).
- A blank-screen UI (belongs in `purpose-interceptor`, future work).
- A kernel scheduler (belongs in `purpose-kernel`, future work).
- Model weights (live on disk, loaded by providers on demand).
- Domain content (lives in external substrates: UniProt, HuggingFace, databases, …).

The framework is intentionally narrow so that each of the above can be built independently against a stable interface.

---

## 2. The stability contract

### 2.1 Stable interfaces (frozen)

These are the types and traits that downstream code should depend on without expecting change:

| Symbol | Crate | Contract |
|---|---|---|
| `VaHera` | `purpose-core` | The four variants (`Call`, `Compose`, `Literal`, `Hole`) are permanent. New variants may be added in a minor version only if all existing consumers handle them gracefully via a fallback. |
| `Value` | `purpose-core` | The six variants (`Null`, `Bool`, `Num`, `Str`, `List`, `Record`) are permanent. |
| `Type` | `purpose-core` | Stable; new variants may be added but consumers may ignore unknown ones. |
| `Operation` | `purpose-core` | Stable struct shape. |
| `Domain` | `purpose-core` | Stable struct shape. |
| `Resolver` trait | `purpose-core` | `async fn compile(&self, utterance: &str) -> Result<VaHera, Error>` — permanent signature. |
| `Provider` trait | `purpose-operations` | `async fn invoke(&self, op: &str, args: &BTreeMap<String, Value>) -> Result<Value, Error>` — permanent signature. |
| `OperationRegistry::register` | `purpose-operations` | Stable API. |

Any breaking change to the above is a major-version event.

### 2.2 Evolving interfaces

These will grow as additional concerns are integrated:

- `Domain` — will gain `pure_intent_sampler`, `theoretical_corpus`, `verifier`, `cascade_position` fields when the factory comes online. Additive only; existing fields keep their meanings.
- `Executor` — will gain hooks for PVE, TEM, CMM, PSS dispatch when the kernel is wired in. Additive.
- `OperationRegistry` — will gain cascade-position indexing when routing is added.

Additions will not remove or rename existing fields.

### 2.3 What lives outside the framework

- All model weights and adapters (loaded through providers).
- All substrate content (loaded through providers).
- All presentation and UI concerns.
- All training-time state (belongs to the factory).

---

## 3. Integration path A — adding a new domain

This is the most common path. A domain contributes an operation vocabulary, a resolver that compiles utterances into vaHera fragments, and one or more providers that execute those fragments' calls.

The existing `purpose-domains-protein` crate is a complete example; use it as a template.

### 3.1 The four things you write

For a new domain "chemistry" targeting ChEMBL and PubChem, you write:

```rust
// 1. A resolver: utterance -> vaHera fragment.
pub struct ChemistryResolver;

#[async_trait::async_trait]
impl purpose_core::Resolver for ChemistryResolver {
    async fn compile(&self, utterance: &str) -> Result<VaHera, Error> {
        let compound = extract_compound_name(utterance)
            .ok_or_else(|| Error::Compile("no compound identified".into()))?;

        let mut args = BTreeMap::new();
        args.insert("name".into(), VaHera::Literal(Value::Str(compound)));

        Ok(VaHera::Compose(vec![
            VaHera::Call { op: "lookup_compound".into(), args },
            VaHera::Call { op: "summarize_compound".into(), args: BTreeMap::new() },
        ]))
    }
}

// 2. An operation vocabulary.
pub fn operations() -> Vec<Operation> {
    vec![
        Operation::new(
            "lookup_compound",
            /* inputs */ btreemap!{ "name".into() => Type::Str },
            /* output */ Type::named("CompoundRecord"),
            "Look up a chemical compound by name or SMILES.",
        ),
        Operation::new(
            "summarize_compound",
            btreemap!{ "input".into() => Type::named("CompoundRecord") },
            Type::Str,
            "Render a compound record as a human-readable summary.",
        ),
    ]
}

// 3. The Domain struct.
pub fn domain() -> Domain {
    Domain {
        name: "chemistry".into(),
        operations: operations(),
        resolver: Arc::new(ChemistryResolver),
    }
}

// 4. A registration function that wires providers to operations.
pub fn register_providers(registry: &mut OperationRegistry) {
    let pubchem = Arc::new(PubChemProvider::new());
    let summary = Arc::new(CompoundSummaryProvider);
    for op in operations() {
        match op.name.as_str() {
            "lookup_compound" => registry.register(op, pubchem.clone()),
            "summarize_compound" => registry.register(op, summary.clone()),
            _ => {}
        }
    }
}
```

That's the whole contract. Once a domain exposes `domain()` and `register_providers(&mut OperationRegistry)`, the rest of the framework — CLI, kernel, interceptor, cascade routing — consumes it without modification.

### 3.2 Resolver implementations you can choose from

A `Resolver` is any `Send + Sync` type that implements `async fn compile(&self, &str) -> Result<VaHera, Error>`. Four kinds exist:

| Kind | Implementation | When to use |
|---|---|---|
| **Hand-coded** | Regex / match expression emits a static vaHera template | MVP, debugging, testing, when the compilation mapping is trivial |
| **LoRA-adapted** | Loads Aperture Base + LoRA weights via Candle, does a forward pass | Production resolvers produced by the factory |
| **Remote** | Makes an RPC to a resolver service | When the resolver lives in a different process / host |
| **Composite** | Combines other resolvers with a dispatch rule | Domain-level routing before cascade descent |

The MVP ships with hand-coded only. The trait is identical for all four.

### 3.3 Multiple domains

Register as many domains as needed. The CLI currently hard-codes one; for a binary that serves many, route the utterance to the right domain via a small `DomainRouter`:

```rust
pub struct DomainRouter {
    domains: Vec<Domain>,
}

impl DomainRouter {
    pub async fn route(&self, utterance: &str) -> Result<(Domain, VaHera), Error> {
        for d in &self.domains {
            match d.resolver.compile(utterance).await {
                Ok(frag) if frag.is_fully_resolved() => return Ok((d.clone(), frag)),
                _ => continue,
            }
        }
        Err(Error::Compile("no domain matched utterance".into()))
    }
}
```

When cascade routing lands (path F below), this stub is replaced with the full cascade tree.

---

## 4. Integration path B — adding a new provider

A provider executes one or more named operations. Every external service — REST API, HuggingFace inference endpoint, local Candle model, database, Python subprocess, file system — integrates as a `Provider`.

### 4.1 The Provider trait

```rust
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error>;
}
```

A provider is usually implemented as a struct holding a connection (HTTP client, model handle, database pool) and a match expression dispatching by operation name.

### 4.2 Pattern: HTTP / REST provider

Exact pattern used by `UniprotProvider`. Holds a `reqwest::Client`, extracts arguments from the map, constructs a URL, parses JSON into `Value`:

```rust
pub struct ChEMBLProvider { client: reqwest::Client }

#[async_trait::async_trait]
impl Provider for ChEMBLProvider {
    async fn invoke(&self, op: &str, args: &BTreeMap<String, Value>) -> Result<Value, Error> {
        match op {
            "lookup_compound" => {
                let name = args.get("name").and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Provider("missing 'name'".into()))?;
                let url = format!("https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={}&format=json",
                                  urlencoding::encode(name));
                let body: serde_json::Value = self.client.get(&url).send().await
                    .map_err(|e| Error::Provider(e.to_string()))?
                    .json().await
                    .map_err(|e| Error::Provider(e.to_string()))?;
                Ok(json_to_value(&body))
            }
            _ => Err(Error::Provider(format!("unsupported op: {}", op))),
        }
    }
}
```

One file, maybe 80 lines.

### 4.3 Pattern: HuggingFace Inference API provider

Wraps the hosted HF Inference endpoint. Operation names map to model IDs; arguments flatten into the HF input schema:

```rust
pub struct HFInferenceProvider {
    client: reqwest::Client,
    token: String,                               // HF access token
    op_map: HashMap<String, String>,             // op name -> model id
}

impl HFInferenceProvider {
    pub fn new(token: String) -> Self {
        let mut op_map = HashMap::new();
        op_map.insert("embed_sequence".into(), "facebook/esm2_t33_650M_UR50D".into());
        op_map.insert("predict_structure".into(), "facebook/esmfold_v1".into());
        Self { client: reqwest::Client::new(), token, op_map }
    }
}

#[async_trait::async_trait]
impl Provider for HFInferenceProvider {
    async fn invoke(&self, op: &str, args: &BTreeMap<String, Value>) -> Result<Value, Error> {
        let model_id = self.op_map.get(op)
            .ok_or_else(|| Error::Provider(format!("unknown HF op: {}", op)))?;
        let url = format!("https://api-inference.huggingface.co/models/{}", model_id);

        // Flatten args into HF request body
        let mut body = serde_json::Map::new();
        for (k, v) in args {
            body.insert(k.clone(), value_to_json(v));
        }

        let resp: serde_json::Value = self.client.post(&url)
            .bearer_auth(&self.token)
            .json(&serde_json::Value::Object(body))
            .send().await.map_err(|e| Error::Provider(e.to_string()))?
            .json().await.map_err(|e| Error::Provider(e.to_string()))?;

        Ok(json_to_value(&resp))
    }
}
```

Every HuggingFace model you want to expose is one `op_map` entry. Adding ESMFold to the framework is literally one line plus the operation signature in whichever domain consumes it.

### 4.4 Pattern: local Candle model provider

For cases where you do not want to round-trip to the HF API — latency, privacy, cost — load the same model locally via Candle:

```rust
pub struct LocalESM2Provider {
    model: candle_transformers::models::esm::Model,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
}

#[async_trait::async_trait]
impl Provider for LocalESM2Provider {
    async fn invoke(&self, op: &str, args: &BTreeMap<String, Value>) -> Result<Value, Error> {
        match op {
            "embed_sequence" => {
                let seq = args.get("sequence").and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Provider("missing 'sequence'".into()))?;
                let embedding = tokio::task::block_in_place(|| {
                    let ids = self.tokenizer.encode(seq, true)
                        .map_err(|e| Error::Provider(e.to_string()))?;
                    // ... forward pass; emit pooled embedding as Value::List
                    Ok::<_, Error>(Value::List(vec![/* f64 per dim */]))
                })?;
                Ok(embedding)
            }
            _ => Err(Error::Provider(format!("unsupported op: {}", op))),
        }
    }
}
```

The provider trait hides whether inference is hosted or local. The resolver emits the same vaHera fragment in both cases; only the provider registration changes.

### 4.5 Pattern: Python subprocess bridge

For HuggingFace models whose Rust / Candle support is immature (AlphaFold, some protein LLMs, some diffusion models), a subprocess bridge is the pragmatic route. The framework makes this clean because the provider trait is all it requires:

```rust
pub struct PythonBridgeProvider {
    py_bin: PathBuf,
    script: PathBuf,   // long-running script communicating over stdin/stdout JSON
    child: Arc<tokio::sync::Mutex<tokio::process::Child>>,
}

#[async_trait::async_trait]
impl Provider for PythonBridgeProvider {
    async fn invoke(&self, op: &str, args: &BTreeMap<String, Value>) -> Result<Value, Error> {
        let request = serde_json::json!({ "op": op, "args": args });
        let mut guard = self.child.lock().await;
        // write request, read response -- one JSON line each
        let response: Value = ...;
        Ok(response)
    }
}
```

The Python side is a short script that reads one JSON line, loads the requested model, runs inference, writes one JSON line. Deployable as one Docker container alongside the Rust binary.

### 4.6 Pattern: database provider

Treat SQL queries as operations:

```rust
pub struct DuckDBProvider { pool: duckdb::Connection }

#[async_trait::async_trait]
impl Provider for DuckDBProvider {
    async fn invoke(&self, op: &str, args: &BTreeMap<String, Value>) -> Result<Value, Error> {
        match op {
            "lookup_metabolite_by_hmdb_id" => {
                let id = args.get("id").and_then(|v| v.as_str())
                    .ok_or_else(|| Error::Provider("missing 'id'".into()))?;
                let row: MetaboliteRow = tokio::task::block_in_place(|| {
                    self.pool.query_row("SELECT * FROM metabolites WHERE hmdb_id = ?", [id], ...)
                })?;
                Ok(row.into_value())
            }
            _ => Err(Error::Provider(format!("unsupported op: {}", op))),
        }
    }
}
```

The resolver never knows a database is involved. The substrate happens to be a database for this operation.

### 4.7 What a provider may not do

- Provide untyped output. Every `Value` returned must be shape-consistent with the registered operation's `output` type. Type violations break the type-checker.
- Assume argument ordering. `args` is a `BTreeMap` — order is alphabetical, not source-order. Extract by name.
- Panic. All errors must be returned as `Err(Error::Provider(...))`. Panics crash the whole runtime.
- Hold locks across `.await`. Use tokio-aware synchronisation (`tokio::sync::Mutex`, `RwLock`) if state must be shared.

---

## 5. Integration path C — embedding Purpose in a host application

When the host is another Rust binary (a server, a daemon, a larger application), depend on the crates directly:

```toml
[dependencies]
purpose-core = { path = "../mechanistic-synthesis/implementation/crates/purpose-core" }
purpose-operations = { path = "../mechanistic-synthesis/implementation/crates/purpose-operations" }
purpose-domains-protein = { path = "../mechanistic-synthesis/implementation/crates/purpose-domains-protein" }
```

Then construct the runtime explicitly:

```rust
let mut registry = OperationRegistry::new();
purpose_domains_protein::register_providers(&mut registry);
// ... register your own providers ...

let executor = Executor::new(registry);
let domain = purpose_domains_protein::domain();

// Per request:
let fragment = domain.resolver.compile(&utterance).await?;
let result = executor.execute(&fragment).await?;
```

For a host that cannot depend on Rust crates directly (Python, Node, Swift), wrap the CLI as a subprocess and communicate over stdin/stdout JSON:

```
$ purpose query "Tell me about SOD1" --raw
{ "primaryAccession": "P00441", ... }
```

The `--raw` flag emits the JSON `Value` directly; the host parses it.

---

## 6. Integrating with the Buhera operating system

The Buhera OS is a separate crate (`purpose-kernel`, not yet implemented) that will wrap the `Executor` and add the five subsystem responsibilities. Here is exactly how Buhera consumes the framework and what each subsystem adds.

### 6.1 What Buhera adds over the bare Executor

| Subsystem | What it adds | Where it hooks |
|---|---|---|
| **CMM** (Categorical Memory Manager) | Caches operation results by argument-hash in a coordinate-indexed store | Before dispatch: lookup; after execution: insert |
| **PSS** (Penultimate State Scheduler) | Orders pending operations by categorical distance to final state | Around the `Executor::execute_with` main loop |
| **DIC** (Demon I/O Controller) | Surgical retrieval — fetches only `I(D; A_Q)` bits from large sources | Special provider kind: `DemonProvider`, inspects the query's residual entropy |
| **PVE** (Proof Validation Engine) | Type-checks and formally verifies each fragment before execution | Wraps `typecheck`, adds refinement-type checks |
| **TEM** (Triple Equivalence Monitor) | Samples system state; checks conservation invariants hold | Independent tokio task, subscribes to executor events |

The kernel is composed of these subsystems plus the framework. Conceptually:

```rust
// purpose-kernel crate (future)
pub struct BuheraKernel {
    cmm: CategoricalMemoryManager,      // wraps Executor result cache
    pss: PenultimateStateScheduler,     // orders pending ops
    dic: DemonIoController,             // surgical retrieval
    pve: ProofValidationEngine,         // extends typecheck
    tem: TripleEquivalenceMonitor,      // independent sampler
    executor: purpose_operations::Executor,
    registry: Arc<OperationRegistry>,
}

impl BuheraKernel {
    pub async fn dispatch(&self, fragment: VaHera) -> Result<Value, Error> {
        self.pve.validate(&fragment)?;                       // PVE gate
        if let Some(cached) = self.cmm.lookup(&fragment) {   // CMM cache
            return Ok(cached);
        }
        let scheduled = self.pss.schedule(fragment).await;   // PSS ordering
        let result = self.executor.execute(&scheduled).await?;
        self.cmm.insert(&scheduled, &result);
        Ok(result)
    }
}
```

### 6.2 What the kernel does NOT change

- The `Resolver` trait. Resolvers emit the same fragments; the kernel consumes them.
- The `Provider` trait. Providers are invoked the same way.
- The `OperationRegistry`. Unchanged.
- The `VaHera` AST. Kernel reads it, does not rewrite it.

This is the crucial property: adding the kernel does not force any existing domain, provider, or resolver to change. Domains written against the MVP continue to work under Buhera without modification.

### 6.3 Migration path

- Stage 1 (now): use `Executor` directly. No kernel.
- Stage 2: add `BuheraKernel` as a new crate. It wraps `Executor`. Existing call sites migrate from `executor.execute(...)` to `kernel.dispatch(...)`. The CLI picks one or the other based on a `--kernel` flag.
- Stage 3: kernel becomes the default. `Executor` remains as a lower-level building block.

---

## 7. Integrating with the Blank Screen Interceptor

The Interceptor is the user-facing layer. It owns the presentation state machine, focus arbitration, session trajectory, and the MSI (which is itself a root `Resolver`). It is a separate crate (`purpose-interceptor`, not yet implemented).

### 7.1 What the Interceptor adds over the bare CLI

| Concern | Owned by Interceptor | Not owned by framework |
|---|---|---|
| Single-surface 4-state machine (blank / prompt / artifact / artifact+prompt) | Yes | — |
| Focus arbitration (single-focus invariant) | Yes | — |
| Session trajectory (precision-by-difference) | Yes | — |
| MSI (root coordinate-extraction resolver) | Yes (as `Arc<dyn Resolver>`) | — |
| No-RPC inter-layer semantics | Yes | — |
| Presentation rendering (TTY / GUI / web) | Yes | — |
| Compilation (utterance → vaHera) | Via its MSI | Delegated to framework |
| Execution | Via `BuheraKernel` or `Executor` | Delegated to framework |

### 7.2 The Interceptor's dependency on the framework

```rust
// purpose-interceptor crate (future)
pub struct Interceptor {
    presentation: PresentationState,
    focus: FocusArbiter,
    session: SessionTrajectory,
    msi: Arc<dyn purpose_core::Resolver>,        // root resolver (may be hand-coded now, trained later)
    kernel: Arc<BuheraKernel>,                   // or Arc<Executor> in stage 1
}

impl Interceptor {
    pub async fn on_utterance(&mut self, utterance: String) -> Result<(), Error> {
        self.presentation.enter_thinking();
        let fragment = self.msi.compile(&utterance).await?;
        let result = self.kernel.dispatch(fragment).await?;
        self.presentation.show_artifact(result);
        Ok(())
    }
}
```

The MSI is constructed by the Interceptor at startup and injected via `Arc<dyn Resolver>`. In the MVP today, the MSI can be a hand-coded resolver (regex-based, deterministic). Later, it becomes a LoRA-trained resolver produced by the factory. The Interceptor does not care which.

### 7.3 The file-operation symmetry property

The Interceptor is where the file-operation symmetry theorem (save ≡ retrieve, differentiated by one axis sign) is enforced. From the framework's perspective:

- "save" becomes an `archive_state` operation — a provider writes state to a coordinate-indexed store.
- "retrieve" becomes a `fetch_state` operation — the same provider reads.

Both operations use the same vaHera grammar. The Interceptor never exposes a file system.

---

## 8. Integrating with cascade routing

The cascade router is the third kind of runtime consumer (after kernel and interceptor). A cascade is a `k`-ary tree of cascade nodes; each internal node is itself a `Resolver` whose job is routing (select child subtree), and each leaf is a compilation `Resolver`.

### 8.1 Cascade node as a resolver

A cascade node is a Resolver that emits a special-purpose vaHera fragment naming its chosen child:

```rust
pub struct CascadeNode {
    name: String,
    resolver: Arc<dyn Resolver>,
    children: HashMap<String, Arc<CascadeNode>>,   // keyed by child name
}

impl CascadeNode {
    pub async fn route(&self, utterance: &str) -> Result<VaHera, Error> {
        // Internal: resolver emits Call { op: "route", args: {"to": <child_name>} }
        // Leaf: resolver emits a domain compilation fragment directly.
        if self.children.is_empty() {
            return self.resolver.compile(utterance).await;  // leaf
        }
        let fragment = self.resolver.compile(utterance).await?;
        let child_name = extract_route_target(&fragment)?;
        let child = self.children.get(&child_name)
            .ok_or_else(|| Error::Compile(format!("no child: {}", child_name)))?;
        Box::pin(child.route(utterance)).await
    }
}
```

Self-similarity is preserved: internal nodes and leaves share the same `Resolver` trait. Only the emitted fragment differs.

### 8.2 Registering cascade nodes

Once the cascade crate exists (`purpose-cascade`, future), registering a node is one call:

```rust
let mut cascade = Cascade::new();
cascade.root(Arc::new(msi_router));
cascade.add_child("biology", biology_router);
cascade.add_child("biology/protein", purpose_domains_protein::domain());
cascade.add_child("biology/chemistry", purpose_domains_chemistry::domain());
// ...
```

The cascade becomes the Interceptor's MSI (it is a Resolver), or the kernel's dispatch target (it implements the same trait).

### 8.3 The self-similarity invariant

Every cascade node — root, intermediate, leaf — is an `Arc<dyn Resolver>`. A single trait, a single interface, composable at any scale. The framework does not distinguish between "router" and "specialist"; they are the same object at different positions.

---

## 9. The data contract

### 9.1 vaHera wire format

Fragments serialise as JSON via serde. Example:

```json
{
  "compose": [
    {
      "call": {
        "op": "lookup_protein_by_gene",
        "args": {
          "gene": { "literal": "SOD1" }
        }
      }
    },
    {
      "call": {
        "op": "summarize_protein",
        "args": {}
      }
    }
  ]
}
```

Every variant of `VaHera` serialises as a single-key object under its variant name (externally-tagged). This is stable.

### 9.2 Value wire format

`Value` serialises untagged — each variant renders as its natural JSON type:

| Variant | JSON |
|---|---|
| `Null` | `null` |
| `Bool(b)` | `true` / `false` |
| `Num(n)` | `3.14` |
| `Str(s)` | `"text"` |
| `List(v)` | `[ ... ]` |
| `Record(m)` | `{ ... }` |

### 9.3 Error wire format

The `Error` enum is `thiserror`-derived; when serialised (e.g., across a process boundary), use `{"kind": "compile", "message": "..."}` or similar — not currently wire-stable. Treat errors as opaque across process boundaries.

---

## 10. Versioning

The workspace follows semver. Crates are versioned together (single workspace version). `0.1.x` is the MVP; expect frequent breaking changes within `0.x`. `1.0` will be cut once:

- At least three domains are registered.
- At least five provider kinds are in use (HTTP, HF-hosted, Candle-local, Python-bridge, SQL).
- The kernel and interceptor crates exist.
- A trained LoRA resolver has replaced at least one hand-coded resolver end-to-end.

Until `1.0`, interfaces may change. After `1.0`, the stability contract in §2 becomes binding.

---

## 11. Reference — the complete public API

### `purpose-core`

```rust
// vahera.rs
pub enum VaHera {
    Call { op: String, args: BTreeMap<String, VaHera> },
    Compose(Vec<VaHera>),
    Literal(Value),
    Hole(String),
}

impl VaHera {
    pub fn call<N: Into<String>>(name: N) -> Self;
    pub fn is_fully_resolved(&self) -> bool;
}

pub enum Value {
    Null, Bool(bool), Num(f64), Str(String),
    List(Vec<Value>), Record(BTreeMap<String, Value>),
}

impl Value {
    pub fn str<S: Into<String>>(s: S) -> Self;
    pub fn as_str(&self) -> Option<&str>;
    pub fn as_record(&self) -> Option<&BTreeMap<String, Value>>;
    pub fn as_list(&self) -> Option<&[Value]>;
}

// types.rs
pub enum Type {
    Str, Num, Bool, Unit,
    List(Box<Type>),
    Named(String),
    Var(String),
}

// operation.rs
pub struct Operation {
    pub name: String,
    pub inputs: BTreeMap<String, Type>,
    pub output: Type,
    pub description: String,
}

// domain.rs
pub struct Domain {
    pub name: String,
    pub operations: Vec<Operation>,
    pub resolver: Arc<dyn Resolver>,
}

#[async_trait]
pub trait Resolver: Send + Sync {
    async fn compile(&self, utterance: &str) -> Result<VaHera, Error>;
}

// typecheck.rs
pub fn typecheck(fragment: &VaHera, ops: &HashMap<String, Operation>) -> Result<Type, Error>;

// error.rs
pub enum Error { Compile(String), Type(String), Provider(String), Parse(String), Internal(String) }
```

### `purpose-operations`

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    async fn invoke(&self, op: &str, args: &BTreeMap<String, Value>) -> Result<Value, Error>;
}

pub struct OperationRegistry { ... }
impl OperationRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, op: Operation, provider: Arc<dyn Provider>);
    pub fn get(&self, name: &str) -> Option<(&Operation, &Arc<dyn Provider>)>;
    pub fn names(&self) -> impl Iterator<Item = &str>;
    pub fn operations(&self) -> impl Iterator<Item = &Operation>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}

pub struct Executor { ... }
impl Executor {
    pub fn new(registry: OperationRegistry) -> Self;
    pub fn registry(&self) -> &OperationRegistry;
    pub async fn execute(&self, program: &VaHera) -> Result<Value, Error>;
    pub fn execute_with<'a>(&'a self, program: &'a VaHera, piped: Option<Value>)
        -> Pin<Box<dyn Future<Output = Result<Value, Error>> + Send + 'a>>;
}

pub mod providers {
    pub struct UniprotProvider { ... }
    pub struct ProteinSummaryProvider;
}
```

### `purpose-domains-protein`

```rust
pub struct ProteinResolver;
pub fn domain() -> Domain;
pub fn operations() -> Vec<Operation>;
pub fn register_providers(registry: &mut OperationRegistry);
```

---

## 12. Quick reference for common integrations

| Want to … | Do this |
|---|---|
| Add a new scientific domain | Copy `purpose-domains-protein` as template; write `domain()`, `operations()`, `register_providers()`; write a resolver. |
| Expose a HuggingFace model | Implement `Provider` with the model ID in an op-map; register it against the operations that want it. |
| Expose a local model via Candle | Same as HF but load weights at construction; dispatch inference in `invoke`. |
| Expose a REST API | Implement `Provider` with a `reqwest::Client`; match on op name in `invoke`. |
| Expose a database | Implement `Provider` holding a connection pool; map operations to queries. |
| Expose a Python model | Implement `Provider` with a stdin/stdout subprocess bridge; serialise args as JSON. |
| Embed Purpose in another Rust binary | Add crates as `path = "..."` dependencies; construct `OperationRegistry` and `Executor`. |
| Embed Purpose in a non-Rust host | Spawn the CLI; communicate via `--raw` JSON on stdout. |
| Add a cascade layer | Wrap domains in `CascadeNode`s; each node is a `Resolver`. |
| Add a Buhera kernel | Wrap `Executor` in `BuheraKernel`; add CMM/PSS/PVE/TEM hooks around `dispatch`. |
| Add an interceptor | New crate owning presentation state + focus + session; inject an `Arc<dyn Resolver>` as its MSI. |
| Replace hand-coded resolver with LoRA | Implement `Resolver` for a Candle-backed adapter; register it in place of the old one. |

---

## 13. Design principles the framework enforces

These principles are load-bearing; integrations that violate them break the architectural guarantees:

1. **vaHera is the interchange format.** Resolvers emit vaHera, executors consume vaHera, verifiers inspect vaHera. No domain-specific side channels.
2. **Operations are typed.** Every operation has a declared signature; every fragment type-checks against the registry before execution.
3. **The empty dictionary principle.** Domain content lives outside the framework — in UniProt, HuggingFace, databases, file systems. Resolver weights encode compilation, not facts.
4. **Self-similarity at all scales.** Routers, specialists, MSI, cascade leaves are the same kind of object (a `Resolver`), distinguished only by training scope.
5. **Additive evolution.** New variants, fields, subsystems may be added; existing ones are never broken. The stability contract in §2 is binding.
6. **No hidden state.** Resolvers and providers are explicit objects with explicit lifetimes. All state is visible, testable, replaceable.

When you are deciding how to add something new, check whether your integration preserves these six principles. If it does, you fit into the architecture. If it does not, either refactor until it does or flag the tension explicitly — the framework can be extended, but not silently.

---

*This document will be updated as new crates, providers, and domains are added. The current revision covers the MVP workspace (`implementation/` with four crates: `purpose-core`, `purpose-operations`, `purpose-domains-protein`, `purpose-cli`). Each future crate — `purpose-kernel`, `purpose-interceptor`, `purpose-cascade`, `purpose-factory`, `purpose-aperture` — will add its own integration section here as it comes online.*
