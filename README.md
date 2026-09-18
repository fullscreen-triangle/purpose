<h1 align="center">Purpose</h1>
<p align="center"><em>The reason for which something has the right to expire</em></p>

<p align="center">
<img src="boltzmann.png" alt="Purpose Logo" width="200"/>
</p>

# Purpose: Accountable Compilation for Domain-Specific Language Models

## Overview

Purpose produces domain-specific language models from a declared body of
knowledge rather than from a general-purpose model paired with retrieval. A
theme — a name, a set of sources, a model shape — is compiled into a trained,
self-contained model: sources are ingested, formed into a verified training
corpus, and used to train a model from scratch or to LoRA-adapt a pretrained
checkpoint, entirely in Rust, on Candle.

The framework has three layers:

1. **Theory.** [`absicht`](absicht/) — *Accountable Compilation: A Unified
   Framework for Producing, Composing, and Cross-Verifying Domain-Specific
   Language Models* — the formal account of what a domain must declare, how
   several trained models compose with a provable error floor, and how two
   models with no shared internal representation certify agreement without
   either disclosing its internal state.
2. **Implementation.** [`purpose-factory`](mechanistic-synthesis/implementation/crates/purpose-factory)
   — the Rust crate implementing the theory's acquisition pipeline: source
   ingestion, corpus construction, training, and export. Exposed as a local
   CLI (`purpose factory`) and, for a caller on a different machine, as a
   token-authenticated HTTP server (`purpose serve`).
3. **Interface.** [`profile-web`](mechanistic-synthesis/profile-web) — a
   personal knowledge-profile application: add papers, structured data, and
   links to a named profile, and train a model from it, against either a
   local `purpose` binary or a remote `purpose serve` instance.

A fourth, unrelated component — the `purpose` **codebase navigation tool** —
ships in the same binary and answers a different question entirely: *where is
X defined in this project?* See [below](#the-purpose-codebase-navigation-tool).

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────────┐
│                    │     │                    │     │                    │
│  Sources           │────▶│  purpose-factory   │────▶│  Domain-Specific   │
│  (papers, CSV,     │     │  ingest → corpus   │     │  Model             │
│  JSON, links)      │     │  → train → export  │     │  (.safetensors)    │
│                    │     │                    │     │                    │
└────────────────────┘     └────────────────────┘     └────────────────────┘
         ▲                            ▲
         │                            │
┌────────────────────┐     ┌────────────────────┐
│                    │     │                    │
│  profile-web       │────▶│  purpose serve     │
│  (personal profile │     │  (bearer-token     │
│  UI)               │     │  HTTP boundary)    │
│                    │     │                    │
└────────────────────┘     └────────────────────┘
```

---

## Table of Contents

0. [The `purpose` Codebase Navigation Tool](#the-purpose-codebase-navigation-tool)
1. [Theoretical Framework — `absicht`](#theoretical-framework--absicht)
2. [`purpose-factory`: Acquisition Pipeline](#purpose-factory-acquisition-pipeline)
   - [Sources](#sources)
   - [Training](#training)
   - [The `theme.toml` Contract](#the-themetoml-contract)
3. [The CLI](#the-cli)
4. [`purpose serve`: The HTTP Boundary](#purpose-serve-the-http-boundary)
5. [`profile-web`: The Personal Knowledge Profile](#profile-web-the-personal-knowledge-profile)
6. [`purpose-factory-ts`](#purpose-factory-ts)
7. [Workspace Layout](#workspace-layout)
8. [Building From Source](#building-from-source)
9. [Scope and Open Problems](#scope-and-open-problems)
10. [License](#license)

---

## The `purpose` Codebase Navigation Tool

> **For AI assistants working in this repo:** read
> [`mechanistic-synthesis/synthesis.md`](mechanistic-synthesis/synthesis.md)
> first, then use `purpose ask` instead of reading files broadly.

Independent of the training framework, the `purpose` binary also answers the
question *"where is X in this project?"* without an assistant having to read
the whole codebase. It builds a local index of a project's symbol definitions
and returns the relevant `file:line` slice for any natural-language question.
It runs entirely on the machine: no API key, no network, no per-query cost.

```bash
# 1. Build the index for the current project (run once; re-run after big changes)
purpose index

# 2. Ask where something is — returns the relevant code slice
purpose ask "where is the database connection set up"
purpose ask "Resolver trait"
```

| Command | What it does |
|---|---|
| `purpose index` | Build/refresh `.purpose/index.json` for the current project |
| `purpose index --root .` | Index only the current folder (not the whole git repo) |
| `purpose ask "<question>"` | Return the relevant `file:line` slice for a question |

The index covers *definitions* (functions, structs, classes, traits,
headings), not call sites, config values, or runtime behaviour. By default it
indexes the enclosing git root; pass `--root` to scope it. Full details are in
[`mechanistic-synthesis/synthesis.md`](mechanistic-synthesis/synthesis.md).

A related subcommand, `purpose ckg`, answers a different question the index
cannot: given a goal, which modules of a codebase are load-bearing for it, and
is that determination itself accountable, or contested. See
`purpose ckg --help` and [`mechanistic-synthesis/implementation/integration.md`](mechanistic-synthesis/implementation/integration.md).

---

## Theoretical Framework — `absicht`

[`absicht/docs/research-domain-specific-models`](absicht/docs/research-domain-specific-models)
contains *Accountable Compilation*, the paper the rest of this framework
implements. It treats three questions as one problem rather than three
independent engineering choices:

- **Acquisition.** A domain need only declare a typed operation vocabulary and
  a verification oracle. Training should occur exclusively at the domain's
  hardest, most complete task instances — its *extremal regime* — because
  competence there provably includes competence on every simpler restriction,
  while the converse fails and the failure gap is bounded away from zero (the
  **Inclusion Theorem**). This is why `purpose-factory` trains once, on a
  theme's full corpus, rather than per-restriction.
- **Composition.** Any finite collection of imperfect, bounded models has an
  aggregate error floor that decreases multiplicatively under federation (the
  **Federation Theorem**); the optimal allocation of a fixed inference budget
  across models is a solvable knapsack problem (the **Cascade Theorem**); and
  no chain of pairwise checks can certify an output below the floor of its
  least-checked member — only a minimal closed loop of three or more
  mutually-checking models can (the **Minimum Loop Theorem**).
- **Verification without disclosure.** Two systems with no shared internal
  representation can certify agreement without either exposing its internal
  state, by relaxing a four-part comparison — answer and provoked follow-up,
  for each side — to a fixed point (the **Quiescence Theorem**); an opaque
  pair whose answers agree but whose provoked follow-ups disagree is
  detectably unreliable, although comparing the answers alone would not show
  this (the **Route-Audit Theorem**). The paper's principal new result is that
  this verification step is *itself* a bounded receiver with its own
  composable error floor (the **Verification-Floor Theorem**) — so
  verification is not a step outside the compositional calculus, but an
  ordinary federation member, routable, budgetable, and further verifiable by
  the same machinery.

Nine computational experiments validate the framework's central claims against
seeded, reproducible simulations; results are stored as JSON records alongside
the manuscript. The paper is self-contained: every object is a finite weighted
graph, a bounded map between finite sets, or a convex/combinatorial
optimisation problem, and every theorem is proved from stated premises alone.

---

## `purpose-factory`: Acquisition Pipeline

[`purpose-factory`](mechanistic-synthesis/implementation/crates/purpose-factory)
is the Rust crate that operationalises the paper's acquisition pipeline: given
a named theme, it ingests sources, verifies and windows them into a training
corpus, trains a model, and exports it as a self-contained directory
(`model.safetensors`, `config.json`, `tokenizer.json`).

### Sources

A theme's sources are any combination of:

- **Local files** — `.tex`, `.pdf`, `.md`, `.txt`, `.csv`, `.json`. LaTeX is
  reduced to prose text; CSV rows and JSON documents are flattened to
  readable `key: value` text rather than fed in as raw structured data.
- **IMAP email** — a mailbox, a search filter, and a pre-obtained
  app-password or token (no OAuth consent flow is performed by the crate
  itself).
- **URLs** — fetched and reduced from HTML to text.

Every admitted (prompt, completion) pair passes a `Verifier` before it enters
the corpus — a default heuristic verifier (length and repetition checks)
ships for themes with no domain-specific oracle, and any caller embedding the
crate directly may supply its own.

### Training

Two paths, selected per theme:

- **From scratch.** A small GPT-2-style causal model, including a fresh
  word-level tokenizer trained on exactly the theme's admitted text. No
  external download; the whole pipeline stays network-free after source
  ingestion.
- **Pretrained (LoRA).** A real LLaMA-family checkpoint is downloaded from the
  HuggingFace Hub and LoRA-adapted: `q_proj`, `v_proj`, and `gate_proj` in
  every transformer block are trainable low-rank adapters over otherwise
  frozen pretrained weights. Because the upstream `candle-transformers` crate
  keeps its own LLaMA implementation's attention projections private, the
  architecture is implemented directly against public Candle primitives
  specifically so LoRA can be spliced into it. The merged export uses the
  checkpoint's own tensor names, so the result is a drop-in replacement for
  the original checkpoint in any HuggingFace-compatible loader. Only
  single-file (`model.safetensors`) checkpoints are supported.

Both paths train with AdamW, merge any LoRA adapters into plain weights
before export, and write a model directory that a `Provider` implementation
(per the integration contract below) can load directly.

### The `theme.toml` Contract

```toml
name = "my-theme"

[sources]
local_files = [{ root = "path/to/papers", extensions = ["tex", "pdf", "md"] }]
urls = ["https://example.com/article"]

# [[sources.imap]]
# host = "imap.gmail.com"
# username = "you@example.com"
# password_env = "THEME_IMAP_PASSWORD"

[model]
vocab_size = 8192
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256

# Uncomment to LoRA-adapt a pretrained checkpoint instead:
# [model.pretrained]
# repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

[training]
epochs = 3
batch_size = 8
learning_rate = 0.0003
lora_rank = 8
lora_alpha = 16.0
```

---

## The CLI

```bash
purpose factory init <name>          # scaffold a starter theme.toml
purpose factory build <theme.toml>   # fetch sources, train, export
purpose factory list                 # list built theme models from the local registry
```

`purpose factory build` records the result to a local JSON registry
(`.purpose/factory/registry.json`) so other tooling can discover what has
been produced without re-running the factory.

---

## `purpose serve`: The HTTP Boundary

```bash
PURPOSE_SERVE_TOKEN=<token> purpose serve --port 8420 --root <dir>
```

`purpose serve` exposes the same acquisition pipeline over HTTP, so a caller —
`profile-web` or any other client — need not run on the same machine or share
a filesystem with it. Every request must present the configured token as
`Authorization: Bearer <token>`; the server refuses to start if
`PURPOSE_SERVE_TOKEN` is unset, so there is no silent no-auth mode.

| Method & path | Behaviour |
|---|---|
| `POST /themes/{name}/sources` | Multipart upload of source files into the named theme |
| `POST /themes/{name}/build` | JSON body selecting model/training configuration; trains and returns the resulting model's metadata |
| `GET /themes/{name}/model/{file}` | Downloads one of `model.safetensors`, `config.json`, `tokenizer.json` |
| `GET /themes` | Lists built theme models |

This is groundwork toward a distributed deployment, not a distributed
deployment itself: the server is a single process with a static shared
secret, suited to a personal or small-team setup, not a public multi-tenant
service.

---

## `profile-web`: The Personal Knowledge Profile

[`profile-web`](mechanistic-synthesis/profile-web) is a Next.js application
presenting a theme as a personal "profile": create one or more named
profiles, add papers, CSVs, JSON files, and links to each, choose a model
(from scratch or a pretrained repo to LoRA-adapt), and train.

It runs against either transport `purpose-factory` supports:

- **Local**, by default — spawns the `purpose` binary as a subprocess and
  shares its filesystem, for a single machine running both the UI and the
  factory.
- **Remote**, when `PURPOSE_SERVE_URL` and `PURPOSE_SERVE_TOKEN` are set —
  uploads sources to, and triggers builds against, a `purpose serve`
  instance over the network.

```bash
cd mechanistic-synthesis/profile-web
npm install
npm run dev
```

---

## `purpose-factory-ts`

[`purpose-factory-ts`](mechanistic-synthesis/purpose-factory-ts)
(`@buhera/purpose-factory-client`) is the TypeScript client `profile-web`
consumes, and is usable directly by any other Node application. It exposes
both transports under namespaced exports:

```ts
import { cli, http, Registry } from "@buhera/purpose-factory-client";

// Local subprocess
const model = await cli.buildTheme("theme.toml", { out: "./out" });

// Remote server
const model = await http.buildTheme(
  "my-theme",
  { model: { kind: "scratch" } },
  { baseUrl: "http://localhost:8420", token: process.env.PURPOSE_SERVE_TOKEN! },
);
```

The package never loads or serves a model itself — that is left to whatever
inference stack the consuming application already uses.

---

## Workspace Layout

```
absicht/                                    Theoretical framework (Accountable Compilation)
  docs/research-domain-specific-models/      The paper, figures, and validation experiments
  web/                                       Landing page

mechanistic-synthesis/
  implementation/                            Rust workspace
    crates/purpose-core/                     vaHera AST, typed operations, the Resolver/Provider contract
    crates/purpose-operations/                Operation registry and executor
    crates/purpose-domains-*/                 Query domains (protein, codebase, ledger, ckg)
    crates/purpose-factory/                   Source ingestion, corpus, training, export, HTTP server
    crates/purpose-cli/                       The `purpose` binary
    integration.md                            The frozen integration contract for the Rust workspace
  purpose-ts/                                 @buhera/purpose — context-residue graph library
  purpose-factory-ts/                         @buhera/purpose-factory-client — TypeScript client for purpose-factory
  profile-web/                                Personal knowledge-profile frontend
```

---

## Building From Source

```bash
# Requires Rust (https://rustup.rs)
cd mechanistic-synthesis/implementation
cargo install --path crates/purpose-cli
# installs `purpose` into ~/.cargo/bin

purpose --help
```

```bash
# TypeScript packages
cd mechanistic-synthesis/purpose-factory-ts && npm install && npm run build
cd ../profile-web && npm install && npm run dev
```

---

## Scope and Open Problems

- Only single-shard (`model.safetensors`) pretrained checkpoints are
  supported; sharded checkpoints are not.
- `purpose serve` authenticates with one static shared token; it does not
  issue, rotate, or expire credentials, and is not intended as a
  multi-tenant service.
- Loading a produced model back into a query-answering `Resolver`/`Provider`
  pair — closing the loop from trained model to in-process inference — is a
  natural next integration, documented in
  [`integration.md`](mechanistic-synthesis/implementation/integration.md)
  §4.4, but not yet built.
- The paper's own stated open problems (`absicht`, §Discussion) — a
  characterisation of exactly which domains satisfy the non-expansive
  projection axiom, which follow-up policies make the route-audit maximally
  discriminating, and the optimal shape of a verification tree under a fixed
  budget — remain open.

The legacy Python implementation this repository originally shipped has been
superseded by the Rust framework described above and is no longer
maintained.

---

## License

MIT — see [`LICENSE`](LICENSE).
