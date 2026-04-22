# Purpose implementation — MVP

Rust workspace implementing the runtime side of the Purpose Model Factory.

## Crates

- `purpose-core` — durable types: vaHera AST, `Domain`, `Resolver` trait, `Operation`, `Type`, `Value`, type-checker.
- `purpose-operations` — `Provider` trait, `OperationRegistry`, vaHera `Executor`, built-in providers (UniProt HTTP, deterministic protein summariser).
- `purpose-domains-protein` — protein Domain Connector with hand-coded resolver.
- `purpose-cli` — the `purpose` binary.

## Build

Requires Rust 1.75+ (install via <https://rustup.rs>).

```
cargo build --release
```

## Run

```
# End-to-end: hit UniProt, format a summary
cargo run -p purpose-cli -- query "Tell me about SOD1"

# Compile only, inspect the vaHera fragment
cargo run -p purpose-cli -- query "What is TP53?" --dry-run

# Raw JSON response
cargo run -p purpose-cli -- query "Tell me about BRCA1" --raw

# List registered operations and their signatures
cargo run -p purpose-cli -- operations
```

## Current scope

One domain (protein). One compilation pattern (`lookup_protein_by_gene |> summarize_protein`). Two providers (UniProt REST, deterministic summariser). The resolver is hand-coded — regex-based gene-symbol extraction, templated vaHera emission.

Every interface in the workspace is the one the full framework uses. A LoRA-trained resolver produced by the factory later swaps into the same `Resolver` trait; every other crate sees no change. Additional domains (chemical compound, metabolite, …) plug in by registering their own `Domain` and providers against the same `OperationRegistry`.

## What is stubbed or absent

- The training factory (`purpose-factory`).
- The Aperture Foundation Model.
- Cascade registry routing (single domain at present).
- The kernel subsystems (CMM / PSS / PVE / TEM / DIC) — the CLI dispatches directly.
- The interceptor's presentation state machine — CLI read-eval-print is the current surface.
- HuggingFace inference provider — interface is designed for it; concrete adapter to be added next.
