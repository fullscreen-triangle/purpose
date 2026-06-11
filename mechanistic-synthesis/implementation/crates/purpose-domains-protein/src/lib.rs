//! Protein compilation domain.
//!
//! Provides:
//! - A `Domain` struct with the protein operation vocabulary.
//! - A hand-coded resolver that compiles simple natural-language queries
//!   about proteins into vaHera fragments (`lookup_protein_by_gene |>
//!   summarize_protein`).
//! - A function to register UniProt and summary providers against the
//!   operation registry.
//!
//! In a later stage this resolver is replaced by a trained LoRA adapter;
//! all other components stay unchanged.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use tracing::debug;

use purpose_core::{Domain, Error, Operation, Resolver, Type, VaHera, Value};
use purpose_operations::providers::{ProteinSummaryProvider, UniprotProvider};
use purpose_operations::OperationRegistry;

/// Hand-coded resolver: extract a gene symbol from the utterance and emit
/// a composition that looks up then summarises the record.
pub struct ProteinResolver;

#[async_trait]
impl Resolver for ProteinResolver {
    async fn compile(&self, utterance: &str) -> Result<VaHera, Error> {
        let gene = extract_gene_symbol(utterance).ok_or_else(|| {
            Error::Compile(
                "no gene/protein identifier recognised in utterance".into(),
            )
        })?;
        debug!(gene = %gene, "protein resolver extracted symbol");

        let mut lookup_args: BTreeMap<String, VaHera> = BTreeMap::new();
        lookup_args.insert(
            "gene".to_string(),
            VaHera::Literal(Value::Str(gene)),
        );

        let summarize_args: BTreeMap<String, VaHera> = BTreeMap::new();

        Ok(VaHera::Compose(vec![
            VaHera::Call {
                op: "lookup_protein_by_gene".to_string(),
                args: lookup_args,
            },
            VaHera::Call {
                op: "summarize_protein".to_string(),
                args: summarize_args,
            },
        ]))
    }
}

/// Extract a candidate gene/protein symbol from the utterance. Matches
/// uppercase identifiers of 2–7 characters (optionally with digits). A
/// small stopword list prevents obvious false positives.
fn extract_gene_symbol(utterance: &str) -> Option<String> {
    let re = Regex::new(r"\b([A-Z][A-Z0-9]{1,6})\b").ok()?;
    let symbol = re
        .captures_iter(utterance)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .find(|s| !is_stopword(s));
    symbol
}

fn is_stopword(s: &str) -> bool {
    matches!(
        s,
        "I" | "A"
            | "AN"
            | "THE"
            | "IS"
            | "ARE"
            | "ABOUT"
            | "ME"
            | "TELL"
            | "WHAT"
            | "WHO"
            | "WHERE"
            | "HOW"
            | "WHY"
            | "WHEN"
            | "DO"
            | "DOES"
            | "CAN"
            | "COULD"
            | "WILL"
            | "WOULD"
            | "SHOULD"
            | "PROTEIN"
            | "GENE"
            | "QUERY"
            | "INFO"
    )
}

/// Build the protein domain connector.
pub fn domain() -> Domain {
    Domain {
        name: "protein".into(),
        operations: operations(),
        resolver: Arc::new(ProteinResolver),
    }
}

/// Operation vocabulary for the protein domain.
pub fn operations() -> Vec<Operation> {
    vec![
        Operation::new(
            "lookup_protein_by_gene",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("gene".into(), Type::Str);
                inputs.insert("organism".into(), Type::Str);
                inputs
            },
            Type::named("ProteinRecord"),
            "Look up a reviewed protein entry by gene symbol and organism (default human).",
        ),
        Operation::new(
            "lookup_uniprot_record",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("accession".into(), Type::Str);
                inputs
            },
            Type::named("ProteinRecord"),
            "Fetch a UniProt record by primary accession.",
        ),
        Operation::new(
            "summarize_protein",
            {
                let mut inputs = BTreeMap::new();
                inputs.insert("input".into(), Type::named("ProteinRecord"));
                inputs
            },
            Type::Str,
            "Produce a human-readable summary of a protein record.",
        ),
    ]
}

/// Register the providers that serve this domain's operations.
///
/// The UniProt provider serves the two lookup operations; the
/// summarisation provider serves the summarise operation.
pub fn register_providers(registry: &mut OperationRegistry) {
    let uniprot = Arc::new(UniprotProvider::new());
    let summary = Arc::new(ProteinSummaryProvider);

    for op in operations() {
        match op.name.as_str() {
            "lookup_protein_by_gene" | "lookup_uniprot_record" => {
                registry.register(op, uniprot.clone());
            }
            "summarize_protein" => {
                registry.register(op, summary.clone());
            }
            _ => {}
        }
    }
}
