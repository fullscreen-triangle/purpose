use async_trait::async_trait;
use purpose_core::{Error, Value};
use std::collections::BTreeMap;

use crate::provider::Provider;

/// Deterministic formatter for UniProt records. Produces a short
/// human-readable summary from a structured response.
///
/// In a later stage this is replaced by a small learned summarisation
/// resolver invoked through the same Provider interface.
pub struct ProteinSummaryProvider;

#[async_trait]
impl Provider for ProteinSummaryProvider {
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error> {
        match op {
            "summarize_protein" => {
                let input = args
                    .get("input")
                    .ok_or_else(|| Error::Provider(
                        "summarize_protein: no 'input' value".into(),
                    ))?;
                Ok(Value::Str(format_protein_summary(input)))
            }
            _ => Err(Error::Provider(format!(
                "summary provider does not handle op: '{}'",
                op
            ))),
        }
    }
}

fn format_protein_summary(v: &Value) -> String {
    let Value::Record(rec) = v else {
        return format!("Unexpected value shape: {:?}", v);
    };

    // UniProt search returns `{"results": [...]}`; a direct accession
    // lookup returns the record directly.
    let protein: &BTreeMap<String, Value> = if let Some(results) = rec.get("results") {
        match results {
            Value::List(items) => match items.first() {
                Some(Value::Record(r)) => r,
                _ => return "No protein matched the query.".to_string(),
            },
            _ => return "Malformed UniProt search response.".to_string(),
        }
    } else {
        rec
    };

    let mut out = String::new();
    let mut push = |k: &str, v: &str| {
        if !v.is_empty() {
            out.push_str(&format!("{}: {}\n", k, v));
        }
    };

    if let Some(Value::Str(acc)) = protein.get("primaryAccession") {
        push("Accession", acc);
    }

    if let Some(Value::Str(name)) = protein.get("uniProtkbId") {
        push("ID", name);
    }

    if let Some(Value::Record(pd)) = protein.get("proteinDescription") {
        if let Some(full_name) = pd
            .get("recommendedName")
            .and_then(|v| v.as_record())
            .and_then(|r| r.get("fullName"))
            .and_then(|v| v.as_record())
            .and_then(|r| r.get("value"))
            .and_then(|v| v.as_str())
        {
            push("Name", full_name);
        }
    }

    if let Some(Value::Record(org)) = protein.get("organism") {
        if let Some(Value::Str(sci)) = org.get("scientificName") {
            push("Organism", sci);
        }
    }

    if let Some(Value::List(genes)) = protein.get("genes") {
        let names: Vec<String> = genes
            .iter()
            .filter_map(|g| {
                g.as_record()
                    .and_then(|r| r.get("geneName"))
                    .and_then(|v| v.as_record())
                    .and_then(|r| r.get("value"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect();
        if !names.is_empty() {
            push("Gene(s)", &names.join(", "));
        }
    }

    if let Some(Value::Record(seq)) = protein.get("sequence") {
        if let Some(Value::Num(len)) = seq.get("length") {
            push("Length", &format!("{} aa", *len as i64));
        }
        if let Some(Value::Num(mass)) = seq.get("molWeight") {
            push("Molecular weight", &format!("{:.0} Da", mass));
        }
    }

    // FUNCTION comments
    if let Some(Value::List(comments)) = protein.get("comments") {
        for c in comments {
            let Value::Record(cr) = c else { continue };
            let ct = cr.get("commentType").and_then(|v| v.as_str()).unwrap_or("");
            match ct {
                "FUNCTION" => {
                    if let Some(Value::List(texts)) = cr.get("texts") {
                        if let Some(desc) = texts
                            .first()
                            .and_then(|t| t.as_record())
                            .and_then(|r| r.get("value"))
                            .and_then(|v| v.as_str())
                        {
                            push("Function", desc);
                        }
                    }
                }
                "SUBCELLULAR LOCATION" => {
                    if let Some(Value::List(locs)) = cr.get("subcellularLocations") {
                        let parts: Vec<String> = locs
                            .iter()
                            .filter_map(|loc| {
                                loc.as_record()
                                    .and_then(|r| r.get("location"))
                                    .and_then(|v| v.as_record())
                                    .and_then(|r| r.get("value"))
                                    .and_then(|v| v.as_str())
                                    .map(str::to_string)
                            })
                            .collect();
                        if !parts.is_empty() {
                            push("Subcellular location", &parts.join("; "));
                        }
                    }
                }
                "DISEASE" => {
                    if let Some(Value::Record(d)) = cr.get("disease") {
                        if let Some(Value::Str(id)) = d.get("diseaseId") {
                            push("Associated disease", id);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if out.is_empty() {
        out = "No summary information available in record.".to_string();
    }
    out
}
