use async_trait::async_trait;
use purpose_core::{Error, Value};
use reqwest::Client;
use std::collections::BTreeMap;
use std::time::Duration;
use tracing::{debug, warn};

use crate::provider::Provider;

/// UniProt REST client. Serves `lookup_protein_by_gene` and
/// `lookup_uniprot_record`.
pub struct UniprotProvider {
    client: Client,
    base_url: String,
}

impl UniprotProvider {
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(20))
            .user_agent("purpose-cli/0.1")
            .build()
            .expect("failed to build reqwest client");
        Self {
            client,
            base_url: "https://rest.uniprot.org/uniprotkb".to_string(),
        }
    }

    pub fn with_base_url<S: Into<String>>(mut self, url: S) -> Self {
        self.base_url = url.into();
        self
    }

    async fn get_json(&self, url: &str) -> Result<serde_json::Value, Error> {
        debug!(url = %url, "uniprot GET");
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| Error::Provider(format!("uniprot request failed: {}", e)))?;

        if !resp.status().is_success() {
            warn!(status = %resp.status(), "uniprot non-success");
            return Err(Error::Provider(format!(
                "uniprot returned status {}",
                resp.status()
            )));
        }

        resp.json()
            .await
            .map_err(|e| Error::Provider(format!("uniprot json parse: {}", e)))
    }
}

impl Default for UniprotProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for UniprotProvider {
    async fn invoke(
        &self,
        op: &str,
        args: &BTreeMap<String, Value>,
    ) -> Result<Value, Error> {
        match op {
            "lookup_protein_by_gene" => {
                let gene = args
                    .get("gene")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        Error::Provider(
                            "lookup_protein_by_gene: missing 'gene' argument".into(),
                        )
                    })?
                    .to_string();
                let organism = args
                    .get("organism")
                    .and_then(|v| v.as_str())
                    .unwrap_or("9606")
                    .to_string();

                let query = format!(
                    "gene_exact:{} AND organism_id:{} AND reviewed:true",
                    gene, organism
                );
                let url = format!(
                    "{}/search?query={}&format=json&size=1",
                    self.base_url,
                    urlencoding::encode(&query)
                );
                let body = self.get_json(&url).await?;
                Ok(value_from_json(&body))
            }

            "lookup_uniprot_record" => {
                let accession = args
                    .get("accession")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        Error::Provider(
                            "lookup_uniprot_record: missing 'accession'".into(),
                        )
                    })?
                    .to_string();

                let url = format!("{}/{}.json", self.base_url, accession);
                let body = self.get_json(&url).await?;
                Ok(value_from_json(&body))
            }

            _ => Err(Error::Provider(format!(
                "uniprot does not handle op: '{}'",
                op
            ))),
        }
    }
}

/// Convert a serde_json::Value into our Value enum.
fn value_from_json(j: &serde_json::Value) -> Value {
    match j {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => Value::Num(n.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(s) => Value::Str(s.clone()),
        serde_json::Value::Array(a) => {
            Value::List(a.iter().map(value_from_json).collect())
        }
        serde_json::Value::Object(o) => {
            let mut m: BTreeMap<String, Value> = BTreeMap::new();
            for (k, v) in o {
                m.insert(k.clone(), value_from_json(v));
            }
            Value::Record(m)
        }
    }
}
