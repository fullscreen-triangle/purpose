//! Turning graded claims into a reply. The generator phrases; it does not
//! know anything the claims do not say, and it is told so. Without a
//! generator — or when it fails — the reply is the claims themselves.

use std::time::Duration;

use chigutiro_core::{Grade, Retrieval};
use serde::Serialize;
use serde_json::json;

#[derive(Debug, Serialize)]
pub struct Answer {
    pub answer: String,
    pub generated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_error: Option<String>,
    #[serde(flatten)]
    pub retrieval: Retrieval,
    /// Latest consolidated voice model, and whether it may encode erased data.
    pub model_version: Option<u64>,
    pub model_tainted: bool,
}

pub fn extractive(r: &Retrieval) -> String {
    if r.claims.is_empty() {
        return "Nothing I hold answers that.".into();
    }
    r.claims
        .iter()
        .map(|c| format!("- [{}] {}", grade_label(c.grade), c.text))
        .collect::<Vec<_>>()
        .join("\n")
}

fn grade_label(g: Grade) -> &'static str {
    match g {
        Grade::Grounded => "grounded",
        Grade::TwoSourced => "two sources",
        Grade::SingleSourced => "one source",
        Grade::Contested => "contested",
        Grade::Declined => "declined",
    }
}

#[derive(Clone)]
pub struct Ollama {
    pub url: String,
    pub model: String,
    pub owner: String,
    client: reqwest::Client,
}

impl Ollama {
    pub fn new(url: String, model: String, owner: String) -> Ollama {
        let client = reqwest::Client::builder().timeout(Duration::from_secs(90)).build().expect("reqwest client");
        Ollama { url: url.trim_end_matches('/').to_string(), model, owner, client }
    }

    pub async fn phrase(&self, query: &str, r: &Retrieval) -> Result<String, String> {
        let system = format!(
            "You are {owner}'s personal model, answering {owner} directly. Answer ONLY from the numbered \
             claims below; they are everything you know. Each claim is tagged with how many independent \
             sources support it: say so when an answer rests on one source, and never smooth over a \
             contested claim — name the disagreement. If the claims do not answer the question, say that \
             plainly. Never invent or estimate a number, date, name, or amount. Be brief.\n\nClaims:\n{claims}",
            owner = self.owner,
            claims = r
                .claims
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{}. [{}; sources: {}] {}", i + 1, grade_label(c.grade), c.sources.join(", "), c.text))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let body = json!({
            "model": self.model,
            "stream": false,
            "options": { "temperature": 0.2 },
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": query },
            ],
        });
        let resp = self
            .client
            .post(format!("{}/api/chat", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("ollama unreachable at {}: {e}", self.url))?;
        if !resp.status().is_success() {
            return Err(format!("ollama returned {}", resp.status()));
        }
        let v: serde_json::Value = resp.json().await.map_err(|e| format!("ollama reply: {e}"))?;
        v["message"]["content"]
            .as_str()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "ollama reply had no message content".into())
    }
}
