//! HTTP boundary. Every route but `/health` requires `Authorization: Bearer
//! <CHIGUTIRO_TOKEN>`. No CORS: the token belongs in a server-side caller
//! (the host framework's backend or API routes), never in a browser.

use std::sync::{Arc, Mutex};

use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chigutiro_core::consolidate;
use chigutiro_core::{Erase, Scope};
use serde::Deserialize;
use serde_json::json;

use crate::answer::{extractive, Answer, Ollama};
use crate::service::Service;

#[derive(Clone)]
pub struct AppState {
    pub service: Arc<Mutex<Service>>,
    pub token: Arc<String>,
    pub generator: Option<Ollama>,
    pub auto_consolidate: bool,
}

pub fn app(state: AppState) -> Router {
    let protected = Router::new()
        .route("/status", get(status))
        .route("/ingest", post(ingest))
        .route("/ask", post(ask))
        .route("/erase", post(erase))
        .route("/consolidate", post(consolidate_now))
        .route("/consolidations", get(consolidations))
        .layer(middleware::from_fn_with_state(state.clone(), auth));
    Router::new()
        .route("/health", get(|| async { Json(json!({ "ok": true, "version": env!("CARGO_PKG_VERSION") })) }))
        .merge(protected)
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<chigutiro_core::Error> for ApiError {
    fn from(e: chigutiro_core::Error) -> Self {
        let code = match e {
            chigutiro_core::Error::InvalidRecord(_) => StatusCode::UNPROCESSABLE_ENTITY,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        ApiError(code, e.to_string())
    }
}

async fn auth(State(state): State<AppState>, req: Request, next: Next) -> Result<Response, ApiError> {
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or("");
    if !constant_time_eq(presented.as_bytes(), state.token.as_bytes()) {
        return Err(ApiError(StatusCode::UNAUTHORIZED, "missing or wrong bearer token".into()));
    }
    Ok(next.run(req).await)
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn lock(state: &AppState) -> std::sync::MutexGuard<'_, Service> {
    // A panic mid-request must not wedge every later request; the service's
    // own invariants are re-established from disk on the next restart.
    state.service.lock().unwrap_or_else(|p| p.into_inner())
}

async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let s = lock(&state);
    Json(serde_json::to_value(s.status()).expect("status serializes"))
}

#[derive(Deserialize)]
struct IngestBody {
    records: Vec<serde_json::Value>,
}

async fn ingest(State(state): State<AppState>, Json(body): Json<IngestBody>) -> Result<Json<serde_json::Value>, ApiError> {
    let report = lock(&state).ingest(body.records)?;
    if state.auto_consolidate && report.accepted > 0 {
        // "Not due" is the usual outcome here, not an error worth surfacing.
        let _ = spawn_consolidation(state.clone(), false);
    }
    Ok(Json(serde_json::to_value(report).expect("report serializes")))
}

#[derive(Deserialize)]
struct AskBody {
    query: String,
    #[serde(default)]
    budget: Option<usize>,
    #[serde(default = "yes")]
    generate: bool,
    /// `"work"` answers from the work scope only; omitted, from everything.
    #[serde(default)]
    scope: Option<Scope>,
}

fn yes() -> bool {
    true
}

async fn ask(State(state): State<AppState>, Json(body): Json<AskBody>) -> Result<Json<Answer>, ApiError> {
    if body.query.trim().is_empty() {
        return Err(ApiError(StatusCode::UNPROCESSABLE_ENTITY, "query is empty".into()));
    }
    if body.scope == Some(Scope::Personal) {
        return Err(ApiError(
            StatusCode::UNPROCESSABLE_ENTITY,
            "there is no personal-only view: pass scope \"work\", or omit scope to ask over everything".into(),
        ));
    }
    // Retrieve under the lock; phrase outside it, so a slow model never
    // blocks ingestion.
    let (retrieval, model_version, model_tainted) = {
        let s = lock(&state);
        let r = s.ask(&body.query, body.budget.map(|b| b.clamp(1, 64)), body.scope);
        (r, consolidate::latest_succeeded(s.state()).map(|c| c.version), consolidate::tainted(s.state()))
    };
    let mut answer = Answer {
        answer: extractive(&retrieval),
        generated: false,
        model: None,
        generation_error: None,
        retrieval,
        model_version,
        model_tainted,
    };
    if let (true, Some(g)) = (body.generate && !answer.retrieval.claims.is_empty(), &state.generator) {
        match g.phrase(&body.query, &answer.retrieval).await {
            Ok(text) => {
                answer.answer = text;
                answer.generated = true;
                answer.model = Some(g.model.clone());
            }
            Err(e) => answer.generation_error = Some(e),
        }
    }
    Ok(Json(answer))
}

async fn erase(State(state): State<AppState>, Json(criteria): Json<Erase>) -> Result<Json<serde_json::Value>, ApiError> {
    let report = lock(&state).erase(&criteria)?;
    Ok(Json(serde_json::to_value(report).expect("report serializes")))
}

#[derive(Deserialize, Default)]
struct ConsolidateBody {
    #[serde(default)]
    force: bool,
}

async fn consolidate_now(State(state): State<AppState>, body: Option<Json<ConsolidateBody>>) -> Response {
    let force = body.map(|b| b.0.force).unwrap_or_default();
    match spawn_consolidation(state, force) {
        Ok(version) => (StatusCode::ACCEPTED, Json(json!({ "started": true, "version": version }))).into_response(),
        Err(reason) => (StatusCode::CONFLICT, Json(json!({ "started": false, "reason": reason }))).into_response(),
    }
}

async fn consolidations(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(lock(&state).consolidations()).expect("serializes"))
}

/// Starts a round in the background if one may start. Training holds no
/// lock; only preparing and recording the outcome do.
pub fn spawn_consolidation(state: AppState, force: bool) -> Result<u64, String> {
    let (plan, bin) = {
        let mut s = lock(&state);
        let plan = s.begin_consolidation(force).map_err(|e| e.to_string())??;
        (plan, s.purpose_bin().expect("checked by begin_consolidation"))
    };
    let version = plan.version;
    tokio::task::spawn_blocking(move || {
        tracing::info!(version, "consolidation started");
        let result = consolidate::run(&plan, &bin);
        match &result {
            Ok(()) => tracing::info!(version, "consolidation succeeded"),
            Err(e) => tracing::warn!(version, "consolidation failed: {e}"),
        }
        if let Err(e) = lock(&state).end_consolidation(version, result) {
            tracing::error!(version, "could not record consolidation outcome: {e}");
        }
    });
    Ok(version)
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::Request;
    use chigutiro_core::consolidate::ConsolidationConfig;
    use chigutiro_core::EngineConfig;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    fn test_app(dir: &std::path::Path) -> Router {
        let service = Service::open(dir, None, EngineConfig::default(), ConsolidationConfig::default(), false).unwrap();
        app(AppState {
            service: Arc::new(Mutex::new(service)),
            token: Arc::new("secret".into()),
            generator: None,
            auto_consolidate: false,
        })
    }

    async fn call(app: &Router, method: &str, path: &str, token: Option<&str>, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let mut req = Request::builder().method(method).uri(path).header("content-type", "application/json");
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {t}"));
        }
        let resp = app.clone().oneshot(req.body(Body::from(body.to_string())).unwrap()).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
    }

    #[tokio::test]
    async fn the_work_scope_is_a_separate_view() {
        let dir = tempfile::tempdir().unwrap();
        let app = test_app(dir.path());
        let today = chrono::Utc::now().format("%Y-%m-%dT06:00:00Z").to_string();
        let (_, report) = call(&app, "POST", "/ingest", Some("secret"), json!({ "records": [
            { "source": "upload:lab-report", "ts": today, "kind": "prose", "title": "Run 14",
              "text": "Transaminase run 14: conversion stalled at 40% without added PLP." },
            { "source": "gmail", "ts": today, "kind": "prose",
              "text": "Transaminase pun for the birthday card, and the flat viewing on Friday." },
        ]})).await;
        assert_eq!(report["accepted"], 2);

        let (_, all) = call(&app, "POST", "/ask", Some("secret"), json!({ "query": "transaminase" })).await;
        let (_, work) = call(&app, "POST", "/ask", Some("secret"), json!({ "query": "transaminase", "scope": "work" })).await;
        assert_eq!(all["claims"].as_array().unwrap().len(), 2);
        assert_eq!(work["claims"].as_array().unwrap().len(), 1, "{work}");
        assert!(!work.to_string().contains("birthday"));

        let (code, _) = call(&app, "POST", "/ask", Some("secret"), json!({ "query": "x", "scope": "personal" })).await;
        assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
        let (_, status) = call(&app, "GET", "/status", Some("secret"), json!(null)).await;
        assert_eq!(status["stats"]["work_records"], 1);
    }

    #[tokio::test]
    async fn ingest_ask_erase_over_http() {
        let dir = tempfile::tempdir().unwrap();
        let app = test_app(dir.path());

        let (code, _) = call(&app, "GET", "/status", None, json!(null)).await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);
        let (code, _) = call(&app, "GET", "/status", Some("wrong"), json!(null)).await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);

        let today = chrono::Utc::now().format("%Y-%m-%dT06:00:00Z").to_string();
        let (code, report) = call(&app, "POST", "/ingest", Some("secret"), json!({ "records": [
            { "source": "garmin", "ts": today, "kind": "measurement", "metric": "HRV", "value": 61, "unit": "ms" },
            { "source": "gmail", "id": "m1", "ts": today, "kind": "prose", "subject": "boss@uni.de",
              "text": "The enzyme screen review moves to Thursday." },
            { "source": "broken", "ts": "not-a-date", "kind": "measurement", "metric": "x", "value": 1 },
        ]})).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(report["accepted"], 2);
        assert_eq!(report["rejected"][0]["index"], 2);

        let (_, ans) = call(&app, "POST", "/ask", Some("secret"), json!({ "query": "hrv" })).await;
        assert_eq!(ans["grade"], "single_sourced");
        assert_eq!(ans["generated"], false);
        assert!(ans["answer"].as_str().unwrap().contains("hrv: latest 61.0 ms"), "{ans}");
        // The route log holds shape only.
        let routes = std::fs::read_to_string(dir.path().join("routes.log")).unwrap();
        assert!(!routes.contains("hrv") && routes.contains("\"receiver\":\"series\""), "{routes}");

        let (_, er) = call(&app, "POST", "/erase", Some("secret"), json!({ "subject": "boss@uni.de" })).await;
        assert_eq!(er["removed"], 1);
        let (_, ans) = call(&app, "POST", "/ask", Some("secret"), json!({ "query": "enzyme screen review" })).await;
        assert_eq!(ans["grade"], "declined");
        let on_disk = std::fs::read_to_string(dir.path().join("records.log")).unwrap();
        assert!(!on_disk.contains("enzyme"), "erased record still on disk");

        let (code, c) = call(&app, "POST", "/consolidate", Some("secret"), json!({})).await;
        assert_eq!(code, StatusCode::CONFLICT);
        assert!(c["reason"].as_str().unwrap().contains("CHIGUTIRO_PURPOSE_BIN"));
    }
}
