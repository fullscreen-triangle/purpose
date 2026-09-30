//! `purpose serve`: an axum HTTP server exposing the theme factory, gated by
//! a static bearer token. It is the local end of Kundai's model platform: a
//! browser page (the mechanistic-synthesis site) calls it on `127.0.0.1` to
//! plan where to train, upload material, run build jobs and watch them, and
//! fetch the results. Every theme lives under the `Workspace` layout;
//! callers never see or choose a local path.

use std::path::Path;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tower_http::cors::{AllowOrigin, AllowPrivateNetwork, CorsLayer};

use crate::error::Error;
use crate::factory::{Factory, Registry, ThemeModel};
use crate::jobs::{BuildRequest, Job, Jobs};
use crate::placement::{self, Catalog, Plan, PlanRequest};
use crate::workspace::{validate_theme_name, Workspace};

/// Extensions `LocalFileSource` knows how to ingest — kept in sync by hand
/// with `LocalFileSource::new`'s default list (source.rs), since that
/// default is a `Vec` built at construction time, not a `const` this module
/// can reference directly.
const SUPPORTED_EXTENSIONS: &[&str] = &["tex", "pdf", "md", "txt", "csv", "json"];

const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;
/// A multipart request may carry several files, each up to `MAX_UPLOAD_BYTES`.
const MAX_REQUEST_BYTES: usize = 8 * MAX_UPLOAD_BYTES;
const MODEL_FILES: &[&str] = &["model.safetensors", "config.json", "tokenizer.json"];

#[derive(Clone)]
struct ServerState {
    ws: Workspace,
    token: Arc<String>,
    jobs: Jobs,
}

/// Builds the router. `root` is the server's working directory (created if
/// missing); `token` must be presented as `Authorization: Bearer <token>` on
/// every request; `allowed_origins` are the web origins (e.g. the platform
/// site) a browser may call from. Preflight requests are answered before
/// authentication, including Chrome's private-network preflight, so a page
/// served over HTTPS can reach this server on `127.0.0.1`.
pub fn app(root: impl AsRef<Path>, token: String, allowed_origins: &[String]) -> Result<Router, Error> {
    let ws = Workspace::new(root.as_ref());
    std::fs::create_dir_all(ws.root())?;
    let state = ServerState {
        jobs: Jobs::open(ws.clone())?,
        ws,
        token: Arc::new(token),
    };

    let origins: Vec<HeaderValue> = allowed_origins
        .iter()
        .filter_map(|o| HeaderValue::from_str(o.trim().trim_end_matches('/')).ok())
        .collect();
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([header::AUTHORIZATION, header::CONTENT_TYPE])
        .allow_private_network(AllowPrivateNetwork::yes());

    Ok(Router::new()
        .route("/health", get(health))
        .route("/plan", post(plan))
        .route("/places", get(places))
        .route("/themes", get(list_themes))
        .route("/themes/{name}/sources", get(list_sources).post(upload_sources))
        .route("/themes/{name}/build", post(build_theme))
        .route("/themes/{name}/model/{file}", get(download_model_file))
        .route("/jobs", get(list_jobs).post(submit_job))
        .route("/jobs/{id}", get(get_job))
        .route("/jobs/{id}/log", get(job_log))
        .route("/jobs/{id}/cancel", post(cancel_job))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        // Outermost, so preflights are answered without a token.
        .layer(cors)
        .with_state(state))
}

async fn auth(
    State(state): State<ServerState>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let presented = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match presented {
        Some(t) if constant_time_eq(t.as_bytes(), state.token.as_bytes()) => next.run(request).await,
        _ => (StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response(),
    }
}

/// Compares without an early exit, so response timing does not reveal how
/// much of a guessed token was right.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match &self {
            Error::Source(_) | Error::Corpus(_) | Error::Config(_) => StatusCode::BAD_REQUEST,
            Error::Train(_) | Error::Export(_) | Error::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Error::Cancelled => StatusCode::CONFLICT,
        };
        (status, Json(serde_json::json!({ "error": self.to_string() }))).into_response()
    }
}

// =====================================================================
// GET /health, POST /plan, GET /places
// =====================================================================

async fn health(State(state): State<ServerState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "ok": true,
        "version": env!("CARGO_PKG_VERSION"),
        "job_running": state.jobs.is_running(),
    }))
}

async fn plan(State(state): State<ServerState>, Json(req): Json<PlanRequest>) -> Result<Json<Plan>, Error> {
    let catalog = Catalog::load(&state.ws.places_path())?;
    Ok(Json(placement::plan(&req, &catalog)))
}

async fn places(State(state): State<ServerState>) -> Result<Json<Catalog>, Error> {
    Ok(Json(Catalog::load(&state.ws.places_path())?))
}

// =====================================================================
// GET /themes
// =====================================================================

async fn list_themes(State(state): State<ServerState>) -> Result<Json<Vec<ThemeModel>>, Error> {
    let models = Registry::new(state.ws.registry_path()).load()?;
    Ok(Json(models))
}

// =====================================================================
// GET, POST /themes/{name}/sources
// =====================================================================

#[derive(Serialize)]
struct SourceFile {
    filename: String,
    bytes: u64,
}

async fn list_sources(
    State(state): State<ServerState>,
    AxumPath(name): AxumPath<String>,
) -> Result<Json<Vec<SourceFile>>, Error> {
    validate_theme_name(&name)?;
    let mut files = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(state.ws.sources_dir(&name)).await {
        while let Some(entry) = entries.next_entry().await.map_err(Error::Io)? {
            let meta = entry.metadata().await.map_err(Error::Io)?;
            if meta.is_file() {
                files.push(SourceFile {
                    filename: entry.file_name().to_string_lossy().into_owned(),
                    bytes: meta.len(),
                });
            }
        }
    }
    files.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(Json(files))
}

#[derive(Serialize)]
struct UploadResponse {
    accepted: Vec<String>,
    rejected: Vec<RejectedFile>,
}

#[derive(Serialize)]
struct RejectedFile {
    filename: String,
    reason: String,
}

async fn upload_sources(
    State(state): State<ServerState>,
    AxumPath(name): AxumPath<String>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, Error> {
    validate_theme_name(&name)?;
    let dir = state.ws.sources_dir(&name);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(Error::Io)?;

    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| Error::Source(format!("multipart read: {e}")))?
    {
        let Some(filename) = field.file_name().map(|s| s.to_string()) else {
            continue;
        };
        let base = sanitize_filename(&filename);
        let Some(base) = base else {
            rejected.push(RejectedFile {
                filename,
                reason: "invalid filename".into(),
            });
            continue;
        };

        if !has_supported_extension(&base) {
            rejected.push(RejectedFile {
                filename: base,
                reason: format!(
                    "unsupported extension (supported: {})",
                    SUPPORTED_EXTENSIONS.join(", ")
                ),
            });
            continue;
        }

        let bytes = field
            .bytes()
            .await
            .map_err(|e| Error::Source(format!("multipart body: {e}")))?;
        if bytes.len() > MAX_UPLOAD_BYTES {
            rejected.push(RejectedFile {
                filename: base,
                reason: format!("exceeds {}MB limit", MAX_UPLOAD_BYTES / (1024 * 1024)),
            });
            continue;
        }

        tokio::fs::write(dir.join(&base), &bytes)
            .await
            .map_err(Error::Io)?;
        accepted.push(base);
    }

    Ok(Json(UploadResponse { accepted, rejected }))
}

fn sanitize_filename(name: &str) -> Option<String> {
    let base = Path::new(name.replace('\\', "/").as_str())
        .file_name()?
        .to_str()?
        .to_string();
    if base.is_empty() || base == "." || base == ".." {
        return None;
    }
    Some(base)
}

fn has_supported_extension(filename: &str) -> bool {
    filename
        .rsplit('.')
        .next()
        .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

// =====================================================================
// POST /themes/{name}/build   (blocking; kept for the TypeScript client)
// =====================================================================

async fn build_theme(
    State(state): State<ServerState>,
    AxumPath(name): AxumPath<String>,
    body: Option<Json<BuildRequest>>,
) -> Result<Json<ThemeModel>, Error> {
    let request = body.map(|Json(b)| b).unwrap_or_default();
    let contract = request.contract(&state.ws, &name)?;
    let model = Factory::build(contract, &state.ws.model_dir(&name)).await?;
    Registry::new(state.ws.registry_path()).record(&model)?;
    Ok(Json(model))
}

// =====================================================================
// /jobs
// =====================================================================

#[derive(Deserialize)]
struct JobRequest {
    theme: String,
    #[serde(flatten)]
    build: BuildRequest,
}

async fn submit_job(State(state): State<ServerState>, Json(req): Json<JobRequest>) -> Result<Json<Job>, Error> {
    Ok(Json(state.jobs.submit(&req.theme, req.build)?))
}

async fn list_jobs(State(state): State<ServerState>) -> Json<Vec<Job>> {
    Json(state.jobs.list())
}

fn no_such_job() -> Response {
    (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "no such job" }))).into_response()
}

async fn get_job(State(state): State<ServerState>, AxumPath(id): AxumPath<String>) -> Response {
    match state.jobs.get(&id) {
        Some(job) => Json(job).into_response(),
        None => no_such_job(),
    }
}

#[derive(Deserialize)]
struct LogQuery {
    tail: Option<usize>,
}

async fn job_log(
    State(state): State<ServerState>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<LogQuery>,
) -> Response {
    match state.jobs.log(&id, q.tail.unwrap_or(200).min(5000)) {
        Some(lines) => Json(lines).into_response(),
        None => no_such_job(),
    }
}

async fn cancel_job(State(state): State<ServerState>, AxumPath(id): AxumPath<String>) -> Response {
    match state.jobs.cancel(&id) {
        Some(job) => Json(job).into_response(),
        None => no_such_job(),
    }
}

// =====================================================================
// GET /themes/{name}/model/{file}
// =====================================================================

async fn download_model_file(
    State(state): State<ServerState>,
    AxumPath((name, file)): AxumPath<(String, String)>,
) -> Result<Response, Error> {
    validate_theme_name(&name)?;
    if !MODEL_FILES.contains(&file.as_str()) {
        return Ok((StatusCode::NOT_FOUND, "no such model file").into_response());
    }

    let path = state.ws.model_dir(&name).join(&file);
    match tokio::fs::read(&path).await {
        Ok(bytes) => Ok(bytes.into_response()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Ok((StatusCode::NOT_FOUND, "theme not built yet").into_response())
        }
        Err(e) => Err(Error::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    const ORIGIN: &str = "https://platform.example";

    fn router(tag: &str) -> (Router, std::path::PathBuf) {
        let root = std::env::temp_dir().join(format!("pf-server-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        (app(&root, "secret-token".into(), &[ORIGIN.to_string()]).unwrap(), root)
    }

    fn preflight(origin: &str) -> Request<Body> {
        Request::builder()
            .method("OPTIONS")
            .uri("/health")
            .header("origin", origin)
            .header("access-control-request-method", "GET")
            .header("access-control-request-headers", "authorization")
            .header("access-control-request-private-network", "true")
            .body(Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn preflight_passes_without_a_token_and_allows_the_private_network() {
        let (app, root) = router("preflight");
        let res = app.oneshot(preflight(ORIGIN)).await.unwrap();
        assert!(res.status().is_success(), "{}", res.status());
        let h = res.headers();
        assert_eq!(h["access-control-allow-origin"], ORIGIN);
        assert_eq!(h["access-control-allow-private-network"], "true");
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn other_origins_get_no_cors_grant() {
        let (app, root) = router("origin");
        let res = app.oneshot(preflight("https://evil.example")).await.unwrap();
        assert!(res.headers().get("access-control-allow-origin").is_none());
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn requests_need_the_right_token() {
        let (app, root) = router("token");
        let get = |token: &str| {
            Request::builder()
                .uri("/health")
                .header("authorization", format!("Bearer {token}"))
                .body(Body::empty())
                .unwrap()
        };
        let bad = app.clone().oneshot(get("secret-tokex")).await.unwrap();
        assert_eq!(bad.status(), StatusCode::UNAUTHORIZED);
        let ok = app.clone().oneshot(get("secret-token")).await.unwrap();
        assert_eq!(ok.status(), StatusCode::OK);

        let plan = Request::builder()
            .method("POST")
            .uri("/plan")
            .header("authorization", "Bearer secret-token")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"params_b":0.5,"tokens":100000,"privacy":"private","goal":"vocabulary"}"#))
            .unwrap();
        let res = app.oneshot(plan).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn token_comparison_is_exact() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"abcd"));
    }
}
