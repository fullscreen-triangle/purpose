//! `purpose serve`: an axum HTTP server exposing the theme factory over the
//! network, gated by a static bearer token, so `purpose-factory`'s caller
//! (e.g. the profile-web frontend) need not run on the same machine or
//! share a filesystem with it. Every theme lives under
//! `<root>/themes/<name>/{sources,model}/`; the server owns that layout —
//! callers never see or choose a local path, since they may not have one.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Multipart, Path as AxumPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::contract::{
    BaseModelSpec, HeuristicVerifier, PretrainedConfig, ScratchConfig, ThemeContract, TrainingSpec,
};
use crate::error::Error;
use crate::factory::{Factory, Registry, ThemeModel};
use crate::source::{LocalFileSource, SourceProvider, UrlSource};

/// Extensions `LocalFileSource` knows how to ingest — kept in sync by hand
/// with `LocalFileSource::new`'s default list (source.rs), since that
/// default is a `Vec` built at construction time, not a `const` this module
/// can reference directly.
const SUPPORTED_EXTENSIONS: &[&str] = &["tex", "pdf", "md", "txt", "csv", "json"];

const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;
const MODEL_FILES: &[&str] = &["model.safetensors", "config.json", "tokenizer.json"];

#[derive(Clone)]
struct ServerState {
    root: PathBuf,
    token: Arc<String>,
}

fn theme_dir(root: &Path, name: &str) -> PathBuf {
    root.join("themes").join(name)
}

fn sources_dir(root: &Path, name: &str) -> PathBuf {
    theme_dir(root, name).join("sources")
}

fn model_dir(root: &Path, name: &str) -> PathBuf {
    theme_dir(root, name).join("model")
}

fn registry_path(root: &Path) -> PathBuf {
    root.join("registry.json")
}

/// Builds the router. `root` is the server's local working directory
/// (created if missing); `token` is the bearer token every request must
/// present via `Authorization: Bearer <token>`.
pub fn app(root: PathBuf, token: String) -> Router {
    let state = ServerState {
        root,
        token: Arc::new(token),
    };

    Router::new()
        .route("/themes", get(list_themes))
        .route("/themes/{name}/sources", post(upload_sources))
        .route("/themes/{name}/build", post(build_theme))
        .route("/themes/{name}/model/{file}", get(download_model_file))
        .layer(middleware::from_fn_with_state(state.clone(), auth))
        .with_state(state)
}

async fn auth(
    State(state): State<ServerState>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let presented = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match presented {
        Some(t) if t == state.token.as_str() => next.run(request).await,
        _ => (StatusCode::UNAUTHORIZED, "missing or invalid bearer token").into_response(),
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let status = match &self {
            Error::Source(_) | Error::Corpus(_) | Error::Config(_) => StatusCode::BAD_REQUEST,
            Error::Train(_) | Error::Export(_) | Error::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        (status, Json(serde_json::json!({ "error": self.to_string() }))).into_response()
    }
}

// =====================================================================
// GET /themes
// =====================================================================

async fn list_themes(State(state): State<ServerState>) -> Result<Json<Vec<ThemeModel>>, Error> {
    let models = Registry::new(registry_path(&state.root)).load()?;
    Ok(Json(models))
}

// =====================================================================
// POST /themes/{name}/sources
// =====================================================================

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
    let dir = sources_dir(&state.root, &name);
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
// POST /themes/{name}/build
// =====================================================================

#[derive(Deserialize)]
struct BuildRequest {
    #[serde(default)]
    urls: Vec<String>,
    #[serde(default)]
    model: BuildModelChoice,
    #[serde(default)]
    training: Option<BuildTrainingSpec>,
}

#[derive(Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BuildModelChoice {
    #[default]
    Scratch,
    Pretrained {
        repo: String,
        revision: Option<String>,
    },
}

#[derive(Deserialize)]
struct BuildTrainingSpec {
    epochs: Option<usize>,
    batch_size: Option<usize>,
    learning_rate: Option<f64>,
    lora_rank: Option<usize>,
    lora_alpha: Option<f64>,
}

async fn build_theme(
    State(state): State<ServerState>,
    AxumPath(name): AxumPath<String>,
    body: Option<Json<BuildRequest>>,
) -> Result<Json<ThemeModel>, Error> {
    let body = body.map(|Json(b)| b).unwrap_or(BuildRequest {
        urls: Vec::new(),
        model: BuildModelChoice::Scratch,
        training: None,
    });

    let dir = sources_dir(&state.root, &name);
    let mut sources: Vec<Box<dyn SourceProvider>> = Vec::new();
    if dir.is_dir() {
        sources.push(Box::new(LocalFileSource::new(dir)));
    }
    if !body.urls.is_empty() {
        sources.push(Box::new(UrlSource::new(body.urls)));
    }
    if sources.is_empty() {
        return Err(Error::Source(format!(
            "theme '{name}' has no sources — upload files or pass urls before building"
        )));
    }

    let base_model = match body.model {
        BuildModelChoice::Scratch => BaseModelSpec::Scratch(ScratchConfig::default()),
        BuildModelChoice::Pretrained { repo, revision } => {
            BaseModelSpec::Pretrained(PretrainedConfig { repo, revision })
        }
    };

    let mut training = TrainingSpec::default();
    if let Some(t) = body.training {
        if let Some(v) = t.epochs {
            training.epochs = v;
        }
        if let Some(v) = t.batch_size {
            training.batch_size = v;
        }
        if let Some(v) = t.learning_rate {
            training.learning_rate = v;
        }
        if let Some(v) = t.lora_rank {
            training.lora_rank = v;
        }
        if let Some(v) = t.lora_alpha {
            training.lora_alpha = v;
        }
    }

    let contract = ThemeContract {
        name: name.clone(),
        sources,
        base_model,
        verifier: Box::new(HeuristicVerifier::default()),
        training,
    };

    let out_dir = model_dir(&state.root, &name);
    let model = Factory::build(contract, &out_dir).await?;

    Registry::new(registry_path(&state.root)).record(&model)?;

    Ok(Json(model))
}

// =====================================================================
// GET /themes/{name}/model/{file}
// =====================================================================

async fn download_model_file(
    State(state): State<ServerState>,
    AxumPath((name, file)): AxumPath<(String, String)>,
) -> Result<Response, Error> {
    if !MODEL_FILES.contains(&file.as_str()) {
        return Ok((StatusCode::NOT_FOUND, "no such model file").into_response());
    }

    let path = model_dir(&state.root, &name).join(&file);
    match tokio::fs::read(&path).await {
        Ok(bytes) => Ok(bytes.into_response()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Ok((StatusCode::NOT_FOUND, "theme not built yet").into_response())
        }
        Err(e) => Err(Error::Io(e)),
    }
}
