//! the slice of the Ollama HTTP API the launcher uses

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::time::Duration;

/// Ollama's default listen address. 127.0.0.1 rather than localhost: on Windows
/// localhost resolves to ::1 first, which the server does not listen on.
pub const BASE: &str = "http://127.0.0.1:11434";

/// the OpenAI-compatible endpoint harnesses are pointed at
pub fn openai_base() -> String {
    format!("{BASE}/v1")
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Details {
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct TagModel {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub details: Details,
}

#[derive(Deserialize)]
struct Tags {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct Show {
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub model_info: Map<String, Value>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Running {
    pub name: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub size_vram: u64,
    #[serde(default)]
    pub context_length: u64,
}

#[derive(Deserialize)]
struct Ps {
    #[serde(default)]
    models: Vec<Running>,
}

fn agent(timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new().timeout(timeout).build()
}

fn get<T: serde::de::DeserializeOwned>(path: &str, timeout: Duration) -> Result<T> {
    let resp = agent(timeout).get(&format!("{BASE}{path}")).call().with_context(|| format!("GET {path}"))?;
    resp.into_json().with_context(|| format!("decoding {path}"))
}

fn post<T: serde::de::DeserializeOwned>(path: &str, body: Value, timeout: Duration) -> Result<T> {
    let resp = agent(timeout).post(&format!("{BASE}{path}")).send_json(body).with_context(|| format!("POST {path}"))?;
    resp.into_json().with_context(|| format!("decoding {path}"))
}

/// true when the server answers at all
pub fn alive(timeout: Duration) -> bool {
    get::<Value>("/api/version", timeout).is_ok()
}

pub fn tags() -> Result<Vec<TagModel>> {
    Ok(get::<Tags>("/api/tags", Duration::from_secs(5))?.models)
}

pub fn show(model: &str) -> Result<Show> {
    post("/api/show", json!({ "model": model }), Duration::from_secs(10))
}

pub fn ps() -> Result<Vec<Running>> {
    Ok(get::<Ps>("/api/ps", Duration::from_secs(3))?.models)
}

/// the micro-batch every model is loaded with. left to itself, Ollama picks 512,
/// 1024 or 2048 per load from its own memory prediction, and the compute buffers
/// grow with it, so the same model at the same context could need up to a GiB
/// more on one load than on the next. 512 is also Ollama's default num_batch,
/// which is what every request a harness makes carries; a runner loaded with
/// any other explicit value would be reloaded by the harness's first request.
pub const NUM_BATCH: u64 = 512;

/// loads a model without generating anything, with the micro-batch pinned to
/// NUM_BATCH. keep_alive -1 pins it in memory, 0 unloads it.
pub fn load(model: &str, keep_alive: i64) -> Result<()> {
    let _: Value = post(
        "/api/generate",
        // an unload carries no options: a runner some other client loaded with a
        // different batch would otherwise be reloaded only to be unloaded
        if keep_alive == 0 {
            json!({ "model": model, "keep_alive": 0 })
        } else {
            json!({ "model": model, "keep_alive": keep_alive, "options": { "num_batch": NUM_BATCH } })
        },
        // a cold load of a large model from a slow disk takes minutes
        Duration::from_secs(15 * 60),
    )?;
    Ok(())
}

/// the running entry for `model`, matching either the name or the model field
pub fn find_running<'a>(running: &'a [Running], model: &str) -> Option<&'a Running> {
    let norm = |s: &str| s.trim_end_matches(":latest").to_ascii_lowercase();
    let want = norm(model);
    running.iter().find(|r| norm(&r.name) == want || norm(&r.model) == want)
}
