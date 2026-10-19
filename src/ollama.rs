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
    /// the chat template; for a GGUF import, its embedded Jinja template
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub modelfile: String,
}

/// the namespace of the models the launcher creates for itself. they are hidden
/// from the model list and removed when the session that needed them ends.
pub const OWN_NAMESPACE: &str = "wauncher/";

/// Ollama's built-in renderer for a model whose own chat template cannot take
/// what Claude Code sends, or None when the template copes or a renderer is
/// already set.
///
/// Claude Code sends its environment block as a system message after the first
/// user message, and Ollama passes it through. the Jinja template Qwen ships in
/// its GGUF files raises "System message must be at the beginning" on that, so
/// every request fails with a 500 before a token is generated. Ollama's own
/// qwen3.5 and qwen3.8 renderers, which the models in Ollama's library use,
/// accept it.
pub fn renderer_for_claude(show: &Show) -> Option<&'static str> {
    if show.modelfile.lines().any(|l| l.starts_with("RENDERER ")) {
        return None;
    }
    if !show.template.contains("System message must be at the beginning") {
        return None;
    }
    if show.model_info.get("general.architecture").and_then(Value::as_str) != Some("qwen35") {
        return None;
    }
    let basename = show.model_info.get("general.basename").and_then(Value::as_str).unwrap_or("");
    Some(if basename.contains("3.8") { "qwen3.8" } else { "qwen3.5" })
}

/// the name a model derived from `model` gets in OWN_NAMESPACE
pub fn derived_name(model: &str) -> String {
    let (base, tag) = model.rsplit_once(':').unwrap_or((model, "latest"));
    let base = base.rsplit('/').next().unwrap_or(base);
    let clean = |s: &str| -> String {
        s.chars()
            .map(|c| c.to_ascii_lowercase())
            .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '-' })
            .collect()
    };
    format!("{OWN_NAMESPACE}{}:{}", clean(base), clean(tag))
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

/// creates `name` from `from` with Ollama's built-in `renderer` and the parser of
/// the same name. it shares the source's blobs, so it takes no disk space.
pub fn create_with_renderer(name: &str, from: &str, renderer: &str) -> Result<()> {
    let _: Value = post(
        "/api/create",
        json!({ "model": name, "from": from, "renderer": renderer, "parser": renderer, "stream": false }),
        Duration::from_secs(60),
    )?;
    Ok(())
}

/// removes a model; its blobs stay for as long as another model uses them
pub fn delete(name: &str) -> Result<()> {
    agent(Duration::from_secs(30))
        .delete(&format!("{BASE}/api/delete"))
        .send_json(json!({ "model": name }))
        .with_context(|| format!("deleting {name}"))?;
    Ok(())
}

/// the running entry for `model`, matching either the name or the model field
pub fn find_running<'a>(running: &'a [Running], model: &str) -> Option<&'a Running> {
    let norm = |s: &str| s.trim_end_matches(":latest").to_ascii_lowercase();
    let want = norm(model);
    running.iter().find(|r| norm(&r.name) == want || norm(&r.model) == want)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(arch: &str, basename: &str, template: &str, modelfile: &str) -> Show {
        let mut model_info = Map::new();
        model_info.insert("general.architecture".into(), Value::from(arch));
        model_info.insert("general.basename".into(), Value::from(basename));
        Show { template: template.into(), modelfile: modelfile.into(), model_info, ..Default::default() }
    }

    const QWEN_TEMPLATE: &str =
        "{%- if not loop.first %}{{- raise_exception('System message must be at the beginning.') }}";

    #[test]
    fn qwen_gguf_templates_get_the_builtin_renderer() {
        assert_eq!(renderer_for_claude(&show("qwen35", "Qwen3.5-4B", QWEN_TEMPLATE, "FROM x")), Some("qwen3.5"));
        assert_eq!(renderer_for_claude(&show("qwen35", "Qwen3.8-27B", QWEN_TEMPLATE, "FROM x")), Some("qwen3.8"));
    }

    #[test]
    fn other_models_are_served_as_they_are() {
        // a renderer is set already
        assert_eq!(
            renderer_for_claude(&show("qwen35", "Qwen3.5-4B", QWEN_TEMPLATE, "FROM x\nRENDERER qwen3.5\n")),
            None
        );
        // the template copes with a late system message
        assert_eq!(renderer_for_claude(&show("qwen35", "Ornith-1.5-9B", "{{ messages }}", "FROM x")), None);
        // no built-in renderer for the architecture
        assert_eq!(renderer_for_claude(&show("gemma4", "gemma-4-12b", QWEN_TEMPLATE, "FROM x")), None);
    }

    #[test]
    fn derived_names_are_valid_and_namespaced() {
        assert_eq!(derived_name("hf.co/unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL"), "wauncher/qwen3.5-4b-gguf:ud-q4_k_xl");
        assert_eq!(derived_name("qwen3.8:27b-q4_K_M"), "wauncher/qwen3.8:27b-q4_k_m");
        assert_eq!(derived_name("mymodel"), "wauncher/mymodel:latest");
    }
}
