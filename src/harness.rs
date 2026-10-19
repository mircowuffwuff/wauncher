//! the coding agents the launcher can start, and how each is pointed at Ollama.
//!
//! - Claude Code and OpenCode go through `ollama launch <integration> --model <m>`,
//!   which sets their endpoint per session (environment for Claude Code,
//!   OPENCODE_CONFIG_CONTENT for OpenCode) and starts them without changing the
//!   working directory
//! - Qwen Code takes the OpenAI-compatible endpoint from OPENAI_BASE_URL,
//!   OPENAI_API_KEY and OPENAI_MODEL plus `--auth-type openai --model <m>`;
//!   environment and flags rank above its settings.json, so no file is touched
//! - Pi has no flag or variable for an endpoint: a provider must exist in
//!   models.json. a "wauncher" provider is added for the session and the file
//!   restored afterwards
//! - Hermes Agent does read OPENAI_BASE_URL, but config.yaml's model.provider
//!   wins over it, so the endpoint is written into config.yaml's model block for
//!   the session and the file restored afterwards
//!
//! a file is backed up beside itself before it is edited. a backup still
//! present on the next launch means a session never got to restore it, and the
//! backup is taken as the original.

use crate::ollama;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Harness {
    Claude,
    Pi,
    Hermes,
    OpenCode,
    Qwen,
}

pub const ALL: [Harness; 5] = [Harness::Claude, Harness::Pi, Harness::Hermes, Harness::OpenCode, Harness::Qwen];

/// Hermes refuses to run agent tools below this context
const HERMES_MIN_CTX: u64 = 64_000;

const BACKUP_SUFFIX: &str = "wauncher-backup";

impl Harness {
    pub fn id(self) -> &'static str {
        match self {
            Harness::Claude => "claude",
            Harness::Pi => "pi",
            Harness::Hermes => "hermes",
            Harness::OpenCode => "opencode",
            Harness::Qwen => "qwen",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Harness::Claude => "Claude Code",
            Harness::Pi => "Pi",
            Harness::Hermes => "Hermes Agent",
            Harness::OpenCode => "OpenCode",
            Harness::Qwen => "Qwen Code",
        }
    }

    pub fn from_id(id: &str) -> Option<Harness> {
        ALL.into_iter().find(|h| h.id() == id)
    }

    /// how the harness is pointed at Ollama, for the selection screen
    pub fn method(self) -> &'static str {
        match self {
            Harness::Claude | Harness::OpenCode => "ollama launch",
            Harness::Qwen => "environment + flags",
            Harness::Pi => "models.json, restored on exit",
            Harness::Hermes => "config.yaml, restored on exit",
        }
    }

    pub fn min_ctx(self) -> Option<u64> {
        match self {
            Harness::Hermes => Some(HERMES_MIN_CTX),
            _ => None,
        }
    }

    /// the harness's executable, or None when it is not installed
    pub fn binary(self) -> Option<PathBuf> {
        if let Some(p) = which(self.id()) {
            return Some(p);
        }
        let env = |k: &str| std::env::var_os(k).map(PathBuf::from);
        let home = env("USERPROFILE");
        let npm = env("APPDATA").map(|a| a.join("npm"));
        let candidates: Vec<Option<PathBuf>> = match self {
            Harness::Claude => vec![home.map(|h| h.join(".local").join("bin").join("claude.exe"))],
            Harness::OpenCode => vec![
                npm.map(|n| n.join("opencode.cmd")),
                home.map(|h| h.join(".opencode").join("bin").join("opencode.exe")),
            ],
            Harness::Pi => vec![npm.map(|n| n.join("pi.cmd"))],
            Harness::Qwen => vec![npm.map(|n| n.join("qwen.cmd"))],
            Harness::Hermes => {
                let venv = |root: PathBuf| root.join("hermes-agent").join("venv").join("Scripts").join("hermes.exe");
                vec![env("HERMES_HOME").map(venv), env("LOCALAPPDATA").map(|l| venv(l.join("hermes")))]
            }
        };
        candidates.into_iter().flatten().find(|p| p.is_file())
    }
}

/// looks a command up on PATH the way cmd.exe does, trying each PATHEXT extension
pub fn which(name: &str) -> Option<PathBuf> {
    let exts = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".into());
    let exts: Vec<&str> = exts.split(';').filter(|e| !e.is_empty()).collect();
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for ext in &exts {
            let p = dir.join(format!("{name}{}", ext.to_ascii_lowercase()));
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// what the harness needs to know about the model it runs
pub struct Session<'a> {
    pub model: &'a str,
    pub ctx: u64,
    pub thinking: bool,
    pub vision: bool,
}

/// a config file edited for one session
pub struct Edited {
    path: PathBuf,
    backup: PathBuf,
    /// the file before the edit; None when it did not exist
    original: Option<Vec<u8>>,
    written: Vec<u8>,
}

impl Edited {
    /// puts the original back, unless something else changed the file during
    /// the session, in which case the file is left alone and the backup kept
    pub fn restore(self) -> Result<Option<String>> {
        let now = std::fs::read(&self.path).ok();
        if now.as_deref() != Some(self.written.as_slice()) {
            return Ok(Some(format!(
                "{} changed during the session, so it was left as it is -- the original is in {}",
                self.path.display(),
                self.backup.display()
            )));
        }
        match &self.original {
            Some(bytes) => std::fs::write(&self.path, bytes),
            None => std::fs::remove_file(&self.path),
        }
        .with_context(|| format!("restoring {}", self.path.display()))?;
        let _ = std::fs::remove_file(&self.backup);
        Ok(None)
    }
}

/// edits `path` through `edit`, which gets the original content (None when the
/// file does not exist) and returns the new content
fn edit_file(path: &Path, edit: impl FnOnce(Option<&[u8]>) -> Result<Vec<u8>>) -> Result<Edited> {
    let mut backup = path.as_os_str().to_owned();
    backup.push(format!(".{BACKUP_SUFFIX}"));
    let backup = PathBuf::from(backup);

    let original = if backup.is_file() {
        Some(std::fs::read(&backup).with_context(|| format!("reading {}", backup.display()))?)
    } else {
        match std::fs::read(path) {
            Ok(b) => {
                std::fs::write(&backup, &b).with_context(|| format!("backing up to {}", backup.display()))?;
                Some(b)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        }
    };
    let written = edit(original.as_deref())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, &written).with_context(|| format!("writing {}", path.display()))?;
    Ok(Edited { path: path.to_path_buf(), backup, original, written })
}

pub struct Prepared {
    pub command: Command,
    pub edits: Vec<Edited>,
}

pub fn prepare(h: Harness, s: &Session, cwd: &Path) -> Result<Prepared> {
    let bin = h.binary().with_context(|| format!("{} is not installed", h.name()))?;
    let mut edits = Vec::new();
    let mut cmd = match h {
        Harness::Claude | Harness::OpenCode => {
            let ollama = crate::server::ollama_exe().context("ollama.exe not found")?;
            let mut c = Command::new(ollama);
            c.args(["launch", h.id(), "--model", s.model]);
            if h == Harness::Claude {
                // ollama launch only tells Claude Code the window of cloud models; without it
                // Claude Code assumes its own default and compacts far too late
                c.env("CLAUDE_CODE_MAX_CONTEXT_TOKENS", s.ctx.to_string());
                if std::env::var_os("CLAUDE_CODE_AUTO_COMPACT_WINDOW").is_none() {
                    c.env("CLAUDE_CODE_AUTO_COMPACT_WINDOW", s.ctx.to_string());
                }
            }
            c
        }
        Harness::Qwen => {
            let mut c = Command::new(bin);
            c.env("OPENAI_API_KEY", "ollama")
                .env("OPENAI_BASE_URL", ollama::openai_base())
                .env("OPENAI_MODEL", s.model)
                .env("QWEN_CODE_SYSTEM_SETTINGS_PATH", qwen_system_settings(s.ctx)?)
                .args(["--auth-type", "openai", "--model", s.model]);
            c
        }
        Harness::Pi => {
            edits.push(edit_file(&pi_models_path()?, |orig| pi_models_json(orig, s))?);
            let mut c = Command::new(bin);
            // pi matches the exact model id before it reads a ":<thinking>" suffix,
            // so a tag such as ":UD-IQ4_XS" is not mistaken for a thinking level
            c.args(["--provider", PI_PROVIDER, "--model", s.model]);
            c
        }
        Harness::Hermes => {
            edits.push(edit_file(&hermes_config_path()?, |orig| hermes_config_yaml(orig, s))?);
            let mut c = Command::new(bin);
            // config.yaml decides; these agree with it in case a stale ~/.hermes/.env says otherwise
            c.env("OPENAI_BASE_URL", ollama::openai_base()).env("OPENAI_API_KEY", "ollama");
            c
        }
    };
    cmd.current_dir(cwd);
    Ok(Prepared { command: cmd, edits })
}

/// Qwen Code sizes the context window from tables keyed on the model name and
/// assumes 200k for a name it does not know, which an hf.co name is, so it
/// would neither compact nor warn before the server's real context runs out.
/// model.generationConfig.contextWindowSize sets the window. system settings
/// rank above user and project settings, and QWEN_CODE_SYSTEM_SETTINGS_PATH
/// picks the system settings file per process, so the launcher writes its own
/// file into its state folder rather than editing one of the user's. it starts
/// from the machine-wide system settings, which the variable would otherwise
/// hide. a provider entry for the same model in the user's modelProviders
/// keeps its own generationConfig, which this does not reach.
fn qwen_system_settings(ctx: u64) -> Result<PathBuf> {
    let machine = std::env::var_os("QWEN_CODE_SYSTEM_SETTINGS_PATH")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("ProgramData").map(|p| PathBuf::from(p).join("qwen-code").join("settings.json")));
    let mut root = match machine.and_then(|p| std::fs::read(p).ok()) {
        Some(b) => serde_json::from_slice(&b).context("the machine-wide Qwen Code settings are not valid JSON")?,
        None => json!({}),
    };
    let Some(obj) = root.as_object_mut() else { bail!("the machine-wide Qwen Code settings are not a JSON object") };
    let model = obj.entry("model").or_insert_with(|| json!({}));
    if !model.is_object() {
        *model = json!({});
    }
    let gen =
        model.as_object_mut().expect("just made an object").entry("generationConfig").or_insert_with(|| json!({}));
    if !gen.is_object() {
        *gen = json!({});
    }
    gen.as_object_mut().expect("just made an object").insert("contextWindowSize".into(), json!(ctx));

    let path = crate::state::dir().join("qwen-system-settings.json");
    std::fs::create_dir_all(crate::state::dir())?;
    std::fs::write(&path, serde_json::to_vec_pretty(&root)?).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

const PI_PROVIDER: &str = "wauncher";

fn pi_models_path() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("PI_CODING_AGENT_DIR") {
        return Ok(PathBuf::from(dir).join("models.json"));
    }
    let home = std::env::var_os("USERPROFILE").context("USERPROFILE is not set")?;
    Ok(PathBuf::from(home).join(".pi").join("agent").join("models.json"))
}

fn pi_models_json(original: Option<&[u8]>, s: &Session) -> Result<Vec<u8>> {
    let mut root: Value = match original {
        Some(b) if !b.is_empty() => serde_json::from_slice(b).context("models.json is not valid JSON")?,
        _ => json!({}),
    };
    let Some(obj) = root.as_object_mut() else { bail!("models.json is not a JSON object") };
    let providers = obj.entry("providers").or_insert_with(|| json!({}));
    let Some(providers) = providers.as_object_mut() else { bail!("models.json: providers is not an object") };
    let input = if s.vision { json!(["text", "image"]) } else { json!(["text"]) };
    providers.insert(
        PI_PROVIDER.into(),
        json!({
            "baseUrl": ollama::openai_base(),
            "api": "openai-completions",
            "apiKey": "ollama",
            "models": [{
                "id": s.model,
                "contextWindow": s.ctx,
                "reasoning": s.thinking,
                "input": input,
            }],
        }),
    );
    Ok(serde_json::to_vec_pretty(&root)?)
}

fn hermes_config_path() -> Result<PathBuf> {
    let home = match std::env::var_os("HERMES_HOME") {
        Some(h) => PathBuf::from(h),
        None => PathBuf::from(std::env::var_os("LOCALAPPDATA").context("LOCALAPPDATA is not set")?).join("hermes"),
    };
    Ok(home.join("config.yaml"))
}

/// the model block Hermes documents for a local Ollama: provider custom, the /v1
/// base url, and the context the server was started with. ollama_num_ctx is the
/// size Hermes asks Ollama for on local endpoints, so it has to agree with the
/// server or the model is reloaded at a different context.
fn hermes_config_yaml(original: Option<&[u8]>, s: &Session) -> Result<Vec<u8>> {
    use serde_yaml::{Mapping, Value as Y};
    let mut root: Y = match original {
        Some(b) if !b.is_empty() => serde_yaml::from_slice(b).context("config.yaml is not valid YAML")?,
        _ => Y::Mapping(Mapping::new()),
    };
    if root.is_null() {
        root = Y::Mapping(Mapping::new());
    }
    let Some(map) = root.as_mapping_mut() else { bail!("config.yaml is not a mapping") };
    let model = map.entry(Y::from("model")).or_insert_with(|| Y::Mapping(Mapping::new()));
    if !model.is_mapping() {
        *model = Y::Mapping(Mapping::new());
    }
    let m = model.as_mapping_mut().expect("just made a mapping");
    m.insert("default".into(), s.model.into());
    m.insert("provider".into(), "custom".into());
    m.insert("base_url".into(), ollama::openai_base().into());
    m.insert("api_key".into(), "ollama".into());
    m.insert("context_length".into(), s.ctx.into());
    m.insert("ollama_num_ctx".into(), s.ctx.into());
    Ok(serde_yaml::to_string(&root)?.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session<'static> {
        Session { model: "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS", ctx: 65536, thinking: true, vision: true }
    }

    #[test]
    fn pi_provider_is_added_beside_existing_ones() {
        let orig = br#"{"providers":{"ollama":{"baseUrl":"x","models":[{"id":"a"}]}},"other":1}"#;
        let out: Value = serde_json::from_slice(&pi_models_json(Some(orig), &session()).unwrap()).unwrap();
        assert_eq!(out["other"], 1);
        assert_eq!(out["providers"]["ollama"]["baseUrl"], "x");
        let p = &out["providers"]["wauncher"];
        assert_eq!(p["baseUrl"], "http://127.0.0.1:11434/v1");
        assert_eq!(p["models"][0]["id"], "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS");
        assert_eq!(p["models"][0]["contextWindow"], 65536);
        assert_eq!(p["models"][0]["reasoning"], true);
    }

    #[test]
    fn hermes_model_block_keeps_other_keys() {
        let orig = b"model:\n  default: something\n  extra: keep\nagent:\n  max_turns: 90\n";
        let out: serde_yaml::Value =
            serde_yaml::from_slice(&hermes_config_yaml(Some(orig), &session()).unwrap()).unwrap();
        assert_eq!(out["model"]["provider"], "custom");
        assert_eq!(out["model"]["default"], "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS");
        assert_eq!(out["model"]["base_url"], "http://127.0.0.1:11434/v1");
        assert_eq!(out["model"]["context_length"], 65536);
        assert_eq!(out["model"]["extra"], "keep");
        assert_eq!(out["agent"]["max_turns"], 90);
    }

    #[test]
    fn edit_restores_the_original_and_removes_the_backup() {
        let dir = std::env::temp_dir().join(format!("wauncher-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("models.json");
        std::fs::write(&f, b"{}").unwrap();

        let e = edit_file(&f, |o| pi_models_json(o, &session())).unwrap();
        assert!(std::fs::read_to_string(&f).unwrap().contains("wauncher"));
        assert!(e.backup.is_file());
        assert_eq!(e.restore().unwrap(), None);
        assert_eq!(std::fs::read(&f).unwrap(), b"{}");

        // a file that did not exist is removed again
        let g = dir.join("config.yaml");
        let e = edit_file(&g, |o| hermes_config_yaml(o, &session())).unwrap();
        assert!(g.is_file());
        e.restore().unwrap();
        assert!(!g.exists());

        // a file changed during the session is left alone
        let e = edit_file(&f, |o| pi_models_json(o, &session())).unwrap();
        std::fs::write(&f, b"{\"changed\":true}").unwrap();
        assert!(e.restore().unwrap().is_some());
        assert_eq!(std::fs::read(&f).unwrap(), b"{\"changed\":true}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// runs Pi, Hermes and Qwen Code stand-ins (batch files that echo what they get) through
    /// prepare, with the config homes in a temp folder. it changes this process's environment:
    /// cargo test stand_ins -- --ignored --nocapture
    #[test]
    #[ignore = "changes PATH, PI_CODING_AGENT_DIR and HERMES_HOME for the whole test process"]
    fn stand_ins() {
        let root = std::env::temp_dir().join(format!("wauncher-stand-ins-{}", std::process::id()));
        let (bin, work) = (root.join("bin"), root.join("work"));
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&work).unwrap();
        for n in ["pi", "hermes", "qwen"] {
            let script = "@echo off\r\necho cwd=%CD%\r\necho args=%*\r\necho OPENAI_BASE_URL=%OPENAI_BASE_URL% OPENAI_MODEL=%OPENAI_MODEL%\r\nif defined PI_CODING_AGENT_DIR type \"%PI_CODING_AGENT_DIR%\\models.json\"\r\nif defined HERMES_HOME type \"%HERMES_HOME%\\config.yaml\"\r\n";
            std::fs::write(bin.join(format!("{n}.cmd")), script).unwrap();
        }
        let path = std::env::join_paths(
            std::iter::once(bin.clone()).chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
        )
        .unwrap();
        std::env::set_var("PATH", path);
        std::env::set_var("PI_CODING_AGENT_DIR", root.join("pi"));
        std::env::set_var("HERMES_HOME", root.join("hermes"));
        std::fs::create_dir_all(root.join("hermes")).unwrap();
        std::fs::write(root.join("hermes").join("config.yaml"), "agent:\n  max_turns: 90\n").unwrap();

        for h in [Harness::Pi, Harness::Hermes, Harness::Qwen] {
            let mut p = prepare(h, &session(), &work).unwrap();
            let out = p.command.output().unwrap();
            let text = String::from_utf8_lossy(&out.stdout).to_string();
            println!("== {}\n{text}", h.name());
            assert!(text.contains(&format!("cwd={}", work.display())), "{text}");
            if h == Harness::Qwen {
                let path =
                    p.command.get_envs().find(|(k, _)| *k == "QWEN_CODE_SYSTEM_SETTINGS_PATH").unwrap().1.unwrap();
                let settings: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
                assert_eq!(settings["model"]["generationConfig"]["contextWindowSize"], 65536);
            }
            for e in p.edits.drain(..) {
                assert_eq!(e.restore().unwrap(), None);
            }
        }
        assert!(!root.join("pi").join("models.json").exists(), "pi models.json removed again");
        assert_eq!(
            std::fs::read_to_string(root.join("hermes").join("config.yaml")).unwrap(),
            "agent:\n  max_turns: 90\n"
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
