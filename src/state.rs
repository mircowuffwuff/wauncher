//! what the launcher remembers between runs, in %APPDATA%\wauncher\state.json

use crate::vram::{KvType, DEFAULT_OVERHEAD_BYTES, DEFAULT_RESERVE_BYTES};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const APP_NAME: &str = "wauncher";

pub const CTX_STEP: u64 = 8 * 1024;
pub const CTX_MIN: u64 = 8 * 1024;
const CTX_DEFAULT: u64 = 32 * 1024;

/// a correction learned from one load: how much dedicated VRAM the load really
/// took, measured on the GPUs' usage counters, minus the uncorrected estimate
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Correction {
    pub bytes: i64,
    /// the context and KV type the correction was measured at
    pub ctx: u64,
    pub kv: KvType,
}

/// the settings an `ollama serve` process was started with. the server reads
/// them once at startup, so a server whose settings differ is restarted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerSettings {
    pub ctx: u64,
    pub kv: KvType,
    pub env: BTreeMap<String, String>,
}

/// an `ollama serve` the launcher started, identified by pid and creation time
/// so a recycled pid is not mistaken for it
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OwnedServer {
    pub pid: u32,
    pub created: u64,
    pub settings: ServerSettings,
}

fn default_ctx() -> u64 {
    CTX_DEFAULT
}

/// variables every `ollama serve` the launcher starts gets, before the ones in
/// state.json's server_env, which add to or override them.
/// - OLLAMA_VULKAN: the Ollama tray app turns the Vulkan backend on through its
///   own settings rather than the environment, so a plain `ollama serve` would
///   come up without it
/// - OLLAMA_KEEP_ALIVE: the model is pinned with keep_alive -1 before the
///   harness starts, but every request the harness makes resets that to the
///   server default; -1 keeps the model resident until the launcher unloads it
/// - GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM: with resizable BAR on, llama.cpp's
///   Vulkan backend places its buffers in the host-visible VRAM memory type.
///   on Windows with AMD's driver only about 8.4 GiB of that stays in VRAM and
///   the rest is backed by system RAM, while Ollama still reports the model as
///   fully on the GPU. a 27B model measured 7 tok/s that way and 34 tok/s in
///   plain device-local memory, which this variable forces. Ollama passes its
///   environment on to llama-server.
pub fn builtin_server_env() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("OLLAMA_VULKAN".to_string(), "1".to_string()),
        ("OLLAMA_KEEP_ALIVE".to_string(), "-1".to_string()),
        ("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM".to_string(), "1".to_string()),
    ])
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub last_model: Option<String>,
    #[serde(default)]
    pub last_harness: Option<String>,
    #[serde(default = "default_ctx")]
    pub ctx: u64,
    #[serde(default)]
    pub kv: KvType,
    #[serde(default)]
    pub corrections: BTreeMap<String, Correction>,
    /// overrides DEFAULT_OVERHEAD_BYTES
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overhead_bytes: Option<u64>,
    /// overrides DEFAULT_RESERVE_BYTES
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserve_bytes: Option<u64>,
    /// added to, and overriding, builtin_server_env
    #[serde(default)]
    pub server_env: BTreeMap<String, String>,
    #[serde(default)]
    pub server: Option<OwnedServer>,
}

impl Default for State {
    fn default() -> Self {
        State {
            last_model: None,
            last_harness: None,
            ctx: CTX_DEFAULT,
            kv: KvType::default(),
            corrections: BTreeMap::new(),
            overhead_bytes: None,
            reserve_bytes: None,
            server_env: BTreeMap::new(),
            server: None,
        }
    }
}

pub fn dir() -> PathBuf {
    let base = std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    base.join(APP_NAME)
}

fn path() -> PathBuf {
    dir().join("state.json")
}

impl State {
    /// a missing or unreadable file gives the defaults; an unreadable one is kept aside
    pub fn load() -> State {
        let p = path();
        let Ok(text) = std::fs::read_to_string(&p) else {
            return State::default();
        };
        match serde_json::from_str::<State>(&text) {
            Ok(mut s) => {
                s.ctx = s.ctx.max(CTX_MIN) / CTX_STEP * CTX_STEP;
                s
            }
            Err(_) => {
                let _ = std::fs::rename(&p, p.with_extension("json.broken"));
                State::default()
            }
        }
    }

    /// written to a temp file and renamed, so a crash never leaves half a file
    pub fn save(&self) -> Result<()> {
        let p = path();
        std::fs::create_dir_all(dir()).with_context(|| format!("creating {}", dir().display()))?;
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, &p).with_context(|| format!("writing {}", p.display()))?;
        Ok(())
    }

    pub fn overhead(&self) -> u64 {
        self.overhead_bytes.unwrap_or(DEFAULT_OVERHEAD_BYTES)
    }

    pub fn reserve(&self) -> u64 {
        self.reserve_bytes.unwrap_or(DEFAULT_RESERVE_BYTES)
    }

    /// the variables a server is started with on top of the inherited environment
    pub fn server_env(&self) -> BTreeMap<String, String> {
        let mut env = builtin_server_env();
        env.extend(self.server_env.iter().map(|(k, v)| (k.clone(), v.clone())));
        env
    }

    pub fn correction(&self, model: &str) -> i64 {
        self.corrections.get(model).map(|c| c.bytes).unwrap_or(0)
    }
}
