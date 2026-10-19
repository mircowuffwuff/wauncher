//! VRAM estimate for a model at a given context and KV cache type.
//!
//! the estimate is weights + KV cache + recurrent state + a flat overhead, plus
//! a per-model correction learned from how much dedicated VRAM the model really
//! took the last time it was loaded. the overhead stands for what the metadata
//! does not show: compute buffers, the vision projector's compute buffer and the
//! driver's own allocations, less the token embeddings, which llama.cpp keeps in
//! system RAM although they count towards the weights.
//!
//! the KV cache is worked out layer by layer from the GGUF metadata Ollama
//! exposes in /api/show model_info, because a growing number of architectures
//! keep a cache in only some of their layers:
//! - hybrid linear-attention models (qwen35, qwen3next) carry
//!   "<arch>.full_attention_interval": only every n-th layer is full attention,
//!   the rest are Gated DeltaNet layers with a fixed-size recurrent state
//! - hybrid SSM models (nemotron_h, granitehybrid, lfm2, ...) carry
//!   "<arch>.attention.head_count_kv" as a per-layer array with 0 on recurrent layers
//! - sliding-window models (gemma3, gemma4, ...) carry
//!   "<arch>.attention.sliding_window_pattern"; their local layers only ever
//!   hold the window, not the whole context
//! - "<arch>.nextn_predict_layers" are multi-token-prediction layers at the end
//!   of the block list that Ollama does not run, so they hold no cache

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// flat allowance for what the metadata does not show, until a load has
/// calibrated the model. measured on Vulkan with the micro-batch at 512, as the
/// model's real usage minus weights, KV cache, recurrent state and mask, it
/// ranged from -0.40 GiB (Ornith 9B) to +0.29 GiB (Qwen3.5-4B), so 0.5 GiB keeps
/// a first estimate from being optimistic.
pub const DEFAULT_OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;

/// dedicated VRAM per GPU that is not usable. Windows starts moving a process's
/// video memory to system RAM before the GPU is full: on a 16 GiB RX 6950 XT
/// (15.95 GiB reported) loads that brought total usage to 14.43 GiB ran at full
/// speed, and loads that needed 14.7 GiB and more had part of their buffers
/// evicted, with total usage stuck between 14.6 and 14.7 GiB. Ollama reports
/// such a model as fully in VRAM; prompt processing runs about 5x slower.
pub const DEFAULT_RESERVE_BYTES: u64 = 1536 * 1024 * 1024;

/// tokens a sliding-window layer holds beyond its window: llama.cpp sizes that
/// cache to the window plus one micro-batch (Gemma 4 12B held 2048 tokens for its
/// 1024-token window with a 1024 micro-batch)
const SWA_BATCH: u64 = crate::ollama::NUM_BATCH;

/// the attention mask llama.cpp reserves in its compute buffer: one f16 per
/// cached token per micro-batch token. it is the part of the compute buffer
/// that grows with the context: with a 1024 micro-batch, Gemma 4 12B's grew by
/// exactly 96k x 1024 x 2 bytes from 32k to 128k
pub fn mask_bytes(ctx: u64) -> u64 {
    ctx * crate::ollama::NUM_BATCH * 2
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KvType {
    #[serde(rename = "f16")]
    F16,
    #[default]
    #[serde(rename = "q8_0")]
    Q8_0,
}

impl KvType {
    /// bytes per cached element. q8_0 stores blocks of 32 int8 values plus one
    /// f16 scale, i.e. 34 bytes per 32 elements.
    pub fn bytes_per_element(self) -> f64 {
        match self {
            KvType::F16 => 2.0,
            KvType::Q8_0 => 34.0 / 32.0,
        }
    }

    /// the value OLLAMA_KV_CACHE_TYPE takes
    pub fn as_str(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
        }
    }

    pub fn toggled(self) -> KvType {
        match self {
            KvType::F16 => KvType::Q8_0,
            KvType::Q8_0 => KvType::F16,
        }
    }
}

/// what one transformer block keeps per cached token
#[derive(Clone, Debug, PartialEq)]
pub enum LayerCache {
    /// recurrent, or sharing another layer's cache
    None,
    /// full attention: `elems` K+V elements per token over the whole context
    Full { elems: u64 },
    /// sliding-window attention: `elems` per token, at most `window` tokens
    Sliding { elems: u64, window: u64 },
}

#[derive(Clone, Debug)]
pub struct Arch {
    /// "<arch>.context_length", 0 when the model does not say
    pub native_ctx: u64,
    pub layers: Vec<LayerCache>,
    /// fixed-size state of the recurrent layers, independent of context
    pub recurrent_bytes: u64,
}

fn as_u64(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64))
}

/// a scalar or per-layer array, spread to one value per layer
fn per_layer_u64(v: Option<&Value>, n: usize) -> Option<Vec<u64>> {
    match v? {
        Value::Array(a) => {
            let mut out: Vec<u64> = a.iter().map(|x| as_u64(x).unwrap_or(0)).collect();
            let last = out.last().copied().unwrap_or(0);
            out.resize(n, last);
            Some(out)
        }
        x => as_u64(x).map(|s| vec![s; n]),
    }
}

impl Arch {
    pub fn from_model_info(info: &Map<String, Value>) -> Option<Arch> {
        let arch = info.get("general.architecture")?.as_str()?.to_string();
        let get = |k: &str| info.get(&format!("{arch}.{k}"));
        let u = |k: &str| get(k).and_then(as_u64);

        let native_ctx = u("context_length").unwrap_or(0);
        let block_count = u("block_count")? as usize;
        let nextn = u("nextn_predict_layers").unwrap_or(0) as usize;
        let n = block_count.saturating_sub(nextn);

        let heads = per_layer_u64(get("attention.head_count"), n).unwrap_or_else(|| vec![0; n]);
        let kv_heads = per_layer_u64(get("attention.head_count_kv"), n).unwrap_or_else(|| heads.clone());

        // head dimension falls back to embedding_length / head_count, as llama.cpp does
        let first_heads = heads.iter().copied().find(|&h| h > 0).unwrap_or(1);
        let head_dim = u("embedding_length").map(|e| e / first_heads).unwrap_or(0);
        let k_len = u("attention.key_length").unwrap_or(head_dim);
        let v_len = u("attention.value_length").unwrap_or(k_len);
        let k_len_swa = u("attention.key_length_swa").unwrap_or(k_len);
        let v_len_swa = u("attention.value_length_swa").unwrap_or(v_len);

        let interval = u("full_attention_interval").filter(|&i| i > 0);
        let window = u("attention.sliding_window").filter(|&w| w > 0);
        let swa_pattern: Vec<bool> = match get("attention.sliding_window_pattern") {
            Some(Value::Array(a)) => a.iter().map(|x| x.as_bool().unwrap_or(false)).collect(),
            // an integer pattern n means every n-th layer is global, the rest local
            Some(x) => match as_u64(x) {
                Some(p) if p > 0 => (0..n).map(|i| !(i as u64 + 1).is_multiple_of(p)).collect(),
                _ => Vec::new(),
            },
            None => Vec::new(),
        };
        let shared_kv = u("attention.shared_kv_layers").unwrap_or(0) as usize;
        // multi-head latent attention caches one compressed latent plus the rope part per token
        let mla = u("attention.kv_lora_rank").map(|r| r + u("rope.dimension_count").unwrap_or(0));

        let mut layers = Vec::with_capacity(n);
        let mut recurrent_layers = 0u64;
        for (i, &kvh) in kv_heads.iter().enumerate() {
            if i >= n.saturating_sub(shared_kv) && shared_kv > 0 {
                layers.push(LayerCache::None);
                continue;
            }
            let is_recurrent = interval.is_some_and(|iv| !(i as u64 + 1).is_multiple_of(iv)) || kvh == 0;
            if is_recurrent {
                recurrent_layers += 1;
                layers.push(LayerCache::None);
                continue;
            }
            if let Some(elems) = mla {
                layers.push(LayerCache::Full { elems });
                continue;
            }
            let sliding = swa_pattern.get(i).copied().unwrap_or(false);
            match (sliding, window) {
                (true, Some(w)) => layers.push(LayerCache::Sliding { elems: kvh * (k_len_swa + v_len_swa), window: w }),
                _ => layers.push(LayerCache::Full { elems: kvh * (k_len + v_len) }),
            }
        }

        // recurrent state, kept in f32: the SSM state (d_state x d_inner) plus the
        // conv window over (d_inner + 2 x groups x d_state) channels
        let recurrent_bytes = match (u("ssm.inner_size"), u("ssm.state_size")) {
            (Some(d_inner), Some(d_state)) if recurrent_layers > 0 => {
                let groups = u("ssm.group_count").unwrap_or(0);
                let conv = u("ssm.conv_kernel").unwrap_or(1).saturating_sub(1);
                let per_layer = d_state * d_inner + conv * (d_inner + 2 * groups * d_state);
                recurrent_layers * per_layer * 4
            }
            _ => 0,
        };

        Some(Arch { native_ctx, layers, recurrent_bytes })
    }

    #[cfg(test)]
    pub fn attention_layers(&self) -> usize {
        self.layers.iter().filter(|l| **l != LayerCache::None).count()
    }

    /// KV cache bytes for `ctx` tokens
    pub fn kv_bytes(&self, ctx: u64, kv: KvType) -> u64 {
        let elems: u64 = self
            .layers
            .iter()
            .map(|l| match *l {
                LayerCache::None => 0,
                LayerCache::Full { elems } => elems * ctx,
                LayerCache::Sliding { elems, window } => elems * ctx.min(window + SWA_BATCH),
            })
            .sum();
        (elems as f64 * kv.bytes_per_element()).round() as u64
    }
}

/// the context a model actually gets: the slider value, capped at the model's
/// native maximum. returns the context and whether it was capped.
pub fn effective_ctx(slider: u64, native: u64) -> (u64, bool) {
    if native > 0 && native < slider {
        (native, true)
    } else {
        (slider, false)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Estimate {
    pub weights: u64,
    pub kv: u64,
    pub recurrent: u64,
    pub mask: u64,
    pub overhead: u64,
    /// learned from the last load of this model, may be negative
    pub correction: i64,
}

impl Estimate {
    pub fn uncorrected(&self) -> i64 {
        (self.weights + self.kv + self.recurrent + self.mask + self.overhead) as i64
    }

    pub fn total(&self) -> i64 {
        self.uncorrected() + self.correction
    }
}

pub fn estimate(weights: u64, arch: Option<&Arch>, ctx: u64, kv: KvType, overhead: u64, correction: i64) -> Estimate {
    Estimate {
        weights,
        kv: arch.map(|a| a.kv_bytes(ctx, kv)).unwrap_or(0),
        recurrent: arch.map(|a| a.recurrent_bytes).unwrap_or(0),
        mask: mask_bytes(ctx),
        overhead,
        correction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn info(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    /// model_info as Ollama reports it for Qwen3.8-27B (unsloth UD-IQ4_XS), trimmed to the keys the math reads
    fn qwen38_27b() -> Map<String, Value> {
        info(json!({
            "general.architecture": "qwen35",
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
            "qwen35.block_count": 65,
            "qwen35.context_length": 262144,
            "qwen35.embedding_length": 5120,
            "qwen35.full_attention_interval": 4,
            "qwen35.nextn_predict_layers": 1,
            "qwen35.ssm.conv_kernel": 4,
            "qwen35.ssm.group_count": 16,
            "qwen35.ssm.inner_size": 6144,
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.time_step_rank": 48
        }))
    }

    #[test]
    fn qwen38_27b_has_16_attention_layers_of_64() {
        let a = Arch::from_model_info(&qwen38_27b()).unwrap();
        assert_eq!(a.layers.len(), 64);
        assert_eq!(a.attention_layers(), 16);
        assert_eq!(a.native_ctx, 262144);
    }

    #[test]
    fn qwen38_27b_kv_is_64_kib_per_token_at_f16() {
        let a = Arch::from_model_info(&qwen38_27b()).unwrap();
        // 16 layers x 4 kv heads x (256 + 256) x 2 bytes
        assert_eq!(a.kv_bytes(1, KvType::F16), 64 * 1024);
        // so 64k tokens are exactly 4 GiB at f16 and 2.125 GiB at q8_0
        assert_eq!(a.kv_bytes(65536, KvType::F16), 4 * 1024 * 1024 * 1024);
        assert_eq!(a.kv_bytes(65536, KvType::Q8_0), 2_281_701_376);
        // and the full native 256k is 16 GiB at f16
        assert_eq!(a.kv_bytes(262144, KvType::F16), 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn qwen38_27b_would_be_4x_larger_if_every_layer_had_kv() {
        let mut m = qwen38_27b();
        m.remove("qwen35.full_attention_interval");
        let a = Arch::from_model_info(&m).unwrap();
        assert_eq!(a.kv_bytes(1, KvType::F16), 256 * 1024);
    }

    #[test]
    fn qwen38_27b_recurrent_state() {
        let a = Arch::from_model_info(&qwen38_27b()).unwrap();
        // 48 Gated DeltaNet layers x (128 x 6144 + 3 x (6144 + 2 x 16 x 128)) x 4 bytes
        assert_eq!(a.recurrent_bytes, 48 * (128 * 6144 + 3 * (6144 + 4096)) * 4);
    }

    #[test]
    fn qwen38_27b_total_estimate() {
        let a = Arch::from_model_info(&qwen38_27b()).unwrap();
        let weights = 15_183_993_522; // UD-IQ4_XS, /api/tags size
        let e = estimate(weights, Some(&a), 65536, KvType::Q8_0, DEFAULT_OVERHEAD_BYTES, 0);
        let gib = e.total() as f64 / GIB;
        // 14.14 weights + 2.125 kv + 0.146 state + 0.0625 mask + 0.5 overhead
        assert!((gib - 16.975).abs() < 0.001, "{gib}");
    }

    #[test]
    fn gemma4_sliding_layers_hold_only_the_window() {
        let pattern: Vec<bool> = (0..48).map(|i| (i + 1) % 6 != 0).collect();
        let kv_heads: Vec<u64> = pattern.iter().map(|&s| if s { 8 } else { 1 }).collect();
        let a = Arch::from_model_info(&info(json!({
            "general.architecture": "gemma4",
            "gemma4.attention.head_count": 16,
            "gemma4.attention.head_count_kv": kv_heads,
            "gemma4.attention.key_length": 512,
            "gemma4.attention.value_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.value_length_swa": 256,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.attention.sliding_window_pattern": pattern,
            "gemma4.block_count": 48,
            "gemma4.context_length": 262144
        })))
        .unwrap();
        // 8 global layers x 1 head x 1024, 40 local layers x 8 heads x 512 over at most 1536 tokens
        let ctx = 131072u64;
        let want = 8 * 1024 * ctx + 40 * 8 * 512 * (1024 + 512);
        assert_eq!(a.kv_bytes(ctx, KvType::F16), want * 2);
        // the global layers' share at 32k and q8_0 is the 272 MiB llama.cpp reported for them
        assert_eq!(a.kv_bytes(32768, KvType::Q8_0) - 40 * 4096 * 1536 * 17 / 16, 272 * 1024 * 1024);
        // below the window both kinds scale with the context
        assert_eq!(a.kv_bytes(1, KvType::F16), (8 * 1024 + 40 * 4096) * 2);
    }

    #[test]
    fn plain_gqa_model_uses_every_layer() {
        let a = Arch::from_model_info(&info(json!({
            "general.architecture": "llama",
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.context_length": 131072
        })))
        .unwrap();
        // head_dim falls back to 4096 / 32 = 128: 32 x 8 x 256 x 2 = 128 KiB per token
        assert_eq!(a.kv_bytes(1, KvType::F16), 128 * 1024);
        assert_eq!(a.recurrent_bytes, 0);
    }

    #[test]
    fn per_layer_zero_kv_heads_mark_recurrent_layers() {
        let a = Arch::from_model_info(&info(json!({
            "general.architecture": "nemotron_h",
            "nemotron_h.attention.head_count": 32,
            "nemotron_h.attention.head_count_kv": [0, 0, 8, 0, 0, 8],
            "nemotron_h.attention.key_length": 128,
            "nemotron_h.attention.value_length": 128,
            "nemotron_h.block_count": 6
        })))
        .unwrap();
        assert_eq!(a.attention_layers(), 2);
        assert_eq!(a.kv_bytes(1, KvType::F16), 2 * 8 * 256 * 2);
    }

    #[test]
    fn capping() {
        assert_eq!(effective_ctx(262144, 131072), (131072, true));
        assert_eq!(effective_ctx(65536, 131072), (65536, false));
        assert_eq!(effective_ctx(65536, 0), (65536, false));
    }
}
