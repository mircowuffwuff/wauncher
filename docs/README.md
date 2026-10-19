# wauncher

a terminal launcher for local coding agents on Windows. pick an Ollama model and a context size, see what it costs in VRAM before anything loads, then start Claude Code, Pi, Hermes Agent, OpenCode or Qwen Code against it in the current folder. it is one self-contained `.exe` that runs in PowerShell and Windows Terminal.

## why Rust

wauncher is written in Rust with ratatui. it builds to a single static `.exe` with no runtime to install, and the `windows` crate gives direct access to the three Windows APIs the VRAM figures need — DXGI, PDH and the process APIs — so there is no PowerShell or WMI round trip on every refresh. Go with Bubble Tea would do the same job; Rust is used because its toolchain is the one already set up here.

## build

needs the Rust toolchain with the MSVC target (`rustup` default on Windows).

```bash
cargo build --release
```

the binary is `target\release\wauncher.exe`. `.cargo/config.toml` links the C runtime statically, so it needs no Visual C++ redistributable — copy it anywhere on `PATH` and run `wauncher` from the folder you want the agent to work in.

```bash
cargo test
```

runs the unit tests, the VRAM math among them. `cargo test live_ -- --ignored --nocapture < NUL` checks Free After against the real machine: for each model it predicts the free VRAM the way the model screen does, loads the model through the launch path — server restart, load, measurement, calibration, spill check — and prints the prediction beside the free VRAM really left, then does the same at the next context with the fresh calibration. `WAUNCHER_LIVE_RUNS` picks the runs, as `model=16,64;other-model=32`, contexts in k. it stops the local Ollama; a model that does not fit is declined at the prompt, which is why stdin comes from `NUL`. `cargo test stand_ins -- --ignored --nocapture` puts batch-file stand-ins for Pi, Hermes Agent and Qwen Code first on `PATH`, with their config folders in a temp directory, and prints the working directory, arguments, environment and config each one is started with.

`wauncher --list` prints the GPUs and the model table once and exits, without touching the server.

## use

**Model screen.** `Q` and `E` move the context slider in 8k steps, from 8k up to the largest native context among the listed models. `K` switches the KV cache between f16 and q8_0. `↑` `↓` select a model, `Enter` goes on, `Esc` quits.

the table lists every installed model with the `tools` capability:

| column | meaning |
| --- | --- |
| Ctx | the slider value, capped at the model's native context. a capped value shows as `128k↓` in yellow; the slider itself keeps its value |
| VRAM Req | the dedicated VRAM the model takes at that context and KV type. `*` marks a model calibrated by an earlier load |
| Free After | usable VRAM − VRAM other programs use − VRAM Req. red when negative: Windows would move part of the model to system RAM. yellow below 1 GiB |

the used-VRAM figure refreshes every two seconds. a model Ollama has loaded right now is marked `●`.

**Harness screen.** harnesses that are not installed are greyed out and cannot be selected. `Enter` launches, `Esc` goes back.

## what a launch does

1. the working directory is the folder wauncher was started from.
2. `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_KV_CACHE_TYPE` and `OLLAMA_FLASH_ATTENTION` are read once, when the server starts. unless the running server is one wauncher started with exactly these settings — checked by pid and process creation time — wauncher stops the Ollama tray app, every `ollama.exe` and the `llama-server.exe` runners under Ollama's install folder — a runner outlives a killed server and keeps its model's VRAM — then starts `ollama serve` hidden, with its output in `%APPDATA%\wauncher\ollama-serve.log`. changing the slider never restarts anything; only a launch does.
3. it waits for the API, then loads the model with `keep_alive: -1` and `num_batch: 512` (see [the micro-batch](#the-micro-batch)).
4. it measures how much dedicated VRAM the load took and saves that as the model's calibration. if the load went over the usable VRAM, or `/api/ps` shows `size_vram < size`, part of the model lives in system RAM; wauncher says so and asks before going on.
5. the harness starts in the same terminal and working directory, and wauncher waits for it to exit.
6. the model is unloaded with `keep_alive: 0`, any config file edited for the session is restored, and the model screen comes back.

`ctrl+c` during the launch steps cancels them. while the harness runs, `ctrl+c` belongs to the harness — wauncher takes the console event without exiting.

stopping every `ollama.exe` also ends an `ollama launch` running in another terminal, since that is the same executable. the agent it started keeps running, but its model is gone until the next request reloads it.

### the server wauncher starts

on top of the inherited environment it sets the three variables above and these, which `server_env` in `state.json` adds to or overrides:

- `OLLAMA_VULKAN=1` — the tray app switches the Vulkan backend on through its own settings, not the environment, so a plain `ollama serve` would come up without it
- `OLLAMA_KEEP_ALIVE=-1` — every request a harness makes resets the model's keep-alive to the server default; `-1` keeps the model loaded until wauncher unloads it
- `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` — see below. Ollama passes its environment on to the llama-server process that runs the model

#### the host-visible VRAM trap

with resizable BAR on, all of VRAM is also exposed to Vulkan as a host-visible memory type, and llama.cpp's Vulkan backend allocates its buffers from that type by default. on Windows with AMD's driver, only about 8.4 GiB of such allocations stays in VRAM; the rest is backed by system RAM, while Ollama still reports the whole model on the GPU. measured on an RX 6950 XT with Qwen3.8-27B UD-IQ3_XXS at 8k context:

| | generation | prompt processing | dedicated VRAM | shared (system RAM) |
| --- | --- | --- | --- | --- |
| default | 7.1 tok/s | 83 tok/s | +8.4 GiB | +3.0 GiB |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | 34 tok/s | 288 tok/s | +11.4 GiB | +0.7 GiB |

the variable makes llama.cpp use the plain device-local memory type. `GGML_VK_ENABLE_MEMORY_PRIORITY` does not help (7.1 tok/s), and running the server with or without a window makes no difference. models under about 8 GiB are not affected either way.

the tray app stays stopped after a launch. start it again from the Start menu to go back to Ollama's own server.

### how each harness is pointed at Ollama

| harness | method | files touched |
| --- | --- | --- |
| Claude Code | `ollama launch claude --model <m>`, plus `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and `CLAUDE_CODE_AUTO_COMPACT_WINDOW` set to the effective context, which `ollama launch` only sets for cloud models | none |
| OpenCode | `ollama launch opencode --model <m>`, which passes its config in `OPENCODE_CONFIG_CONTENT` | none |
| Qwen Code | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` and `--auth-type openai --model <m>`; environment and flags rank above `settings.json` | none |
| Pi | a `wauncher` provider in `models.json` (`%PI_CODING_AGENT_DIR%` or `~\.pi\agent`), then `pi --provider wauncher --model <m>`. Pi has no flag or variable for a custom endpoint | `models.json`, restored on exit |
| Hermes Agent | the `model:` block Hermes documents for a local Ollama — `provider: custom`, `base_url`, `context_length`, `ollama_num_ctx` — in `config.yaml` (`%HERMES_HOME%` or `%LOCALAPPDATA%\hermes`). `OPENAI_BASE_URL` alone loses to a `model.provider` already in `config.yaml` | `config.yaml`, restored on exit |

`ollama launch` starts both of its harnesses without setting a working directory, so they inherit wauncher's. a harness that is not on `PATH` is also looked for where its installer puts it: `~\.local\bin` for Claude Code, `%APPDATA%\npm` for the npm-installed ones, and the Hermes venv under `%LOCALAPPDATA%\hermes`.

an edited file is copied to `<file>.wauncher-backup` first. on exit the original goes back, unless something else changed the file during the session — then the file stays as it is, and the backup stays beside it. a backup still present at the next launch means a session never got to restore it, and wauncher takes the backup as the original.

Hermes needs at least 64,000 tokens of context; the harness screen warns when the effective context is below that.

## the VRAM estimate

**Free After** is what the model screen is for: whether the model fits, at this context, without Windows moving part of it to system RAM.

```
Free After = usable VRAM − VRAM other programs use − VRAM Req
```

### usable VRAM

Windows starts evicting a process's video memory to system RAM before the GPU is full, and Ollama does not notice. on the RX 6950 XT (15.95 GiB reported), with Qwen3.8-27B UD-IQ3_XXS through Ollama:

| context | dedicated VRAM in use after the load | evicted to system RAM | prompt processing | generation |
| --- | --- | --- | --- | --- |
| 8k | 13.66 GiB | — | 286 tok/s | 34.2 tok/s |
| 24k | 14.16 GiB | — | 286 tok/s | 34.1 tok/s |
| 32k | 14.43 GiB | — | 285 tok/s | 34.1 tok/s |
| 40k | 14.63 GiB | ~0.1 GiB | 60 tok/s | 34.2 tok/s |
| 52k | 14.69 GiB | ~0.5 GiB | 60 tok/s | 34.2 tok/s |
| 64k | 14.59 GiB | ~1.1 GiB | 60 tok/s | 34.2 tok/s |

past about 14.5 GiB, usage stops rising and the rest goes to system RAM. even 0.1 GiB of that cut prompt processing to a fifth, since it is the compute buffers that go, while generation speed looks normal. usable VRAM is therefore each discrete GPU's size less a reserve of 1.5 GiB — 14.45 GiB here — and `reserve_bytes` in `state.json` overrides the reserve. the DXGI budget is no better a guide: it read 15.20 GiB with nothing loaded and 13.78 GiB with a model holding 11 GiB, while the point where eviction starts stayed put.

**total** is the sum of `DedicatedVideoMemory` from DXGI over the discrete GPUs. DXGI reports it as a 64-bit value, unlike WMI's `AdapterRAM`, which stops at 4 GiB. an adapter counts as discrete unless it is a software adapter, has Microsoft's vendor id, or has less than 2 GiB of dedicated memory — an iGPU's "dedicated" memory is only the few hundred MiB the firmware reserves from system RAM.

**used** is the `\GPU Adapter Memory(*)\Dedicated Usage` performance counter, summed over the discrete GPUs. its instances are named after the adapter LUID, which is how they are matched to the DXGI adapters. models Ollama has loaded are in that counter too, so their `size_vram` from `/api/ps` is subtracted. that figure leaves out a model's compute buffers and vision projector, so while a model is loaded, Free After for the others reads low rather than high.

### VRAM Req

```
weights + KV cache + recurrent state + attention mask + overhead + calibration
```

**weights** are the model's size from `/api/tags`, which includes the vision projector.

**KV cache** is worked out per layer from the GGUF metadata in `/api/show` `model_info`:

```
layers with a cache × n_kv_heads × (key_length + value_length) × ctx × bytes per element
```

with 2 bytes per element for f16 and 34/32 = 1.0625 for q8_0 (blocks of 32 int8 values plus one f16 scale). which layers have a cache comes from the metadata, not from assuming all of them do:

- `<arch>.full_attention_interval` — hybrid linear-attention models such as `qwen35`. only every n-th layer is full attention; the others are Gated DeltaNet layers with a fixed-size state
- `<arch>.attention.head_count_kv` as a per-layer array — a 0 marks a recurrent layer (`nemotron_h`, `granitehybrid`, …)
- `<arch>.attention.sliding_window_pattern` — sliding-window layers hold at most the window plus one 512-token micro-batch, with `key_length_swa` and `value_length_swa` (`gemma3`, `gemma4`)
- `<arch>.nextn_predict_layers` — multi-token-prediction layers at the end of the block list, which hold no cache
- `<arch>.attention.kv_lora_rank` — multi-head latent attention caches one latent plus the rope dimensions per token

for Qwen3.8-27B, Ollama reports `qwen35.block_count = 65`, `nextn_predict_layers = 1`, `full_attention_interval = 4`, `head_count_kv = 4` and `key_length = value_length = 256`. that gives 64 layers, 16 of them with a cache, and 16 × 4 × 512 × 2 = 65,536 bytes — 64 KiB per token at f16. 64k tokens are 4 GiB at f16 and 2.125 GiB at q8_0, the KV buffer llama.cpp reports for it to the MiB; Gemma 4 12B's 612 MiB at 32k matches too. the tests in `src/vram.rs` check these figures.

**recurrent state** of hybrid models is `d_state × d_inner + (conv_kernel − 1) × (d_inner + 2 × groups × d_state)` f32 values per recurrent layer — 149.6 MiB for Qwen3.8-27B, as llama.cpp reports, independent of context.

**attention mask** is the part of llama.cpp's compute buffer that grows with the context: one f16 per cached token per micro-batch token, `ctx × 512 × 2` bytes — 128 MiB at 128k. loaded with a 1024 micro-batch, Gemma 4 12B's compute buffer grew from 32k to 128k by exactly the mask's share, 96k × 1024 × 2 bytes = 192 MiB.

**overhead** stands for the rest of what the metadata does not show: the fixed part of the compute buffers, the vision projector's compute buffer and the driver's own allocations, less the token embeddings, which stay in system RAM although `/api/tags` counts them. measured as real usage minus the terms above, it ranged from −0.40 GiB (Ornith 9B) to +0.29 GiB (Qwen3.5-4B). the default of 0.5 GiB keeps a first estimate from being optimistic; `overhead_bytes` in `state.json` overrides it.

### the micro-batch

left to itself, Ollama picks llama-server's micro-batch per load — 512, 1024 or 2048, from the context and its own memory prediction — and the compute buffers grow with it: Qwen3.5-4B got 1024 at 16k and 2048 at 64k, and its compute buffer went from 170 to 532 MiB. the same model could then need a different amount on every load, and a calibration made at one context would not carry to another. wauncher loads every model with `num_batch: 512`. that is also the default every request from a harness carries, and Ollama compares the two before reusing a runner: any other explicit value would make the harness's first request reload the model. a larger micro-batch can process long prompts faster on small models, which this gives up; large models near the VRAM limit get 512 from Ollama anyway.

**calibration.** wauncher reads the dedicated-usage counters just before a load and just after it, and the difference is what the model really takes. its difference from the uncorrected estimate is stored per model in `state.json` and added to every later estimate for it. `/api/ps` is no use here: its `size` leaves out the compute buffers, the vision projector and the driver's allocations, over 1 GiB for a 27B model. a load that went over the usable VRAM is not used for calibration, because the evicted part is missing from the measurement.

measured with the live test below: each model predicted fresh, loaded to calibrate, then predicted at a second context and loaded again.

| model | fresh prediction error | calibrated at | predicted at | calibrated prediction error |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B UD-Q4_K_XL | −0.71 GiB at 16k | 16k | 128k | +0.01 GiB |
| Ornith 1.5 9B Q5_K_M | −1.40 GiB at 32k | 32k | 128k | −0.04 GiB |
| Gemma 4 12B UD-IQ3_XXS | −0.74 GiB at 32k | 32k | 128k | +0.10 GiB |
| Qwen3.8-27B UD-IQ3_XXS | −1.29 GiB at 8k | 8k | 24k | +0.01 GiB |

a negative error is a pessimistic prediction, and the fresh column was measured with a 1 GiB overhead; with the default of 0.5 GiB it is half a GiB smaller. at 32k, Qwen3.8-27B predicted −0.15 GiB free and loaded to −0.11 GiB, so the launch flagged the eviction and the model was declined.

## files

| path | contents |
| --- | --- |
| `%APPDATA%\wauncher\state.json` | last model, last harness, slider value, KV type, calibration, server settings, the server wauncher started |
| `%APPDATA%\wauncher\ollama-serve.log` | output of the last `ollama serve` wauncher started |

the last model is saved when `Enter` leaves the model screen, together with the slider and KV type; the last harness when `Enter` launches it. an unreadable `state.json` is moved to `state.json.broken` and the defaults are used.

## out of scope

MCP servers, MTP speculative decoding, llama-server as a backend beside Ollama, and browser-based harnesses are not part of wauncher. the parts they would touch are separate: the server process in `src/server.rs`, the harnesses in `src/harness.rs`, the API client in `src/ollama.rs`, and the estimate in `src/vram.rs`, which only needs model metadata. another backend is new modules beside these rather than a rewrite.
