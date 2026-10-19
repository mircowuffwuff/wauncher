//! the launch sequence, run in the plain terminal between the two TUI sessions:
//! make sure the server runs with the right settings, load the model, start the
//! harness in the working directory, and unload the model when it exits

use crate::gpu;
use crate::harness::{self, Harness, Session};
use crate::ollama;
use crate::server;
use crate::state::{self, Correction, ServerSettings, State};
use crate::vram::{Estimate, KvType, GIB};
use anyhow::{bail, Context, Result};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub struct Request {
    pub model: String,
    pub ctx: u64,
    pub kv: KvType,
    pub harness: Harness,
    pub thinking: bool,
    pub vision: bool,
    pub estimate: Estimate,
    pub cwd: PathBuf,
}

fn log(msg: impl AsRef<str>) {
    println!("\x1b[2mwauncher:\x1b[0m {}", msg.as_ref());
}

fn warn(msg: impl AsRef<str>) {
    println!("\x1b[33mwauncher:\x1b[0m {}", msg.as_ref());
}

fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / GIB)
}

/// runs `f` on a thread and waits for it, giving up early when `cancel` is set.
/// the thread is left to finish on its own.
fn cancellable<T: Send + 'static>(cancel: &AtomicBool, f: impl FnOnce() -> Result<T> + Send + 'static) -> Result<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    loop {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(r) => return r,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(e) => bail!("worker thread vanished -- {e}"),
        }
    }
}

/// asks a yes/no question on the terminal; anything but y is no
fn confirm(question: &str) -> bool {
    print!("{question} [y/N] ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).is_ok() && line.trim().eq_ignore_ascii_case("y")
}

/// the server is restarted unless it is the one this launcher started, still
/// running, with exactly the settings wanted now
fn ensure_server(want: &ServerSettings, st: &mut State, cancel: &AtomicBool) -> Result<()> {
    let ours = st.server.as_ref().is_some_and(|o| o.settings == *want && server::is_running(o));
    if ours && ollama::alive(Duration::from_secs(2)) {
        log("Ollama is already running with these settings");
        return Ok(());
    }
    log("stopping the Ollama server and tray app");
    server::stop_all()?;
    st.server = None;
    st.save()?;

    let vars: Vec<String> = server::server_env(want).iter().map(|(k, v)| format!("{k}={v}")).collect();
    log(format!("starting ollama serve -- {}", vars.join(" ")));
    let log_path = state::dir().join("ollama-serve.log");
    st.server = Some(server::start_owned(want, &log_path)?);
    st.save()?;
    server::wait_ready(Duration::from_secs(60), cancel)
        .with_context(|| format!("waiting for the server, its log is {}", log_path.display()))
}

pub fn run(req: &Request, st: &mut State, cancel: &AtomicBool) -> Result<()> {
    cancel.store(false, Ordering::Relaxed);
    println!();
    log(format!(
        "{} with {}, {}k context, {} KV cache",
        req.harness.name(),
        req.model,
        req.ctx / 1024,
        req.kv.as_str()
    ));

    let Some(served) = load_checked(req, st, cancel)? else {
        return Ok(());
    };

    let session = Session { model: &served, ctx: req.ctx, thinking: req.thinking, vision: req.vision };
    let result = run_harness(req.harness, &session, req);

    log("unloading the model");
    if let Err(e) = ollama::load(&served, 0) {
        warn(format!("unloading failed -- {e:#}"));
    }
    drop_served(req, &served);
    result
}

/// the model the harness is pointed at: the selected one, or a model derived
/// from it with Ollama's built-in renderer where its own chat template cannot
/// take the harness's requests
fn served_model(req: &Request) -> Result<String> {
    if req.harness != Harness::Claude {
        return Ok(req.model.clone());
    }
    let show = ollama::show(&req.model)?;
    let Some(renderer) = ollama::renderer_for_claude(&show) else {
        return Ok(req.model.clone());
    };
    let name = ollama::derived_name(&req.model);
    log(format!(
        "the model's own chat template rejects Claude Code's requests -- serving it as {name} with Ollama's {renderer} renderer"
    ));
    ollama::create_with_renderer(&name, &req.model, renderer).context("creating the derived model")?;
    Ok(name)
}

/// removes a model served_model derived; it shares its blobs with the selected model
fn drop_served(req: &Request, served: &str) {
    if served != req.model {
        if let Err(e) = ollama::delete(served) {
            warn(format!("removing {served} failed -- {e:#}"));
        }
    }
}

/// steps 2 to 4: the server with the right settings, the model loaded and
/// measured, calibration saved. returns the model the harness is to use, or
/// None when the user declined a model that did not fit, which is then
/// unloaded again.
fn load_checked(req: &Request, st: &mut State, cancel: &AtomicBool) -> Result<Option<String>> {
    let want = ServerSettings { ctx: req.ctx, kv: req.kv, env: st.server_env() };
    ensure_server(&want, st, cancel)?;
    let served = served_model(req)?;

    // the model's real footprint is the rise in dedicated usage across the load.
    // a restarted server's old runner may still be releasing its memory, so the
    // baseline is taken after a moment
    let adapters = gpu::discrete_adapters();
    let usable = gpu::usable(&adapters, st.reserve());
    let mut sampler = gpu::UsageSampler::new().ok();
    std::thread::sleep(Duration::from_secs(1));
    let before = gpu::used(&mut sampler, &adapters);

    log(format!("loading {} (estimated {})", req.model, gib(req.estimate.total().max(0) as u64)));
    let started = Instant::now();
    let model = served.clone();
    if let Err(e) = cancellable(cancel, move || ollama::load(&model, -1)) {
        let _ = ollama::load(&served, 0);
        drop_served(req, &served);
        return Err(e.context("loading the model"));
    }
    log(format!("loaded in {:.0}s", started.elapsed().as_secs_f64()));
    std::thread::sleep(Duration::from_secs(2));
    let after = gpu::used(&mut sampler, &adapters);

    let running = ollama::ps()?;
    let r = ollama::find_running(&running, &served).context("the model is not in /api/ps after loading")?.clone();
    if r.context_length > 0 && r.context_length != req.ctx {
        warn(format!(
            "the model runs with {} tokens of context, not {} -- its Modelfile may set num_ctx",
            r.context_length, req.ctx
        ));
    }

    let mut spilled = Vec::new();
    if r.size_vram < r.size {
        spilled.push(format!(
            "Ollama put only {} of {} in VRAM, {} runs from system RAM",
            gib(r.size_vram),
            gib(r.size),
            gib(r.size - r.size_vram)
        ));
    }
    match (before, after) {
        (Some(b), Some(a)) if a <= usable && spilled.is_empty() => {
            // calibration against the measured rise rather than /api/ps: Ollama's size
            // leaves out the compute buffers, the vision projector and the driver's own
            // allocations, which together came to over 1 GiB for a 27B model
            let real = a.saturating_sub(b);
            let correction = real as i64 - req.estimate.uncorrected();
            st.corrections.insert(req.model.clone(), Correction { bytes: correction, ctx: req.ctx, kv: req.kv });
            st.save()?;
            log(format!(
                "the model takes {} of VRAM, estimated {} -- correction {:+.2} GiB saved",
                gib(real),
                gib(req.estimate.total().max(0) as u64),
                correction as f64 / GIB
            ));
        }
        (Some(_), Some(a)) => {
            if a > usable {
                // Windows evicts buffers to system RAM near this point while Ollama still
                // reports them in VRAM; the measured rise then undercounts, so it is not
                // used for calibration
                spilled.push(format!(
                    "{} of dedicated VRAM is in use, above the usable {} -- Windows moves what does not fit to system RAM, which makes prompt processing several times slower",
                    gib(a),
                    gib(usable)
                ));
            }
            log("not calibrated: the load did not fit in VRAM");
        }
        _ => log("not calibrated: the GPU memory counters are unavailable"),
    }

    if !spilled.is_empty() {
        for s in &spilled {
            warn(s);
        }
        if !confirm("continue anyway?") {
            log("unloading the model");
            ollama::load(&served, 0)?;
            drop_served(req, &served);
            return Ok(None);
        }
    }
    Ok(Some(served))
}

fn run_harness(h: Harness, s: &Session, req: &Request) -> Result<()> {
    let mut prepared = harness::prepare(h, s, &req.cwd)?;
    log(format!("starting {} in {}", h.name(), req.cwd.display()));
    println!();
    let status = prepared.command.status().with_context(|| format!("starting {}", h.name()));
    println!();
    for edit in prepared.edits.drain(..) {
        match edit.restore() {
            Ok(None) => {}
            Ok(Some(note)) => warn(note),
            Err(e) => warn(format!("{e:#}")),
        }
    }
    let status = status?;
    if !status.success() {
        warn(format!("{} exited with {status}", h.name()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::Harness;
    use crate::vram::{self, Arch};

    /// predicts Free After the way the model screen does, loads through the real
    /// launch path (server, load, measurement, calibration, spill check), measures
    /// the free VRAM that is really left, and does it again at a second context
    /// with the fresh calibration. models that do not fit are declined at the
    /// prompt, so stdin should not be a terminal:
    /// cargo test live_ -- --ignored --nocapture < NUL
    #[test]
    #[ignore = "stops and restarts the local Ollama server and loads models"]
    fn live_free_after() {
        let runs: Vec<(String, Vec<u64>)> = match std::env::var("WAUNCHER_LIVE_RUNS") {
            // "model=ctx,ctx;model=ctx"
            Ok(v) => v
                .split(';')
                .filter_map(|r| r.split_once('='))
                .map(|(m, c)| (m.to_string(), c.split(',').map(|k| k.parse::<u64>().unwrap() * 1024).collect()))
                .collect(),
            Err(_) => vec![("hf.co/unsloth/Qwen3.5-4B-GGUF:UD-Q4_K_XL".into(), vec![16384, 65536])],
        };
        let mut st = State::load();
        let cancel = AtomicBool::new(false);
        let adapters = crate::gpu::discrete_adapters();
        let usable = crate::gpu::usable(&adapters, st.reserve());
        let mut sampler = crate::gpu::UsageSampler::new().ok();
        let tags = ollama::tags().unwrap();
        println!("usable {:.2} GiB", usable as f64 / GIB);
        for (model, ctxs) in runs {
            st.corrections.remove(&model);
            let weights = tags.iter().find(|t| t.name == model).expect("model installed").size;
            let show = ollama::show(&model).unwrap();
            let arch = Arch::from_model_info(&show.model_info);
            for ctx in ctxs {
                let est =
                    vram::estimate(weights, arch.as_ref(), ctx, KvType::Q8_0, st.overhead(), st.correction(&model));
                let req = Request {
                    model: model.clone(),
                    ctx,
                    kv: KvType::Q8_0,
                    harness: Harness::Pi,
                    thinking: true,
                    vision: true,
                    estimate: est,
                    cwd: std::env::temp_dir(),
                };
                // the server is restarted first, so the prediction is made against what
                // other programs use once nothing is loaded
                let want = ServerSettings { ctx, kv: KvType::Q8_0, env: st.server_env() };
                ensure_server(&want, &mut st, &cancel).unwrap();
                std::thread::sleep(Duration::from_secs(1));
                let others = crate::gpu::used(&mut sampler, &adapters).unwrap();
                let predicted = usable as i64 - others as i64 - est.total();
                let calibrated = st.corrections.contains_key(&model);
                let served = load_checked(&req, &mut st, &cancel).unwrap();
                let proceeded = served.is_some();
                let after = crate::gpu::used(&mut sampler, &adapters).unwrap();
                let real = usable as i64 - after as i64;
                println!(
                    "RESULT {model} {}k {}: predicted free {:+.2} GiB, real free {:+.2} GiB, error {:+.2} GiB{}",
                    ctx / 1024,
                    if calibrated { "calibrated" } else { "fresh" },
                    predicted as f64 / GIB,
                    real as f64 / GIB,
                    (predicted - real) as f64 / GIB,
                    if proceeded { "" } else { " (declined: did not fit)" }
                );
                if let Some(served) = served {
                    let _ = ollama::load(&served, 0);
                    drop_served(&req, &served);
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }
}
