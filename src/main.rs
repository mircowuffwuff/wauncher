//! wauncher: a terminal launcher for local coding agents on Ollama

mod app;
mod gpu;
mod harness;
mod launch;
mod ollama;
mod server;
mod state;
mod vram;

use anyhow::Result;
use app::{Action, App, ModelRow, Msg};
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Duration;

/// set by ctrl+c and ctrl+break while the launch sequence runs in the plain terminal
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// how often the used-VRAM figure is refreshed
const USAGE_INTERVAL: Duration = Duration::from_secs(2);

/// the harness shares this console, so its ctrl+c reaches the launcher too and
/// would kill it by default. a handler routine takes the event instead; unlike
/// ignoring ctrl+c outright, a handler is not inherited by child processes.
#[cfg(windows)]
fn install_ctrl_handler() {
    use windows::Win32::Foundation::{BOOL, FALSE, TRUE};
    use windows::Win32::System::Console::{SetConsoleCtrlHandler, CTRL_BREAK_EVENT, CTRL_C_EVENT};

    unsafe extern "system" fn handler(kind: u32) -> BOOL {
        if kind == CTRL_C_EVENT || kind == CTRL_BREAK_EVENT {
            INTERRUPTED.store(true, Ordering::Relaxed);
            TRUE
        } else {
            FALSE
        }
    }
    let _ = unsafe { SetConsoleCtrlHandler(Some(handler), true) };
}

#[cfg(not(windows))]
fn install_ctrl_handler() {}

/// lists the installed models with the tools capability
fn load_models() -> Result<Vec<ModelRow>> {
    let mut rows = Vec::new();
    for t in ollama::tags()? {
        if t.name.starts_with(ollama::OWN_NAMESPACE) {
            continue;
        }
        // a model removed between /api/tags and /api/show is skipped, not fatal
        let Ok(show) = ollama::show(&t.name) else { continue };
        if !show.capabilities.iter().any(|c| c == "tools") {
            continue;
        }
        let has = |c: &str| show.capabilities.iter().any(|x| x == c);
        rows.push(ModelRow {
            quant: t.details.quantization_level.clone(),
            weights: t.size,
            arch: vram::Arch::from_model_info(&show.model_info),
            thinking: has("thinking"),
            vision: has("vision"),
            name: t.name,
        });
    }
    rows.sort_by_key(|a| a.name.to_lowercase());
    Ok(rows)
}

/// loads the model list, starting Ollama first when nothing answers. a server
/// started here gets the user's own environment plus the launcher's extra
/// variables, and is not recorded as owned, so launching restarts it with the
/// exact settings.
fn models_worker(tx: Sender<Msg>, server_env: Vec<(String, String)>) {
    if !ollama::alive(Duration::from_secs(2)) {
        let _ = tx.send(Msg::Status("Ollama is not running, starting it…".into()));
        if let Err(e) = server::start(&server_env, &state::dir().join("ollama-serve.log")) {
            let _ = tx.send(Msg::Status(format!("Could not start Ollama: {e:#}")));
        }
    }
    loop {
        match load_models() {
            Ok(rows) => {
                let _ = tx.send(Msg::Models(Ok(rows)));
                return;
            }
            Err(e) => {
                if tx.send(Msg::Models(Err(format!("{e:#}")))).is_err() {
                    return;
                }
            }
        }
        std::thread::sleep(Duration::from_secs(3));
    }
}

fn usage_worker(tx: Sender<Msg>, adapters: Vec<gpu::Adapter>) {
    let mut sampler = gpu::UsageSampler::new().ok();
    loop {
        let used = gpu::used(&mut sampler, &adapters);
        let running = ollama::ps().unwrap_or_default();
        let msg = Msg::Usage {
            used,
            ollama: running.iter().map(|r| r.size_vram).sum(),
            loaded: running.into_iter().map(|r| r.name).collect(),
        };
        if tx.send(msg).is_err() {
            return;
        }
        std::thread::sleep(USAGE_INTERVAL);
    }
}

fn pause() {
    println!("\npress Enter to return");
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
}

/// `--list`: prints the GPUs and the model table once, at the saved slider and
/// KV type, without touching the server
fn list(st: &state::State, adapters: &[gpu::Adapter]) -> Result<()> {
    let gib = |b: i64| format!("{:.2}", b as f64 / vram::GIB);
    let used = gpu::used(&mut gpu::UsageSampler::new().ok(), adapters);
    let running = ollama::ps().unwrap_or_default();
    let ollama_vram: u64 = running.iter().map(|r| r.size_vram).sum();
    let total: u64 = adapters.iter().map(|a| a.dedicated).sum();
    let usable = gpu::usable(adapters, st.reserve());
    for a in adapters {
        println!(
            "gpu  {} -- {} GiB, budget {} GiB -- luid {:08x}_{:08x}",
            a.name,
            gib(a.dedicated as i64),
            gib(a.budget as i64),
            a.luid.0,
            a.luid.1
        );
    }
    let others = used.map(|u| u.saturating_sub(ollama_vram));
    println!(
        "vram total {} GiB, usable {} GiB, in use {} GiB, of that Ollama {} GiB",
        gib(total as i64),
        gib(usable as i64),
        used.map(|u| gib(u as i64)).unwrap_or("?".into()),
        gib(ollama_vram as i64)
    );
    println!("slider {}k, kv {}\n", st.ctx / 1024, st.kv.as_str());
    println!(
        "{:<48} {:>8} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8}",
        "model", "quant", "weights", "ctx", "attn", "kv", "req", "free"
    );
    for m in load_models()? {
        let (ctx, capped) = vram::effective_ctx(st.ctx, m.native_ctx());
        let est = vram::estimate(m.weights, m.arch.as_ref(), ctx, st.kv, st.overhead(), st.correction(&m.name));
        let attn = m
            .arch
            .as_ref()
            .map(|a| {
                format!("{}/{}", a.layers.iter().filter(|l| **l != vram::LayerCache::None).count(), a.layers.len())
            })
            .unwrap_or("?".into());
        let free = others.map(|o| gib(usable as i64 - o as i64 - est.total())).unwrap_or("?".into());
        println!(
            "{:<48} {:>8} {:>8} {:>6}{} {:>8} {:>8} {:>8} {:>8}",
            m.name,
            m.quant,
            gib(m.weights as i64),
            format!("{}k", ctx / 1024),
            if capped { "↓" } else { " " },
            attn,
            gib(est.kv as i64),
            gib(est.total()),
            free
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    install_ctrl_handler();
    let cwd = std::env::current_dir()?;
    let st = state::State::load();
    let adapters = gpu::discrete_adapters();
    if std::env::args().any(|a| a == "--list") {
        return list(&st, &adapters);
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let extra_env: Vec<(String, String)> = st.server_env().into_iter().collect();
    {
        let tx = tx.clone();
        std::thread::spawn(move || models_worker(tx, extra_env));
    }
    {
        let adapters = adapters.clone();
        std::thread::spawn(move || usage_worker(tx, adapters));
    }

    let mut app = App::new(st, adapters, cwd);
    let mut terminal = ratatui::init();
    let result = loop {
        while let Ok(msg) = rx.try_recv() {
            app.on_msg(msg);
        }
        if let Err(e) = terminal.draw(|f| app.draw(f)) {
            break Err(e.into());
        }
        match event::poll(Duration::from_millis(200)) {
            Ok(false) => continue,
            Ok(true) => {}
            Err(e) => break Err(e.into()),
        }
        let key = match event::read() {
            Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => key,
            Ok(_) => continue,
            Err(e) => break Err(e.into()),
        };
        match app.on_key(key) {
            Action::None => {}
            Action::Quit => break Ok(()),
            Action::Launch(req) => {
                ratatui::restore();
                if let Err(e) = launch::run(&req, &mut app.state, &INTERRUPTED) {
                    println!("\x1b[31mwauncher:\x1b[0m launch failed -- {e:#}");
                    pause();
                }
                terminal = ratatui::init();
                terminal.clear()?;
            }
        }
    };
    ratatui::restore();
    result
}
