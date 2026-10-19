//! starting and stopping the Ollama server process

use crate::ollama;
use crate::state::{OwnedServer, ServerSettings};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// the image names of the Ollama tray app, of the CLI, which is also the server,
/// and of the llama-server runner Ollama starts for each loaded model
const TRAY_EXE: &str = "ollama app.exe";
const OLLAMA_EXE: &str = "ollama.exe";
const RUNNER_EXE: &str = "llama-server.exe";

pub fn ollama_exe() -> Option<PathBuf> {
    crate::harness::which("ollama").or_else(|| {
        let p = PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("Programs").join("Ollama").join(OLLAMA_EXE);
        p.is_file().then_some(p)
    })
}

/// the variables a server is started with on top of the inherited environment
pub fn server_env(s: &ServerSettings) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = s.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    env.push(("OLLAMA_CONTEXT_LENGTH".into(), s.ctx.to_string()));
    env.push(("OLLAMA_KV_CACHE_TYPE".into(), s.kv.as_str().into()));
    // a quantized KV cache needs flash attention; Ollama falls back to f16 without it
    env.push(("OLLAMA_FLASH_ATTENTION".into(), "1".into()));
    env
}

/// whether `owned` is still running, by pid and creation time
pub fn is_running(owned: &OwnedServer) -> bool {
    process_created(owned.pid) == Some(owned.created)
}

/// stops the tray app first, so it does not bring a server straight back, then
/// every ollama.exe, then the llama-server runners under Ollama's install
/// folder. a runner outlives a server that is killed, and goes on holding its
/// model's VRAM; a llama-server installed anywhere else is not Ollama's and is
/// left alone.
pub fn stop_all() -> Result<()> {
    for name in [TRAY_EXE, OLLAMA_EXE] {
        for pid in pids_named(name) {
            kill(pid).with_context(|| format!("stopping {name} (pid {pid})"))?;
        }
    }
    if let Some(root) = ollama_exe().and_then(|p| p.parent().map(Path::to_path_buf)) {
        for pid in pids_named(RUNNER_EXE) {
            if image_path(pid).is_some_and(|p| p.starts_with(&root)) {
                kill(pid).with_context(|| format!("stopping {RUNNER_EXE} (pid {pid})"))?;
            }
        }
    }
    Ok(())
}

/// starts `ollama serve` with exactly `settings` and returns it as owned
pub fn start_owned(settings: &ServerSettings, log: &Path) -> Result<OwnedServer> {
    let (pid, created) = start(&server_env(settings), log)?;
    Ok(OwnedServer { pid, created, settings: settings.clone() })
}

/// starts `ollama serve` without a window, detached from this console so the
/// harness's ctrl+c and the terminal closing do not reach it. `env` goes on top
/// of the inherited environment. returns the pid and creation time.
pub fn start(env: &[(String, String)], log: &Path) -> Result<(u32, u64)> {
    let exe = ollama_exe().context("ollama.exe not found on PATH or in %LOCALAPPDATA%\\Programs\\Ollama")?;
    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let out = std::fs::File::create(log).with_context(|| format!("creating {}", log.display()))?;
    let err = out.try_clone()?;

    let mut cmd = Command::new(&exe);
    cmd.arg("serve").envs(env.iter().cloned()).stdin(Stdio::null()).stdout(out).stderr(err);
    let child = spawn_detached(&mut cmd)?;
    let pid = child.id();
    let created = process_created(pid).context("reading the server's creation time")?;
    Ok((pid, created))
}

/// waits for the API to answer, giving up after `timeout` or when `cancel` is set
pub fn wait_ready(timeout: Duration, cancel: &AtomicBool) -> Result<()> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        if ollama::alive(Duration::from_millis(500)) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    bail!("Ollama did not answer on {} within {}s", ollama::BASE, timeout.as_secs())
}

#[cfg(windows)]
fn spawn_detached(cmd: &mut Command) -> Result<std::process::Child> {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    // breaking away from the terminal's job object keeps the server alive when the
    // terminal closes; a job that forbids breakaway refuses the flag outright
    cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB);
    match cmd.spawn() {
        Ok(c) => Ok(c),
        Err(_) => {
            cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
            cmd.spawn().context("starting ollama serve")
        }
    }
}

#[cfg(not(windows))]
fn spawn_detached(cmd: &mut Command) -> Result<std::process::Child> {
    cmd.spawn().context("starting ollama serve")
}

#[cfg(windows)]
fn pids_named(image: &str) -> Vec<u32> {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };

    let mut out = Vec::new();
    let Ok(snap) = (unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) }) else {
        return out;
    };
    let mut entry = PROCESSENTRY32W { dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
    let mut ok = unsafe { Process32FirstW(snap, &mut entry) }.is_ok();
    while ok {
        let len = entry.szExeFile.iter().position(|&c| c == 0).unwrap_or(entry.szExeFile.len());
        let name = String::from_utf16_lossy(&entry.szExeFile[..len]);
        if name.eq_ignore_ascii_case(image) {
            out.push(entry.th32ProcessID);
        }
        ok = unsafe { Process32NextW(snap, &mut entry) }.is_ok();
    }
    let _ = unsafe { CloseHandle(snap) };
    out
}

#[cfg(windows)]
fn kill(pid: u32) -> Result<()> {
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows::Win32::System::Threading::{
        OpenProcess, TerminateProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
    };

    // a process that is already gone is fine
    let Ok(h) = (unsafe { OpenProcess(PROCESS_TERMINATE | PROCESS_SYNCHRONIZE, false, pid) }) else {
        return Ok(());
    };
    let result = unsafe { TerminateProcess(h, 1) };
    let waited = unsafe { WaitForSingleObject(h, 10_000) };
    let _ = unsafe { CloseHandle(h) };
    result?;
    if waited != WAIT_OBJECT_0 {
        bail!("did not exit within 10s");
    }
    Ok(())
}

/// the full path of a process's executable
#[cfg(windows)]
fn image_path(pid: u32) -> Option<PathBuf> {
    use windows::core::PWSTR;
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut buf = [0u16; 1024];
    let mut len = buf.len() as u32;
    let r = unsafe { QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, PWSTR(buf.as_mut_ptr()), &mut len) };
    let _ = unsafe { CloseHandle(h) };
    r.ok()?;
    Some(PathBuf::from(String::from_utf16_lossy(&buf[..len as usize])))
}

#[cfg(not(windows))]
fn image_path(_pid: u32) -> Option<PathBuf> {
    None
}

/// the process creation time as a FILETIME value, or None when it is not running
#[cfg(windows)]
fn process_created(pid: u32) -> Option<u64> {
    use windows::Win32::Foundation::{CloseHandle, FILETIME, STILL_ACTIVE};
    use windows::Win32::System::Threading::{
        GetExitCodeProcess, GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut created = FILETIME::default();
    let (mut a, mut b, mut c) = (FILETIME::default(), FILETIME::default(), FILETIME::default());
    let times = unsafe { GetProcessTimes(h, &mut created, &mut a, &mut b, &mut c) };
    let mut code = 0u32;
    let exit = unsafe { GetExitCodeProcess(h, &mut code) };
    let _ = unsafe { CloseHandle(h) };
    times.ok()?;
    exit.ok()?;
    if code != STILL_ACTIVE.0 as u32 {
        return None;
    }
    Some(((created.dwHighDateTime as u64) << 32) | created.dwLowDateTime as u64)
}

#[cfg(not(windows))]
fn pids_named(_image: &str) -> Vec<u32> {
    Vec::new()
}

#[cfg(not(windows))]
fn kill(_pid: u32) -> Result<()> {
    Ok(())
}

#[cfg(not(windows))]
fn process_created(_pid: u32) -> Option<u64> {
    None
}
