//! the two screens: model selection with the context slider, and harness selection

use crate::gpu::Adapter;
use crate::harness::{self, Harness};
use crate::launch::Request;
use crate::state::{State, CTX_MIN, CTX_STEP};
use crate::vram::{self, Arch, Estimate, GIB};
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState};
use ratatui::Frame;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ModelRow {
    pub name: String,
    pub quant: String,
    pub weights: u64,
    pub arch: Option<Arch>,
    pub thinking: bool,
    pub vision: bool,
}

impl ModelRow {
    pub fn native_ctx(&self) -> u64 {
        self.arch.as_ref().map(|a| a.native_ctx).unwrap_or(0)
    }
}

pub enum Msg {
    Models(Result<Vec<ModelRow>, String>),
    Status(String),
    /// dedicated VRAM in use on the discrete GPUs (None when the counters are
    /// unreadable), VRAM held by models Ollama has loaded, and their names
    Usage {
        used: Option<u64>,
        ollama: u64,
        loaded: Vec<String>,
    },
}

pub enum Action {
    None,
    Quit,
    Launch(Request),
}

#[derive(PartialEq, Eq)]
enum Screen {
    Models,
    Harness,
}

pub struct App {
    pub state: State,
    adapters: Vec<Adapter>,
    cwd: PathBuf,
    models: Option<Vec<ModelRow>>,
    models_err: Option<String>,
    status: Option<String>,
    used: Option<u64>,
    ollama_vram: u64,
    loaded: Vec<String>,
    screen: Screen,
    table: TableState,
    harness_sel: usize,
    installed: [bool; 5],
}

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;

fn gib(b: i64) -> String {
    let v = b as f64 / GIB;
    if v < 0.0 {
        format!("−{:.1} GiB", -v)
    } else {
        format!("{v:.1} GiB")
    }
}

fn k(ctx: u64) -> String {
    format!("{}k", ctx / 1024)
}

fn detect_installed() -> [bool; 5] {
    harness::ALL.map(|h| h.binary().is_some())
}

impl App {
    pub fn new(state: State, adapters: Vec<Adapter>, cwd: PathBuf) -> App {
        let harness_sel = state
            .last_harness
            .as_deref()
            .and_then(Harness::from_id)
            .and_then(|h| harness::ALL.iter().position(|&x| x == h))
            .unwrap_or(0);
        App {
            state,
            adapters,
            cwd,
            models: None,
            models_err: None,
            status: None,
            used: None,
            ollama_vram: 0,
            loaded: Vec::new(),
            screen: Screen::Models,
            table: TableState::default(),
            harness_sel,
            installed: detect_installed(),
        }
    }

    pub fn on_msg(&mut self, msg: Msg) {
        match msg {
            Msg::Models(Ok(models)) => {
                let sel =
                    self.state.last_model.as_deref().and_then(|m| models.iter().position(|r| r.name == m)).unwrap_or(0);
                self.table.select((!models.is_empty()).then_some(sel));
                self.models = Some(models);
                self.models_err = None;
                self.status = None;
            }
            Msg::Models(Err(e)) => self.models_err = Some(e),
            Msg::Status(s) => self.status = Some(s),
            Msg::Usage { used, ollama, loaded } => {
                self.used = used;
                self.ollama_vram = ollama;
                self.loaded = loaded;
            }
        }
    }

    fn usable_vram(&self) -> u64 {
        crate::gpu::usable(&self.adapters, self.state.reserve())
    }

    fn total_vram(&self) -> u64 {
        self.adapters.iter().map(|a| a.dedicated).sum()
    }

    /// the slider's upper bound: the largest native context among the listed models
    fn ctx_max(&self) -> u64 {
        let max = self.models.iter().flatten().map(|m| m.native_ctx()).max().unwrap_or(0);
        let max = if max == 0 { self.state.ctx } else { max };
        max.max(CTX_MIN)
    }

    /// the slider value as shown and used, held inside the current bounds
    fn slider(&self) -> u64 {
        self.state.ctx.clamp(CTX_MIN, self.ctx_max())
    }

    fn selected(&self) -> Option<&ModelRow> {
        self.models.as_ref()?.get(self.table.selected()?)
    }

    fn numbers(&self, m: &ModelRow) -> (u64, bool, Estimate, Option<i64>) {
        let (ctx, capped) = vram::effective_ctx(self.slider(), m.native_ctx());
        let est = vram::estimate(
            m.weights,
            m.arch.as_ref(),
            ctx,
            self.state.kv,
            self.state.overhead(),
            self.state.correction(&m.name),
        );
        // models Ollama has loaded are in the usage counters too; they would be counted twice
        let free = self.used.filter(|_| !self.adapters.is_empty()).map(|used| {
            let others = used.saturating_sub(self.ollama_vram);
            self.usable_vram() as i64 - others as i64 - est.total()
        });
        (ctx, capped, est, free)
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Action {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Action::Quit;
        }
        match self.screen {
            Screen::Models => self.on_key_models(key),
            Screen::Harness => self.on_key_harness(key),
        }
    }

    fn on_key_models(&mut self, key: KeyEvent) -> Action {
        let n = self.models.as_ref().map(|m| m.len()).unwrap_or(0);
        match key.code {
            KeyCode::Esc => return Action::Quit,
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                self.state.ctx = self.slider().saturating_sub(CTX_STEP).max(CTX_MIN);
            }
            KeyCode::Char('e') | KeyCode::Char('E') => {
                self.state.ctx = (self.slider() + CTX_STEP).min(self.ctx_max());
            }
            KeyCode::Char('k') | KeyCode::Char('K') => self.state.kv = self.state.kv.toggled(),
            KeyCode::Up if n > 0 => {
                let i = self.table.selected().unwrap_or(0);
                self.table.select(Some((i + n - 1) % n));
            }
            KeyCode::Down if n > 0 => {
                let i = self.table.selected().unwrap_or(0);
                self.table.select(Some((i + 1) % n));
            }
            KeyCode::Enter => {
                if let Some(m) = self.selected() {
                    self.state.last_model = Some(m.name.clone());
                    self.state.ctx = self.slider();
                    if let Err(e) = self.state.save() {
                        self.status = Some(format!("Could not save state: {e:#}"));
                    }
                    self.installed = detect_installed();
                    if !self.installed[self.harness_sel] {
                        self.move_harness(1);
                    }
                    self.screen = Screen::Harness;
                }
            }
            _ => {}
        }
        Action::None
    }

    /// moves the harness selection by `dir`, skipping harnesses that are not installed
    fn move_harness(&mut self, dir: isize) {
        let n = harness::ALL.len() as isize;
        let mut i = self.harness_sel as isize;
        for _ in 0..n {
            i = (i + dir).rem_euclid(n);
            if self.installed[i as usize] {
                self.harness_sel = i as usize;
                return;
            }
        }
    }

    fn on_key_harness(&mut self, key: KeyEvent) -> Action {
        match key.code {
            KeyCode::Esc => self.screen = Screen::Models,
            KeyCode::Up => self.move_harness(-1),
            KeyCode::Down => self.move_harness(1),
            KeyCode::Enter if self.installed[self.harness_sel] => {
                let Some(m) = self.selected().cloned() else { return Action::None };
                let h = harness::ALL[self.harness_sel];
                self.state.last_harness = Some(h.id().to_string());
                if let Err(e) = self.state.save() {
                    self.status = Some(format!("Could not save state: {e:#}"));
                }
                let (ctx, _, estimate, _) = self.numbers(&m);
                self.screen = Screen::Models;
                return Action::Launch(Request {
                    model: m.name,
                    ctx,
                    kv: self.state.kv,
                    harness: h,
                    thinking: m.thinking,
                    vision: m.vision,
                    estimate,
                    cwd: self.cwd.clone(),
                });
            }
            _ => {}
        }
        Action::None
    }

    pub fn draw(&mut self, f: &mut Frame) {
        match self.screen {
            Screen::Models => self.draw_models(f),
            Screen::Harness => self.draw_harness(f),
        }
    }

    fn title(&self) -> Line<'static> {
        let gpus = if self.adapters.is_empty() {
            "no discrete GPU found".to_string()
        } else {
            self.adapters.iter().map(|a| a.name.clone()).collect::<Vec<_>>().join(" + ")
        };
        Line::from(vec![
            Span::styled(" wauncher ", Style::new().fg(Color::Black).bg(ACCENT).bold()),
            Span::raw("  "),
            Span::styled(format!("{gpus} · {}", gib(self.total_vram() as i64)), Style::new().fg(DIM)),
            Span::raw("  "),
            Span::styled(self.cwd.display().to_string(), Style::new().fg(DIM)),
        ])
    }

    fn slider_line(&self, width: u16) -> Line<'static> {
        let value = self.slider();
        let max = self.ctx_max();
        let label = format!(" {:>5} ", k(value));
        let kv = format!(" KV {} ", self.state.kv.as_str());
        // "Q ◀ " + bar + " ▶ E" + label + kv
        let fixed = 4 + 4 + label.chars().count() + kv.chars().count() + 2;
        let bar_w = (width as usize).saturating_sub(fixed).max(10);
        let steps = ((max - CTX_MIN) / CTX_STEP).max(1);
        let pos = (((value - CTX_MIN) / CTX_STEP) as usize * (bar_w - 1)) / steps as usize;
        let filled: String = "━".repeat(pos);
        let empty: String = "─".repeat(bar_w - 1 - pos);
        let kv_style = match self.state.kv {
            vram::KvType::F16 => Style::new().fg(Color::Black).bg(Color::Magenta).bold(),
            vram::KvType::Q8_0 => Style::new().fg(Color::Black).bg(Color::Green).bold(),
        };
        Line::from(vec![
            Span::styled("Q ◀ ", Style::new().fg(DIM)),
            Span::styled(filled, Style::new().fg(ACCENT)),
            Span::styled("●", Style::new().fg(ACCENT).bold()),
            Span::styled(empty, Style::new().fg(DIM)),
            Span::styled(" ▶ E", Style::new().fg(DIM)),
            Span::styled(label, Style::new().bold()),
            Span::raw(" "),
            Span::styled(kv, kv_style),
        ])
    }

    fn draw_models(&mut self, f: &mut Frame) {
        let [title, slider, table, legend, usage, help] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(f.area());

        f.render_widget(Paragraph::new(self.title()), title);

        let block = Block::default().borders(Borders::ALL).border_style(Style::new().fg(DIM)).title(format!(
            " Context  {}–{} ",
            k(CTX_MIN),
            k(self.ctx_max())
        ));
        let inner = block.inner(slider);
        f.render_widget(block, slider);
        f.render_widget(Paragraph::new(self.slider_line(inner.width)), inner);

        self.draw_table(f, table);

        let models = self.models.as_deref().unwrap_or_default();
        let mut notes = Vec::new();
        if models.iter().any(|m| self.numbers(m).1) {
            notes.push(Line::from(vec![
                Span::styled("↓", Style::new().fg(Color::Yellow).bold()),
                Span::raw(format!(" capped at the model's native context; the slider stays at {}", k(self.slider()))),
            ]));
        }
        if models.iter().any(|m| self.state.corrections.contains_key(&m.name)) {
            notes.push(Line::from(vec![
                Span::styled("*", Style::new().fg(ACCENT).bold()),
                Span::raw(" calibrated against the last load   "),
                Span::styled("●", Style::new().fg(Color::Green)),
                Span::raw(" loaded now"),
            ]));
        }
        f.render_widget(Paragraph::new(notes).style(Style::new().fg(DIM)), legend);

        let used = match self.used {
            Some(u) => format!(
                "Usable VRAM {} · in use by other programs {}, by Ollama {}",
                gib(self.usable_vram() as i64),
                gib(u.saturating_sub(self.ollama_vram) as i64),
                gib(self.ollama_vram as i64)
            ),
            None => format!("Usable VRAM {} · usage counters unavailable", gib(self.usable_vram() as i64)),
        };
        let line = match &self.status {
            Some(s) => Line::from(vec![Span::styled(s.clone(), Style::new().fg(Color::Yellow))]),
            None => Line::from(Span::styled(used, Style::new().fg(DIM))),
        };
        f.render_widget(Paragraph::new(line), usage);

        f.render_widget(
            Paragraph::new(help_line(&[
                ("↑↓", "Select"),
                ("Enter", "Next"),
                ("Q/E", "Context"),
                ("K", "KV Cache"),
                ("Esc", "Quit"),
            ])),
            help,
        );
    }

    fn draw_table(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).border_style(Style::new().fg(DIM)).title(" Models ");
        let Some(models) = self.models.clone() else {
            let msg = match &self.models_err {
                Some(e) => Line::from(Span::styled(format!("Cannot reach Ollama: {e}"), Style::new().fg(Color::Red))),
                None => Line::from(Span::styled("Loading models…", Style::new().fg(DIM))),
            };
            f.render_widget(Paragraph::new(msg).block(block), area);
            return;
        };
        if models.is_empty() {
            let msg = Line::from(Span::styled("No installed model has the tools capability.", Style::new().fg(DIM)));
            f.render_widget(Paragraph::new(msg).block(block), area);
            return;
        }

        let header = Row::new(["Name", "Quant", "Weights", "Ctx", "VRAM Req", "Free After"].map(Cell::from))
            .style(Style::new().fg(DIM).add_modifier(Modifier::BOLD));
        let rows = models.iter().map(|m| {
            let (ctx, capped, est, free) = self.numbers(m);
            let loaded = self.loaded.iter().any(|l| l == &m.name);
            let name = Line::from(vec![
                Span::raw(m.name.clone()),
                if loaded { Span::styled(" ●", Style::new().fg(Color::Green)) } else { Span::raw("") },
            ]);
            let ctx_cell = if capped {
                Cell::from(Span::styled(format!("{}↓", k(ctx)), Style::new().fg(Color::Yellow).bold()))
            } else {
                Cell::from(k(ctx))
            };
            let req = if self.state.corrections.contains_key(&m.name) {
                format!("{}*", gib(est.total()))
            } else {
                format!("{} ", gib(est.total()))
            };
            let free_cell = match free {
                Some(v) if v < 0 => Cell::from(Span::styled(gib(v), Style::new().fg(Color::Red).bold())),
                Some(v) if (v as f64) < GIB => Cell::from(Span::styled(gib(v), Style::new().fg(Color::Yellow))),
                Some(v) => Cell::from(Span::styled(gib(v), Style::new().fg(Color::Green))),
                None => Cell::from(Span::styled("?", Style::new().fg(DIM))),
            };
            Row::new(vec![
                Cell::from(name),
                Cell::from(m.quant.clone()),
                Cell::from(gib(m.weights as i64)),
                ctx_cell,
                Cell::from(req),
                free_cell,
            ])
        });
        let widths = [
            Constraint::Fill(1),
            Constraint::Length(8),
            Constraint::Length(9),
            Constraint::Length(6),
            Constraint::Length(10),
            Constraint::Length(10),
        ];
        let table = Table::new(rows, widths)
            .header(header)
            .block(block)
            .column_spacing(2)
            .row_highlight_style(Style::new().bg(Color::Rgb(40, 50, 70)).bold())
            .highlight_symbol("▶ ");
        f.render_stateful_widget(table, area, &mut self.table);
    }

    fn draw_harness(&mut self, f: &mut Frame) {
        let [title, summary, list, note, help] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(3),
            Constraint::Length(harness::ALL.len() as u16 + 2),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(f.area());
        f.render_widget(Paragraph::new(self.title()), title);

        let Some(m) = self.selected().cloned() else { return };
        let (ctx, capped, est, free) = self.numbers(&m);
        let mut spans = vec![
            Span::styled(m.name.clone(), Style::new().bold()),
            Span::styled("   ctx ", Style::new().fg(DIM)),
            Span::raw(k(ctx)),
        ];
        if capped {
            spans.push(Span::styled("↓", Style::new().fg(Color::Yellow).bold()));
        }
        spans.extend([
            Span::styled("   KV ", Style::new().fg(DIM)),
            Span::raw(self.state.kv.as_str()),
            Span::styled("   needs ", Style::new().fg(DIM)),
            Span::raw(gib(est.total())),
        ]);
        if let Some(v) = free {
            let style = if v < 0 { Style::new().fg(Color::Red).bold() } else { Style::new().fg(Color::Green) };
            spans.extend([Span::styled("   free after ", Style::new().fg(DIM)), Span::styled(gib(v), style)]);
        }
        let block = Block::default().borders(Borders::ALL).border_style(Style::new().fg(DIM)).title(" Model ");
        f.render_widget(Paragraph::new(Line::from(spans)).block(block), summary);

        let rows = harness::ALL.iter().enumerate().map(|(i, &h)| {
            let installed = self.installed[i];
            let short = h.min_ctx().filter(|&min| ctx < min);
            let detail = match (installed, short) {
                (false, _) => Span::styled("not installed", Style::new().fg(DIM)),
                (true, Some(min)) => Span::styled(
                    format!("needs at least {}k context", min.div_ceil(1024)),
                    Style::new().fg(Color::Yellow),
                ),
                (true, None) => Span::styled(h.method(), Style::new().fg(DIM)),
            };
            let name_style =
                if installed { Style::new() } else { Style::new().fg(DIM).add_modifier(Modifier::CROSSED_OUT) };
            Row::new(vec![Cell::from(Span::styled(h.name(), name_style)), Cell::from(detail)])
        });
        let mut state = TableState::default().with_selected(Some(self.harness_sel));
        let table = Table::new(rows, [Constraint::Length(16), Constraint::Fill(1)])
            .block(Block::default().borders(Borders::ALL).border_style(Style::new().fg(DIM)).title(" Harness "))
            .column_spacing(2)
            .row_highlight_style(Style::new().bg(Color::Rgb(40, 50, 70)).bold())
            .highlight_symbol("▶ ");
        f.render_stateful_widget(table, list, &mut state);

        let mut lines =
            vec![Line::from(Span::styled(format!("Starts in {}", self.cwd.display()), Style::new().fg(DIM)))];
        if let Some(s) = &self.status {
            lines.push(Line::from(Span::styled(s.clone(), Style::new().fg(Color::Yellow))));
        }
        f.render_widget(Paragraph::new(lines), note);

        f.render_widget(Paragraph::new(help_line(&[("↑↓", "Select"), ("Enter", "Launch"), ("Esc", "Back")])), help);
    }
}

fn help_line(keys: &[(&str, &str)]) -> Line<'static> {
    let mut spans = Vec::new();
    for (key, what) in keys {
        spans.push(Span::styled(format!(" {key} "), Style::new().fg(Color::Black).bg(DIM)));
        spans.push(Span::styled(format!(" {what}   "), Style::new().fg(DIM)));
    }
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use serde_json::json;

    fn row(name: &str, weights: u64, native: u64) -> ModelRow {
        let info = json!({
            "general.architecture": "qwen35",
            "qwen35.attention.head_count": 24,
            "qwen35.attention.head_count_kv": 4,
            "qwen35.attention.key_length": 256,
            "qwen35.attention.value_length": 256,
            "qwen35.block_count": 65,
            "qwen35.context_length": native,
            "qwen35.full_attention_interval": 4,
            "qwen35.nextn_predict_layers": 1
        });
        ModelRow {
            name: name.into(),
            quant: "IQ4_XS".into(),
            weights,
            arch: Arch::from_model_info(info.as_object().unwrap()),
            thinking: true,
            vision: true,
        }
    }

    fn app() -> App {
        let adapters =
            vec![Adapter { name: "AMD Radeon RX 6950 XT".into(), luid: (0, 1), dedicated: 16 << 30, budget: 15 << 30 }];
        let st = State { ctx: 192 * 1024, ..State::default() };
        let mut a = App::new(st, adapters, PathBuf::from(r"Y:\Local Projects\wauncher"));
        a.on_msg(Msg::Models(Ok(vec![
            row("hf.co/unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS", 15_183_993_522, 262144),
            row("small-model:4b", 3_587_679_553, 131072),
        ])));
        a.on_msg(Msg::Usage { used: Some(2_500_000_000), ollama: 0, loaded: vec![] });
        a
    }

    fn render(a: &mut App) -> String {
        let mut t = Terminal::new(TestBackend::new(110, 20)).unwrap();
        t.draw(|f| a.draw(f)).unwrap();
        let buf = t.backend().buffer().clone();
        (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width).map(|x| buf[(x, y)].symbol().to_string()).collect::<String>().trim_end().to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn screens_render() {
        let mut a = app();
        let s = render(&mut a);
        println!("{s}");
        assert!(s.contains("192k"));
        assert!(s.contains("128k↓"), "the 128k model is capped");
        assert!(s.contains("capped at the model's native context"));
        assert_eq!(a.state.ctx, 192 * 1024, "capping leaves the slider alone");

        a.on_key(KeyEvent::from(KeyCode::Enter));
        let s = render(&mut a);
        println!("{s}");
        assert!(s.contains("Harness"));
    }

    #[test]
    fn slider_keys() {
        let mut a = app();
        a.on_key(KeyEvent::from(KeyCode::Char('e')));
        assert_eq!(a.state.ctx, 200 * 1024);
        for _ in 0..100 {
            a.on_key(KeyEvent::from(KeyCode::Char('e')));
        }
        assert_eq!(a.state.ctx, 256 * 1024, "max is the largest native context");
        for _ in 0..100 {
            a.on_key(KeyEvent::from(KeyCode::Char('q')));
        }
        assert_eq!(a.state.ctx, 8 * 1024);
        a.on_key(KeyEvent::from(KeyCode::Char('k')));
        assert_eq!(a.state.kv, vram::KvType::F16);
    }
}
