use crate::api::{self, ChatEvent, ChatMessage, ChatRequest};
use crate::theme;
use egui::{Align, Color32, CornerRadius, Layout, Margin, RichText, ScrollArea, Vec2};
use std::sync::mpsc;
use std::time::Instant;

#[derive(PartialEq, Clone, Copy)]
enum Role {
    User,
    Assistant,
}

struct DisplayMessage {
    role: Role,
    content: String,
}

#[derive(Clone, Copy)]
enum PaneSide {
    Left,
    Right,
}

struct PaneState {
    label: String,
    endpoint: String,
    model: String,
    available_models: Vec<String>,
    messages: Vec<DisplayMessage>,
    input: String,
    request_in_flight: bool,
    last_stats: Option<String>,
    response_rx: Option<mpsc::Receiver<ChatEvent>>,
    model_rx: Option<mpsc::Receiver<Vec<String>>>,
    completed_at: Option<Instant>,
}

impl PaneState {
    fn new(label: &str, endpoint: &str) -> Self {
        Self {
            label: label.into(),
            endpoint: endpoint.into(),
            model: String::new(),
            available_models: Vec::new(),
            messages: Vec::new(),
            input: String::new(),
            request_in_flight: false,
            last_stats: None,
            response_rx: None,
            model_rx: None,
            completed_at: None,
        }
    }
}

pub struct ChatApp {
    temperature: f32,
    max_tokens: u32,
    system_prompt: String,
    show_system_prompt: bool,
    compare_prompt: String,
    compare_started: Option<Instant>,
    left: PaneState,
    right: PaneState,
    runtime: tokio::runtime::Handle,
    theme_applied: bool,
}

impl ChatApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, runtime: tokio::runtime::Handle) -> Self {
        let mut app = Self {
            temperature: 0.7,
            max_tokens: 2048,
            system_prompt: "You are a helpful assistant.".into(),
            show_system_prompt: false,
            compare_prompt: String::new(),
            compare_started: None,
            left: PaneState::new("Endpoint A", "http://127.0.0.1:8080/v1/chat/completions"),
            right: PaneState::new("Endpoint B", "http://127.0.0.1:8081/v1/chat/completions"),
            runtime,
            theme_applied: false,
        };
        app.refresh_models(PaneSide::Left);
        app.refresh_models(PaneSide::Right);
        app
    }

    fn pane(&self, side: PaneSide) -> &PaneState {
        match side {
            PaneSide::Left => &self.left,
            PaneSide::Right => &self.right,
        }
    }

    fn pane_mut(&mut self, side: PaneSide) -> &mut PaneState {
        match side {
            PaneSide::Left => &mut self.left,
            PaneSide::Right => &mut self.right,
        }
    }

    fn refresh_models(&mut self, side: PaneSide) {
        let endpoint = self.pane(side).endpoint.clone();
        let (tx, rx) = mpsc::channel();
        self.pane_mut(side).model_rx = Some(rx);
        let _guard = self.runtime.enter();
        api::fetch_models(endpoint, tx);
    }

    fn api_messages(&self, side: PaneSide) -> Vec<ChatMessage> {
        let mut messages = Vec::new();
        if !self.system_prompt.trim().is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: self.system_prompt.trim().into(),
            });
        }
        messages.extend(self.pane(side).messages.iter().map(|message| {
            ChatMessage {
                role: match message.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                }
                .into(),
                content: message.content.clone(),
            }
        }));
        messages
    }

    fn send(&mut self, side: PaneSide, prompt: String) {
        if prompt.trim().is_empty() || self.pane(side).request_in_flight {
            return;
        }

        self.pane_mut(side).messages.push(DisplayMessage {
            role: Role::User,
            content: prompt.trim().into(),
        });
        let request = ChatRequest {
            model: {
                let model = &self.pane(side).model;
                if model.is_empty() {
                    "default".into()
                } else {
                    model.clone()
                }
            },
            messages: self.api_messages(side),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            stream: false,
        };

        let endpoint = self.pane(side).endpoint.clone();
        let (tx, rx) = mpsc::channel();
        let pane = self.pane_mut(side);
        pane.messages.push(DisplayMessage {
            role: Role::Assistant,
            content: String::new(),
        });
        pane.response_rx = Some(rx);
        pane.request_in_flight = true;
        pane.last_stats = None;
        pane.completed_at = None;

        let _guard = self.runtime.enter();
        api::request_chat(endpoint, request, tx);
    }

    fn send_input(&mut self, side: PaneSide) {
        let prompt = std::mem::take(&mut self.pane_mut(side).input);
        self.send(side, prompt);
    }

    fn compare(&mut self) {
        let prompt = self.compare_prompt.trim().to_string();
        if prompt.is_empty() || self.left.request_in_flight || self.right.request_in_flight {
            return;
        }
        self.compare_started = Some(Instant::now());
        self.send(PaneSide::Left, prompt.clone());
        self.send(PaneSide::Right, prompt);
    }

    fn poll(&mut self, side: PaneSide) {
        if let Some(rx) = &self.pane(side).model_rx {
            match rx.try_recv() {
                Ok(models) => {
                    let pane = self.pane_mut(side);
                    pane.available_models = models;
                    if pane.model.is_empty() {
                        pane.model = pane.available_models.first().cloned().unwrap_or_default();
                    }
                    pane.model_rx = None;
                }
                Err(mpsc::TryRecvError::Disconnected) => self.pane_mut(side).model_rx = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }

        loop {
            let event = match self.pane(side).response_rx.as_ref().map(|rx| rx.try_recv()) {
                Some(Ok(event)) => event,
                Some(Err(mpsc::TryRecvError::Empty)) | None => break,
                Some(Err(mpsc::TryRecvError::Disconnected)) => {
                    let pane = self.pane_mut(side);
                    pane.request_in_flight = false;
                    pane.response_rx = None;
                    break;
                }
            };
            let pane = self.pane_mut(side);
            match event {
                ChatEvent::Content(content) => {
                    if let Some(message) = pane.messages.last_mut() {
                        message.content.push_str(&content);
                    }
                }
                ChatEvent::Done { elapsed_secs } => {
                    pane.last_stats = Some(format!("Response in {elapsed_secs:.1}s"));
                    pane.request_in_flight = false;
                    pane.response_rx = None;
                    pane.completed_at = Some(Instant::now());
                    break;
                }
                ChatEvent::Error(error) => {
                    if let Some(message) = pane.messages.last_mut() {
                        if message.content.is_empty() {
                            message.content = format!("[Error: {error}]");
                        }
                    }
                    pane.request_in_flight = false;
                    pane.response_rx = None;
                    pane.completed_at = Some(Instant::now());
                    break;
                }
            }
        }
    }

    fn render_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("rvLLM endpoint comparison")
                    .size(18.0)
                    .strong(),
            );
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.compare_prompt)
                    .desired_width((ui.available_width() - 170.0).max(200.0))
                    .hint_text("Prompt both endpoints"),
            );
            let enter = response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let enabled = !self.compare_prompt.trim().is_empty()
                && !self.left.request_in_flight
                && !self.right.request_in_flight;
            if ui
                .add_enabled(enabled, egui::Button::new("Compare"))
                .clicked()
                || (enter && enabled)
            {
                self.compare();
            }
        });

        ui.horizontal(|ui| {
            ui.label(format!("Temperature {:.2}", self.temperature));
            ui.add(egui::Slider::new(&mut self.temperature, 0.0..=1.0));
            ui.label(format!("Max output {}", self.max_tokens));
            ui.add(egui::Slider::new(&mut self.max_tokens, 1..=4096).logarithmic(true));
            if ui.button("System prompt").clicked() {
                self.show_system_prompt = !self.show_system_prompt;
            }
            if let (Some(start), Some(left), Some(right)) = (
                self.compare_started,
                self.left.completed_at,
                self.right.completed_at,
            ) {
                let left_secs = left.duration_since(start).as_secs_f64();
                let right_secs = right.duration_since(start).as_secs_f64();
                ui.label(format!("A {left_secs:.2}s · B {right_secs:.2}s"));
            }
        });

        if self.show_system_prompt {
            ui.text_edit_singleline(&mut self.system_prompt);
        }
        ui.separator();
    }

    fn render_pane(
        pane: &mut PaneState,
        ui: &mut egui::Ui,
        id: &str,
        accent: Color32,
        border: Color32,
        header: Color32,
        runtime: &tokio::runtime::Handle,
    ) -> bool {
        let mut send_requested = false;
        egui::Frame::new()
            .stroke(egui::Stroke::new(1.5, border))
            .corner_radius(CornerRadius::same(8))
            .show(ui, |ui| {
                egui::Frame::new()
                    .fill(header)
                    .inner_margin(Margin::symmetric(10, 6))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&pane.label).strong().color(accent));
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui.button("Clear").clicked() {
                                    pane.messages.clear();
                                    pane.last_stats = None;
                                }
                            });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Endpoint");
                            let response = ui.text_edit_singleline(&mut pane.endpoint);
                            if response.lost_focus()
                                && ui.input(|input| input.key_pressed(egui::Key::Enter))
                            {
                                let (tx, rx) = mpsc::channel();
                                pane.model_rx = Some(rx);
                                let _guard = runtime.enter();
                                api::fetch_models(pane.endpoint.clone(), tx);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Model");
                            if pane.available_models.is_empty() {
                                ui.label("No models loaded");
                            } else {
                                egui::ComboBox::from_id_salt(format!("model-{id}"))
                                    .selected_text(&pane.model)
                                    .show_ui(ui, |ui| {
                                        for model in &pane.available_models {
                                            ui.selectable_value(
                                                &mut pane.model,
                                                model.clone(),
                                                model,
                                            );
                                        }
                                    });
                            }
                        });
                    });

                let chat_height = (ui.available_height() - 52.0).max(120.0);
                ScrollArea::vertical()
                    .id_salt(format!("messages-{id}"))
                    .stick_to_bottom(true)
                    .max_height(chat_height)
                    .show(ui, |ui| {
                        if pane.messages.is_empty() {
                            ui.centered_and_justified(|ui| ui.label("Start a conversation"));
                        }
                        for message in &pane.messages {
                            render_message(ui, message);
                        }
                        if let Some(stats) = &pane.last_stats {
                            ui.label(RichText::new(stats).small().color(theme::TEXT_DIM));
                        }
                    });

                ui.horizontal(|ui| {
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut pane.input)
                            .desired_width((ui.available_width() - 60.0).max(100.0))
                            .hint_text("Message"),
                    );
                    let enter = response.has_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter));
                    let enabled = !pane.input.trim().is_empty() && !pane.request_in_flight;
                    if ui.add_enabled(enabled, egui::Button::new("Send")).clicked()
                        || (enter && enabled)
                    {
                        send_requested = true;
                    }
                });
            });
        send_requested
    }
}

fn render_message(ui: &mut egui::Ui, message: &DisplayMessage) {
    let (background, alignment, label) = match message.role {
        Role::User => (theme::BG_USER_BUBBLE, Align::RIGHT, "You"),
        Role::Assistant => (theme::BG_ASSISTANT_BUBBLE, Align::LEFT, "rvLLM"),
    };
    ui.with_layout(Layout::top_down(alignment), |ui| {
        ui.label(RichText::new(label).small().color(theme::TEXT_DIM));
        egui::Frame::new()
            .fill(background)
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::symmetric(10, 6))
            .show(ui, |ui| {
                ui.label(if message.content.is_empty() {
                    "…"
                } else {
                    &message.content
                });
            });
    });
}

impl eframe::App for ChatApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            theme::apply_theme(context);
            self.theme_applied = true;
        }
        self.poll(PaneSide::Left);
        self.poll(PaneSide::Right);
        if self.left.request_in_flight || self.right.request_in_flight {
            context.request_repaint();
        }

        egui::TopBottomPanel::top("top").show(context, |ui| self.render_top_bar(ui));
        let mut send_left = false;
        let mut send_right = false;
        egui::CentralPanel::default().show(context, |ui| {
            let width = (ui.available_width() - 8.0) / 2.0;
            ui.horizontal(|ui| {
                ui.allocate_ui(Vec2::new(width, ui.available_height()), |ui| {
                    send_left = Self::render_pane(
                        &mut self.left,
                        ui,
                        "left",
                        theme::GPU_ACCENT,
                        theme::GPU_BORDER,
                        theme::GPU_HEADER_BG,
                        &self.runtime,
                    );
                });
                ui.allocate_ui(Vec2::new(width, ui.available_height()), |ui| {
                    send_right = Self::render_pane(
                        &mut self.right,
                        ui,
                        "right",
                        theme::TPU_ACCENT,
                        theme::TPU_BORDER,
                        theme::TPU_HEADER_BG,
                        &self.runtime,
                    );
                });
            });
        });
        if send_left {
            self.send_input(PaneSide::Left);
        }
        if send_right {
            self.send_input(PaneSide::Right);
        }
    }
}
