use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, Style, TextStyle, Visuals};

pub const BG_DARK: Color32 = Color32::from_rgb(13, 13, 15);
pub const BG_PANEL: Color32 = Color32::from_rgb(22, 22, 28);
pub const BG_INPUT: Color32 = Color32::from_rgb(30, 30, 38);
pub const BG_USER_BUBBLE: Color32 = Color32::from_rgb(45, 45, 58);
pub const BG_ASSISTANT_BUBBLE: Color32 = Color32::from_rgb(28, 28, 36);
pub const ACCENT: Color32 = Color32::from_rgb(99, 102, 241);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(129, 132, 255);
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(230, 230, 240);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(140, 140, 160);
pub const TEXT_DIM: Color32 = Color32::from_rgb(90, 90, 110);
pub const BORDER: Color32 = Color32::from_rgb(50, 50, 65);
pub const SUCCESS: Color32 = Color32::from_rgb(52, 211, 153);
pub const ERROR: Color32 = Color32::from_rgb(248, 113, 113);

// Pane accent colors
pub const GPU_ACCENT: Color32 = Color32::from_rgb(70, 130, 230);
pub const GPU_BORDER: Color32 = Color32::from_rgb(45, 80, 140);
pub const GPU_HEADER_BG: Color32 = Color32::from_rgb(18, 25, 42);

pub const TPU_ACCENT: Color32 = Color32::from_rgb(52, 211, 153);
pub const TPU_BORDER: Color32 = Color32::from_rgb(30, 110, 80);
pub const TPU_HEADER_BG: Color32 = Color32::from_rgb(16, 30, 24);

pub const RACE_TIMER: Color32 = Color32::from_rgb(255, 200, 60);
pub const WINNER_GOLD: Color32 = Color32::from_rgb(255, 215, 0);

pub fn apply_theme(ctx: &egui::Context) {
    let mut style = Style::default();

    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(20.0, FontFamily::Proportional),
    );
    style
        .text_styles
        .insert(TextStyle::Body, FontId::new(15.0, FontFamily::Proportional));
    style.text_styles.insert(
        TextStyle::Monospace,
        FontId::new(14.0, FontFamily::Monospace),
    );
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(14.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(12.0, FontFamily::Proportional),
    );

    let mut visuals = Visuals::dark();
    visuals.panel_fill = BG_PANEL;
    visuals.window_fill = BG_PANEL;
    visuals.extreme_bg_color = BG_INPUT;
    visuals.faint_bg_color = BG_DARK;
    visuals.override_text_color = Some(TEXT_PRIMARY);

    visuals.widgets.noninteractive.bg_fill = BG_PANEL;
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_SECONDARY);
    visuals.widgets.noninteractive.corner_radius = CornerRadius::same(6);

    visuals.widgets.inactive.bg_fill = BG_INPUT;
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    visuals.widgets.inactive.corner_radius = CornerRadius::same(6);

    visuals.widgets.hovered.bg_fill = ACCENT_HOVER;
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(6);

    visuals.widgets.active.bg_fill = ACCENT;
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.active.corner_radius = CornerRadius::same(6);

    visuals.selection.bg_fill = ACCENT.linear_multiply(0.4);
    visuals.selection.stroke = Stroke::new(1.0, ACCENT);

    visuals.window_corner_radius = CornerRadius::same(10);
    visuals.window_stroke = Stroke::new(1.0, BORDER);

    style.visuals = visuals;
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.window_margin = egui::Margin::same(12);

    ctx.set_style(style);
}
