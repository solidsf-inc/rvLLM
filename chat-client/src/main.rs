mod api;
mod app;
mod theme;

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to create tokio runtime");

    let rt_handle = runtime.handle().clone();
    let _guard = runtime.enter();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_min_inner_size([1000.0, 600.0])
            .with_title("rvLLM Race"),
        ..Default::default()
    };

    eframe::run_native(
        "rvLLM Race",
        options,
        Box::new(move |cc| Ok(Box::new(app::ChatApp::new(cc, rt_handle)))),
    )
    .expect("failed to run eframe app");
}
