#![allow(non_snake_case)]

mod app;
mod colormap;
mod core;
mod mesher;
mod solvers;

use app::RustyFlowApp;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("RustyFlow — Structured O-Grid Mesher & Full Potential CFD Workbench")
            .with_inner_size([1480.0, 920.0])
            .with_min_inner_size([1060.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RustyFlow",
        native_options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(RustyFlowApp::default()))
        }),
    )
}
