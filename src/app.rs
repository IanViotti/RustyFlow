use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use egui::{Color32, Pos2, Rect, RichText, Sense, Shape, Stroke, Vec2};
use egui_plot::{Line, Plot, PlotBounds, PlotPoint, PlotPoints};
use ndarray::Array2;

use crate::colormap::Colormap;
use crate::core::grid::{
    compute_mesh_quality, MeshQualitySummary, MeshScalarField, NodeQualityMetrics, Point,
};
use crate::core::solver_trait::{CfdSolution2D, FlowScalarField, SolverFamily};
use crate::mesher::config::{AirfoilType, Config as MesherConfig, EllipticEquation};
use crate::mesher::elliptic::EllipticStats;
use crate::mesher::{io as mesher_io, run_mesher_pipeline, MesherEvent};
use crate::solvers::full_potential::config::Config as FullPotentialConfig;
use crate::solvers::full_potential::io as fp_io;
use crate::solvers::full_potential::solver_core::SolverStats;
use crate::solvers::full_potential::solver_utils::{Grid as SolverGrid, PointData};
use crate::solvers::{run_full_potential_pipeline, SolverEvent};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SidebarTab {
    Mesher,
    Solver,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BottomTab {
    SurfaceCpAndAero,
    CfdConvergence,
    MesherDiagnostics,
    QualityAndLogs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CameraPreset {
    Airfoil,
    LeadingEdge,
    TrailingEdge,
    Farfield,
}

/// Unified selector for coloring the 2D physical viewport by either Mesh Quality or CFD Flowfield.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewportScalar {
    Mesh(MeshScalarField),
    Flow(FlowScalarField),
}

impl ViewportScalar {
    fn label(&self) -> String {
        match self {
            Self::Mesh(m) => format!("Mesh: {}", m.label()),
            Self::Flow(f) => format!("CFD: {}", f.label()),
        }
    }

    fn is_none(&self) -> bool {
        matches!(self, Self::Mesh(MeshScalarField::None))
    }
}

pub struct RustyFlowApp {
    // UI navigation state
    sidebar_tab: SidebarTab,
    bottom_tab: BottomTab,

    // Mesher configuration
    mesh_config: MesherConfig,
    poisson_ds_wall: f64,
    poisson_a_decay: f64,
    poisson_b_decay: f64,
    mesher_snapshot_interval: usize,

    // Active mesh & quality data
    grid: Option<Array2<Point>>,
    quality_metrics: Option<Array2<NodeQualityMetrics>>,
    quality_summary: MeshQualitySummary,

    // Mesher background worker & telemetry
    mesher_rx: Option<Receiver<MesherEvent>>,
    mesher_cancel: Arc<AtomicBool>,
    is_meshing: bool,
    mesher_residual_history: Vec<[f64; 2]>, // [iter, log10(residual)]
    last_mesher_stats: Option<EllipticStats>,

    // CFD Solver configuration & state
    selected_solver: SolverFamily,
    fp_config: FullPotentialConfig,
    solver_snapshot_interval: usize,
    cfd_solution: Option<CfdSolution2D>,
    raw_fp_domain: Option<(SolverGrid, Array2<PointData>)>,
    solver_rx: Option<Receiver<SolverEvent>>,
    solver_cancel: Arc<AtomicBool>,
    is_solving: bool,
    cfd_max_res_history: Vec<[f64; 2]>, // [iter, log10(max_res)]
    cfd_rms_res_history: Vec<[f64; 2]>, // [iter, log10(rms_res)]
    cfd_gamma_history: Vec<[f64; 2]>,   // [iter, circulation]
    last_solver_stats: Option<SolverStats>,
    last_solver_duration: Option<Duration>,
    last_beta_sup: f64,

    // Surface plot options
    invert_cp_axis: bool,
    plot_surface_mach: bool,

    // Viewport settings
    viewport_scalar: ViewportScalar,
    scalar_range: (f64, f64),
    colormap: Colormap,
    show_grid_lines: bool,
    grid_line_stride: usize,
    show_airfoil_wall: bool,
    show_wake_cut: bool,
    show_nodes: bool,
    show_sonic_line: bool,
    show_iso_contours: bool,
    iso_contour_levels: usize,
    show_velocity_vectors: bool,
    enable_hover_inspector: bool,
    pending_camera: Option<CameraPreset>,

    // Global status & logs
    status_text: String,
    error_banner: Option<String>,
    logs: Vec<String>,
}

impl Default for RustyFlowApp {
    fn default() -> Self {
        let default_mesh_cfg = if let Ok(content) = fs::read_to_string("job_config.toml") {
            toml::from_str::<MesherConfig>(&content).unwrap_or_default()
        } else {
            MesherConfig::default()
        };

        let (ds_wall, a_decay, b_decay) = match default_mesh_cfg.elliptic_equation {
            EllipticEquation::Poisson {
                ds_wall,
                a_decay,
                b_decay,
            } => (ds_wall, a_decay, b_decay),
            _ => (0.005, 1.5, 1.5),
        };

        let mut fp_cfg = FullPotentialConfig::default();
        fp_cfg.compute_derived_variables();

        let mut app = Self {
            sidebar_tab: SidebarTab::Mesher,
            bottom_tab: BottomTab::SurfaceCpAndAero,
            mesh_config: default_mesh_cfg,
            poisson_ds_wall: ds_wall,
            poisson_a_decay: a_decay,
            poisson_b_decay: b_decay,
            mesher_snapshot_interval: 25,
            grid: None,
            quality_metrics: None,
            quality_summary: MeshQualitySummary::default(),
            mesher_rx: None,
            mesher_cancel: Arc::new(AtomicBool::new(false)),
            is_meshing: false,
            mesher_residual_history: Vec::new(),
            last_mesher_stats: None,
            selected_solver: SolverFamily::FullPotential,
            fp_config: fp_cfg,
            solver_snapshot_interval: 25,
            cfd_solution: None,
            raw_fp_domain: None,
            solver_rx: None,
            solver_cancel: Arc::new(AtomicBool::new(false)),
            is_solving: false,
            cfd_max_res_history: Vec::new(),
            cfd_rms_res_history: Vec::new(),
            cfd_gamma_history: Vec::new(),
            last_solver_stats: None,
            last_solver_duration: None,
            last_beta_sup: 2.5,
            invert_cp_axis: true,
            plot_surface_mach: false,
            viewport_scalar: ViewportScalar::Mesh(MeshScalarField::None),
            scalar_range: (0.0, 1.0),
            colormap: Colormap::Turbo,
            show_grid_lines: true,
            grid_line_stride: 1,
            show_airfoil_wall: true,
            show_wake_cut: true,
            show_nodes: false,
            show_sonic_line: true,
            show_iso_contours: false,
            iso_contour_levels: 18,
            show_velocity_vectors: false,
            enable_hover_inspector: true,
            pending_camera: Some(CameraPreset::Airfoil),
            status_text: "Ready".to_string(),
            error_banner: None,
            logs: Vec::new(),
        };

        app.log("RustyFlow initialized. Generating initial O-grid preview...");
        app.generate_parabolic_preview_sync();
        app
    }
}

impl RustyFlowApp {
    fn log(&mut self, msg: impl Into<String>) {
        let text = msg.into();
        self.logs.push(text);
        if self.logs.len() > 500 {
            self.logs.remove(0);
        }
    }

    fn update_grid(&mut self, new_grid: Array2<Point>, recompute_quality: bool) {
        // Clear stale CFD solution if mesh dimensions changed
        if let Some(sol) = &self.cfd_solution {
            if sol.grid.dim() != new_grid.dim() {
                self.cfd_solution = None;
                self.raw_fp_domain = None;
                if matches!(self.viewport_scalar, ViewportScalar::Flow(_)) {
                    self.viewport_scalar = ViewportScalar::Mesh(MeshScalarField::None);
                }
            }
        }

        if recompute_quality {
            let (metrics, summary) = compute_mesh_quality(&new_grid);
            self.quality_summary = summary;
            self.quality_metrics = Some(metrics);
            self.recompute_scalar_range();
        }
        self.grid = Some(new_grid);
    }

    fn recompute_scalar_range(&mut self) {
        match self.viewport_scalar {
            ViewportScalar::Mesh(MeshScalarField::None) => {
                self.scalar_range = (0.0, 1.0);
            }
            ViewportScalar::Mesh(field) => {
                if let Some(metrics) = &self.quality_metrics {
                    let mut min_v = f64::MAX;
                    let mut max_v = f64::MIN;
                    for m in metrics.iter() {
                        let v = field.sample(m);
                        if v.is_finite() {
                            min_v = min_v.min(v);
                            max_v = max_v.max(v);
                        }
                    }
                    if min_v > max_v || (max_v - min_v).abs() < 1e-12 {
                        self.scalar_range = (min_v - 0.5, max_v + 0.5);
                    } else {
                        self.scalar_range = (min_v, max_v);
                    }
                }
            }
            ViewportScalar::Flow(field) => {
                if let Some(sol) = &self.cfd_solution {
                    let (ni, nj) = sol.grid.dim();
                    let mut min_v = f64::MAX;
                    let mut max_v = f64::MIN;
                    for j in 0..nj {
                        for i in 0..ni {
                            let v = sol.sample_field(field, i, j);
                            if v.is_finite() {
                                min_v = min_v.min(v);
                                max_v = max_v.max(v);
                            }
                        }
                    }
                    if min_v > max_v || (max_v - min_v).abs() < 1e-12 {
                        self.scalar_range = (min_v - 0.1, max_v + 0.1);
                    } else {
                        self.scalar_range = (min_v, max_v);
                    }
                }
            }
        }
    }

    fn sample_active_scalar(&self, i: usize, j: usize) -> f64 {
        match self.viewport_scalar {
            ViewportScalar::Mesh(field) => self
                .quality_metrics
                .as_ref()
                .map(|m| field.sample(&m[[i, j]]))
                .unwrap_or(0.0),
            ViewportScalar::Flow(field) => self
                .cfd_solution
                .as_ref()
                .map(|sol| sol.sample_field(field, i, j))
                .unwrap_or(0.0),
        }
    }

    fn generate_parabolic_preview_sync(&mut self) {
        self.mesh_config.sanitize();
        let mut grid = mesher_io::create_grid(
            self.mesh_config.longitudinal_points,
            self.mesh_config.normal_points,
        );
        crate::mesher::geometry::insert_geometry(&self.mesh_config, &mut grid);
        crate::mesher::parabolic::generate_parabolic_mesh(&self.mesh_config, &mut grid);
        self.auto_adjust_stride(
            self.mesh_config.longitudinal_points,
            self.mesh_config.normal_points,
        );
        self.cfd_solution = None;
        self.raw_fp_domain = None;
        if matches!(self.viewport_scalar, ViewportScalar::Flow(_)) {
            self.viewport_scalar = ViewportScalar::Mesh(MeshScalarField::None);
        }
        self.update_grid(grid, true);
        self.status_text = format!(
            "Parabolic Mesh Ready ({} x {})",
            self.mesh_config.longitudinal_points, self.mesh_config.normal_points
        );
    }

    fn auto_adjust_stride(&mut self, ni: usize, nj: usize) {
        if ni * nj > 50_000 {
            self.grid_line_stride = 4;
        } else if ni * nj > 20_000 {
            self.grid_line_stride = 2;
        } else {
            self.grid_line_stride = 1;
        }
    }

    fn start_mesher_worker(&mut self) {
        if self.is_meshing || self.is_solving {
            return;
        }
        self.mesh_config.sanitize();
        self.auto_adjust_stride(
            self.mesh_config.longitudinal_points,
            self.mesh_config.normal_points,
        );
        self.error_banner = None;
        self.mesher_residual_history.clear();
        self.last_mesher_stats = None;
        self.cfd_solution = None;
        self.raw_fp_domain = None;
        if matches!(self.viewport_scalar, ViewportScalar::Flow(_)) {
            self.viewport_scalar = ViewportScalar::Mesh(MeshScalarField::None);
        }
        self.is_meshing = true;
        self.bottom_tab = BottomTab::MesherDiagnostics;

        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.mesher_cancel = cancel_flag.clone();

        let (tx, rx) = mpsc::channel();
        self.mesher_rx = Some(rx);

        let cfg = self.mesh_config.clone();
        let snap_int = self.mesher_snapshot_interval;

        self.status_text = format!(
            "Meshing {} x {}...",
            cfg.longitudinal_points, cfg.normal_points
        );
        self.log(format!(
            "Started mesh job '{}' ({:?}, {} x {}, smoother: {:?})",
            cfg.meshname,
            cfg.airfoil_type,
            cfg.longitudinal_points,
            cfg.normal_points,
            cfg.elliptic_equation
        ));

        thread::spawn(move || {
            run_mesher_pipeline(cfg, tx, cancel_flag, snap_int);
        });
    }

    fn start_solver_worker(&mut self) {
        if self.is_meshing || self.is_solving {
            return;
        }
        let Some(grid) = self.grid.clone() else {
            self.error_banner = Some("Please generate or import a mesh first.".to_string());
            return;
        };
        if self.quality_summary.negative_jacobian_count > 0 {
            self.error_banner = Some(format!(
                "Active mesh contains {} inverted cells (J <= 0). Smooth or adjust the mesh before running CFD.",
                self.quality_summary.negative_jacobian_count
            ));
            return;
        }

        self.fp_config.compute_derived_variables();
        self.error_banner = None;
        self.cfd_max_res_history.clear();
        self.cfd_rms_res_history.clear();
        self.cfd_gamma_history.clear();
        self.last_solver_stats = None;
        self.is_solving = true;

        // Automatically switch viewport to Mach Number and bottom panel to Surface Cp
        if !matches!(self.viewport_scalar, ViewportScalar::Flow(_)) {
            self.viewport_scalar = ViewportScalar::Flow(FlowScalarField::Mach);
        }
        self.bottom_tab = BottomTab::SurfaceCpAndAero;

        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.solver_cancel = cancel_flag.clone();

        let (tx, rx) = mpsc::channel();
        self.solver_rx = Some(rx);

        let cfg = self.fp_config.clone();
        let snap_int = self.solver_snapshot_interval;

        self.status_text = format!(
            "Solving CFD (M={:.2}, AoA={:.1}°)...",
            cfg.Mach_inf, cfg.AoA
        );
        self.log(format!(
            "Started Full Potential AF2 Solver: job='{}', M_inf={:.3}, AoA={:.2}°, Rg={:.2}, grid={}x{}",
            cfg.jobname,
            cfg.Mach_inf,
            cfg.AoA,
            cfg.Rg,
            grid.dim().0,
            grid.dim().1
        ));

        thread::spawn(move || {
            run_full_potential_pipeline(grid, cfg, tx, cancel_flag, snap_int);
        });
    }

    fn poll_worker_events(&mut self, ctx: &egui::Context) {
        // 1. Poll Mesher events
        let mut mesher_done = false;
        let mut mesher_events = Vec::new();
        if let Some(rx) = &self.mesher_rx {
            while let Ok(ev) = rx.try_recv() {
                mesher_events.push(ev);
            }
        }
        for ev in mesher_events {
            match ev {
                MesherEvent::ParabolicCompleted { grid, duration } => {
                    self.log(format!(
                        "Parabolic marching step completed in {:.2} ms.",
                        duration.as_secs_f64() * 1000.0
                    ));
                    self.update_grid(grid, false);
                }
                MesherEvent::Iteration {
                    iter,
                    max_iter,
                    residual,
                    snapshot,
                } => {
                    let log_res = residual.max(1e-16).log10();
                    self.mesher_residual_history.push([iter as f64, log_res]);
                    self.status_text = format!(
                        "Smoothing Mesh: Iter {}/{} | Res: {:.2e}",
                        iter, max_iter, residual
                    );
                    if let Some(snap_grid) = snapshot {
                        self.update_grid(snap_grid, false);
                    }
                }
                MesherEvent::Finished {
                    grid,
                    stats,
                    total_duration,
                } => {
                    self.update_grid(grid, true);
                    self.last_mesher_stats = Some(stats);
                    if stats.cancelled {
                        self.status_text = format!(
                            "Mesher Stopped (Iter {}, Res: {:.2e})",
                            stats.iterations, stats.final_residual
                        );
                        self.log(format!(
                            "Mesher stopped at iteration {} (Residual: {:.3e}).",
                            stats.iterations, stats.final_residual
                        ));
                    } else if stats.converged {
                        self.status_text = format!(
                            "Mesh Ready ({} iters in {:.2} s)",
                            stats.iterations,
                            total_duration.as_secs_f64()
                        );
                        self.log(format!(
                            "Mesh generation converged in {} iterations ({:.2} s). Final residual: {:.3e}",
                            stats.iterations,
                            total_duration.as_secs_f64(),
                            stats.final_residual
                        ));
                    } else {
                        self.status_text = format!(
                            "Mesh Max Iter ({} iters, Res: {:.2e})",
                            stats.iterations, stats.final_residual
                        );
                        self.log(format!(
                            "Mesh generation reached max iterations ({}) in {:.2} s. Final residual: {:.3e}",
                            stats.iterations,
                            total_duration.as_secs_f64(),
                            stats.final_residual
                        ));
                    }
                    mesher_done = true;
                }
                MesherEvent::Error { message } => {
                    self.error_banner = Some(message.clone());
                    self.status_text = "Error: Mesher Diverged".to_string();
                    self.log(format!("MESHER ERROR: {}", message));
                    mesher_done = true;
                }
            }
        }
        if mesher_done {
            self.is_meshing = false;
            self.mesher_rx = None;
        }

        // 2. Poll CFD Solver events
        let mut solver_done = false;
        let mut solver_events = Vec::new();
        if let Some(rx) = &self.solver_rx {
            while let Ok(ev) = rx.try_recv() {
                solver_events.push(ev);
            }
        }
        for ev in solver_events {
            match ev {
                SolverEvent::Iteration {
                    iter,
                    max_iter,
                    max_res,
                    rms_res,
                    circulation,
                    beta_sup,
                    snapshot,
                } => {
                    self.cfd_max_res_history
                        .push([iter as f64, max_res.max(1e-16).log10()]);
                    self.cfd_rms_res_history
                        .push([iter as f64, rms_res.max(1e-16).log10()]);
                    self.cfd_gamma_history.push([iter as f64, circulation]);
                    self.last_beta_sup = beta_sup;
                    self.status_text = format!(
                        "Solving AF2: Iter {}/{} | Max Res: {:.2e} | Γ: {:+.4}",
                        iter, max_iter, max_res, circulation
                    );
                    if let Some(snap_sol) = snapshot {
                        self.cfd_solution = Some(snap_sol);
                        self.recompute_scalar_range();
                    }
                }
                SolverEvent::Finished {
                    solution,
                    raw_fp_domain,
                    stats,
                    total_duration,
                } => {
                    let cl = solution.aero.cl;
                    let cd = solution.aero.cd;
                    let m_max = solution.aero.max_mach;
                    self.cfd_solution = Some(solution);
                    self.raw_fp_domain = raw_fp_domain;
                    self.last_solver_stats = Some(stats);
                    self.last_solver_duration = Some(total_duration);
                    self.recompute_scalar_range();

                    if stats.cancelled {
                        self.status_text = format!(
                            "CFD Stopped at Iter {} (Res: {:.2e})",
                            stats.iterations, stats.final_max_res
                        );
                        self.log(format!(
                            "CFD Solver stopped by user at iteration {} (Max Res = {:.3e}, Γ = {:+.5}).",
                            stats.iterations, stats.final_max_res, stats.final_gamma
                        ));
                    } else if stats.converged {
                        self.status_text = format!(
                            "CFD Converged ({} iters, {:.2} s) | Cl={:+.4}, M_max={:.3}",
                            stats.iterations,
                            total_duration.as_secs_f64(),
                            cl,
                            m_max
                        );
                        self.log(format!(
                            "CFD Converged in {} iterations ({:.2} s): Max Res={:.3e}, RMS Res={:.3e}, Γ={:+.5}, Cl={:+.4}, Cd={:+.4}, M_max={:.3}",
                            stats.iterations,
                            total_duration.as_secs_f64(),
                            stats.final_max_res,
                            stats.final_rms_res,
                            stats.final_gamma,
                            cl,
                            cd,
                            m_max
                        ));
                    } else {
                        self.status_text = format!(
                            "CFD Max Iter ({}) | Res: {:.2e} | Cl={:+.4}",
                            stats.iterations, stats.final_max_res, cl
                        );
                        self.log(format!(
                            "CFD reached max iterations ({}) in {:.2} s: Max Res={:.3e}, Γ={:+.5}, Cl={:+.4}, Cd={:+.4}",
                            stats.iterations,
                            total_duration.as_secs_f64(),
                            stats.final_max_res,
                            stats.final_gamma,
                            cl,
                            cd
                        ));
                    }
                    solver_done = true;
                }
                SolverEvent::Error { message } => {
                    self.error_banner = Some(message.clone());
                    self.status_text = "Error: CFD Solver Diverged".to_string();
                    self.log(format!("CFD SOLVER ERROR: {}", message));
                    solver_done = true;
                }
            }
        }
        if solver_done {
            self.is_solving = false;
            self.solver_rx = None;
        }

        if self.is_meshing || self.is_solving {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
    }

    fn export_current_mesh(&mut self) {
        let Some(grid) = &self.grid else {
            return;
        };
        match mesher_io::init_mesh_directory(&self.mesh_config) {
            Ok(dir) => {
                let vtk_path = PathBuf::from(&dir).join("mesh.vtk");
                let csv_path = PathBuf::from(&dir).join("mesh.csv");
                let toml_path = PathBuf::from(&dir).join("job_config.toml");

                let vtk_res = mesher_io::export_vtk_structured_grid(grid, &vtk_path);
                let csv_res = mesher_io::save_grid_to_csv(grid, &csv_path);
                let _ = toml::to_string_pretty(&self.mesh_config)
                    .ok()
                    .and_then(|s| fs::write(&toml_path, s).ok());

                match (vtk_res, csv_res) {
                    (Ok(_), Ok(_)) => {
                        let msg = format!("Exported mesh to {}/mesh.vtk & mesh.csv", dir);
                        self.status_text = msg.clone();
                        self.log(msg);
                    }
                    (Err(e), _) | (_, Err(e)) => {
                        let msg = format!("Mesh export failed: {}", e);
                        self.error_banner = Some(msg.clone());
                        self.log(msg);
                    }
                }
            }
            Err(e) => {
                let msg = format!("Could not create mesh output folder: {}", e);
                self.error_banner = Some(msg.clone());
                self.log(msg);
            }
        }
    }

    fn export_current_cfd_solution(&mut self) {
        let Some((solver_grid, domain)) = &self.raw_fp_domain else {
            return;
        };
        match fp_io::export_full_potential_results(&mut self.fp_config, solver_grid, domain) {
            Ok(out_dir) => {
                let msg = format!(
                    "Exported CFD results to {}/ (solution.vtk, Cp_distribution.csv, solver_config.txt)",
                    out_dir
                );
                self.status_text = format!("Exported CFD to {}/", out_dir);
                self.log(msg);
            }
            Err(e) => {
                let msg = format!("Failed to export CFD solution: {}", e);
                self.error_banner = Some(msg.clone());
                self.log(msg);
            }
        }
    }

    fn render_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.heading(
                RichText::new("🦀 RustyFlow")
                    .strong()
                    .color(Color32::from_rgb(240, 140, 60)),
            );
            ui.label(RichText::new("Mesher & CFD Workbench").weak());
            ui.separator();

            // Quick Presets selector
            egui::ComboBox::from_id_salt("preset_selector")
                .selected_text("⚡ Quick Presets...")
                .show_ui(ui, |ui| {
                    ui.label(RichText::new("Mesh Presets (RustyMesher)").weak().small());
                    if ui
                        .selectable_label(false, "🕸️ NACA 0012 — Laplace Smoother (213 x 65)")
                        .clicked()
                    {
                        self.mesh_config = MesherConfig::preset_naca0012_laplace();
                        self.generate_parabolic_preview_sync();
                        self.pending_camera = Some(CameraPreset::Airfoil);
                    }
                    if ui
                        .selectable_label(false, "🕸️ NACA 0012 — Poisson Wall Control (213 x 65)")
                        .clicked()
                    {
                        self.mesh_config = MesherConfig::preset_naca0012_poisson();
                        self.poisson_ds_wall = 0.005;
                        self.poisson_a_decay = 1.5;
                        self.poisson_b_decay = 1.5;
                        self.generate_parabolic_preview_sync();
                        self.pending_camera = Some(CameraPreset::Airfoil);
                    }
                    if ui
                        .selectable_label(false, "🕸️ NACA 0012 — Fine Grid (713 x 150)")
                        .clicked()
                    {
                        self.mesh_config = MesherConfig::preset_naca0012_fine();
                        self.poisson_ds_wall = 0.005;
                        self.poisson_a_decay = 1.5;
                        self.poisson_b_decay = 1.5;
                        self.generate_parabolic_preview_sync();
                        self.pending_camera = Some(CameraPreset::Airfoil);
                    }
                    if ui
                        .selectable_label(false, "🕸️ Biconvex 10% — Laplace (213 x 65)")
                        .clicked()
                    {
                        self.mesh_config = MesherConfig::preset_biconvex_standard();
                        self.generate_parabolic_preview_sync();
                        self.pending_camera = Some(CameraPreset::Airfoil);
                    }

                    ui.separator();
                    ui.label(RichText::new("Flight Presets (RustPotential)").weak().small());
                    if ui
                        .selectable_label(false, "🚀 Subsonic Symmetric (M = 0.30, AoA = 0.0°)")
                        .clicked()
                    {
                        self.fp_config = FullPotentialConfig::preset_subsonic_symmetric();
                        self.sidebar_tab = SidebarTab::Solver;
                    }
                    if ui
                        .selectable_label(false, "🚀 Transonic Lifting Shock (M = 0.80, AoA = 1.0°)")
                        .clicked()
                    {
                        self.fp_config = FullPotentialConfig::preset_transonic_lifting();
                        self.sidebar_tab = SidebarTab::Solver;
                    }
                    if ui
                        .selectable_label(false, "🚀 Transonic Strong Shock (M = 0.85, AoA = 0.0°)")
                        .clicked()
                    {
                        self.fp_config = FullPotentialConfig::preset_transonic_shock_zero_aoa();
                        self.sidebar_tab = SidebarTab::Solver;
                    }
                    if ui
                        .selectable_label(false, "🚀 Subsonic High Lift (M = 0.50, AoA = 4.0°)")
                        .clicked()
                    {
                        self.fp_config = FullPotentialConfig::preset_subsonic_high_lift();
                        self.sidebar_tab = SidebarTab::Solver;
                    }
                });

            ui.separator();

            if ui
                .button("📥 Import VTK Mesh")
                .on_hover_text("Load an existing .vtk structured grid file")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("VTK Structured Grid", &["vtk"])
                    .pick_file()
                {
                    match mesher_io::import_vtk_structured_grid(&path) {
                        Ok(loaded_grid) => {
                            let (ni, nj) = loaded_grid.dim();
                            self.mesh_config.longitudinal_points = ni;
                            self.mesh_config.normal_points = nj;
                            self.auto_adjust_stride(ni, nj);
                            self.cfd_solution = None;
                            self.raw_fp_domain = None;
                            self.update_grid(loaded_grid, true);
                            self.pending_camera = Some(CameraPreset::Airfoil);
                            let msg = format!(
                                "Imported VTK mesh '{}' ({} x {})",
                                path.display(),
                                ni,
                                nj
                            );
                            self.status_text = msg.clone();
                            self.log(msg);
                        }
                        Err(e) => {
                            self.error_banner = Some(e);
                        }
                    }
                }
            }

            if ui
                .add_enabled(self.grid.is_some(), egui::Button::new("📤 Export Mesh"))
                .on_hover_text("Export mesh to job_files/<meshname>/mesh.vtk & mesh.csv")
                .clicked()
            {
                self.export_current_mesh();
            }

            if ui
                .add_enabled(
                    self.raw_fp_domain.is_some(),
                    egui::Button::new("💾 Export CFD Solution"),
                )
                .on_hover_text(
                    "Export solution.vtk, Cp_distribution.csv, and solver_config.txt to job_files/",
                )
                .clicked()
            {
                self.export_current_cfd_solution();
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let badge_color = if self.is_meshing || self.is_solving {
                    Color32::from_rgb(230, 170, 40)
                } else if self.error_banner.is_some() {
                    Color32::from_rgb(220, 70, 70)
                } else {
                    Color32::from_rgb(80, 190, 120)
                };
                if self.is_meshing || self.is_solving {
                    ui.spinner();
                }
                ui.label(RichText::new(&self.status_text).color(badge_color).strong());
            });
        });
    }

    fn render_left_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.sidebar_tab,
                SidebarTab::Mesher,
                RichText::new("🕸️ 1. Mesh Generator").strong(),
            );
            ui.selectable_value(
                &mut self.sidebar_tab,
                SidebarTab::Solver,
                RichText::new("🚀 2. CFD Solver").strong(),
            );
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| match self.sidebar_tab {
            SidebarTab::Mesher => self.render_mesher_sidebar(ui),
            SidebarTab::Solver => self.render_solver_sidebar(ui),
        });
    }

    fn render_mesher_sidebar(&mut self, ui: &mut egui::Ui) {
        let mut geom_changed = false;

        ui.collapsing(RichText::new("📐 Airfoil & Domain Geometry").strong(), |ui| {
            egui::Grid::new("geom_grid")
                .num_columns(2)
                .spacing([10.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Mesh Name:");
                    ui.text_edit_singleline(&mut self.mesh_config.meshname);
                    ui.end_row();

                    ui.label("Airfoil Profile:");
                    egui::ComboBox::from_id_salt("airfoil_type_combo")
                        .selected_text(match self.mesh_config.airfoil_type {
                            AirfoilType::NACA00XX => "NACA 00XX (4-Digit)",
                            AirfoilType::Biconvex => "Parabolic Biconvex",
                        })
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_value(
                                    &mut self.mesh_config.airfoil_type,
                                    AirfoilType::NACA00XX,
                                    "NACA 00XX (4-Digit)",
                                )
                                .changed()
                            {
                                geom_changed = true;
                            }
                            if ui
                                .selectable_value(
                                    &mut self.mesh_config.airfoil_type,
                                    AirfoilType::Biconvex,
                                    "Parabolic Biconvex",
                                )
                                .changed()
                            {
                                geom_changed = true;
                            }
                        });
                    ui.end_row();

                    ui.label("Thickness (t/c):")
                        .on_hover_text("Maximum thickness-to-chord ratio (e.g. 0.12 for NACA 0012)");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.mesh_config.t, 0.02..=0.30)
                                .step_by(0.005)
                                .fixed_decimals(3),
                        )
                        .changed()
                    {
                        geom_changed = true;
                    }
                    ui.end_row();

                    ui.label("Farfield Radius (r_max):")
                        .on_hover_text("Outer circular boundary radius in chord lengths");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.mesh_config.r_max, 3.0..=50.0)
                                .suffix(" c")
                                .fixed_decimals(1),
                        )
                        .changed()
                    {
                        geom_changed = true;
                    }
                    ui.end_row();
                });
        })
        .fully_open();

        ui.add_space(4.0);

        ui.collapsing(RichText::new("🔢 O-Grid Topology & Clustering").strong(), |ui| {
            egui::Grid::new("topo_grid")
                .num_columns(2)
                .spacing([10.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Longitudinal (N_ξ):")
                        .on_hover_text("Nodes wrapped around airfoil (4k + 1 so LE, mid-chord, and TE are symmetric)");
                    let mut ni = self.mesh_config.longitudinal_points;
                    if ui
                        .add(egui::DragValue::new(&mut ni).range(21..=1501).speed(4))
                        .changed()
                    {
                        let rem = (ni.max(21) - 1) % 4;
                        if rem != 0 {
                            ni += 4 - rem;
                        }
                        self.mesh_config.longitudinal_points = ni;
                        geom_changed = true;
                    }
                    ui.end_row();

                    ui.label("Normal (N_η):")
                        .on_hover_text("Radial layers from airfoil surface (j=0) to farfield (j=N_η-1)");
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.mesh_config.normal_points)
                                .range(10..=500)
                                .speed(1),
                        )
                        .changed()
                    {
                        geom_changed = true;
                    }
                    ui.end_row();

                    ui.label("LE/TE Clustering (r):")
                        .on_hover_text("1.0 = uniform; > 1.0 clusters nodes at Leading and Trailing Edges");
                    if ui
                        .add(
                            egui::Slider::new(
                                &mut self.mesh_config.stretching_factor,
                                1.000..=1.120,
                            )
                            .fixed_decimals(4),
                        )
                        .changed()
                    {
                        geom_changed = true;
                    }
                    ui.end_row();

                    ui.label("Radial Progression (q):")
                        .on_hover_text("Geometric progression ratio for normal spacing in Parabolic marching");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.mesh_config.q, 1.000..=1.200)
                                .fixed_decimals(4),
                        )
                        .changed()
                    {
                        geom_changed = true;
                    }
                    ui.end_row();
                });

            let total_pts =
                self.mesh_config.longitudinal_points * self.mesh_config.normal_points;
            ui.label(
                RichText::new(format!(
                    "Total Grid Size: {} nodes ({} cells)",
                    total_pts,
                    (self.mesh_config.longitudinal_points - 1)
                        * (self.mesh_config.normal_points - 1)
                ))
                .small()
                .weak(),
            );
        })
        .fully_open();

        if geom_changed && !self.is_meshing && !self.is_solving {
            self.generate_parabolic_preview_sync();
        }

        ui.add_space(4.0);

        ui.collapsing(RichText::new("⚙️ Elliptic Smoother (AF1 / ADI)").strong(), |ui| {
            let is_parabolic = matches!(
                self.mesh_config.elliptic_equation,
                EllipticEquation::ParabolicOnly
            );
            let is_laplace = matches!(
                self.mesh_config.elliptic_equation,
                EllipticEquation::Laplace
            );
            let is_poisson = matches!(
                self.mesh_config.elliptic_equation,
                EllipticEquation::Poisson { .. }
            );

            ui.horizontal(|ui| {
                if ui.selectable_label(is_parabolic, "Parabolic Only").clicked() {
                    self.mesh_config.elliptic_equation = EllipticEquation::ParabolicOnly;
                }
                if ui.selectable_label(is_laplace, "Laplace (AF1)").clicked() {
                    self.mesh_config.elliptic_equation = EllipticEquation::Laplace;
                }
                if ui
                    .selectable_label(is_poisson, "Poisson (Wall Control)")
                    .clicked()
                {
                    self.mesh_config.elliptic_equation = EllipticEquation::Poisson {
                        ds_wall: self.poisson_ds_wall,
                        a_decay: self.poisson_a_decay,
                        b_decay: self.poisson_b_decay,
                    };
                }
            });

            if !is_parabolic {
                ui.separator();
                egui::Grid::new("af1_grid")
                    .num_columns(2)
                    .spacing([10.0, 6.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Relaxation (ω):");
                        ui.add(
                            egui::Slider::new(&mut self.mesh_config.omega, 0.5..=1.85)
                                .fixed_decimals(2),
                        );
                        ui.end_row();

                        ui.label("Alpha Low (α_L):");
                        ui.add(
                            egui::DragValue::new(&mut self.mesh_config.alpha_l)
                                .range(0.01..=10.0)
                                .speed(0.05),
                        );
                        ui.end_row();

                        ui.label("Alpha High (α_H):");
                        ui.add(
                            egui::DragValue::new(&mut self.mesh_config.alpha_h)
                                .range(1.0..=200.0)
                                .speed(0.5),
                        );
                        ui.end_row();

                        ui.label("Max Iterations:");
                        ui.add(
                            egui::DragValue::new(&mut self.mesh_config.max_iter)
                                .range(10..=20000)
                                .speed(50),
                        );
                        ui.end_row();

                        ui.label("Tolerance:");
                        ui.add(
                            egui::DragValue::new(&mut self.mesh_config.conv_criterion)
                                .range(1e-10..=1e-2)
                                .speed(1e-7)
                                .custom_formatter(|n, _| format!("{:.1e}", n)),
                        );
                        ui.end_row();

                        ui.label("Live Frame Interval:");
                        ui.add(
                            egui::DragValue::new(&mut self.mesher_snapshot_interval)
                                .range(1..=500)
                                .suffix(" iters"),
                        );
                        ui.end_row();
                    });
            }

            if is_poisson {
                ui.separator();
                ui.label(RichText::new("Poisson Boundary Control (Sorenson & Steger)").strong());
                let mut poisson_updated = false;
                egui::Grid::new("poisson_grid")
                    .num_columns(2)
                    .spacing([10.0, 6.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Wall Spacing (ds_wall):");
                        if ui
                            .add(
                                egui::DragValue::new(&mut self.poisson_ds_wall)
                                    .range(0.0002..=0.05)
                                    .speed(0.0005)
                                    .fixed_decimals(5),
                            )
                            .changed()
                        {
                            poisson_updated = true;
                        }
                        ui.end_row();

                        ui.label("Decay Factor 'a':");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.poisson_a_decay, 0.1..=5.0)
                                    .fixed_decimals(2),
                            )
                            .changed()
                        {
                            poisson_updated = true;
                        }
                        ui.end_row();

                        ui.label("Decay Factor 'b':");
                        if ui
                            .add(
                                egui::Slider::new(&mut self.poisson_b_decay, 0.1..=5.0)
                                    .fixed_decimals(2),
                            )
                            .changed()
                        {
                            poisson_updated = true;
                        }
                        ui.end_row();
                    });

                if poisson_updated {
                    self.mesh_config.elliptic_equation = EllipticEquation::Poisson {
                        ds_wall: self.poisson_ds_wall,
                        a_decay: self.poisson_a_decay,
                        b_decay: self.poisson_b_decay,
                    };
                }
            }
        })
        .fully_open();

        ui.add_space(8.0);

        ui.vertical_centered_justified(|ui| {
            if !self.is_meshing {
                let btn_label = match self.mesh_config.elliptic_equation {
                    EllipticEquation::ParabolicOnly => "▶ Generate Parabolic O-Grid",
                    EllipticEquation::Laplace => "▶ Generate & Smooth (Laplace AF1)",
                    EllipticEquation::Poisson { .. } => "▶ Generate & Smooth (Poisson AF1)",
                };
                if ui
                    .add_enabled(
                        !self.is_solving,
                        egui::Button::new(RichText::new(btn_label).strong().size(14.5))
                            .fill(Color32::from_rgb(40, 110, 190))
                            .min_size(Vec2::new(ui.available_width(), 32.0)),
                    )
                    .clicked()
                {
                    self.start_mesher_worker();
                }

                ui.add_space(4.0);
                if ui
                    .add_enabled(
                        !self.is_solving,
                        egui::Button::new("⚡ Reset to Fast Parabolic Preview"),
                    )
                    .clicked()
                {
                    self.generate_parabolic_preview_sync();
                }
            } else if ui
                .add_sized(
                    [ui.available_width(), 32.0],
                    egui::Button::new(
                        RichText::new("⏹ Stop Elliptic Smoother")
                            .strong()
                            .size(14.5),
                    )
                    .fill(Color32::from_rgb(190, 55, 55)),
                )
                .clicked()
            {
                self.mesher_cancel.store(true, Ordering::Relaxed);
            }
        });
    }

    fn render_solver_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.collapsing(RichText::new("🧠 CFD Solver Engine Architecture").strong(), |ui| {
            for family in [
                SolverFamily::FullPotential,
                SolverFamily::Euler2D,
                SolverFamily::Rans2D,
            ] {
                ui.radio_value(&mut self.selected_solver, family, family.label());
            }
        })
        .fully_open();

        if !self.selected_solver.is_implemented() {
            ui.add_space(8.0);
            ui.group(|ui| {
                ui.colored_label(
                    Color32::from_rgb(240, 180, 70),
                    "This solver module is part of the extensible architecture and is reserved for future implementation.",
                );
                ui.label("Select 'Full Potential (AF2 + Holst Upwind + Circulation)' to run simulations now.");
            });
            return;
        }

        ui.add_space(4.0);

        ui.collapsing(RichText::new("✈️ Flight & Thermodynamic State").strong(), |ui| {
            let mut changed = false;
            egui::Grid::new("fp_flight_grid")
                .num_columns(2)
                .spacing([10.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Job Name:");
                    ui.text_edit_singleline(&mut self.fp_config.jobname);
                    ui.end_row();

                    ui.label("Freestream Mach (M_∞):");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.fp_config.Mach_inf, 0.05..=1.25)
                                .step_by(0.01)
                                .fixed_decimals(3),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    ui.end_row();

                    ui.label("Angle of Attack (AoA):")
                        .on_hover_text("Freestream incidence angle α in degrees (drives Kutta circulation Γ)");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.fp_config.AoA, -8.0..=12.0)
                                .step_by(0.1)
                                .suffix("°")
                                .fixed_decimals(2),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    ui.end_row();

                    ui.label("Heat Ratio (γ):");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.fp_config.gamma, 1.10..=1.67)
                                .fixed_decimals(2),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                    ui.end_row();
                });

            if changed {
                self.fp_config.compute_derived_variables();
            }

            let regime = if self.fp_config.Mach_inf < 0.75 {
                ("Subsonic", Color32::from_rgb(90, 190, 250))
            } else if self.fp_config.Mach_inf <= 1.15 {
                ("Transonic (Shock-Capturing)", Color32::from_rgb(245, 175, 65))
            } else {
                ("Supersonic", Color32::from_rgb(240, 100, 100))
            };

            ui.horizontal(|ui| {
                ui.label(RichText::new("Regime:").small().weak());
                ui.label(RichText::new(regime.0).small().strong().color(regime.1));
                ui.label(
                    RichText::new(format!(
                        "| u_∞={:.3}, v_∞={:.3}, ρ_∞={:.3}",
                        self.fp_config.u_inf, self.fp_config.v_inf, self.fp_config.rho_inf
                    ))
                    .small()
                    .weak(),
                );
            });
        })
        .fully_open();

        ui.add_space(4.0);

        ui.collapsing(RichText::new("⚙️ AF2 & Circulation Controls").strong(), |ui| {
            egui::Grid::new("fp_af2_grid")
                .num_columns(2)
                .spacing([10.0, 6.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Max Iterations:");
                    ui.add(
                        egui::DragValue::new(&mut self.fp_config.max_iterations)
                            .range(10..=20000)
                            .speed(25),
                    );
                    ui.end_row();

                    ui.label("Convergence Tol:");
                    ui.add(
                        egui::DragValue::new(&mut self.fp_config.convergence_residual)
                            .range(1e-10..=1e-2)
                            .speed(1e-6)
                            .custom_formatter(|n, _| format!("{:.1e}", n)),
                    );
                    ui.end_row();

                    ui.label("Relaxation (ω):");
                    ui.add(
                        egui::Slider::new(&mut self.fp_config.omega, 0.2..=1.6)
                            .fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Circulation Relax (R_g):")
                        .on_hover_text("Trailing-edge Kutta condition relaxation factor (0.2–0.5 recommended)");
                    ui.add(
                        egui::Slider::new(&mut self.fp_config.Rg, 0.05..=1.0)
                            .fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Supersonic Beta (β_mult):")
                        .on_hover_text("Initial temporal damping multiplier in supersonic zones (M > 1)");
                    ui.add(
                        egui::Slider::new(&mut self.fp_config.beta_mult, 0.5..=8.0)
                            .fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Alpha Cycle Length (M):");
                    ui.add(
                        egui::DragValue::new(&mut self.fp_config.alpha_sequence_length)
                            .range(1..=32),
                    );
                    ui.end_row();

                    ui.label("Holst Upwind Coeff (C):")
                        .on_hover_text("Artificial density upwind bias coefficient for shock capturing");
                    ui.add(
                        egui::Slider::new(&mut self.fp_config.C, 0.5..=3.0)
                            .fixed_decimals(2),
                    );
                    ui.end_row();

                    ui.label("Export Metrics in VTK:");
                    ui.checkbox(&mut self.fp_config.export_metrics, "Include J, A1, A2, A3");
                    ui.end_row();

                    ui.label("Live Frame Interval:");
                    ui.add(
                        egui::DragValue::new(&mut self.solver_snapshot_interval)
                            .range(1..=500)
                            .suffix(" iters"),
                    );
                    ui.end_row();
                });
        })
        .fully_open();

        ui.add_space(8.0);

        // Primary CFD Action Buttons
        ui.vertical_centered_justified(|ui| {
            if !self.is_solving {
                if ui
                    .add_enabled(
                        self.grid.is_some() && !self.is_meshing,
                        egui::Button::new(
                            RichText::new("🚀 Run Full Potential Solver (AF2)")
                                .strong()
                                .size(15.0),
                        )
                        .fill(Color32::from_rgb(35, 145, 90))
                        .min_size(Vec2::new(ui.available_width(), 36.0)),
                    )
                    .clicked()
                {
                    self.start_solver_worker();
                }
            } else if ui
                .add_sized(
                    [ui.available_width(), 36.0],
                    egui::Button::new(RichText::new("⏹ Stop CFD Solver").strong().size(15.0))
                        .fill(Color32::from_rgb(190, 55, 55)),
                )
                .clicked()
            {
                self.solver_cancel.store(true, Ordering::Relaxed);
            }
        });

        // Quick Aerodynamic Summary Box in Sidebar if a solution exists
        if let Some(sol) = &self.cfd_solution {
            ui.add_space(6.0);
            ui.group(|ui| {
                ui.label(
                    RichText::new("📊 Live Aerodynamic Coefficients")
                        .strong()
                        .color(Color32::from_rgb(120, 210, 255)),
                );
                let a = &sol.aero;
                egui::Grid::new("sidebar_aero_card")
                    .num_columns(2)
                    .spacing([12.0, 3.0])
                    .striped(true)
                    .show(ui, |ui| {
                        ui.label("Lift Coeff (Cl):");
                        ui.label(
                            RichText::new(format!(
                                "{:+.4} (KJ: {:+.4})",
                                a.cl, a.cl_kutta_joukowski
                            ))
                            .strong(),
                        );
                        ui.end_row();

                        ui.label("Drag Coeff (Cd):");
                        ui.label(format!("{:+.4}", a.cd));
                        ui.end_row();

                        ui.label("Moment (Cm_c/4):");
                        ui.label(format!("{:+.4}", a.cm_c4));
                        ui.end_row();

                        ui.label("Circulation (Γ):");
                        ui.label(format!("{:+.5}", a.circulation));
                        ui.end_row();

                        ui.label("Mach Range:");
                        ui.label(format!("[{:.3}, {:.3}]", a.min_mach, a.max_mach));
                        ui.end_row();

                        ui.label("Suction Cp_min:");
                        ui.label(format!("{:+.4} (x/c={:.2})", a.cp_min, a.x_cp_min));
                        ui.end_row();

                        ui.label("Stagnation Cp_max:");
                        ui.label(format!(
                            "{:+.4} (th: {:+.4})",
                            a.cp_max, a.cp_stag_theory
                        ));
                        ui.end_row();
                    });
            });
        }
    }

    fn render_bottom_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(
                &mut self.bottom_tab,
                BottomTab::SurfaceCpAndAero,
                "🎯 Surface Cp & Aerodynamics",
            );
            ui.selectable_value(
                &mut self.bottom_tab,
                BottomTab::CfdConvergence,
                "📉 CFD Convergence (AF2)",
            );
            ui.selectable_value(
                &mut self.bottom_tab,
                BottomTab::MesherDiagnostics,
                "📈 Mesher Convergence & Spacing",
            );
            ui.selectable_value(
                &mut self.bottom_tab,
                BottomTab::QualityAndLogs,
                "📊 Mesh Quality & Logs",
            );
        });
        ui.separator();

        match self.bottom_tab {
            BottomTab::SurfaceCpAndAero => self.render_surface_cp_panel(ui),
            BottomTab::CfdConvergence => self.render_cfd_convergence_plot(ui),
            BottomTab::MesherDiagnostics => self.render_mesher_diagnostics_panel(ui),
            BottomTab::QualityAndLogs => self.render_quality_and_logs(ui),
        }
    }

    fn render_surface_cp_panel(&mut self, ui: &mut egui::Ui) {
        let Some(sol) = &self.cfd_solution else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new(
                        "Switch to '🚀 2. CFD Solver' and click 'Run Full Potential Solver (AF2)' to plot Upper/Lower Surface Cp and Mach distributions.",
                    )
                    .weak(),
                );
            });
            return;
        };

        let avail = ui.available_size();
        ui.horizontal(|ui| {
            // Left ~70%: Interactive Surface Cp or Mach plot
            ui.allocate_ui(Vec2::new((avail.x * 0.72).max(200.0), avail.y), |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.plot_surface_mach, false, "Surface Cp(x/c)");
                    ui.selectable_value(&mut self.plot_surface_mach, true, "Surface Mach(x/c)");
                    if !self.plot_surface_mach {
                        ui.separator();
                        ui.checkbox(&mut self.invert_cp_axis, "Invert Y-Axis (-Cp Upward)");
                    }
                });

                let aero = &sol.aero;
                if !self.plot_surface_mach {
                    let sign = if self.invert_cp_axis { -1.0 } else { 1.0 };
                    let upper_pts: PlotPoints = aero
                        .upper_cp
                        .iter()
                        .map(|p| [p[0], sign * p[1]])
                        .collect();
                    let lower_pts: PlotPoints = aero
                        .lower_cp
                        .iter()
                        .map(|p| [p[0], sign * p[1]])
                        .collect();

                    let inv = self.invert_cp_axis;
                    Plot::new("surface_cp_plot")
                        .x_axis_label("Chordwise Position (x/c)")
                        .y_axis_label(if inv { "-Cp (Suction Upward)" } else { "Cp" })
                        .y_axis_formatter(move |mark, _range| {
                            let val = if inv { -mark.value } else { mark.value };
                            format!("{:+.2}", val)
                        })
                        .label_formatter(move |name, pt| {
                            let cp_val = if inv { -pt.y } else { pt.y };
                            if name.is_empty() {
                                format!("x/c = {:.4}\nCp = {:+.4}", pt.x, cp_val)
                            } else {
                                format!("{}\nx/c = {:.4}\nCp = {:+.4}", name, pt.x, cp_val)
                            }
                        })
                        .legend(egui_plot::Legend::default())
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(upper_pts)
                                    .name("Upper Surface (Extradorso)")
                                    .color(Color32::from_rgb(70, 165, 255))
                                    .width(2.2),
                            );
                            plot_ui.line(
                                Line::new(lower_pts)
                                    .name("Lower Surface (Intradorso)")
                                    .color(Color32::from_rgb(245, 90, 90))
                                    .width(2.2),
                            );
                            if let Some(cp_star) = aero.cp_star {
                                if aero.cp_min <= cp_star + 0.35 {
                                    plot_ui.line(
                                        Line::new(PlotPoints::from(vec![
                                            [0.0, sign * cp_star],
                                            [1.0, sign * cp_star],
                                        ]))
                                        .name(format!("Sonic Cp* = {:+.3} (M=1)", cp_star))
                                        .color(Color32::from_rgb(90, 220, 120))
                                        .style(egui_plot::LineStyle::Dashed { length: 6.0 }),
                                    );
                                }
                            }
                        });
                } else {
                    let upper_m: PlotPoints = aero.upper_mach.iter().copied().collect();
                    let lower_m: PlotPoints = aero.lower_mach.iter().copied().collect();
                    Plot::new("surface_mach_plot")
                        .x_axis_label("Chordwise Position (x/c)")
                        .y_axis_label("Local Surface Mach (M)")
                        .legend(egui_plot::Legend::default())
                        .show(ui, |plot_ui| {
                            plot_ui.line(
                                Line::new(upper_m)
                                    .name("Upper Surface Mach")
                                    .color(Color32::from_rgb(70, 165, 255))
                                    .width(2.2),
                            );
                            plot_ui.line(
                                Line::new(lower_m)
                                    .name("Lower Surface Mach")
                                    .color(Color32::from_rgb(245, 90, 90))
                                    .width(2.2),
                            );
                            plot_ui.line(
                                Line::new(PlotPoints::from(vec![[0.0, 1.0], [1.0, 1.0]]))
                                    .name("Sonic Threshold (M = 1.0)")
                                    .color(Color32::from_rgb(90, 220, 120))
                                    .style(egui_plot::LineStyle::Dashed { length: 6.0 }),
                            );
                        });
                }
            });

            ui.separator();

            // Right ~28%: Detailed Aerodynamic Summary Table
            ui.vertical(|ui| {
                let a = &sol.aero;
                ui.label(
                    RichText::new(format!(
                        "{} — M={:.3}, AoA={:.2}°, γ={:.2}",
                        sol.solver_family.label(),
                        a.mach_inf,
                        a.aoa_deg,
                        a.gamma_heat
                    ))
                    .strong(),
                );
                egui::ScrollArea::vertical().show(ui, |ui| {
                    egui::Grid::new("bottom_aero_table")
                        .num_columns(2)
                        .spacing([12.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("Lift Coeff (Cl):");
                            ui.label(RichText::new(format!("{:+.5}", a.cl)).strong());
                            ui.end_row();

                            ui.label("Kutta-Joukowski Cl:");
                            ui.label(format!("{:+.5}", a.cl_kutta_joukowski));
                            ui.end_row();

                            ui.label("Drag Coeff (Cd):");
                            ui.label(RichText::new(format!("{:+.5}", a.cd)).strong());
                            ui.end_row();

                            ui.label("Moment (Cm_c/4):");
                            ui.label(format!("{:+.5}", a.cm_c4));
                            ui.end_row();

                            ui.label("Normal / Axial (Cn, Ca):");
                            ui.label(format!("{:+.4}, {:+.4}", a.cn, a.ca));
                            ui.end_row();

                            ui.label("Circulation (Γ):");
                            ui.label(format!("{:+.6}", a.circulation));
                            ui.end_row();

                            ui.label("Suction Peak (Cp_min):");
                            ui.label(format!("{:+.4} @ x/c={:.3}", a.cp_min, a.x_cp_min));
                            ui.end_row();

                            ui.label("Stagnation (Cp_max):");
                            ui.label(format!(
                                "{:+.4} @ x/c={:.3} (Th: {:+.4})",
                                a.cp_max, a.x_cp_max, a.cp_stag_theory
                            ));
                            ui.end_row();

                            if let Some(cp_star) = a.cp_star {
                                ui.label("Critical Sonic (Cp*):");
                                ui.label(format!("{:+.4}", cp_star));
                                ui.end_row();
                            }
                        });
                });
            });
        });
    }

    fn render_cfd_convergence_plot(&mut self, ui: &mut egui::Ui) {
        if self.cfd_max_res_history.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new(
                        "Run the Full Potential AF2 Solver to monitor live Max Residual, RMS Residual, and Circulation Γ convergence.",
                    )
                    .weak(),
                );
            });
            return;
        }

        let max_pts: PlotPoints = self.cfd_max_res_history.iter().copied().collect();
        let rms_pts: PlotPoints = self.cfd_rms_res_history.iter().copied().collect();
        let tol_log = self.fp_config.convergence_residual.max(1e-16).log10();
        let max_x = self
            .cfd_max_res_history
            .last()
            .map(|p| p[0])
            .unwrap_or(100.0)
            .max(10.0);

        Plot::new("cfd_af2_convergence_plot")
            .x_axis_label("Iteration")
            .y_axis_label("Log10(Residual |L(φ)|)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(max_pts)
                        .name("Log10(Max Residual)")
                        .color(Color32::from_rgb(250, 120, 75))
                        .width(2.0),
                );
                plot_ui.line(
                    Line::new(rms_pts)
                        .name("Log10(RMS Residual)")
                        .color(Color32::from_rgb(80, 185, 250))
                        .width(1.8),
                );
                plot_ui.line(
                    Line::new(PlotPoints::from(vec![[0.0, tol_log], [max_x, tol_log]]))
                        .name("Target Tolerance")
                        .color(Color32::from_rgb(100, 220, 120))
                        .style(egui_plot::LineStyle::Dashed { length: 6.0 }),
                );
            });
    }

    fn render_mesher_diagnostics_panel(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |cols| {
            // Left column: Elliptic Mesher convergence
            if self.mesher_residual_history.is_empty() {
                cols[0].centered_and_justified(|ui| {
                    ui.label(
                        RichText::new(
                            "Run Laplace or Poisson Elliptic Smoother to view mesh residual history.",
                        )
                        .weak(),
                    );
                });
            } else {
                let pts: PlotPoints = self.mesher_residual_history.iter().copied().collect();
                let tol_log = self.mesh_config.conv_criterion.max(1e-16).log10();
                let max_x = self
                    .mesher_residual_history
                    .last()
                    .map(|p| p[0])
                    .unwrap_or(100.0)
                    .max(10.0);

                Plot::new("elliptic_convergence_plot")
                    .x_axis_label("Iteration")
                    .y_axis_label("Log10(Max |Δx, Δy|)")
                    .legend(egui_plot::Legend::default())
                    .show(&mut cols[0], |plot_ui| {
                        plot_ui.line(
                            Line::new(pts)
                                .name("Log10(Max |Δx, Δy|)")
                                .color(Color32::from_rgb(80, 180, 250))
                                .width(2.0),
                        );
                        plot_ui.line(
                            Line::new(PlotPoints::from(vec![[0.0, tol_log], [max_x, tol_log]]))
                                .name("Target Tolerance")
                                .color(Color32::from_rgb(100, 220, 120))
                                .style(egui_plot::LineStyle::Dashed { length: 6.0 }),
                        );
                    });
            }

            // Right column: Airfoil Surface Clustering
            if let Some(grid) = &self.grid {
                let (ni, _) = grid.dim();
                let mut spacing_pts = Vec::with_capacity(ni - 1);
                for i in 0..(ni - 1) {
                    let p0 = grid[[i, 0]];
                    let p1 = grid[[i + 1, 0]];
                    let xm = 0.5 * (p0.x + p1.x);
                    let ds = ((p1.x - p0.x).powi(2) + (p1.y - p0.y).powi(2)).sqrt();
                    spacing_pts.push([xm, ds]);
                }

                Plot::new("surface_clustering_plot")
                    .x_axis_label("Chordwise Position (x/c)")
                    .y_axis_label("Surface Panel Length Δs_ξ")
                    .legend(egui_plot::Legend::default())
                    .show(&mut cols[1], |plot_ui| {
                        plot_ui.line(
                            Line::new(PlotPoints::from(spacing_pts))
                                .name("Surface Spacing Δs_ξ(x/c)")
                                .color(Color32::from_rgb(245, 165, 65))
                                .width(2.0),
                        );
                    });
            }
        });
    }

    fn render_quality_and_logs(&mut self, ui: &mut egui::Ui) {
        ui.columns(2, |cols| {
            cols[0].label(RichText::new("Geometric & Metric Quality Summary").strong());
            let s = &self.quality_summary;
            egui::Grid::new("quality_summary_table")
                .num_columns(4)
                .spacing([14.0, 4.0])
                .striped(true)
                .show(&mut cols[0], |ui| {
                    ui.label("Grid Dimensions:");
                    ui.label(format!("{} x {}", s.ni, s.nj));
                    ui.label("Total Nodes / Cells:");
                    ui.label(format!("{} / {}", s.total_nodes, s.total_cells));
                    ui.end_row();

                    ui.label("Min Wall Normal Δs_η:");
                    ui.label(format!("{:.5}", s.min_wall_spacing_normal));
                    ui.label("Max Wall Normal Δs_η:");
                    ui.label(format!("{:.5}", s.max_wall_spacing_normal));
                    ui.end_row();

                    ui.label("Min Wall Tangent Δs_ξ:");
                    ui.label(format!("{:.5}", s.min_wall_spacing_tangent));
                    ui.label("Avg Wall Ortho Dev:");
                    ui.label(format!("{:.2}°", s.wall_avg_orthogonality_dev_deg));
                    ui.end_row();

                    ui.label("Max Ortho Deviation:");
                    ui.label(format!("{:.2}°", s.max_orthogonality_dev_deg));
                    ui.label("Max Aspect Ratio:");
                    ui.label(format!("{:.2}", s.max_aspect_ratio));
                    ui.end_row();

                    ui.label("Min Jacobian J:");
                    ui.label(format!("{:.3e}", s.min_jacobian));
                    ui.label("Inverted Cells (J ≤ 0):");
                    ui.colored_label(
                        if s.negative_jacobian_count == 0 {
                            Color32::from_rgb(80, 210, 120)
                        } else {
                            Color32::from_rgb(230, 80, 80)
                        },
                        format!("{}", s.negative_jacobian_count),
                    );
                    ui.end_row();
                });

            cols[1].label(RichText::new("Event Log").strong());
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(&mut cols[1], |ui| {
                    for line in &self.logs {
                        ui.label(RichText::new(line).monospace().small());
                    }
                });
        });
    }

    fn render_viewport(&mut self, ui: &mut egui::Ui) {
        if let Some(err) = self.error_banner.clone() {
            ui.horizontal(|ui| {
                ui.colored_label(Color32::from_rgb(245, 90, 90), format!("⚠ {}", err));
                if ui.small_button("Dismiss").clicked() {
                    self.error_banner = None;
                }
            });
            ui.separator();
        }

        // Viewport toolbar
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Camera:").strong());
            if ui.button("🔍 Airfoil").clicked() {
                self.pending_camera = Some(CameraPreset::Airfoil);
            }
            if ui.button("LE").clicked() {
                self.pending_camera = Some(CameraPreset::LeadingEdge);
            }
            if ui.button("TE").clicked() {
                self.pending_camera = Some(CameraPreset::TrailingEdge);
            }
            if ui.button("🌐 Farfield").clicked() {
                self.pending_camera = Some(CameraPreset::Farfield);
            }

            ui.separator();
            ui.label(RichText::new("Field:").strong());
            let prev_scalar = self.viewport_scalar;
            let has_cfd = self.cfd_solution.is_some();
            egui::ComboBox::from_id_salt("viewport_scalar_combo")
                .selected_text(self.viewport_scalar.label())
                .show_ui(ui, |ui| {
                    if has_cfd {
                        ui.label(RichText::new("CFD Flowfield").weak().small());
                        for f in FlowScalarField::ALL {
                            let item = ViewportScalar::Flow(f);
                            ui.selectable_value(&mut self.viewport_scalar, item, item.label());
                        }
                        ui.separator();
                    }
                    ui.label(RichText::new("Mesh & Metrics").weak().small());
                    for m in [
                        MeshScalarField::None,
                        MeshScalarField::OrthogonalityDev,
                        MeshScalarField::AspectRatio,
                        MeshScalarField::CellArea,
                        MeshScalarField::Log10Jacobian,
                        MeshScalarField::MetricA1,
                        MeshScalarField::MetricA2,
                        MeshScalarField::MetricA3,
                    ] {
                        let item = ViewportScalar::Mesh(m);
                        ui.selectable_value(&mut self.viewport_scalar, item, item.label());
                    }
                });
            if self.viewport_scalar != prev_scalar {
                self.recompute_scalar_range();
            }

            if !self.viewport_scalar.is_none() {
                egui::ComboBox::from_id_salt("colormap_combo")
                    .width(120.0)
                    .selected_text(self.colormap.label())
                    .show_ui(ui, |ui| {
                        for cmap in Colormap::ALL {
                            ui.selectable_value(&mut self.colormap, cmap, cmap.label());
                        }
                    });
                ui.checkbox(&mut self.show_iso_contours, "Iso-Lines");
            }

            if has_cfd {
                ui.checkbox(&mut self.show_sonic_line, "Sonic M=1")
                    .on_hover_text("Draw the M = 1.0 sonic contour line in magenta");
                ui.checkbox(&mut self.show_velocity_vectors, "Vectors")
                    .on_hover_text("Overlay local velocity vectors (u, v)");
            }

            ui.separator();
            ui.checkbox(&mut self.show_grid_lines, "Grid");
            if self.show_grid_lines || self.show_velocity_vectors || self.show_nodes {
                egui::ComboBox::from_id_salt("stride_combo")
                    .width(58.0)
                    .selected_text(format!("1/{}", self.grid_line_stride))
                    .show_ui(ui, |ui| {
                        for s in [1usize, 2, 4, 8] {
                            ui.selectable_value(
                                &mut self.grid_line_stride,
                                s,
                                format!("Stride {}", s),
                            );
                        }
                    });
            }
            ui.checkbox(&mut self.show_airfoil_wall, "Airfoil");
            ui.checkbox(&mut self.show_wake_cut, "Wake");
            ui.checkbox(&mut self.enable_hover_inspector, "Probe");
        });

        ui.separator();

        let Some(grid) = &self.grid else {
            ui.centered_and_justified(|ui| {
                ui.label("No mesh generated.");
            });
            return;
        };

        let r_max = self.mesh_config.r_max;
        let pending_cam = self.pending_camera.take();

        let total_rect = ui.available_rect_before_wrap();
        let colorbar_width = if !self.viewport_scalar.is_none() {
            76.0
        } else {
            0.0
        };
        let plot_width = (total_rect.width() - colorbar_width).max(100.0);

        ui.horizontal(|ui| {
            ui.set_height(total_rect.height());

            let plot_response = ui
                .allocate_ui(Vec2::new(plot_width, total_rect.height()), |ui| {
                    Plot::new("physical_mesh_viewport")
                        .data_aspect(1.0)
                        .show_axes(true)
                        .show_grid(true)
                        .allow_drag(true)
                        .allow_zoom(true)
                        .allow_scroll(true)
                        .x_axis_label("x / c")
                        .y_axis_label("y / c")
                        .show(ui, |plot_ui| {
                            if let Some(preset) = pending_cam {
                                let bounds = match preset {
                                    CameraPreset::Airfoil => {
                                        PlotBounds::from_min_max([-0.35, -0.65], [1.35, 0.65])
                                    }
                                    CameraPreset::LeadingEdge => {
                                        PlotBounds::from_min_max([-0.08, -0.10], [0.15, 0.10])
                                    }
                                    CameraPreset::TrailingEdge => {
                                        PlotBounds::from_min_max([0.85, -0.10], [1.12, 0.10])
                                    }
                                    CameraPreset::Farfield => PlotBounds::from_min_max(
                                        [0.5 - r_max * 1.1, -r_max * 1.1],
                                        [0.5 + r_max * 1.1, r_max * 1.1],
                                    ),
                                };
                                plot_ui.set_plot_bounds(bounds);
                            }
                        })
                })
                .inner;

            let transform = &plot_response.transform;
            let frame_rect = *transform.frame();
            let painter = ui.painter_at(frame_rect);

            let (ni, nj) = grid.dim();

            // 1. Precompute screen-space Pos2 and active scalar values for all grid nodes
            let mut screen_pts = Array2::<Pos2>::from_elem((ni, nj), Pos2::ZERO);
            let mut scalar_vals = Array2::<f64>::zeros((ni, nj));
            let active_scalar = !self.viewport_scalar.is_none();

            for j in 0..nj {
                for i in 0..ni {
                    let pt = grid[[i, j]];
                    screen_pts[[i, j]] =
                        transform.position_from_point(&PlotPoint::new(pt.x, pt.y));
                    if active_scalar {
                        scalar_vals[[i, j]] = self.sample_active_scalar(i, j);
                    }
                }
            }

            // 2. Render Gouraud-shaded structured quad mesh if a scalar field is active
            if active_scalar {
                let (min_v, max_v) = self.scalar_range;
                let inv_span = 1.0 / (max_v - min_v).max(1e-12);
                let mut mesh = egui::Mesh::default();
                mesh.vertices.reserve(ni * nj);
                mesh.indices.reserve((ni - 1) * (nj - 1) * 6);

                for j in 0..nj {
                    for i in 0..ni {
                        let val = scalar_vals[[i, j]];
                        let t = ((val - min_v) * inv_span).clamp(0.0, 1.0);
                        let color = self.colormap.sample(t);
                        mesh.vertices.push(egui::epaint::Vertex {
                            pos: screen_pts[[i, j]],
                            uv: egui::epaint::WHITE_UV,
                            color,
                        });
                    }
                }

                for j in 0..(nj - 1) {
                    let row0 = (j * ni) as u32;
                    let row1 = ((j + 1) * ni) as u32;
                    for i in 0..(ni - 1) {
                        let idx00 = row0 + (i as u32);
                        let idx10 = idx00 + 1;
                        let idx01 = row1 + (i as u32);
                        let idx11 = idx01 + 1;

                        mesh.indices
                            .extend_from_slice(&[idx00, idx10, idx11, idx00, idx11, idx01]);
                    }
                }

                painter.add(Shape::mesh(mesh));

                // 2b. Optional Iso-Contour lines across the active scalar field
                if self.show_iso_contours && self.iso_contour_levels >= 2 {
                    let n_levels = self.iso_contour_levels;
                    let iso_stroke =
                        Stroke::new(1.0, Color32::from_rgba_unmultiplied(15, 15, 20, 155));
                    for k in 1..n_levels {
                        let frac = (k as f64) / (n_levels as f64);
                        let iso_val = min_v + frac * (max_v - min_v);
                        draw_iso_contour_on_grid(
                            &painter,
                            &screen_pts,
                            &scalar_vals,
                            iso_val,
                            iso_stroke,
                        );
                    }
                }
            }

            // 3. Render Sonic M = 1.0 Line if CFD solution is present and toggle is active
            if self.show_sonic_line {
                if let Some(sol) = &self.cfd_solution {
                    if sol.aero.max_mach >= 1.0 {
                        draw_iso_contour_on_grid(
                            &painter,
                            &screen_pts,
                            &sol.mach,
                            1.0,
                            Stroke::new(2.4, Color32::from_rgb(255, 60, 220)),
                        );
                    }
                }
            }

            // 4. Render solid airfoil interior mask
            if ni > 2 {
                let airfoil_poly: Vec<Pos2> = (0..(ni - 1)).map(|i| screen_pts[[i, 0]]).collect();
                painter.add(Shape::convex_polygon(
                    airfoil_poly,
                    Color32::from_rgba_unmultiplied(32, 36, 44, 240),
                    Stroke::NONE,
                ));
            }

            // 5. Render O-Grid Wireframe lines
            if self.show_grid_lines {
                let stride = self.grid_line_stride.max(1);
                let line_color = if !active_scalar {
                    Color32::from_rgba_unmultiplied(90, 165, 235, 135)
                } else {
                    Color32::from_rgba_unmultiplied(20, 20, 25, 75)
                };
                let stroke = Stroke::new(1.0, line_color);

                for j in (0..nj).step_by(stride) {
                    let pts: Vec<Pos2> = (0..ni).map(|i| screen_pts[[i, j]]).collect();
                    painter.add(Shape::line(pts, stroke));
                }
                if (nj - 1) % stride != 0 {
                    let pts: Vec<Pos2> = (0..ni).map(|i| screen_pts[[i, nj - 1]]).collect();
                    painter.add(Shape::line(pts, stroke));
                }

                for i in (0..(ni - 1)).step_by(stride) {
                    let pts: Vec<Pos2> = (0..nj).map(|j| screen_pts[[i, j]]).collect();
                    painter.add(Shape::line(pts, stroke));
                }
            }

            // 6. Highlight Wake Cut line (i = 0 / i = ni - 1)
            if self.show_wake_cut {
                let wake_pts: Vec<Pos2> = (0..nj).map(|j| screen_pts[[0, j]]).collect();
                painter.add(Shape::line(
                    wake_pts,
                    Stroke::new(1.6, Color32::from_rgba_unmultiplied(240, 180, 60, 180)),
                ));
            }

            // 7. Highlight Airfoil Surface (j = 0)
            if self.show_airfoil_wall {
                let wall_pts: Vec<Pos2> = (0..ni).map(|i| screen_pts[[i, 0]]).collect();
                painter.add(Shape::line(
                    wall_pts,
                    Stroke::new(2.4, Color32::from_rgb(245, 245, 250)),
                ));
            }

            // 8. Optional Velocity Vectors Overlay
            if self.show_velocity_vectors {
                if let Some(sol) = &self.cfd_solution {
                    let stride = (self.grid_line_stride * 2).max(2);
                    let q_ref = self.fp_config.q_inf.max(0.1);
                    let arrow_stroke =
                        Stroke::new(1.2, Color32::from_rgba_unmultiplied(250, 250, 255, 200));

                    for j in (0..nj).step_by(stride) {
                        for i in (0..(ni - 1)).step_by(stride) {
                            let p0 = screen_pts[[i, j]];
                            if !frame_rect.contains(p0) {
                                continue;
                            }
                            let u = sol.velocity_u[[i, j]];
                            let v = sol.velocity_v[[i, j]];
                            let mag = (u * u + v * v).sqrt();
                            if mag < 1e-6 {
                                continue;
                            }
                            let len_px = (14.0 * (mag / q_ref).clamp(0.25, 2.2)) as f32;
                            // Note: screen Y points downward, physical Y points upward
                            let dir = Vec2::new((u / mag) as f32, -(v / mag) as f32);
                            let p1 = p0 + dir * len_px;
                            painter.line_segment([p0, p1], arrow_stroke);

                            let perp = Vec2::new(-dir.y, dir.x);
                            let head_base = p1 - dir * 4.0;
                            painter.line_segment([p1, head_base + perp * 2.2], arrow_stroke);
                            painter.line_segment([p1, head_base - perp * 2.2], arrow_stroke);
                        }
                    }
                }
            }

            // 9. Hover Probe Inspector
            if self.enable_hover_inspector {
                if let Some(hover_pos) = plot_response.response.hover_pos() {
                    if frame_rect.contains(hover_pos) {
                        let mut best_ij = (0usize, 0usize);
                        let mut best_dist_sq = f32::MAX;
                        for j in 0..nj {
                            for i in 0..(ni - 1) {
                                let d2 = screen_pts[[i, j]].distance_sq(hover_pos);
                                if d2 < best_dist_sq {
                                    best_dist_sq = d2;
                                    best_ij = (i, j);
                                }
                            }
                        }

                        if best_dist_sq < 25.0 * 25.0 {
                            let (hi, hj) = best_ij;
                            let target_pos = screen_pts[[hi, hj]];
                            painter.circle_stroke(
                                target_pos,
                                5.5,
                                Stroke::new(2.0, Color32::from_rgb(255, 220, 60)),
                            );

                            let pt = grid[[hi, hj]];
                            let mut tooltip = format!(
                                "Node (i={}, j={})\nx = {:.5}, y = {:.5}",
                                hi, hj, pt.x, pt.y
                            );
                            if let Some(sol) = &self.cfd_solution {
                                tooltip.push_str(&format!(
                                    "\n────────────────────\nMach : {:.4}\nCp   : {:+.4}\nρ/ρt : {:.4}\n(u,v): ({:+.4}, {:+.4})\nφ    : {:+.4}",
                                    sol.mach[[hi, hj]],
                                    sol.cp[[hi, hj]],
                                    sol.density[[hi, hj]],
                                    sol.velocity_u[[hi, hj]],
                                    sol.velocity_v[[hi, hj]],
                                    sol.potential[[hi, hj]]
                                ));
                            } else if let Some(metrics) = &self.quality_metrics {
                                let m = &metrics[[hi, hj]];
                                tooltip.push_str(&format!(
                                    "\n────────────────────\nOrtho Dev: {:.2}°\nAspect   : {:.2}\nCell Area: {:.2e}\nJacobian : {:.2e}",
                                    m.orthogonality_dev_deg, m.aspect_ratio, m.cell_area, m.jacobian
                                ));
                            }
                            egui::show_tooltip_at_pointer(
                                ui.ctx(),
                                ui.layer_id(),
                                egui::Id::new("mesh_node_inspector"),
                                |ui| {
                                    ui.label(RichText::new(tooltip).monospace().small());
                                },
                            );
                        }
                    }
                }
            }

            // 10. Vertical Colorbar Legend
            if active_scalar {
                self.render_colorbar(ui, total_rect.height());
            }
        });
    }

    fn render_colorbar(&self, ui: &mut egui::Ui, height: f32) {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(68.0, height), Sense::hover());
        let painter = ui.painter_at(rect);

        let bar_top = rect.top() + 24.0;
        let bar_bottom = (rect.bottom() - 24.0).max(bar_top + 20.0);
        let bar_rect = Rect::from_min_max(
            Pos2::new(rect.left() + 6.0, bar_top),
            Pos2::new(rect.left() + 22.0, bar_bottom),
        );

        let steps = 64usize;
        for k in 0..steps {
            let t0 = (k as f32) / (steps as f32);
            let t1 = ((k + 1) as f32) / (steps as f32);
            let y0 = bar_rect.bottom() - t0 * bar_rect.height();
            let y1 = bar_rect.bottom() - t1 * bar_rect.height();
            let c = self.colormap.sample(t0 as f64);
            painter.rect_filled(
                Rect::from_min_max(
                    Pos2::new(bar_rect.left(), y1),
                    Pos2::new(bar_rect.right(), y0),
                ),
                0.0,
                c,
            );
        }
        painter.rect_stroke(
            bar_rect,
            0.0,
            Stroke::new(1.0, Color32::GRAY),
            egui::StrokeKind::Outside,
        );

        let (min_v, max_v) = self.scalar_range;
        let mid_v = 0.5 * (min_v + max_v);
        let font = egui::FontId::monospace(10.5);

        painter.text(
            Pos2::new(bar_rect.right() + 4.0, bar_rect.top()),
            egui::Align2::LEFT_CENTER,
            format!("{:.2}", max_v),
            font.clone(),
            Color32::LIGHT_GRAY,
        );
        painter.text(
            Pos2::new(bar_rect.right() + 4.0, bar_rect.center().y),
            egui::Align2::LEFT_CENTER,
            format!("{:.2}", mid_v),
            font.clone(),
            Color32::LIGHT_GRAY,
        );
        painter.text(
            Pos2::new(bar_rect.right() + 4.0, bar_rect.bottom()),
            egui::Align2::LEFT_CENTER,
            format!("{:.2}", min_v),
            font,
            Color32::LIGHT_GRAY,
        );
    }
}

/// Extracts and draws iso-contour segments across a structured 2D quad grid by
/// splitting each quad into the exact same two triangles `(00, 10, 11)` and `(00, 11, 01)`
/// used by the Gouraud `egui::Mesh`.
fn draw_iso_contour_on_grid(
    painter: &egui::Painter,
    screen_pts: &Array2<Pos2>,
    field: &Array2<f64>,
    iso: f64,
    stroke: Stroke,
) {
    let (ni, nj) = field.dim();
    for j in 0..(nj - 1) {
        for i in 0..(ni - 1) {
            let p00 = screen_pts[[i, j]];
            let p10 = screen_pts[[i + 1, j]];
            let p11 = screen_pts[[i + 1, j + 1]];
            let p01 = screen_pts[[i, j + 1]];

            let v00 = field[[i, j]];
            let v10 = field[[i + 1, j]];
            let v11 = field[[i + 1, j + 1]];
            let v01 = field[[i, j + 1]];

            draw_iso_triangle(painter, p00, v00, p10, v10, p11, v11, iso, stroke);
            draw_iso_triangle(painter, p00, v00, p11, v11, p01, v01, iso, stroke);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_iso_triangle(
    painter: &egui::Painter,
    p0: Pos2,
    v0: f64,
    p1: Pos2,
    v1: f64,
    p2: Pos2,
    v2: f64,
    iso: f64,
    stroke: Stroke,
) {
    let mut hits = [Pos2::ZERO; 3];
    let mut count = 0usize;

    let edges = [((p0, v0), (p1, v1)), ((p1, v1), (p2, v2)), ((p2, v2), (p0, v0))];
    for &((pa, va), (pb, vb)) in &edges {
        let da = va - iso;
        let db = vb - iso;
        if (da >= 0.0 && db < 0.0) || (da < 0.0 && db >= 0.0) {
            let denom = vb - va;
            if denom.abs() > 1e-14 {
                let t = ((iso - va) / denom).clamp(0.0, 1.0) as f32;
                hits[count] = pa + (pb - pa) * t;
                count += 1;
            }
        }
    }

    if count >= 2 {
        painter.line_segment([hits[0], hits[1]], stroke);
    }
}

impl eframe::App for RustyFlowApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_worker_events(ctx);

        egui::TopBottomPanel::top("top_toolbar_panel").show(ctx, |ui| {
            self.render_top_bar(ui);
        });

        egui::SidePanel::left("left_control_sidebar")
            .resizable(true)
            .default_width(345.0)
            .width_range(295.0..=480.0)
            .show(ctx, |ui| {
                self.render_left_sidebar(ui);
            });

        egui::TopBottomPanel::bottom("bottom_analytics_panel")
            .resizable(true)
            .default_height(245.0)
            .height_range(150.0..=460.0)
            .show(ctx, |ui| {
                self.render_bottom_panel(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_viewport(ui);
        });
    }
}
