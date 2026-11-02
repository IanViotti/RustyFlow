pub mod full_potential;

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};
use ndarray::Array2;

use crate::core::grid::Point;
use crate::core::solver_trait::CfdSolution2D;
use crate::solvers::full_potential::config::Config as FullPotentialConfig;
use crate::solvers::full_potential::solver_core::{solve_full_potential, SolverStats};
use crate::solvers::full_potential::solver_utils::{build_cfd_solution, Grid, PointData};

/// Events streamed from any background CFD Solver worker thread to the GUI thread.
pub enum SolverEvent {
    Iteration {
        iter: usize,
        max_iter: usize,
        max_res: f64,
        rms_res: f64,
        circulation: f64,
        beta_sup: f64,
        snapshot: Option<CfdSolution2D>,
    },
    Finished {
        solution: CfdSolution2D,
        raw_fp_domain: Option<(Grid, Array2<PointData>)>,
        stats: SolverStats,
        total_duration: Duration,
    },
    Error {
        message: String,
    },
}

/// Runs the Full Potential AF2 solver on a background worker thread and streams telemetry via `tx`.
pub fn run_full_potential_pipeline(
    mesher_grid: Array2<Point>,
    mut config: FullPotentialConfig,
    tx: Sender<SolverEvent>,
    cancel_flag: Arc<AtomicBool>,
    snapshot_interval: usize,
) {
    let start = Instant::now();
    config.compute_derived_variables();

    let solver_grid = Grid::from_mesher_grid(&mesher_grid);

    match solve_full_potential(
        &mesher_grid,
        &solver_grid,
        &config,
        Some(&tx),
        Some(&cancel_flag),
        snapshot_interval,
    ) {
        Ok((domain, stats)) => {
            let solution = build_cfd_solution(
                &config,
                &solver_grid,
                &mesher_grid,
                &domain,
                stats.final_gamma,
            );
            let _ = tx.send(SolverEvent::Finished {
                solution,
                raw_fp_domain: Some((solver_grid, domain)),
                stats,
                total_duration: start.elapsed(),
            });
        }
        Err(err_msg) => {
            let _ = tx.send(SolverEvent::Error { message: err_msg });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::config::{Config as MesherConfig, EllipticEquation};
    use crate::mesher::{run_mesher_pipeline, MesherEvent};
    use std::sync::mpsc;

    #[test]
    fn test_end_to_end_mesher_to_full_potential_solver() {
        // 1. Generate a 93x25 NACA0012 mesh
        let mut m_cfg = MesherConfig::preset_naca0012_laplace();
        m_cfg.longitudinal_points = 93;
        m_cfg.normal_points = 25;
        m_cfg.r_max = 6.5;
        m_cfg.elliptic_equation = EllipticEquation::ParabolicOnly;

        let (m_tx, m_rx) = mpsc::channel();
        run_mesher_pipeline(m_cfg, m_tx, Arc::new(AtomicBool::new(false)), 0);

        let mut grid = None;
        for ev in m_rx {
            if let MesherEvent::Finished { grid: g, .. } = ev {
                grid = Some(g);
            }
        }
        let grid = grid.expect("Mesh should be generated");

        // 2. Solve Subsonic Lifting Flow (M = 0.3, AoA = 2.0 deg)
        let mut fp_cfg = FullPotentialConfig::preset_subsonic_symmetric();
        fp_cfg.Mach_inf = 0.30;
        fp_cfg.AoA = 2.0;
        fp_cfg.Rg = 0.35;
        fp_cfg.max_iterations = 400;

        let (s_tx, s_rx) = mpsc::channel();
        run_full_potential_pipeline(grid, fp_cfg, s_tx, Arc::new(AtomicBool::new(false)), 0);

        let mut final_sol = None;
        for ev in s_rx {
            match ev {
                SolverEvent::Finished {
                    solution, stats, ..
                } => {
                    assert!(stats.iterations > 0);
                    assert!(stats.final_max_res.is_finite());
                    final_sol = Some(solution);
                }
                SolverEvent::Error { message } => {
                    panic!("Solver failed: {}", message);
                }
                _ => {}
            }
        }

        let sol = final_sol.expect("CFD solution should be produced");
        assert!(
            sol.aero.cl > 0.1,
            "Positive AoA should produce positive Lift Coefficient Cl, got {}",
            sol.aero.cl
        );
        assert!(
            sol.aero.circulation > 0.0,
            "Positive AoA should produce positive circulation Gamma, got {}",
            sol.aero.circulation
        );
    }
}

