pub mod config;
pub mod elliptic;
pub mod geometry;
pub mod io;
pub mod parabolic;

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};
use ndarray::Array2;

use crate::core::grid::Point;
use crate::mesher::config::{Config, EllipticEquation};
use crate::mesher::elliptic::EllipticStats;

/// Events streamed from the background Mesher worker thread to the GUI thread.
pub enum MesherEvent {
    /// Sent immediately after the fast parabolic (Nakamura marching) mesh is generated.
    ParabolicCompleted {
        grid: Array2<Point>,
        duration: Duration,
    },
    /// Sent during the iterative elliptic smoother (AF1 / ADI).
    Iteration {
        iter: usize,
        max_iter: usize,
        residual: f64,
        snapshot: Option<Array2<Point>>,
    },
    /// Sent when the entire mesh generation pipeline finishes.
    Finished {
        grid: Array2<Point>,
        stats: EllipticStats,
        total_duration: Duration,
    },
    /// Sent if an error (such as numerical divergence) occurs.
    Error {
        message: String,
    },
}

/// Runs the full mesh generation pipeline synchronously or on a background worker thread,
/// streaming live progress events through `tx`.
pub fn run_mesher_pipeline(
    mut config: Config,
    tx: Sender<MesherEvent>,
    cancel_flag: Arc<AtomicBool>,
    snapshot_interval: usize,
) {
    let start_total = Instant::now();
    config.sanitize();

    // 1. Allocate grid & insert inner (airfoil) + outer (farfield) boundaries
    let mut grid = io::create_grid(config.longitudinal_points, config.normal_points);
    geometry::insert_geometry(&config, &mut grid);

    // 2. Step 1: Parabolic marching mesh (Nakamura)
    let start_para = Instant::now();
    parabolic::generate_parabolic_mesh(&config, &mut grid);
    let para_duration = start_para.elapsed();

    let _ = tx.send(MesherEvent::ParabolicCompleted {
        grid: grid.clone(),
        duration: para_duration,
    });

    if config.elliptic_equation == EllipticEquation::ParabolicOnly {
        let _ = tx.send(MesherEvent::Finished {
            grid,
            stats: EllipticStats {
                iterations: 0,
                final_residual: 0.0,
                converged: true,
                cancelled: false,
            },
            total_duration: start_total.elapsed(),
        });
        return;
    }

    // 3. Step 2: Elliptic PDE smoother (Laplace or Steger-Sorenson Poisson via AF1/ADI)
    match elliptic::elliptic_smoother(
        &config,
        &mut grid,
        Some(&tx),
        Some(&cancel_flag),
        snapshot_interval,
    ) {
        Ok(stats) => {
            let _ = tx.send(MesherEvent::Finished {
                grid,
                stats,
                total_duration: start_total.elapsed(),
            });
        }
        Err(err_msg) => {
            let _ = tx.send(MesherEvent::Error { message: err_msg });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::grid::compute_mesh_quality;
    use std::sync::mpsc;

    #[test]
    fn test_mesher_pipeline_laplace_and_poisson() {
        let mut cfg = Config::preset_naca0012_laplace();
        cfg.longitudinal_points = 93;
        cfg.normal_points = 25;
        cfg.r_max = 6.5;
        cfg.alpha_h = 10.0;
        cfg.max_iter = 200;

        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        run_mesher_pipeline(cfg, tx, cancel, 50);

        let mut final_grid = None;
        for ev in rx {
            match ev {
                MesherEvent::Finished { grid, stats, .. } => {
                    assert!(stats.iterations > 0);
                    assert!(stats.final_residual.is_finite());
                    final_grid = Some(grid);
                }
                MesherEvent::Error { message } => {
                    panic!("Mesher returned error: {}", message);
                }
                _ => {}
            }
        }
        let grid = final_grid.expect("Mesher should finish successfully");
        let (_, summary) = compute_mesh_quality(&grid);
        assert_eq!(summary.ni, 93);
        assert_eq!(summary.nj, 25);
        assert_eq!(summary.negative_jacobian_count, 0);
    }
}

