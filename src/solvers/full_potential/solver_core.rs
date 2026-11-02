#![allow(non_snake_case)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use ndarray::Array2;

use crate::core::grid::Point;
use crate::solvers::full_potential::af2;
use crate::solvers::full_potential::config::Config;
use crate::solvers::full_potential::solver_utils::{self, Grid, PointData};
use crate::solvers::SolverEvent;

/// Statistics recorded during solver convergence
#[derive(Clone, Copy, Debug)]
pub struct SolverStats {
    pub iterations: usize,
    pub converged: bool,
    pub cancelled: bool,
    pub final_max_res: f64,
    pub final_rms_res: f64,
    pub final_gamma: f64,
}

/// Main entry point for the Full Potential CFD solver.
/// Initializes the domain, calculates metrics, and runs the iterative AF2 scheme.
pub fn solve_full_potential(
    mesher_grid: &Array2<Point>,
    solver_grid: &Grid,
    config: &Config,
    tx: Option<&Sender<SolverEvent>>,
    cancel_flag: Option<&Arc<AtomicBool>>,
    snapshot_interval: usize,
) -> Result<(Array2<PointData>, SolverStats), String> {
    // 1. Initialize domain arrays
    let mut domain = solver_utils::initialize_domain(config, solver_grid);

    // 2. Calculate geometric metrics in-place
    solver_utils::calculate_metrics(solver_grid, &mut domain);

    // 3. Enter the main iterative solver loop (AF2 scheme)
    let stats = main_solver_loop(
        config,
        mesher_grid,
        solver_grid,
        &mut domain,
        tx,
        cancel_flag,
        snapshot_interval,
    )?;

    // 4. Calculate true Cartesian velocities and final flow variables for post-processing/VTK
    solver_utils::calculate_field(config, solver_grid, &mut domain, stats.final_gamma);

    Ok((domain, stats))
}

fn main_solver_loop(
    config: &Config,
    mesher_grid: &Array2<Point>,
    grid: &Grid,
    domain: &mut Array2<PointData>,
    tx: Option<&Sender<SolverEvent>>,
    cancel_flag: Option<&Arc<AtomicBool>>,
    snapshot_interval: usize,
) -> Result<SolverStats, String> {
    // Sequence of Holst's alpha parameter to accelerate convergence
    let alphas = solver_utils::alpha_seq(config, grid);
    let omega = config.omega; // Relaxation factor
    let m_cycle = alphas.len(); // Length of the alpha cycle (M)

    // Initialization of Beta control variables (Page 50 of lecture notes)
    let beta_sub: f64 = 0.3;
    let mut beta_sup: f64 = config.beta_mult;
    let b_high = beta_sup + 1.0;
    let b_low = (beta_sup - 1.0).max(0.1);

    // Circulation history for Kutta condition & implicit wake-cut relaxation (Pages 54-57)
    let mut gamma_n: f64 = 0.0;
    let mut gamma_nm1: f64 = 0.0;
    let mut gamma_nm2: f64 = 0.0;

    // Residual history to compare iteration n with (n - M)
    let mut prev_rms_res: f64 = 0.0;
    let mut prev_max_res: f64 = 0.0;
    let mut last_max_res: f64 = 0.0;
    let mut last_rms_res: f64 = 0.0;

    let mut residual = Array2::<f64>::zeros((grid.imax, grid.jmax));

    for iteration in 1..=config.max_iterations {
        if let Some(flag) = cancel_flag {
            if flag.load(Ordering::Relaxed) {
                return Ok(SolverStats {
                    iterations: iteration.saturating_sub(1),
                    converged: false,
                    cancelled: true,
                    final_max_res: last_max_res,
                    final_rms_res: last_rms_res,
                    final_gamma: gamma_n,
                });
            }
        }

        // Fix alpha cycle indexing: iteration 1 uses alphas[0] = alpha_H
        let alpha = alphas[(iteration - 1) % m_cycle];

        // Update circulation Gamma^n via Kutta condition at trailing edge (Pages 55-56)
        if iteration > 1 {
            let ny = grid.jmax - 1;
            // (phi_xi)_{NX, NY} = 0 => tilde_Gamma^n = phi^n_{2, NY} - phi^n_{NX-1, NY}
            let gamma_tilde =
                domain[[1, ny]].flow_ij.phi - domain[[grid.imax - 2, ny]].flow_ij.phi;
            let gamma_bar = config.Rg * gamma_tilde + (1.0 - config.Rg) * gamma_nm1;
            let gamma_est = 2.0 * gamma_nm1 - gamma_nm2;
            gamma_n = 0.5 * (gamma_bar + gamma_est);
        }

        // Step A: Enforce boundary conditions (farfield vortex + wake cut potential jump)
        enforce_boundary_conditions(config, grid, domain, gamma_n);

        // Step B: Update thermodynamics and Holst artificial density across domain
        update_thermodynamics(config, grid, domain, gamma_n);

        // 1. Calculate the Residual field L(phi) for interior and wall (j = 1..grid.jmax)
        let mut max_res = 0.0;
        let mut sum_sq_res = 0.0;
        let mut count = 0.0;
        residual.fill(0.0);

        for i in 1..grid.imax {
            for j in 1..grid.jmax {
                let res = calc_L_phi_ij(grid, domain, i, j);
                residual[[i, j]] = res;

                let abs_res = res.abs();
                if abs_res > max_res {
                    max_res = abs_res;
                }

                sum_sq_res += res * res;
                count += 1.0;
            }
        }

        // Apply periodicity condition at the wake cut
        for j in 1..grid.jmax {
            residual[[0, j]] = residual[[grid.imax - 1, j]];
        }

        let rms_res = if count > 0.0 {
            (sum_sq_res / count).sqrt()
        } else {
            0.0
        };

        if !max_res.is_finite() || !rms_res.is_finite() || max_res > 1e6 {
            return Err(format!(
                "CFD solver diverged at iteration {} (Max Res = {:.3e}). Try reducing omega, increasing beta_mult, or refining the mesh.",
                iteration, max_res
            ));
        }

        last_max_res = max_res;
        last_rms_res = rms_res;

        // Initialize baseline residuals on iteration 1
        if iteration == 1 {
            prev_rms_res = rms_res;
            prev_max_res = max_res;
        }

        // 2. Dynamic Update Logic (Page 50 of notes: every M iterations)
        if iteration > 1
            && iteration % m_cycle == 0
            && prev_rms_res > 1e-15
            && prev_max_res > 1e-15
        {
            let ratio = (rms_res / prev_rms_res) + (max_res / prev_max_res);

            if ratio < 2.0 {
                beta_sup *= 0.98;
            } else if ratio > 2.1 {
                beta_sup *= 1.1;
            }

            beta_sup = beta_sup.clamp(b_low, b_high);

            prev_rms_res = rms_res;
            prev_max_res = max_res;
        }

        if let Some(sender) = tx {
            let send_snap =
                snapshot_interval > 0 && (iteration % snapshot_interval == 0 || iteration == 1);
            let snapshot = if send_snap {
                let mut snap_dom = domain.clone();
                solver_utils::calculate_field(config, grid, &mut snap_dom, gamma_n);
                Some(solver_utils::build_cfd_solution(
                    config,
                    grid,
                    mesher_grid,
                    &snap_dom,
                    gamma_n,
                ))
            } else {
                None
            };

            let _ = sender.send(SolverEvent::Iteration {
                iter: iteration,
                max_iter: config.max_iterations,
                max_res,
                rms_res,
                circulation: gamma_n,
                beta_sup,
                snapshot,
            });
        }

        let is_converged = max_res < config.convergence_residual;
        if is_converged {
            return Ok(SolverStats {
                iterations: iteration,
                converged: true,
                cancelled: false,
                final_max_res: max_res,
                final_rms_res: rms_res,
                final_gamma: gamma_n,
            });
        }

        // 3. Execute the AF2 Scheme
        af2::execute_af2(
            grid, domain, &residual, alpha, beta_sub, beta_sup, omega, gamma_n, gamma_nm1,
        )?;

        // Advance circulation history for next iteration
        gamma_nm2 = gamma_nm1;
        gamma_nm1 = gamma_n;
    }

    Ok(SolverStats {
        iterations: config.max_iterations,
        converged: false,
        cancelled: false,
        final_max_res: last_max_res,
        final_rms_res: last_rms_res,
        final_gamma: gamma_n,
    })
}

pub fn update_thermodynamics(
    config: &Config,
    grid: &Grid,
    domain: &mut Array2<PointData>,
    gamma_n: f64,
) {
    let gamma = config.gamma;
    let c_rho = (gamma - 1.0) / (gamma + 1.0);
    let mach_factor = 2.0 / (gamma + 1.0);

    // Helper for potential at (i + 1, j) on the same branch as (i, j):
    // At i = imax - 1 (NX in 1-based), node NX+1 wraps to i = 1 (2 in 1-based) with jump -Gamma^n.
    let phi_plus = |i_idx: usize, j_idx: usize, dom: &Array2<PointData>| -> f64 {
        if i_idx == grid.imax - 1 {
            dom[[1, j_idx]].flow_ij.phi - gamma_n
        } else {
            dom[[i_idx + 1, j_idx]].flow_ij.phi
        }
    };

    // =========================================================
    // CYCLE 1: Isentropic Properties, Mach, and Contravariant Velocities
    // j = 0 is Farfield, j = grid.jmax - 1 is Airfoil Surface
    // =========================================================
    for i in 1..grid.imax {
        for j in 0..grid.jmax {
            // --- Faces Xi (i+1/2, j) ---
            let phi_xi_ip12j = phi_plus(i, j, domain) - domain[[i, j]].flow_ij.phi;
            let phi_eta_ip12j = if j == grid.jmax - 1 {
                // Wall tangency at (i+1/2, wall)
                -(domain[[i, j]].metrics_ip12j.A2 / domain[[i, j]].metrics_ip12j.A3) * phi_xi_ip12j
            } else if j == 0 {
                // Farfield forward difference
                0.5 * ((domain[[i, 1]].flow_ij.phi - domain[[i, 0]].flow_ij.phi)
                    + (phi_plus(i, 1, domain) - phi_plus(i, 0, domain)))
            } else {
                0.25 * ((domain[[i, j + 1]].flow_ij.phi - domain[[i, j - 1]].flow_ij.phi)
                    + (phi_plus(i, j + 1, domain) - phi_plus(i, j - 1, domain)))
            };

            domain[[i, j]].flow_ip12j.U = domain[[i, j]].metrics_ip12j.A1 * phi_xi_ip12j
                + domain[[i, j]].metrics_ip12j.A2 * phi_eta_ip12j;
            domain[[i, j]].flow_ip12j.V = domain[[i, j]].metrics_ip12j.A2 * phi_xi_ip12j
                + domain[[i, j]].metrics_ip12j.A3 * phi_eta_ip12j;

            let q2_ip12j =
                domain[[i, j]].flow_ip12j.U * phi_xi_ip12j + domain[[i, j]].flow_ip12j.V * phi_eta_ip12j;
            let base_ip12j = 1.0 - c_rho * q2_ip12j;
            domain[[i, j]].flow_ip12j.rho = base_ip12j.max(1e-5).powf(1.0 / (gamma - 1.0));

            // --- Faces Eta (i, j+1/2) for j < grid.jmax - 1 ---
            if j < grid.jmax - 1 {
                let phi_eta_ijp12 = domain[[i, j + 1]].flow_ij.phi - domain[[i, j]].flow_ij.phi;
                let phi_xi_ijp12 = 0.25
                    * ((phi_plus(i, j, domain) - domain[[i - 1, j]].flow_ij.phi)
                        + (phi_plus(i, j + 1, domain) - domain[[i - 1, j + 1]].flow_ij.phi));

                domain[[i, j]].flow_ijp12.U = domain[[i, j]].metrics_ijp12.A1 * phi_xi_ijp12
                    + domain[[i, j]].metrics_ijp12.A2 * phi_eta_ijp12;
                domain[[i, j]].flow_ijp12.V = domain[[i, j]].metrics_ijp12.A2 * phi_xi_ijp12
                    + domain[[i, j]].metrics_ijp12.A3 * phi_eta_ijp12;

                let q2_ijp12 = domain[[i, j]].flow_ijp12.U * phi_xi_ijp12
                    + domain[[i, j]].flow_ijp12.V * phi_eta_ijp12;
                let base_ijp12 = 1.0 - c_rho * q2_ijp12;
                domain[[i, j]].flow_ijp12.rho = base_ijp12.max(1e-5).powf(1.0 / (gamma - 1.0));
            }

            // --- Integer Node (i, j) for Mach and Contravariant Velocity ---
            if j > 0 {
                let phi_xi_ij = 0.5 * (phi_plus(i, j, domain) - domain[[i - 1, j]].flow_ij.phi);
                let phi_eta_ij = if j == grid.jmax - 1 {
                    // Wall tangency: V = 0 => phi_eta = -(A2/A3)*phi_xi
                    -(domain[[i, j]].metrics_ij.A2 / domain[[i, j]].metrics_ij.A3.max(1e-12))
                        * phi_xi_ij
                } else {
                    0.5 * (domain[[i, j + 1]].flow_ij.phi - domain[[i, j - 1]].flow_ij.phi)
                };

                let u_contra = domain[[i, j]].metrics_ij.A1 * phi_xi_ij
                    + domain[[i, j]].metrics_ij.A2 * phi_eta_ij;
                let v_contra = domain[[i, j]].metrics_ij.A2 * phi_xi_ij
                    + domain[[i, j]].metrics_ij.A3 * phi_eta_ij;
                let q2_ij = u_contra * phi_xi_ij + v_contra * phi_eta_ij;

                let base_ij = 1.0 - c_rho * q2_ij;
                domain[[i, j]].flow_ij.rho = base_ij.max(1e-5).powf(1.0 / (gamma - 1.0));
                domain[[i, j]].flow_ij.M = (mach_factor * (q2_ij / base_ij.max(1e-5))).sqrt();

                // Store contravariant velocity for AF2 directional damping
                domain[[i, j]].flow_ij.u = u_contra;
            }
        }
    }

    // Synchronize boundaries before computing artificial density upwind
    enforce_boundary_conditions(config, grid, domain, gamma_n);

    // =========================================================
    // CYCLE 2: Artificial Density (Holst's Upwind Formulation)
    // =========================================================
    let c1 = 0.633939; // Sonic density for gamma = 1.4
    let c2 = 4.9325;
    let c = config.C;

    for i in 1..grid.imax {
        let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };

        for j in 0..grid.jmax {
            // Xi-face transition parameter
            let v_ip12j = if domain[[i, j]].flow_ip12j.U >= 0.0 {
                ((c1 - domain[[i, j]].flow_ij.rho) * c2 * c).clamp(0.0, 1.0)
            } else {
                ((c1 - domain[[i_plus, j]].flow_ij.rho) * c2 * c).clamp(0.0, 1.0)
            };

            // Xi upwind index
            let r: i32 = if domain[[i, j]].flow_ip12j.U >= 0.0 { -1 } else { 1 };
            let mut i_upwind = i as i32 + r;
            if i_upwind == grid.imax as i32 {
                i_upwind = 1;
            } else if i_upwind == 0 {
                i_upwind = (grid.imax - 1) as i32;
            }
            let i_upwind_mapped = i_upwind as usize;

            domain[[i, j]].flow_ip12j.rho_art = (1.0 - v_ip12j) * domain[[i, j]].flow_ip12j.rho
                + v_ip12j * domain[[i_upwind_mapped, j]].flow_ip12j.rho;

            // Eta-face transition parameter (for j < grid.jmax - 1)
            if j < grid.jmax - 1 {
                let v_ijp12 = if domain[[i, j]].flow_ijp12.V >= 0.0 {
                    ((c1 - domain[[i, j]].flow_ij.rho) * c2 * c).clamp(0.0, 1.0)
                } else {
                    ((c1 - domain[[i, j + 1]].flow_ij.rho) * c2 * c).clamp(0.0, 1.0)
                };

                let s: i32 = if domain[[i, j]].flow_ijp12.V >= 0.0 { -1 } else { 1 };
                let j_upwind = if j == 0 && s == -1 {
                    0
                } else {
                    (j as i32 + s).clamp(0, (grid.jmax - 2) as i32) as usize
                };

                domain[[i, j]].flow_ijp12.rho_art = (1.0 - v_ijp12) * domain[[i, j]].flow_ijp12.rho
                    + v_ijp12 * domain[[i, j_upwind]].flow_ijp12.rho;
            }
        }
    }

    // Final boundary sync for the iteration
    enforce_boundary_conditions(config, grid, domain, gamma_n);
}

fn calc_L_phi_ij(grid: &Grid, domain: &Array2<PointData>, i: usize, j: usize) -> f64 {
    // Left Xi face (i - 1/2)
    let flux_xi_minus = domain[[i - 1, j]].flow_ip12j.rho_art * domain[[i - 1, j]].flow_ip12j.U
        / domain[[i - 1, j]].metrics_ip12j.J;

    // Right Xi face (i + 1/2)
    let flux_xi_plus = domain[[i, j]].flow_ip12j.rho_art * domain[[i, j]].flow_ip12j.U
        / domain[[i, j]].metrics_ip12j.J;

    let d_flux_xi = flux_xi_plus - flux_xi_minus;

    let d_flux_eta = if j == grid.jmax - 1 {
        // Airfoil wall condition (Page 52 of lecture notes):
        // (rho V / J)_{i, NY + 1/2} = - (rho V / J)_{i, NY - 1/2}
        let flux_eta_minus = domain[[i, j - 1]].flow_ijp12.rho_art
            * domain[[i, j - 1]].flow_ijp12.V
            / domain[[i, j - 1]].metrics_ijp12.J;
        -2.0 * flux_eta_minus
    } else {
        let flux_eta_minus = domain[[i, j - 1]].flow_ijp12.rho_art
            * domain[[i, j - 1]].flow_ijp12.V
            / domain[[i, j - 1]].metrics_ijp12.J;
        let flux_eta_plus = domain[[i, j]].flow_ijp12.rho_art * domain[[i, j]].flow_ijp12.V
            / domain[[i, j]].metrics_ijp12.J;
        flux_eta_plus - flux_eta_minus
    };

    d_flux_xi + d_flux_eta
}

pub fn enforce_boundary_conditions(
    config: &Config,
    grid: &Grid,
    domain: &mut Array2<PointData>,
    gamma_n: f64,
) {
    let imax = grid.imax;
    let jmax = grid.jmax;

    // 1. Farfield Condition at j = 0 (Page 54 of lecture notes):
    // phi_{i, 1} = phi_inf + phi_vortex
    // where phi_vortex = -(Gamma / 2*pi) * arctan(beta * y / (x - 0.25))
    let beta_prandtl = (1.0 - config.Mach_inf.powi(2)).max(1e-6).sqrt();
    let two_pi = 2.0 * std::f64::consts::PI;

    for i in 0..imax {
        let x = grid.points[[i, 0]].x;
        let y = grid.points[[i, 0]].y;
        let phi_inf = config.u_inf * x + config.v_inf * y;

        // Angle theta in [0, 2*pi] measured counter-clockwise from the wake cut (x > 0.25, y = 0)
        let theta = if i == 0 {
            0.0
        } else if i == imax - 1 {
            two_pi
        } else {
            let mut ang = (beta_prandtl * y).atan2(x - 0.25);
            if ang < 0.0 {
                ang += two_pi;
            }
            ang
        };

        let phi_vortex = -(gamma_n / two_pi) * theta;

        domain[[i, 0]].flow_ij.phi = phi_inf + phi_vortex;
        domain[[i, 0]].flow_ij.rho = config.rho_inf;
        domain[[i, 0]].flow_ij.M = config.Mach_inf;
        domain[[i, 0]].flow_ij.u = config.u_inf;
        domain[[i, 0]].flow_ij.v = config.v_inf;
    }

    // 2. Wake Cut Condition (Pages 54-55 of lecture notes):
    // Physical properties (velocity, density, Mach) are continuous across the wake cut,
    // whereas velocity potential has a branch-cut jump of +Gamma^n:
    // phi^n_{1, j} = phi^n_{NX, j} + Gamma^n
    for j in 0..jmax {
        domain[[0, j]].flow_ij = domain[[imax - 1, j]].flow_ij;
        domain[[0, j]].flow_ij.phi = domain[[imax - 1, j]].flow_ij.phi + gamma_n;
        domain[[0, j]].flow_ip12j = domain[[imax - 1, j]].flow_ip12j;
        if j < jmax - 1 {
            domain[[0, j]].flow_ijp12 = domain[[imax - 1, j]].flow_ijp12;
        }
    }
}
