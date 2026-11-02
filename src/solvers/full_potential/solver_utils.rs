#![allow(non_snake_case)]

use ndarray::Array2;
use crate::core::grid::Point;
use crate::core::solver_trait::{AerodynamicSummary, CfdSolution2D, SolverFamily};
use crate::solvers::full_potential::config::Config;

/// Grid struct holding grid dimensions and the 2D array of physical points
/// oriented in the Full Potential Solver lecture note convention:
/// - `j = 0` at Farfield, `j = jmax - 1` at Airfoil Surface (eta points inward)
/// - `i = 0` at TE, `i` increases counter-clockwise (upper surface -> LE -> lower surface -> TE)
#[derive(Clone, Debug)]
pub struct Grid {
    pub imax: usize,
    pub jmax: usize,
    pub points: Array2<Point>,
}

impl Grid {
    /// Converts a canonical RustyFlow/RustyMesher grid (`j=0` at airfoil wall, `i` clockwise)
    /// into the Solver's O-grid convention (`j=0` at farfield, `i` counter-clockwise).
    pub fn from_mesher_grid(mesher_grid: &Array2<Point>) -> Self {
        let (imax, jmax) = mesher_grid.dim();
        let mut points = Array2::from_elem((imax, jmax), Point { x: 0.0, y: 0.0 });
        for j_raw in 0..jmax {
            let j = jmax - 1 - j_raw;
            for i_raw in 0..imax {
                let i = imax - 1 - i_raw;
                points[[i, j]] = mesher_grid[[i_raw, j_raw]];
            }
        }
        Self { imax, jmax, points }
    }
}

/// Structure for Geometric Metrics
/// Holds the Jacobian and the transformation coefficients (A1, A2, A3)
#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    pub J: f64,
    pub A1: f64,
    pub A2: f64,
    pub A3: f64,
}

/// Physical variables evaluated exactly at the grid node (i, j)
#[derive(Clone, Copy, Debug, Default)]
pub struct NodalFlow {
    pub phi: f64, // Velocity potential (the primary unknown solved by AF2)
    pub M: f64,   // Local Mach number
    pub rho: f64, // Local density
    pub u: f64,   // Cartesian velocity u (or contravariant during iteration)
    pub v: f64,   // Cartesian velocity v
}

/// Physical variables evaluated at the intermediate half-points (i+1/2, j) or (i, j+1/2)
#[derive(Clone, Copy, Debug, Default)]
pub struct HalfPointFlow {
    pub U: f64,       // Contravariant velocity U at half-point
    pub V: f64,       // Contravariant velocity V at half-point
    pub rho: f64,     // Isentropic density at the half-point
    pub rho_art: f64, // Artificial density at the half-point (Holst's Upwinding)
}

/// Master structure grouping data for the node (i,j) and its adjacent forward half-points
#[derive(Clone, Copy, Debug, Default)]
pub struct PointData {
    // Integer Point Data: (i, j)
    pub flow_ij: NodalFlow,
    pub metrics_ij: Metrics,

    // Half-Point Data in Xi direction: (i+1/2, j)
    pub flow_ip12j: HalfPointFlow,
    pub metrics_ip12j: Metrics,

    // Half-Point Data in Eta direction: (i, j+1/2)
    pub flow_ijp12: HalfPointFlow,
    pub metrics_ijp12: Metrics,
}

/// Creates the data grid and initializes the nodal fields (freestream potential with AoA)
pub fn initialize_domain(config: &Config, grid: &Grid) -> Array2<PointData> {
    let mut domain = Array2::<PointData>::default((grid.imax, grid.jmax));

    for i in 0..grid.imax {
        for j in 0..grid.jmax {
            let x = grid.points[[i, j]].x;
            let y = grid.points[[i, j]].y;

            domain[[i, j]].flow_ij = NodalFlow {
                // Initialize velocity potential assuming uniform freestream at angle of attack AoA
                phi: config.u_inf * x + config.v_inf * y,
                M: config.Mach_inf,
                rho: config.rho_inf,
                u: config.u_inf,
                v: config.v_inf,
            };

            domain[[i, j]].flow_ip12j = HalfPointFlow {
                U: config.u_inf,
                V: config.v_inf,
                rho: config.rho_inf,
                rho_art: config.rho_inf,
            };

            domain[[i, j]].flow_ijp12 = HalfPointFlow {
                U: config.u_inf,
                V: config.v_inf,
                rho: config.rho_inf,
                rho_art: config.rho_inf,
            };
        }
    }

    domain
}

/// Calculates transformation metrics for the O-grid:
/// - j = 0 is Farfield
/// - j = jmax - 1 is Airfoil Surface
pub fn calculate_metrics(grid: &Grid, domain: &mut Array2<PointData>) {
    for j in 0..grid.jmax {
        for i in 0..grid.imax {
            // Periodic wrap-around along Xi (i)
            let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
            let i_minus = if i == 0 { grid.imax - 2 } else { i - 1 };

            // 1. Xi direction (Central difference across wake cut)
            let x_xi = (grid.points[[i_plus, j]].x - grid.points[[i_minus, j]].x) / 2.0;
            let y_xi = (grid.points[[i_plus, j]].y - grid.points[[i_minus, j]].y) / 2.0;

            // 2. Eta direction (j = 0 is farfield; j = jmax - 1 is airfoil wall)
            let (x_eta, y_eta) = if j == 0 {
                // Forward difference (into domain from farfield)
                (
                    0.5 * (-3.0 * grid.points[[i, 0]].x
                        + 4.0 * grid.points[[i, 1]].x
                        - grid.points[[i, 2]].x),
                    0.5 * (-3.0 * grid.points[[i, 0]].y
                        + 4.0 * grid.points[[i, 1]].y
                        - grid.points[[i, 2]].y),
                )
            } else if j == grid.jmax - 1 {
                // Backward difference (at airfoil wall towards domain)
                (
                    0.5 * (3.0 * grid.points[[i, j]].x
                        - 4.0 * grid.points[[i, j - 1]].x
                        + grid.points[[i, j - 2]].x),
                    0.5 * (3.0 * grid.points[[i, j]].y
                        - 4.0 * grid.points[[i, j - 1]].y
                        + grid.points[[i, j - 2]].y),
                )
            } else {
                // Central difference in interior
                (
                    (grid.points[[i, j + 1]].x - grid.points[[i, j - 1]].x) / 2.0,
                    (grid.points[[i, j + 1]].y - grid.points[[i, j - 1]].y) / 2.0,
                )
            };

            let J_ij = 1.0 / (x_xi * y_eta - x_eta * y_xi);
            domain[[i, j]].metrics_ij = Metrics {
                J: J_ij,
                A1: (J_ij * y_eta).powi(2) + (-J_ij * x_eta).powi(2),
                A2: (J_ij * y_eta) * (-J_ij * y_xi) + (-J_ij * x_eta) * (J_ij * x_xi),
                A3: (-J_ij * y_xi).powi(2) + (J_ij * x_xi).powi(2),
            };

            // Faces (Half-Points)
            let x_xi_ip12 = grid.points[[i_plus, j]].x - grid.points[[i, j]].x;
            let y_xi_ip12 = grid.points[[i_plus, j]].y - grid.points[[i, j]].y;

            let (x_eta_ip12, y_eta_ip12) = if j == 0 {
                (
                    0.5 * (0.5
                        * (-3.0 * grid.points[[i_plus, 0]].x
                            + 4.0 * grid.points[[i_plus, 1]].x
                            - grid.points[[i_plus, 2]].x)
                        + 0.5
                            * (-3.0 * grid.points[[i, 0]].x
                                + 4.0 * grid.points[[i, 1]].x
                                - grid.points[[i, 2]].x)),
                    0.5 * (0.5
                        * (-3.0 * grid.points[[i_plus, 0]].y
                            + 4.0 * grid.points[[i_plus, 1]].y
                            - grid.points[[i_plus, 2]].y)
                        + 0.5
                            * (-3.0 * grid.points[[i, 0]].y
                                + 4.0 * grid.points[[i, 1]].y
                                - grid.points[[i, 2]].y)),
                )
            } else if j == grid.jmax - 1 {
                (
                    0.5 * (0.5
                        * (3.0 * grid.points[[i_plus, j]].x
                            - 4.0 * grid.points[[i_plus, j - 1]].x
                            + grid.points[[i_plus, j - 2]].x)
                        + 0.5
                            * (3.0 * grid.points[[i, j]].x
                                - 4.0 * grid.points[[i, j - 1]].x
                                + grid.points[[i, j - 2]].x)),
                    0.5 * (0.5
                        * (3.0 * grid.points[[i_plus, j]].y
                            - 4.0 * grid.points[[i_plus, j - 1]].y
                            + grid.points[[i_plus, j - 2]].y)
                        + 0.5
                            * (3.0 * grid.points[[i, j]].y
                                - 4.0 * grid.points[[i, j - 1]].y
                                + grid.points[[i, j - 2]].y)),
                )
            } else {
                (
                    0.5 * ((grid.points[[i_plus, j + 1]].x - grid.points[[i_plus, j - 1]].x) / 2.0
                        + (grid.points[[i, j + 1]].x - grid.points[[i, j - 1]].x) / 2.0),
                    0.5 * ((grid.points[[i_plus, j + 1]].y - grid.points[[i_plus, j - 1]].y) / 2.0
                        + (grid.points[[i, j + 1]].y - grid.points[[i, j - 1]].y) / 2.0),
                )
            };

            let J_ip12 = 1.0 / (x_xi_ip12 * y_eta_ip12 - x_eta_ip12 * y_xi_ip12);
            domain[[i, j]].metrics_ip12j = Metrics {
                J: J_ip12,
                A1: (J_ip12 * y_eta_ip12).powi(2) + (-J_ip12 * x_eta_ip12).powi(2),
                A2: (J_ip12 * y_eta_ip12) * (-J_ip12 * y_xi_ip12)
                    + (-J_ip12 * x_eta_ip12) * (J_ip12 * x_xi_ip12),
                A3: (-J_ip12 * y_xi_ip12).powi(2) + (J_ip12 * x_xi_ip12).powi(2),
            };

            if j < grid.jmax - 1 {
                let x_eta_jp12 = grid.points[[i, j + 1]].x - grid.points[[i, j]].x;
                let y_eta_jp12 = grid.points[[i, j + 1]].y - grid.points[[i, j]].y;

                let x_xi_jp12 = 0.5
                    * ((grid.points[[i_plus, j + 1]].x - grid.points[[i_minus, j + 1]].x) / 2.0
                        + (grid.points[[i_plus, j]].x - grid.points[[i_minus, j]].x) / 2.0);
                let y_xi_jp12 = 0.5
                    * ((grid.points[[i_plus, j + 1]].y - grid.points[[i_minus, j + 1]].y) / 2.0
                        + (grid.points[[i_plus, j]].y - grid.points[[i_minus, j]].y) / 2.0);

                let J_jp12 = 1.0 / (x_xi_jp12 * y_eta_jp12 - x_eta_jp12 * y_xi_jp12);
                domain[[i, j]].metrics_ijp12 = Metrics {
                    J: J_jp12,
                    A1: (J_jp12 * y_eta_jp12).powi(2) + (-J_jp12 * x_eta_jp12).powi(2),
                    A2: (J_jp12 * y_eta_jp12) * (-J_jp12 * y_xi_jp12)
                        + (-J_jp12 * x_eta_jp12) * (J_jp12 * x_xi_jp12),
                    A3: (-J_jp12 * y_xi_jp12).powi(2) + (J_jp12 * x_xi_jp12).powi(2),
                };
            }
        }
    }

    // Exact periodic closure: node 0 receives identical metrics as node imax - 1
    for j in 0..grid.jmax {
        domain[[0, j]].metrics_ij = domain[[grid.imax - 1, j]].metrics_ij;
        domain[[0, j]].metrics_ip12j = domain[[grid.imax - 1, j]].metrics_ip12j;
        if j < grid.jmax - 1 {
            domain[[0, j]].metrics_ijp12 = domain[[grid.imax - 1, j]].metrics_ijp12;
        }
    }
}

/// Reconstructs the true Cartesian velocity field (u, v), density, and Mach number
/// for all grid nodes after AF2 convergence (or for an intermediate snapshot).
pub fn calculate_field(
    config: &Config,
    grid: &Grid,
    domain: &mut Array2<PointData>,
    gamma_n: f64,
) {
    let gamma = config.gamma;
    let m_inf = config.Mach_inf;
    let c_rho = (gamma - 1.0) / (gamma + 1.0);
    let mach_factor = 2.0 / (gamma + 1.0);

    // Ensure wake-cut potential jump is synchronized before taking derivatives
    for j in 0..grid.jmax {
        domain[[0, j]].flow_ij.phi = domain[[grid.imax - 1, j]].flow_ij.phi + gamma_n;
    }

    for j in 1..grid.jmax {
        for i in 1..grid.imax {
            let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
            let i_minus = if i == 1 { grid.imax - 1 } else { i - 1 };

            // Spatial derivatives of coordinates
            let x_xi = (grid.points[[i_plus, j]].x - grid.points[[i_minus, j]].x) / 2.0;
            let y_xi = (grid.points[[i_plus, j]].y - grid.points[[i_minus, j]].y) / 2.0;

            let (x_eta, y_eta) = if j == grid.jmax - 1 {
                // Airfoil surface (backward 2nd order difference)
                (
                    0.5 * (3.0 * grid.points[[i, j]].x
                        - 4.0 * grid.points[[i, j - 1]].x
                        + grid.points[[i, j - 2]].x),
                    0.5 * (3.0 * grid.points[[i, j]].y
                        - 4.0 * grid.points[[i, j - 1]].y
                        + grid.points[[i, j - 2]].y),
                )
            } else {
                // Interior (central difference)
                (
                    (grid.points[[i, j + 1]].x - grid.points[[i, j - 1]].x) / 2.0,
                    (grid.points[[i, j + 1]].y - grid.points[[i, j - 1]].y) / 2.0,
                )
            };

            let jacobian = 1.0 / (x_xi * y_eta - x_eta * y_xi);

            let xi_x = jacobian * y_eta;
            let xi_y = -jacobian * x_eta;
            let eta_x = -jacobian * y_xi;
            let eta_y = jacobian * x_xi;

            // Potential derivatives across branch cut:
            // Left neighbor (i - 1): for i = 1, domain[[0, j]].flow_ij.phi is already phi_{NX, j} + Gamma
            // Right neighbor (i + 1): for i = imax - 1 (NX), ghost node NX+1 has phi_{2, j} - Gamma
            let phi_plus = if i == grid.imax - 1 {
                domain[[1, j]].flow_ij.phi - gamma_n
            } else {
                domain[[i + 1, j]].flow_ij.phi
            };
            let phi_xi = (phi_plus - domain[[i - 1, j]].flow_ij.phi) / 2.0;

            let phi_eta = if j == grid.jmax - 1 {
                // Wall flow tangency: V = A2*phi_xi + A3*phi_eta = 0 => phi_eta = -(A2/A3)*phi_xi
                let a2 = domain[[i, j]].metrics_ij.A2;
                let a3 = domain[[i, j]].metrics_ij.A3;
                if a3.abs() > 1e-12 {
                    -(a2 / a3) * phi_xi
                } else {
                    0.5 * (3.0 * domain[[i, j]].flow_ij.phi
                        - 4.0 * domain[[i, j - 1]].flow_ij.phi
                        + domain[[i, j - 2]].flow_ij.phi)
                }
            } else {
                (domain[[i, j + 1]].flow_ij.phi - domain[[i, j - 1]].flow_ij.phi) / 2.0
            };

            // True Cartesian velocities (u, v)
            let u = xi_x * phi_xi + eta_x * phi_eta;
            let v = xi_y * phi_xi + eta_y * phi_eta;
            let q2 = u * u + v * v;

            // Isentropic thermodynamics
            let a_squared = (1.0 - c_rho * q2).max(1e-5);
            let rho = a_squared.powf(1.0 / (gamma - 1.0));
            let mach = (mach_factor * (q2 / a_squared)).sqrt();

            domain[[i, j]].flow_ij.u = u;
            domain[[i, j]].flow_ij.v = v;
            domain[[i, j]].flow_ij.rho = rho;
            domain[[i, j]].flow_ij.M = mach;
        }
    }

    // At the sharp trailing-edge wall node (imax - 1, jmax - 1), central coordinate
    // differencing across the O-grid cusp is singular (x_1 == x_{NX-1} => x_xi = 0).
    // Reconstruct the TE wall state as the average of the adjacent upper and lower surface nodes.
    let j_wall = grid.jmax - 1;
    if grid.imax > 3 {
        let u_te = 0.5 * (domain[[1, j_wall]].flow_ij.u + domain[[grid.imax - 2, j_wall]].flow_ij.u);
        let v_te = 0.5 * (domain[[1, j_wall]].flow_ij.v + domain[[grid.imax - 2, j_wall]].flow_ij.v);
        let q2_te = u_te * u_te + v_te * v_te;
        let a2_te = (1.0 - c_rho * q2_te).max(1e-5);
        let rho_te = a2_te.powf(1.0 / (gamma - 1.0));
        let mach_te = (mach_factor * (q2_te / a2_te)).sqrt();
        domain[[grid.imax - 1, j_wall]].flow_ij.u = u_te;
        domain[[grid.imax - 1, j_wall]].flow_ij.v = v_te;
        domain[[grid.imax - 1, j_wall]].flow_ij.rho = rho_te;
        domain[[grid.imax - 1, j_wall]].flow_ij.M = mach_te;
    }

    // Farfield boundary at j = 0
    for i in 0..grid.imax {
        domain[[i, 0]].flow_ij.u = config.u_inf;
        domain[[i, 0]].flow_ij.v = config.v_inf;
        domain[[i, 0]].flow_ij.rho = config.rho_inf;
        domain[[i, 0]].flow_ij.M = m_inf;
    }

    // Sync wake cut (continuous velocity/density/Mach, discontinuous potential by +Gamma)
    for j in 0..grid.jmax {
        domain[[0, j]].flow_ij = domain[[grid.imax - 1, j]].flow_ij;
        domain[[0, j]].flow_ij.phi = domain[[grid.imax - 1, j]].flow_ij.phi + gamma_n;
    }
}

pub fn alpha_seq(config: &Config, grid: &Grid) -> Vec<f64> {
    let min_spacing = compute_min_spacing(grid);

    let alpha_h = 1.0 / min_spacing;
    let alpha_l = 1.0;

    let m = config.alpha_sequence_length;
    let mut alpha_k = Vec::with_capacity(m);

    if m <= 1 {
        alpha_k.push(alpha_h);
        return alpha_k;
    }

    // Geometric progression: ratio = (alpha_L / alpha_H)^(1 / (M - 1))
    let ratio = (alpha_l / alpha_h).powf(1.0 / (m as f64 - 1.0));

    // Alpha sequence from alpha_H down to alpha_L
    for k in 0..m {
        let alpha = alpha_h * ratio.powi(k as i32);
        alpha_k.push(alpha);
    }

    alpha_k
}

pub fn compute_min_spacing(grid: &Grid) -> f64 {
    let mut min_spacing = f64::MAX;

    // Airfoil surface is at j = jmax - 1
    let j_wall = grid.jmax - 1;

    for i in 0..grid.imax {
        let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };

        let dx_xi = grid.points[[i_plus, j_wall]].x - grid.points[[i, j_wall]].x;
        let dy_xi = grid.points[[i_plus, j_wall]].y - grid.points[[i, j_wall]].y;
        let ds_xi = (dx_xi.powi(2) + dy_xi.powi(2)).sqrt();

        if ds_xi > 1e-14 && ds_xi < min_spacing {
            min_spacing = ds_xi;
        }

        // Distance from wall (j_wall) to adjacent interior node (j_wall - 1)
        let dx_eta = grid.points[[i, j_wall]].x - grid.points[[i, j_wall - 1]].x;
        let dy_eta = grid.points[[i, j_wall]].y - grid.points[[i, j_wall - 1]].y;
        let ds_eta = (dx_eta.powi(2) + dy_eta.powi(2)).sqrt();

        if ds_eta > 1e-14 && ds_eta < min_spacing {
            min_spacing = ds_eta;
        }
    }

    min_spacing.max(1e-8)
}

/// Converts the internal Solver `domain` into the solver-agnostic `CfdSolution2D`
/// aligned with RustyFlow's canonical `(i_mesher, j_mesher)` indexing, and computes
/// all integrated aerodynamic coefficients (`Cl`, `Cd`, `Cm_c4`, `Cn`, `Ca`, `Cp*`, etc.).
pub fn build_cfd_solution(
    config: &Config,
    solver_grid: &Grid,
    mesher_grid: &Array2<Point>,
    domain: &Array2<PointData>,
    final_gamma: f64,
) -> CfdSolution2D {
    let imax = solver_grid.imax;
    let jmax = solver_grid.jmax;

    let gamma = config.gamma;
    let m_inf = config.Mach_inf;
    let rho_inf = config.rho_inf;
    let cp_multiplier = 2.0 / (gamma * m_inf.powi(2));

    let mut mach = Array2::<f64>::zeros((imax, jmax));
    let mut cp = Array2::<f64>::zeros((imax, jmax));
    let mut density = Array2::<f64>::zeros((imax, jmax));
    let mut velocity_u = Array2::<f64>::zeros((imax, jmax));
    let mut velocity_v = Array2::<f64>::zeros((imax, jmax));
    let mut potential = Array2::<f64>::zeros((imax, jmax));

    let mut max_mach = 0.0_f64;
    let mut min_mach = f64::MAX;

    for j_raw in 0..jmax {
        let j_sol = jmax - 1 - j_raw;
        for i_raw in 0..imax {
            let i_sol = imax - 1 - i_raw;
            let flow = &domain[[i_sol, j_sol]].flow_ij;
            let cp_val = cp_multiplier * ((flow.rho / rho_inf).powf(gamma) - 1.0);

            mach[[i_raw, j_raw]] = flow.M;
            cp[[i_raw, j_raw]] = cp_val;
            density[[i_raw, j_raw]] = flow.rho;
            velocity_u[[i_raw, j_raw]] = flow.u;
            velocity_v[[i_raw, j_raw]] = flow.v;
            potential[[i_raw, j_raw]] = flow.phi;

            if flow.M > max_mach {
                max_mach = flow.M;
            }
            if flow.M < min_mach {
                min_mach = flow.M;
            }
        }
    }

    // Compute aerodynamic coefficients along the wall in solver convention (counter-clockwise, j_sol = jmax - 1)
    let j_wall = jmax - 1;
    let mut cp_min = f64::MAX;
    let mut x_cp_min = 0.0_f64;
    let mut cp_max = -f64::MAX;
    let mut x_cp_max = 0.0_f64;

    let mut cn = 0.0_f64;
    let mut ca = 0.0_f64;
    let mut cm_c4 = 0.0_f64;

    let mut x_le = f64::MAX;
    let mut x_te = f64::MIN;
    let mut idx_le = 0usize;

    for i in 0..(imax - 1) {
        let x = solver_grid.points[[i, j_wall]].x;
        if x < x_le {
            x_le = x;
            idx_le = i;
        }
        if x > x_te {
            x_te = x;
        }
    }
    let chord = (x_te - x_le).max(1e-12);
    let x_ref = x_le + 0.25 * chord;

    for i in 0..(imax - 1) {
        let x0 = solver_grid.points[[i, j_wall]].x;
        let y0 = solver_grid.points[[i, j_wall]].y;
        let rho0 = domain[[i, j_wall]].flow_ij.rho;
        let cp0 = cp_multiplier * ((rho0 / rho_inf).powf(gamma) - 1.0);

        let x1 = solver_grid.points[[i + 1, j_wall]].x;
        let y1 = solver_grid.points[[i + 1, j_wall]].y;
        let rho1 = domain[[i + 1, j_wall]].flow_ij.rho;
        let cp1 = cp_multiplier * ((rho1 / rho_inf).powf(gamma) - 1.0);

        if cp0 < cp_min {
            cp_min = cp0;
            x_cp_min = (x0 - x_le) / chord;
        }
        if cp0 > cp_max {
            cp_max = cp0;
            x_cp_max = (x0 - x_le) / chord;
        }

        let cp_avg = 0.5 * (cp0 + cp1);
        let xm = 0.5 * (x0 + x1);
        let ym = 0.5 * (y0 + y1);
        let dx = x1 - x0;
        let dy = y1 - y0;

        cn += cp_avg * dx / chord;
        ca -= cp_avg * dy / chord;
        cm_c4 += (cp_avg * (x_ref - xm) * dx + cp_avg * ym * dy) / (chord * chord);
    }

    let cl = cn * config.alpha_rad.cos() - ca * config.alpha_rad.sin();
    let cd = cn * config.alpha_rad.sin() + ca * config.alpha_rad.cos();
    let cl_kutta_joukowski = if config.q_inf > 1e-12 {
        2.0 * final_gamma / config.q_inf
    } else {
        0.0
    };

    // Exact theoretical isentropic stagnation Cp
    let p_ratio_stag = (1.0 + 0.5 * (gamma - 1.0) * m_inf.powi(2)).powf(gamma / (gamma - 1.0));
    let cp_stag_theory = (2.0 / (gamma * m_inf.powi(2))) * (p_ratio_stag - 1.0);

    // Sonic pressure coefficient Cp* (M = 1)
    let cp_star = if m_inf > 0.05 {
        let ratio = (2.0 + (gamma - 1.0) * m_inf.powi(2)) / (gamma + 1.0);
        Some((2.0 / (gamma * m_inf.powi(2))) * (ratio.powf(gamma / (gamma - 1.0)) - 1.0))
    } else {
        None
    };

    // Split wall distribution into Upper Surface (i = 0..=idx_le) and Lower Surface (i = idx_le..imax)
    let mut upper_cp = Vec::new();
    let mut lower_cp = Vec::new();
    let mut upper_mach = Vec::new();
    let mut lower_mach = Vec::new();

    for i in (0..=idx_le).rev() {
        let x = solver_grid.points[[i, j_wall]].x;
        let rho = domain[[i, j_wall]].flow_ij.rho;
        let m = domain[[i, j_wall]].flow_ij.M;
        let cp_val = cp_multiplier * ((rho / rho_inf).powf(gamma) - 1.0);
        upper_cp.push([x, cp_val]);
        upper_mach.push([x, m]);
    }

    for i in idx_le..imax {
        let x = solver_grid.points[[i, j_wall]].x;
        let rho = domain[[i, j_wall]].flow_ij.rho;
        let m = domain[[i, j_wall]].flow_ij.M;
        let cp_val = cp_multiplier * ((rho / rho_inf).powf(gamma) - 1.0);
        lower_cp.push([x, cp_val]);
        lower_mach.push([x, m]);
    }

    let aero = AerodynamicSummary {
        mach_inf: m_inf,
        aoa_deg: config.AoA,
        gamma_heat: gamma,
        circulation: final_gamma,
        cl,
        cl_kutta_joukowski,
        cd,
        cm_c4,
        cn,
        ca,
        cp_min,
        x_cp_min,
        cp_max,
        x_cp_max,
        cp_stag_theory,
        cp_star,
        min_mach: if min_mach == f64::MAX { 0.0 } else { min_mach },
        max_mach,
        upper_cp,
        lower_cp,
        upper_mach,
        lower_mach,
    };

    CfdSolution2D {
        solver_family: SolverFamily::FullPotential,
        grid: mesher_grid.clone(),
        mach,
        cp,
        density,
        velocity_u,
        velocity_v,
        potential,
        aero,
    }
}
