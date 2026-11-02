use ndarray::Array2;
use crate::solvers::full_potential::solver_utils::{Grid, PointData};

/// Main orchestrator for the AF2 numerical scheme
#[allow(clippy::too_many_arguments)]
pub fn execute_af2(
    grid: &Grid,
    domain: &mut Array2<PointData>,
    residual: &Array2<f64>,
    alpha: f64,
    beta_sub: f64,
    beta_sup: f64,
    omega: f64,
    gamma_n: f64,
    gamma_nm1: f64,
) -> Result<(), String> {
    let mut f_inter = Array2::<f64>::zeros((grid.imax, grid.jmax));
    let mut delta_phi = Array2::<f64>::zeros((grid.imax, grid.jmax));

    // =========================================================
    // Memory pre-allocation for Step 2 (Periodic Tridiagonal)
    // N is the number of unique nodes in the O-grid ring
    // =========================================================
    let n = grid.imax - 1;
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    // Auxiliary buffers for Sherman-Morrison
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];
    let mut d_z = vec![0.0; n];

    // =========================================================
    // STEP 1: Radial Sweep (Eta direction)
    // Solves the Bidiagonal system from Wall to Farfield (decreasing j)
    // =========================================================
    for i in 1..grid.imax {
        step1_bidiagonal_eta(i, alpha, omega, grid, domain, residual, &mut f_inter);
    }

    // =========================================================
    // STEP 2: Azimuthal Sweep (Xi direction)
    // Solves the Periodic Tridiagonal system ring by ring
    // j = 0 is Dirichlet farfield, so j goes from 1 to grid.jmax - 1 (wall)
    // =========================================================
    for j in 1..grid.jmax {
        step2_tridiagonal_xi(
            j,
            alpha,
            beta_sub,
            beta_sup,
            gamma_n,
            gamma_nm1,
            grid,
            domain,
            &f_inter,
            &mut delta_phi,
            &mut a,
            &mut b,
            &mut c,
            &mut d,
            &mut y,
            &mut z,
            &mut c_prime,
            &mut d_prime,
            &mut d_z,
        )?;
    }

    // =========================================================
    // Final Update of the Velocity Potential
    // =========================================================
    for i in 1..grid.imax {
        for j in 1..grid.jmax {
            domain[[i, j]].flow_ij.phi += delta_phi[[i, j]];
        }
    }

    // Synchronize the wake cut potential jump with current circulation
    for j in 0..grid.jmax {
        domain[[0, j]].flow_ij.phi = domain[[grid.imax - 1, j]].flow_ij.phi + gamma_n;
    }

    Ok(())
}

/// Solves the first step of AF2. 
/// Employs a forward difference operator in Eta, so the sweep marches backward (decreasing j)
/// from the wall (j = ny = jmax - 1) towards the farfield (j = 0).
fn step1_bidiagonal_eta(
    i: usize,
    alpha: f64,
    omega: f64,
    grid: &Grid,
    domain: &Array2<PointData>,
    residual: &Array2<f64>,
    f_inter: &mut Array2<f64>,
) {
    let ny = grid.jmax - 1; // Wall is at ny = jmax - 1

    // At the wall boundary j = ny:
    // f_{i, ny+1} = 0 (Page 53 of lecture notes)
    // (alpha + a_wall) * f_{i, ny} = alpha * omega * residual_{i, ny}
    let a_wall = domain[[i, ny - 1]].flow_ijp12.rho_art
        * domain[[i, ny - 1]].metrics_ijp12.A3
        / domain[[i, ny - 1]].metrics_ijp12.J;

    f_inter[[i, ny]] = (alpha * omega * residual[[i, ny]]) / (alpha + a_wall);

    // Reverse sweep (from wall towards farfield: j from ny - 1 down to 1)
    for j in (1..ny).rev() {
        let a_plus = domain[[i, j]].flow_ijp12.rho_art
            * domain[[i, j]].metrics_ijp12.A3
            / domain[[i, j]].metrics_ijp12.J;

        let rhs = alpha * omega * residual[[i, j]];

        // Direct bidiagonal substitution
        f_inter[[i, j]] = (rhs + a_plus * f_inter[[i, j + 1]]) / (alpha + a_plus);
    }

    // Dirichlet boundary condition at Farfield for intermediate function
    f_inter[[i, 0]] = 0.0;
}

/// Solves the second step of AF2 incorporating temporal damping (Beta)
/// and implicit circulation wake-cut conditions (Pages 56-57).
/// Uses Sherman-Morrison to enforce cyclic coupling across the wake cut.
#[allow(clippy::too_many_arguments)]
fn step2_tridiagonal_xi(
    j: usize,
    alpha: f64,
    beta_sub: f64,
    beta_sup: f64,
    gamma_n: f64,
    gamma_nm1: f64,
    grid: &Grid,
    domain: &Array2<PointData>,
    f_inter: &Array2<f64>,
    delta_phi: &mut Array2<f64>,
    a: &mut [f64],
    b: &mut [f64],
    c: &mut [f64],
    d: &mut [f64],
    y: &mut [f64],
    z: &mut [f64],
    c_prime: &mut [f64],
    d_prime: &mut [f64],
    d_z: &mut [f64],
) -> Result<(), String> {
    let n = grid.imax - 1;

    // 1. Build the matrix coefficients for the ring 'j'
    for idx in 0..n {
        let i = idx + 1;

        // Topological mapping for O-Grid periodicity
        let i_minus = if i == 1 { grid.imax - 1 } else { i - 1 };

        // Spatial central difference operators based on A1 metric (Xi direction)
        let b_plus = domain[[i, j]].flow_ip12j.rho_art * domain[[i, j]].metrics_ip12j.A1
            / domain[[i, j]].metrics_ip12j.J;
        let b_minus = domain[[i_minus, j]].flow_ip12j.rho_art
            * domain[[i_minus, j]].metrics_ip12j.A1
            / domain[[i_minus, j]].metrics_ip12j.J;

        a[idx] = -b_minus;
        b[idx] = alpha + b_plus + b_minus;
        c[idx] = -b_plus;

        // RHS: intermediate function + delayed eta contribution: alpha * Delta_phi_{i, j-1}
        d[idx] = f_inter[[i, j]];
        if j > 1 {
            d[idx] += alpha * delta_phi[[i, j - 1]];
        }

        // Temporal damping injection (Page 48 of lecture notes: +/- alpha * beta * delta_xi)
        let mach_local = domain[[i, j]].flow_ij.M;
        let u_contra = domain[[i, j]].flow_ij.u;

        let beta_local = if mach_local > 1.0 { beta_sup } else { beta_sub };
        let stabilizer = alpha * beta_local;

        // Upwind discretization based on contravariant velocity direction
        if u_contra >= 0.0 {
            // Forward flow: backward difference -> + alpha * beta * (C_i - C_{i-1})
            b[idx] += stabilizer; // Reinforce main diagonal (node i)
            a[idx] -= stabilizer; // Subtract from upstream neighbor (node i-1)
        } else {
            // Reverse flow: forward difference -> - alpha * beta * (C_{i+1} - C_i)
            b[idx] += stabilizer; // Reinforce main diagonal (node i)
            c[idx] -= stabilizer; // Subtract from downstream neighbor (node i+1)
        }
    }

    // Implicit wake-cut boundary condition for circulation (Pages 56-57 of lecture notes):
    // At i = 2 (idx = 0): C^n_{1, j} = C^n_{NX, j} + (Gamma^n - Gamma^{n-1})
    // At i = NX (idx = n - 1): C^n_{NX+1, j} = C^n_{2, j} - (Gamma^n - Gamma^{n-1})
    let d_gamma = gamma_n - gamma_nm1;
    d[0] -= a[0] * d_gamma;
    d[n - 1] += c[n - 1] * d_gamma;

    // 2. Sherman-Morrison Modification for Cyclic Tridiagonal System
    let gamma = -b[0];

    b[0] -= gamma;
    b[n - 1] -= c[n - 1] * a[0] / gamma;

    d_z.fill(0.0);
    d_z[0] = gamma;
    d_z[n - 1] = c[n - 1];

    // 3. Solve auxiliary systems in-place (Zero heap allocation)
    solve_standard_tdma_inplace(a, b, c, d, y, c_prime, d_prime)?;
    solve_standard_tdma_inplace(a, b, c, d_z, z, c_prime, d_prime)?;

    // 4. Recombine using Sherman-Morrison formula
    let fact = (y[0] + a[0] * y[n - 1] / gamma) / (1.0 + z[0] + a[0] * z[n - 1] / gamma);

    for idx in 0..n {
        let i = idx + 1;
        delta_phi[[i, j]] = y[idx] - fact * z[idx];
    }

    // 5. Wake Cut delta_phi at i = 0
    delta_phi[[0, j]] = delta_phi[[grid.imax - 1, j]] + d_gamma;

    Ok(())
}

/// Helper: Standard Thomas Algorithm.
/// Uses pre-allocated slices to avoid dynamic memory allocation during sweeps.
fn solve_standard_tdma_inplace(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    x: &mut [f64],
    c_prime: &mut [f64],
    d_prime: &mut [f64],
) -> Result<(), String> {
    let n = b.len();

    // Forward Sweep
    if b[0].abs() < 1e-14 || !b[0].is_finite() {
        return Err("Ill-conditioned tridiagonal matrix at index 0.".to_string());
    }
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < 1e-14 || !denom.is_finite() {
            return Err(format!(
                "Ill-conditioned matrix encountered at index {}. Check alpha sequence or flow parameters.",
                i
            ));
        }
        c_prime[i] = c[i] / denom;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Backward Substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(())
}
