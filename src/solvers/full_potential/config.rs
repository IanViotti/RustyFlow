#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

fn default_rg() -> f64 {
    0.5
}

/// Main configuration structure that holds all parameters required for the 
/// Full Potential CFD solver process. Compatible with RustPotential's `job_config.toml`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    // Basic identifiers
    pub meshname: String,

    pub jobname: String,

    pub export_metrics: bool,

    pub Mach_inf: f64,

    #[serde(default, alias = "alpha_deg", alias = "alpha", alias = "aoa")]
    pub AoA: f64,

    #[serde(default = "default_rg", alias = "r_g", alias = "rg")]
    pub Rg: f64,

    pub max_iterations: usize,

    pub convergence_residual: f64,

    pub gamma: f64,

    pub omega: f64,

    pub beta_mult: f64,

    pub alpha_sequence_length: usize,

    pub C: f64,

    #[serde(skip)]
    pub output_dir: String,

    #[serde(skip)]
    pub alpha_rad: f64,

    #[serde(skip)]
    pub q_inf: f64,

    #[serde(skip)]
    pub u_inf: f64,

    #[serde(skip)]
    pub v_inf: f64,

    #[serde(skip)]
    pub rho_inf: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self::preset_transonic_lifting()
    }
}

impl Config {
    /// Populates derived physical variables after reading or modifying the configuration.
    pub fn compute_derived_variables(&mut self) {
        self.Mach_inf = self.Mach_inf.clamp(0.01, 1.35);
        self.gamma = self.gamma.clamp(1.05, 1.67);
        self.Rg = self.Rg.clamp(0.0, 1.0);
        self.alpha_sequence_length = self.alpha_sequence_length.max(1);

        self.alpha_rad = self.AoA.to_radians();

        // Non-dimensional freestream speed magnitude (normalized by critical sound speed a*)
        self.q_inf = ((self.gamma + 1.0) * self.Mach_inf.powi(2)
            / (2.0 + (self.gamma - 1.0) * self.Mach_inf.powi(2)))
            .sqrt();

        // Cartesian freestream velocity components
        self.u_inf = self.q_inf * self.alpha_rad.cos();
        self.v_inf = self.q_inf * self.alpha_rad.sin();

        // Non-dimensional freestream density (normalized by stagnation density rho_t)
        self.rho_inf = (1.0 - (self.gamma - 1.0) / (self.gamma + 1.0) * self.q_inf.powi(2))
            .max(1e-6)
            .powf(1.0 / (self.gamma - 1.0));
    }

    pub fn preset_subsonic_symmetric() -> Self {
        let mut cfg = Self {
            meshname: "mesh/naca0012_mesh.vtk".to_string(),
            jobname: "naca0012_subsonic".to_string(),
            export_metrics: true,
            Mach_inf: 0.30,
            AoA: 0.0,
            Rg: 0.5,
            max_iterations: 1000,
            convergence_residual: 1e-5,
            gamma: 1.4,
            omega: 1.0,
            beta_mult: 2.5,
            alpha_sequence_length: 8,
            C: 1.5,
            output_dir: String::new(),
            alpha_rad: 0.0,
            q_inf: 0.0,
            u_inf: 0.0,
            v_inf: 0.0,
            rho_inf: 1.0,
        };
        cfg.compute_derived_variables();
        cfg
    }

    pub fn preset_transonic_lifting() -> Self {
        let mut cfg = Self {
            meshname: "mesh/naca0012_fine_mesh.vtk".to_string(),
            jobname: "naca0012_AoA1_M08".to_string(),
            export_metrics: true,
            Mach_inf: 0.80,
            AoA: 1.0,
            Rg: 0.25,
            max_iterations: 2000,
            convergence_residual: 1e-5,
            gamma: 1.4,
            omega: 1.0,
            beta_mult: 2.5,
            alpha_sequence_length: 8,
            C: 1.5,
            output_dir: String::new(),
            alpha_rad: 0.0,
            q_inf: 0.0,
            u_inf: 0.0,
            v_inf: 0.0,
            rho_inf: 1.0,
        };
        cfg.compute_derived_variables();
        cfg
    }

    pub fn preset_transonic_shock_zero_aoa() -> Self {
        let mut cfg = Self {
            meshname: "mesh/naca0012_fine_mesh.vtk".to_string(),
            jobname: "naca0012_M085".to_string(),
            export_metrics: true,
            Mach_inf: 0.85,
            AoA: 0.0,
            Rg: 0.5,
            max_iterations: 2000,
            convergence_residual: 1e-5,
            gamma: 1.4,
            omega: 1.0,
            beta_mult: 2.5,
            alpha_sequence_length: 8,
            C: 1.5,
            output_dir: String::new(),
            alpha_rad: 0.0,
            q_inf: 0.0,
            u_inf: 0.0,
            v_inf: 0.0,
            rho_inf: 1.0,
        };
        cfg.compute_derived_variables();
        cfg
    }

    pub fn preset_subsonic_high_lift() -> Self {
        let mut cfg = Self {
            meshname: "mesh/naca0012_mesh.vtk".to_string(),
            jobname: "naca0012_AoA4_M05".to_string(),
            export_metrics: true,
            Mach_inf: 0.50,
            AoA: 4.0,
            Rg: 0.35,
            max_iterations: 1500,
            convergence_residual: 1e-5,
            gamma: 1.4,
            omega: 1.0,
            beta_mult: 2.5,
            alpha_sequence_length: 8,
            C: 1.5,
            output_dir: String::new(),
            alpha_rad: 0.0,
            q_inf: 0.0,
            u_inf: 0.0,
            v_inf: 0.0,
            rho_inf: 1.0,
        };
        cfg.compute_derived_variables();
        cfg
    }
}
