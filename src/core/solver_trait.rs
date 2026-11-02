use ndarray::Array2;
use serde::{Deserialize, Serialize};
use crate::core::grid::Point;

/// Identifies the CFD solver family in RustyFlow.
/// Designed to scale cleanly from Full Potential to Euler and RANS solvers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverFamily {
    FullPotential,
    Euler2D,
    Rans2D,
}

impl SolverFamily {
    pub fn label(&self) -> &'static str {
        match self {
            Self::FullPotential => "Full Potential (AF2 + Holst Upwind + Circulation)",
            Self::Euler2D => "Euler 2D (Compressible Inviscid - Planned)",
            Self::Rans2D => "RANS 2D (Viscous Turbulent - Planned)",
        }
    }

    pub fn is_implemented(&self) -> bool {
        match self {
            Self::FullPotential => true,
            Self::Euler2D => false,
            Self::Rans2D => false,
        }
    }
}

/// Scalar fields produced by a CFD solver that can be rendered in the 2D viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowScalarField {
    Mach,
    Cp,
    Density,
    Potential,
    VelocityMag,
    VelocityU,
    VelocityV,
}

impl FlowScalarField {
    pub const ALL: [FlowScalarField; 7] = [
        Self::Mach,
        Self::Cp,
        Self::Density,
        Self::Potential,
        Self::VelocityMag,
        Self::VelocityU,
        Self::VelocityV,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Mach => "Mach Number (M)",
            Self::Cp => "Pressure Coefficient (Cp)",
            Self::Density => "Density (ρ / ρ_t)",
            Self::Potential => "Velocity Potential (φ)",
            Self::VelocityMag => "Velocity Magnitude (|V| / a*)",
            Self::VelocityU => "Velocity u (Cartesian)",
            Self::VelocityV => "Velocity v (Cartesian)",
        }
    }
}

/// Aerodynamic force, moment, and extrema summary computed from the wall pressure distribution.
#[derive(Clone, Debug, Default)]
pub struct AerodynamicSummary {
    pub mach_inf: f64,
    pub aoa_deg: f64,
    pub gamma_heat: f64,
    pub circulation: f64,
    pub cl: f64,
    pub cl_kutta_joukowski: f64,
    pub cd: f64,
    pub cm_c4: f64,
    pub cn: f64,
    pub ca: f64,
    pub cp_min: f64,
    pub x_cp_min: f64,
    pub cp_max: f64,
    pub x_cp_max: f64,
    pub cp_stag_theory: f64,
    pub cp_star: Option<f64>,
    pub min_mach: f64,
    pub max_mach: f64,
    /// Upper surface (extradorso) sorted by x/c: Vec<[x, cp]>
    pub upper_cp: Vec<[f64; 2]>,
    /// Lower surface (intradorso) sorted by x/c: Vec<[x, cp]>
    pub lower_cp: Vec<[f64; 2]>,
    /// Upper surface Mach distribution sorted by x/c: Vec<[x, M]>
    pub upper_mach: Vec<[f64; 2]>,
    /// Lower surface Mach distribution sorted by x/c: Vec<[x, M]>
    pub lower_mach: Vec<[f64; 2]>,
}

/// Generic flow solution container produced by any 2D structured CFD solver.
/// All 2D arrays are stored in RustyFlow's canonical grid indexing `(i_mesher, j_mesher)`
/// where `j = 0` is the airfoil surface and `j = nj - 1` is the farfield boundary.
#[derive(Clone, Debug)]
pub struct CfdSolution2D {
    pub solver_family: SolverFamily,
    pub grid: Array2<Point>,
    pub mach: Array2<f64>,
    pub cp: Array2<f64>,
    pub density: Array2<f64>,
    pub velocity_u: Array2<f64>,
    pub velocity_v: Array2<f64>,
    pub potential: Array2<f64>,
    pub aero: AerodynamicSummary,
}

impl CfdSolution2D {
    pub fn sample_field(&self, field: FlowScalarField, i: usize, j: usize) -> f64 {
        match field {
            FlowScalarField::Mach => self.mach[[i, j]],
            FlowScalarField::Cp => self.cp[[i, j]],
            FlowScalarField::Density => self.density[[i, j]],
            FlowScalarField::Potential => self.potential[[i, j]],
            FlowScalarField::VelocityMag => {
                let u = self.velocity_u[[i, j]];
                let v = self.velocity_v[[i, j]];
                (u * u + v * v).sqrt()
            }
            FlowScalarField::VelocityU => self.velocity_u[[i, j]],
            FlowScalarField::VelocityV => self.velocity_v[[i, j]],
        }
    }
}
