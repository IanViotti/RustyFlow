#![allow(non_snake_case)]

use serde::{Deserialize, Serialize};

/// Main configuration structure that holds all parameters required for the 
/// grid generation process. This struct is populated by reading the `job_config.toml` file
/// or adjusted interactively through the RustyFlow GUI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    // Basic identifiers
    /// Name of the mesh/job, used for creating output directories and files.
    pub meshname: String,
    
    /// The physical shape of the inner boundary (e.g., Biconvex or NACA).
    pub airfoil_type: AirfoilType,  
    
    /// Outer boundary radius or distance. Defines how far the computational 
    /// domain extends from the airfoil surface.
    pub r_max: f64,
    
    // Grid topology parameters
    /// Number of points in the longitudinal direction (xi / ξ coordinate).
    /// Typically represents the wrap-around direction in an O-grid.
    pub longitudinal_points: usize,
    
    /// Number of points in the normal direction (eta / η coordinate).
    /// Represents the radial lines extending from the airfoil to the outer boundary.
    pub normal_points: usize,
    
    /// Geometric stretching factor used to cluster grid points closer to the 
    /// Leading Edge (LE) and Trailing Edge (TE) along the airfoil surface.
    pub stretching_factor: f64, 
    
    // Geometry definition parameters
    /// Geometric progression factor used specifically in the parabolic meshing step 
    /// to cluster radial lines closer to the airfoil wall.
    pub q: f64,
    
    /// Maximum thickness-to-chord ratio (t/c) of the airfoil.
    pub t: f64,
    
    // Numerical solver parameters
    /// Relaxation factor used in iterative schemes (like SOR - Successive Over-Relaxation).
    /// If omega = 1.0, it acts as standard Gauss-Seidel.
    pub omega: f64,
    
    /// Lower bound for the acceleration parameter sequence used in the 
    /// AF1 (Approximate Factorization) algorithm to speed up convergence.
    pub alpha_l: f64,
    
    /// Upper bound for the acceleration parameter sequence in the AF1 algorithm.
    pub alpha_h: f64,
    
    /// Selects the underlying partial differential equation (PDE) for the 
    /// elliptic grid generator (ParabolicOnly, Laplace, or Poisson with grid control).
    pub elliptic_equation: EllipticEquation, 
    
    /// Maximum number of iterations allowed for the elliptic solver before it stops.
    pub max_iter: usize,
    
    /// Convergence tolerance. The solver stops when the maximum residual 
    /// error falls below this value.
    pub conv_criterion: f64,
}

/// Defines the mathematical formulation used to generate the internal grid points.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum EllipticEquation {
    /// Skips elliptic PDE smoothing, retaining the fast algebraic/parabolic Nakamura marching grid.
    ParabolicOnly,

    /// Solves the homogeneous Laplace equation (∇²ξ = 0, ∇²η = 0).
    /// Tends to produce smooth grids but lacks direct control over spacing near the boundaries.
    Laplace,
    
    /// Solves the inhomogeneous Poisson equation (∇²ξ = P, ∇²η = Q).
    /// Introduces source terms (P, Q) to enforce grid clustering and orthogonality near the boundaries.
    Poisson {
        /// Target normal grid spacing (ds) at the airfoil surface (inner boundary wall).
        ds_wall: f64, 
        
        /// Decay factor 'a' that dictates how quickly the longitudinal grid control 
        /// fades out as it moves away from the source.
        a_decay: f64, 
        
        /// Decay factor 'b' that dictates how quickly the normal grid control 
        /// (attraction to the wall) fades out towards the outer boundary.
        b_decay: f64, 
    }, 
}

/// Available airfoil geometries for the inner boundary condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AirfoilType {
    /// A symmetrical biconvex airfoil formed by parabolic arcs.
    Biconvex,
    
    /// A standard 4-digit symmetrical NACA airfoil (e.g., NACA 0012).
    NACA00XX, 
}

impl Default for Config {
    fn default() -> Self {
        Self::preset_naca0012_laplace()
    }
}

impl Config {
    /// Ensures longitudinal_points satisfies `4 * k + 1` (so Leading Edge is at `mid` and
    /// mid-chord crests at `x = 0.5` are symmetric without duplicate nodes) and >= 21,
    /// and normal_points >= 5.
    pub fn sanitize(&mut self) {
        if self.longitudinal_points < 21 {
            self.longitudinal_points = 21;
        }
        let rem = (self.longitudinal_points - 1) % 4;
        if rem != 0 {
            self.longitudinal_points += 4 - rem;
        }
        if self.normal_points < 5 {
            self.normal_points = 5;
        }
        self.t = self.t.clamp(0.01, 0.40);
        self.r_max = self.r_max.max(2.0);
        self.stretching_factor = self.stretching_factor.clamp(1.0, 1.30);
        self.q = self.q.clamp(1.0, 1.50);
        self.omega = self.omega.clamp(0.1, 1.95);
        self.alpha_l = self.alpha_l.max(1e-4);
        self.alpha_h = self.alpha_h.max(self.alpha_l);
        self.conv_criterion = self.conv_criterion.clamp(1e-12, 1e-1);
    }

    pub fn preset_naca0012_laplace() -> Self {
        Self {
            meshname: "naca0012_laplace".to_string(),
            airfoil_type: AirfoilType::NACA00XX,
            t: 0.12,
            r_max: 20.0,
            longitudinal_points: 213,
            normal_points: 65,
            stretching_factor: 1.04,
            q: 1.07,
            omega: 1.3,
            alpha_l: 0.5,
            alpha_h: 20.0,
            elliptic_equation: EllipticEquation::Laplace,
            max_iter: 1500,
            conv_criterion: 1e-6,
        }
    }

    pub fn preset_naca0012_poisson() -> Self {
        Self {
            meshname: "naca0012_poisson".to_string(),
            airfoil_type: AirfoilType::NACA00XX,
            t: 0.12,
            r_max: 20.0,
            longitudinal_points: 213,
            normal_points: 65,
            stretching_factor: 1.04,
            q: 1.07,
            omega: 1.3,
            alpha_l: 0.5,
            alpha_h: 20.0,
            elliptic_equation: EllipticEquation::Poisson {
                ds_wall: 0.005,
                a_decay: 1.5,
                b_decay: 1.5,
            },
            max_iter: 2000,
            conv_criterion: 1e-6,
        }
    }

    pub fn preset_naca0012_fine() -> Self {
        Self {
            meshname: "naca0012_fine".to_string(),
            airfoil_type: AirfoilType::NACA00XX,
            t: 0.12,
            r_max: 20.0,
            longitudinal_points: 713,
            normal_points: 150,
            stretching_factor: 1.03,
            q: 1.05,
            omega: 1.3,
            alpha_l: 0.5,
            alpha_h: 20.0,
            elliptic_equation: EllipticEquation::Poisson {
                ds_wall: 0.005,
                a_decay: 1.5,
                b_decay: 1.5,
            },
            max_iter: 3000,
            conv_criterion: 1e-6,
        }
    }

    pub fn preset_biconvex_standard() -> Self {
        Self {
            meshname: "biconvex_mesh".to_string(),
            airfoil_type: AirfoilType::Biconvex,
            t: 0.10,
            r_max: 15.0,
            longitudinal_points: 213,
            normal_points: 65,
            stretching_factor: 1.04,
            q: 1.07,
            omega: 1.3,
            alpha_l: 0.5,
            alpha_h: 20.0,
            elliptic_equation: EllipticEquation::Laplace,
            max_iter: 1500,
            conv_criterion: 1e-6,
        }
    }
}
