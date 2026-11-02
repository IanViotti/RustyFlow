use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Basic structure representing a 2D Cartesian coordinate in the physical domain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Per-node geometric and quality metrics for structured 2D O-grids.
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeQualityMetrics {
    /// Transformation Jacobian J = 1 / (x_xi * y_eta - x_eta * y_xi)
    pub jacobian: f64,
    /// Approximate physical cell area |x_xi * y_eta - x_eta * y_xi|
    pub cell_area: f64,
    /// Deviation from 90-degree orthogonality (in degrees, 0.0 = perfectly orthogonal)
    pub orthogonality_dev_deg: f64,
    /// Local grid aspect ratio max(|r_xi|, |r_eta|) / min(|r_xi|, |r_eta|)
    pub aspect_ratio: f64,
    /// Contravariant metric tensor component A1 = xi_x^2 + xi_y^2
    pub a1: f64,
    /// Contravariant metric tensor component A2 = xi_x * eta_x + xi_y * eta_y
    pub a2: f64,
    /// Contravariant metric tensor component A3 = eta_x^2 + eta_y^2
    pub a3: f64,
}

/// Global summary of mesh quality and geometric characteristics.
#[derive(Clone, Copy, Debug, Default)]
pub struct MeshQualitySummary {
    pub ni: usize,
    pub nj: usize,
    pub total_nodes: usize,
    pub total_cells: usize,
    pub min_wall_spacing_normal: f64,
    pub max_wall_spacing_normal: f64,
    pub min_wall_spacing_tangent: f64,
    pub max_orthogonality_dev_deg: f64,
    pub wall_avg_orthogonality_dev_deg: f64,
    pub max_aspect_ratio: f64,
    pub min_jacobian: f64,
    pub negative_jacobian_count: usize,
}

/// Scalar fields available for visualization on the mesh alone (pre-CFD).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshScalarField {
    None,
    OrthogonalityDev,
    AspectRatio,
    CellArea,
    Log10Jacobian,
    MetricA1,
    MetricA2,
    MetricA3,
}

impl MeshScalarField {
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "Wireframe Only (No Scalar)",
            Self::OrthogonalityDev => "Orthogonality Deviation (°)",
            Self::AspectRatio => "Grid Aspect Ratio",
            Self::CellArea => "Log10(Cell Area)",
            Self::Log10Jacobian => "Log10(Jacobian J)",
            Self::MetricA1 => "Log10(Metric A1)",
            Self::MetricA2 => "Metric A2 (Cross-Derivative)",
            Self::MetricA3 => "Log10(Metric A3)",
        }
    }

    pub fn sample(&self, m: &NodeQualityMetrics) -> f64 {
        match self {
            Self::None => 0.0,
            Self::OrthogonalityDev => m.orthogonality_dev_deg,
            Self::AspectRatio => m.aspect_ratio,
            Self::CellArea => m.cell_area.max(1e-12).log10(),
            Self::Log10Jacobian => m.jacobian.abs().max(1e-12).log10(),
            Self::MetricA1 => m.a1.max(1e-12).log10(),
            Self::MetricA2 => m.a2,
            Self::MetricA3 => m.a3.max(1e-12).log10(),
        }
    }
}

/// Computes node-by-node quality metrics and a global summary for a Mesher grid
/// where `j = 0` is the airfoil surface and `j = nj - 1` is the farfield boundary.
pub fn compute_mesh_quality(
    grid: &Array2<Point>,
) -> (Array2<NodeQualityMetrics>, MeshQualitySummary) {
    let (ni, nj) = grid.dim();
    let mut metrics = Array2::<NodeQualityMetrics>::default((ni, nj));

    if ni < 3 || nj < 3 {
        return (metrics, MeshQualitySummary::default());
    }

    let n_unique = ni - 1; // i = ni - 1 overlaps i = 0 at the wake cut

    let mut min_wall_normal = f64::MAX;
    let mut max_wall_normal = 0.0_f64;
    let mut min_wall_tangent = f64::MAX;
    let mut max_ortho_dev = 0.0_f64;
    let mut wall_ortho_sum = 0.0_f64;
    let mut max_ar = 0.0_f64;
    let mut min_jac = f64::MAX;
    let mut neg_jac_count = 0usize;

    for j in 0..nj {
        for i in 0..n_unique {
            let i_plus = if i == n_unique - 1 { 0 } else { i + 1 };
            let i_minus = if i == 0 { n_unique - 1 } else { i - 1 };

            // Central difference along xi (wrapping around wake cut)
            let x_xi = 0.5 * (grid[[i_plus, j]].x - grid[[i_minus, j]].x);
            let y_xi = 0.5 * (grid[[i_plus, j]].y - grid[[i_minus, j]].y);

            // Second-order differences along eta (j=0 wall, j=nj-1 farfield)
            let (x_eta, y_eta) = if j == 0 {
                (
                    0.5 * (-3.0 * grid[[i, 0]].x + 4.0 * grid[[i, 1]].x - grid[[i, 2]].x),
                    0.5 * (-3.0 * grid[[i, 0]].y + 4.0 * grid[[i, 1]].y - grid[[i, 2]].y),
                )
            } else if j == nj - 1 {
                (
                    0.5 * (3.0 * grid[[i, j]].x - 4.0 * grid[[i, j - 1]].x + grid[[i, j - 2]].x),
                    0.5 * (3.0 * grid[[i, j]].y - 4.0 * grid[[i, j - 1]].y + grid[[i, j - 2]].y),
                )
            } else {
                (
                    0.5 * (grid[[i, j + 1]].x - grid[[i, j - 1]].x),
                    0.5 * (grid[[i, j + 1]].y - grid[[i, j - 1]].y),
                )
            };

            // Both (i, j) in Mesher convention and inverted (i, j) in Solver convention
            // form a right-handed coordinate system with det = x_xi * y_eta - x_eta * y_xi > 0.
            let det = x_xi * y_eta - x_eta * y_xi;
            let cell_area = det.abs();
            let jacobian = if det.abs() > 1e-16 { 1.0 / det } else { 0.0 };

            if det <= 0.0 {
                neg_jac_count += 1;
            }
            if jacobian < min_jac {
                min_jac = jacobian;
            }

            let len_xi = (x_xi * x_xi + y_xi * y_xi).sqrt();
            let len_eta = (x_eta * x_eta + y_eta * y_eta).sqrt();

            let ortho_dev_deg = if len_xi > 1e-14 && len_eta > 1e-14 {
                let cos_theta = ((x_xi * x_eta + y_xi * y_eta) / (len_xi * len_eta)).clamp(-1.0, 1.0);
                let angle_deg = cos_theta.acos().to_degrees();
                (90.0 - angle_deg).abs()
            } else {
                0.0
            };

            let min_len = len_xi.min(len_eta);
            let max_len = len_xi.max(len_eta);
            let aspect_ratio = if min_len > 1e-14 {
                max_len / min_len
            } else {
                1.0
            };

            let xi_x = jacobian * y_eta;
            let xi_y = -jacobian * x_eta;
            let eta_x = -jacobian * y_xi;
            let eta_y = jacobian * x_xi;

            let a1 = xi_x * xi_x + xi_y * xi_y;
            let a2 = xi_x * eta_x + xi_y * eta_y;
            let a3 = eta_x * eta_x + eta_y * eta_y;

            metrics[[i, j]] = NodeQualityMetrics {
                jacobian,
                cell_area,
                orthogonality_dev_deg: ortho_dev_deg,
                aspect_ratio,
                a1,
                a2,
                a3,
            };

            if ortho_dev_deg > max_ortho_dev {
                max_ortho_dev = ortho_dev_deg;
            }
            if aspect_ratio > max_ar {
                max_ar = aspect_ratio;
            }

            if j == 0 {
                wall_ortho_sum += ortho_dev_deg;

                let dx_n = grid[[i, 1]].x - grid[[i, 0]].x;
                let dy_n = grid[[i, 1]].y - grid[[i, 0]].y;
                let ds_n = (dx_n * dx_n + dy_n * dy_n).sqrt();
                if ds_n < min_wall_normal {
                    min_wall_normal = ds_n;
                }
                if ds_n > max_wall_normal {
                    max_wall_normal = ds_n;
                }

                let dx_t = grid[[i_plus, 0]].x - grid[[i, 0]].x;
                let dy_t = grid[[i_plus, 0]].y - grid[[i, 0]].y;
                let ds_t = (dx_t * dx_t + dy_t * dy_t).sqrt();
                if ds_t < min_wall_tangent {
                    min_wall_tangent = ds_t;
                }
            }
        }

        // Periodic closure at wake cut i = ni - 1
        metrics[[ni - 1, j]] = metrics[[0, j]];
    }

    let summary = MeshQualitySummary {
        ni,
        nj,
        total_nodes: ni * nj,
        total_cells: (ni - 1) * (nj - 1),
        min_wall_spacing_normal: if min_wall_normal == f64::MAX { 0.0 } else { min_wall_normal },
        max_wall_spacing_normal: max_wall_normal,
        min_wall_spacing_tangent: if min_wall_tangent == f64::MAX { 0.0 } else { min_wall_tangent },
        max_orthogonality_dev_deg: max_ortho_dev,
        wall_avg_orthogonality_dev_deg: wall_ortho_sum / (n_unique as f64),
        max_aspect_ratio: max_ar,
        min_jacobian: if min_jac == f64::MAX { 0.0 } else { min_jac },
        negative_jacobian_count: neg_jac_count,
    };

    (metrics, summary)
}
