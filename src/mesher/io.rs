use ndarray::Array2;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::core::grid::Point;
use crate::mesher::config::Config;

/// Initializes the output directory for the current mesh job.
/// Ensures that the required folder structure exists before saving files.
pub fn init_mesh_directory(config: &Config) -> io::Result<String> {
    let dir = format!("job_files/{}", config.meshname);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Allocates the 2D array representing the computational grid (xi, eta).
/// Initializes all points at the origin (0.0, 0.0) before the solver maps them to physical space.
pub fn create_grid(longitudinal_points: usize, normal_points: usize) -> Array2<Point> {
    Array2::from_elem(
        (longitudinal_points, normal_points),
        Point { x: 0.0, y: 0.0 },
    )
}

/// Exports the grid coordinates to a basic CSV format.
/// Useful for quick data inspection, debugging, or custom Python plotting scripts.
pub fn save_grid_to_csv(grid: &Array2<Point>, filepath: &Path) -> io::Result<()> {
    let file = File::create(filepath)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "i,j,x,y")?;
    for ((i, j), point) in grid.indexed_iter() {
        writeln!(writer, "{},{},{},{}", i, j, point.x, point.y)?;
    }

    Ok(())
}

/// Exports the 2D grid to the VTK Legacy format (Structured Grid).
/// This is the standard output format to visualize the mesh topology, 
/// orthogonality, and quality using post-processing software like ParaView,
/// and is directly compatible with RustPotential.
pub fn export_vtk_structured_grid(grid: &Array2<Point>, filepath: &Path) -> io::Result<()> {
    let shape = grid.shape();
    let ni = shape[0];
    let nj = shape[1];
    let num_points = ni * nj;

    let file = File::create(filepath)?;
    let mut writer = BufWriter::new(file);

    // --- 1. VTK Legacy Header ---
    writeln!(writer, "# vtk DataFile Version 3.0")?;
    writeln!(writer, "Malha Parabolica 2D O-Mesh")?;
    writeln!(writer, "ASCII")?;
    writeln!(writer, "DATASET STRUCTURED_GRID")?;
    writeln!(writer, "DIMENSIONS {} {} 1", ni, nj)?;
    writeln!(writer, "POINTS {} float", num_points)?;

    // --- 2. Writing Coordinates ---
    // VTK requires points to be listed varying 'i' fastest, then 'j', then 'k'.
    for j in 0..nj {
        for i in 0..ni {
            let pt = &grid[[i, j]];
            writeln!(writer, "{:.6} {:.6} 0.0", pt.x, pt.y)?;
        }
    }

    Ok(())
}

/// Imports a 2D structured O-grid from a VTK Legacy `.vtk` file.
/// Automatically detects if the file has `j = 0` at the farfield (e.g. a solver output VTK)
/// and normalizes it so `j = 0` is always the airfoil surface and `j = nj - 1` is the farfield.
pub fn import_vtk_structured_grid(filepath: &Path) -> Result<Array2<Point>, String> {
    let file = File::open(filepath).map_err(|e| format!("Failed to open VTK file: {}", e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut imax = 0usize;
    let mut jmax = 0usize;
    let mut grid: Option<Array2<Point>> = None;

    while let Some(Ok(line)) = lines.next() {
        let trimmed = line.trim();
        if trimmed.starts_with("DIMENSIONS") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 3 {
                return Err("Invalid DIMENSIONS line in VTK file.".to_string());
            }
            imax = parts[1]
                .parse::<usize>()
                .map_err(|_| "Failed to parse imax dimension from VTK.".to_string())?;
            jmax = parts[2]
                .parse::<usize>()
                .map_err(|_| "Failed to parse jmax dimension from VTK.".to_string())?;
        } else if trimmed.starts_with("POINTS") {
            if imax == 0 || jmax == 0 {
                return Err("VTK POINTS section encountered before valid DIMENSIONS.".to_string());
            }
            let mut arr = Array2::from_elem((imax, jmax), Point { x: 0.0, y: 0.0 });
            for j in 0..jmax {
                for i in 0..imax {
                    if let Some(Ok(pt_line)) = lines.next() {
                        let coords: Vec<f64> = pt_line
                            .split_whitespace()
                            .filter_map(|s| s.parse::<f64>().ok())
                            .collect();
                        if coords.len() >= 2 {
                            arr[[i, j]] = Point {
                                x: coords[0],
                                y: coords[1],
                            };
                        }
                    }
                }
            }
            grid = Some(arr);
            break;
        }
    }

    let mut arr = grid.ok_or_else(|| "No POINTS section found in VTK file.".to_string())?;

    // Check if j=0 is farfield (average radius from (0.5, 0) larger at j=0 than at j=jmax-1)
    let r_j0: f64 = (0..imax)
        .map(|i| ((arr[[i, 0]].x - 0.5).powi(2) + arr[[i, 0]].y.powi(2)).sqrt())
        .sum::<f64>()
        / (imax as f64);
    let r_jmax: f64 = (0..imax)
        .map(|i| ((arr[[i, jmax - 1]].x - 0.5).powi(2) + arr[[i, jmax - 1]].y.powi(2)).sqrt())
        .sum::<f64>()
        / (imax as f64);

    if r_j0 > r_jmax {
        // Flip i and j back to Mesher convention (j=0 at airfoil wall)
        let mut flipped = Array2::from_elem((imax, jmax), Point { x: 0.0, y: 0.0 });
        for j in 0..jmax {
            for i in 0..imax {
                flipped[[i, j]] = arr[[imax - 1 - i, jmax - 1 - j]];
            }
        }
        arr = flipped;
    }

    Ok(arr)
}

/// Thomas algorithm (Tridiagonal Matrix Algorithm - TDMA)
///
/// Solves a tridiagonal linear system of equations efficiently in O(N) time:
///
///     a_j x_{j-1} + b_j x_j + c_j x_{j+1} = d_j
pub fn thomas_algorithm(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut c_star = vec![0.0; n];
    let mut d_star = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Step 1: Forward sweep - Implicitly building L and U
    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for i in 1..n {
        let m = 1.0 / (b[i] - a[i] * c_star[i - 1]);
        if i < n - 1 {
            c_star[i] = c[i] * m;
        }
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) * m;
    }

    // Step 2: Backward substitution
    x[n - 1] = d_star[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }

    x
}

/// Solves a cyclic (periodic) tridiagonal system using the Sherman-Morrison formula.
/// 
/// This is necessary for O-grid topologies. In an O-grid, the wrap-around 
/// direction (longitudinal/xi) forms a closed loop, meaning the first node and the 
/// last node on a given constant-eta line are adjacent (or identical at the wake cut).
pub fn periodic_thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();

    // Defines gamma (usually -b[0] to ensure numerical stability)
    let gamma = -b[0];

    // Creates the modified main diagonal (Matrix T)
    let mut bb = b.to_vec();
    bb[0] = b[0] - gamma;
    bb[n - 1] = b[n - 1] - a[0] * c[n - 1] / gamma;

    // Assembles vector u (only the endpoints contain values)
    let mut u = vec![0.0; n];
    u[0] = gamma;
    u[n - 1] = c[n - 1];

    // Double Resolution:
    // 1. Solves Ty = d using the modified main diagonal
    let y: Vec<f64> = thomas_algorithm(a, &bb, c, d);

    // 2. Solves Tq = u
    let q = thomas_algorithm(a, &bb, c, &u);

    // 3. Final reconstruction of the solution combining the results
    let v_n_minus_1 = a[0] / gamma;

    // Dot products (v dot y) and (v dot q)
    let v_dot_y = y[0] + v_n_minus_1 * y[n - 1];
    let v_dot_q = q[0] + v_n_minus_1 * q[n - 1];

    let factor = v_dot_y / (1.0 + v_dot_q);

    let mut x = vec![0.0; n];
    for i in 0..n {
        x[i] = y[i] - factor * q[i];
    }

    x
}
