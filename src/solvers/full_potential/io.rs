use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::SystemTime;
use ndarray::Array2;

use crate::solvers::full_potential::config::Config;
use crate::solvers::full_potential::solver_utils::{Grid, PointData};

/// Generates a timestamp string YYYY-MM-DD_HH-MM-SS from system time.
pub fn current_timestamp_string() -> String {
    let now = SystemTime::now();
    let duration = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = duration.as_secs();

    let sec = total_secs % 60;
    let total_min = total_secs / 60;
    let min = total_min % 60;
    let total_hours = total_min / 60;
    let hour = total_hours % 24;

    let mut days = (total_hours / 24) as i64;
    let mut year = 1970i32;

    loop {
        let leap = is_leap_year(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days >= days_in_year {
            days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }

    let leap = is_leap_year(year);
    let days_in_months = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    let mut month = 1usize;
    for &dim in &days_in_months {
        if days >= dim as i64 {
            days -= dim as i64;
            month += 1;
        } else {
            break;
        }
    }
    let day = (days + 1) as usize;

    format!(
        "{:04}-{:02}-{:02}_{:02}-{:02}-{:02}",
        year, month, day, hour, min, sec
    )
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Creates a unique output directory incorporating jobname and date/timestamp,
/// and exports `solution.vtk`, `Cp_distribution.csv`, `solver_config.txt`, and `job_config.toml`.
pub fn export_full_potential_results(
    config: &mut Config,
    grid: &Grid,
    domain: &Array2<PointData>,
) -> io::Result<String> {
    let timestamp = current_timestamp_string();
    let output_dir = format!("job_files/{}_{}", config.jobname, timestamp);
    fs::create_dir_all(&output_dir)?;
    config.output_dir = output_dir.clone();

    let dir_path = Path::new(&output_dir);
    export_vtk_fields(config, grid, domain, &dir_path.join("solution.vtk"))?;
    export_cp_csv(config, grid, domain, &dir_path.join("Cp_distribution.csv"))?;
    save_solver_configuration(config, grid, &dir_path.join("solver_config.txt"))?;

    if let Ok(toml_str) = toml::to_string_pretty(config) {
        let _ = fs::write(dir_path.join("job_config.toml"), toml_str);
    }

    Ok(output_dir)
}

pub fn export_vtk_fields(
    config: &Config,
    grid: &Grid,
    domain: &Array2<PointData>,
    filepath: &Path,
) -> io::Result<()> {
    let file = File::create(filepath)?;
    let mut writer = BufWriter::new(file);

    let imax = grid.imax;
    let jmax = grid.jmax;
    let total_points = imax * jmax;

    writeln!(writer, "# vtk DataFile Version 3.0")?;
    writeln!(writer, "Full Potential CFD Results")?;
    writeln!(writer, "ASCII")?;
    writeln!(writer, "DATASET STRUCTURED_GRID")?;
    writeln!(writer, "DIMENSIONS {} {} 1", imax, jmax)?;

    writeln!(writer, "POINTS {} float", total_points)?;
    for j in 0..jmax {
        for i in 0..imax {
            let pt = &grid.points[[i, j]];
            writeln!(writer, "{:.6} {:.6} 0.000000", pt.x, pt.y)?;
        }
    }

    writeln!(writer, "POINT_DATA {}", total_points)?;

    writeln!(writer, "SCALARS Potential float 1")?;
    writeln!(writer, "LOOKUP_TABLE default")?;
    for j in 0..jmax {
        for i in 0..imax {
            writeln!(writer, "{:.6}", domain[[i, j]].flow_ij.phi)?;
        }
    }

    writeln!(writer, "SCALARS Mach_Number float 1")?;
    writeln!(writer, "LOOKUP_TABLE default")?;
    for j in 0..jmax {
        for i in 0..imax {
            writeln!(writer, "{:.6}", domain[[i, j]].flow_ij.M)?;
        }
    }

    writeln!(writer, "SCALARS Density float 1")?;
    writeln!(writer, "LOOKUP_TABLE default")?;
    for j in 0..jmax {
        for i in 0..imax {
            writeln!(writer, "{:.6}", domain[[i, j]].flow_ij.rho)?;
        }
    }

    writeln!(writer, "VECTORS Velocity float")?;
    for j in 0..jmax {
        for i in 0..imax {
            writeln!(
                writer,
                "{:.6} {:.6} 0.000000",
                domain[[i, j]].flow_ij.u,
                domain[[i, j]].flow_ij.v
            )?;
        }
    }

    if config.export_metrics {
        writeln!(writer, "SCALARS Jacobian float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.8}", domain[[i, j]].metrics_ij.J)?;
            }
        }

        writeln!(writer, "SCALARS Metric_A1 float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.8}", domain[[i, j]].metrics_ij.A1)?;
            }
        }

        writeln!(writer, "SCALARS Metric_A2 float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.8}", domain[[i, j]].metrics_ij.A2)?;
            }
        }

        writeln!(writer, "SCALARS Metric_A3 float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.8}", domain[[i, j]].metrics_ij.A3)?;
            }
        }
    }

    Ok(())
}

pub fn export_cp_csv(
    config: &Config,
    grid: &Grid,
    domain: &Array2<PointData>,
    filepath: &Path,
) -> io::Result<()> {
    let file = File::create(filepath)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "x,y,Cp")?;

    let gamma = config.gamma;
    let m_inf = config.Mach_inf;
    let rho_inf = config.rho_inf;
    let cp_multiplier = 2.0 / (gamma * m_inf.powi(2));

    let j_wall = grid.jmax - 1;

    for i in 0..(grid.imax - 1) {
        let x = grid.points[[i, j_wall]].x;
        let y = grid.points[[i, j_wall]].y;
        let rho_local = domain[[i, j_wall]].flow_ij.rho;
        let cp = cp_multiplier * ((rho_local / rho_inf).powf(gamma) - 1.0);
        writeln!(writer, "{:.6},{:.6},{:.6}", x, y, cp)?;
    }

    Ok(())
}

pub fn save_solver_configuration(
    config: &Config,
    grid: &Grid,
    filepath: &Path,
) -> io::Result<()> {
    let file = File::create(filepath)?;
    let mut writer = BufWriter::new(file);

    let total_nodes = grid.imax * grid.jmax;
    let regime = if config.Mach_inf < 0.8 {
        "Subsonic"
    } else if config.Mach_inf <= 1.2 {
        "Transonic"
    } else {
        "Supersonic"
    };

    writeln!(
        writer,
        "================================================================================"
    )?;
    writeln!(
        writer,
        "                    RUSTPOTENTIAL SOLVER CONFIGURATION                          "
    )?;
    writeln!(
        writer,
        "================================================================================"
    )?;
    writeln!(writer, "Job Name                : {}", config.jobname)?;
    writeln!(writer, "Output Directory        : {}", config.output_dir)?;
    writeln!(writer)?;
    writeln!(
        writer,
        "--------------------------------------------------------------------------------"
    )?;
    writeln!(
        writer,
        "  MESH SPECIFICATIONS                                                           "
    )?;
    writeln!(
        writer,
        "--------------------------------------------------------------------------------"
    )?;
    writeln!(writer, "Mesh File               : {}", config.meshname)?;
    writeln!(
        writer,
        "Grid Dimensions (imax)  : {} (Azimuthal / Xi direction)",
        grid.imax
    )?;
    writeln!(
        writer,
        "Grid Dimensions (jmax)  : {} (Radial / Eta direction)",
        grid.jmax
    )?;
    writeln!(writer, "Total Grid Nodes        : {}", total_nodes)?;
    writeln!(
        writer,
        "Wake Cut Node Pair      : i = 0 <-> i = {}",
        grid.imax - 1
    )?;
    writeln!(writer, "Farfield Boundary       : j = 0")?;
    writeln!(writer, "Airfoil Wall Boundary   : j = {}", grid.jmax - 1)?;
    writeln!(writer)?;
    writeln!(
        writer,
        "--------------------------------------------------------------------------------"
    )?;
    writeln!(
        writer,
        "  FLOW & FLIGHT PARAMETERS                                                      "
    )?;
    writeln!(
        writer,
        "--------------------------------------------------------------------------------"
    )?;
    writeln!(
        writer,
        "Freestream Mach (M_inf) : {:.6} ({})",
        config.Mach_inf, regime
    )?;
    writeln!(
        writer,
        "Angle of Attack (AoA)   : {:.4} deg ({:.6} rad)",
        config.AoA, config.alpha_rad
    )?;
    writeln!(writer, "Specific Heat Ratio (g) : {:.4}", config.gamma)?;
    writeln!(
        writer,
        "Freestream Speed (q_inf): {:.6} (non-dimensional)",
        config.q_inf
    )?;
    writeln!(writer, "Freestream Velocity u   : {:.6}", config.u_inf)?;
    writeln!(writer, "Freestream Velocity v   : {:.6}", config.v_inf)?;
    writeln!(writer, "Freestream Density (rho): {:.6}", config.rho_inf)?;
    writeln!(writer)?;
    writeln!(
        writer,
        "--------------------------------------------------------------------------------"
    )?;
    writeln!(
        writer,
        "  AF2 NUMERICAL SCHEME CONTROLS                                                 "
    )?;
    writeln!(
        writer,
        "--------------------------------------------------------------------------------"
    )?;
    writeln!(writer, "Max Iterations          : {}", config.max_iterations)?;
    writeln!(
        writer,
        "Convergence Tolerance   : {:.2e}",
        config.convergence_residual
    )?;
    writeln!(writer, "Relaxation Factor (w)   : {:.4}", config.omega)?;
    writeln!(writer, "Circulation Relax (Rg)  : {:.4}", config.Rg)?;
    writeln!(
        writer,
        "Alpha Sequence Length M : {}",
        config.alpha_sequence_length
    )?;
    writeln!(writer, "Supersonic Damping Beta : {:.4}", config.beta_mult)?;
    writeln!(writer, "Holst Upwind Coeff (C)  : {:.4}", config.C)?;
    writeln!(
        writer,
        "Export Geometric Metrics: {}",
        config.export_metrics
    )?;
    writeln!(
        writer,
        "================================================================================"
    )?;

    Ok(())
}
