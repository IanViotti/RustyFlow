# RustyFlow 🦀🌊

**Unified 2D Structured O-Grid Mesher & Full Potential CFD Workbench**

[![Language: Rust](https://img.shields.io/badge/Language-Rust-orange.svg)](https://www.rust-lang.org/)
[![GUI: egui / eframe](https://img.shields.io/badge/GUI-egui%20%2F%20eframe-blue.svg)](https://github.com/emilk/egui)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RustyFlow** is a standalone, hardware-accelerated desktop CFD workbench written in 100% Rust. It integrates [**RustyMesher**](https://github.com/IanViotti/RustyMesher) (Parabolic & Elliptic PDE O-Grid generation) and [**RustPotential**](https://github.com/IanViotti/RustPotential) (Conservative Full Potential AF2 solver with Kutta circulation and Holst artificial density shock capturing) into a single interactive window—allowing you to generate grids, inspect mesh quality, solve subsonic/transonic aerodynamics, and visualize flowfields and $C_p$ distributions in real time.

---

## Table of Contents

1. [Overview & Key Capabilities](#1-overview--key-capabilities)
2. [Compilation & Quick Start](#2-compilation--quick-start)
3. [Theoretical Summary](#3-theoretical-summary)
   - [Parabolic & Elliptic O-Grid Generation](#31-parabolic--elliptic-o-grid-generation)
   - [Full Potential Equation & Holst Artificial Density](#32-full-potential-equation--holst-artificial-density)
   - [AF2 Scheme, Kutta Condition & Lifting Circulation](#33-af2-scheme-kutta-condition--lifting-circulation)
4. [Complete Parameter Reference](#4-complete-parameter-reference)
   - [Top Toolbar & Global Presets](#41-top-toolbar--global-presets)
   - [Stage 1: Mesh Generator Parameters (`RustyMesher`)](#42-stage-1-mesh-generator-parameters-rustymesher)
   - [Stage 2: CFD Solver Parameters (`RustPotential`)](#43-stage-2-cfd-solver-parameters-rustpotential)
   - [2D Viewport & Visualization Controls](#44-2d-viewport--visualization-controls)
   - [Bottom Analytics & Aerodynamic Outputs](#45-bottom-analytics--aerodynamic-outputs)
5. [TOML Configuration File Reference](#5-toml-configuration-file-reference)
6. [Project Architecture & Adding Future Solvers (Euler / RANS)](#6-project-architecture--adding-future-solvers-euler--rans)
7. [Practical Tuning & Troubleshooting Guide](#7-practical-tuning--troubleshooting-guide)
8. [Academic Disclaimer & License](#8-academic-disclaimer--license)

---

## 1. Overview & Key Capabilities

- **Zero-Copy In-Memory Pipeline**: Passes `ndarray::Array2` grids and flowfields directly between the mesher, the CFD solver, and the 60 FPS GPU renderer without intermediate disk bottlenecks.
- **Real-Time Iteration Telemetry**: Watch the O-grid smooth during AF1/ADI iterations and watch transonic shock waves, supersonic pockets ($M \ge 1$), and surface $C_p$ curves form live during AF2 iterations.
- **Comprehensive Aerodynamic Post-Processing**:
  - Gouraud-shaded 2D colormaps (`Mach`, `Cp`, `Density`, `Potential`, `Velocity`, and `Mesh Metrics`).
  - Exact triangle-interpolated **Sonic Line ($M = 1.0$)**, **Iso-Contour Lines**, and **Velocity Vectors**.
  - Interactive **Surface $C_p(x/c)$** and **Surface $M(x/c)$** plots with inverted aerodynamic Y-axis (`-Cp` upward) and critical sonic $C_p^*$ marker.
  - Automatic integration of Lift ($C_l$ & Kutta-Joukowski $C_{l,\text{KJ}}$), Drag ($C_d$), Quarter-Chord Pitching Moment ($C_{m,c/4}$), Normal/Axial forces ($C_n, C_a$), and Circulation ($\Gamma$).
- **Full ParaView & Script Compatibility**: Import and export legacy `.vtk` structured grids, `.csv` distributions, `solver_config.txt`, and `.toml` configuration files.

---

## 2. Compilation & Quick Start

### Prerequisites
- [Rust Toolchain](https://rustup.rs/) (1.80+ / Edition 2024).

### Build & Run
From the root directory (`RustyFlow/`), compile and launch in release mode for maximum numerical performance:

```powershell
cargo run --release
```

Or run the compiled standalone executable directly:

```powershell
.\target\release\RustyFlow.exe
```

### 30-Second Workflow
1. **Select a Preset**: Click **⚡ Quick Presets...** in the top toolbar to choose a mesh and flight configuration (e.g., *NACA 0012 — Poisson Wall Control* + *Transonic Lifting Shock $M = 0.80, \text{AoA} = 1.0^\circ$*).
2. **Generate & Smooth Mesh**: In the left sidebar (**🕸️ 1. Mesh Generator**), adjust geometry or clustering parameters (the parabolic preview updates instantly) and click **▶ Generate & Smooth**.
3. **Run CFD Solver**: Switch to **🚀 2. CFD Solver** and click **🚀 Run Full Potential Solver (AF2)**.
4. **Inspect Results**: Use the viewport toolbar to toggle **Sonic M=1**, **Iso-Lines**, **Vectors**, or hover over nodes with **Probe**, and inspect the **🎯 Surface Cp & Aerodynamics** tab at the bottom.

---

## 3. Theoretical Summary

### 3.1 Parabolic & Elliptic O-Grid Generation
`RustyFlow` maps the physical domain $(x, y)$ around the airfoil to a uniform computational O-grid $(\xi, \eta)$:
- **Step 1 — Parabolic Marching (Nakamura)**: Advances layer-by-layer from the airfoil surface ($j = 0$) to the circular farfield ($j = N_\eta - 1$), blending locally orthogonal wall projections with straight-line outer boundary interpolation via periodic Thomas (Sherman-Morrison) sweeps.
- **Step 2 — Elliptic PDE Smoother (Thompson / Sorenson & Steger)**: Solves the curvilinear elliptic system via **AF1 (Alternating Direction Implicit)** approximate factorization:
  $$a \, \vec{r}_{\xi\xi} - 2b \, \vec{r}_{\xi\eta} + c \, \vec{r}_{\eta\eta} + J^{-2} \left( P \, \vec{r}_\xi + Q \, \vec{r}_\eta \right) = 0$$
  - **Laplace Mode**: Sets $P = Q = 0$ for maximum global smoothness.
  - **Poisson Mode**: Dynamically evaluates wall source terms $(P_0, Q_0)$ at $j=0$ to enforce strict boundary orthogonality ($\vec{r}_\xi \cdot \vec{r}_\eta = 0$) and target first-cell normal spacing $\Delta s_{\text{wall}}$, decaying exponentially into the field as $P_{i,j} = P_0(i) e^{-a j}$ and $Q_{i,j} = Q_0(i) e^{-b j}$.

### 3.2 Full Potential Equation & Holst Artificial Density
The steady 2D compressible, inviscid, irrotational, isentropic flow is governed by the strong conservative Full Potential equation in generalized curvilinear coordinates:

$$\frac{\partial}{\partial \xi} \left( \frac{\tilde{\rho} U}{J} \right) + \frac{\partial}{\partial \eta} \left( \frac{\bar{\rho} V}{J} \right) = 0$$

where contravariant velocities are $U = A_1 \phi_\xi + A_2 \phi_\eta$ and $V = A_2 \phi_\xi + A_3 \phi_\eta$, and isentropic density (normalized by stagnation density $\rho_t$ and critical sound speed $a^*$) is:

$$\rho = \left[ 1 - \frac{\gamma - 1}{\gamma + 1} (U \phi_\xi + V \phi_\eta) \right]^{\frac{1}{\gamma - 1}}$$

To capture transonic shock waves stably ($M > 1$), **Holst's Artificial Density** introduces an upwind bias:

$$\tilde{\rho}_{i+1/2, j} = (1 - \nu) \rho_{i+1/2, j} + \nu \rho_{i+r+1/2, j}, \quad \nu = \text{clamp}\left( (C_1 - \rho) C_2 C, \, 0, \, 1 \right)$$

where $C_1 = \left(\frac{2}{\gamma + 1}\right)^{\frac{1}{\gamma - 1}}$ is the sonic density threshold.

### 3.3 AF2 Scheme, Kutta Condition & Lifting Circulation
The residual operator $L(\phi^n)_{i,j}$ is driven to zero using the two-step **AF2 (Approximate Factorization 2)** scheme:
1. **Step 1 (Radial Bidiagonal Sweep in $\eta$)**: Marches inward from the airfoil wall to the farfield.
2. **Step 2 (Azimuthal Periodic Tridiagonal Sweep in $\xi$)**: Solves each O-grid ring via the Sherman-Morrison cyclic formula, injecting directional temporal damping $\pm \alpha \beta \delta_\xi$ ($\beta_{\text{sub}} = 0.3$, dynamic $\beta_{\text{sup}}$ for $M > 1$).

For lifting flows ($\text{AoA} \neq 0$), circulation $\Gamma^n$ is updated at each iteration via the trailing-edge **Kutta condition** ($(\phi_\xi)_{TE} = 0$):

$$\tilde{\Gamma}^n = \phi^n_{2, NY} - \phi^n_{NX-1, NY}, \quad \bar{\Gamma}^n = R_g \tilde{\Gamma}^n + (1 - R_g)\Gamma^{n-1}, \quad \Gamma^n = \frac{1}{2}\left(\bar{\Gamma}^n + 2\Gamma^{n-1} - \Gamma^{n-2}\right)$$

which imposes a potential jump $\phi_{0, j} = \phi_{NX, j} + \Gamma^n$ across the wake cut and adds a compressible point-vortex potential $\phi_{\text{vortex}} = -\frac{\Gamma^n}{2\pi} \text{atan2}\left(\sqrt{1 - M_\infty^2}\, y, \, x - 0.25\right)$ on the farfield boundary.

---

## 4. Complete Parameter Reference

### 4.1 Top Toolbar & Global Presets

| UI Control | Description |
| :--- | :--- |
| **⚡ Quick Presets...** | Loads pre-configured mesh and flight setups with one click:<br>• **Mesh Presets**: *NACA 0012 Laplace (213×65)*, *NACA 0012 Poisson Wall Control (213×65)*, *NACA 0012 Fine Grid (713×150)*, *Biconvex 10% Laplace (213×65)*.<br>• **Flight Presets**: *Subsonic Symmetric ($M=0.30, \text{AoA}=0^\circ$)*, *Transonic Lifting Shock ($M=0.80, \text{AoA}=1^\circ$)*, *Transonic Strong Shock ($M=0.85, \text{AoA}=0^\circ$)*, *Subsonic High Lift ($M=0.50, \text{AoA}=4^\circ$)*. |
| **📥 Import VTK Mesh** | Opens a native file dialog to load any structured `.vtk` O-grid (automatically detects and normalizes wall vs. farfield orientation). |
| **📤 Export Mesh** | Exports the active grid to `job_files/<meshname>/mesh.vtk`, `mesh.csv`, and `job_config.toml`. |
| **💾 Export CFD Solution** | Exports the converged CFD solution to a timestamped folder `job_files/<jobname>_<YYYY-MM-DD_HH-MM-SS>/` containing `solution.vtk`, `Cp_distribution.csv`, `solver_config.txt`, and `job_config.toml`. |

---

### 4.2 Stage 1: Mesh Generator Parameters (`RustyMesher`)

Located in the left sidebar under **🕸️ 1. Mesh Generator**:

#### Airfoil & Domain Geometry
| Parameter | TOML Key | Range / Type | Default | Detailed Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Mesh Name** | `meshname` | `String` | `"naca0012_poisson"` | Identifier used for output subfolder creation inside `job_files/`. |
| **Airfoil Profile** | `airfoil_type` | `NACA00XX` \| `Biconvex` | `NACA00XX` | Inner boundary geometry:<br>• `NACA00XX`: Standard symmetrical NACA 4-digit polynomial with closed trailing edge.<br>• `Biconvex`: Parabolic arc profile $y = \pm 2 t x (1 - x)$. |
| **Thickness ($t/c$)** | `t` | `0.02 ..= 0.30` | `0.12` | Maximum airfoil thickness-to-chord ratio (e.g., `0.12` = 12% thickness for NACA 0012). |
| **Farfield Radius ($r_{\max}$)** | `r_max` | `3.0 ..= 50.0` | `20.0` | Radius of the outer circular computational boundary in chord lengths, centered at mid-chord $(0.5, 0.0)$. Larger values (`15–25 c`) minimize farfield boundary interference in transonic flows. |

#### O-Grid Topology & Clustering
| Parameter | TOML Key | Range / Type | Default | Detailed Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Longitudinal ($N_\xi$)** | `longitudinal_points` | `21 ..= 1501` ($4k+1$) | `213` | Number of nodes wrapped around the airfoil surface. Automatically snapped to $4k+1$ so the Leading Edge ($x=0$), mid-chord crests ($x=0.5$), and Trailing Edge ($x=1$) are strictly symmetric, with $i=0$ and $i=N_\xi-1$ overlapping at the wake cut. |
| **Normal ($N_\eta$)** | `normal_points` | `10 ..= 500` | `65` | Number of radial grid layers from the airfoil wall ($j=0$) to the farfield circle ($j=N_\eta-1$). |
| **LE/TE Clustering ($r$)** | `stretching_factor` | `1.000 ..= 1.120` | `1.04` | Chordwise geometric progression ratio along the airfoil surface:<br>• `1.000`: Uniform spacing along $x/c$.<br>• `> 1.000`: Clusters nodes tightly at the Leading Edge ($x=0$) and Trailing Edge ($x=1$) to resolve stagnation points, Kutta condition gradients, and shock waves. |
| **Radial Progression ($q$)** | `q` | `1.000 ..= 1.200` | `1.07` | Geometric progression factor ($\Delta s_{j+1} = q \Delta s_j$) controlling radial layer clustering near the airfoil wall during the Step 1 Parabolic marching pass. |

#### Elliptic Smoother (`AF1 / ADI`) & Poisson Control
| Parameter | TOML Key | Range / Type | Default | Detailed Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Smoothing Formulation** | `elliptic_equation` | `ParabolicOnly` \| `Laplace` \| `Poisson` | `Poisson` | Selects the internal grid solver:<br>• `ParabolicOnly`: Fast algebraic/parabolic Nakamura march only.<br>• `Laplace`: Homogeneous elliptic smoothing ($\nabla^2 \xi = 0, \nabla^2 \eta = 0$).<br>• `Poisson`: Inhomogeneous elliptic smoothing with Steger-Sorenson boundary control. |
| **Relaxation ($\omega$)** | `omega` | `0.50 ..= 1.85` | `1.30` | Over-relaxation factor multiplying the elliptic PDE residual in the AF1 scheme. |
| **Alpha Low ($\alpha_L$)** | `alpha_l` | `0.01 ..= 10.0` | `0.50` | Minimum acceleration parameter in the 6-step AF1 geometric sequence (damps global low-frequency grid errors). |
| **Alpha High ($\alpha_H$)** | `alpha_h` | `1.0 ..= 200.0` | `20.0` | Maximum acceleration parameter in the AF1 sequence (damps local high-frequency errors). |
| **Max Iterations** | `max_iter` | `10 ..= 20000` | `1500` | Maximum global AF1/ADI smoothing cycles. |
| **Tolerance** | `conv_criterion` | `1e-10 ..= 1e-2` | `1e-6` | Convergence threshold on maximum node displacement $\max_{i,j}(|\Delta x_{i,j}|, |\Delta y_{i,j}|)$. |
| **Live Frame Interval** | *(GUI only)* | `1 ..= 500` iters | `25` | Number of elliptic iterations between live mesh snapshots sent to the 2D viewport. |
| **Wall Spacing ($\Delta s_{\text{wall}}$)** | `ds_wall` | `0.0002 ..= 0.05` | `0.005` | *(Poisson only)* Target distance from the airfoil wall ($j=0$) to the first radial ring ($j=1$) measured strictly perpendicular to the surface. |
| **Decay Factor $a$** | `a_decay` | `0.10 ..= 5.00` | `1.50` | *(Poisson only)* Exponential decay rate ($e^{-a j}$) of longitudinal spacing control source term $P_{i,j}$ away from the wall. |
| **Decay Factor $b$** | `b_decay` | `0.10 ..= 5.00` | `1.50` | *(Poisson only)* Exponential decay rate ($e^{-b j}$) of normal spacing/orthogonality control source term $Q_{i,j}$ away from the wall. |

---

### 4.3 Stage 2: CFD Solver Parameters (`RustPotential`)

Located in the left sidebar under **🚀 2. CFD Solver**:

#### Flight & Thermodynamic State
| Parameter | TOML Key | Range / Type | Default | Detailed Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Solver Engine** | *(GUI only)* | `FullPotential` \| `Euler2D` \| `Rans2D` | `FullPotential` | Selects the active CFD formulation from the extensible solver architecture. |
| **Job Name** | `jobname` | `String` | `"naca0012_AoA1_M08"` | Prefix used when exporting timestamped solution folders inside `job_files/`. |
| **Freestream Mach ($M_\infty$)** | `Mach_inf` | `0.05 ..= 1.25` | `0.80` | Undisturbed freestream Mach number:<br>• $M_\infty < 0.75$: Subsonic flow.<br>• $0.75 \le M_\infty \le 1.15$: Transonic flow with supersonic pockets and shock waves.<br>• $M_\infty > 1.15$: Low supersonic regime. |
| **Angle of Attack ($\text{AoA}$)** | `AoA` | `-8.0° ..= +12.0°` | `1.0°` | Freestream incidence angle $\alpha$ in degrees. Determines $u_\infty = q_\infty \cos\alpha$, $v_\infty = q_\infty \sin\alpha$, and drives lifting circulation $\Gamma$ via the Kutta condition. |
| **Heat Ratio ($\gamma$)** | `gamma` | `1.10 ..= 1.67` | `1.40` | Ratio of specific heats $c_p / c_v$ (`1.40` for standard air). |

#### AF2 & Circulation Numerical Controls
| Parameter | TOML Key | Range / Type | Default | Detailed Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Max Iterations** | `max_iterations` | `10 ..= 20000` | `2000` | Maximum number of AF2 pseudo-time iterations. |
| **Convergence Tol** | `convergence_residual` | `1e-10 ..= 1e-2` | `1e-5` | Convergence criterion on the maximum mass conservation residual $\max_{i,j} |L(\phi)_{i,j}|$. |
| **Relaxation ($\omega$)** | `omega` | `0.20 ..= 1.60` | `1.00` | Relaxation factor multiplying the residual RHS in Step 1 of the AF2 scheme. |
| **Circulation Relax ($R_g$)** | `Rg` | `0.05 ..= 1.00` | `0.25` | Under-relaxation weight for the trailing-edge Kutta circulation update. Lower values (`0.20–0.35`) improve stability in transonic lifting cases; `0.50` is standard for subsonic flows. |
| **Supersonic Beta ($\beta_{\text{mult}}$)** | `beta_mult` | `0.50 ..= 8.00` | `2.50` | Initial temporal damping multiplier $\beta_{\text{sup}}$ applied in locally supersonic cells ($M > 1$). Dynamically self-adjusts every $M$ iterations within $[\max(0.1, \beta_{\text{mult}}-1), \beta_{\text{mult}}+1]$ based on residual ratio $R_t^n$. |
| **Alpha Cycle Length ($M$)** | `alpha_sequence_length` | `1 ..= 32` | `8` | Length of Holst's geometric pseudo-time sequence $\alpha_k \in [\alpha_L, \alpha_H]$ where $\alpha_H = 1/\Delta s_{\min}$ and $\alpha_L = 1.0$. |
| **Holst Upwind Coeff ($C$)** | `C` | `0.50 ..= 3.00` | `1.50` | Artificial compressibility upwind multiplier in $\nu = \text{clamp}((C_1 - \rho) C_2 C, 0, 1)$. Higher values (`1.5–2.0`) add shock-capturing stability for strong transonic shocks. |
| **Export Metrics in VTK** | `export_metrics` | `bool` | `true` | When enabled, writes transformation metrics (`Jacobian`, `Metric_A1`, `Metric_A2`, `Metric_A3`) into `solution.vtk`. |
| **Live Frame Interval** | *(GUI only)* | `1 ..= 500` iters | `25` | Iteration interval between live 2D flowfield and $C_p$ plot updates during solver execution. |

---

### 4.4 2D Viewport & Visualization Controls

Located in the top strip of the central 2D viewport:

| Control | Options / Behavior |
| :--- | :--- |
| **Camera Presets** | • **🔍 Airfoil**: Fits the airfoil chord ($x \in [-0.35, 1.35]$) with locked 1:1 physical aspect ratio.<br>• **LE**: Zooms into the Leading Edge ($x \approx 0$) to inspect nose clustering and stagnation.<br>• **TE**: Zooms into the Trailing Edge ($x \approx 1$) to inspect Kutta condition closure and wake cut.<br>• **🌐 Farfield**: Fits the entire circular O-grid domain ($[-r_{\max}, +r_{\max}]$). |
| **Field (`ViewportScalar`)** | Selects the scalar field used for Gouraud-shaded quad coloring:<br>• **CFD Flowfield**: `Mach Number (M)`, `Pressure Coefficient (Cp)`, `Density (ρ / ρ_t)`, `Velocity Potential (φ)`, `Velocity Magnitude (|V| / a*)`, `Velocity u`, `Velocity v`.<br>• **Mesh & Metrics**: `Wireframe Only`, `Orthogonality Deviation (°)`, `Grid Aspect Ratio`, `Log10(Cell Area)`, `Log10(Jacobian J)`, `Log10(Metric A1)`, `Metric A2 (Cross-Derivative)`, `Log10(Metric A3)`. |
| **Palette (`Colormap`)** | Scientific colormaps: `Turbo (High Contrast)`, `Viridis (Perceptual)`, `Cool-Warm (Diverging)`, `Classic Jet`. |
| **Iso-Lines** | Overlays 18 linearly spaced iso-value contour lines across the active scalar field. |
| **Sonic M=1** | Draws the exact **$M = 1.0$ sonic contour line** in bright magenta, delineating supersonic pockets and shock waves. |
| **Vectors** | Overlays scaled Cartesian velocity vectors $(u, v)$ indicating local flow direction and speed. |
| **Grid & Stride** | Toggles O-grid wireframe lines ($\xi$-rings and $\eta$-radials) and sets display decimation (`Stride 1, 2, 4, 8`) for ultra-fine grids. |
| **Airfoil / Wake / Probe** | Toggles the highlighted airfoil surface ($j=0$), wake branch cut ($i=0$), and interactive cursor hover inspector tooltip. |

---

### 4.5 Bottom Analytics & Aerodynamic Outputs

| Tab / Metric | Symbol / Formula | Description |
| :--- | :--- | :--- |
| **Lift Coefficient** | $C_l = C_n \cos\alpha - C_a \sin\alpha$ | Integrated from surface pressure panels around the closed airfoil contour. |
| **Kutta-Joukowski Lift** | $C_{l,\text{KJ}} = \frac{2\Gamma}{q_\infty c}$ | Theoretical circulation lift computed directly from the converged trailing-edge potential jump $\Gamma$. |
| **Drag Coefficient** | $C_d = C_n \sin\alpha + C_a \cos\alpha$ | Pressure + wave drag coefficient (near zero in subsonic potential flow; positive in transonic shock regimes). |
| **Pitching Moment** | $C_{m,c/4}$ | Pitching moment coefficient about the quarter-chord point $(x_{\text{LE}} + 0.25c, 0)$, nose-up positive. |
| **Normal & Axial Force** | $C_n = \frac{1}{c}\oint C_p \, dx, \; C_a = -\frac{1}{c}\oint C_p \, dy$ | Force coefficients in airfoil body axes. |
| **Suction Peak ($C_{p,\min}$)** | $\min_{\text{wall}} C_p$ | Minimum pressure coefficient on the airfoil surface and its chordwise location $x/c$. |
| **Stagnation ($C_{p,\max}$)** | $\max_{\text{wall}} C_p$ | Maximum wall pressure coefficient compared against exact isentropic stagnation theory $C_{p,\text{stag}} = \frac{2}{\gamma M_\infty^2}\left[\left(1 + \frac{\gamma-1}{2}M_\infty^2\right)^{\frac{\gamma}{\gamma-1}} - 1\right]$. |
| **Critical Sonic $C_p^*$** | $C_p(M=1)$ | Pressure coefficient corresponding to local Mach $M = 1.0$: $C_p^* = \frac{2}{\gamma M_\infty^2}\left[\left(\frac{2 + (\gamma-1)M_\infty^2}{\gamma+1}\right)^{\frac{\gamma}{\gamma-1}} - 1\right]$. Plotted as a dashed green line on the $C_p$ graph. |

---

## 5. TOML Configuration File Reference

### Mesh Configuration (`job_config.toml` — `RustyMesher` format)
```toml
meshname = "naca0012_poisson"
airfoil_type = "NACA00XX"      # "NACA00XX" or "Biconvex"
t = 0.12
r_max = 20.0

longitudinal_points = 213      # Must satisfy 4k + 1 (e.g. 93, 213, 713)
normal_points = 65
stretching_factor = 1.04
q = 1.07

omega = 1.3
alpha_l = 0.5
alpha_h = 20.0
max_iter = 1500
conv_criterion = 1e-6

# Options: "ParabolicOnly", "Laplace", or [elliptic_equation.Poisson]
[elliptic_equation.Poisson]
ds_wall = 0.005
a_decay = 1.5
b_decay = 1.5
```

### exported Solver Configuration (`job_files/<run>/job_config.toml` — `RustPotential` format)
```toml
meshname = "mesh/naca0012_fine_mesh.vtk"
jobname = "naca0012_AoA1_M08"
export_metrics = true

Mach_inf = 0.80
AoA = 1.0
Rg = 0.25
gamma = 1.4

max_iterations = 2000
convergence_residual = 1e-5
omega = 1.0
beta_mult = 2.5
alpha_sequence_length = 8
C = 1.5
```

---

## 6. Project Architecture & Adding Future Solvers (Euler / RANS)

```text
RustyFlow/
├── Cargo.toml
├── job_config.toml
├── README.md
└── src/
    ├── main.rs                         # Native window setup & entry point
    ├── app.rs                          # Unified GUI state, sidebar, viewport & plots
    ├── colormap.rs                     # Scientific colormaps (Turbo, Viridis, CoolWarm, Jet)
    ├── core/
    │   ├── grid.rs                     # Canonical 2D structured grid & quality metric evaluator
    │   └── solver_trait.rs             # Solver-agnostic CfdSolution2D, FlowScalarField & SolverFamily
    ├── mesher/                         # Integrated RustyMesher engine
    │   ├── mod.rs                      # Background mesher worker & MesherEvent stream
    │   ├── config.rs                   # Mesher configuration & presets
    │   ├── geometry.rs                 # Airfoil & farfield boundary generators
    │   ├── parabolic.rs                # Nakamura space-marching parabolic generator
    │   ├── elliptic.rs                 # AF1/ADI Laplace & Poisson smoother
    │   └── io.rs                       # VTK, CSV & TOML I/O
    └── solvers/                        # Pluggable CFD Solver engines
        ├── mod.rs                      # Background solver dispatcher & SolverEvent stream
        └── full_potential/             # Integrated RustPotential engine
            ├── config.rs               # Flight & AF2 configuration + presets
            ├── af2.rs                  # AF2 bidiagonal/tridiagonal sweeps & Sherman-Morrison
            ├── solver_core.rs          # Main loop, Kutta circulation & Holst artificial density
            ├── solver_utils.rs         # Metrics, Cartesian reconstruction & CfdSolution2D builder
            └── io.rs                   # VTK solution, Cp CSV & solver_config.txt exporters
```

### How to Add a New Solver (e.g., `Euler2D` or `Rans2D`)
1. Add your solver implementation under `src/solvers/euler_2d/` (or `src/solvers/rans_2d/`).
2. Enable its variant in [`SolverFamily::is_implemented()`](src/core/solver_trait.rs).
3. Populate the solver-agnostic [`CfdSolution2D`](src/core/solver_trait.rs) struct (`mach`, `cp`, `density`, `velocity_u`, `velocity_v`, `potential`, and `aero`) and emit `SolverEvent::Iteration` / `SolverEvent::Finished` from `src/solvers/mod.rs`. All 2D viewport colormaps, iso-contours, sonic lines, vector overlays, hover probes, and $C_p$ plots will work automatically.

---

## 7. Practical Tuning & Troubleshooting Guide

- **Poisson Smoother Divergence**:
  - If `ds_wall` is much smaller than the natural parabolic spacing or `a_decay`/`b_decay` are too small on a coarse mesh, the Steger-Sorenson source terms can cause grid lines to cross. Increase `ds_wall` (e.g., `0.005`), increase `a_decay` / `b_decay` (`1.5–2.5`), or reduce `omega` (`1.1–1.3`).
- **Transonic Lifting Convergence ($\text{Mach}_\infty \ge 0.80, \text{AoA} > 0^\circ$)**:
  - In strong transonic lifting flows, reduce `Rg` (Circulation Relaxation) to `0.20–0.30` and increase `beta_mult` (`2.5–3.5`) or `C` (`1.5–2.0`) to prevent oscillations of the shock wave and trailing-edge Kutta circulation.

---

## 8. Academic Disclaimer & License

> **Notice**: Developed for educational and academic research purposes in Computational Fluid Dynamics (CC-297 at ITA — Instituto Tecnológico de Aeronáutica). Provided **"as is"** under the **MIT License**.

Copyright (c) 2026 Ian Viotti.
