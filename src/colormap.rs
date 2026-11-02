use egui::Color32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    Turbo,
    Viridis,
    CoolWarm,
    Jet,
}

impl Colormap {
    pub const ALL: [Colormap; 4] = [
        Colormap::Turbo,
        Colormap::Viridis,
        Colormap::CoolWarm,
        Colormap::Jet,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Turbo => "Turbo (High Contrast)",
            Self::Viridis => "Viridis (Perceptual)",
            Self::CoolWarm => "Cool-Warm (Diverging)",
            Self::Jet => "Classic Jet",
        }
    }

    /// Maps a normalized value `t` in `[0.0, 1.0]` to an `egui::Color32`.
    pub fn sample(&self, t: f64) -> Color32 {
        let t = t.clamp(0.0, 1.0);
        let (r, g, b) = match self {
            Self::Turbo => sample_turbo(t),
            Self::Viridis => sample_viridis(t),
            Self::CoolWarm => sample_coolwarm(t),
            Self::Jet => sample_jet(t),
        };
        Color32::from_rgb(
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
        )
    }
}

fn lerp3(a: (f64, f64, f64), b: (f64, f64, f64), t: f64) -> (f64, f64, f64) {
    (
        a.0 + (b.0 - a.0) * t,
        a.1 + (b.1 - a.1) * t,
        a.2 + (b.2 - a.2) * t,
    )
}

fn sample_stops(stops: &[(f64, f64, f64)], t: f64) -> (f64, f64, f64) {
    let n = stops.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    if n == 1 {
        return stops[0];
    }
    let scaled = t.clamp(0.0, 1.0) * ((n - 1) as f64);
    let idx = (scaled.floor() as usize).min(n - 2);
    let frac = scaled - (idx as f64);
    lerp3(stops[idx], stops[idx + 1], frac)
}

fn sample_turbo(t: f64) -> (f64, f64, f64) {
    const STOPS: [(f64, f64, f64); 9] = [
        (0.18995, 0.07176, 0.23217),
        (0.25107, 0.35237, 0.86540),
        (0.15844, 0.63956, 0.92610),
        (0.16830, 0.86210, 0.69290),
        (0.47960, 0.97290, 0.34130),
        (0.82260, 0.91630, 0.20660),
        (0.98590, 0.68720, 0.13690),
        (0.92610, 0.34180, 0.06950),
        (0.47960, 0.01583, 0.01055),
    ];
    sample_stops(&STOPS, t)
}

fn sample_viridis(t: f64) -> (f64, f64, f64) {
    const STOPS: [(f64, f64, f64); 7] = [
        (0.26700, 0.00487, 0.32942),
        (0.28262, 0.14093, 0.45752),
        (0.22974, 0.32236, 0.54571),
        (0.12757, 0.56695, 0.55056),
        (0.20803, 0.71870, 0.47287),
        (0.56550, 0.84454, 0.26213),
        (0.99325, 0.90616, 0.14394),
    ];
    sample_stops(&STOPS, t)
}

fn sample_coolwarm(t: f64) -> (f64, f64, f64) {
    const STOPS: [(f64, f64, f64); 7] = [
        (0.230, 0.299, 0.754),
        (0.404, 0.533, 0.933),
        (0.671, 0.765, 0.969),
        (0.865, 0.865, 0.865),
        (0.965, 0.690, 0.569),
        (0.898, 0.408, 0.325),
        (0.706, 0.016, 0.150),
    ];
    sample_stops(&STOPS, t)
}

fn sample_jet(t: f64) -> (f64, f64, f64) {
    const STOPS: [(f64, f64, f64); 7] = [
        (0.0, 0.0, 0.5),
        (0.0, 0.0, 1.0),
        (0.0, 0.8, 1.0),
        (0.2, 1.0, 0.2),
        (1.0, 0.9, 0.0),
        (1.0, 0.2, 0.0),
        (0.5, 0.0, 0.0),
    ];
    sample_stops(&STOPS, t)
}
