//! 4D multilinear interpolation of gain matrices over the motor-speed hypercube.
//!
//! The gain lookup tables are indexed by the 4 motor speeds (BF motor index
//! order).  At runtime we interpolate across the 16 vertices of the
//! surrounding 4-cube to get smooth gain variation.
//!
//! The interpolation works on raw f32 arrays internally (see
//! [`GainTable::blend`]) but returns typed `MixedUnitMatrix` values — the
//! `from_nalgebra` wrapping is zero-cost (`PhantomData` only).

use crate::gains::{GainStorage, GainTable, REF_FLOATS};
use crate::types::*;

use nalgebra::SMatrix;

/// Interpolated gain set at a single operating point, fully typed.
pub struct InterpolatedGains {
    pub ki: KiMatrix,
    pub kd: KdMatrix,
    pub kp: KpMatrix,
    pub binv: BInvMatrix,
    /// Reference-model terms (see [`crate::gains::REF_FLOATS`]); zero when
    /// the blob has no reference model.
    pub ref_model: [f32; REF_FLOATS],
}

impl InterpolatedGains {
    pub fn zero() -> Self {
        Self {
            ki: KiMatrix::from_nalgebra(SMatrix::zeros()),
            kd: KdMatrix::from_nalgebra(SMatrix::zeros()),
            kp: KpMatrix::from_nalgebra(SMatrix::zeros()),
            binv: BInvMatrix::from_nalgebra(SMatrix::zeros()),
            ref_model: [0.0; REF_FLOATS],
        }
    }
}

fn clampf(x: f32, min: f32, max: f32) -> f32 {
    if x < min { min } else if x > max { max } else { x }
}

/// The 16 hypercube vertices around an operating point: grid-point index
/// and multilinear weight.
struct Vertices {
    flat: [usize; 16],
    weight: [f32; 16],
}

impl Vertices {
    fn new(n: usize, floor_idx: &[usize; 4], frac: &[f32; 4]) -> Self {
        let mut flat = [0usize; 16];
        let mut weight = [0.0f32; 16];
        for vertex in 0..16usize {
            let mut idx = [0usize; 4];
            let mut w = 1.0f32;
            for m in 0..4 {
                let bit = (vertex >> m) & 1;
                idx[m] = floor_idx[m] + bit;
                w *= if bit == 0 { 1.0 - frac[m] } else { frac[m] };
            }
            flat[vertex] = ((idx[0] * n + idx[1]) * n + idx[2]) * n + idx[3];
            weight[vertex] = w;
        }
        Self { flat, weight }
    }

    /// Weighted sum of a table with `W` values per grid point.
    fn interpolate<const W: usize>(&self, table: &GainTable) -> [f32; W] {
        table.blend::<W>(&self.flat, &self.weight)
    }

    /// Interpolate a 4×3 (motor × axis) gain table.
    fn interpolate_4x3(&self, table: &GainTable) -> SMatrix<f32, 4, 3> {
        SMatrix::<f32, 4, 3>::from_row_slice(&self.interpolate::<12>(table))
    }
}

/// Interpolate gains at the given motor speeds (in model units).
///
/// Returns fully typed `MixedUnitMatrix` gain matrices — the wrapping is
/// zero-cost since `MixedUnitMatrix` is `PhantomData` over `SMatrix`.
pub fn interpolate_gains(
    motor_rps: &[f32; 4],
    storage: &GainStorage,
) -> InterpolatedGains {
    let header = &storage.header;
    let n = header.resolution as usize;
    let step = (header.max_rps - header.min_rps) / (n as f32 - 1.0);
    let max_idx = n - 1;

    let mut floor_idx = [0usize; 4];
    let mut frac = [0.0f32; 4];

    for m in 0..4 {
        let clamped = clampf(motor_rps[m], header.min_rps, header.max_rps);
        let continuous_idx = (clamped - header.min_rps) / step;
        let fi = continuous_idx as usize;
        floor_idx[m] = if fi >= max_idx { max_idx - 1 } else { fi };
        frac[m] = continuous_idx - floor_idx[m] as f32;
    }

    let vertices = Vertices::new(n, &floor_idx, &frac);

    InterpolatedGains {
        ki: KiMatrix::from_nalgebra(vertices.interpolate_4x3(&storage.ki)),
        kd: KdMatrix::from_nalgebra(vertices.interpolate_4x3(&storage.kd)),
        kp: KpMatrix::from_nalgebra(vertices.interpolate_4x3(&storage.kp)),
        binv: BInvMatrix::from_nalgebra(vertices.interpolate_4x3(&storage.binv)),
        ref_model: if storage.has_ref_model {
            vertices.interpolate::<REF_FLOATS>(&storage.ref_model)
        } else {
            [0.0; REF_FLOATS]
        },
    }
}
