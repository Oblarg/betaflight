//! 4D multilinear interpolation of gain matrices over the motor-speed hypercube.
//!
//! The gain lookup tables are indexed by the 4 motor speeds (BF motor index
//! order).  At runtime we interpolate across the 16 vertices of the
//! surrounding 4-cube to get smooth gain variation.
//!
//! Each table holds quantities of its gain's unit (see [`GainTable::blend`]);
//! the interpolated entries are gauged into typed `MixedUnitMatrix` values,
//! which checks at compile time that every table feeds the matrix of its unit.

use crate::gains::{
    ref_slots_from_motors, GainStorage, GainTable, ReferenceModel, TableEntry, REF_FLOATS, REF_MOTOR_SLOTS,
};
use crate::types::*;

use nalgebra::SMatrix;
use whippyalgebra::nalgebra::UniformUnitMatrix;

/// Interpolated gain set at a single operating point, fully typed.
pub struct InterpolatedGains {
    pub ki: KiMatrix,
    pub kd: KdMatrix,
    pub kp: KpMatrix,
    pub binv: BInvMatrix,
    /// Reference-model terms; zero when the blob has no reference model.
    pub ref_model: ReferenceModel,
}

impl InterpolatedGains {
    pub fn zero() -> Self {
        Self {
            ki: KiMatrix::from_nalgebra(SMatrix::zeros()),
            kd: KdMatrix::from_nalgebra(SMatrix::zeros()),
            kp: KpMatrix::from_nalgebra(SMatrix::zeros()),
            binv: BInvMatrix::from_nalgebra(SMatrix::zeros()),
            ref_model: ReferenceModel::from_slots(&[0.0; REF_FLOATS]),
        }
    }
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
    fn interpolate<E: TableEntry, const W: usize>(&self, table: &GainTable<E>) -> [E; W] {
        table.blend::<W>(&self.flat, &self.weight)
    }
}

/// A 4×3 (motor × axis) table's interpolated entries as a uniform matrix,
/// ready to gauge into the gain's mixed-unit type.
macro_rules! interpolate_4x3 {
    ($vertices:expr, $table:expr) => {{
        let entries = $vertices.interpolate::<_, 12>(&$table);
        UniformUnitMatrix::<_, SMatrix<f32, 4, 3>>::from_fn(|i, j| entries[3 * i + j]).gauge()
    }};
}

/// Interpolate gains at the given motor speeds (shaft rev/s).
pub fn interpolate_gains(
    motor_rps: &[RotPerSec; 4],
    storage: &GainStorage,
) -> InterpolatedGains {
    let header = &storage.header;
    let n = header.resolution as usize;
    let step = (header.max_rps - header.min_rps) / (n as f32 - 1.0);
    let max_idx = n - 1;

    let mut floor_idx = [0usize; 4];
    let mut frac = [0.0f32; 4];

    for m in 0..4 {
        let clamped = motor_rps[m].clamp(header.min_rps, header.max_rps);
        let continuous_idx: f32 = ((clamped - header.min_rps) / step).into();
        let fi = continuous_idx as usize;
        floor_idx[m] = if fi >= max_idx { max_idx - 1 } else { fi };
        frac[m] = continuous_idx - floor_idx[m] as f32;
    }

    let vertices = Vertices::new(n, &floor_idx, &frac);

    InterpolatedGains {
        ki: interpolate_4x3!(vertices, storage.ki),
        kd: interpolate_4x3!(vertices, storage.kd),
        kp: interpolate_4x3!(vertices, storage.kp),
        binv: interpolate_4x3!(vertices, storage.binv),
        ref_model: ReferenceModel::from_slots(&if !storage.has_ref_model {
            [0.0; REF_FLOATS]
        } else if storage.ref_per_motor {
            per_motor_ref(storage, &floor_idx, &frac)
        } else {
            vertices.interpolate::<f32, REF_FLOATS>(&storage.ref_model)
        }),
    }
}

/// The v8 reference model: each motor's slots interpolated along its own speed.
fn per_motor_ref(storage: &GainStorage, floor_idx: &[usize; 4], frac: &[f32; 4]) -> [f32; REF_FLOATS] {
    let motors = core::array::from_fn(|m| {
        storage.ref_model.blend_slots::<REF_MOTOR_SLOTS>(
            4 * REF_MOTOR_SLOTS,
            m * REF_MOTOR_SLOTS,
            &[floor_idx[m], floor_idx[m] + 1],
            &[1.0 - frac[m], frac[m]],
        )
    });
    ref_slots_from_motors(&storage.ref_const, &motors)
}
