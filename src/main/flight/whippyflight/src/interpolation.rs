//! 4D multilinear interpolation of gain matrices over the motor-speed hypercube.
//!
//! The gain lookup tables are indexed by 4 motor speeds (front-left, back-left,
//! front-right, back-right).  At runtime we interpolate across the 16 vertices
//! of the surrounding 4-cube to get smooth gain variation.

use crate::gains::GainStorage;

/// Interpolated gain set at a single operating point.  These are raw f32
/// arrays (not yet wrapped in MixedUnitMatrix) because the interpolation
/// loop works on scalars; the caller wraps the result in typed matrices.
pub struct InterpolatedGains {
    pub ki: [[f32; 3]; 4],
    pub kd: [[f32; 3]; 4],
    pub kp: [[f32; 3]; 4],
    pub binv: [[f32; 3]; 4],
}

impl InterpolatedGains {
    pub const fn zero() -> Self {
        Self {
            ki: [[0.0; 3]; 4],
            kd: [[0.0; 3]; 4],
            kp: [[0.0; 3]; 4],
            binv: [[0.0; 3]; 4],
        }
    }
}

/// Clamp a value to [min, max].
fn clampf(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

/// Interpolate gains at the given motor speeds (in rot/s).
pub fn interpolate_gains(
    motor_rps: &[f32; 4],
    storage: &GainStorage,
) -> InterpolatedGains {
    let header = &storage.header;
    let n = header.resolution as usize;
    let step = (header.max_rps - header.min_rps) / (n as f32 - 1.0);
    let max_idx = n - 1;

    // For each motor, compute floor index and interpolation fraction
    let mut floor_idx = [0usize; 4];
    let mut frac = [0.0f32; 4];

    for m in 0..4 {
        let clamped = clampf(motor_rps[m], header.min_rps, header.max_rps);
        let continuous_idx = (clamped - header.min_rps) / step;
        let fi = continuous_idx as usize;
        floor_idx[m] = if fi >= max_idx { max_idx - 1 } else { fi };
        frac[m] = continuous_idx - floor_idx[m] as f32;
    }

    let mut result = InterpolatedGains::zero();

    // Iterate over all 16 vertices of the 4-cube (2⁴ corners)
    for vertex in 0..16u8 {
        // Decode which corner: bit i → floor (0) or ceil (1) for motor i
        let idx = [
            floor_idx[0] + ((vertex >> 0) & 1) as usize,
            floor_idx[1] + ((vertex >> 1) & 1) as usize,
            floor_idx[2] + ((vertex >> 2) & 1) as usize,
            floor_idx[3] + ((vertex >> 3) & 1) as usize,
        ];

        // Multilinear weight: product of (1 - frac) or frac for each axis
        let mut weight = 1.0f32;
        for m in 0..4 {
            if (vertex >> m) & 1 == 0 {
                weight *= 1.0 - frac[m];
            } else {
                weight *= frac[m];
            }
        }

        // Flat index into the gain tables: [fl][bl][fr][br] row-major
        let flat = ((idx[0] * n + idx[1]) * n + idx[2]) * n + idx[3];
        let base = flat * 12; // 4 motors × 3 axes

        // Accumulate weighted gains
        for motor in 0..4 {
            for axis in 0..3 {
                let offset = base + motor * 3 + axis;
                result.ki[motor][axis] += weight * storage.ki[offset];
                result.kd[motor][axis] += weight * storage.kd[offset];
                result.kp[motor][axis] += weight * storage.kp[offset];
                result.binv[motor][axis] += weight * storage.binv[offset];
            }
        }
    }

    result
}
