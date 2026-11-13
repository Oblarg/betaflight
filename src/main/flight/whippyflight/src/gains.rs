//! Gain data storage and binary format parsing.
//!
//! The gain binary format (`.wfg`) starts with a common header:
//!
//! | Offset | Size    | Field           | Description                          |
//! |--------|---------|-----------------|--------------------------------------|
//! | 0x00   | 4       | magic           | `0x4C515247` ("LQRG")                |
//! | 0x04   | 2       | version         | Format version (1–7)                 |
//! | 0x06   | 1       | resolution      | N (grid points per motor axis)       |
//! | 0x07   | 1       | ff_horizon (v3) | Feedforward horizon h (ticks)        |
//! | 0x08   | 4       | min_rps         | `f32` — minimum motor RPS            |
//! | 0x0C   | 4       | max_rps         | `f32` — maximum motor RPS            |
//! | 0x10   | 4       | dt              | `f32` — loop period (s)              |
//! | 0x14   | 4       | integral_tau    | `f32` — integral time constant       |
//! | 0x18   | 4       | derivative_tau  | `f32` — derivative smoothing time    |
//! | 0x1C   | 4       | total_size      | `u32` — total blob size in bytes     |
//!
//! Versions 1–4 then store the tables as f32:
//!
//! | Offset | Size    | Field           | Description                          |
//! |--------|---------|-----------------|--------------------------------------|
//! | 0x20   | var     | ki_table        | `f32[N⁴×4×3]`                        |
//! | ...    | var     | kd_table        | `f32[N⁴×4×3]`                        |
//! | ...    | var     | kp_table        | `f32[N⁴×4×3]`                        |
//! | ...    | var     | binv_table      | `f32[N⁴×4×3]`                        |
//! | ...    | var     | ref_table (v3)  | `f32[N⁴×REF_FLOATS]` reference model |
//! | ...    | 64      | affine (v2+)    | `f32[4×4]` row-major linearizer G    |
//! | ...    | 8       | motor (v4)      | `f32[2]` motor drag [k_ω, k_ω²]      |
//! | EOF−4  | 4       | crc32           | CRC32 over bytes `0..EOF−4`          |
//!
//! Version 5 quantizes the tables to `u16` to halve their size (they are
//! copied to RAM, which is scarce on F411-class targets).  Each table slot
//! (one of the W floats per grid point) is mapped linearly over its own
//! range across the grid, value = lo + step·q:
//!
//! | Offset | Size    | Field           | Description                          |
//! |--------|---------|-----------------|--------------------------------------|
//! | 0x20   | var     | quant params    | per table (ki, kd, kp, binv, ref):   |
//! |        |         |                 | `f32[W]` lo, then `f32[W]` step      |
//! | ...    | 64      | affine          | `f32[4×4]` row-major linearizer G    |
//! | ...    | 8       | motor           | `f32[2]` motor drag [k_ω, k_ω²]      |
//! | ...    | 16      | balance (v6)    | `f32[4]` hover speed ratio per motor |
//! | ...    | 20      | drag (v7)       | `f32[5]` airframe drag, see below    |
//! | ...    | 8       | target (v7)     | `f32[2]` setpoint target limits      |
//! | ...    | 8       | hover (v7)      | `f32[2]` hover rps, integral limit V |
//! | ...    | var     | table data      | per table: `u16[N⁴×W]` q             |
//! | EOF−4  | 4       | crc32           | CRC32 over bytes `0..EOF−4`          |
//!
//! W is 12 for the gain tables and [`REF_FLOATS`] for the reference model;
//! N⁴·W is always even, so every section stays 4-byte aligned.
//!
//! Version 8 stores the reference model per motor.  Each of its blocks
//! depends on at most one motor's speed (the rate and motor-speed dynamics
//! are diagonal and the plant is block-triangular, so column m of every
//! rate ← speed or rate ← voltage block, and motor m's own decay and
//! gain, follow motor m alone), so it is a 1D table over each motor's speed
//! rather than an N⁴ one.  Multilinear interpolation of such a table is
//! exactly the 1D interpolation, so nothing is approximated.  The
//! quantization block's reference entry becomes:
//!
//! | Offset | Size    | Field           | Description                          |
//! |--------|---------|-----------------|--------------------------------------|
//! | ...    | 24      | ref constants   | `f32[6]` ad_rr, fh_rr (speed-free)   |
//! | ...    | 352     | ref quant       | `f32[44]` lo, then `f32[44]` step    |
//!
//! and its table `u16[N × 4 × REF_MOTOR_SLOTS]`, indexed
//! `[grid index · 4 + motor] · REF_MOTOR_SLOTS + slot` (see
//! [`REF_MOTOR_SLOTS`]).  At N = 4 the blob is 25 848 bytes rather than
//! ≈ 52 KB.
//!
//! Version 1 blobs have no affine block; the linearizer is then zero.
//! Versions 1–2 have no reference model; `binv` is then a one-tick B⁻¹ and
//! the controller falls back to `u_ff = B⁻¹·Δr`.  In v3+ `binv` is the
//! h-tick inverse G_h⁺ (see [`crate::controller`]).  Versions 1–3 have no
//! motor block; the idle voltage floor is then 0 V.  Versions 1–5 have no
//! balance block; the collective is then spread uniformly.  Versions 1–6
//! have no airframe drag block (the mode is then off), no target limits
//! (the setpoint estimator's defaults then apply) and no hover speed (the
//! controller then skips takeoff/landing detection and the trim-sized
//! integral bound).

use core::ops::{Add, Mul};

use crate::types::*;
use nalgebra::SMatrix;
use whippyalgebra::nalgebra::UniformUnitMatrix;
use whippyunits::qty;

/// Magic bytes: "LQRG" in little-endian.
const GAIN_MAGIC: u32 = 0x4C51_5247;

/// Format version without the affine linearizer block.
const GAIN_VERSION_V1: u16 = 1;

/// Format version with the affine linearizer but no reference model.
const GAIN_VERSION_V2: u16 = 2;

/// Format version with the feedforward reference model.
const GAIN_VERSION_V3: u16 = 3;

/// Format version that adds the motor drag coefficients.
const GAIN_VERSION_V4: u16 = 4;

/// Format version with u16-quantized tables.
const GAIN_VERSION_V5: u16 = 5;

/// Format version that adds the thrust balance.
const GAIN_VERSION_V6: u16 = 6;

/// Format version that adds the airframe drag and target-limit blocks.
const GAIN_VERSION_V7: u16 = 7;

/// Current format version: the reference model per motor.
const GAIN_VERSION: u16 = 8;

/// Floats per grid point in the reference-model table.
///
/// The blocks are the pieces of the discrete (ZOH) model in deviation
/// coordinates, δx = [r (3 rates); δω (4 motor speeds)], u = 4 voltages:
///
/// ```text
///   One tick:   δx[k+1] = Ad·δx[k] + Bd·u[k]
///
///   ┌  r'  ┐   ┌ diag(ad_rr)    ad_rw     ┐ ┌ r  ┐   ┌   bd_r     ┐
///   │      │ = │   (3×3)        (3×4)     │ │    │ + │   (3×4)    │ u
///   │  δω' │   │     0       diag(ad_ww)  │ │ δω │   │ diag(bd_w) │
///   └      ┘   └   (4×3)        (4×4)     ┘ └    ┘   └   (4×4)    ┘
///
///   h ticks, free response (attitude rows of Ad^h):
///
///   F_h = [ diag(fh_rr)   fh_rw ]
///           (3×3)        (3×4)
/// ```
///
/// The rate→rate and motor→motor blocks are diagonal (rates and motors
/// each decay independently), and motor speeds don't depend on rates, so
/// the lower-left block is zero.  Only the non-zero blocks are stored,
/// diagonals as vectors, in BF axis order with motors by index:
///
/// | Offset | Block   | Shape | Meaning                                 |
/// |--------|---------|-------|-----------------------------------------|
/// | 0      | `ad_rr` | 3     | one-tick rate decay e^{ρ·dt}            |
/// | 3      | `ad_rw` | 3×4   | one-tick rate ← motor speed             |
/// | 15     | `ad_ww` | 4     | one-tick motor-speed decay              |
/// | 19     | `bd_r`  | 3×4   | one-tick rate ← voltage                 |
/// | 31     | `bd_w`  | 4     | one-tick motor speed ← voltage          |
/// | 35     | `fh_rr` | 3     | h-tick free response, rate ← rate       |
/// | 38     | `fh_rw` | 3×4   | h-tick free response, rate ← motor speed|
pub const REF_FLOATS: usize = 50;

/// Speed-independent reference-model floats (v8): `ad_rr` then `fh_rr`.
pub const REF_CONST_FLOATS: usize = 6;

/// Reference-model values per motor per grid speed (v8), from motor m's
/// columns of the blocks in [`REF_FLOATS`]:
///
/// | Slot | Value          |
/// |------|----------------|
/// | 0–2  | `ad_rw[0..3][m]` |
/// | 3    | `ad_ww[m]`     |
/// | 4–6  | `bd_r[0..3][m]` |
/// | 7    | `bd_w[m]`      |
/// | 8–10 | `fh_rw[0..3][m]` |
pub const REF_MOTOR_SLOTS: usize = 11;

/// Width of the v8 reference table's rows: all motors at one grid speed.
const REF_MOTOR_WIDTH: usize = 4 * REF_MOTOR_SLOTS;

/// Assemble the [`REF_FLOATS`] slots from the v8 speed-free values and each
/// motor's [`REF_MOTOR_SLOTS`] at its own speed.
pub fn ref_slots_from_motors(
    constant: &[f32; REF_CONST_FLOATS],
    motors: &[[f32; REF_MOTOR_SLOTS]; 4],
) -> [f32; REF_FLOATS] {
    let mut r = [0.0f32; REF_FLOATS];
    r[0..3].copy_from_slice(&constant[0..3]);
    r[35..38].copy_from_slice(&constant[3..6]);
    for (m, s) in motors.iter().enumerate() {
        for i in 0..3 {
            r[3 + 4 * i + m] = s[i];
            r[19 + 4 * i + m] = s[4 + i];
            r[38 + 4 * i + m] = s[8 + i];
        }
        r[15 + m] = s[3];
        r[31 + m] = s[7];
    }
    r
}

/// The reference-model blocks of [`REF_FLOATS`] at one operating point,
/// with their units.  Motor speeds and rates are both rot/s, so the
/// rate ← speed blocks and every decay are dimensionless.
#[derive(Clone, Copy)]
pub struct ReferenceModel {
    pub ad_rr: [f32; 3],
    pub ad_rw: RateFromSpeedMatrix,
    pub ad_ww: [f32; 4],
    pub bd_r: RateFromVoltMatrix,
    pub bd_w: [RpsPerVolt; 4],
    pub fh_rr: [f32; 3],
    pub fh_rw: RateFromSpeedMatrix,
}

impl ReferenceModel {
    /// Decode the table slots (see [`REF_FLOATS`] for the layout).
    pub fn from_slots(r: &[f32; REF_FLOATS]) -> Self {
        let rate_from_speed = |o: usize| {
            UniformUnitMatrix::<_, SMatrix<f32, 3, 4>>::from_fn(|i, j| {
                <qty!(1, f32)>::new(r[o + 4 * i + j])
            })
            .gauge::<AttitudeAxes, MotorSpeedAxes>()
        };
        Self {
            ad_rr: [r[0], r[1], r[2]],
            ad_rw: rate_from_speed(3),
            ad_ww: [r[15], r[16], r[17], r[18]],
            bd_r: UniformUnitMatrix::<_, SMatrix<f32, 3, 4>>::from_fn(|i, j| {
                RpsPerVolt::new(r[19 + 4 * i + j])
            })
            .gauge::<AttitudeAxes, MotorAxes>(),
            bd_w: [31, 32, 33, 34].map(|o| RpsPerVolt::new(r[o])),
            fh_rr: [r[35], r[36], r[37]],
            fh_rw: rate_from_speed(38),
        }
    }
}

/// Floats in the affine linearizer block.
pub const AFFINE_FLOATS: usize = 16;

/// Floats in the motor block.
pub const MOTOR_FLOATS: usize = 2;

/// Steady-state motor drag, V = k_ω·ω + k_ω²·ω|ω| (main.R's motor fit
/// without the k_α·ω̇ term).
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct MotorDrag {
    pub k_w: VoltPerRps,
    pub k_w2: VoltPerRpsSq,
}

impl MotorDrag {
    pub const ZERO: Self = Self { k_w: VoltPerRps::new(0.0), k_w2: VoltPerRpsSq::new(0.0) };
}

/// Floats in the thrust-balance block.
pub const BALANCE_FLOATS: usize = 4;

/// Floats in the airframe drag block (v7):
/// `[lp_hz, roll←ax, roll←ay, pitch←ax, pitch←ay]`.
///
/// Rotor drag on tall frames acts well above the CG, so airspeed torques
/// pitch and roll.  In flight the accelerometer reads thrust (body z) plus
/// that drag, so its body x/y components, one-pole low-passed at `lp_hz`,
/// measure it; the coefficients (rot/m, i.e. rot/s² per m/s², BF axes) give the angular
/// acceleration it causes.  `lp_hz = 0` turns the mode off.
pub const DRAG_FLOATS: usize = 5;

/// The airframe drag block, see [`DRAG_FLOATS`]: angular acceleration per
/// unit of body specific force.
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct AirframeDrag {
    pub lp: Hertz,
    /// Roll angular acceleration per body [x, y] specific force.
    pub roll: [RotPerMeter; 2],
    /// Pitch angular acceleration per body [x, y] specific force.
    pub pitch: [RotPerMeter; 2],
}

impl AirframeDrag {
    pub const OFF: Self = Self {
        lp: Hertz::new(0.0),
        roll: [RotPerMeter::new(0.0); 2],
        pitch: [RotPerMeter::new(0.0); 2],
    };
}

/// Floats in the target-limit block (v7): the setpoint estimator's
/// `[slew (rot/s²), jerk (rot/s³)]` bounds on the feedforward target, set
/// from the airframe's pitch/roll authority (see
/// [`crate::setpoint`]).  A value ≤ 0 selects the estimator's default.
pub const TARGET_FLOATS: usize = 2;

/// The target-limit block, see [`TARGET_FLOATS`].
#[derive(Clone, Copy, PartialEq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct TargetLimits {
    pub slew: RotPerSecSq,
    pub jerk: RotPerSecCu,
}

impl TargetLimits {
    pub const DEFAULT: Self = Self { slew: RotPerSecSq::new(0.0), jerk: RotPerSecCu::new(0.0) };
}

/// Floats in the hover block (v7): [hover rps, integral limit V] — the
/// fitted airframe's median hover motor speed (shaft rev/s), for
/// takeoff/landing detection, and the per-motor integral voltage that trims
/// a generous CG offset at hover (gen_wfg `build_integral_limit`).  0 when
/// unknown.
pub const HOVER_FLOATS: usize = 2;

/// Maximum supported resolution (N).  This bounds the format, not the
/// target: the blob is copied to RAM, and at N=5 a v5 blob is ≈ 123 KB.
pub const MAX_RESOLUTION: usize = 5;

/// Floats per grid point in each of the 4 gain tables (4 motors × 3 axes).
const GAIN_SLOTS: usize = 12;

/// Total floats per gain table at a given resolution: N⁴ × 4 motors × 3 axes.
const fn table_floats(n: usize) -> usize {
    n * n * n * n * GAIN_SLOTS
}

/// Values in the reference-model table at a given resolution.
const fn ref_floats(n: usize, per_motor: bool) -> usize {
    if per_motor { n * REF_MOTOR_WIDTH } else { n * n * n * n * REF_FLOATS }
}

/// Binary header size in bytes.
const HEADER_SIZE: usize = 0x20;

// Tables are read in place as native f32/u16, which must match the blob's
// little-endian encoding.
#[cfg(target_endian = "big")]
compile_error!("the .wfg tables are little-endian and are read in place");

/// A value a lookup table can hold: a quantity of the table's unit, or a
/// bare `f32` for the reference-model table, whose slots differ in unit
/// (decoded by [`ReferenceModel::from_slots`]).
pub trait TableEntry: bytemuck::Pod + Default + Add<Output = Self> + Mul<f32, Output = Self> {}

impl<T: bytemuck::Pod + Default + Add<Output = T> + Mul<f32, Output = T>> TableEntry for T {}

/// Storage of one lookup table, read in place from the blob.
#[derive(Clone, Copy)]
enum TableData<E: 'static> {
    F32(&'static [E]),
    /// value[i] = lo[i % W] + step[i % W] · q[i]
    U16 { q: &'static [u16], lo: &'static [E], step: &'static [E] },
}

/// A lookup table with `width` values per grid point, indexed
/// [grid point] · width + slot.
#[derive(Clone, Copy)]
pub struct GainTable<E: 'static> {
    data: TableData<E>,
}

impl<E: TableEntry> GainTable<E> {
    pub const EMPTY: Self = Self { data: TableData::F32(&[]) };

    pub fn len(&self) -> usize {
        match self.data {
            TableData::F32(v) => v.len(),
            TableData::U16 { q, .. } => q.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Start of the table's per-point data within the blob.
    pub fn data_ptr(&self) -> *const u8 {
        match self.data {
            TableData::F32(v) => v.as_ptr() as *const u8,
            TableData::U16 { q, .. } => q.as_ptr() as *const u8,
        }
    }

    /// Decoded value at flat index `i`.
    pub fn get(&self, i: usize) -> E {
        match self.data {
            TableData::F32(v) => v[i],
            TableData::U16 { q, lo, step } => {
                let slot = i % lo.len();
                lo[slot] + step[slot] * q[i] as f32
            }
        }
    }

    /// Weighted sum over grid points of the `W` values at each, i.e.
    /// Σ_v weight[v] · table[point[v]·W + i].  The weights must sum to 1:
    /// quantized tables sum the raw codes and apply lo/step once per slot.
    pub fn blend<const W: usize>(&self, point: &[usize], weight: &[f32]) -> [E; W] {
        self.blend_slots::<W>(W, 0, point, weight)
    }

    /// [`Self::blend`] of the `K` slots from `first` in rows of `width`.
    pub fn blend_slots<const K: usize>(&self, width: usize, first: usize, point: &[usize], weight: &[f32]) -> [E; K] {
        match self.data {
            TableData::F32(v) => {
                let mut out = [E::default(); K];
                for (&p, &w) in point.iter().zip(weight) {
                    let row = &v[p * width + first..p * width + first + K];
                    for (o, &x) in out.iter_mut().zip(row) {
                        *o = *o + x * w;
                    }
                }
                out
            }
            TableData::U16 { q, lo, step } => {
                let mut code = [0.0f32; K];
                for (&p, &w) in point.iter().zip(weight) {
                    let row = &q[p * width + first..p * width + first + K];
                    for (o, &x) in code.iter_mut().zip(row) {
                        *o += w * x as f32;
                    }
                }
                core::array::from_fn(|i| lo[first + i] + step[first + i] * code[i])
            }
        }
    }
}

/// Parsed gain header.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct GainHeader {
    pub resolution: u8,
    pub min_rps: RotPerSec,
    pub max_rps: RotPerSec,
    pub dt: Seconds,
    pub integral_tau: Seconds,
    pub derivative_tau: Seconds,
    /// Feedforward inversion horizon in ticks (v3; 0 when absent).
    pub ff_horizon: u8,
}

/// Validated view of a gain blob.  The lookup tables are slices into the
/// blob itself (the glue's RAM copy on target), so the blob must outlive the
/// controller and be 4-byte aligned; see [`load_gains`].
pub struct GainStorage {
    pub header: GainHeader,
    pub loaded: bool,

    // The 4 gain tables, each N⁴ × 12 values.  Indexed as
    // [((m0·N + m1)·N + m2)·N + m3] * 12 + motor * 3 + axis
    pub ki: GainTable<VoltPerRot>,
    pub kd: GainTable<VoltPerRpsPerSec>,
    pub kp: GainTable<VoltPerRps>,
    pub binv: GainTable<VoltPerRps>,

    /// Reference-model table: v3–v7 [grid point] * REF_FLOATS + offset,
    /// v8 per motor (see [`REF_MOTOR_SLOTS`]); empty when the blob has none
    /// (v1–v2).
    pub ref_model: GainTable<f32>,
    /// Whether `ref_model` was present in the blob (v3+).
    pub has_ref_model: bool,
    /// Whether `ref_model` is per motor (v8).
    pub ref_per_motor: bool,
    /// The reference model's speed-free values (v8), see [`REF_CONST_FLOATS`].
    pub ref_const: [f32; REF_CONST_FLOATS],

    /// Affine linearizer G, row-major [motor][signed-square motor]:
    /// u_aff = G · ω|ω|.  Constant over the grid.
    pub affine: [VoltPerRpsSq; AFFINE_FLOATS],

    /// Steady-state motor drag.  Zero when the blob has none (v1–v3).
    pub motor_drag: MotorDrag,

    /// Per-motor hover speed ratio for a torque-free hover, geometric mean 1
    /// (1/√(d_fb·d_lr) from the arm lengths).  Uniform before v6.
    pub thrust_balance: [f32; BALANCE_FLOATS],

    /// Airframe drag model, see [`DRAG_FLOATS`].  Zero (off) before v7.
    pub airframe_drag: AirframeDrag,

    /// Setpoint target limits, see [`TARGET_FLOATS`].  Zero (defaults)
    /// before v7.
    pub target_limits: TargetLimits,

    /// Hover motor speed (shaft rev/s), see [`HOVER_FLOATS`].  0 before v7.
    pub hover_rps: RotPerSec,
    /// Per-motor integral voltage bound, see [`HOVER_FLOATS`].  0 before v7.
    pub integral_limit_v: Volt,
}

impl GainStorage {
    pub const fn new() -> Self {
        Self {
            header: GainHeader {
                resolution: 0,
                min_rps: RotPerSec::new(0.0),
                max_rps: RotPerSec::new(0.0),
                dt: Seconds::new(0.0),
                integral_tau: Seconds::new(0.0),
                derivative_tau: Seconds::new(0.0),
                ff_horizon: 0,
            },
            loaded: false,
            ki: GainTable::EMPTY,
            kd: GainTable::EMPTY,
            kp: GainTable::EMPTY,
            binv: GainTable::EMPTY,
            ref_model: GainTable::EMPTY,
            has_ref_model: false,
            ref_per_motor: false,
            ref_const: [0.0; REF_CONST_FLOATS],
            affine: [VoltPerRpsSq::new(0.0); AFFINE_FLOATS],
            motor_drag: MotorDrag::ZERO,
            thrust_balance: [1.0; BALANCE_FLOATS],
            airframe_drag: AirframeDrag::OFF,
            target_limits: TargetLimits::DEFAULT,
            hover_rps: RotPerSec::new(0.0),
            integral_limit_v: Volt::new(0.0),
        }
    }

    /// The affine linearizer G as a matrix.
    pub fn affine_matrix(&self) -> AffineMatrix {
        UniformUnitMatrix::<_, SMatrix<f32, 4, 4>>::from_fn(|i, j| self.affine[4 * i + j])
            .gauge::<MotorAxes, MotorSpeedSqAxes>()
    }
}

/// Sequential typed reader over an aligned blob.  Tables are viewed in
/// place: blob alignment is checked in `load_gains` and every section
/// offset is a multiple of 4.
struct Cursor {
    data: &'static [u8],
    offset: usize,
}

impl Cursor {
    fn take<E: bytemuck::Pod>(&mut self, count: usize) -> &'static [E] {
        let bytes = &self.data[self.offset..self.offset + count * core::mem::size_of::<E>()];
        self.offset += bytes.len();
        debug_assert!(bytes.as_ptr() as usize % core::mem::align_of::<E>() == 0);
        // SAFETY: `bytes` is in bounds and aligned (see the struct docs), and
        // `E: Pod` accepts any bit pattern.  Not `bytemuck::cast_slice`: its
        // formatting panic path pulls unwinding symbols into the SITL link.
        unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<E>(), count) }
    }

    fn one<E: bytemuck::Pod>(&mut self) -> E {
        self.take::<E>(1)[0]
    }

    /// A quantized table's per-slot (lo, step).
    fn quant<E: bytemuck::Pod>(&mut self, width: usize) -> (&'static [E], &'static [E]) {
        (self.take(width), self.take(width))
    }
}

/// Read a little-endian `u32` from a byte slice at the given offset.
fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

/// Read a little-endian `u16` from a byte slice at the given offset.
fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Read a value stored as a little-endian `f32` at the given offset.
fn read_f32<E: bytemuck::Pod>(data: &[u8], offset: usize) -> E {
    bytemuck::pod_read_unaligned(&data[offset..offset + 4])
}

/// Simple CRC32 (ISO 3309 / ITU-T V.42 polynomial).
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Parse and validate a gain binary blob, pointing `storage` at its tables.
///
/// The tables are not copied: `data` must stay valid and unmodified for the
/// life of the program (on target, the glue's static RAM copy) and must be
/// 4-byte aligned.
pub fn load_gains(data: &'static [u8], storage: &mut GainStorage) -> GainLoadResult {
    storage.loaded = false;

    // Minimum size: header + CRC
    if data.len() < HEADER_SIZE + 4 {
        return GainLoadResult::TooShort;
    }

    if data.as_ptr() as usize % core::mem::align_of::<f32>() != 0 {
        return GainLoadResult::Misaligned;
    }

    // Validate magic
    let magic = read_u32(data, 0x00);
    if magic != GAIN_MAGIC {
        return GainLoadResult::BadMagic;
    }

    // Validate version
    let version = read_u16(data, 0x04);
    if !(GAIN_VERSION_V1..=GAIN_VERSION).contains(&version) {
        return GainLoadResult::BadVersion;
    }
    let has_affine = version >= GAIN_VERSION_V2;
    let has_ref_model = version >= GAIN_VERSION_V3;
    let has_motor = version >= GAIN_VERSION_V4;
    let quantized = version >= GAIN_VERSION_V5;
    let has_balance = version >= GAIN_VERSION_V6;
    let has_drag = version >= GAIN_VERSION_V7;
    let per_motor = version >= GAIN_VERSION;

    let resolution = data[0x06] as usize;
    if resolution > MAX_RESOLUTION || resolution == 0 {
        return GainLoadResult::ResolutionTooLarge;
    }

    let floats_per_table = table_floats(resolution);
    let affine_floats = if has_affine { AFFINE_FLOATS } else { 0 };
    let ref_count = if has_ref_model { ref_floats(resolution, per_motor) } else { 0 };
    let motor_floats = if has_motor { MOTOR_FLOATS } else { 0 }
        + if has_balance { BALANCE_FLOATS } else { 0 }
        + if has_drag { DRAG_FLOATS + TARGET_FLOATS + HOVER_FLOATS } else { 0 };
    let table_values = 4 * floats_per_table + ref_count;
    let body_bytes = if quantized {
        let quant_floats = 2 * 4 * GAIN_SLOTS
            + if per_motor { REF_CONST_FLOATS + 2 * REF_MOTOR_WIDTH } else { 2 * REF_FLOATS };
        (quant_floats + affine_floats + motor_floats) * 4 + table_values * 2
    } else {
        (table_values + affine_floats + motor_floats) * 4
    };

    // Validate total size
    let total_size = read_u32(data, 0x1C) as usize;
    if data.len() < total_size || total_size < HEADER_SIZE + body_bytes + 4 {
        return GainLoadResult::TooShort;
    }

    // Validate CRC32 over everything except the last 4 bytes
    let payload = &data[..total_size - 4];
    let expected_crc = read_u32(data, total_size - 4);
    let actual_crc = crc32(payload);
    if actual_crc != expected_crc {
        return GainLoadResult::BadChecksum;
    }

    // Parse header
    storage.header = GainHeader {
        resolution: resolution as u8,
        min_rps: read_f32(data, 0x08),
        max_rps: read_f32(data, 0x0C),
        dt: read_f32(data, 0x10),
        integral_tau: read_f32(data, 0x14),
        derivative_tau: read_f32(data, 0x18),
        ff_horizon: if has_ref_model { data[0x07] } else { 0 },
    };

    // Point at the tables in place
    let mut cur = Cursor { data, offset: HEADER_SIZE };
    if quantized {
        let ki_q = cur.quant(GAIN_SLOTS);
        let kd_q = cur.quant(GAIN_SLOTS);
        let kp_q = cur.quant(GAIN_SLOTS);
        let binv_q = cur.quant(GAIN_SLOTS);
        storage.ref_const = [0.0; REF_CONST_FLOATS];
        if per_motor {
            storage.ref_const.copy_from_slice(cur.take(REF_CONST_FLOATS));
        }
        let ref_q = cur.quant(if per_motor { REF_MOTOR_WIDTH } else { REF_FLOATS });
        storage.affine.copy_from_slice(cur.take(AFFINE_FLOATS));
        storage.motor_drag = MotorDrag { k_w: cur.one(), k_w2: cur.one() };
        storage.thrust_balance = [1.0; BALANCE_FLOATS];
        if has_balance {
            storage.thrust_balance.copy_from_slice(cur.take(BALANCE_FLOATS));
        }
        storage.airframe_drag = AirframeDrag::OFF;
        storage.target_limits = TargetLimits::DEFAULT;
        storage.hover_rps = RotPerSec::new(0.0);
        storage.integral_limit_v = Volt::new(0.0);
        if has_drag {
            storage.airframe_drag = AirframeDrag {
                lp: cur.one(),
                roll: [cur.one(), cur.one()],
                pitch: [cur.one(), cur.one()],
            };
            storage.target_limits = TargetLimits { slew: cur.one(), jerk: cur.one() };
            storage.hover_rps = cur.one();
            storage.integral_limit_v = cur.one();
        }
        fn table<E: TableEntry>(cur: &mut Cursor, (lo, step): (&'static [E], &'static [E]), count: usize) -> GainTable<E> {
            GainTable { data: TableData::U16 { q: cur.take(count), lo, step } }
        }
        storage.ki = table(&mut cur, ki_q, floats_per_table);
        storage.kd = table(&mut cur, kd_q, floats_per_table);
        storage.kp = table(&mut cur, kp_q, floats_per_table);
        storage.binv = table(&mut cur, binv_q, floats_per_table);
        storage.ref_model = table(&mut cur, ref_q, ref_count);
    } else {
        fn table<E: TableEntry>(cur: &mut Cursor, count: usize) -> GainTable<E> {
            GainTable { data: TableData::F32(cur.take(count)) }
        }
        storage.ki = table(&mut cur, floats_per_table);
        storage.kd = table(&mut cur, floats_per_table);
        storage.kp = table(&mut cur, floats_per_table);
        storage.binv = table(&mut cur, floats_per_table);
        storage.ref_model = table(&mut cur, ref_count);
        storage.ref_const = [0.0; REF_CONST_FLOATS];
        storage.affine = [VoltPerRpsSq::new(0.0); AFFINE_FLOATS];
        if has_affine {
            storage.affine.copy_from_slice(cur.take(AFFINE_FLOATS));
        }
        storage.motor_drag = MotorDrag::ZERO;
        if has_motor {
            storage.motor_drag = MotorDrag { k_w: cur.one(), k_w2: cur.one() };
        }
        storage.thrust_balance = [1.0; BALANCE_FLOATS];
        storage.airframe_drag = AirframeDrag::OFF;
        storage.target_limits = TargetLimits::DEFAULT;
        storage.hover_rps = RotPerSec::new(0.0);
        storage.integral_limit_v = Volt::new(0.0);
    }

    storage.has_ref_model = has_ref_model;
    storage.ref_per_motor = per_motor;

    storage.loaded = true;
    GainLoadResult::Ok
}
