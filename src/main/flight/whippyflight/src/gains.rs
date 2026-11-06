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

use crate::types::GainLoadResult;

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

/// Current format version: adds the airframe drag and target-limit blocks.
const GAIN_VERSION: u16 = 7;

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

/// Floats in the affine linearizer block.
pub const AFFINE_FLOATS: usize = 16;

/// Floats in the motor block.
pub const MOTOR_FLOATS: usize = 2;

/// Floats in the thrust-balance block.
pub const BALANCE_FLOATS: usize = 4;

/// Floats in the airframe drag block (v7):
/// `[lp_hz, roll←ax, roll←ay, pitch←ax, pitch←ay]`.
///
/// Rotor drag on tall frames acts well above the CG, so airspeed torques
/// pitch and roll.  In flight the accelerometer reads thrust (body z) plus
/// that drag, so its body x/y components (g), one-pole low-passed at
/// `lp_hz`, measure it; the coefficients (rot/s² per g, BF axes) give the
/// angular acceleration it causes.  `lp_hz = 0` turns the mode off.
pub const DRAG_FLOATS: usize = 5;

/// Floats in the target-limit block (v7): the setpoint estimator's
/// `[slew (rot/s²), jerk (rot/s³)]` bounds on the feedforward target, set
/// from the airframe's pitch/roll authority (see
/// [`crate::setpoint`]).  A value ≤ 0 selects the estimator's default.
pub const TARGET_FLOATS: usize = 2;

/// Floats in the hover block (v7): [hover rps, integral limit V] — the
/// fitted airframe's median hover motor speed (model rps), for
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

/// Floats in the reference-model table at a given resolution.
const fn ref_floats(n: usize) -> usize {
    n * n * n * n * REF_FLOATS
}

/// Binary header size in bytes.
const HEADER_SIZE: usize = 0x20;

// Tables are read in place as native f32/u16, which must match the blob's
// little-endian encoding.
#[cfg(target_endian = "big")]
compile_error!("the .wfg tables are little-endian and are read in place");

/// Storage of one lookup table, read in place from the blob.
#[derive(Clone, Copy)]
enum TableData {
    F32(&'static [f32]),
    /// value[i] = lo[i % W] + step[i % W] · q[i]
    U16 { q: &'static [u16], lo: &'static [f32], step: &'static [f32] },
}

/// A lookup table with `width` values per grid point, indexed
/// [grid point] · width + slot.
#[derive(Clone, Copy)]
pub struct GainTable {
    data: TableData,
}

impl GainTable {
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
    pub fn get(&self, i: usize) -> f32 {
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
    pub fn blend<const W: usize>(&self, point: &[usize], weight: &[f32]) -> [f32; W] {
        let mut out = [0.0f32; W];
        match self.data {
            TableData::F32(v) => {
                for (&p, &w) in point.iter().zip(weight) {
                    let row = &v[p * W..p * W + W];
                    for (o, &x) in out.iter_mut().zip(row) {
                        *o += w * x;
                    }
                }
            }
            TableData::U16 { q, lo, step } => {
                for (&p, &w) in point.iter().zip(weight) {
                    let row = &q[p * W..p * W + W];
                    for (o, &x) in out.iter_mut().zip(row) {
                        *o += w * x as f32;
                    }
                }
                for ((o, &l), &s) in out.iter_mut().zip(lo).zip(step) {
                    *o = l + s * *o;
                }
            }
        }
        out
    }
}

/// Parsed gain header.
#[derive(Clone, Copy, Debug)]
pub struct GainHeader {
    pub resolution: u8,
    pub min_rps: f32,
    pub max_rps: f32,
    pub dt: f32,
    pub integral_tau: f32,
    pub derivative_tau: f32,
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
    pub ki: GainTable,
    pub kd: GainTable,
    pub kp: GainTable,
    pub binv: GainTable,

    /// Reference-model table, [grid point] * REF_FLOATS + offset; empty
    /// when the blob has none (v1–v2).
    pub ref_model: GainTable,
    /// Whether `ref_model` was present in the blob (v3+).
    pub has_ref_model: bool,

    /// Affine linearizer G, row-major [motor][signed-square motor]:
    /// u_aff = G · ω|ω|.  Constant over the grid.
    pub affine: [f32; AFFINE_FLOATS],

    /// Steady-state motor drag, V = k_ω·ω + k_ω²·ω|ω| (main.R's motor fit
    /// without the k_α·ω̇ term): [k_ω (V per model rps), k_ω² (V per
    /// model rps²)].  Zero when the blob has none (v1–v3).
    pub motor_drag: [f32; MOTOR_FLOATS],

    /// Per-motor hover speed ratio for a torque-free hover, geometric mean 1
    /// (1/√(d_fb·d_lr) from the arm lengths).  Uniform before v6.
    pub thrust_balance: [f32; BALANCE_FLOATS],

    /// Airframe drag model, see [`DRAG_FLOATS`].  Zero (off) before v7.
    pub airframe_drag: [f32; DRAG_FLOATS],

    /// Setpoint target limits, see [`TARGET_FLOATS`].  Zero (defaults)
    /// before v7.
    pub target_limits: [f32; TARGET_FLOATS],

    /// Hover motor speed (model rps), see [`HOVER_FLOATS`].  0 before v7.
    pub hover_rps: f32,
    /// Per-motor integral voltage bound (V), see [`HOVER_FLOATS`].  0 before v7.
    pub integral_limit_v: f32,
}

impl GainStorage {
    pub const fn new() -> Self {
        Self {
            header: GainHeader {
                resolution: 0,
                min_rps: 0.0,
                max_rps: 0.0,
                dt: 0.0,
                integral_tau: 0.0,
                derivative_tau: 0.0,
                ff_horizon: 0,
            },
            loaded: false,
            ki: GainTable::EMPTY,
            kd: GainTable::EMPTY,
            kp: GainTable::EMPTY,
            binv: GainTable::EMPTY,
            ref_model: GainTable::EMPTY,
            has_ref_model: false,
            affine: [0.0; AFFINE_FLOATS],
            motor_drag: [0.0; MOTOR_FLOATS],
            thrust_balance: [1.0; BALANCE_FLOATS],
            airframe_drag: [0.0; DRAG_FLOATS],
            target_limits: [0.0; TARGET_FLOATS],
            hover_rps: 0.0,
            integral_limit_v: 0.0,
        }
    }
}

/// View `count` f32s starting at byte `offset` of an aligned blob.
fn f32_slice(data: &'static [u8], offset: usize, count: usize) -> &'static [f32] {
    let bytes = &data[offset..offset + count * 4];
    debug_assert!(bytes.as_ptr() as usize % core::mem::align_of::<f32>() == 0);
    // SAFETY: `bytes` is in bounds, 4-byte aligned (blob alignment is checked
    // in `load_gains` and every table offset is a multiple of 4), and every
    // bit pattern is a valid f32.
    unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const f32, count) }
}

/// View `count` u16s starting at byte `offset` of an aligned blob.
fn u16_slice(data: &'static [u8], offset: usize, count: usize) -> &'static [u16] {
    let bytes = &data[offset..offset + count * 2];
    debug_assert!(bytes.as_ptr() as usize % core::mem::align_of::<u16>() == 0);
    // SAFETY: as for `f32_slice`; every offset is even.
    unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const u16, count) }
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

/// Read a little-endian `f32` from a byte slice at the given offset.
fn read_f32(data: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
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
    let has_drag = version >= GAIN_VERSION;

    let resolution = data[0x06] as usize;
    if resolution > MAX_RESOLUTION || resolution == 0 {
        return GainLoadResult::ResolutionTooLarge;
    }

    let floats_per_table = table_floats(resolution);
    let affine_floats = if has_affine { AFFINE_FLOATS } else { 0 };
    let ref_count = if has_ref_model { ref_floats(resolution) } else { 0 };
    let motor_floats = if has_motor { MOTOR_FLOATS } else { 0 }
        + if has_balance { BALANCE_FLOATS } else { 0 }
        + if has_drag { DRAG_FLOATS + TARGET_FLOATS + HOVER_FLOATS } else { 0 };
    let table_values = 4 * floats_per_table + ref_count;
    let body_bytes = if quantized {
        let quant_floats = 2 * (4 * GAIN_SLOTS + REF_FLOATS);
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
    let mut offset = HEADER_SIZE;
    let mut next_f32 = |count: usize| {
        let table = f32_slice(data, offset, count);
        offset += count * 4;
        table
    };
    let widths = [GAIN_SLOTS, GAIN_SLOTS, GAIN_SLOTS, GAIN_SLOTS, REF_FLOATS];
    let counts = [floats_per_table, floats_per_table, floats_per_table, floats_per_table, ref_count];
    let mut tables = [GainTable::EMPTY; 5];

    if quantized {
        let mut quant = [(&[][..], &[][..]); 5];
        for (q, &w) in quant.iter_mut().zip(&widths) {
            *q = (next_f32(w), next_f32(w));
        }
        storage.affine.copy_from_slice(next_f32(AFFINE_FLOATS));
        storage.motor_drag.copy_from_slice(next_f32(MOTOR_FLOATS));
        storage.thrust_balance = [1.0; BALANCE_FLOATS];
        if has_balance {
            storage.thrust_balance.copy_from_slice(next_f32(BALANCE_FLOATS));
        }
        storage.airframe_drag = [0.0; DRAG_FLOATS];
        storage.target_limits = [0.0; TARGET_FLOATS];
        storage.hover_rps = 0.0;
        storage.integral_limit_v = 0.0;
        if has_drag {
            storage.airframe_drag.copy_from_slice(next_f32(DRAG_FLOATS));
            storage.target_limits.copy_from_slice(next_f32(TARGET_FLOATS));
            [storage.hover_rps, storage.integral_limit_v] = next_f32(HOVER_FLOATS).try_into().unwrap();
        }
        for ((table, &(lo, step)), &count) in tables.iter_mut().zip(&quant).zip(&counts) {
            *table = GainTable { data: TableData::U16 { q: u16_slice(data, offset, count), lo, step } };
            offset += count * 2;
        }
    } else {
        for (table, &count) in tables.iter_mut().zip(&counts) {
            *table = GainTable { data: TableData::F32(next_f32(count)) };
        }
        storage.affine = [0.0; AFFINE_FLOATS];
        if has_affine {
            storage.affine.copy_from_slice(next_f32(AFFINE_FLOATS));
        }
        storage.motor_drag = [0.0; MOTOR_FLOATS];
        if has_motor {
            storage.motor_drag.copy_from_slice(next_f32(MOTOR_FLOATS));
        }
        storage.thrust_balance = [1.0; BALANCE_FLOATS];
        storage.airframe_drag = [0.0; DRAG_FLOATS];
        storage.target_limits = [0.0; TARGET_FLOATS];
        storage.hover_rps = 0.0;
        storage.integral_limit_v = 0.0;
    }

    let [ki, kd, kp, binv, ref_model] = tables;
    storage.ki = ki;
    storage.kd = kd;
    storage.kp = kp;
    storage.binv = binv;
    storage.ref_model = ref_model;
    storage.has_ref_model = has_ref_model;

    storage.loaded = true;
    GainLoadResult::Ok
}
