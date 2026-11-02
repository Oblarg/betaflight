//! Gain data storage and binary format parsing.
//!
//! The gain binary format (`.wfg`) is:
//!
//! | Offset | Size    | Field           | Description                          |
//! |--------|---------|-----------------|--------------------------------------|
//! | 0x00   | 4       | magic           | `0x4C515247` ("LQRG")                |
//! | 0x04   | 2       | version         | Format version (1)                   |
//! | 0x06   | 1       | resolution      | N (grid points per motor axis)       |
//! | 0x07   | 1       | reserved        | Padding                              |
//! | 0x08   | 4       | min_rps         | `f32` — minimum motor RPS            |
//! | 0x0C   | 4       | max_rps         | `f32` — maximum motor RPS            |
//! | 0x10   | 4       | dt              | `f32` — loop period (s)              |
//! | 0x14   | 4       | integral_tau    | `f32` — integral time constant       |
//! | 0x18   | 4       | derivative_tau  | `f32` — derivative smoothing time    |
//! | 0x1C   | 4       | total_size      | `u32` — total blob size in bytes     |
//! | 0x20   | var     | ki_table        | `f32[N⁴×4×3]`                        |
//! | ...    | var     | kd_table        | `f32[N⁴×4×3]`                        |
//! | ...    | var     | kp_table        | `f32[N⁴×4×3]`                        |
//! | ...    | var     | binv_table      | `f32[N⁴×4×3]`                        |
//! | EOF−4  | 4       | crc32           | CRC32 over bytes `0..EOF−4`          |

use crate::types::GainLoadResult;

/// Magic bytes: "LQRG" in little-endian.
const GAIN_MAGIC: u32 = 0x4C51_5247;

/// Current format version.
const GAIN_VERSION: u16 = 1;

/// Maximum supported resolution (N).  At N=5, each table is 5⁴×4×3 = 7500
/// floats = 30 KB; four tables = 120 KB.  We statically allocate for this max.
pub const MAX_RESOLUTION: usize = 5;

/// Total floats per gain table at a given resolution: N⁴ × 4 motors × 3 axes.
const fn table_floats(n: usize) -> usize {
    n * n * n * n * 4 * 3
}

/// Total floats across all 4 gain tables.
pub const MAX_TABLE_FLOATS: usize = table_floats(MAX_RESOLUTION);

/// Binary header size in bytes.
const HEADER_SIZE: usize = 0x20;

/// Parsed gain header.
#[derive(Clone, Copy, Debug)]
pub struct GainHeader {
    pub resolution: u8,
    pub min_rps: f32,
    pub max_rps: f32,
    pub dt: f32,
    pub integral_tau: f32,
    pub derivative_tau: f32,
}

/// Static storage for gain lookup tables.  Allocated once, populated by
/// [`load_gains`].
pub struct GainStorage {
    pub header: GainHeader,
    pub loaded: bool,

    // Flat f32 arrays for the 4 gain tables.  Indexed as
    // [fl_idx * N³ + bl_idx * N² + fr_idx * N + br_idx] * 12 + motor * 3 + axis
    pub ki: [f32; MAX_TABLE_FLOATS],
    pub kd: [f32; MAX_TABLE_FLOATS],
    pub kp: [f32; MAX_TABLE_FLOATS],
    pub binv: [f32; MAX_TABLE_FLOATS],
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
            },
            loaded: false,
            ki: [0.0; MAX_TABLE_FLOATS],
            kd: [0.0; MAX_TABLE_FLOATS],
            kp: [0.0; MAX_TABLE_FLOATS],
            binv: [0.0; MAX_TABLE_FLOATS],
        }
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

/// Parse and validate a gain binary blob, populating `storage`.
pub fn load_gains(data: &[u8], storage: &mut GainStorage) -> GainLoadResult {
    storage.loaded = false;

    // Minimum size: header + CRC
    if data.len() < HEADER_SIZE + 4 {
        return GainLoadResult::TooShort;
    }

    // Validate magic
    let magic = read_u32(data, 0x00);
    if magic != GAIN_MAGIC {
        return GainLoadResult::BadMagic;
    }

    // Validate version
    let version = read_u16(data, 0x04);
    if version != GAIN_VERSION {
        return GainLoadResult::BadVersion;
    }

    let resolution = data[0x06] as usize;
    if resolution > MAX_RESOLUTION || resolution == 0 {
        return GainLoadResult::ResolutionTooLarge;
    }

    // Validate total size
    let total_size = read_u32(data, 0x1C) as usize;
    if data.len() < total_size {
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
    };

    // Parse gain tables
    let floats_per_table = table_floats(resolution);
    let bytes_per_table = floats_per_table * 4;
    let mut offset = HEADER_SIZE;

    let tables: [&mut [f32; MAX_TABLE_FLOATS]; 4] = [
        &mut storage.ki,
        &mut storage.kd,
        &mut storage.kp,
        &mut storage.binv,
    ];

    for table in tables {
        for i in 0..floats_per_table {
            table[i] = read_f32(data, offset + i * 4);
        }
        offset += bytes_per_table;
    }

    storage.loaded = true;
    GainLoadResult::Ok
}
