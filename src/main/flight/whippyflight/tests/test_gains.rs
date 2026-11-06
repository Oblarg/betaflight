//! Tests for gain binary parsing and validation.

mod common;

use whippyflight_control::gains::*;
use whippyflight_control::types::GainLoadResult;

/// Build a valid gain binary blob with the given resolution and table values.
/// Each table is filled with `fill_value` for simplicity.
fn make_gain_blob(resolution: u8, fill_value: f32) -> Vec<u8> {
    let n = resolution as usize;
    let floats_per_table = n * n * n * n * 4 * 3;
    let bytes_per_table = floats_per_table * 4;
    let payload_size = 0x20 + 4 * bytes_per_table;
    let total_size = payload_size + 4; // + CRC32

    let mut buf = vec![0u8; total_size];

    // Magic: "LQRG" = 0x4C515247 LE
    buf[0..4].copy_from_slice(&0x4C51_5247u32.to_le_bytes());
    // Version: 1
    buf[4..6].copy_from_slice(&1u16.to_le_bytes());
    // Resolution
    buf[6] = resolution;
    // Reserved
    buf[7] = 0;
    // min_rps
    buf[0x08..0x0C].copy_from_slice(&100.0f32.to_le_bytes());
    // max_rps
    buf[0x0C..0x10].copy_from_slice(&500.0f32.to_le_bytes());
    // dt
    buf[0x10..0x14].copy_from_slice(&0.000125f32.to_le_bytes()); // 8 kHz
    // integral_tau
    buf[0x14..0x18].copy_from_slice(&0.1f32.to_le_bytes());
    // derivative_tau
    buf[0x18..0x1C].copy_from_slice(&0.01f32.to_le_bytes());
    // total_size
    buf[0x1C..0x20].copy_from_slice(&(total_size as u32).to_le_bytes());

    // Fill all 4 tables with fill_value
    let mut offset = 0x20;
    for _ in 0..4 {
        for _ in 0..floats_per_table {
            buf[offset..offset + 4].copy_from_slice(&fill_value.to_le_bytes());
            offset += 4;
        }
    }

    // CRC32 over everything except the last 4 bytes
    let crc = crc32_compute(&buf[..payload_size]);
    buf[payload_size..payload_size + 4].copy_from_slice(&crc.to_le_bytes());

    buf
}

/// Build a gain blob where each table has distinct values per motor×axis,
/// allowing interpolation tests to distinguish them.
#[allow(dead_code)]
fn make_distinguishable_gain_blob(resolution: u8) -> Vec<u8> {
    let n = resolution as usize;
    let floats_per_table = n * n * n * n * 4 * 3;
    let bytes_per_table = floats_per_table * 4;
    let payload_size = 0x20 + 4 * bytes_per_table;
    let total_size = payload_size + 4;

    let mut buf = vec![0u8; total_size];

    // Header (same as make_gain_blob)
    buf[0..4].copy_from_slice(&0x4C51_5247u32.to_le_bytes());
    buf[4..6].copy_from_slice(&1u16.to_le_bytes());
    buf[6] = resolution;
    buf[0x08..0x0C].copy_from_slice(&100.0f32.to_le_bytes());
    buf[0x0C..0x10].copy_from_slice(&500.0f32.to_le_bytes());
    buf[0x10..0x14].copy_from_slice(&0.000125f32.to_le_bytes());
    buf[0x14..0x18].copy_from_slice(&0.1f32.to_le_bytes());
    buf[0x18..0x1C].copy_from_slice(&0.01f32.to_le_bytes());
    buf[0x1C..0x20].copy_from_slice(&(total_size as u32).to_le_bytes());

    // Fill tables: table_idx * 1000 + grid_point * 12 + motor * 3 + axis
    // This makes every entry unique and predictable.
    let mut offset = 0x20;
    for table_idx in 0..4u32 {
        for grid_point in 0..floats_per_table / 12 {
            for motor in 0..4u32 {
                for axis in 0..3u32 {
                    let val = table_idx as f32 * 1000.0
                        + grid_point as f32 * 0.1
                        + motor as f32 * 0.01
                        + axis as f32 * 0.001;
                    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
                    offset += 4;
                }
            }
        }
    }

    let crc = crc32_compute(&buf[..payload_size]);
    buf[payload_size..payload_size + 4].copy_from_slice(&crc.to_le_bytes());

    buf
}

/// Recompute CRC32 (ISO 3309) — mirrors the implementation in gains.rs.
fn crc32_compute(data: &[u8]) -> u32 {
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

// =========================================================================
// Gain parsing tests
// =========================================================================

#[test]
fn load_valid_gains_resolution_2() {
    let blob = make_gain_blob(2, 1.0);
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);

    assert_eq!(result, GainLoadResult::Ok);
    assert!(storage.loaded);
    assert_eq!(storage.header.resolution, 2);
    assert_eq!(storage.header.min_rps, 100.0);
    assert_eq!(storage.header.max_rps, 500.0);
    assert!((storage.header.dt - 0.000125).abs() < 1e-9);
    assert!((storage.header.integral_tau - 0.1).abs() < 1e-9);
    assert!((storage.header.derivative_tau - 0.01).abs() < 1e-9);

    // All table entries should be 1.0
    let n = 2usize;
    let floats = n * n * n * n * 4 * 3;
    for i in 0..floats {
        assert_eq!(storage.ki.get(i), 1.0, "ki[{i}]");
        assert_eq!(storage.kd.get(i), 1.0, "kd[{i}]");
        assert_eq!(storage.kp.get(i), 1.0, "kp[{i}]");
        assert_eq!(storage.binv.get(i), 1.0, "binv[{i}]");
    }
}

#[test]
fn load_valid_gains_resolution_5() {
    let blob = make_gain_blob(5, 0.5);
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);

    assert_eq!(result, GainLoadResult::Ok);
    assert!(storage.loaded);
    assert_eq!(storage.header.resolution, 5);
}

#[test]
fn reject_empty_buffer() {
    let mut storage = GainStorage::new();
    let result = load_gains(&[], &mut storage);
    assert_eq!(result, GainLoadResult::TooShort);
    assert!(!storage.loaded);
}

#[test]
fn reject_truncated_header() {
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&[0u8; 16]), &mut storage);
    assert_eq!(result, GainLoadResult::TooShort);
    assert!(!storage.loaded);
}

#[test]
fn reject_bad_magic() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[0] = 0xFF; // corrupt magic
    // Recompute CRC so we test magic check, not CRC check
    let payload_size = blob.len() - 4;
    let crc = crc32_compute(&blob[..payload_size]);
    blob[payload_size..].copy_from_slice(&crc.to_le_bytes());

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::BadMagic);
}

#[test]
fn reject_bad_version() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[4..6].copy_from_slice(&99u16.to_le_bytes()); // bad version
    let payload_size = blob.len() - 4;
    let crc = crc32_compute(&blob[..payload_size]);
    blob[payload_size..].copy_from_slice(&crc.to_le_bytes());

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::BadVersion);
}

#[test]
fn reject_resolution_zero() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[6] = 0; // resolution = 0
    let payload_size = blob.len() - 4;
    let crc = crc32_compute(&blob[..payload_size]);
    blob[payload_size..].copy_from_slice(&crc.to_le_bytes());

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::ResolutionTooLarge);
}

#[test]
fn reject_resolution_too_large() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[6] = 6; // resolution = 6 > MAX_RESOLUTION(5)
    let payload_size = blob.len() - 4;
    let crc = crc32_compute(&blob[..payload_size]);
    blob[payload_size..].copy_from_slice(&crc.to_le_bytes());

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::ResolutionTooLarge);
}

#[test]
fn reject_corrupted_crc() {
    let mut blob = make_gain_blob(2, 1.0);
    // Corrupt a data byte without updating CRC
    blob[0x20] ^= 0xFF;

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::BadChecksum);
}

#[test]
fn reject_truncated_data() {
    let blob = make_gain_blob(2, 1.0);
    // Pass only half the blob
    let half = blob.len() / 2;

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob[..half]), &mut storage);
    // Should fail with TooShort (total_size > data.len())
    assert_eq!(result, GainLoadResult::TooShort);
}

#[test]
fn load_then_reload_clears_previous() {
    let blob1 = make_gain_blob(2, 1.0);
    let blob2 = make_gain_blob(3, 2.0);

    let mut storage = GainStorage::new();

    let result = load_gains(common::static_blob(&blob1), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    assert_eq!(storage.header.resolution, 2);
    assert_eq!(storage.ki.get(0), 1.0);

    let result = load_gains(common::static_blob(&blob2), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    assert_eq!(storage.header.resolution, 3);
    assert_eq!(storage.ki.get(0), 2.0);
}

#[test]
fn failed_load_does_not_set_loaded() {
    let blob = make_gain_blob(2, 1.0);
    let mut storage = GainStorage::new();

    // First: valid load
    load_gains(common::static_blob(&blob), &mut storage);
    assert!(storage.loaded);

    // Second: invalid load (truncated)
    load_gains(common::static_blob(&[0u8; 10]), &mut storage);
    assert!(!storage.loaded);
}

#[test]
fn reject_misaligned_blob() {
    // Tables are read in place as f32, so the blob must be 4-byte aligned.
    let blob = make_gain_blob(2, 1.0);
    let mut padded = vec![0u8];
    padded.extend_from_slice(&blob);
    let misaligned = &common::static_blob(&padded)[1..];
    let mut storage = GainStorage::new();
    assert_eq!(load_gains(misaligned, &mut storage), GainLoadResult::Misaligned);
    assert!(!storage.loaded);
}

#[test]
fn tables_are_views_into_blob() {
    let blob = common::static_blob(&make_gain_blob(2, 1.0));
    let mut storage = GainStorage::new();
    assert_eq!(load_gains(blob, &mut storage), GainLoadResult::Ok);
    let range = blob.as_ptr_range();
    for table in [storage.ki, storage.kd, storage.kp, storage.binv] {
        assert!(range.contains(&table.data_ptr()), "table was copied");
    }
}
