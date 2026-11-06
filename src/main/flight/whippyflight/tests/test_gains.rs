//! Tests for gain binary parsing and validation.

mod common;

use common::put;
use whippyflight_control::gains::*;
use whippyflight_control::types::*;

/// Fill `count` floats at `offset` with `value`.
fn fill<Q: bytemuck::Pod>(buf: &mut [u8], offset: &mut usize, count: usize, value: Q) {
    for _ in 0..count {
        put(buf, *offset, value);
        *offset += 4;
    }
}

/// Build a valid gain binary blob with the given resolution and table values.
/// Each table is filled with `fill_value` (in its own unit) for simplicity.
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
    put(&mut buf, 0x08, RotPerSec::new(100.0)); // min_rps
    put(&mut buf, 0x0C, RotPerSec::new(500.0)); // max_rps
    put(&mut buf, 0x10, Seconds::new(0.000125)); // dt: 8 kHz
    put(&mut buf, 0x14, Seconds::new(0.1)); // integral_tau
    put(&mut buf, 0x18, Seconds::new(0.01)); // derivative_tau
    // total_size
    buf[0x1C..0x20].copy_from_slice(&(total_size as u32).to_le_bytes());

    // Fill all 4 tables (ki, kd, kp, binv) with fill_value
    let mut offset = 0x20;
    fill(&mut buf, &mut offset, floats_per_table, VoltPerRot::new(fill_value));
    fill(&mut buf, &mut offset, floats_per_table, VoltPerRpsPerSec::new(fill_value));
    fill(&mut buf, &mut offset, floats_per_table, VoltPerRps::new(fill_value));
    fill(&mut buf, &mut offset, floats_per_table, VoltPerRps::new(fill_value));

    // CRC32 over everything except the last 4 bytes
    let crc = common::crc32(&buf[..payload_size]);
    buf[payload_size..payload_size + 4].copy_from_slice(&crc.to_le_bytes());

    buf
}

/// Recompute the CRC after editing `blob`'s payload.
fn refresh_crc(blob: &mut [u8]) {
    let payload_size = blob.len() - 4;
    let crc = common::crc32(&blob[..payload_size]);
    blob[payload_size..].copy_from_slice(&crc.to_le_bytes());
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
    assert_eq!(storage.header.min_rps, RotPerSec::new(100.0));
    assert_eq!(storage.header.max_rps, RotPerSec::new(500.0));
    let tol = Seconds::new(1e-9);
    assert!((storage.header.dt - Seconds::new(0.000125)).abs() < tol);
    assert!((storage.header.integral_tau - Seconds::new(0.1)).abs() < tol);
    assert!((storage.header.derivative_tau - Seconds::new(0.01)).abs() < tol);

    // All table entries should be 1.0
    let n = 2usize;
    let floats = n * n * n * n * 4 * 3;
    for i in 0..floats {
        assert_eq!(storage.ki.get(i), VoltPerRot::new(1.0), "ki[{i}]");
        assert_eq!(storage.kd.get(i), VoltPerRpsPerSec::new(1.0), "kd[{i}]");
        assert_eq!(storage.kp.get(i), VoltPerRps::new(1.0), "kp[{i}]");
        assert_eq!(storage.binv.get(i), VoltPerRps::new(1.0), "binv[{i}]");
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
    refresh_crc(&mut blob);

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::BadMagic);
}

#[test]
fn reject_bad_version() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[4..6].copy_from_slice(&99u16.to_le_bytes()); // bad version
    refresh_crc(&mut blob);

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::BadVersion);
}

#[test]
fn reject_resolution_zero() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[6] = 0; // resolution = 0
    refresh_crc(&mut blob);

    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::ResolutionTooLarge);
}

#[test]
fn reject_resolution_too_large() {
    let mut blob = make_gain_blob(2, 1.0);
    blob[6] = 6; // resolution = 6 > MAX_RESOLUTION(5)
    refresh_crc(&mut blob);

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
    assert_eq!(storage.ki.get(0), VoltPerRot::new(1.0));

    let result = load_gains(common::static_blob(&blob2), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    assert_eq!(storage.header.resolution, 3);
    assert_eq!(storage.ki.get(0), VoltPerRot::new(2.0));
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
    let tables = [storage.ki.data_ptr(), storage.kd.data_ptr(), storage.kp.data_ptr(), storage.binv.data_ptr()];
    for ptr in tables {
        assert!(range.contains(&ptr), "table was copied");
    }
}
