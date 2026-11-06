//! Tests for 4D multilinear gain interpolation.

mod common;

use whippyflight_control::gains::{GainStorage, load_gains};
use whippyflight_control::interpolation::interpolate_gains;

/// Build a gain blob at resolution N=2 where each table has a known gradient:
/// the value at each grid vertex encodes its position in the hypercube.
///
/// For a 4-motor × 3-axis matrix at vertex (i0, i1, i2, i3),
/// entry (motor, axis) = table_base + i0 + i1*2 + i2*4 + i3*8 + motor*0.01 + axis*0.001
///
/// This lets us verify that:
/// - At grid corners, interpolation returns exact vertex values
/// - At midpoints, interpolation returns the average of vertices
fn make_gradient_blob() -> Vec<u8> {
    let n = 2usize;
    let floats_per_table = n * n * n * n * 4 * 3;
    let bytes_per_table = floats_per_table * 4;
    let payload_size = 0x20 + 4 * bytes_per_table;
    let total_size = payload_size + 4;
    let mut buf = vec![0u8; total_size];

    // Header
    buf[0..4].copy_from_slice(&0x4C51_5247u32.to_le_bytes());
    buf[4..6].copy_from_slice(&1u16.to_le_bytes());
    buf[6] = 2;
    buf[0x08..0x0C].copy_from_slice(&100.0f32.to_le_bytes()); // min_rps
    buf[0x0C..0x10].copy_from_slice(&200.0f32.to_le_bytes()); // max_rps
    buf[0x10..0x14].copy_from_slice(&0.000125f32.to_le_bytes());
    buf[0x14..0x18].copy_from_slice(&0.1f32.to_le_bytes());
    buf[0x18..0x1C].copy_from_slice(&0.01f32.to_le_bytes());
    buf[0x1C..0x20].copy_from_slice(&(total_size as u32).to_le_bytes());

    let table_bases = [0.0f32, 100.0, 200.0, 300.0]; // ki, kd, kp, binv
    let mut offset = 0x20;

    for (_table_idx, &base) in table_bases.iter().enumerate() {
        for i0 in 0..n {
            for i1 in 0..n {
                for i2 in 0..n {
                    for i3 in 0..n {
                        for motor in 0..4u32 {
                            for axis in 0..3u32 {
                                let vertex_val = i0 + i1 * 2 + i2 * 4 + i3 * 8;
                                let val = base
                                    + vertex_val as f32
                                    + motor as f32 * 0.01
                                    + axis as f32 * 0.001;
                                buf[offset..offset + 4]
                                    .copy_from_slice(&val.to_le_bytes());
                                offset += 4;
                            }
                        }
                    }
                }
            }
        }
    }

    let crc = crc32_compute(&buf[..payload_size]);
    buf[payload_size..payload_size + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

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

fn load_gradient_gains() -> GainStorage {
    let blob = make_gradient_blob();
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, whippyflight_control::types::GainLoadResult::Ok);
    storage
}

#[test]
fn interpolation_at_min_corner() {
    let storage = load_gradient_gains();

    // All motors at min_rps (100.0) → vertex (0,0,0,0)
    let motor_rps = [100.0, 100.0, 100.0, 100.0];
    let interp = interpolate_gains(&motor_rps, &storage);

    // At vertex (0,0,0,0): vertex_val = 0
    // kp (table_base=200.0): entry (0,0) = 200.0 + 0 + 0*0.01 + 0*0.001 = 200.0
    let kp_raw = interp.kp.nalgebra();
    assert!((kp_raw[(0, 0)] - 200.0).abs() < 1e-4, "kp(0,0) = {}", kp_raw[(0, 0)]);

    // ki (table_base=0.0): entry (1,2) = 0.0 + 0 + 1*0.01 + 2*0.001 = 0.012
    let ki_raw = interp.ki.nalgebra();
    assert!((ki_raw[(1, 2)] - 0.012).abs() < 1e-4, "ki(1,2) = {}", ki_raw[(1, 2)]);
}

#[test]
fn interpolation_at_max_corner() {
    let storage = load_gradient_gains();

    // All motors at max_rps (200.0) → vertex (1,1,1,1)
    let motor_rps = [200.0, 200.0, 200.0, 200.0];
    let interp = interpolate_gains(&motor_rps, &storage);

    // vertex (1,1,1,1): vertex_val = 1 + 2 + 4 + 8 = 15
    // kp (base=200.0): entry (0,0) = 200.0 + 15 = 215.0
    let kp_raw = interp.kp.nalgebra();
    assert!((kp_raw[(0, 0)] - 215.0).abs() < 1e-3, "kp(0,0) = {}", kp_raw[(0, 0)]);
}

#[test]
fn interpolation_at_midpoint() {
    let storage = load_gradient_gains();

    // All motors at midpoint (150.0) → frac = [0.5, 0.5, 0.5, 0.5]
    // Result should be average of all 16 vertices
    let motor_rps = [150.0, 150.0, 150.0, 150.0];
    let interp = interpolate_gains(&motor_rps, &storage);

    // Average of vertex_val over all 16 vertices:
    // vertices have vals 0..15, average = 7.5
    // kp (base=200.0): entry (0,0) = 200.0 + 7.5 = 207.5
    let kp_raw = interp.kp.nalgebra();
    assert!(
        (kp_raw[(0, 0)] - 207.5).abs() < 1e-2,
        "kp(0,0) at midpoint = {}",
        kp_raw[(0, 0)]
    );
}

#[test]
fn interpolation_clamps_below_min() {
    let storage = load_gradient_gains();

    // Motor speeds below min_rps should clamp to min corner
    let motor_rps = [0.0, 0.0, 0.0, 0.0];
    let interp = interpolate_gains(&motor_rps, &storage);

    // Should match vertex (0,0,0,0) — same as min corner test
    let kp_raw = interp.kp.nalgebra();
    assert!((kp_raw[(0, 0)] - 200.0).abs() < 1e-4, "kp(0,0) = {}", kp_raw[(0, 0)]);
}

#[test]
fn interpolation_clamps_above_max() {
    let storage = load_gradient_gains();

    // Motor speeds above max_rps should clamp to max corner
    let motor_rps = [9999.0, 9999.0, 9999.0, 9999.0];
    let interp = interpolate_gains(&motor_rps, &storage);

    // Should match vertex (1,1,1,1)
    let kp_raw = interp.kp.nalgebra();
    assert!((kp_raw[(0, 0)] - 215.0).abs() < 1e-3, "kp(0,0) = {}", kp_raw[(0, 0)]);
}

#[test]
fn interpolation_distinguishes_tables() {
    let storage = load_gradient_gains();

    let motor_rps = [100.0, 100.0, 100.0, 100.0]; // min corner
    let interp = interpolate_gains(&motor_rps, &storage);

    // At vertex (0,0,0,0), entry (0,0):
    // ki (base=0):   0.0
    // kd (base=100): 100.0
    // kp (base=200): 200.0
    // binv (base=300): 300.0
    let ki_raw = interp.ki.nalgebra();
    let kd_raw = interp.kd.nalgebra();
    let kp_raw = interp.kp.nalgebra();
    let binv_raw = interp.binv.nalgebra();

    assert!((ki_raw[(0, 0)] - 0.0).abs() < 1e-4);
    assert!((kd_raw[(0, 0)] - 100.0).abs() < 1e-4);
    assert!((kp_raw[(0, 0)] - 200.0).abs() < 1e-4);
    assert!((binv_raw[(0, 0)] - 300.0).abs() < 1e-4);
}

#[test]
fn interpolation_asymmetric_motors() {
    let storage = load_gradient_gains();

    // Motor 0 (fl) at min, motor 1 (bl) at max, others at min
    // This tests that each motor axis interpolates independently
    let motor_rps = [100.0, 200.0, 100.0, 100.0];
    let interp = interpolate_gains(&motor_rps, &storage);

    // frac = [0.0, 1.0, 0.0, 0.0]
    // Active vertices: (0, 1, 0, 0) with weight 1.0
    // vertex_val = 0 + 1*2 + 0 + 0 = 2
    // kp entry (0,0) = 200.0 + 2 = 202.0
    let kp_raw = interp.kp.nalgebra();
    assert!(
        (kp_raw[(0, 0)] - 202.0).abs() < 1e-3,
        "kp(0,0) with asymmetric motors = {}",
        kp_raw[(0, 0)]
    );
}
