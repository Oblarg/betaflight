//! Tests for 4D multilinear gain interpolation.

mod common;

use common::put;
use whippyflight_control::gains::{GainStorage, load_gains};
use whippyflight_control::interpolation::interpolate_gains;
use whippyflight_control::types::*;

/// Write one gradient table at `offset`: entry (motor, axis) at vertex
/// (i0, i1, i2, i3) = `unit` × (base + i0 + i1*2 + i2*4 + i3*8 + motor*0.01 + axis*0.001).
fn gradient_table<Q>(buf: &mut [u8], offset: &mut usize, n: usize, unit: Q, base: f32)
where
    Q: bytemuck::Pod + core::ops::Mul<f32, Output = Q>,
{
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
                            put(buf, *offset, unit * val);
                            *offset += 4;
                        }
                    }
                }
            }
        }
    }
}

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
    put(&mut buf, 0x08, RotPerSec::new(100.0)); // min_rps
    put(&mut buf, 0x0C, RotPerSec::new(200.0)); // max_rps
    put(&mut buf, 0x10, Seconds::new(0.000125));
    put(&mut buf, 0x14, Seconds::new(0.1));
    put(&mut buf, 0x18, Seconds::new(0.01));
    buf[0x1C..0x20].copy_from_slice(&(total_size as u32).to_le_bytes());

    let mut offset = 0x20;
    gradient_table(&mut buf, &mut offset, n, VoltPerRot::new(1.0), 0.0); // ki
    gradient_table(&mut buf, &mut offset, n, VoltPerRpsPerSec::new(1.0), 100.0); // kd
    gradient_table(&mut buf, &mut offset, n, VoltPerRps::new(1.0), 200.0); // kp
    gradient_table(&mut buf, &mut offset, n, VoltPerRps::new(1.0), 300.0); // binv

    let crc = common::crc32(&buf[..payload_size]);
    buf[payload_size..payload_size + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

fn load_gradient_gains() -> GainStorage {
    let blob = make_gradient_blob();
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    storage
}

fn rps(v: f32) -> [RotPerSec; 4] {
    [RotPerSec::new(v); 4]
}

#[test]
fn interpolation_at_min_corner() {
    let storage = load_gradient_gains();

    // All motors at min_rps (100.0) → vertex (0,0,0,0)
    let interp = interpolate_gains(&rps(100.0), &storage);

    // At vertex (0,0,0,0): vertex_val = 0
    // kp (table_base=200.0): entry (0,0) = 200.0 + 0 + 0*0.01 + 0*0.001 = 200.0
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!((kp - VoltPerRps::new(200.0)).abs() < VoltPerRps::new(1e-4), "kp(0,0) = {kp:?}");

    // ki (table_base=0.0): entry (1,2) = 0.0 + 0 + 1*0.01 + 2*0.001 = 0.012
    let ki = interp.ki.as_uniform().get(1, 2);
    assert!((ki - VoltPerRot::new(0.012)).abs() < VoltPerRot::new(1e-4), "ki(1,2) = {ki:?}");
}

#[test]
fn interpolation_at_max_corner() {
    let storage = load_gradient_gains();

    // All motors at max_rps (200.0) → vertex (1,1,1,1)
    let interp = interpolate_gains(&rps(200.0), &storage);

    // vertex (1,1,1,1): vertex_val = 1 + 2 + 4 + 8 = 15
    // kp (base=200.0): entry (0,0) = 200.0 + 15 = 215.0
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!((kp - VoltPerRps::new(215.0)).abs() < VoltPerRps::new(1e-3), "kp(0,0) = {kp:?}");
}

#[test]
fn interpolation_at_midpoint() {
    let storage = load_gradient_gains();

    // All motors at midpoint (150.0) → frac = [0.5, 0.5, 0.5, 0.5]
    // Result should be average of all 16 vertices
    let interp = interpolate_gains(&rps(150.0), &storage);

    // Average of vertex_val over all 16 vertices:
    // vertices have vals 0..15, average = 7.5
    // kp (base=200.0): entry (0,0) = 200.0 + 7.5 = 207.5
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!(
        (kp - VoltPerRps::new(207.5)).abs() < VoltPerRps::new(1e-2),
        "kp(0,0) at midpoint = {kp:?}"
    );
}

#[test]
fn interpolation_clamps_below_min() {
    let storage = load_gradient_gains();

    // Motor speeds below min_rps should clamp to min corner
    let interp = interpolate_gains(&rps(0.0), &storage);

    // Should match vertex (0,0,0,0) — same as min corner test
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!((kp - VoltPerRps::new(200.0)).abs() < VoltPerRps::new(1e-4), "kp(0,0) = {kp:?}");
}

#[test]
fn interpolation_clamps_above_max() {
    let storage = load_gradient_gains();

    // Motor speeds above max_rps should clamp to max corner
    let interp = interpolate_gains(&rps(9999.0), &storage);

    // Should match vertex (1,1,1,1)
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!((kp - VoltPerRps::new(215.0)).abs() < VoltPerRps::new(1e-3), "kp(0,0) = {kp:?}");
}

#[test]
fn interpolation_distinguishes_tables() {
    let storage = load_gradient_gains();

    let interp = interpolate_gains(&rps(100.0), &storage); // min corner

    // At vertex (0,0,0,0), entry (0,0):
    // ki (base=0):   0.0
    // kd (base=100): 100.0
    // kp (base=200): 200.0
    // binv (base=300): 300.0
    let ki = interp.ki.as_uniform().get(0, 0);
    let kd = interp.kd.as_uniform().get(0, 0);
    let kp = interp.kp.as_uniform().get(0, 0);
    let binv = interp.binv.as_uniform().get(0, 0);

    assert!((ki - VoltPerRot::new(0.0)).abs() < VoltPerRot::new(1e-4));
    assert!((kd - VoltPerRpsPerSec::new(100.0)).abs() < VoltPerRpsPerSec::new(1e-4));
    assert!((kp - VoltPerRps::new(200.0)).abs() < VoltPerRps::new(1e-4));
    assert!((binv - VoltPerRps::new(300.0)).abs() < VoltPerRps::new(1e-4));
}

#[test]
fn interpolation_asymmetric_motors() {
    let storage = load_gradient_gains();

    // Motor 0 (fl) at min, motor 1 (bl) at max, others at min
    // This tests that each motor axis interpolates independently
    let motor_rps = [100.0, 200.0, 100.0, 100.0].map(RotPerSec::new);
    let interp = interpolate_gains(&motor_rps, &storage);

    // frac = [0.0, 1.0, 0.0, 0.0]
    // Active vertices: (0, 1, 0, 0) with weight 1.0
    // vertex_val = 0 + 1*2 + 0 + 0 = 2
    // kp entry (0,0) = 200.0 + 2 = 202.0
    let kp = interp.kp.as_uniform().get(0, 0);
    assert!(
        (kp - VoltPerRps::new(202.0)).abs() < VoltPerRps::new(1e-3),
        "kp(0,0) with asymmetric motors = {kp:?}"
    );
}
