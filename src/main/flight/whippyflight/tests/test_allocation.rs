//! Tests for the motor allocator.

use whippyflight_control::allocation::{allocate, balanced_direction};

const V_LO: f32 = 0.5;
const V_HI: f32 = 16.0;
const SYMMETRIC: [f32; 4] = [1.26, 0.74, 1.26, 0.74];

fn approx(a: [f32; 4], b: [f32; 4], tol: f32) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
}

#[test]
fn unconstrained_command_passes_through_with_pilot_collective() {
    let u = [0.3, -0.2, -0.1, 0.2];
    let a = allocate(u, SYMMETRIC, 8.0, V_LO, V_HI);
    let expected = [0, 1, 2, 3].map(|i| u[i] + 8.0 * SYMMETRIC[i]);
    assert!(approx(a.voltage, expected, 1e-5), "{:?}", a.voltage);
    assert_eq!(a.command_scale, 1.0);
    assert!(!a.saturated);
}

#[test]
fn zero_throttle_lifts_lowest_motor_to_idle() {
    let u = [0.3, -0.2, -0.1, 0.2];
    let a = allocate(u, SYMMETRIC, 0.0, V_LO, V_HI);
    let lowest = a.voltage.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!((lowest - V_LO).abs() < 1e-5, "{:?}", a.voltage);
    // The command itself is untouched: v − c·n = u.
    let residual = [0, 1, 2, 3].map(|i| a.voltage[i] - a.collective_v * SYMMETRIC[i]);
    assert!(approx(residual, u, 1e-5));
    assert!(!a.saturated);
}

#[test]
fn full_throttle_lowers_collective_to_keep_command() {
    let u = [2.0, -2.0, -1.0, 1.0];
    let a = allocate(u, SYMMETRIC, V_HI, V_LO, V_HI);
    let highest = a.voltage.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!((highest - V_HI).abs() < 1e-4, "{:?}", a.voltage);
    let residual = [0, 1, 2, 3].map(|i| a.voltage[i] - a.collective_v * SYMMETRIC[i]);
    assert!(approx(residual, u, 1e-4));
    assert!(!a.saturated);
}

#[test]
fn oversized_command_is_scaled_not_clipped() {
    let u = [20.0, -20.0, -10.0, 10.0];
    let a = allocate(u, SYMMETRIC, 8.0, V_LO, V_HI);
    assert!(a.command_scale < 1.0 && a.command_scale > 0.0);
    assert!(a.saturated);
    for &v in &a.voltage {
        assert!(v >= V_LO - 1e-4 && v <= V_HI + 1e-4, "{:?}", a.voltage);
    }
    // Direction preserved: v − c·n = s·u.
    let residual = [0, 1, 2, 3].map(|i| a.voltage[i] - a.collective_v * SYMMETRIC[i]);
    let scaled = u.map(|x| x * a.command_scale);
    assert!(approx(residual, scaled, 1e-3), "{residual:?} vs {scaled:?}");
}

#[test]
fn negative_direction_entry_is_respected() {
    // Diagonal-asymmetric operating point: motor 3 must drop as c rises.
    let n = [2.41, 0.86, 1.14, -0.415];
    let u = [0.0; 4];
    let a = allocate(u, n, 0.0, 0.0, V_HI);
    for &v in &a.voltage {
        assert!(v >= -1e-5 && v <= V_HI + 1e-5, "{:?}", a.voltage);
    }
}

#[test]
fn infeasible_falls_back_to_clipping() {
    // With n₃ < 0 and v_lo > 0, zero command can't keep all motors above idle.
    let n = [2.41, 0.86, 1.14, -0.415];
    let a = allocate([0.0; 4], n, 4.0, V_LO, V_HI);
    assert!(a.saturated);
    for &v in &a.voltage {
        assert!(v >= V_LO && v <= V_HI, "{:?}", a.voltage);
    }
}

const K_W: f32 = 0.00666413;
const K_W2: f32 = 2.47091361e-06;
// d_front/d_back = 1/1.7, d_left/d_right = 1/1: back motors at 1/√1.7 of the front.
const BALANCE: [f32; 4] = [1.1419, 0.8758, 1.1419, 0.8758];

/// gen_wfg's linearizer: u_aff = −k_ω²·ω|ω| per motor.
fn affine() -> [f32; 16] {
    let mut g = [0.0; 16];
    for i in 0..4 {
        g[i * 5] = -K_W2;
    }
    g
}

#[test]
fn balanced_collective_holds_balanced_speeds() {
    // Steady state with the linearizer in the loop: c·nᵢ − k_ω²·ωᵢ² = k_ω·ωᵢ + k_ω²·ωᵢ².
    for c in [0.5f32, 2.0, 8.0] {
        let n = balanced_direction(BALANCE, [K_W, K_W2], &affine(), c);
        assert!((n.iter().sum::<f32>() - 4.0).abs() < 1e-4, "{n:?}");
        let speed = |v: f32| (-K_W + (K_W * K_W + 8.0 * K_W2 * v).sqrt()) / (4.0 * K_W2);
        let w_bar = speed(c * n[0]) / BALANCE[0];
        for i in 1..4 {
            let w = speed(c * n[i]);
            assert!((w / BALANCE[i] - w_bar).abs() < 1e-3 * w_bar, "c={c}: motor {i} at {w}, {n:?}");
        }
    }
}

#[test]
fn front_motors_carry_more_collective() {
    // Front motors (shorter arm) carry more of the collective, more so as
    // the quadratic drag dominates.
    let lo = balanced_direction(BALANCE, [K_W, K_W2], &affine(), 0.5);
    let hi = balanced_direction(BALANCE, [K_W, K_W2], &affine(), 12.0);
    assert!(lo[0] > lo[1] && hi[0] > hi[1]);
    assert!(hi[0] / hi[1] > lo[0] / lo[1]);
}

#[test]
fn without_motor_drag_duty_follows_speed() {
    let n = balanced_direction(BALANCE, [0.0, 0.0], &[0.0; 16], 4.0);
    let mean = BALANCE.iter().sum::<f32>() / 4.0;
    for i in 0..4 {
        assert!((n[i] - BALANCE[i] / mean).abs() < 1e-5, "{n:?}");
    }
}

#[test]
fn uniform_balance_gives_uniform_direction() {
    assert_eq!(balanced_direction([1.0; 4], [0.0, 0.0], &[0.0; 16], 4.0), [1.0; 4]);
    let n = balanced_direction([1.0; 4], [K_W, K_W2], &affine(), 4.0);
    assert!(n.iter().all(|x| (x - 1.0).abs() < 1e-5), "{n:?}");
}
