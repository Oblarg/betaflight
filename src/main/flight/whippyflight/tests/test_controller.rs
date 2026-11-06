//! Tests for the LQR controller logic.

mod common;

use common::put;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{GainStorage, load_gains};
use whippyflight_control::types::*;

const DT: Seconds = Seconds::new(0.000125);
const BATTERY: Volt = Volt::new(12.0);

/// Diagonal gains for [`make_identity_gains`].
struct Diag {
    kp: VoltPerRps,
    ki: VoltPerRot,
    kd: VoltPerRpsPerSec,
    binv: VoltPerRps,
}

fn diag(kp: f32, ki: f32, kd: f32, binv: f32) -> Diag {
    Diag { kp: VoltPerRps::new(kp), ki: VoltPerRot::new(ki), kd: VoltPerRpsPerSec::new(kd), binv: VoltPerRps::new(binv) }
}

/// Build a gain blob with identity-like gains for testing.
/// Kp = diag(kp), Ki = diag(ki), Kd = diag(kd), Binv = diag(binv)
/// (extended to 4×3: motor i maps to axis i for i < 3, motor 3 = zeros).
fn make_identity_gains(resolution: u8, gains: &Diag, dt: Seconds) -> Vec<u8> {
    let n = resolution as usize;
    let floats_per_table = n * n * n * n * 4 * 3;
    let bytes_per_table = floats_per_table * 4;
    let payload_size = 0x20 + 4 * bytes_per_table;
    let total_size = payload_size + 4;
    let mut buf = vec![0u8; total_size];

    buf[0..4].copy_from_slice(&0x4C51_5247u32.to_le_bytes());
    buf[4..6].copy_from_slice(&1u16.to_le_bytes());
    buf[6] = resolution;
    put(&mut buf, 0x08, RotPerSec::new(100.0));
    put(&mut buf, 0x0C, RotPerSec::new(500.0));
    put(&mut buf, 0x10, dt);
    put(&mut buf, 0x14, Seconds::new(0.1)); // integral_tau
    put(&mut buf, 0x18, Seconds::new(0.01)); // derivative_tau
    buf[0x1C..0x20].copy_from_slice(&(total_size as u32).to_le_bytes());

    fn table<Q: bytemuck::Pod + Default>(buf: &mut [u8], offset: &mut usize, grid_points: usize, diag: Q) {
        for _grid_point in 0..grid_points {
            // 4 motors × 3 axes: identity-like (motor i, axis i) = diag
            for motor in 0..4u32 {
                for axis in 0..3u32 {
                    let val = if motor < 3 && motor == axis { diag } else { Q::default() };
                    put(buf, *offset, val);
                    *offset += 4;
                }
            }
        }
    }
    let mut offset = 0x20;
    let grid_points = floats_per_table / 12;
    table(&mut buf, &mut offset, grid_points, gains.ki);
    table(&mut buf, &mut offset, grid_points, gains.kd);
    table(&mut buf, &mut offset, grid_points, gains.kp);
    table(&mut buf, &mut offset, grid_points, gains.binv);

    let crc = common::crc32(&buf[..payload_size]);
    buf[payload_size..payload_size + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

fn load_identity_gains(kp: f32, ki: f32, kd: f32, binv: f32) -> GainStorage {
    let blob = make_identity_gains(2, &diag(kp, ki, kd, binv), DT);
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    storage
}

fn make_sensors(gyro: [f32; 3], battery_v: f32) -> SensorData {
    SensorData {
        gyro_rps: gyro.map(RotPerSec::new),
        motor_rps: [RotPerSec::new(200.0); 4],
        battery_voltage: Volt::new(battery_v),
        accel_g: [0.0; 3],
    }
}

fn make_setpoint(rates: [f32; 3], throttle: f32) -> SetpointData {
    SetpointData {
        rate_rps: rates.map(RotPerSec::new),
        throttle,
    }
}

// =========================================================================
// Controller tests
// =========================================================================

#[test]
fn no_gains_returns_zero_output() {
    let mut ctrl = ControllerState::new();
    let storage = GainStorage::new(); // loaded = false

    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint = make_setpoint([1.0, 0.0, 0.0], 0.5);

    let output = ctrl.update(&sensors, &setpoint, &storage);
    assert_eq!(output.motor_duty, [0.0; 4]);
    assert_eq!(ctrl.telemetry().status, 0);
}

#[test]
fn zero_error_produces_throttle_only() {
    let gains = load_identity_gains(1.0, 0.0, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    // Gyro matches setpoint exactly → zero error → zero feedback/feedforward
    let sensors = make_sensors([1.0, 0.0, 0.0], 12.0);
    let setpoint = make_setpoint([1.0, 0.0, 0.0], 0.5);

    let output = ctrl.update(&sensors, &setpoint, &gains);

    // Zero control voltage: duty is the throttle (collective = throttle·Vbat
    // along a uniform direction, since binv is zero; v1 blob → no idle floor).
    let expected_duty = 0.5;
    for i in 0..4 {
        assert!(
            (output.motor_duty[i] - expected_duty).abs() < 1e-4,
            "motor[{i}] = {} (expected {})",
            output.motor_duty[i],
            expected_duty
        );
    }
    assert_eq!(ctrl.telemetry().status, 1);
}

#[test]
fn proportional_error_generates_feedback() {
    let gains = load_identity_gains(1.0, 0.0, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    // Setpoint = [1, 0, 0], gyro = [0, 0, 0] → error = [1, 0, 0]
    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint = make_setpoint([1.0, 0.0, 0.0], 0.5);

    let _output = ctrl.update(&sensors, &setpoint, &gains);

    // Kp is identity-like: motor 0 gets error[0]*kp = 1.0*1.0 = 1.0 V
    let telem = ctrl.telemetry();
    assert!(
        telem.feedback_v[0].abs() > Volt::new(0.5),
        "feedback_v[0] should be nonzero, got {:?}",
        telem.feedback_v[0]
    );
    assert!(
        telem.error_prop[0] > RotPerSec::new(0.9),
        "error_prop[0] should be ~1.0, got {:?}",
        telem.error_prop[0]
    );
}

#[test]
fn opposite_error_reduces_duty() {
    let gains = load_identity_gains(1.0, 0.0, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    // Negative error: setpoint = [0,0,0], gyro = [1,0,0] → error = [-1,0,0]
    let sensors = make_sensors([1.0, 0.0, 0.0], 12.0);
    let setpoint = make_setpoint([0.0, 0.0, 0.0], 0.5);

    let _output = ctrl.update(&sensors, &setpoint, &gains);

    // Motor 0 feedback voltage should be negative
    let telem = ctrl.telemetry();
    assert!(
        telem.feedback_v[0] < Volt::new(-0.5),
        "negative error should produce negative feedback, got {:?}",
        telem.feedback_v[0]
    );
}

#[test]
fn integrator_accumulates_over_iterations() {
    let gains = load_identity_gains(0.0, 1.0, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint = make_setpoint([1.0, 0.0, 0.0], 0.5);

    // Run several iterations to let the leaky integrator accumulate
    let mut prev_integral = Rotation::new(0.0);
    for i in 0..50 {
        ctrl.update(&sensors, &setpoint, &gains);
        let current = ctrl.telemetry().error_integral[0];
        if i > 0 {
            assert!(
                current >= prev_integral,
                "integral should grow: iter {i}, prev={prev_integral:?}, cur={current:?}"
            );
        }
        prev_integral = current;
    }

    assert!(
        prev_integral > RotPerSec::new(0.01) * DT,
        "integral should have accumulated, got {:?}",
        prev_integral
    );
}

#[test]
fn integrator_clamps() {
    let gains = load_identity_gains(0.0, 1.0, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint = make_setpoint([100.0, 0.0, 0.0], 0.5); // huge error

    // Run many iterations — integrator should clamp at ±0.25 rot/s·dt
    for _ in 0..10000 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    let integral = ctrl.telemetry().error_integral[0];
    let limit = RotPerSec::new(0.25) * DT;
    assert!(
        (integral - limit).abs() <= Rotation::new(1e-9),
        "integral should clamp at {limit:?}, got {:?}",
        integral
    );
}

#[test]
fn integral_voltage_bounded() {
    let gains = load_identity_gains(0.0, 1e6, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint = make_setpoint([1.0, 0.0, 0.0], 0.5);
    for _ in 0..10000 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    let fb = ctrl.telemetry().feedback_v;
    let limit = 0.2 * BATTERY;
    assert!(fb.iter().all(|v| v.abs() <= limit * (1.0 + 1e-5)), "integral voltage {fb:?} exceeds {limit:?}");
    assert!(fb.iter().any(|v| v.abs() > 0.5 * limit), "integral didn't build: {fb:?}");
}

#[test]
fn derivative_responds_to_error_change() {
    let gains = load_identity_gains(0.0, 0.0, 1.0, 0.0);
    let mut ctrl = ControllerState::new();

    // First iteration: setpoint=[0,0,0], gyro=[0,0,0] → error=[0,0,0]
    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint = make_setpoint([0.0, 0.0, 0.0], 0.5);
    ctrl.update(&sensors, &setpoint, &gains);

    // Second iteration: error jumps to [1,0,0] → derivative should be nonzero
    let setpoint2 = make_setpoint([1.0, 0.0, 0.0], 0.5);
    ctrl.update(&sensors, &setpoint2, &gains);

    let deriv = ctrl.telemetry().error_derivative[0];
    assert!(
        deriv.abs() > RotPerSecSq::new(0.1),
        "derivative should respond to error change, got {:?}",
        deriv
    );
}

#[test]
fn feedforward_responds_to_setpoint_change() {
    let gains = load_identity_gains(0.0, 0.0, 0.0, 1.0);
    let mut ctrl = ControllerState::new();

    // First iteration: setpoint=[0,0,0]
    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint0 = make_setpoint([0.0, 0.0, 0.0], 0.5);
    ctrl.update(&sensors, &setpoint0, &gains);

    // Second iteration: setpoint jumps to [1,0,0]
    let setpoint1 = make_setpoint([1.0, 0.0, 0.0], 0.5);
    ctrl.update(&sensors, &setpoint1, &gains);

    let ff = ctrl.telemetry().feedforward_v[0];
    assert!(
        ff.abs() > Volt::new(0.01),
        "feedforward should respond to setpoint step, got {:?}",
        ff
    );
}

#[test]
fn reset_clears_state() {
    let gains = load_identity_gains(1.0, 1.0, 1.0, 1.0);
    let mut ctrl = ControllerState::new();

    // Run several iterations to build up integrator/derivative state
    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint = make_setpoint([1.0, 0.5, 0.0], 0.5);
    for _ in 0..100 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    // State should be nonzero
    let integral_before = ctrl.telemetry().error_integral[0];
    assert!(integral_before.abs() > RotPerSec::new(0.001) * DT);

    // Reset
    ctrl.reset();

    // Run one update with zero error to read telemetry
    let sensors_match = make_sensors([1.0, 0.5, 0.0], 12.0);
    ctrl.update(&sensors_match, &setpoint, &gains);

    // After reset, integral should be near zero (only one iter of accumulation)
    let integral_after = ctrl.telemetry().error_integral[0];
    assert!(
        integral_after.abs() < integral_before.abs() * 0.1,
        "integral should reset: before={integral_before:?}, after={integral_after:?}"
    );
}

#[test]
fn output_clamped_to_unit_interval() {
    let gains = load_identity_gains(100.0, 0.0, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    // Enormous error with huge Kp should produce enormous voltage,
    // but output duty should clamp to [0, 1]
    let sensors = make_sensors([0.0; 3], 12.0);
    let setpoint = make_setpoint([100.0, 100.0, 100.0], 1.0);
    let output = ctrl.update(&sensors, &setpoint, &gains);

    for i in 0..4 {
        assert!(
            output.motor_duty[i] >= 0.0 && output.motor_duty[i] <= 1.0,
            "motor[{i}] = {} should be in [0, 1]",
            output.motor_duty[i]
        );
    }
}

#[test]
fn low_battery_guard() {
    let gains = load_identity_gains(1.0, 0.0, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    // Battery voltage = 0 (should be guarded to 1.0 to prevent division by zero)
    let sensors = make_sensors([0.0; 3], 0.0);
    let setpoint = make_setpoint([1.0, 0.0, 0.0], 0.5);
    let output = ctrl.update(&sensors, &setpoint, &gains);

    // Should not panic or produce NaN/Inf
    for i in 0..4 {
        assert!(
            output.motor_duty[i].is_finite(),
            "motor[{i}] should be finite with zero battery, got {}",
            output.motor_duty[i]
        );
    }
}

#[test]
fn throttle_maps_to_collective() {
    let gains = load_identity_gains(0.0, 0.0, 0.0, 0.0);
    let mut ctrl = ControllerState::new();

    // All gains zero, zero error → output = pure throttle
    let sensors = make_sensors([0.0; 3], 12.0);

    let out_low = ctrl.update(&sensors, &make_setpoint([0.0; 3], 0.0), &gains);
    ctrl.reset();
    let out_mid = ctrl.update(&sensors, &make_setpoint([0.0; 3], 0.5), &gains);
    ctrl.reset();
    let out_high = ctrl.update(&sensors, &make_setpoint([0.0; 3], 1.0), &gains);

    assert!((out_low.motor_duty[0] - 0.00).abs() < 1e-4);
    assert!((out_mid.motor_duty[0] - 0.50).abs() < 1e-4);
    assert!((out_high.motor_duty[0] - 1.00).abs() < 1e-4);
}
