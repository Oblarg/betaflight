//! Tests for Betaflight interop: unit conversion, axis ordering, motor mapping.
//!
//! These tests simulate what the C glue layer does: take BF-native values
//! (deg/s, centivolts, eRPM, DShot) and convert them to WhippyFlight units
//! (rot/s, V, duty [0,1]) before calling the controller.

mod common;

use common::put;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{GainStorage, load_gains};
use whippyflight_control::types::*;
use whippyunits::api::rescale_f32;
use whippyunits::quantity;

// =========================================================================
// Conversion helpers (mirror what the C glue layer does)
// =========================================================================

/// Simulate BF C glue: gyro.gyroADCf[] (deg/s, raw C floats) → SensorData.gyro_rps.
fn bf_gyro_to_rps(dps: [f32; 3]) -> [RotPerSec; 3] {
    [
        dps[0] * BF_DPS_TO_RPS,
        dps[1] * BF_DPS_TO_RPS,
        dps[2] * BF_DPS_TO_RPS,
    ]
}

/// Simulate BF C glue: getSetpointRate(axis) (deg/s) → SetpointData.rate_rps.
fn bf_setpoint_to_rps(dps: [f32; 3]) -> [RotPerSec; 3] {
    bf_gyro_to_rps(dps) // same conversion
}

/// Simulate BF C glue: getBatteryVoltage() (uint16 centivolts) → volts.
fn bf_cv_to_volts(centivolts: u16) -> Volt {
    centivolts as f32 * BF_CENTIVOLTS_TO_V
}

/// Simulate BF C glue: escSensorData[i].rpm (eRPM/100) → mechanical rot/s.
///
/// In BF: `erpmToRpm(erpm) = erpm * ERPM_PER_LSB / (motorPoleCount / 2)`.
fn bf_erpm_to_rps(erpm_lsb: i16, pole_count: u8) -> RotPerSec {
    let rpm = (erpm_lsb as f32) * BF_ERPM_LSB / (pole_count as f32 / 2.0);
    rescale_f32(quantity!(rpm, rot / min, f32))
}

/// Simulate BF C glue: motor duty [0,1] → DShot value.
///
/// `dshot = DSHOT_MIN_THROTTLE + duty * (DSHOT_MAX_THROTTLE - DSHOT_MIN_THROTTLE)`
const DSHOT_MIN: f32 = 48.0;
const DSHOT_MAX: f32 = 2047.0;

fn duty_to_dshot(duty: f32) -> f32 {
    DSHOT_MIN + duty * (DSHOT_MAX - DSHOT_MIN)
}

fn dshot_to_duty(dshot: f32) -> f32 {
    (dshot - DSHOT_MIN) / (DSHOT_MAX - DSHOT_MIN)
}

// =========================================================================
// Test gain builder (reused from test_controller.rs)
// =========================================================================

/// Kp = diag(kp) on motors 0..3, every other table zero.
fn make_kp_gains(resolution: u8, kp: VoltPerRps, dt: Seconds) -> Vec<u8> {
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
    put(&mut buf, 0x14, Seconds::new(0.1));
    put(&mut buf, 0x18, Seconds::new(0.01));
    buf[0x1C..0x20].copy_from_slice(&(total_size as u32).to_le_bytes());

    // Tables in blob order ki, kd, kp, binv; ki, kd and binv stay zero.
    let mut offset = 0x20 + 2 * bytes_per_table;
    for _grid_point in 0..(floats_per_table / 12) {
        for motor in 0..4u32 {
            for axis in 0..3u32 {
                let val = if motor < 3 && motor == axis { kp } else { VoltPerRps::new(0.0) };
                put(&mut buf, offset, val);
                offset += 4;
            }
        }
    }

    let crc = common::crc32(&buf[..payload_size]);
    buf[payload_size..payload_size + 4].copy_from_slice(&crc.to_le_bytes());
    buf
}

fn load_test_gains() -> GainStorage {
    let blob = make_kp_gains(2, VoltPerRps::new(1.0), Seconds::new(0.000125));
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    storage
}

const MOTORS_200: [RotPerSec; 4] = [RotPerSec::new(200.0); 4];

// =========================================================================
// Unit conversion tests
// =========================================================================

#[test]
fn bf_dps_to_rps_conversion() {
    // 360 deg/s = 1 rot/s
    let rps = bf_gyro_to_rps([360.0, 180.0, 0.0]);
    let tol = RotPerSec::new(1e-6);
    assert!((rps[0] - RotPerSec::new(1.0)).abs() < tol, "360 dps should be 1 rps, got {:?}", rps[0]);
    assert!((rps[1] - RotPerSec::new(0.5)).abs() < tol, "180 dps should be 0.5 rps, got {:?}", rps[1]);
    assert!(rps[2].abs() < tol, "0 dps should be 0 rps, got {:?}", rps[2]);
}

#[test]
fn bf_centivolts_to_volts() {
    let tol = Volt::new(1e-4);
    // 1680 cV = 16.80 V (typical 4S battery)
    let v = bf_cv_to_volts(1680);
    assert!((v - Volt::new(16.80)).abs() < tol, "1680 cV should be 16.80 V, got {v:?}");

    // 0 cV = 0.0 V
    assert_eq!(bf_cv_to_volts(0), Volt::new(0.0));

    // 2520 cV = 25.20 V (6S battery)
    let v6s = bf_cv_to_volts(2520);
    assert!((v6s - Volt::new(25.20)).abs() < tol);
}

#[test]
fn bf_erpm_to_rps_conversion() {
    // Typical 14-pole motor (7 pole pairs)
    // eRPM/100 = 350 → eRPM = 35000 → mech RPM = 35000 / 7 = 5000 → rot/s = 83.33
    let rps = bf_erpm_to_rps(350, 14);
    let expected = RotPerSec::new(350.0 * 100.0 / 7.0 / 60.0); // 83.333...
    assert!(
        (rps - expected).abs() < RotPerSec::new(0.01),
        "eRPM/100=350 with 14 poles should be {expected:?}, got {rps:?}"
    );

    // Zero RPM
    assert_eq!(bf_erpm_to_rps(0, 14), RotPerSec::new(0.0));
}

#[test]
fn dshot_round_trip() {
    // Duty 0 → DShot 48 (DSHOT_MIN_THROTTLE)
    assert!((duty_to_dshot(0.0) - DSHOT_MIN).abs() < 0.01);
    // Duty 1 → DShot 2047 (DSHOT_MAX_THROTTLE)
    assert!((duty_to_dshot(1.0) - DSHOT_MAX).abs() < 0.01);
    // Round-trip
    for duty in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let recovered = dshot_to_duty(duty_to_dshot(duty));
        assert!(
            (recovered - duty).abs() < 1e-6,
            "duty {duty} round-trip failed: got {recovered}"
        );
    }
}

// =========================================================================
// Axis ordering tests
// =========================================================================

/// Verify that axis index 0 = roll, 1 = pitch, 2 = yaw.
/// We inject error on one axis at a time and check that the correct
/// telemetry axis responds.
#[test]
fn axis_order_roll_pitch_yaw() {
    let gains = load_test_gains();
    let big = RotPerSec::new(0.9);
    let small = RotPerSec::new(0.01);

    // BF: gyro.gyroADCf = [0, 0, 0] dps
    // BF: setpoint = [360, 0, 0] dps on ROLL (FD_ROLL=0)
    // After conversion: setpoint_rps = [1.0, 0.0, 0.0]
    {
        let mut ctrl = ControllerState::new();
        let sensors = SensorData {
            gyro_rps: bf_gyro_to_rps([0.0, 0.0, 0.0]),
            motor_rps: MOTORS_200,
            battery_voltage: bf_cv_to_volts(1680),
            accel_g: [0.0; 3],
        };
        let setpoint = SetpointData {
            rate_rps: bf_setpoint_to_rps([360.0, 0.0, 0.0]),
            throttle: 0.5,
        };
        ctrl.update(&sensors, &setpoint, &gains);

        let telem = ctrl.telemetry();
        // Roll error should be ~1.0 rot/s
        assert!(
            telem.error_prop[0].abs() > big,
            "roll error should be ~1.0, got {:?}", telem.error_prop[0]
        );
        // Pitch and yaw error should be ~0
        assert!(
            telem.error_prop[1].abs() < small,
            "pitch error should be ~0, got {:?}", telem.error_prop[1]
        );
        assert!(
            telem.error_prop[2].abs() < small,
            "yaw error should be ~0, got {:?}", telem.error_prop[2]
        );
    }

    // BF: setpoint = [0, 360, 0] dps on PITCH (FD_PITCH=1)
    {
        let mut ctrl = ControllerState::new();
        let sensors = SensorData {
            gyro_rps: bf_gyro_to_rps([0.0, 0.0, 0.0]),
            motor_rps: MOTORS_200,
            battery_voltage: bf_cv_to_volts(1680),
            accel_g: [0.0; 3],
        };
        let setpoint = SetpointData {
            rate_rps: bf_setpoint_to_rps([0.0, 360.0, 0.0]),
            throttle: 0.5,
        };
        ctrl.update(&sensors, &setpoint, &gains);

        let telem = ctrl.telemetry();
        assert!(
            telem.error_prop[1].abs() > big,
            "pitch error should be ~1.0, got {:?}", telem.error_prop[1]
        );
    }

    // BF: setpoint = [0, 0, 360] dps on YAW (FD_YAW=2)
    {
        let mut ctrl = ControllerState::new();
        let sensors = SensorData {
            gyro_rps: bf_gyro_to_rps([0.0, 0.0, 0.0]),
            motor_rps: MOTORS_200,
            battery_voltage: bf_cv_to_volts(1680),
            accel_g: [0.0; 3],
        };
        let setpoint = SetpointData {
            rate_rps: bf_setpoint_to_rps([0.0, 0.0, 360.0]),
            throttle: 0.5,
        };
        ctrl.update(&sensors, &setpoint, &gains);

        let telem = ctrl.telemetry();
        assert!(
            telem.error_prop[2].abs() > big,
            "yaw error should be ~1.0, got {:?}", telem.error_prop[2]
        );
    }
}

// =========================================================================
// Motor ordering tests
// =========================================================================

/// Verify that motor index 0 = REAR_R, 1 = FRONT_R, 2 = REAR_L, 3 = FRONT_L.
///
/// With identity-like gains (motor i maps to axis i for i<3), error on axis 0
/// (roll) should produce feedback voltage on motor 0 (REAR_R) only.
#[test]
fn motor_order_matches_bf_quadx() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    // Error on roll only → motor 0 (REAR_R) gets Kp voltage
    let sensors = SensorData {
        gyro_rps: [RotPerSec::new(0.0); 3],
        motor_rps: MOTORS_200,
        battery_voltage: Volt::new(16.8),
        accel_g: [0.0; 3],
    };
    let setpoint = SetpointData {
        rate_rps: [RotPerSec::new(1.0), RotPerSec::new(0.0), RotPerSec::new(0.0)], // roll error only
        throttle: 0.5,
    };
    ctrl.update(&sensors, &setpoint, &gains);

    let telem = ctrl.telemetry();
    // Motor 0 (REAR_R) should have feedback voltage from roll
    assert!(
        telem.feedback_v[0].abs() > Volt::new(0.5),
        "REAR_R (motor 0) should get roll feedback, got {:?}", telem.feedback_v[0]
    );
    // Motor 3 (FRONT_L) has no gain (only 3 motors mapped in identity matrix)
    assert!(
        telem.feedback_v[3].abs() < Volt::new(1e-6),
        "FRONT_L (motor 3) should get zero feedback (unmapped), got {:?}", telem.feedback_v[3]
    );
}

// =========================================================================
// Full round-trip: BF-native values → convert → controller → duty → DShot
// =========================================================================

#[test]
fn full_bf_round_trip() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    // Simulate BF native values for a typical hovering quad:
    // - Gyro: [2, -1, 0.5] deg/s  (near zero, slight drift)
    // - Setpoint: [0, 0, 0] deg/s (sticks centered)
    // - Battery: 1520 cV = 15.20V (3S discharged)
    // - Motor eRPM: [300, 310, 295, 305] eRPM/100, 14-pole motor
    // - Throttle: 0.45

    let bf_gyro_dps = [2.0f32, -1.0, 0.5];
    let bf_setpoint_dps = [0.0f32, 0.0, 0.0];
    let bf_battery_cv: u16 = 1520;
    let bf_motor_erpm: [i16; 4] = [300, 310, 295, 305];
    let pole_count: u8 = 14;
    let bf_throttle: f32 = 0.45;

    // Convert (what the C glue does)
    let sensors = SensorData {
        gyro_rps: bf_gyro_to_rps(bf_gyro_dps),
        motor_rps: bf_motor_erpm.map(|erpm| bf_erpm_to_rps(erpm, pole_count)),
        battery_voltage: bf_cv_to_volts(bf_battery_cv),
        accel_g: [0.0; 3],
    };
    let setpoint = SetpointData {
        rate_rps: bf_setpoint_to_rps(bf_setpoint_dps),
        throttle: bf_throttle,
    };

    // Verify conversions
    assert!((sensors.gyro_rps[0] - RotPerSec::new(2.0 / 360.0)).abs() < RotPerSec::new(1e-6));
    assert!((sensors.battery_voltage - Volt::new(15.20)).abs() < Volt::new(0.01));
    assert!(sensors.motor_rps[0] > RotPerSec::new(0.0));

    // Run controller
    let output = ctrl.update(&sensors, &setpoint, &gains);

    // Convert output to DShot (what the C glue does on the way back)
    let dshot_values: Vec<f32> = output.motor_duty.iter().map(|&d| duty_to_dshot(d)).collect();

    // All motors should produce valid DShot values
    for (i, &ds) in dshot_values.iter().enumerate() {
        assert!(
            ds >= DSHOT_MIN && ds <= DSHOT_MAX,
            "motor[{i}] DShot={ds} out of range [{DSHOT_MIN}, {DSHOT_MAX}]"
        );
    }

    // With centered sticks (zero setpoint) and slight gyro drift,
    // the controller should produce mostly-throttle output with small corrections
    for (i, &duty) in output.motor_duty.iter().enumerate() {
        assert!(
            (duty - bf_throttle).abs() < 0.1,
            "motor[{i}] duty={duty} should be near throttle={bf_throttle}"
        );
    }
}

/// Verify that negative gyro rates (quad spinning) convert correctly
/// and produce the right sign of correction.
#[test]
fn negative_gyro_converts_correctly() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    // BF: gyro = [-720, 0, 0] dps → -2 rot/s roll
    // BF: setpoint = [0, 0, 0] dps → error = 0 - (-2) = +2 rot/s
    let sensors = SensorData {
        gyro_rps: bf_gyro_to_rps([-720.0, 0.0, 0.0]),
        motor_rps: MOTORS_200,
        battery_voltage: bf_cv_to_volts(1680),
        accel_g: [0.0; 3],
    };
    let setpoint = SetpointData {
        rate_rps: bf_setpoint_to_rps([0.0, 0.0, 0.0]),
        throttle: 0.5,
    };
    ctrl.update(&sensors, &setpoint, &gains);

    let telem = ctrl.telemetry();
    // Roll error should be positive (want to stop the negative spin)
    assert!(
        telem.error_prop[0] > RotPerSec::new(1.5),
        "roll error should be ~2.0 (correcting -2 rps spin), got {:?}", telem.error_prop[0]
    );
}

/// The glue's conversion constants agree with whippyunits' own unit scales.
#[test]
fn conversion_constant_sanity() {
    let deg_per_sec: RotPerSec = rescale_f32(quantity!(1.0, deg / s, f32));
    assert!((BF_DPS_TO_RPS - deg_per_sec).abs() < RotPerSec::new(1e-9));

    let centivolt: Volt = rescale_f32(quantity!(1.0, cV, f32));
    assert!((BF_CENTIVOLTS_TO_V - centivolt).abs() < Volt::new(1e-9));

    // ERPM_PER_LSB = 100
    assert!((BF_ERPM_LSB - 100.0).abs() < 1e-6);
}
