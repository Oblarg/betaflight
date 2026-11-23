//! Tests for Betaflight interop: unit conversion, axes, motor mapping.
//!
//! The C glue hands BF-native values (deg/s, centivolts, eRPM/100, FLU
//! axes) to the FFI as [`BfSensorData`]; these tests convert them the same
//! way and check the controller sees the right rates, voltages and motor
//! speeds in FRD, and that the telemetry going back ([`BfTelemetry`]) is in
//! the units and axes BF logs.

mod common;

use whippyflight_betaflight::bf::*;
use whippyflight_betaflight::ffi::*;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::spec::BlobSpec;
use whippyflight_control::gains::{GainStorage, load_gains};
use whippyflight_control::types::*;
use whippyunits::api::rescale_f32;
use whippyunits::value;

// =========================================================================
// Conversion helpers
// =========================================================================

/// `motor_poles` of the motors these tests read.
const POLES: u8 = 14;

/// BF's readings as the glue passes them, converted as the FFI does.
fn bf_sensors(gyro: [DegPerSec; 3], motor_erpm: [HundredErpm; 4], battery: CentiVolt) -> SensorData {
    BfSensorData {
        gyro,
        motor_erpm,
        battery,
        accel_g: [0.0; 3],
        motor_poles: POLES,
    }
    .to_sensor_data()
}

/// `getRawSetpoint(axis)` (deg/s, FLU), converted as `whippyflight_setpoint_packet` does.
fn bf_setpoint_to_rps(rate: [DegPerSec; 3]) -> [RotPerSec; 3] {
    flu_frd(rate.map(rescale_f32))
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
// Test gain builder
// =========================================================================

/// Kp = diag(kp) on motors 0..3, every other table zero, for BF's props-in
/// quad X.
fn make_kp_gains(resolution: u8, kp: VoltPerRps, dt: Seconds) -> Vec<u8> {
    let mut spec = BlobSpec::new(resolution);
    spec.dt = value!(dt, s, f32);
    let kp = value!(kp, V / (rot / s), f32);
    BlobSpec::fill(&mut spec.kp, |motor, axis| if motor == axis { kp } else { 0.0 });
    spec.layout = quadx_layout(false);
    spec.encode()
}

fn load_test_gains() -> GainStorage {
    let blob = make_kp_gains(2, VoltPerRps::new(1.0), Seconds::new(0.000125));
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(&blob), &mut storage);
    assert_eq!(result, GainLoadResult::Ok);
    storage
}

/// eRPM/100 counts of a 14-pole motor at 200 rot/s: 12000 RPM × 7 pole pairs.
const ERPM_200: [HundredErpm; 4] = [HundredErpm::new(840.0); 4];
const MOTORS_200: [RotPerSec; 4] = [RotPerSec::new(200.0); 4];

// =========================================================================
// Unit conversion tests
// =========================================================================

#[test]
fn bf_dps_to_rps_conversion() {
    // 360 deg/s = 1 rot/s; BF's pitch (nose down) is FRD's −pitch.
    let rps = bf_sensors([360.0, 180.0, 0.0].map(DegPerSec::new), ERPM_200, CentiVolt::new(1680.0)).gyro_rps;
    let tol = RotPerSec::new(1e-6);
    assert!((rps[0] - RotPerSec::new(1.0)).abs() < tol, "360 dps should be 1 rps, got {:?}", rps[0]);
    assert!((rps[1] - RotPerSec::new(-0.5)).abs() < tol, "180 dps should be -0.5 rps, got {:?}", rps[1]);
    assert!(rps[2].abs() < tol, "0 dps should be 0 rps, got {:?}", rps[2]);
}

#[test]
fn bf_centivolts_to_volts() {
    let tol = Volt::new(1e-4);
    let volts = |cv: CentiVolt| bf_sensors([0.0; 3].map(DegPerSec::new), ERPM_200, cv).battery_voltage;
    // 1680 cV = 16.80 V (typical 4S battery)
    assert!((volts(CentiVolt::new(1680.0)) - Volt::new(16.80)).abs() < tol, "1680 cV should be 16.80 V, got {:?}", volts(CentiVolt::new(1680.0)));
    assert_eq!(volts(CentiVolt::new(0.0)), Volt::new(0.0));
    // 2520 cV = 25.20 V (6S battery)
    assert!((volts(CentiVolt::new(2520.0)) - Volt::new(25.20)).abs() < tol);
}

#[test]
fn bf_erpm_to_shaft_speed() {
    // 14 poles, 7 pole pairs: 350 eRPM/100 = 35000 eRPM = 5000 RPM = 83.3 rot/s.
    let speed = bf_sensors([0.0; 3].map(DegPerSec::new), [350.0, 0.0, 1.0, 840.0].map(HundredErpm::new), CentiVolt::new(1680.0)).motor_rps;
    let tol = RotPerSec::new(1e-4);
    assert!((speed[0] - RotPerSec::new(5000.0 / 60.0)).abs() < tol, "got {:?}", speed[0]);
    assert_eq!(speed[1], RotPerSec::new(0.0));
    assert!((speed[2] - RotPerSec::new(100.0 / 60.0 / 7.0)).abs() < RotPerSec::new(1e-6), "got {:?}", speed[2]);
    assert!((speed[3] - RotPerSec::new(200.0)).abs() < tol, "got {:?}", speed[3]);
}

/// The same eRPM is a slower shaft on a motor with more poles.
#[test]
fn shaft_speed_follows_the_pole_count() {
    let erpm = HundredErpm::new(840.0);
    let tol = RotPerSec::new(1e-4);
    assert!((mechanical_from_erpm(erpm, 14) - RotPerSec::new(200.0)).abs() < tol);
    assert!((mechanical_from_erpm(erpm, 12) - RotPerSec::new(1400.0 / 6.0)).abs() < tol);
    // An unset pole count reads as one pole pair.
    assert_eq!(mechanical_from_erpm(erpm, 0), mechanical_from_erpm(erpm, 2));
}

/// BF's gyro and setpoint are FLU: roll as is, pitch and yaw negated.
#[test]
fn bf_axes_are_flu() {
    let rps = bf_sensors([90.0, 180.0, 360.0].map(DegPerSec::new), ERPM_200, CentiVolt::new(1680.0)).gyro_rps;
    assert_eq!(rps.map(|r| value!(r, rot / s, f32)), [0.25, -0.5, -1.0]);
    let sp = bf_setpoint_to_rps([90.0, 180.0, 360.0].map(DegPerSec::new));
    assert_eq!(sp.map(|r| value!(r, rot / s, f32)), [0.25, -0.5, -1.0]);
}

/// Telemetry leaves in the units and axes BF logs it in: `pidData` floats
/// per logged count, `int16_t` fields rounded and saturated.
#[test]
fn bf_telemetry_encoding() {
    let mut t = *ControllerState::new().telemetry();
    t.error_prop = [RotPerSec::new(1.0), RotPerSec::new(-0.5), RotPerSec::new(0.001)];
    t.error_integral_rate = [RotPerSec::new(0.25); 3];
    t.error_derivative = [RotPerSecSq::new(1.0); 3];
    t.feedback_v = [Volt::new(1.25), Volt::new(-0.0004), Volt::new(40.0), Volt::new(f32::NAN)];
    t.setpoint_rps = [RotPerSec::new(0.0001); 3];
    t.ff_target_rps = [RotPerSec::new(0.5); 3];
    t.command_scale = 0.9876;
    let bf = BfTelemetry::encode(&t, [RotPerSec::new(-1.0); 3]);

    let close = |a: f32, b: f32| (a - b).abs() < 1e-3 * b.abs().max(1.0);
    let p = bf.pid_p.map(|x| value!(x, 0.1 * deg / s, f32));
    assert!(close(p[0], 3600.0) && close(p[1], 1800.0) && close(p[2], -3.6), "{p:?}");
    let i = bf.pid_i.map(|x| value!(x, 0.1 * deg / s, f32));
    assert!(close(i[0], 900.0) && close(i[1], -900.0) && close(i[2], -900.0), "{i:?}");
    let d = bf.pid_d.map(|x| value!(x, deg / s ^ 2, f32));
    assert!(close(d[0], 360.0) && close(d[1], -360.0) && close(d[2], -360.0), "{d:?}");
    let f = bf.pid_f.map(|x| value!(x, 0.1 * deg / s, f32));
    assert!(close(f[0], 0.36) && close(f[1], -0.36) && close(f[2], -0.36), "{f:?}");

    assert_eq!(bf.feedback.map(|v| value!(v, mV, i16)), [1250, 0, i16::MAX, 0]);
    assert_eq!(bf.raw_setpoint.map(|r| value!(r, 0.1 * deg / s, i16)), [-3600, 3600, 3600]);
    assert_eq!(value!(bf.command_scale, 0.001, i16), 988);
    assert_eq!(bf.pid_setpoint.map(|r| value!(r, deg / s, f32)), [180.0, -180.0, -180.0]);
}

/// `int16_t` fields round as `roundf` does, saturating, with NaN to 0.
#[test]
fn bf_debug_rounding() {
    let cases = [
        0.0, 0.4999999, 0.5, -0.5, 1.5, -2.5, 2.4999998, -7.5000005, 32766.5, 32767.4, -32768.4,
        1e6, -1e6, 1e12, -1e12, f32::MAX, f32::INFINITY, f32::NEG_INFINITY,
    ];
    for x in cases {
        let got = value!(to_debug(Volt::new(x)), V, i16);
        assert_eq!(got, libm::roundf(x) as i16, "{x}");
    }
    assert_eq!(value!(to_debug(Volt::new(f32::NAN)), V, i16), 0);
}

/// Each motor's effort leaves in mV (and 0.1 rev/s), on its own BF motor index.
#[test]
fn bf_effort_encoding() {
    let mut t = *ControllerState::new().telemetry();
    let per_motor = |base: f32| core::array::from_fn(|m| Volt::new(base + m as f32 * 0.001));
    t.feedback_p_v = per_motor(1.0);
    t.feedback_i_v = per_motor(-0.2);
    t.feedback_d_v = per_motor(0.03);
    t.feedforward_v = per_motor(2.0);
    t.affine_v = per_motor(3.0);
    t.schedule_jitter_v = per_motor(0.004);
    t.applied_v = per_motor(4.0);
    t.filtered_rps = core::array::from_fn(|m| RotPerSec::new(300.0 + m as f32));
    let bf = BfTelemetry::encode(&t, [RotPerSec::new(0.0); 3]);

    for (m, e) in bf.effort.iter().enumerate() {
        let m = m as i16;
        let mv = [e.p, e.i, e.d, e.feedforward, e.affine, e.schedule_jitter, e.applied]
            .map(|v| value!(v, mV, i16));
        assert_eq!(mv, [1000 + m, -200 + m, 30 + m, 2000 + m, 3000 + m, 4 + m, 4000 + m]);
        assert_eq!(value!(e.filtered_speed, 0.1 * rot / s, i16), 3000 + 10 * m);
    }
}

/// `BfTelemetry` matches `wf_bf_telemetry_t` in whippyflight_glue.h.
#[test]
fn bf_telemetry_layout() {
    use core::mem::{offset_of, size_of};
    assert_eq!(offset_of!(BfTelemetry, pid_p), 0);
    assert_eq!(offset_of!(BfTelemetry, pid_i), 12);
    assert_eq!(offset_of!(BfTelemetry, pid_d), 24);
    assert_eq!(offset_of!(BfTelemetry, pid_f), 36);
    assert_eq!(offset_of!(BfTelemetry, feedback), 48);
    assert_eq!(offset_of!(BfTelemetry, raw_setpoint), 56);
    assert_eq!(offset_of!(BfTelemetry, command_scale), 62);
    assert_eq!(offset_of!(BfTelemetry, pid_setpoint), 64);
    assert_eq!(offset_of!(BfTelemetry, effort), 76);
    assert_eq!(size_of::<BfTelemetry>(), 140);
    assert_eq!(offset_of!(BfEffort, p), 0);
    assert_eq!(offset_of!(BfEffort, i), 2);
    assert_eq!(offset_of!(BfEffort, d), 4);
    assert_eq!(offset_of!(BfEffort, feedforward), 6);
    assert_eq!(offset_of!(BfEffort, affine), 8);
    assert_eq!(offset_of!(BfEffort, schedule_jitter), 10);
    assert_eq!(offset_of!(BfEffort, applied), 12);
    assert_eq!(offset_of!(BfEffort, filtered_speed), 14);
    assert_eq!(size_of::<BfEffort>(), 16);
}

/// The glue's accelerometer reading is a count of g₀ on BF's FLU axes: at
/// rest BF reads +1 g₀ on z (up), FRD −g₀ (z down).
#[test]
fn bf_accel_g_to_mps2() {
    let accel = BfSensorData {
        gyro: [DegPerSec::new(0.0); 3],
        motor_erpm: [HundredErpm::new(0.0); 4],
        battery: CentiVolt::new(0.0),
        accel_g: [0.0, -0.5, 1.0],
        motor_poles: POLES,
    }
    .to_sensor_data()
    .accel
    .map(|a| value!(a, m / s ^ 2, f32));
    let expected = [0.0, 4.903325, -9.80665];
    for (a, e) in accel.into_iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{accel:?}");
    }
}

/// `BfSensorData` matches `wf_sensor_data_t` in whippyflight_glue.h.
#[test]
fn bf_sensor_data_layout() {
    use core::mem::{offset_of, size_of};
    assert_eq!(offset_of!(BfSensorData, gyro), 0);
    assert_eq!(offset_of!(BfSensorData, motor_erpm), 12);
    assert_eq!(offset_of!(BfSensorData, battery), 28);
    assert_eq!(offset_of!(BfSensorData, accel_g), 32);
    assert_eq!(offset_of!(BfSensorData, motor_poles), 44);
    assert_eq!(size_of::<BfSensorData>(), 48);
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
        let sensors = bf_sensors([0.0, 0.0, 0.0].map(DegPerSec::new), ERPM_200, CentiVolt::new(1680.0));
        let setpoint = SetpointData {
            rate_rps: bf_setpoint_to_rps([360.0, 0.0, 0.0].map(DegPerSec::new)),
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
        let sensors = bf_sensors([0.0, 0.0, 0.0].map(DegPerSec::new), ERPM_200, CentiVolt::new(1680.0));
        let setpoint = SetpointData {
            rate_rps: bf_setpoint_to_rps([0.0, 360.0, 0.0].map(DegPerSec::new)),
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
        let sensors = bf_sensors([0.0, 0.0, 0.0].map(DegPerSec::new), ERPM_200, CentiVolt::new(1680.0));
        let setpoint = SetpointData {
            rate_rps: bf_setpoint_to_rps([0.0, 0.0, 360.0].map(DegPerSec::new)),
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
        accel: [MetersPerSecSq::new(0.0); 3],
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
    // - Motor eRPM: [300, 310, 295, 305] eRPM/100
    // - Throttle: 0.45

    let bf_gyro = [2.0, -1.0, 0.5].map(DegPerSec::new);
    let bf_setpoint = [0.0, 0.0, 0.0].map(DegPerSec::new);
    let bf_battery = CentiVolt::new(1520.0);
    let bf_motor_erpm = [300.0, 310.0, 295.0, 305.0].map(HundredErpm::new);
    let bf_throttle: f32 = 0.45;

    let sensors = bf_sensors(bf_gyro, bf_motor_erpm, bf_battery);
    let setpoint = SetpointData {
        rate_rps: bf_setpoint_to_rps(bf_setpoint),
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
            (DSHOT_MIN..=DSHOT_MAX).contains(&ds),
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
    let sensors = bf_sensors([-720.0, 0.0, 0.0].map(DegPerSec::new), ERPM_200, CentiVolt::new(1680.0));
    let setpoint = SetpointData {
        rate_rps: bf_setpoint_to_rps([0.0, 0.0, 0.0].map(DegPerSec::new)),
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

// =========================================================================
// FFI: layout check and BF axes end to end
// =========================================================================

/// One test, since the FFI's state is global.
#[test]
fn ffi_checks_layout_and_speaks_bf_axes() {
    let blob = common::static_blob(&make_kp_gains(2, VoltPerRps::new(1.0), Seconds::new(0.000125)));
    let init = |mixer, reversed| unsafe { whippyflight_init(blob.as_ptr(), blob.len() as u32, mixer, reversed) };
    assert_eq!(init(MIXER_QUADX, true), GainLoadResult::LayoutMismatch, "props out");
    assert_eq!(init(MIXER_QUADX - 1, false), GainLoadResult::LayoutMismatch, "quad +");

    let slot_order = common::static_blob(&BlobSpec::new(2).encode());
    assert_eq!(
        unsafe { whippyflight_init(slot_order.as_ptr(), slot_order.len() as u32, MIXER_QUADX, false) },
        GainLoadResult::LayoutMismatch,
    );

    assert_eq!(init(MIXER_QUADX, false), GainLoadResult::Ok);
    // BF gyro pitching at +360 deg/s (nose down), sticks centred: BF's
    // own rate error is −360 deg/s, and the telemetry logs it so.
    let sensors = BfSensorData {
        gyro: [0.0, 360.0, 0.0].map(DegPerSec::new),
        motor_erpm: ERPM_200,
        battery: CentiVolt::new(1680.0),
        accel_g: [0.0, 0.0, 1.0],
        motor_poles: POLES,
    };
    unsafe { whippyflight_update(&sensors, 0.5, Micros::new(1000)) };
    let telemetry = unsafe { &*whippyflight_telemetry() };
    let p = telemetry.pid_p.map(|x| value!(x, 0.1 * deg / s, f32));
    assert!(p[0].abs() < 1.0 && (p[1] + 3600.0).abs() < 1.0 && p[2].abs() < 1.0, "{p:?}");
}
