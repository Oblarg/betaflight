//! Integration tests using real DARE-generated gains from test_gains.wfg.
//!
//! These tests load the actual .wfg blob produced by gen_wfg.py, run the
//! full controller pipeline (gain loading → interpolation → LQR feedback →
//! duty output), and verify physically meaningful behaviour:
//!
//! - Gains load successfully (CRC, header, tables)
//! - Hover produces roughly equal motor duty
//! - Pitch/roll/yaw disturbances produce correct differential motor response
//! - Output stays bounded [0, 1]
//! - Multiple loop iterations converge (integral/derivative state evolves)

mod common;

use common::units::{Rate, Time};
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage, REF_FLOATS};
use whippyflight_control::types::*;
use whippyunits::value;

/// Load the test_gains.wfg blob.
static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn load_test_gains() -> GainStorage {
    let mut storage = GainStorage::new();
    let result = load_gains(common::static_blob(TEST_WFG), &mut storage);
    assert_eq!(result, GainLoadResult::Ok, "Failed to load test_gains.wfg");
    assert!(storage.loaded);
    storage
}

const HOVER_RPS: RotPerSec = RotPerSec::new(275.0);
const FULL: Volt = Volt::new(16.8);
const ZERO_V: Volt = Volt::new(0.0);

fn rates(r: [f32; 3]) -> [RotPerSec; 3] {
    r.map(RotPerSec::new)
}

fn make_hover_sensors(battery_v: Volt) -> SensorData {
    SensorData {
        gyro_rps: rates([0.0, 0.0, 0.0]), // [roll, pitch, yaw] — no rotation
        motor_rps: [HOVER_RPS; 4],        // mid-range hover
        battery_voltage: battery_v,
        accel_g: [0.0; 3],
    }
}

/// Arm weights of the coefficient JSON test_gains.wfg was generated from:
/// [front, back, left, right] (gen_wfg arm_weights).
fn arm_weights() -> [f32; 4] {
    let m = common::plant::model();
    [m.w_front, m.w_back, m.w_left, m.w_right].map(|w| w as f32)
}

/// Is `x` between `a` and `b` (either order)?
fn between(x: f32, a: f32, b: f32) -> bool {
    x > a.min(b) - 1e-4 && x < a.max(b) + 1e-4
}

fn make_hover_setpoint() -> SetpointData {
    SetpointData {
        rate_rps: rates([0.0, 0.0, 0.0]),
        throttle: 0.5,
    }
}

/// |e| of an integral-error vector.
fn magnitude(e: [Rotation; 3]) -> Rotation {
    (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt()
}

/// For messages.
fn rot(x: Rotation) -> f32 {
    value!(x, rot, f32)
}

// ── Gain loading ────────────────────────────────────────────────────────

#[test]
fn real_gains_load_successfully() {
    let gains = load_test_gains();
    assert_eq!(gains.header.resolution, 3);
    assert!((gains.header.min_rps - RotPerSec::new(50.0)).abs() < RotPerSec::new(0.01));
    // gen_wfg derives the top of the grid from the motor fit on a full
    // battery, well above hover.
    let max_rps: Rate = gains.header.max_rps.lossy_into();
    assert!(max_rps > 1.5 * common::plant::model().hover_rps);
    assert!((gains.header.dt - Seconds::new(0.00025)).abs() < Seconds::new(1e-6));
    assert!((gains.header.integral_tau - Seconds::new(10.0)).abs() < Seconds::new(0.01));
    assert!((gains.header.derivative_tau - Seconds::new(0.02)).abs() < Seconds::new(0.01));
}

#[test]
fn real_gains_are_nonzero() {
    let gains = load_test_gains();
    let n = gains.header.resolution as usize;
    let table_len = n * n * n * n * 4 * 3;

    // Each table should have significant nonzero entries
    let ki_nz = (0..table_len).filter(|&i| gains.ki.get(i).abs() > VoltPerRot::new(1e-8)).count();
    let kd_nz = (0..table_len).filter(|&i| gains.kd.get(i).abs() > VoltPerRpsPerSec::new(1e-8)).count();
    let kp_nz = (0..table_len).filter(|&i| gains.kp.get(i).abs() > VoltPerRps::new(1e-8)).count();
    let binv_nz = (0..table_len).filter(|&i| gains.binv.get(i).abs() > VoltPerRps::new(1e-8)).count();

    assert!(ki_nz > table_len / 2, "Ki mostly zero: {ki_nz}/{table_len}");
    assert!(kd_nz > table_len / 2, "Kd mostly zero: {kd_nz}/{table_len}");
    assert!(kp_nz > table_len / 2, "Kp mostly zero: {kp_nz}/{table_len}");
    assert!(binv_nz > 0, "B⁻¹ entirely zero");
}

// ── Hover equilibrium ───────────────────────────────────────────────────

#[test]
fn hover_produces_roughly_equal_motors() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    let sensors = make_hover_sensors(FULL);
    let setpoint = make_hover_setpoint();

    // Run a few iterations to let state settle
    let mut output = ctrl.update(&sensors, &setpoint, &gains);
    for _ in 0..100 {
        output = ctrl.update(&sensors, &setpoint, &gains);
    }

    // In hover (zero error, zero rotation), feedback should be ~0, so duty
    // is the collective (throttle 0.5) along the thrust-balanced direction.
    // Speeds balance the lever arms, ω ∝ 1/√d, so e.g. front:back voltage
    // lies between √(d_B/d_F) (linear drag) and d_B/d_F (quadratic drag);
    // likewise left:right.
    let duties = &output.motor_duty;
    let mean = duties.iter().sum::<f32>() / 4.0;
    println!("Hover duties: [{:.4}, {:.4}, {:.4}, {:.4}], mean={:.4}",
             duties[0], duties[1], duties[2], duties[3], mean);

    assert!(mean > 0.3 && mean < 0.8,
            "Mean duty {mean} out of expected hover range");
    let [w_front, w_back, w_left, w_right] = arm_weights();
    let (fb, lr) = (w_back / w_front, w_right / w_left);
    let ratio = (duties[0] + duties[2]) / (duties[1] + duties[3]);
    assert!(between(ratio, fb.sqrt(), fb), "front/back duty ratio {ratio}, expected in (√{fb}, {fb})");
    let ratio = (duties[0] + duties[1]) / (duties[2] + duties[3]);
    assert!(between(ratio, lr.sqrt(), lr), "left/right duty ratio {ratio}, expected in (√{lr}, {lr})");
}

// ── Pitch disturbance ───────────────────────────────────────────────────

#[test]
fn pitch_disturbance_produces_differential_response() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    // Settle at hover first
    let hover_sensors = make_hover_sensors(FULL);
    let setpoint = make_hover_setpoint();
    for _ in 0..200 {
        ctrl.update(&hover_sensors, &setpoint, &gains);
    }

    // Inject positive pitch rate disturbance (nose up)
    // gyro reads positive pitch rate, but setpoint is zero → negative error
    let disturbed = SensorData {
        gyro_rps: rates([0.0, 0.5, 0.0]),  // [roll=0, pitch=+0.5 rot/s, yaw=0]
        ..hover_sensors
    };

    let output = ctrl.update(&disturbed, &setpoint, &gains);
    let d = &output.motor_duty;

    println!("Pitch disturbance duties: REAR_R={:.4}, FRONT_R={:.4}, REAR_L={:.4}, FRONT_L={:.4}",
             d[0], d[1], d[2], d[3]);

    // BF QUADX: [REAR_R=0, FRONT_R=1, REAR_L=2, FRONT_L=3]
    // Positive pitch rate (nose up) → need to correct by pitching nose down
    // → increase rear motors, decrease front motors (or vice versa).
    // The key test: front and rear pairs must be differential.
    let front_avg = (d[1] + d[3]) / 2.0;  // FRONT_R + FRONT_L
    let rear_avg  = (d[0] + d[2]) / 2.0;  // REAR_R + REAR_L

    assert!((front_avg - rear_avg).abs() > 0.01,
            "No differential response to pitch: front={front_avg:.4}, rear={rear_avg:.4}");

    // All outputs must be bounded
    for (i, duty) in d.iter().enumerate() {
        assert!(*duty >= 0.0 && *duty <= 1.0,
                "Motor {i} duty {duty} out of bounds");
    }
}

// ── Roll disturbance ────────────────────────────────────────────────────

#[test]
fn roll_disturbance_produces_differential_response() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    let hover_sensors = make_hover_sensors(FULL);
    let setpoint = make_hover_setpoint();
    for _ in 0..200 {
        ctrl.update(&hover_sensors, &setpoint, &gains);
    }

    // Inject positive roll disturbance
    let disturbed = SensorData {
        gyro_rps: rates([0.5, 0.0, 0.0]),  // [roll=+0.5 rot/s, pitch=0, yaw=0]
        ..hover_sensors
    };

    let output = ctrl.update(&disturbed, &setpoint, &gains);
    let d = &output.motor_duty;

    println!("Roll disturbance duties: REAR_R={:.4}, FRONT_R={:.4}, REAR_L={:.4}, FRONT_L={:.4}",
             d[0], d[1], d[2], d[3]);

    // BF QUADX: [REAR_R=0, FRONT_R=1, REAR_L=2, FRONT_L=3]
    // Roll disturbance → left and right pairs must be differential.
    let right_avg = (d[0] + d[1]) / 2.0;  // REAR_R + FRONT_R
    let left_avg  = (d[2] + d[3]) / 2.0;  // REAR_L + FRONT_L

    assert!((right_avg - left_avg).abs() > 0.01,
            "No differential response to roll: right={right_avg:.4}, left={left_avg:.4}");

    for (i, duty) in d.iter().enumerate() {
        assert!(*duty >= 0.0 && *duty <= 1.0,
                "Motor {i} duty {duty} out of bounds");
    }
}

// ── Yaw disturbance ────────────────────────────────────────────────────

#[test]
fn yaw_disturbance_produces_diagonal_response() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    let hover_sensors = make_hover_sensors(FULL);
    let setpoint = make_hover_setpoint();
    for _ in 0..200 {
        ctrl.update(&hover_sensors, &setpoint, &gains);
    }

    // Inject yaw disturbance
    let disturbed = SensorData {
        gyro_rps: rates([0.0, 0.0, 0.5]),  // [roll=0, pitch=0, yaw=+0.5 rot/s]
        ..hover_sensors
    };

    let output = ctrl.update(&disturbed, &setpoint, &gains);
    let d = &output.motor_duty;

    println!("Yaw disturbance duties: REAR_R={:.4}, FRONT_R={:.4}, REAR_L={:.4}, FRONT_L={:.4}",
             d[0], d[1], d[2], d[3]);

    // Yaw: diagonal pairs spin in the same direction in QUADX.
    // BF QUADX: REAR_R(0) & FRONT_L(3) share one spin direction,
    //           FRONT_R(1) & REAR_L(2) share the other.
    let diag_a = (d[0] + d[3]) / 2.0;  // REAR_R + FRONT_L
    let diag_b = (d[1] + d[2]) / 2.0;  // FRONT_R + REAR_L

    assert!((diag_a - diag_b).abs() > 0.01,
            "No diagonal differential for yaw: diag_a={diag_a:.4}, diag_b={diag_b:.4}");

    for (i, duty) in d.iter().enumerate() {
        assert!(*duty >= 0.0 && *duty <= 1.0,
                "Motor {i} duty {duty} out of bounds");
    }
}

// ── Convergence over time ───────────────────────────────────────────────

#[test]
fn integral_accumulates_under_sustained_error() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    // Small sustained roll command — gyro reads zero, so error persists.
    let sensors = make_hover_sensors(FULL);
    let setpoint = SetpointData {
        rate_rps: rates([0.02, 0.0, 0.0]),
        throttle: 0.5,
    };

    // Run one iteration to capture initial telemetry
    ctrl.update(&sensors, &setpoint, &gains);
    let early_mag = magnitude(ctrl.telemetry().error_integral);

    // Run many more iterations
    for _ in 0..2000 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    let telem = ctrl.telemetry();
    assert!(telem.status == 1, "Controller not active");

    let late_mag = magnitude(telem.error_integral);
    println!("Integral mag: after 1 iter={:.6}, after 2001 iters={:.6} rot", rot(early_mag), rot(late_mag));
    println!("Integral state: [{:.5}, {:.5}, {:.5}] rot",
             rot(telem.error_integral[0]), rot(telem.error_integral[1]), rot(telem.error_integral[2]));

    // The integral state must grow over time with sustained error.
    assert!(late_mag > early_mag * 5.0,
            "Integral didn't grow significantly: early={early_mag:?}, late={late_mag:?}");

    // The integral should be primarily in the roll axis (index 0)
    assert!(telem.error_integral[0].abs() > telem.error_integral[1].abs() * 10.0,
            "Integral not concentrated on roll axis");

    // Steady-state output should still produce a nonzero motor differential.
    let output = ctrl.update(&sensors, &setpoint, &gains);
    let right_avg = (output.motor_duty[0] + output.motor_duty[1]) / 2.0;
    let left_avg  = (output.motor_duty[2] + output.motor_duty[3]) / 2.0;
    let diff = (right_avg - left_avg).abs();
    println!("Steady-state roll differential: {diff:.6}");
    assert!(diff > 0.001, "No roll differential at steady state");
}

// ── Battery voltage scaling ─────────────────────────────────────────────

#[test]
fn lower_battery_increases_duty_for_same_correction() {
    let gains = load_test_gains();

    // Very small pitch disturbance (0.01 rot/s) — well within the linear
    // range so the duty doesn't clamp and the battery scaling is visible.
    // The collective (throttle·n) is battery-independent in duty, so compare
    // the correction alone: disturbed minus undisturbed after equal history.
    let setpoint = make_hover_setpoint();
    let correction_spread = |battery_v: Volt| {
        let sensors = make_hover_sensors(battery_v);
        let disturbed = SensorData { gyro_rps: rates([0.0, 0.01, 0.0]), ..sensors };
        let mut ctrl = ControllerState::new();
        for _ in 0..50 {
            ctrl.update(&sensors, &setpoint, &gains);
        }
        let mut baseline = ControllerState::new();
        for _ in 0..50 {
            baseline.update(&sensors, &setpoint, &gains);
        }
        let out = ctrl.update(&disturbed, &setpoint, &gains);
        let base = baseline.update(&sensors, &setpoint, &gains);
        (0..4).map(|m| (out.motor_duty[m] - base.motor_duty[m]).abs()).sum::<f32>()
    };
    let spread_high = correction_spread(FULL);
    let spread_low = correction_spread(Volt::new(14.0));

    println!("Spread @16.8V={spread_high:.6}, @14.0V={spread_low:.6}");
    assert!(spread_low > spread_high * 1.01,
            "Lower battery didn't increase duty spread: high={spread_high}, low={spread_low}");
}

// ── Reset clears state ──────────────────────────────────────────────────

#[test]
fn reset_clears_integral_and_derivative_state() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();

    // Build up integral state (small enough an error not to saturate, which
    // would hold the integrator)
    let sensors = make_hover_sensors(FULL);
    let error = RotPerSec::new(0.02);
    let setpoint = SetpointData {
        rate_rps: [error, RotPerSec::new(0.0), RotPerSec::new(0.0)],
        throttle: 0.5,
    };
    for _ in 0..500 {
        ctrl.update(&sensors, &setpoint, &gains);
    }

    let pre_integral_mag = magnitude(ctrl.telemetry().error_integral);
    // 500 ticks of a 0.02 rot/s error, each adding (dt/τ_int)·dt·e; allow half.
    let dt = gains.header.dt;
    let expected = 500.0 * f32::from(dt / gains.header.integral_tau) * dt * error;
    assert!(pre_integral_mag > 0.5 * expected, "No integral buildup before reset");

    // Reset
    ctrl.reset();

    // Run one hover iteration
    let hover_setpoint = make_hover_setpoint();
    ctrl.update(&sensors, &hover_setpoint, &gains);
    let post_integral_mag = magnitude(ctrl.telemetry().error_integral);

    println!("Integral: pre-reset={:.5}, post-reset={:.5} rot", rot(pre_integral_mag), rot(post_integral_mag));
    assert!(post_integral_mag < pre_integral_mag * 0.01,
            "Reset didn't clear integral: pre={pre_integral_mag:?}, post={post_integral_mag:?}");
}

// ── Affine linearizer ───────────────────────────────────────────────────

/// motorVelSq of the coefficient JSON (V per (eRPM/600)²).
fn k_omega_sq() -> VoltPerRpsSq {
    common::plant::model().motor_vel_sq.lossy_into()
}

#[test]
fn affine_block_is_loaded() {
    let gains = load_test_gains();
    assert!(gains.affine.iter().any(|&g| g != VoltPerRpsSq::new(0.0)), "v2 blob has an all-zero affine block");
}

#[test]
fn affine_cancels_quadratic_motor_term_at_hover() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();
    ctrl.update(&make_hover_sensors(FULL), &make_hover_setpoint(), &gains);

    let expected = -k_omega_sq() * HOVER_RPS * HOVER_RPS;
    for (m, &v) in ctrl.telemetry().affine_v.iter().enumerate() {
        assert!((v - expected).abs() < 1e-3 * expected.abs(),
                "motor {m}: affine {v:?}, expected {expected:?}");
    }
}

#[test]
fn affine_tracks_each_motor_speed() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();
    let speeds = [300.0, 250.0, 250.0, 300.0].map(RotPerSec::new);
    let sensors = SensorData { motor_rps: speeds, ..make_hover_sensors(FULL) };
    ctrl.update(&sensors, &make_hover_setpoint(), &gains);

    for (m, &v) in ctrl.telemetry().affine_v.iter().enumerate() {
        let expected = -k_omega_sq() * speeds[m] * speeds[m];
        assert!((v - expected).abs() < 1e-2 * expected.abs(),
                "motor {m}: affine {v:?}, expected ≈{expected:?}");
    }
}

#[test]
fn affine_uses_grid_clamped_speed() {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();
    let sensors = SensorData { motor_rps: [RotPerSec::new(0.0); 4], ..make_hover_sensors(FULL) };
    ctrl.update(&sensors, &make_hover_setpoint(), &gains);

    let min = gains.header.min_rps;
    let expected = -k_omega_sq() * min * min;
    for &v in ctrl.telemetry().affine_v.iter() {
        assert!((v - expected).abs() < 1e-2 * expected.abs());
    }
}

// ── Axis / motor conventions (regression frame) ─────────────────────────
//
// main.R fits raw blackbox columns, so these encode the logged quad:
// +roll accel from motors 2,3; +pitch from motors 0,2; +yaw from 1,2.

fn feedback_for_rate_error(axis: usize) -> [Volt; 4] {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();
    let mut setpoint = make_hover_setpoint();
    setpoint.rate_rps[axis] = RotPerSec::new(0.1);
    ctrl.update(&make_hover_sensors(FULL), &setpoint, &gains);
    ctrl.telemetry().feedback_v
}

#[test]
fn positive_roll_error_drives_motors_2_3() {
    let v = feedback_for_rate_error(0);
    assert!(v[2] > ZERO_V && v[3] > ZERO_V && v[0] < ZERO_V && v[1] < ZERO_V, "{v:?}");
}

#[test]
fn positive_pitch_error_drives_motors_0_2() {
    let v = feedback_for_rate_error(1);
    assert!(v[0] > ZERO_V && v[2] > ZERO_V && v[1] < ZERO_V && v[3] < ZERO_V, "{v:?}");
}

#[test]
fn positive_yaw_error_drives_motors_1_2() {
    let v = feedback_for_rate_error(2);
    assert!(v[1] > ZERO_V && v[2] > ZERO_V && v[0] < ZERO_V && v[3] < ZERO_V, "{v:?}");
}

// ── Reference-model feedforward ─────────────────────────────────────────

/// Stick flick amplitude: small enough that hover collective plus
/// feedforward stays off the rails (as tools/ff_reference.py assumes).
const FLICK: RotPerSec = RotPerSec::new(0.4);

/// Run the controller through a FLICK stick flick on `axis` (linear over
/// 30 ms, then PT3-smoothed at 190 Hz like BF's rc_smoothing at 500 Hz RX),
/// with the gyro following the setpoint so feedback stays idle and the
/// outputs never saturate (saturation bleeds the reference model).
/// Returns per-tick feedforward voltages.
fn stick_flick_feedforward(axis: usize, ticks: usize) -> Vec<[Volt; 4]> {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();
    let dt = gains.header.dt;
    let pt3_cutoff = Hertz::new(190.0 * 1.96);
    let a = 1.0 - (-2.0 * core::f32::consts::PI * f32::from(pt3_cutoff * dt)).exp();
    let mut s = [RotPerSec::new(0.0); 3];
    let mut out = Vec::with_capacity(ticks);
    for k in 0..ticks {
        let raw = FLICK * f32::from(k as f32 * dt / Seconds::new(0.03)).min(1.0);
        s[0] = s[0] + a * (raw - s[0]);
        s[1] = s[1] + a * (s[0] - s[1]);
        s[2] = s[2] + a * (s[1] - s[2]);
        let mut sp = make_hover_setpoint();
        sp.rate_rps[axis] = s[2];
        let mut sensors = make_hover_sensors(FULL);
        sensors.gyro_rps = sp.rate_rps;
        ctrl.update(&sensors, &sp, &gains);
        out.push(ctrl.telemetry().feedforward_v);
    }
    out
}

fn peak(ff: &[[Volt; 4]]) -> Volt {
    ff.iter().flatten().fold(ZERO_V, |m, v| m.max(v.abs()))
}

#[test]
fn blob_has_reference_model() {
    let gains = load_test_gains();
    assert!(gains.has_ref_model);
    assert_eq!(gains.header.ff_horizon, 8);
}

// Reference values: tools/ff_reference.py (float64, uniform 275 rps, h = 8).

#[test]
fn pitch_flick_feedforward_matches_reference() {
    let ff = stick_flick_feedforward(1, 1200);
    let p = peak(&ff);
    assert!((p - Volt::new(4.331)).abs() < Volt::new(0.22), "peak pitch feedforward {p:?}, reference 4.331 V");
    let last = ff.last().unwrap();
    assert!(last.iter().all(|v| v.abs() < Volt::new(0.03)), "settled feedforward {last:?}");
    // Direction: +pitch request drives motors 0,2 (as feedback does).
    let early = ff[40];
    assert!(early[0] > ZERO_V && early[2] > ZERO_V && early[1] < ZERO_V && early[3] < ZERO_V, "{early:?}");
}

#[test]
fn yaw_flick_feedforward_matches_reference() {
    let ff = stick_flick_feedforward(2, 1200);
    let p = peak(&ff);
    assert!((p - Volt::new(0.586)).abs() < Volt::new(0.03), "peak yaw feedforward {p:?}, reference 0.586 V");
    // Holding a yaw rate needs a steady drag-compensating voltage.
    let last = ff.last().unwrap();
    assert!(last.iter().all(|v| (v.abs() - Volt::new(0.069)).abs() < Volt::new(0.004)),
            "settled yaw feedforward {last:?}");
    assert!(last[1] > ZERO_V && last[2] > ZERO_V, "{last:?}");
}

#[test]
fn feedforward_has_no_nyquist_chatter() {
    let ff = stick_flick_feedforward(1, 1200);
    let du: Vec<Volt> = ff.windows(2).map(|w| w[1][0] - w[0][0]).collect();
    let opposite = |a: Volt, b: Volt| (a < ZERO_V && b > ZERO_V) || (a > ZERO_V && b < ZERO_V);
    let flips = du.windows(2).filter(|w| opposite(w[0], w[1]) && w[0].abs() > Volt::new(1e-4)).count();
    assert!(flips < du.len() / 20, "{flips} sign flips in {} ticks", du.len());
}

/// The reference model predicts what the plant does under the feedforward:
/// flying a small, unsaturated flick on the nonlinear plant, the feedback's
/// error (reference-model rate − gyro) stays a small fraction of the flick.
/// The model must include the affine linearizer's feedback through measured
/// motor speed, which acts as extra motor drag.
#[test]
fn reference_model_predicts_the_plant() {
    use common::plant::{duty_volts, model, Plant};
    use whippyflight_control::setpoint::SetpointEstimator;
    let gains = load_test_gains();
    let dt: Time = gains.header.dt.lossy_into();
    let amp = RotPerSec::new(0.1);
    for axis in [0usize, 1] {
        let mut ctrl = ControllerState::new();
        let mut est = SetpointEstimator::new();
        est.configure_for(&gains);
        let mut plant = Plant::hovering(1.0);
        let mut worst = RotPerSec::new(0.0);
        // Settle from the uniform start (not torque-free on unequal arms).
        let start = Time::new(1.0);
        for k in 0..f64::from((start + Time::new(0.5)) / dt) as usize {
            let t = k as f64 * dt;
            let now_us = common::micros64(t);
            if k % 8 == 0 {
                let mut stick = [RotPerSec::new(0.0); 3];
                stick[axis] = amp * f64::from((t - start) / Time::new(0.03)).clamp(0.0, 1.0) as f32;
                est.packet(stick, now_us, Seconds::new(0.002));
            }
            let (rate, target) = est.tick(now_us, dt.lossy_into());
            let sp = SetpointData { rate_rps: rate, throttle: model().throttle_at(1.0) };
            let s = SensorData { gyro_rps: plant.gyro_bf(), motor_rps: plant.motor_rps(),
                                 battery_voltage: model().battery_v(), accel_g: [0.0; 3] };
            let volts = duty_volts(ctrl.update_with_target(&s, &sp, target, &gains).motor_duty);
            assert_eq!(ctrl.telemetry().command_scale, 1.0, "saturated at {t:?}");
            for _ in 0..4 {
                plant.rk4(&volts, dt / 4.0);
            }
            if t >= start {
                worst = worst.max(ctrl.telemetry().error_prop[axis].abs());
            }
        }
        // Without the linearizer's feedback in the model: ≈5% here.
        assert!(worst <= 0.01 * amp, "axis {axis}: reference model off by {:.1}% of the flick",
                100.0 * f32::from(worst / amp));
    }
}

#[test]
fn reference_model_seeds_without_kick() {
    // Arming with the stick already deflected must not fire feedforward.
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();
    let mut sp = make_hover_setpoint();
    sp.rate_rps = rates([0.5, -0.3, 0.2]);
    ctrl.update(&make_hover_sensors(FULL), &sp, &gains);
    let ff = ctrl.telemetry().feedforward_v;
    // Only the steady hold term for the seeded rate (no step response).
    assert!(ff.iter().all(|v| v.abs() < Volt::new(1.0)), "{ff:?}");
}

#[test]
fn min_rps_floor_overrides_reported_speed() {
    // A dead/zero telemetry motor must be linearized at the floor, not 0.
    let gains = load_test_gains();
    let floor = RotPerSec::new(120.0);
    let stopped = [RotPerSec::new(0.0); 4];
    let run = |rps: [RotPerSec; 4], set_floor: bool| {
        let mut ctrl = ControllerState::new();
        if set_floor { ctrl.set_min_rps(floor); }
        let sensors = SensorData { motor_rps: rps, ..make_hover_sensors(FULL) };
        let mut sp = make_hover_setpoint();
        sp.rate_rps[1] = RotPerSec::new(0.1);
        ctrl.update(&sensors, &sp, &gains);
        *ctrl.telemetry()
    };
    let floored = run(stopped, true);
    let at_floor = run([floor; 4], false);
    assert_eq!(floored.feedback_v, at_floor.feedback_v);
    assert_eq!(floored.affine_v, at_floor.affine_v);

    // A floor below the grid minimum falls back to the grid minimum.
    let low = run(stopped, false);
    let mut ctrl = ControllerState::new();
    ctrl.set_min_rps(RotPerSec::new(1.0));
    let sensors = SensorData { motor_rps: stopped, ..make_hover_sensors(FULL) };
    let mut sp = make_hover_setpoint();
    sp.rate_rps[1] = RotPerSec::new(0.1);
    ctrl.update(&sensors, &sp, &gains);
    assert_eq!(ctrl.telemetry().feedback_v, low.feedback_v);
}

// ── Allocation (idle floor / collective) ────────────────────────────────

/// motorVel of the coefficient JSON (V per eRPM/600).
fn k_omega() -> VoltPerRps {
    common::plant::model().motor_vel.lossy_into()
}

fn hover_voltages(throttle: f32, motor_rps: RotPerSec, min_rps: RotPerSec) -> ([Volt; 4], LqrTelemetry) {
    let gains = load_test_gains();
    let mut ctrl = ControllerState::new();
    ctrl.set_min_rps(min_rps);
    let sensors = SensorData { motor_rps: [motor_rps; 4], ..make_hover_sensors(FULL) };
    let sp = SetpointData { throttle, ..make_hover_setpoint() };
    let out = ctrl.update(&sensors, &sp, &gains);
    (out.motor_duty.map(|d| d * FULL), *ctrl.telemetry())
}

#[test]
fn blob_has_motor_drag() {
    let gains = load_test_gains();
    assert!((gains.motor_drag.k_w - k_omega()).abs() < VoltPerRps::new(1e-9));
    assert!((gains.motor_drag.k_w2 - k_omega_sq()).abs() < VoltPerRpsSq::new(1e-12));
}

#[test]
fn zero_throttle_idles_at_linearization_floor() {
    let floor = RotPerSec::new(120.0);
    let (v, telem) = hover_voltages(0.0, floor, floor);
    let idle = k_omega() * floor + k_omega_sq() * floor * floor;
    let lowest = v.iter().fold(Volt::new(f32::INFINITY), |m, &x| m.min(x));
    assert!((lowest - idle).abs() < Volt::new(1e-3), "lowest motor {lowest:?}, idle {idle:?} ({v:?})");
    assert!(telem.collective_v > ZERO_V, "idle should lift collective: {:?}", telem.collective_v);
    assert_eq!(telem.command_scale, 1.0);
}

#[test]
fn blob_has_thrust_balance() {
    // Motors [FL, BL, FR, BR] hover at ω ∝ 1/√(d_fb·d_lr).
    let b = load_test_gains().thrust_balance;
    let [w_front, w_back, w_left, w_right] = arm_weights();
    let want = [w_front * w_left, w_back * w_left, w_front * w_right, w_back * w_right].map(|d| 1.0 / d.sqrt());
    for m in 1..4 {
        assert!((b[m] / b[0] - want[m] / want[0]).abs() < 1e-5, "{b:?} vs {want:?}");
    }
    assert!((b.iter().product::<f32>() - 1.0).abs() < 1e-5, "{b:?}");
}

#[test]
fn hover_is_torque_free() {
    // Settle the motors on the controller's own hover voltages (zero rate
    // error, so feedback is idle) and check the lever-arm-weighted thrusts
    // cancel on pitch and roll.  Throttles keep the speeds inside the grid
    // (≤ 500 rps at 16.8 V), where the linearizer sees the true speeds.
    let gains = load_test_gains();
    let [w_front, w_back, w_left, w_right] = arm_weights();
    let (k_w, k_w2) = (k_omega(), k_omega_sq());
    for throttle in [0.05f32, 0.1, 0.2] {
        let mut ctrl = ControllerState::new();
        let mut omega = [HOVER_RPS; 4];
        for _ in 0..50 {
            let sensors = SensorData { motor_rps: omega, ..make_hover_sensors(FULL) };
            let sp = SetpointData { throttle, ..make_hover_setpoint() };
            let v = ctrl.update(&sensors, &sp, &gains).motor_duty.map(|d| d * FULL);
            omega = v.map(|v| (-k_w + (k_w * k_w + 4.0 * k_w2 * v).sqrt()) / (2.0 * k_w2));
        }
        let sq = omega.map(|w| w * w);
        let pitch = w_front * (sq[0] + sq[2]) - w_back * (sq[1] + sq[3]);
        let roll = w_left * (sq[0] + sq[1]) - w_right * (sq[2] + sq[3]);
        let total = sq[0] + sq[1] + sq[2] + sq[3];
        assert!(pitch.abs() < 1e-3 * total && roll.abs() < 1e-3 * total,
                "throttle {throttle}: ω = {omega:?}, pitch {pitch:?}, roll {roll:?}");
    }
}

#[test]
fn quantized_tables_are_views_into_blob() {
    assert_eq!(u16::from_le_bytes([TEST_WFG[4], TEST_WFG[5]]), 7, "expected a v7 blob");
    let blob = common::static_blob(TEST_WFG);
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(blob, &mut gains), GainLoadResult::Ok);
    let range = blob.as_ptr_range();
    let tables = [gains.ki.data_ptr(), gains.kd.data_ptr(), gains.kp.data_ptr(), gains.binv.data_ptr(),
                  gains.ref_model.data_ptr()];
    for ptr in tables {
        assert!(range.contains(&ptr), "table was copied");
    }
}

#[test]
fn quantized_blend_matches_decoded_values() {
    let gains = load_test_gains();
    let point = [0usize, 13, 40, 80];
    let weight = [0.1f32, 0.2, 0.3, 0.4];
    let blended = gains.ref_model.blend::<REF_FLOATS>(&point, &weight);
    for (i, &b) in blended.iter().enumerate() {
        let direct: f32 = point.iter().zip(&weight)
            .map(|(&p, &w)| w * gains.ref_model.get(p * REF_FLOATS + i))
            .sum();
        assert!((b - direct).abs() <= 1e-5 * direct.abs().max(1e-3),
            "slot {i}: blend {b} vs decoded {direct}");
    }
}
