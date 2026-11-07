//! Airframe drag rejection on the nonlinear plant.  Scaffolding: the mode is
//! not validated by flight data yet, so only the blob-format tests run by
//! default; the flight tests and the eval are ignored:
//!
//!   cargo test --release --target aarch64-apple-darwin --features std \
//!       --test eval_airframe_drag -- --ignored --nocapture
//!
//! The plant hovers at zero rate setpoint while rotor drag torques pitch and
//! roll as α = C·a, a being the body x/y specific force of a scripted
//! forward run: accelerate, cruise, brake hard, with a weaving turn.  The
//! accelerometer reads a plus noise.  The controller flies test_gains.wfg
//! with no drag block (feedback alone rejects the torque), and with the
//! block at the true C and at mis-fit C.

mod common;

use common::plant::{duty_volts, Plant};
use common::units::{Accel, AngAccel, Freq, PerMeter, Rate, SpeedSq, Time};
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, AirframeDrag, GainStorage, TargetLimits};
use whippyflight_control::types::*;
use whippyunits::value;

static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn battery_v() -> Volt {
    common::plant::model().battery_v()
}

/// Hover throttle of the plant model.
fn throttle() -> f32 {
    common::plant::model().throttle_at(1.0)
}
const LP: Hertz = Hertz::new(5.0);
fn g0() -> Accel {
    STANDARD_GRAVITY.lossy_into()
}

/// Plant drag [roll←ax, roll←ay, pitch←ax, pitch←ay], the size of the
/// Purple Haze roll and Mako pitch fits (given per g₀).
fn c_true() -> [PerMeter; 4] {
    [-4.5, -17.4, 10.6, 4.1].map(|c| AngAccel::new(c) / g0())
}
const T_END: Time = Time::new(2.6);
const SETTLE: Time = Time::new(0.3);
const WEAVE: Freq = Freq::new(1.5);

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

/// Body x/y specific force of the scripted run.
fn run_accel(t: Time) -> [Accel; 2] {
    let ramp = |start: Time, len: f64| s_curve(f64::from((t - start) / Time::new(len)));
    let ax = -1.0 * ramp(SETTLE, 0.3)
        + 1.8 * ramp(Time::new(1.4), 0.15)
        - 0.8 * ramp(Time::new(2.0), 0.3);
    let weave = ramp(Time::new(0.5), 0.2) - ramp(Time::new(1.9), 0.2);
    let ay = 0.6 * weave * f64::from(2.0 * core::f64::consts::PI * WEAVE * t).sin();
    [ax, ay].map(|g| g * g0())
}

fn drag_block(c: [RotPerMeter; 4]) -> AirframeDrag {
    AirframeDrag { lp: LP, roll: [c[0], c[1]], pitch: [c[2], c[3]] }
}

fn load(drag: Option<[RotPerMeter; 4]>) -> GainStorage {
    let blob = match drag {
        Some(c) => common::with_airframe_drag(TEST_WFG, LP, c),
        None => TEST_WFG.to_vec(),
    };
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(&blob), &mut gains), GainLoadResult::Ok);
    gains
}

struct Result {
    rms: [Rate; 2],
    peak: [Rate; 2],
}

/// Fly the run; RMS and peak |rate| on roll and pitch.
fn fly(gains: &GainStorage) -> Result {
    let dt: Time = gains.header.dt.lossy_into();
    let mut ctrl = ControllerState::new();
    let mut plant = Plant::hovering(1.0);
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut noise = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let (mut sum_sq, mut peak, mut n) = ([SpeedSq::new(0.0); 2], [Rate::new(0.0); 2], 0usize);
    for k in 0..f64::from(T_END / dt) as usize {
        let t = k as f64 * dt;
        let a = run_accel(t);
        let sensors = SensorData {
            gyro_rps: plant.gyro_bf(),
            motor_rps: plant.motor_rps(),
            battery_voltage: battery_v(),
            accel: [a[0] + 0.1 * noise() * g0(), a[1] + 0.1 * noise() * g0(), g0()].map(|x| x.lossy_into()),
        };
        let zero = [RotPerSec::new(0.0); 3];
        let out = ctrl.update(&sensors, &SetpointData { rate_rps: zero, throttle: throttle() }, gains);
        let volts = duty_volts(out.motor_duty);
        let c = c_true();
        let torque: [AngAccel; 3] = [c[0] * a[0] + c[1] * a[1], c[2] * a[0] + c[3] * a[1], AngAccel::new(0.0)];
        for _ in 0..4 {
            plant.rk4_with_torque(&volts, dt / 4.0, torque);
        }
        if t >= SETTLE {
            let g = plant.gyro_bf();
            for axis in 0..2 {
                let rate: Rate = g[axis].lossy_into();
                sum_sq[axis] = sum_sq[axis] + rate * rate;
                peak[axis] = peak[axis].max(rate.abs());
            }
            n += 1;
        }
    }
    Result { rms: sum_sq.map(|s| (s / n as f64).sqrt()), peak }
}

/// The blob's coefficients for `s` × the plant's drag `c`.
fn scaled(c: [PerMeter; 4], s: f64) -> [RotPerMeter; 4] {
    c.map(|x| (x * s).lossy_into())
}

#[test]
fn v7_drag_block_parses() {
    let c = [1.0, 2.0, 3.0, 4.0].map(RotPerMeter::new);
    let gains = load(Some(c));
    assert_eq!(gains.airframe_drag, drag_block(c));
    let plain = load(None);
    assert_eq!(plain.airframe_drag, AirframeDrag::OFF);
    assert_eq!(plain.thrust_balance, gains.thrust_balance);
    assert_eq!(plain.motor_drag, gains.motor_drag);
    assert_eq!(plain.target_limits, gains.target_limits);
    let limits = plain.target_limits;
    assert!(limits.slew > RotPerSecSq::new(0.0) && limits.jerk > RotPerSecCu::new(0.0), "{limits:?}");
}

#[test]
fn v6_blob_loads_with_drag_off() {
    static LEGACY_WFG: &[u8] = include_bytes!("../tools/legacy_test_gains.wfg");
    assert_eq!(u16::from_le_bytes([LEGACY_WFG[4], LEGACY_WFG[5]]), 6);
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(LEGACY_WFG), &mut gains), GainLoadResult::Ok);
    assert_eq!(gains.airframe_drag, AirframeDrag::OFF);
    assert_eq!(gains.target_limits, TargetLimits::DEFAULT);
}

#[test]
#[ignore = "airframe drag is scaffolding until flight data supports it"]
fn zero_drag_block_flies_like_none() {
    let off = fly(&load(None));
    let zero = fly(&load(Some([RotPerMeter::new(0.0); 4])));
    for axis in 0..2 {
        assert!((off.rms[axis] - zero.rms[axis]).abs() < 1e-9 * off.rms[axis].max(Rate::new(1e-9)));
    }
}

#[test]
#[ignore = "airframe drag is scaffolding until flight data supports it"]
fn drag_model_rejects_rotor_drag() {
    let off = fly(&load(None));
    let on = fly(&load(Some(scaled(c_true(), 1.0))));
    for axis in 0..2 {
        assert!(on.rms[axis] < 0.5 * off.rms[axis],
                "axis {axis}: rms {:?} with the drag model vs {:?} without", on.rms[axis], off.rms[axis]);
    }
}

#[test]
#[ignore]
fn airframe_drag_eval() {
    println!("forward run with weave, zero rate setpoint; |rate| in deg/s (roll, pitch)");
    println!("  {:28} {:>15} {:>15}", "", "rms", "peak");
    let c = c_true();
    let roll_only = scaled([c[0], c[1], PerMeter::new(0.0), PerMeter::new(0.0)], 1.0);
    let cases: [(&str, Option<[RotPerMeter; 4]>); 6] = [
        ("no drag block", None),
        ("drag block, true C", Some(scaled(c_true(), 1.0))),
        ("drag block, 0.7 × C", Some(scaled(c_true(), 0.7))),
        ("drag block, 1.3 × C", Some(scaled(c_true(), 1.3))),
        ("drag block, roll terms only", Some(roll_only)),
        ("drag block, wrong sign", Some(scaled(c_true(), -1.0))),
    ];
    let dps = |r: Rate| value!(r, deg / s);
    for (name, drag) in cases {
        let r = fly(&load(drag));
        println!("  {name:28} {:6.2} {:6.2}   {:6.2} {:6.2}",
                 dps(r.rms[0]), dps(r.rms[1]), dps(r.peak[0]), dps(r.peak[1]));
    }
}
