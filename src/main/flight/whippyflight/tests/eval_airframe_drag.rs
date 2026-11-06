//! Airframe drag rejection on the nonlinear plant.  Scaffolding: the mode is
//! not validated by flight data yet, so only the blob-format tests run by
//! default; the flight tests and the eval are ignored:
//!
//!   cargo test --release --target aarch64-apple-darwin --features std \
//!       --test eval_airframe_drag -- --ignored --nocapture
//!
//! The plant hovers at zero rate setpoint while rotor drag torques pitch and
//! roll as α = C·a, a being the body x/y specific force (g) of a scripted
//! forward run: accelerate, cruise, brake hard, with a weaving turn.  The
//! accelerometer reads a plus noise.  The controller flies test_gains.wfg
//! with no drag block (feedback alone rejects the torque), and with the
//! block at the true C and at mis-fit C.

mod common;

use common::plant::Plant;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::types::*;

static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn battery_v() -> f32 {
    common::plant::model().battery_v()
}

/// Hover throttle of the plant model.
fn throttle() -> f32 {
    common::plant::model().throttle_at(1.0)
}
const LP_HZ: f32 = 5.0;
/// Plant drag torque, rot/s² per g: [roll←ax, roll←ay, pitch←ax, pitch←ay],
/// the size of the Purple Haze roll and Mako pitch fits.
const C_TRUE: [f64; 4] = [-4.5, -17.4, 10.6, 4.1];
const T_END: f64 = 2.6;
const SETTLE_S: f64 = 0.3;

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

/// Body x/y specific force (g) of the scripted run.
fn run_accel(t: f64) -> [f64; 2] {
    let ax = -1.0 * s_curve((t - SETTLE_S) / 0.3)
        + 1.8 * s_curve((t - 1.4) / 0.15)
        - 0.8 * s_curve((t - 2.0) / 0.3);
    let weave = s_curve((t - 0.5) / 0.2) - s_curve((t - 1.9) / 0.2);
    let ay = 0.6 * weave * (2.0 * core::f64::consts::PI * 1.5 * t).sin();
    [ax, ay]
}

fn load(drag: Option<[f32; 4]>) -> GainStorage {
    let blob = match drag {
        Some(c) => common::with_airframe_drag(TEST_WFG, [LP_HZ, c[0], c[1], c[2], c[3]]),
        None => TEST_WFG.to_vec(),
    };
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(&blob), &mut gains), GainLoadResult::Ok);
    gains
}

struct Result {
    rms: [f64; 2],
    peak: [f64; 2],
}

/// Fly the run; RMS and peak |rate| (rot/s) on roll and pitch.
fn fly(gains: &GainStorage) -> Result {
    let dt = gains.header.dt as f64;
    let mut ctrl = ControllerState::new();
    let mut plant = Plant::hovering(1.0);
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut noise = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let (mut sum_sq, mut peak, mut n) = ([0.0; 2], [0.0f64; 2], 0usize);
    for k in 0..(T_END / dt) as usize {
        let t = k as f64 * dt;
        let a = run_accel(t);
        let sensors = SensorData {
            gyro_rps: plant.gyro_bf(),
            motor_rps: plant.omega.map(|w| w as f32),
            battery_voltage: battery_v(),
            accel_g: [(a[0] + 0.1 * noise()) as f32, (a[1] + 0.1 * noise()) as f32, 1.0],
        };
        let out = ctrl.update(&sensors, &SetpointData { rate_rps: [0.0; 3], throttle: throttle() }, gains);
        let volts = out.motor_duty.map(|d| (d * battery_v()) as f64);
        let torque = [C_TRUE[0] * a[0] + C_TRUE[1] * a[1], C_TRUE[2] * a[0] + C_TRUE[3] * a[1], 0.0];
        for _ in 0..4 {
            plant.rk4_with_torque(&volts, dt / 4.0, torque);
        }
        if t >= SETTLE_S {
            let g = plant.gyro_bf();
            for axis in 0..2 {
                sum_sq[axis] += (g[axis] as f64).powi(2);
                peak[axis] = peak[axis].max((g[axis] as f64).abs());
            }
            n += 1;
        }
    }
    Result { rms: sum_sq.map(|s| (s / n as f64).sqrt()), peak }
}

fn scaled(c: [f64; 4], s: f64) -> [f32; 4] {
    c.map(|x| (x * s) as f32)
}

#[test]
fn v7_drag_block_parses() {
    let gains = load(Some([1.0, 2.0, 3.0, 4.0]));
    assert_eq!(gains.airframe_drag, [LP_HZ, 1.0, 2.0, 3.0, 4.0]);
    let plain = load(None);
    assert_eq!(plain.airframe_drag, [0.0; 5]);
    assert_eq!(plain.thrust_balance, gains.thrust_balance);
    assert_eq!(plain.motor_drag, gains.motor_drag);
    assert_eq!(plain.target_limits, gains.target_limits);
    assert!(plain.target_limits.iter().all(|&x| x > 0.0), "{:?}", plain.target_limits);
}

#[test]
fn v6_blob_loads_with_drag_off() {
    static LEGACY_WFG: &[u8] = include_bytes!("../tools/legacy_test_gains.wfg");
    assert_eq!(u16::from_le_bytes([LEGACY_WFG[4], LEGACY_WFG[5]]), 6);
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(LEGACY_WFG), &mut gains), GainLoadResult::Ok);
    assert_eq!(gains.airframe_drag, [0.0; 5]);
    assert_eq!(gains.target_limits, [0.0; 2]);
}

#[test]
#[ignore = "airframe drag is scaffolding until flight data supports it"]
fn zero_drag_block_flies_like_none() {
    let off = fly(&load(None));
    let zero = fly(&load(Some([0.0; 4])));
    for axis in 0..2 {
        assert!((off.rms[axis] - zero.rms[axis]).abs() < 1e-9 * off.rms[axis].max(1e-9));
    }
}

#[test]
#[ignore = "airframe drag is scaffolding until flight data supports it"]
fn drag_model_rejects_rotor_drag() {
    let off = fly(&load(None));
    let on = fly(&load(Some(scaled(C_TRUE, 1.0))));
    for axis in 0..2 {
        assert!(on.rms[axis] < 0.5 * off.rms[axis],
                "axis {axis}: rms {:.4} with the drag model vs {:.4} without", on.rms[axis], off.rms[axis]);
    }
}

#[test]
#[ignore]
fn airframe_drag_eval() {
    println!("forward run with weave, zero rate setpoint; |rate| in deg/s (roll, pitch)");
    println!("  {:28} {:>15} {:>15}", "", "rms", "peak");
    let cases: [(&str, Option<[f32; 4]>); 6] = [
        ("no drag block", None),
        ("drag block, true C", Some(scaled(C_TRUE, 1.0))),
        ("drag block, 0.7 × C", Some(scaled(C_TRUE, 0.7))),
        ("drag block, 1.3 × C", Some(scaled(C_TRUE, 1.3))),
        ("drag block, roll terms only", Some([C_TRUE[0] as f32, C_TRUE[1] as f32, 0.0, 0.0])),
        ("drag block, wrong sign", Some(scaled(C_TRUE, -1.0))),
    ];
    for (name, drag) in cases {
        let r = fly(&load(drag));
        println!("  {name:28} {:6.2} {:6.2}   {:6.2} {:6.2}",
                 360.0 * r.rms[0], 360.0 * r.rms[1], 360.0 * r.peak[0], 360.0 * r.peak[1]);
    }
}
