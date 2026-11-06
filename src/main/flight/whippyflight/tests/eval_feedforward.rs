//! Stick-flick tracking on the nonlinear plant, from gentle to saturating.
//! Not a regression test; run with
//!
//!   cargo test --release --target aarch64-apple-darwin --features std \
//!       --test eval_feedforward -- --ignored --nocapture
//!
//! Flies test_gains.wfg, or the blob at $WFG_PATH (zero its feedback gains
//! to see the feedforward + affine linearizer alone), through roll and pitch
//! flicks of 0.05–4 rot/s rising over 30 ms and 10 ms.  Each flick is flown
//! twice — with and without the stick — and the metrics are taken on the
//! difference.  `flick_summary` prints one SUMMARY line for
//! tools/sweep_qr.py.
//!
//! Overrides: FF_SLEW / FF_JERK (target slew rot/s² and jerk rot/s³ limits).

mod common;

use common::plant::Plant;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::setpoint::{target_limits, SetpointEstimator, DEFAULT_LAMBDA_500HZ};
use whippyflight_control::types::*;

static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn battery_v() -> f32 {
    common::plant::model().battery_v()
}

/// Hover throttle of the plant model.
fn throttle() -> f32 {
    common::plant::model().throttle_at(1.0)
}
const F_RX: f64 = 500.0;
const SETTLE_S: f64 = 0.2;
const HOLD_S: f64 = 0.3;
const REST_S: f64 = 0.3;
const AMPS: [f64; 6] = [0.05, 0.2, 0.5, 1.0, 2.0, 4.0];
const RISES: [f64; 2] = [0.030, 0.010];

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).map_or(default, |v| v.parse().unwrap_or_else(|_| panic!("{name}")))
}

struct Trace {
    gyro: Vec<f64>,
    setpoint: Vec<f64>,
    target: Vec<f64>,
    stick: Vec<f64>,
    saturated: Vec<bool>,
}

/// Fly a flick to `amp` rot/s on BF axis `axis`, rising over `rise_s`.
fn fly(gains: &GainStorage, axis: usize, amp: f64, rise_s: f64) -> Trace {
    let dt = gains.header.dt as f64;
    let stick = |t: f64| {
        let t = t - SETTLE_S;
        amp * (s_curve(t / rise_s) - s_curve((t - rise_s - HOLD_S) / rise_s))
    };
    let mut ctrl = ControllerState::new();
    let mut sp = SetpointEstimator::new();
    let (slew, jerk) = target_limits(gains);
    sp.configure(DEFAULT_LAMBDA_500HZ, env_f32("FF_SLEW", slew), env_f32("FF_JERK", jerk));
    let mut plant = Plant::hovering(1.0);
    let t_end = SETTLE_S + 2.0 * rise_s + HOLD_S + REST_S;
    let mut next_pkt = 0.0;
    let mut tr = Trace { gyro: vec![], setpoint: vec![], target: vec![], stick: vec![], saturated: vec![] };
    for k in 0..(t_end / dt) as usize {
        let t = k as f64 * dt;
        let now_us = (t * 1e6) as u32;
        if t >= next_pkt {
            let mut rc = [0.0f32; 3];
            rc[axis] = stick(t) as f32;
            sp.packet(rc, now_us, (1.0 / F_RX) as f32);
            next_pkt += 1.0 / F_RX;
        }
        let (rate, target) = sp.tick(now_us, dt as f32);
        let sensors = SensorData {
            gyro_rps: plant.gyro_bf(),
            motor_rps: plant.omega.map(|w| w as f32),
            battery_voltage: battery_v(),
            accel_g: [0.0; 3],
        };
        let out = ctrl.update_with_target(&sensors, &SetpointData { rate_rps: rate, throttle: throttle() },
                                          target, gains);
        let volts = out.motor_duty.map(|d| (d * battery_v()) as f64);
        for _ in 0..4 {
            plant.rk4(&volts, dt / 4.0);
        }
        if t >= SETTLE_S {
            tr.gyro.push(plant.gyro_bf()[axis] as f64);
            tr.setpoint.push(rate[axis] as f64);
            tr.target.push(target[axis] as f64);
            tr.stick.push(stick(t));
            tr.saturated.push(ctrl.telemetry().command_scale < 0.999);
        }
    }
    tr
}

struct Metrics {
    target_ovs: f64,
    target_lag_ms: f64,
    gyro_lag_ms: f64,
    rise90_ms: f64,
    overshoot: f64,
    hold_err: f64,
    track_rms: f64,
    residual: f64,
    sat_frac: f64,
}

fn measure(gains: &GainStorage, axis: usize, amp: f64, rise_s: f64) -> Metrics {
    let dt = gains.header.dt as f64;
    let base = fly(gains, axis, 0.0, rise_s);
    let run = fly(gains, axis, amp, rise_s);
    let y: Vec<f64> = run.gyro.iter().zip(&base.gyro).map(|(a, b)| a - b).collect();
    let idx = |t: f64| ((t / dt) as usize).min(y.len() - 1);
    let (release, end) = (idx(rise_s + HOLD_S), idx(2.0 * rise_s + HOLD_S));
    let rise90 = y[..release].iter().position(|&v| v >= 0.9 * amp).map_or(f64::INFINITY, |i| i as f64 * dt);
    let peak = y[..release].iter().cloned().fold(f64::MIN, f64::max);
    let hold = &y[idx(rise_s + HOLD_S - 0.1)..release];
    let track = (0..idx(2.0 * rise_s + HOLD_S + 0.05))
        .map(|i| (y[i] - run.setpoint[i]).powi(2))
        .sum::<f64>() / idx(2.0 * rise_s + HOLD_S + 0.05) as f64;
    let half = |v: &[f64]| v[..release].iter().position(|&x| x >= 0.5 * amp).unwrap_or(release) as f64 * dt;
    let t_stick = half(&run.stick);
    Metrics {
        target_ovs: run.target[..release].iter().cloned().fold(f64::MIN, f64::max) / amp - 1.0,
        target_lag_ms: 1e3 * (half(&run.target) - t_stick),
        gyro_lag_ms: 1e3 * (half(&y) - t_stick),
        rise90_ms: 1e3 * rise90,
        overshoot: peak / amp - 1.0,
        hold_err: hold.iter().sum::<f64>() / hold.len() as f64 / amp - 1.0,
        track_rms: track.sqrt() / amp,
        residual: y[idx(2.0 * rise_s + HOLD_S + 0.05)..].iter().fold(0.0f64, |m, v| m.max(v.abs())) / amp,
        sat_frac: run.saturated[..end].iter().filter(|&&s| s).count() as f64 / end as f64,
    }
}

fn load() -> GainStorage {
    let blob = match std::env::var("WFG_PATH") {
        Ok(path) => std::fs::read(path).expect("read $WFG_PATH"),
        Err(_) => TEST_WFG.to_vec(),
    };
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(&blob), &mut gains), GainLoadResult::Ok);
    gains
}

/// Every (axis, rise, amplitude) flick, in table order.
fn all_flicks(gains: &GainStorage) -> Vec<(usize, f64, f64, Metrics)> {
    let mut out = vec![];
    for axis in [0, 1] {
        for rise_s in RISES {
            for amp in AMPS {
                out.push((axis, rise_s, amp, measure(gains, axis, amp, rise_s)));
            }
        }
    }
    out
}

#[test]
#[ignore]
fn feedforward_tracking() {
    let gains = load();
    println!("h = {} ticks, vs nonlinear plant", gains.header.ff_horizon);
    let mut last = (usize::MAX, 0.0);
    for (axis, rise_s, amp, m) in all_flicks(&gains) {
        if (axis, rise_s) != last {
            println!("\n{} flick over {:.0} ms", ["roll", "pitch"][axis], rise_s * 1e3);
            println!("  {:>6} {:>8} {:>8} {:>8} {:>8} {:>10} {:>9} {:>10} {:>9} {:>6}",
                     "amp", "tgt ovs", "tgt lag", "gyro lag", "rise90", "overshoot", "hold err", "track rms", "residual", "sat");
            last = (axis, rise_s);
        }
        println!("  {amp:6.2} {:+7.1}% {:6.2}ms {:6.2}ms {:6.1}ms {:+9.1}% {:+8.1}% {:9.1}% {:8.1}% {:5.1}%",
                 100.0 * m.target_ovs, m.target_lag_ms, m.gyro_lag_ms,
                 m.rise90_ms, 100.0 * m.overshoot, 100.0 * m.hold_err,
                 100.0 * m.track_rms, 100.0 * m.residual, 100.0 * m.sat_frac);
    }
}

/// Worst cases over roll/pitch and both rise times: big flicks are ≥ 1 rot/s
/// (saturating), small ones below.  `track_small` is the 0.2 rot/s, 30 ms
/// flick's RMS tracking error.
#[test]
#[ignore]
fn flick_summary() {
    let gains = load();
    let (mut big_ovs, mut big_rise, mut small_ovs, mut track_small) = (f64::MIN, f64::MIN, f64::MIN, f64::MIN);
    for (_, rise_s, amp, m) in all_flicks(&gains) {
        if amp >= 1.0 {
            big_ovs = big_ovs.max(m.overshoot);
            big_rise = big_rise.max(m.rise90_ms);
        } else {
            small_ovs = small_ovs.max(m.overshoot);
        }
        if amp == 0.2 && rise_s == 0.030 {
            track_small = track_small.max(m.track_rms);
        }
    }
    println!("SUMMARY flick_big_overshoot={big_ovs:.3} flick_big_rise90_ms={big_rise:.1} \
              flick_small_overshoot={small_ovs:.3} flick_track_small={track_small:.3}");
}
