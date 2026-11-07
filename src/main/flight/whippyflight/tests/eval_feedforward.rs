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

use common::plant::{duty_volts, Plant};
use common::units::{Freq, Rate, SpeedSq, Time};
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::setpoint::{target_limits, SetpointEstimator, DEFAULT_LAMBDA_500HZ};
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
const F_RX: Freq = Freq::new(500.0);
const SETTLE: Time = Time::new(0.2);
const HOLD: Time = Time::new(0.3);
const REST: Time = Time::new(0.3);
const AMPS: [Rate; 6] =
    [Rate::new(0.05), Rate::new(0.2), Rate::new(0.5), Rate::new(1.0), Rate::new(2.0), Rate::new(4.0)];
const RISES: [Time; 2] = [Time::new(0.030), Time::new(0.010)];

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().map(|v| v.parse().unwrap_or_else(|_| panic!("{name}")))
}

fn ms(t: Time) -> f64 {
    value!(t, ms)
}

struct Trace {
    gyro: Vec<Rate>,
    setpoint: Vec<Rate>,
    target: Vec<Rate>,
    stick: Vec<Rate>,
    saturated: Vec<bool>,
}

/// Fly a flick to `amp` on BF axis `axis`, rising over `rise`.
fn fly(gains: &GainStorage, axis: usize, amp: Rate, rise: Time) -> Trace {
    let dt: Time = gains.header.dt.lossy_into();
    let stick = |t: Time| {
        let t = t - SETTLE;
        amp * (s_curve(f64::from(t / rise)) - s_curve(f64::from((t - rise - HOLD) / rise)))
    };
    let mut ctrl = ControllerState::new();
    let mut sp = SetpointEstimator::new();
    let (slew, jerk) = target_limits(gains);
    sp.configure(
        DEFAULT_LAMBDA_500HZ,
        env_f32("FF_SLEW").map_or(slew, RotPerSecSq::new),
        env_f32("FF_JERK").map_or(jerk, RotPerSecCu::new),
    );
    let mut plant = Plant::hovering(1.0);
    let t_end = SETTLE + 2.0 * rise + HOLD + REST;
    let mut next_pkt = Time::new(0.0);
    let mut tr = Trace { gyro: vec![], setpoint: vec![], target: vec![], stick: vec![], saturated: vec![] };
    for k in 0..f64::from(t_end / dt) as usize {
        let t = k as f64 * dt;
        let now_us = common::micros64(t);
        if t >= next_pkt {
            let mut rc = [RotPerSec::new(0.0); 3];
            rc[axis] = stick(t).lossy_into();
            sp.packet(rc, now_us, (1.0 / F_RX).lossy_into());
            next_pkt = next_pkt + 1.0 / F_RX;
        }
        let (rate, target) = sp.tick(now_us, dt.lossy_into());
        let sensors = SensorData {
            gyro_rps: plant.gyro_bf(),
            motor_rps: plant.motor_rps(),
            battery_voltage: battery_v(),
            accel: [MetersPerSecSq::new(0.0); 3],
        };
        let out = ctrl.update_with_target(&sensors, &SetpointData { rate_rps: rate, throttle: throttle() },
                                          target, gains);
        let volts = duty_volts(out.motor_duty);
        for _ in 0..4 {
            plant.rk4(&volts, dt / 4.0);
        }
        if t >= SETTLE {
            tr.gyro.push(plant.gyro_bf()[axis].lossy_into());
            tr.setpoint.push(rate[axis].lossy_into());
            tr.target.push(target[axis].lossy_into());
            tr.stick.push(stick(t));
            tr.saturated.push(ctrl.telemetry().command_scale < 0.999);
        }
    }
    tr
}

struct Metrics {
    target_ovs: f64,
    target_lag: Time,
    gyro_lag: Time,
    rise90: Time,
    overshoot: f64,
    hold_err: f64,
    track_rms: f64,
    residual: f64,
    sat_frac: f64,
}

fn measure(gains: &GainStorage, axis: usize, amp: Rate, rise: Time) -> Metrics {
    let dt: Time = gains.header.dt.lossy_into();
    let base = fly(gains, axis, Rate::new(0.0), rise);
    let run = fly(gains, axis, amp, rise);
    let y: Vec<Rate> = run.gyro.iter().zip(&base.gyro).map(|(&a, &b)| a - b).collect();
    let idx = |t: Time| (f64::from(t / dt) as usize).min(y.len() - 1);
    let (release, end) = (idx(rise + HOLD), idx(2.0 * rise + HOLD));
    let rise90 = y[..release].iter().position(|&v| v >= 0.9 * amp).map_or(Time::new(f64::INFINITY), |i| i as f64 * dt);
    let peak = y[..release].iter().fold(Rate::new(f64::MIN), |m, &v| m.max(v));
    let hold = &y[idx(rise + HOLD - Time::new(0.1))..release];
    let track_end = idx(2.0 * rise + HOLD + Time::new(0.05));
    let track = (0..track_end)
        .map(|i| (y[i] - run.setpoint[i]) * (y[i] - run.setpoint[i]))
        .fold(SpeedSq::new(0.0), |acc, e| acc + e) / track_end as f64;
    let half = |v: &[Rate]| v[..release].iter().position(|&x| x >= 0.5 * amp).unwrap_or(release) as f64 * dt;
    let t_stick = half(&run.stick);
    let target_peak = run.target[..release].iter().fold(Rate::new(f64::MIN), |m, &v| m.max(v));
    let hold_mean = hold.iter().fold(Rate::new(0.0), |acc, &v| acc + v) / hold.len() as f64;
    let residual = y[track_end..].iter().fold(Rate::new(0.0), |m, v| m.max(v.abs()));
    Metrics {
        target_ovs: f64::from(target_peak / amp) - 1.0,
        target_lag: half(&run.target) - t_stick,
        gyro_lag: half(&y) - t_stick,
        rise90,
        overshoot: f64::from(peak / amp) - 1.0,
        hold_err: f64::from(hold_mean / amp) - 1.0,
        track_rms: f64::from(track.sqrt() / amp),
        residual: f64::from(residual / amp),
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
fn all_flicks(gains: &GainStorage) -> Vec<(usize, Time, Rate, Metrics)> {
    let mut out = vec![];
    for axis in [0, 1] {
        for rise in RISES {
            for amp in AMPS {
                out.push((axis, rise, amp, measure(gains, axis, amp, rise)));
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
    let mut last = (usize::MAX, Time::new(0.0));
    for (axis, rise, amp, m) in all_flicks(&gains) {
        if (axis, rise) != last {
            println!("\n{} flick over {:.0} ms", ["roll", "pitch"][axis], ms(rise));
            println!("  {:>6} {:>8} {:>8} {:>8} {:>8} {:>10} {:>9} {:>10} {:>9} {:>6}",
                     "amp", "tgt ovs", "tgt lag", "gyro lag", "rise90", "overshoot", "hold err", "track rms", "residual", "sat");
            last = (axis, rise);
        }
        println!("  {:6.2} {:+7.1}% {:6.2}ms {:6.2}ms {:6.1}ms {:+9.1}% {:+8.1}% {:9.1}% {:8.1}% {:5.1}%",
                 value!(amp, rot / s), 100.0 * m.target_ovs, ms(m.target_lag), ms(m.gyro_lag),
                 ms(m.rise90), 100.0 * m.overshoot, 100.0 * m.hold_err,
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
    let (mut big_ovs, mut big_rise, mut small_ovs, mut track_small) =
        (f64::MIN, Time::new(f64::MIN), f64::MIN, f64::MIN);
    for (_, rise, amp, m) in all_flicks(&gains) {
        if amp >= Rate::new(1.0) {
            big_ovs = big_ovs.max(m.overshoot);
            big_rise = big_rise.max(m.rise90);
        } else {
            small_ovs = small_ovs.max(m.overshoot);
        }
        if amp == Rate::new(0.2) && rise == Time::new(0.030) {
            track_small = track_small.max(m.track_rms);
        }
    }
    println!("SUMMARY flick_big_overshoot={big_ovs:.3} flick_big_rise90_ms={:.1} \
              flick_small_overshoot={small_ovs:.3} flick_track_small={track_small:.3}", ms(big_rise));
}
