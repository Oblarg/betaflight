//! How much gyro-path lag the closed loop tolerates.  Not a regression test;
//! run with
//!
//!   cargo test --release --target aarch64-apple-darwin --features std \
//!       --test eval_gyro_filter -- --ignored --nocapture
//!
//! The controller (test_gains.wfg, or the blob at $WFG_PATH) flies the
//! nonlinear plant, as in eval_setpoint_reconstruction, with the gyro passed
//! through BF-style PT1 filters and/or a pure delay before the controller
//! sees it.  `lag_margin_summary` prints one line for tools/sweep_qr.py.
//!
//! The "logged" path is the plant airframe's own gyro low-pass stages
//! (common::plant, from the fit's log header).
//!
//! Noise metrics hover on the logged path with measured gyro and eRPM noise
//! replayed on top (common::noise, $NOISE_PATH): `rough_v` is the RMS of the
//! 50 Hz-high-passed motor voltages (tools/extract_noise.py prints BF's own
//! PID's on the same log), `jitter` the RMS true body rate it causes.
//! `NOISE_SCALE` scales the noise.

mod common;

use common::noise::{Noise, fixture_noise};
use common::plant::{duty_volts, Plant};
use common::units::{AngAccel, Freq, Millis, Rate, SpeedSq, Time, Voltage, VoltageSq};
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::setpoint::SetpointEstimator;
use whippyflight_control::sim::margin::{self, detrended_p2p, GyroPath, HOVER_SCALE};
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
const SETTLE: Time = Time::new(1.0);
const RISE: Time = Time::new(0.030);
const HOLD: Time = Time::new(0.5);
const REST: Time = Time::new(0.6);

fn ms(t: Time) -> f64 {
    value!(t, ms)
}

fn rps(r: Rate) -> f64 {
    value!(r, rot / s)
}

/// The plant airframe's logged gyro low-pass stages, undelayed or delayed.
fn logged_path(delay_ticks: usize, dt: Time) -> GyroPath {
    GyroPath::new(&logged_cutoffs(), delay_ticks, dt)
}

/// The plant airframe's gyro low-pass stages as PT1 cutoffs: BF's PT2/PT3
/// are 2/3 PT1s at the cutoff × CUTOFF_CORRECTION_PT2/PT3 (filter.c).
fn logged_cutoffs() -> Vec<Freq> {
    margin::pt1_cutoffs(common::plant::model()).unwrap_or_else(|e| panic!("{e}"))
}

/// Display name of the logged gyro path, e.g. "PT1 275 + PT1 550".
fn logged_name() -> String {
    let stages = &common::plant::model().gyro_lpf;
    if stages.is_empty() {
        return "none".into();
    }
    stages.iter().map(|(k, hz)| format!("{k} {:.0}", value!(*hz, Hz))).collect::<Vec<_>>().join(" + ")
}


struct Metrics {
    rise90: Time,
    overshoot: f64,
    /// Peak-to-peak roll rate over the last 0.3 s, about its linear trend:
    /// oscillation, not a slow settle still in progress.
    rest_p2p: Rate,
    sat_frac: f64,
    recover: Time,
    /// ∫|roll rate| dt / KICK after the kick: continuous where `recover`
    /// jumps whenever an undershoot lobe crosses the 10% threshold.
    kick_iae: Time,
    rest_pitch: Rate,
    rough_v: Voltage,
    /// True body rates [yaw, pitch, roll] from SETTLE on.
    trace: Vec<[Rate; 3]>,
}

const ROUGH_HP: Freq = Freq::new(50.0); // tools/extract_noise.py ROUGH_HP_HZ

const KICK: Rate = Rate::new(0.2); // added to the roll rate at SETTLE by `kick`

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

/// Hover with zero stick, kick the roll rate by KICK and time its decay to 10%.
fn kick(gains: &GainStorage, path: GyroPath) -> Metrics {
    fly_inner(gains, Rate::new(0.0), KICK, SETTLE, path, None)
}

/// Fly a roll flick to `amp` and back through `path`.
fn fly(gains: &GainStorage, amp: Rate, path: GyroPath) -> Metrics {
    fly_inner(gains, amp, Rate::new(0.0), SETTLE, path, None)
}

/// Mean hover pitch rate once the (slow) integrator has settled.
fn hover_pitch(gains: &GainStorage, path: GyroPath) -> Rate {
    fly_inner(gains, Rate::new(0.0), Rate::new(0.0), Time::new(6.0), path, None).rest_pitch
}

fn plant_dt(gains: &GainStorage) -> Time {
    gains.header.dt.lossy_into()
}

/// Hover on the logged gyro path with measured sensor noise: motor-voltage
/// roughness and the RMS body-rate jitter (vs the same flight without noise).
fn noisy_hover(gains: &GainStorage, gyro_scale: f32, motor_scale: f32) -> (Voltage, Rate) {
    let dt = plant_dt(gains);
    let still = Rate::new(0.0);
    let clean = fly_inner(gains, still, still, SETTLE, logged_path(0, dt), None);
    let noisy = fly_inner(gains, still, still, SETTLE, logged_path(0, dt),
                          Some(&fixture_noise(Time::new(0.0), gyro_scale, motor_scale)));
    let jitter_sq = noisy.trace.iter().zip(&clean.trace)
        .map(|(a, b)| {
            (0..3).fold(SpeedSq::new(0.0), |acc, i| {
                let d = a[i] - b[i];
                acc + d * d
            }) / 3.0
        })
        .fold(SpeedSq::new(0.0), |acc, e| acc + e) / clean.trace.len() as f64;
    (noisy.rough_v, jitter_sq.sqrt())
}

fn noise_scale() -> f32 {
    std::env::var("NOISE_SCALE").map_or(1.0, |s| s.parse().expect("NOISE_SCALE"))
}

/// [`margin::trim`] on the eval plant's logged gyro path: (attitude cost, deg; residual pitch rate).
fn trim(gains: &GainStorage) -> (f64, Rate) {
    let (angle, residual) = margin::trim(gains, common::plant::model(), &logged_cutoffs());
    (value!(angle, deg), residual)
}


fn fly_inner(gains: &GainStorage, amp: Rate, kick: Rate, settle: Time, path: GyroPath,
             noise: Option<&Noise>) -> Metrics {
    fly_inner_with(gains, amp, kick, settle, path, noise, AngAccel::new(0.0), 2.0 * RISE + HOLD + REST)
}

/// [`fly_inner`] with a pitch angular acceleration `torque_pitch` (BF axes)
/// applied from `settle` on, flying `after` past `settle`.
#[allow(clippy::too_many_arguments)]
fn fly_inner_with(gains: &GainStorage, amp: Rate, kick: Rate, settle: Time, mut path: GyroPath,
                  noise: Option<&Noise>, torque_pitch: AngAccel, after: Time) -> Metrics {
    let dt = plant_dt(gains);
    let stick = |t: Time| {
        let t = t - settle;
        amp * (s_curve(f64::from(t / RISE)) - s_curve(f64::from((t - RISE - HOLD) / RISE)))
    };
    let mut ctrl = ControllerState::new();
    let mut sp = SetpointEstimator::new();
    sp.configure_for(gains);
    let mut plant = Plant::hovering(HOVER_SCALE);
    let mut volts = [Voltage::new(0.0); 4];
    let t_end = settle + after;
    let (mut next_pkt, mut peak, mut rise90) = (Time::new(0.0), Rate::new(f64::MIN), Time::new(f64::INFINITY));
    let mut rest = Vec::new();
    let (mut pitch_sum, mut pitch_n) = (Rate::new(0.0), 0usize);
    let (mut sat, mut n) = (0usize, 0usize);
    let (mut kicked, mut recover, mut iae) = (false, Time::new(0.0), Time::new(0.0));
    let hp_k = {
        let w = f64::from(2.0 * core::f64::consts::PI * ROUGH_HP * dt);
        w / (w + 1.0)
    };
    let (mut volts_lp, mut rough_sq, mut trace) = ([Voltage::new(0.0); 4], VoltageSq::new(0.0), Vec::new());
    for k in 0..f64::from(t_end / dt) as usize {
        let t = k as f64 * dt;
        let now_us = common::micros64(t);
        if !kicked && t >= settle {
            plant.rate[2] = plant.rate[2] - kick; // model roll is -BF roll
            kicked = true;
        }
        if t >= next_pkt {
            let zero = RotPerSec::new(0.0);
            sp.packet([stick(t).lossy_into(), zero, zero], now_us, (1.0 / F_RX).lossy_into());
            next_pkt = next_pkt + 1.0 / F_RX;
        }
        let (rate, target) = sp.tick(now_us, dt.lossy_into());
        let mut gyro = path.filter(plant.gyro_bf());
        let mut motor_rps = plant.motor_rps();
        if let Some(noise) = noise {
            let s = noise.at(t);
            (0..3).for_each(|a| gyro[a] = gyro[a] + s.gyro[a]);
            (0..4).for_each(|m| motor_rps[m] = motor_rps[m] + s.motor[m]);
        }
        let sensors = SensorData { gyro_rps: gyro, motor_rps, battery_voltage: battery_v(), accel: [MetersPerSecSq::new(0.0); 3] };
        let out = ctrl.update_with_target(&sensors, &SetpointData { rate_rps: rate, throttle: throttle() },
                                          target, gains);
        volts = duty_volts(out.motor_duty);
        for m in 0..4 {
            volts_lp[m] = volts_lp[m] + hp_k * (volts[m] - volts_lp[m]);
        }
        let no_torque = AngAccel::new(0.0);
        let torque = [no_torque, if t >= settle { torque_pitch } else { no_torque }, no_torque];
        for _ in 0..4 {
            plant.rk4_with_torque(&volts, dt / 4.0, torque);
        }
        let y: Rate = plant.gyro_bf()[0].lossy_into();
        let rel = t - settle;
        if rel >= Time::new(0.0) {
            n += 1;
            let step = (0..4).fold(VoltageSq::new(0.0), |acc, m| {
                let hp = volts[m] - volts_lp[m];
                acc + hp * hp
            });
            rough_sq = rough_sq + step / 4.0;
            trace.push(plant.rate);
            if kick > Rate::new(0.0) {
                iae = iae + y.abs() * dt / kick;
                if y.abs() > 0.1 * kick {
                    recover = rel;
                }
            }
            if ctrl.telemetry().command_scale < 0.999 {
                sat += 1;
            }
            if rel < RISE + HOLD {
                peak = peak.max(y);
                if y >= 0.9 * amp && !rise90.is_finite() {
                    rise90 = rel;
                }
            }
            if rel >= t_end - settle - Time::new(0.3) {
                rest.push((t, y));
                let pitch: Rate = plant.gyro_bf()[1].lossy_into();
                pitch_sum = pitch_sum + pitch;
                pitch_n += 1;
            }
        }
    }
    let _ = volts;
    Metrics {
        rise90,
        overshoot: f64::from(peak / amp) - 1.0,
        rest_p2p: detrended_p2p(&rest),
        sat_frac: sat as f64 / n as f64,
        recover,
        kick_iae: iae,
        rest_pitch: pitch_sum / pitch_n as f64,
        rough_v: (rough_sq / n as f64).sqrt(),
        trace,
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

/// [`margin::delay_margin`] on the eval plant.
fn delay_margin(gains: &GainStorage, cutoffs: &[Freq]) -> Millis {
    margin::delay_margin(gains, common::plant::model(), cutoffs)
}

#[test]
#[ignore]
fn lag_margin_summary() {
    let gains = load();
    let dt = plant_dt(&gains);
    let nominal = fly(&gains, Rate::new(0.5), GyroPath::new(&[], 0, dt));
    let logged = logged_cutoffs();
    let flown = fly(&gains, Rate::new(0.5), logged_path(0, dt));
    let flown_small = fly(&gains, Rate::new(0.05), logged_path(0, dt));
    let margin = |cutoffs: &[Freq]| value!(delay_margin(&gains, cutoffs), ms);
    let hover = kick(&gains, GyroPath::new(&[], 0, dt));
    let logged_kick = kick(&gains, logged_path(0, dt));
    let (rough_v, jitter) = noisy_hover(&gains, noise_scale(), noise_scale());
    let (trim_deg, trim_residual) = trim(&gains);
    println!("SUMMARY rise90_ms={:.1} overshoot={:.3} sat={:.3} hover_pitch={:.4} kick_ms={:.1} \
              logged_kick_ms={:.1} kick_iae_ms={:.2} logged_kick_iae_ms={:.2} logged_overshoot={:.3} \
              logged_p2p={:.4} logged_sat={:.3} logged_small_p2p={:.4} delay_margin_ms={:.2} \
              logged_delay_margin_ms={:.2} rough_v={:.4} jitter={:.5} trim_deg={:.2} trim_residual={:.5}",
             ms(nominal.rise90), nominal.overshoot, nominal.sat_frac,
             rps(hover_pitch(&gains, logged_path(0, dt))), ms(hover.recover),
             ms(logged_kick.recover), ms(hover.kick_iae), ms(logged_kick.kick_iae),
             flown.overshoot, rps(flown.rest_p2p), flown.sat_frac, rps(flown_small.rest_p2p),
             margin(&[]), margin(&logged), value!(rough_v, V), rps(jitter), trim_deg, rps(trim_residual));
}

#[test]
#[ignore]
fn noise_breakdown() {
    let gains = load();
    println!("hover on the logged gyro path ({}), measured noise ×scale", logged_name());
    println!("  {:>6} {:>14} {:>14} {:>14}", "scale", "gyro only", "eRPM only", "both");
    for scale in [1.0f32, 3.0, 10.0] {
        let cell = |g, m| {
            let (r, j) = noisy_hover(&gains, g, m);
            format!("{:.4}V/{:.5}", value!(r, V), rps(j))
        };
        println!("  {:>6} {:>14} {:>14} {:>14}", scale, cell(scale, 0.0), cell(0.0, scale), cell(scale, scale));
    }
    println!("  (roughness V / body-rate jitter rot/s)");
}

#[test]
#[ignore]
fn gyro_lag_tolerance() {
    let gains = load();
    let dt = plant_dt(&gains);
    let logged = logged_cutoffs();
    let logged_label = format!("{} (logged)", logged_name());
    let sitl = [Freq::new(250.0), Freq::new(500.0)];
    let fast = [Freq::new(500.0), Freq::new(1000.0)];
    let paths: [(&str, &[Freq], usize); 9] = [
        ("none", &[], 0),
        ("PT1 250 + PT1 500 (SITL)", &sitl, 0),
        (&logged_label, &logged, 0),
        ("PT1 500 + PT1 1000", &fast, 0),
        ("delay 1 tick", &[], 1),
        ("delay 2 ticks", &[], 2),
        ("delay 3 ticks", &[], 3),
        ("delay 4 ticks", &[], 4),
        ("delay 6 ticks", &[], 6),
    ];
    for amp in [Rate::new(0.05), Rate::new(0.5)] {
        println!("\nroll flick 0 → {} rot/s over {:.0} ms and back", rps(amp), ms(RISE));
        println!("  {:32} {:>9} {:>10} {:>14} {:>6}", "gyro path", "rise90", "overshoot", "rest p2p", "sat");
        for &(name, cutoffs, delay) in &paths {
            let m = fly(&gains, amp, GyroPath::new(cutoffs, delay, dt));
            println!("  {:32} {:7.1}ms {:+9.1}% {:10.4} rot/s {:5.1}%",
                     name, ms(m.rise90), 100.0 * m.overshoot, rps(m.rest_p2p), 100.0 * m.sat_frac);
        }
    }
}
