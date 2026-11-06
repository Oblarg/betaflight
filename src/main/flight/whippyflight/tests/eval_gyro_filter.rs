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

use common::noise::Noise;
use common::plant::Plant;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::setpoint::SetpointEstimator;
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
const SETTLE_S: f64 = 1.0;
const RISE_S: f64 = 0.030;
const HOLD_S: f64 = 0.5;
const REST_S: f64 = 0.6;

/// Gyro path: PT1 stages (BF's pt1FilterGain) then a delay in PID ticks,
/// fractional delays linearly interpolated between ticks.
struct GyroPath {
    gains: Vec<f64>,
    state: Vec<[f64; 3]>,
    /// Filtered samples, newest last.
    history: std::collections::VecDeque<[f64; 3]>,
    delay_ticks: f64,
}

impl GyroPath {
    fn new(cutoffs_hz: &[f64], delay_ticks: usize, dt: f64) -> Self {
        Self::delayed(cutoffs_hz, delay_ticks as f64, dt)
    }

    fn delayed(cutoffs_hz: &[f64], delay_ticks: f64, dt: f64) -> Self {
        let gains = cutoffs_hz
            .iter()
            .map(|f| {
                let w = 2.0 * core::f64::consts::PI * f * dt;
                w / (w + 1.0)
            })
            .collect::<Vec<_>>();
        let n = gains.len();
        let history = vec![[0.0; 3]; delay_ticks as usize + 2].into();
        Self { gains, state: vec![[0.0; 3]; n], history, delay_ticks }
    }

    /// The plant airframe's logged gyro low-pass stages.
    fn logged(delay_ticks: usize, dt: f64) -> Self {
        Self::new(&logged_cutoffs(), delay_ticks, dt)
    }
}

/// The plant airframe's gyro low-pass stages as PT1 cutoffs: BF's PT2/PT3
/// are 2/3 PT1s at the cutoff × CUTOFF_CORRECTION_PT2/PT3 (filter.c).
fn logged_cutoffs() -> Vec<f64> {
    common::plant::model()
        .gyro_lpf
        .iter()
        .flat_map(|(kind, hz)| match kind.as_str() {
            "PT1" => vec![*hz],
            "PT2" => vec![hz * 1.553_773_974; 2],
            "PT3" => vec![hz * 1.961_459_177; 3],
            _ => panic!("gyro LPF type {kind} is not modeled"),
        })
        .collect()
}

/// Display name of the logged gyro path, e.g. "PT1 275 + PT1 550".
fn logged_name() -> String {
    let stages = &common::plant::model().gyro_lpf;
    if stages.is_empty() {
        return "none".into();
    }
    stages.iter().map(|(k, hz)| format!("{k} {hz:.0}")).collect::<Vec<_>>().join(" + ")
}

impl GyroPath {

    fn filter(&mut self, x: [f32; 3]) -> [f32; 3] {
        let mut y = x.map(|v| v as f64);
        for (k, s) in self.gains.iter().zip(self.state.iter_mut()) {
            for a in 0..3 {
                s[a] += k * (y[a] - s[a]);
            }
            y = *s;
        }
        self.history.pop_front();
        self.history.push_back(y);
        let (n, f) = (self.delay_ticks.floor() as usize, self.delay_ticks.fract());
        let last = self.history.len() - 1;
        let (a, b) = (self.history[last - n], self.history[last - n - 1]);
        core::array::from_fn(|i| ((1.0 - f) * a[i] + f * b[i]) as f32)
    }
}

struct Metrics {
    rise90_ms: f64,
    overshoot: f64,
    /// Peak-to-peak roll rate over the last 0.3 s, about its linear trend:
    /// oscillation, not a slow settle still in progress.
    rest_p2p: f64,
    sat_frac: f64,
    recover_ms: f64,
    /// ∫|roll rate| dt / KICK after the kick: continuous where recover_ms
    /// jumps whenever an undershoot lobe crosses the 10% threshold.
    kick_iae_ms: f64,
    rest_pitch: f64,
    rough_v: f64,
    /// True body rates [yaw, pitch, roll] from SETTLE_S on.
    trace: Vec<[f64; 3]>,
}

const ROUGH_HP_HZ: f64 = 50.0; // tools/extract_noise.py ROUGH_HP_HZ

const KICK: f64 = 0.2; // rot/s, added to the roll rate at SETTLE_S by `kick`

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

/// Hover with zero stick, kick the roll rate by KICK and time its decay to 10%.
fn kick(gains: &GainStorage, path: GyroPath) -> Metrics {
    fly_inner(gains, 0.0, KICK, SETTLE_S, path, None)
}

/// Fly a roll flick to `amp` rot/s and back through `path`.
fn fly(gains: &GainStorage, amp: f64, path: GyroPath) -> Metrics {
    fly_inner(gains, amp, 0.0, SETTLE_S, path, None)
}

/// Mean hover pitch rate once the (slow) integrator has settled.
fn hover_pitch(gains: &GainStorage, path: GyroPath) -> f64 {
    fly_inner(gains, 0.0, 0.0, 6.0, path, None).rest_pitch
}

/// Hover on the logged gyro path with measured sensor noise: motor-voltage
/// roughness and the RMS body-rate jitter (vs the same flight without noise).
fn noisy_hover(gains: &GainStorage, gyro_scale: f32, motor_scale: f32) -> (f64, f64) {
    let dt = gains.header.dt as f64;
    let clean = fly_inner(gains, 0.0, 0.0, SETTLE_S, GyroPath::logged(0, dt), None);
    let noisy = fly_inner(gains, 0.0, 0.0, SETTLE_S, GyroPath::logged(0, dt),
                          Some(&Noise::new(0.0, gyro_scale, motor_scale)));
    let jitter_sq = noisy.trace.iter().zip(&clean.trace)
        .map(|(a, b)| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>() / 3.0)
        .sum::<f64>() / clean.trace.len() as f64;
    (noisy.rough_v, jitter_sq.sqrt())
}

fn noise_scale() -> f32 {
    std::env::var("NOISE_SCALE").map_or(1.0, |s| s.parse().expect("NOISE_SCALE"))
}

/// Fraction of the front-back arm the CG shifts by in `trim`.
const TRIM_CG_SHIFT: f64 = 0.05;
/// How long `trim` flies after the shift.
const TRIM_S: f64 = 3.0;

/// Pitch angular acceleration (BF axes, rot/s²) a TRIM_CG_SHIFT CG shift
/// causes at the eval hover: the hover thrust k_T·Σω² on a lever arm
/// TRIM_CG_SHIFT of the (unit) front-back arm.
fn trim_torque() -> f64 {
    let m = common::plant::model();
    TRIM_CG_SHIFT * m.pitch_prop_vel_sq * 4.0 * (HOVER_SCALE * m.hover_rps).powi(2)
}

/// Integrator trim: hover, shift the CG by TRIM_CG_SHIFT (a step pitch
/// torque) at SETTLE_S, and return (attitude it costs, deg, = |∫ pitch
/// rate| over TRIM_S; residual pitch rate over the last 0.5 s, rot/s).  The
/// rate loop alone holds a constant rate error under a constant torque;
/// the (leaky) integrator is what brings it back toward zero.
fn trim(gains: &GainStorage) -> (f64, f64) {
    let dt = gains.header.dt as f64;
    let m = fly_inner_with(gains, 0.0, 0.0, SETTLE_S, GyroPath::logged(0, dt), None, trim_torque(), TRIM_S);
    let pitch: Vec<f64> = m.trace.iter().map(|r| r[1]).collect();
    let angle = pitch.iter().sum::<f64>() * dt * 360.0;
    let tail = (0.5 / dt) as usize;
    let residual = pitch[pitch.len() - tail..].iter().sum::<f64>() / tail as f64;
    (angle.abs(), residual)
}

const HOVER_SCALE: f64 = 250.0 / 270.0;

fn fly_inner(gains: &GainStorage, amp: f64, kick: f64, settle_s: f64, path: GyroPath,
             noise: Option<&Noise>) -> Metrics {
    fly_inner_with(gains, amp, kick, settle_s, path, noise, 0.0, 2.0 * RISE_S + HOLD_S + REST_S)
}

/// [`fly_inner`] with a pitch angular acceleration `torque_pitch` (BF axes,
/// rot/s²) applied from `settle_s` on, flying `after_s` past `settle_s`.
#[allow(clippy::too_many_arguments)]
fn fly_inner_with(gains: &GainStorage, amp: f64, kick: f64, settle_s: f64, mut path: GyroPath,
                  noise: Option<&Noise>, torque_pitch: f64, after_s: f64) -> Metrics {
    let dt = gains.header.dt as f64;
    let stick = |t: f64| {
        let t = t - settle_s;
        amp * (s_curve(t / RISE_S) - s_curve((t - RISE_S - HOLD_S) / RISE_S))
    };
    let mut ctrl = ControllerState::new();
    let mut sp = SetpointEstimator::new();
    sp.configure_for(gains);
    let mut plant = Plant::hovering(HOVER_SCALE);
    let mut volts = [0.0f64; 4];
    let t_end = settle_s + after_s;
    let (mut next_pkt, mut peak, mut rise90) = (0.0, f64::MIN, f64::INFINITY);
    let mut rest = Vec::new();
    let (mut pitch_sum, mut pitch_n) = (0.0, 0usize);
    let (mut sat, mut n) = (0usize, 0usize);
    let (mut kicked, mut recover, mut iae) = (false, 0.0, 0.0);
    let hp_k = {
        let w = 2.0 * core::f64::consts::PI * ROUGH_HP_HZ * dt;
        w / (w + 1.0)
    };
    let (mut volts_lp, mut rough_sq, mut trace) = ([0.0f64; 4], 0.0, Vec::new());
    for k in 0..(t_end / dt) as usize {
        let t = k as f64 * dt;
        let now_us = (t * 1e6) as u32;
        if !kicked && t >= settle_s {
            plant.rate[2] -= kick; // model roll is -BF roll
            kicked = true;
        }
        if t >= next_pkt {
            sp.packet([stick(t) as f32, 0.0, 0.0], now_us, (1.0 / F_RX) as f32);
            next_pkt += 1.0 / F_RX;
        }
        let (rate, target) = sp.tick(now_us, dt as f32);
        let mut gyro = path.filter(plant.gyro_bf());
        let mut motor_rps = plant.omega.map(|w| w as f32);
        if let Some(noise) = noise {
            let s = noise.at(t);
            (0..3).for_each(|a| gyro[a] += s.gyro[a]);
            (0..4).for_each(|m| motor_rps[m] += s.motor[m]);
        }
        let sensors = SensorData { gyro_rps: gyro, motor_rps, battery_voltage: battery_v(), accel_g: [0.0; 3] };
        let out = ctrl.update_with_target(&sensors, &SetpointData { rate_rps: rate, throttle: throttle() },
                                          target, gains);
        volts = out.motor_duty.map(|d| (d * battery_v()) as f64);
        for m in 0..4 {
            volts_lp[m] += hp_k * (volts[m] - volts_lp[m]);
        }
        let torque = [0.0, if t >= settle_s { torque_pitch } else { 0.0 }, 0.0];
        for _ in 0..4 {
            plant.rk4_with_torque(&volts, dt / 4.0, torque);
        }
        let y = plant.gyro_bf()[0] as f64;
        let rel = t - settle_s;
        if rel >= 0.0 {
            n += 1;
            rough_sq += (0..4).map(|m| (volts[m] - volts_lp[m]).powi(2)).sum::<f64>() / 4.0;
            trace.push(plant.rate);
            if kick > 0.0 {
                iae += y.abs() * dt / kick;
                if y.abs() > 0.1 * kick {
                    recover = rel;
                }
            }
            if ctrl.telemetry().command_scale < 0.999 {
                sat += 1;
            }
            if rel < RISE_S + HOLD_S {
                peak = peak.max(y);
                if y >= 0.9 * amp && rise90.is_infinite() {
                    rise90 = rel;
                }
            }
            if rel >= t_end - settle_s - 0.3 {
                rest.push((t, y));
                pitch_sum += plant.gyro_bf()[1] as f64;
                pitch_n += 1;
            }
        }
    }
    let _ = volts;
    Metrics {
        rise90_ms: 1e3 * rise90,
        overshoot: peak / amp - 1.0,
        rest_p2p: detrended_p2p(&rest),
        sat_frac: sat as f64 / n as f64,
        recover_ms: 1e3 * recover,
        kick_iae_ms: 1e3 * iae,
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

/// Largest extra gyro delay (ms) on top of `cutoffs` that still flies both
/// SUMMARY flicks cleanly: bisected over 0–MAX_MARGIN_MS with a fractional
/// delay, so it ranks candidates instead of saturating at a tick cap.  −1
/// if the path isn't clean without extra delay; MAX_MARGIN_MS if it never
/// breaks.  In ms rather than ticks so loop rates compare.
fn delay_margin_ms(gains: &GainStorage, cutoffs: &[f64]) -> f64 {
    let dt = gains.header.dt as f64;
    let ok = |ms: f64| {
        [0.05, 0.5].iter().all(|&a| clean(&fly(gains, a, GyroPath::delayed(cutoffs, ms * 1e-3 / dt, dt))))
    };
    if !ok(0.0) {
        return -1.0;
    }
    if ok(MAX_MARGIN_MS) {
        return MAX_MARGIN_MS;
    }
    let (mut lo, mut hi) = (0.0, MAX_MARGIN_MS);
    while hi - lo > MARGIN_RESOLUTION_MS {
        let mid = 0.5 * (lo + hi);
        if ok(mid) { lo = mid } else { hi = mid }
    }
    lo
}

const MAX_MARGIN_MS: f64 = 16.0;
const MARGIN_RESOLUTION_MS: f64 = 0.1;

/// Peak-to-peak of `(t, y)` samples about their least-squares line.
fn detrended_p2p(samples: &[(f64, f64)]) -> f64 {
    let n = samples.len() as f64;
    let (mt, my) = samples.iter().fold((0.0, 0.0), |(a, b), &(t, y)| (a + t / n, b + y / n));
    let (stt, sty) = samples
        .iter()
        .fold((0.0, 0.0), |(a, b), &(t, y)| (a + (t - mt).powi(2), b + (t - mt) * (y - my)));
    let slope = if stt > 0.0 { sty / stt } else { 0.0 };
    let (lo, hi) = samples.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &(t, y)| {
        let r = y - my - slope * (t - mt);
        (lo.min(r), hi.max(r))
    });
    hi - lo
}

/// Settled without limit cycling or chattering on the rails.
fn clean(m: &Metrics) -> bool {
    m.rest_p2p < 0.005 && m.sat_frac < 0.1
}

#[test]
#[ignore]
fn lag_margin_summary() {
    let gains = load();
    let dt = gains.header.dt as f64;
    let nominal = fly(&gains, 0.5, GyroPath::new(&[], 0, dt));
    let logged = logged_cutoffs();
    let flown = fly(&gains, 0.5, GyroPath::logged(0, dt));
    let flown_small = fly(&gains, 0.05, GyroPath::logged(0, dt));
    let margin = |cutoffs: &[f64]| delay_margin_ms(&gains, cutoffs);
    let hover = kick(&gains, GyroPath::new(&[], 0, dt));
    let logged_kick = kick(&gains, GyroPath::logged(0, dt));
    let (rough_v, jitter) = noisy_hover(&gains, noise_scale(), noise_scale());
    let (trim_deg, trim_residual) = trim(&gains);
    println!("SUMMARY rise90_ms={:.1} overshoot={:.3} sat={:.3} hover_pitch={:.4} kick_ms={:.1} \
              logged_kick_ms={:.1} kick_iae_ms={:.2} logged_kick_iae_ms={:.2} logged_overshoot={:.3} \
              logged_p2p={:.4} logged_sat={:.3} logged_small_p2p={:.4} delay_margin_ms={:.2} \
              logged_delay_margin_ms={:.2} rough_v={:.4} jitter={:.5} trim_deg={:.2} trim_residual={:.5}",
             nominal.rise90_ms, nominal.overshoot, nominal.sat_frac,
             hover_pitch(&gains, GyroPath::logged(0, dt)), hover.recover_ms,
             logged_kick.recover_ms, hover.kick_iae_ms, logged_kick.kick_iae_ms,
             flown.overshoot, flown.rest_p2p, flown.sat_frac, flown_small.rest_p2p,
             margin(&[]), margin(&logged), rough_v, jitter, trim_deg, trim_residual);
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
            format!("{r:.4}V/{j:.5}")
        };
        println!("  {:>6} {:>14} {:>14} {:>14}", scale, cell(scale, 0.0), cell(0.0, scale), cell(scale, scale));
    }
    println!("  (roughness V / body-rate jitter rot/s)");
}

#[test]
#[ignore]
fn gyro_lag_tolerance() {
    let gains = load();
    let dt = gains.header.dt as f64;
    let logged = logged_cutoffs();
    let logged_label = format!("{} (logged)", logged_name());
    let paths: [(&str, &[f64], usize); 9] = [
        ("none", &[], 0),
        ("PT1 250 + PT1 500 (SITL)", &[250.0, 500.0], 0),
        (&logged_label, &logged, 0),
        ("PT1 500 + PT1 1000", &[500.0, 1000.0], 0),
        ("delay 1 tick", &[], 1),
        ("delay 2 ticks", &[], 2),
        ("delay 3 ticks", &[], 3),
        ("delay 4 ticks", &[], 4),
        ("delay 6 ticks", &[], 6),
    ];
    for amp in [0.05, 0.5] {
        println!("\nroll flick 0 → {amp} rot/s over {:.0} ms and back", RISE_S * 1e3);
        println!("  {:32} {:>9} {:>10} {:>14} {:>6}", "gyro path", "rise90", "overshoot", "rest p2p", "sat");
        for &(name, cutoffs, delay) in &paths {
            let m = fly(&gains, amp, GyroPath::new(cutoffs, delay, dt));
            println!("  {:32} {:7.1}ms {:+9.1}% {:10.4} rot/s {:5.1}%",
                     name, m.rise90_ms, 100.0 * m.overshoot, m.rest_p2p, 100.0 * m.sat_frac);
        }
    }
}
