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
use common::plant::{duty_volts, Plant};
use common::units::{Angle, AngAccel, Freq, Millis, Rate, SpeedSq, Time, TimeSq, Voltage, VoltageSq};
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::setpoint::SetpointEstimator;
use whippyflight_control::types::*;
use whippyunits::api::rescale;
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

/// Gyro path: PT1 stages (BF's pt1FilterGain) then a delay in PID ticks,
/// fractional delays linearly interpolated between ticks.
struct GyroPath {
    gains: Vec<f64>,
    state: Vec<[Rate; 3]>,
    /// Filtered samples, newest last.
    history: std::collections::VecDeque<[Rate; 3]>,
    delay_ticks: f64,
}

impl GyroPath {
    fn new(cutoffs: &[Freq], delay_ticks: usize, dt: Time) -> Self {
        Self::delayed(cutoffs, delay_ticks as f64, dt)
    }

    fn delayed(cutoffs: &[Freq], delay_ticks: f64, dt: Time) -> Self {
        let gains = cutoffs
            .iter()
            .map(|&f| {
                let w = f64::from(2.0 * core::f64::consts::PI * f * dt);
                w / (w + 1.0)
            })
            .collect::<Vec<_>>();
        let n = gains.len();
        let history = vec![[Rate::new(0.0); 3]; delay_ticks as usize + 2].into();
        Self { gains, state: vec![[Rate::new(0.0); 3]; n], history, delay_ticks }
    }

    /// The plant airframe's logged gyro low-pass stages.
    fn logged(delay_ticks: usize, dt: Time) -> Self {
        Self::new(&logged_cutoffs(), delay_ticks, dt)
    }
}

/// The plant airframe's gyro low-pass stages as PT1 cutoffs: BF's PT2/PT3
/// are 2/3 PT1s at the cutoff × CUTOFF_CORRECTION_PT2/PT3 (filter.c).
fn logged_cutoffs() -> Vec<Freq> {
    common::plant::model()
        .gyro_lpf
        .iter()
        .flat_map(|&(ref kind, hz)| match kind.as_str() {
            "PT1" => vec![hz],
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
    stages.iter().map(|(k, hz)| format!("{k} {:.0}", value!(*hz, Hz))).collect::<Vec<_>>().join(" + ")
}

impl GyroPath {

    fn filter(&mut self, x: [RotPerSec; 3]) -> [RotPerSec; 3] {
        let mut y: [Rate; 3] = x.map(|v| v.lossy_into());
        for (&k, s) in self.gains.iter().zip(self.state.iter_mut()) {
            for a in 0..3 {
                s[a] = s[a] + k * (y[a] - s[a]);
            }
            y = *s;
        }
        self.history.pop_front();
        self.history.push_back(y);
        let (n, f) = (self.delay_ticks.floor() as usize, self.delay_ticks.fract());
        let last = self.history.len() - 1;
        let (a, b) = (self.history[last - n], self.history[last - n - 1]);
        core::array::from_fn(|i| ((1.0 - f) * a[i] + f * b[i]).lossy_into())
    }
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
    let clean = fly_inner(gains, still, still, SETTLE, GyroPath::logged(0, dt), None);
    let noisy = fly_inner(gains, still, still, SETTLE, GyroPath::logged(0, dt),
                          Some(&Noise::new(Time::new(0.0), gyro_scale, motor_scale)));
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

/// Fraction of the front-back arm the CG shifts by in `trim`.
const TRIM_CG_SHIFT: f64 = 0.05;
/// How long `trim` flies after the shift.
const TRIM: Time = Time::new(3.0);

/// Pitch angular acceleration (BF axes) a TRIM_CG_SHIFT CG shift causes at
/// the eval hover: the hover thrust k_T·Σω² on a lever arm TRIM_CG_SHIFT of
/// the (unit) front-back arm.
fn trim_torque() -> AngAccel {
    let m = common::plant::model();
    let w = HOVER_SCALE * m.hover_rps;
    TRIM_CG_SHIFT * m.pitch_prop_vel_sq * 4.0 * (w * w)
}

/// Integrator trim: hover, shift the CG by TRIM_CG_SHIFT (a step pitch
/// torque) at SETTLE, and return (attitude it costs, deg, = |∫ pitch
/// rate| over TRIM; residual pitch rate over the last 0.5 s).  The rate
/// loop alone holds a constant rate error under a constant torque; the
/// (leaky) integrator is what brings it back toward zero.
fn trim(gains: &GainStorage) -> (f64, Rate) {
    let dt = plant_dt(gains);
    let still = Rate::new(0.0);
    let m = fly_inner_with(gains, still, still, SETTLE, GyroPath::logged(0, dt), None, trim_torque(), TRIM);
    let pitch: Vec<Rate> = m.trace.iter().map(|r| r[1]).collect();
    let angle: Angle = pitch.iter().fold(Rate::new(0.0), |acc, &r| acc + r) * dt;
    let tail = f64::from(Time::new(0.5) / dt) as usize;
    let residual = pitch[pitch.len() - tail..].iter().fold(Rate::new(0.0), |acc, &r| acc + r) / tail as f64;
    (value!(angle, deg).abs(), residual)
}

const HOVER_SCALE: f64 = 250.0 / 270.0;

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

/// Largest extra gyro delay on top of `cutoffs` that still flies both
/// SUMMARY flicks cleanly: bisected over 0–MAX_MARGIN with a fractional
/// delay, so it ranks candidates instead of saturating at a tick cap.  −1 ms
/// if the path isn't clean without extra delay; MAX_MARGIN if it never
/// breaks.  A time rather than ticks so loop rates compare.
fn delay_margin(gains: &GainStorage, cutoffs: &[Freq]) -> Millis {
    let dt = plant_dt(gains);
    let ok = |delay: Millis| {
        let delay: Time = rescale(delay);
        [Rate::new(0.05), Rate::new(0.5)]
            .iter()
            .all(|&a| clean(&fly(gains, a, GyroPath::delayed(cutoffs, f64::from(delay / dt), dt))))
    };
    if !ok(Millis::new(0.0)) {
        return Millis::new(-1.0);
    }
    if ok(MAX_MARGIN) {
        return MAX_MARGIN;
    }
    let (mut lo, mut hi) = (Millis::new(0.0), MAX_MARGIN);
    while hi - lo > MARGIN_RESOLUTION {
        let mid = 0.5 * (lo + hi);
        if ok(mid) { lo = mid } else { hi = mid }
    }
    lo
}

const MAX_MARGIN: Millis = Millis::new(16.0);
const MARGIN_RESOLUTION: Millis = Millis::new(0.1);

/// Peak-to-peak of `(t, y)` samples about their least-squares line.
fn detrended_p2p(samples: &[(Time, Rate)]) -> Rate {
    let n = samples.len() as f64;
    let (mt, my) = samples.iter().fold((Time::new(0.0), Rate::new(0.0)), |(a, b), &(t, y)| (a + t / n, b + y / n));
    let (stt, sty) = samples.iter().fold((TimeSq::new(0.0), Angle::new(0.0)), |(a, b), &(t, y)| {
        (a + (t - mt) * (t - mt), b + (t - mt) * (y - my))
    });
    let slope = if stt > TimeSq::new(0.0) { sty / stt } else { AngAccel::new(0.0) };
    let (lo, hi) = samples.iter().fold((Rate::new(f64::MAX), Rate::new(f64::MIN)), |(lo, hi), &(t, y)| {
        let r = y - my - slope * (t - mt);
        (lo.min(r), hi.max(r))
    });
    hi - lo
}

/// Settled without limit cycling or chattering on the rails.
fn clean(m: &Metrics) -> bool {
    m.rest_p2p < Rate::new(0.005) && m.sat_frac < 0.1
}

#[test]
#[ignore]
fn lag_margin_summary() {
    let gains = load();
    let dt = plant_dt(&gains);
    let nominal = fly(&gains, Rate::new(0.5), GyroPath::new(&[], 0, dt));
    let logged = logged_cutoffs();
    let flown = fly(&gains, Rate::new(0.5), GyroPath::logged(0, dt));
    let flown_small = fly(&gains, Rate::new(0.05), GyroPath::logged(0, dt));
    let margin = |cutoffs: &[Freq]| value!(delay_margin(&gains, cutoffs), ms);
    let hover = kick(&gains, GyroPath::new(&[], 0, dt));
    let logged_kick = kick(&gains, GyroPath::logged(0, dt));
    let (rough_v, jitter) = noisy_hover(&gains, noise_scale(), noise_scale());
    let (trim_deg, trim_residual) = trim(&gains);
    println!("SUMMARY rise90_ms={:.1} overshoot={:.3} sat={:.3} hover_pitch={:.4} kick_ms={:.1} \
              logged_kick_ms={:.1} kick_iae_ms={:.2} logged_kick_iae_ms={:.2} logged_overshoot={:.3} \
              logged_p2p={:.4} logged_sat={:.3} logged_small_p2p={:.4} delay_margin_ms={:.2} \
              logged_delay_margin_ms={:.2} rough_v={:.4} jitter={:.5} trim_deg={:.2} trim_residual={:.5}",
             ms(nominal.rise90), nominal.overshoot, nominal.sat_frac,
             rps(hover_pitch(&gains, GyroPath::logged(0, dt))), ms(hover.recover),
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
