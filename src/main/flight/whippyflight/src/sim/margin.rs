//! Robustness of the closed loop, with the gyro passed through the airframe's
//! gyro low-pass stages (as PT1s): the delay margin, how much extra gyro lag
//! it tolerates (the largest pure delay a gentle and a medium roll flick
//! still fly cleanly with), and the integrator trim, how much attitude a CG
//! shift costs before the integral takes up the torque.  Also how fast a
//! knock dies out and how much measured sensor noise roughens the motors at
//! hover: with the delay margin and trim, the tradeoffs the LQR weights set.

use whippyunits::api::rescale;

use super::flick::roughness;
use super::noise::Noise;
use super::plant::{Model, Plant};
use super::units::*;
use crate::controller::ControllerState;
use crate::gains::GainStorage;
use crate::setpoint::SetpointEstimator;
use crate::types::*;

const F_RX: Freq = Freq::new(500.0);
const SETTLE: Time = Time::new(1.0);
const RISE: Time = Time::new(0.030);
const HOLD: Time = Time::new(0.5);
const REST: Time = Time::new(0.6);
/// Motor speeds the margin flights hover at, of the model's hover speed.
pub const HOVER_SCALE: f64 = 250.0 / 270.0;
/// The flicks that must fly cleanly.
const AMPS: [Rate; 2] = [Rate::new(0.05), Rate::new(0.5)];
pub const MAX_MARGIN: Millis = Millis::new(16.0);
const MARGIN_RESOLUTION: Millis = Millis::new(0.1);
/// Fraction of the front-back arm the CG shifts by in [`trim`].
pub const TRIM_CG_SHIFT: f64 = 0.05;
/// How long [`trim`] flies after the shift.
pub const TRIM: Time = Time::new(3.0);
/// The roll-rate knock [`kick_iae`] recovers from.
pub const KICK: Rate = Rate::new(0.2);

/// Gyro path: PT1 stages (BF's pt1FilterGain) then a delay in PID ticks,
/// fractional delays linearly interpolated between ticks.
pub struct GyroPath {
    gains: Vec<f64>,
    state: Vec<[Rate; 3]>,
    /// Filtered samples, newest last.
    history: std::collections::VecDeque<[Rate; 3]>,
    delay_ticks: f64,
}

impl GyroPath {
    pub fn new(cutoffs: &[Freq], delay_ticks: usize, dt: Time) -> Self {
        Self::delayed(cutoffs, delay_ticks as f64, dt)
    }

    pub fn delayed(cutoffs: &[Freq], delay_ticks: f64, dt: Time) -> Self {
        let gains = cutoffs
            .iter()
            .map(|&f| {
                let w = f64::from(2.0 * core::f64::consts::PI * f * dt);
                w / (w + 1.0)
            })
            .collect::<Vec<_>>();
        let n = gains.len();
        let history = vec![[Rate::new(0.0); 3]; delay_ticks as usize + 2].into();
        Self {
            gains,
            state: vec![[Rate::new(0.0); 3]; n],
            history,
            delay_ticks,
        }
    }

    pub fn filter(&mut self, x: [RotPerSec; 3]) -> [RotPerSec; 3] {
        let mut y: [Rate; 3] = x.map(|v| v.lossy_into());
        for (&k, s) in self.gains.iter().zip(self.state.iter_mut()) {
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
        core::array::from_fn(|i| ((1.0 - f) * a[i] + f * b[i]).lossy_into())
    }
}

/// `model`'s gyro low-pass stages as PT1 cutoffs: BF's PT2/PT3 are 2/3 PT1s
/// at the cutoff × CUTOFF_CORRECTION_PT2/PT3 (filter.c).
pub fn pt1_cutoffs(model: &Model) -> Result<Vec<Freq>, String> {
    let mut out = vec![];
    for (kind, hz) in &model.gyro_lpf {
        match kind.as_str() {
            "PT1" => out.push(*hz),
            "PT2" => out.extend([*hz * 1.553_773_974; 2]),
            "PT3" => out.extend([*hz * 1.961_459_177; 3]),
            _ => return Err(format!("gyro LPF type {kind} is not modeled")),
        }
    }
    Ok(out)
}

/// Peak-to-peak of `(t, y)` samples about their least-squares line.
pub fn detrended_p2p(samples: &[(Time, Rate)]) -> Rate {
    let n = samples.len() as f64;
    let (mt, my) = samples
        .iter()
        .fold((Time::new(0.0), Rate::new(0.0)), |(a, b), &(t, y)| {
            (a + t / n, b + y / n)
        });
    let (stt, sty) = samples
        .iter()
        .fold((TimeSq::new(0.0), Angle::new(0.0)), |(a, b), &(t, y)| {
            (a + (t - mt) * (t - mt), b + (t - mt) * (y - my))
        });
    let slope = if stt > TimeSq::new(0.0) {
        sty / stt
    } else {
        AngAccel::new(0.0)
    };
    let (lo, hi) = samples.iter().fold(
        (Rate::new(f64::MAX), Rate::new(f64::MIN)),
        |(lo, hi), &(t, y)| {
            let r = y - my - slope * (t - mt);
            (lo.min(r), hi.max(r))
        },
    );
    hi - lo
}

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

struct Flight {
    /// Detrended peak-to-peak roll rate over the last 0.3 s.
    rest_p2p: Rate,
    /// Fraction of ticks the motors saturated.
    sat_frac: f64,
    /// True body rates (model axes) from SETTLE on.
    rates: Vec<[Rate; 3]>,
    /// Commanded motor voltages from SETTLE on.
    volts: Vec<[Voltage; 4]>,
}

/// What disturbs a [`flight`] at SETTLE: a pitch angular acceleration
/// (BF axes) from then on, a step in roll rate, and sensor noise throughout.
struct Disturbance<'n> {
    torque_pitch: AngAccel,
    kick: Rate,
    noise: Option<&'n Noise>,
}

const CALM: Disturbance<'static> = Disturbance {
    torque_pitch: AngAccel::new(0.0),
    kick: Rate::new(0.0),
    noise: None,
};

/// Fly a roll flick to `amp` through `path`, disturbed by `disturbance`, for
/// `after` past SETTLE.
fn flight(
    gains: &GainStorage,
    model: &Model,
    amp: Rate,
    mut path: GyroPath,
    disturbance: Disturbance,
    after: Time,
) -> Flight {
    let Disturbance {
        torque_pitch,
        kick,
        noise,
    } = disturbance;
    let dt: Time = gains.header.dt.lossy_into();
    let stick = |t: Time| {
        let t = t - SETTLE;
        amp * (s_curve(f64::from(t / RISE)) - s_curve(f64::from((t - RISE - HOLD) / RISE)))
    };
    let mut ctrl = ControllerState::new();
    let mut sp = SetpointEstimator::new();
    sp.configure_for(gains);
    model.configure(&mut ctrl, &mut sp);
    let mut plant = Plant::hovering(model, HOVER_SCALE);
    let throttle = model.throttle_at(1.0);
    let t_end = SETTLE + after;
    let mut next_pkt = Time::new(0.0);
    let (mut rest, mut sat, mut n, mut rates) = (Vec::new(), 0usize, 0usize, Vec::new());
    let (mut kicked, mut volts_trace) = (false, Vec::new());
    for k in 0..f64::from(t_end / dt) as usize {
        let t = k as f64 * dt;
        let now_us = super::flick::micros(t);
        if !kicked && t >= SETTLE {
            plant.rate[2] -= kick; // model roll is -BF roll
            kicked = true;
        }
        if t >= next_pkt {
            let zero = RotPerSec::new(0.0);
            sp.packet(
                [stick(t).lossy_into(), zero, zero],
                now_us,
                (1.0 / F_RX).lossy_into(),
            );
            next_pkt += 1.0 / F_RX;
        }
        let (rate, target) = sp.tick(now_us, dt.lossy_into());
        let mut gyro_rps = path.filter(plant.gyro_bf());
        let mut motor_rps = plant.motor_rps();
        if let Some(noise) = noise {
            let s = noise.at(t);
            (0..3).for_each(|a| gyro_rps[a] += s.gyro[a]);
            (0..4).for_each(|m| motor_rps[m] += s.motor[m]);
        }
        let sensors = SensorData {
            gyro_rps,
            motor_rps,
            battery_voltage: model.battery_v(),
            accel: [MetersPerSecSq::new(0.0); 3],
        };
        let out = ctrl.update_with_target(
            &sensors,
            &SetpointData {
                rate_rps: rate,
                throttle,
            },
            target,
            gains,
        );
        let volts = model.duty_volts(out.motor_duty);
        let no_torque = AngAccel::new(0.0);
        let torque = [
            no_torque,
            if t >= SETTLE { torque_pitch } else { no_torque },
            no_torque,
        ];
        for _ in 0..4 {
            plant.rk4_with_torque(&volts, dt / 4.0, torque);
        }
        if t >= SETTLE {
            n += 1;
            rates.push(plant.rate);
            volts_trace.push(volts);
            if ctrl.telemetry().command_scale < 0.999 {
                sat += 1;
            }
            if t - SETTLE >= t_end - SETTLE - Time::new(0.3) {
                rest.push((t, plant.gyro_bf()[0].lossy_into()));
            }
        }
    }
    Flight {
        rest_p2p: detrended_p2p(&rest),
        sat_frac: sat as f64 / n as f64,
        rates,
        volts: volts_trace,
    }
}

/// Integrator trim: hover, shift the CG by [`TRIM_CG_SHIFT`] (a step pitch
/// torque: the hover thrust on that lever arm of the unit front-back arm) at
/// SETTLE, gyro through `cutoffs`, and return (attitude it costs, = |∫ pitch
/// rate| over [`TRIM`]; residual pitch rate over the last 0.5 s).  The rate
/// loop alone holds a constant rate error under a constant torque; the
/// (leaky) integrator is what brings it back toward zero.
pub fn trim(gains: &GainStorage, model: &Model, cutoffs: &[Freq]) -> (Angle, Rate) {
    let dt: Time = gains.header.dt.lossy_into();
    let w = HOVER_SCALE * model.hover_rps;
    let torque = TRIM_CG_SHIFT * model.pitch_prop_vel_sq * 4.0 * (w * w);
    let f = flight(
        gains,
        model,
        Rate::new(0.0),
        GyroPath::new(cutoffs, 0, dt),
        Disturbance {
            torque_pitch: torque,
            ..CALM
        },
        TRIM,
    );
    let pitch: Vec<Rate> = f.rates.iter().map(|r| r[1]).collect();
    let angle: Angle = pitch.iter().fold(Rate::new(0.0), |acc, &r| acc + r) * dt;
    let tail = f64::from(Time::new(0.5) / dt) as usize;
    let residual = pitch[pitch.len() - tail..]
        .iter()
        .fold(Rate::new(0.0), |acc, &r| acc + r)
        / tail as f64;
    (angle.abs(), residual)
}

/// Largest extra gyro delay on top of `cutoffs` that still flies both
/// flicks cleanly (settled without limit cycling or chattering on the
/// rails): bisected over 0–[`MAX_MARGIN`] with a fractional delay, so it
/// ranks candidates instead of saturating at a tick cap.  −1 ms if the path
/// isn't clean without extra delay; [`MAX_MARGIN`] if it never breaks.  A
/// time rather than ticks so loop rates compare.
pub fn delay_margin(gains: &GainStorage, model: &Model, cutoffs: &[Freq]) -> Millis {
    let dt: Time = gains.header.dt.lossy_into();
    let ok = |delay: Millis| {
        let delay: Time = rescale(delay);
        AMPS.iter().all(|&a| {
            let path = GyroPath::delayed(cutoffs, f64::from(delay / dt), dt);
            let f = flight(gains, model, a, path, CALM, 2.0 * RISE + HOLD + REST);
            f.rest_p2p < Rate::new(0.005) && f.sat_frac < 0.1
        })
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

/// Hover, knock the roll rate by [`KICK`] at SETTLE, gyro through `cutoffs`,
/// and return ∫|roll rate| dt / KICK: how long the knock takes to die out,
/// continuous where a time to 10% would jump with each undershoot lobe.
pub fn kick_iae(gains: &GainStorage, model: &Model, cutoffs: &[Freq]) -> Millis {
    let dt: Time = gains.header.dt.lossy_into();
    let f = flight(
        gains,
        model,
        Rate::new(0.0),
        GyroPath::new(cutoffs, 0, dt),
        Disturbance { kick: KICK, ..CALM },
        2.0 * RISE + HOLD + REST,
    );
    let iae: Time = f
        .rates
        .iter()
        .fold(Rate::new(0.0), |acc, r| acc + r[2].abs())
        * dt
        / KICK;
    rescale(iae)
}

/// Hover with `noise` on the gyro (after `cutoffs`) and motor speeds, and
/// return the motor roughness it causes: as [`super::flick::Metrics::rough_v`],
/// against the same hover without it.
pub fn hover_roughness(
    gains: &GainStorage,
    model: &Model,
    cutoffs: &[Freq],
    noise: &Noise,
) -> Voltage {
    let dt: Time = gains.header.dt.lossy_into();
    let hover = |noise| {
        flight(
            gains,
            model,
            Rate::new(0.0),
            GyroPath::new(cutoffs, 0, dt),
            Disturbance { noise, ..CALM },
            2.0 * RISE + HOLD + REST,
        )
    };
    roughness(&hover(Some(noise)).volts, &hover(None).volts, dt)
}
