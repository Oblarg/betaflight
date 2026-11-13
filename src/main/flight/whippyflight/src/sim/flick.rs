//! Stick flicks on the nonlinear plant: the controller and setpoint
//! estimator fly a raised-cosine roll or pitch flick from hover, held and
//! released, with RC packets at a receiver rate like the real link.
//!
//! Each flick is flown twice, with and without the stick, and the response
//! is the difference, so the hover's own settling doesn't count against it.

use whippyunits::value;

use super::noise::Noise;
use super::plant::{Model, Plant};
use super::units::*;
use crate::controller::ControllerState;
use crate::gains::GainStorage;
use crate::setpoint::{SetpointEstimator, target_limits};
use crate::types::*;

pub const F_RX: Freq = Freq::new(500.0);
/// Hover before the stick moves, recorded from its end.
pub const SETTLE: Time = Time::new(0.2);
pub const HOLD: Time = Time::new(0.3);
pub const REST: Time = Time::new(0.3);
/// The flicks the evals and the Configurator fly, gentle to saturating.
pub const AMPS: [Rate; 6] = [
    Rate::new(0.05),
    Rate::new(0.2),
    Rate::new(0.5),
    Rate::new(1.0),
    Rate::new(2.0),
    Rate::new(4.0),
];
pub const RISES: [Time; 2] = [Time::new(0.030), Time::new(0.010)];
/// Betaflight axes flown: roll, pitch.
pub const AXES: [usize; 2] = [0, 1];
/// Motor roughness is the RMS of the commanded voltages high-passed here
/// (tools/extract_noise.py ROUGH_HP_HZ, BF's own figure for a log).
pub const ROUGH_HP: Freq = Freq::new(50.0);

fn s_curve(x: f64) -> f64 {
    0.5 * (1.0 - (core::f64::consts::PI * x.clamp(0.0, 1.0)).cos())
}

/// The controller's microsecond clock at plant time `t` (truncated).
pub fn micros(t: Time) -> Micros {
    Micros::new(value!(t, us) as u32)
}

/// One flight, per controller tick from the end of [`SETTLE`], on the flicked axis.
pub struct Trace {
    pub gyro: Vec<Rate>,
    /// The setpoint estimator's reconstructed stick.
    pub setpoint: Vec<Rate>,
    /// The feedforward target it hands the controller.
    pub target: Vec<Rate>,
    pub stick: Vec<Rate>,
    /// Whether the allocator scaled the command down to fit the motors.
    pub saturated: Vec<bool>,
    /// Commanded motor voltages.
    pub volts: Vec<[Voltage; 4]>,
}

/// The setpoint estimator's target slew (rot/s²) and jerk (rot/s³) limits;
/// [`target_limits`] of the blob when not overridden.
#[derive(Clone, Copy, Default)]
pub struct Limits {
    pub slew: Option<f32>,
    pub jerk: Option<f32>,
}

/// Fly a flick to `amp` on BF axis `axis`, rising over `rise`.
pub fn fly(
    gains: &GainStorage,
    model: &Model,
    axis: usize,
    amp: Rate,
    rise: Time,
    limits: Limits,
) -> Trace {
    fly_with(gains, model, axis, amp, rise, limits, None)
}

/// [`fly`] with `noise` added to the gyro and motor speeds the controller reads.
pub fn fly_with(
    gains: &GainStorage,
    model: &Model,
    axis: usize,
    amp: Rate,
    rise: Time,
    limits: Limits,
    noise: Option<&Noise>,
) -> Trace {
    let dt: Time = gains.header.dt.lossy_into();
    let stick = |t: Time| {
        let t = t - SETTLE;
        amp * (s_curve(f64::from(t / rise)) - s_curve(f64::from((t - rise - HOLD) / rise)))
    };
    let mut ctrl = ControllerState::new();
    let mut sp = SetpointEstimator::new();
    let (slew, jerk) = target_limits(gains);
    sp.configure(
        model.stick_tracking,
        limits.slew.map_or(slew, RotPerSecSq::new),
        limits.jerk.map_or(jerk, RotPerSecCu::new),
    );
    model.configure(&mut ctrl, &mut sp);
    let mut plant = Plant::hovering(model, 1.0);
    let throttle = model.throttle_at(1.0);
    let t_end = SETTLE + 2.0 * rise + HOLD + REST;
    let mut next_pkt = Time::new(0.0);
    let mut tr = Trace {
        gyro: vec![],
        setpoint: vec![],
        target: vec![],
        stick: vec![],
        saturated: vec![],
        volts: vec![],
    };
    for k in 0..f64::from(t_end / dt) as usize {
        let t = k as f64 * dt;
        let now_us = micros(t);
        if t >= next_pkt {
            let mut rc = [RotPerSec::new(0.0); 3];
            rc[axis] = stick(t).lossy_into();
            sp.packet(rc, now_us, (1.0 / F_RX).lossy_into());
            next_pkt += 1.0 / F_RX;
        }
        let (rate, target) = sp.tick(now_us, dt.lossy_into());
        let (mut gyro_rps, mut motor_rps) = (plant.gyro_bf(), plant.motor_rps());
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
        for _ in 0..4 {
            plant.rk4(&volts, dt / 4.0);
        }
        if t >= SETTLE {
            tr.gyro.push(plant.gyro_bf()[axis].lossy_into());
            tr.setpoint.push(rate[axis].lossy_into());
            tr.target.push(target[axis].lossy_into());
            tr.stick.push(stick(t));
            tr.saturated.push(ctrl.telemetry().command_scale < 0.999);
            tr.volts.push(volts);
        }
    }
    tr
}

/// How one flick was tracked; ratios are of the flick's amplitude.
#[derive(Clone, Copy, Debug)]
pub struct Metrics {
    /// Feedforward target overshoot.
    pub target_ovs: f64,
    /// Target and gyro lag behind the stick at half amplitude.
    pub target_lag: Time,
    pub gyro_lag: Time,
    /// Time to 90% of the amplitude; infinite if never reached before release.
    pub rise90: Time,
    pub overshoot: f64,
    /// Mean error over the last 100 ms of the hold.
    pub hold_err: f64,
    /// RMS gyro − setpoint through the release and 50 ms after.
    pub track_rms: f64,
    /// Largest gyro excursion after that.
    pub residual: f64,
    /// Fraction of ticks the motors saturated.
    pub sat_frac: f64,
    /// What sensor noise adds to the motor voltages over the flick: RMS of the
    /// [`ROUGH_HP`]-high-passed difference from the noise-free flight, mean
    /// over motors.  None when flown without noise.
    pub rough_v: Option<Voltage>,
}

/// A flick's response: the gyro difference from the stick-free flight, and
/// the stick, setpoint, target and saturation of the flight with the stick.
pub struct Flick {
    pub dt: Time,
    pub response: Vec<Rate>,
    pub run: Trace,
    pub metrics: Metrics,
}

pub fn measure(
    gains: &GainStorage,
    model: &Model,
    axis: usize,
    amp: Rate,
    rise: Time,
    limits: Limits,
) -> Flick {
    measure_with(gains, model, axis, amp, rise, limits, None)
}

/// [`measure`], and with `noise` also the motor roughness it causes (a third flight).
pub fn measure_with(
    gains: &GainStorage,
    model: &Model,
    axis: usize,
    amp: Rate,
    rise: Time,
    limits: Limits,
    noise: Option<&Noise>,
) -> Flick {
    let dt: Time = gains.header.dt.lossy_into();
    let base = fly(gains, model, axis, Rate::new(0.0), rise, limits);
    let run = fly(gains, model, axis, amp, rise, limits);
    let y: Vec<Rate> = run
        .gyro
        .iter()
        .zip(&base.gyro)
        .map(|(&a, &b)| a - b)
        .collect();
    let idx = |t: Time| (f64::from(t / dt) as usize).min(y.len() - 1);
    let (release, end) = (idx(rise + HOLD), idx(2.0 * rise + HOLD));
    let rise90 = y[..release]
        .iter()
        .position(|&v| v >= 0.9 * amp)
        .map_or(Time::new(f64::INFINITY), |i| i as f64 * dt);
    let peak = y[..release]
        .iter()
        .fold(Rate::new(f64::MIN), |m, &v| m.max(v));
    let hold = &y[idx(rise + HOLD - Time::new(0.1))..release];
    let track_end = idx(2.0 * rise + HOLD + Time::new(0.05));
    let track = (0..track_end)
        .map(|i| (y[i] - run.setpoint[i]) * (y[i] - run.setpoint[i]))
        .fold(SpeedSq::new(0.0), |acc, e| acc + e)
        / track_end as f64;
    let half = |v: &[Rate]| {
        v[..release]
            .iter()
            .position(|&x| x >= 0.5 * amp)
            .unwrap_or(release) as f64
            * dt
    };
    let t_stick = half(&run.stick);
    let target_peak = run.target[..release]
        .iter()
        .fold(Rate::new(f64::MIN), |m, &v| m.max(v));
    let hold_mean = hold.iter().fold(Rate::new(0.0), |acc, &v| acc + v) / hold.len() as f64;
    let residual = y[track_end..]
        .iter()
        .fold(Rate::new(0.0), |m, v| m.max(v.abs()));
    let metrics = Metrics {
        target_ovs: f64::from(target_peak / amp) - 1.0,
        target_lag: half(&run.target) - t_stick,
        gyro_lag: half(&y) - t_stick,
        rise90,
        overshoot: f64::from(peak / amp) - 1.0,
        hold_err: f64::from(hold_mean / amp) - 1.0,
        track_rms: f64::from(track.sqrt() / amp),
        residual: f64::from(residual / amp),
        sat_frac: run.saturated[..end].iter().filter(|&&s| s).count() as f64 / end as f64,
        rough_v: noise.map(|noise| {
            let noisy = fly_with(gains, model, axis, amp, rise, limits, Some(noise));
            roughness(&noisy.volts, &run.volts, dt)
        }),
    };
    Flick {
        dt,
        response: y,
        run,
        metrics,
    }
}

/// RMS of the [`ROUGH_HP`] one-pole high-pass of `a − b`, mean over motors.
pub(crate) fn roughness(a: &[[Voltage; 4]], b: &[[Voltage; 4]], dt: Time) -> Voltage {
    let w = f64::from(2.0 * core::f64::consts::PI * ROUGH_HP * dt);
    let k = w / (w + 1.0);
    let diff = |i: usize, m: usize| a[i][m] - b[i][m];
    let mut lp: [Voltage; 4] = core::array::from_fn(|m| diff(0, m));
    let mut sum = VoltageSq::new(0.0);
    for i in 0..a.len() {
        for (m, lp) in lp.iter_mut().enumerate() {
            *lp += k * (diff(i, m) - *lp);
            let hp = diff(i, m) - *lp;
            sum += hp * hp / 4.0;
        }
    }
    (sum / a.len() as f64).sqrt()
}

/// One entry of [`all_flicks`]: BF axis, rise time, amplitude, and the flick.
pub type FlickRun = (usize, Time, Rate, Flick);

/// Every (axis, rise, amplitude) flick, in table order.
pub fn all_flicks(gains: &GainStorage, model: &Model, limits: Limits) -> Vec<FlickRun> {
    let mut out = vec![];
    for axis in AXES {
        for rise in RISES {
            for amp in AMPS {
                out.push((
                    axis,
                    rise,
                    amp,
                    measure(gains, model, axis, amp, rise, limits),
                ));
            }
        }
    }
    out
}
