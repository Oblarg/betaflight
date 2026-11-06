//! Closed-loop comparison of setpoint reconstructions for the reference
//! feedforward.  Not a regression test; run with
//!
//!   cargo test --target aarch64-apple-darwin --features std \
//!       --test eval_setpoint_reconstruction -- --ignored --nocapture
//!
//! The real controller (test_gains.wfg) flies the nonlinear plant that
//! gen_wfg.py linearizes, with coefficients from the regression the blob was
//! generated from.  Stick inputs reach it as an RX packet stream (jitter,
//! drops, 11-bit quantization) and are reconstructed per tick by each
//! candidate.

mod common;

use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::setpoint::{target_limits, SetpointEstimator};
use whippyflight_control::types::*;
use whippyunits::value;

use common::plant::{duty_volts, Plant};
use common::units::{Freq, Rate, SpeedSq, Time, Voltage, VoltageSq};

static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn battery_v() -> Volt {
    common::plant::model().battery_v()
}

/// Hover throttle of the plant model, motors ~11% above hover.
fn throttle() -> f32 {
    common::plant::model().throttle_at(300.0 / 270.0)
}
const F_RX: Freq = Freq::new(500.0);
const MAX_RATE: Rate = Rate::new(2.0); // at full stick
const JITTER: Time = Time::new(50e-6);
const DROP_P: f64 = 0.02;
const SETTLE: Time = Time::new(2.0);

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// RX packet arrival times and quantized stick values for the whole run.
fn packets(stick: &dyn Fn(Time) -> Rate, t_end: Time) -> Vec<(Time, Rate)> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let quant = 2.0 * MAX_RATE / 2048.0;
    let mut out = Vec::new();
    let mut n = 0;
    loop {
        let t = n as f64 / F_RX + JITTER * rng.normal();
        n += 1;
        if t > t_end {
            break;
        }
        if rng.uniform() < DROP_P || t < Time::new(0.0) {
            continue;
        }
        out.push((t, f64::from(stick(t) / quant).round() * quant));
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

#[derive(Clone, Copy)]
enum Recon {
    Raw,
    Pt3 { cutoff: Freq },
    /// The firmware's [`SetpointEstimator`].
    /// `jerk` ≤ 0: slew limit only.
    Estimator { lambda: f32, slew: RotPerSecSq, jerk: RotPerSecCu },
}

impl Recon {
    fn name(&self) -> String {
        match *self {
            Recon::Raw => "raw staircase, r+hΔr".into(),
            Recon::Pt3 { cutoff } => format!("PT3 {:.0} Hz, r+hΔr (BF)", value!(cutoff, Hz)),
            Recon::Estimator { lambda, slew, .. } if !slew.is_finite() => format!("α-β λ={lambda}, no slew limit"),
            Recon::Estimator { lambda, slew, jerk } if jerk > RotPerSecCu::new(0.0) => {
                format!("α-β λ={lambda}, slew {} rot/s², jerk {} rot/s³",
                        value!(slew, rot / s ^ 2, f32), value!(jerk, rot / s ^ 3, f32))
            }
            Recon::Estimator { lambda, slew, .. } => format!("α-β λ={lambda}, slew {} rot/s²", value!(slew, rot / s ^ 2, f32)),
        }
    }
}

/// Per-tick setpoint r(t) and optional feedforward target.
struct Reconstructor {
    kind: Recon,
    axis: usize,
    dt: Time,
    latest: Rate,
    pt3: [Rate; 3],
    estimator: SetpointEstimator,
}

impl Reconstructor {
    fn new(kind: Recon, axis: usize, dt: Time) -> Self {
        let mut estimator = SetpointEstimator::new();
        if let Recon::Estimator { lambda, slew, jerk } = kind {
            estimator.configure(lambda, slew, jerk);
        }
        Self { kind, axis, dt, latest: Rate::new(0.0), pt3: [Rate::new(0.0); 3], estimator }
    }

    fn packet(&mut self, t: Time, v: Rate) {
        self.latest = v;
        let mut rate = [RotPerSec::new(0.0); 3];
        rate[self.axis] = v.lossy_into();
        self.estimator.packet(rate, common::micros64(t), (1.0 / F_RX).lossy_into());
    }

    fn tick(&mut self, t: Time) -> (Rate, Option<Rate>) {
        match self.kind {
            Recon::Raw => (self.latest, None),
            Recon::Pt3 { cutoff } => {
                let w = f64::from(2.0 * core::f64::consts::PI * cutoff * 1.961459177 * self.dt);
                let k = w / (w + 1.0);
                self.pt3[0] = self.pt3[0] + k * (self.latest - self.pt3[0]);
                self.pt3[1] = self.pt3[1] + k * (self.pt3[0] - self.pt3[1]);
                self.pt3[2] = self.pt3[2] + k * (self.pt3[1] - self.pt3[2]);
                (self.pt3[2], None)
            }
            Recon::Estimator { .. } => {
                let (rate, target) = self.estimator.tick(common::micros64(t), self.dt.lossy_into());
                (rate[self.axis].lossy_into(), Some(target[self.axis].lossy_into()))
            }
        }
    }
}

struct Metrics {
    track_rms: Rate,
    overshoot: Rate,
    volt_rough: Voltage,
    sat_frac: f64,
}

/// Fly `stick` on BF `axis` (offset by the settle time) with reconstruction `kind`.
fn fly(gains: &GainStorage, axis: usize, stick: &dyn Fn(Time) -> Rate, dur: Time, kind: Recon) -> Metrics {
    let dt: Time = gains.header.dt.lossy_into();
    let substeps = 4;
    let t_end = SETTLE + dur;
    let shifted = |t: Time| stick(t - SETTLE);
    let pkts = packets(&shifted, t_end);
    let mut recon = Reconstructor::new(kind, axis, dt);
    let mut ctrl = ControllerState::new();
    let mut plant = Plant::hovering(300.0 / 270.0);
    let mut volts = [Voltage::new(0.0); 4];
    let mut next = 0;
    let (mut se, mut n, mut rough, mut sat) = (SpeedSq::new(0.0), 0usize, VoltageSq::new(0.0), 0usize);
    let mut peak = Rate::new(f64::MIN);
    let final_value = stick(dur);
    let ticks = f64::from(t_end / dt) as usize;
    for k in 0..ticks {
        let t = k as f64 * dt;
        while next < pkts.len() && pkts[next].0 <= t {
            recon.packet(pkts[next].0, pkts[next].1);
            next += 1;
        }
        let (r, target) = recon.tick(t);
        let mut rate = [RotPerSec::new(0.0); 3];
        rate[axis] = r.lossy_into();
        let sp = SetpointData { rate_rps: rate, throttle: throttle() };
        let sensors = SensorData {
            gyro_rps: plant.gyro_bf(),
            motor_rps: plant.motor_rps(),
            battery_voltage: battery_v(),
            accel_g: [0.0; 3],
        };
        let out = match target {
            Some(tg) => {
                let mut tgt = [RotPerSec::new(0.0); 3];
                tgt[axis] = tg.lossy_into();
                ctrl.update_with_target(&sensors, &sp, tgt, gains)
            }
            None => ctrl.update(&sensors, &sp, gains),
        };
        let new_volts = duty_volts(out.motor_duty);
        if t >= SETTLE {
            let y: Rate = sensors.gyro_rps[axis].lossy_into();
            let e = y - shifted(t);
            se = se + e * e;
            peak = peak.max(y);
            let step = (0..4).fold(VoltageSq::new(0.0), |acc, i| {
                let dv = new_volts[i] - volts[i];
                acc + dv * dv
            });
            rough = rough + step / 4.0;
            if ctrl.telemetry().command_scale < 0.999 {
                sat += 1;
            }
            n += 1;
        }
        volts = new_volts;
        for _ in 0..substeps {
            plant.rk4(&volts, dt / substeps as f64);
        }
    }
    Metrics {
        track_rms: (se / n as f64).sqrt(),
        overshoot: peak - final_value,
        volt_rough: (rough / n as f64).sqrt(),
        sat_frac: sat as f64 / n as f64,
    }
}

/// 0 → 1 rot/s stick flick starting at 50 ms, rising over `rise`.
fn flick(rise: Time) -> impl Fn(Time) -> Rate {
    move |t: Time| {
        let x = f64::from((t - Time::new(0.05)) / rise).clamp(0.0, 1.0);
        Rate::new(1.0) * (0.5 * (1.0 - (core::f64::consts::PI * x).cos()))
    }
}

#[test]
#[ignore]
fn compare_setpoint_reconstructions() {
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(TEST_WFG), &mut gains), GainLoadResult::Ok);
    let (slew, jerk) = target_limits(&gains);

    let no_jerk = RotPerSecCu::new(0.0);
    let candidates = [
        Recon::Raw,
        Recon::Pt3 { cutoff: 0.375 * F_RX },
        Recon::Estimator { lambda: 8.0, slew: RotPerSecSq::new(f32::INFINITY), jerk: no_jerk },
        Recon::Estimator { lambda: 8.0, slew: RotPerSecSq::new(100.0), jerk: no_jerk },
        Recon::Estimator { lambda: 8.0, slew: RotPerSecSq::new(50.0), jerk: no_jerk },
        Recon::Estimator { lambda: 8.0, slew: RotPerSecSq::new(35.0), jerk: no_jerk },
        Recon::Estimator { lambda: 8.0, slew, jerk },
    ];
    let slow = flick(Time::new(0.030));
    let medium = flick(Time::new(0.015));
    let snap = flick(Time::new(0.008));
    let sine = |t: Time| {
        if t < Time::new(0.0) {
            Rate::new(0.0)
        } else {
            Rate::new(1.0) * f64::from(2.0 * core::f64::consts::PI * Freq::new(6.0) * t).sin()
        }
    };
    let (short, long) = (Time::new(0.3), Time::new(0.5));
    let scenarios: [(&str, usize, &dyn Fn(Time) -> Rate, Time); 7] = [
        ("pitch flick 0→1 rot/s, 30 ms", 1, &slow, short),
        ("pitch flick 0→1 rot/s, 15 ms", 1, &medium, short),
        ("pitch snap  0→1 rot/s, 8 ms", 1, &snap, short),
        ("roll flick  0→1 rot/s, 30 ms", 0, &slow, short),
        ("roll flick  0→1 rot/s, 15 ms", 0, &medium, short),
        ("roll snap   0→1 rot/s, 8 ms", 0, &snap, short),
        ("pitch 6 Hz ±1 rot/s", 1, &sine, long),
    ];
    for (label, axis, stick, dur) in scenarios {
        println!("\n{label}");
        println!("  {:30} {:>10} {:>10} {:>12} {:>8}", "reconstruction", "track_rms", "overshoot", "ΔV_rms/tick", "sat");
        for kind in candidates {
            let m = fly(&gains, axis, stick, dur, kind);
            println!("  {:30} {:10.4} {:+10.4} {:12.4} {:7.1}%",
                     kind.name(), value!(m.track_rms, rot / s), value!(m.overshoot, rot / s),
                     value!(m.volt_rough, V), 100.0 * m.sat_frac);
        }
    }
}
