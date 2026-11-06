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

use common::plant::Plant;

static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn battery_v() -> f32 {
    common::plant::model().battery_v()
}

/// Hover throttle of the plant model, motors ~11% above hover.
fn throttle() -> f32 {
    common::plant::model().throttle_at(300.0 / 270.0)
}
const F_RX: f64 = 500.0;
const MAX_RATE: f64 = 2.0; // rot/s at full stick
const JITTER_S: f64 = 50e-6;
const DROP_P: f64 = 0.02;
const SETTLE_S: f64 = 2.0;

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
fn packets(stick: &dyn Fn(f64) -> f64, t_end: f64) -> Vec<(f64, f64)> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let quant = 2.0 * MAX_RATE / 2048.0;
    let mut out = Vec::new();
    let mut n = 0;
    loop {
        let t = n as f64 / F_RX + JITTER_S * rng.normal();
        n += 1;
        if t > t_end {
            break;
        }
        if rng.uniform() < DROP_P || t < 0.0 {
            continue;
        }
        out.push((t, (stick(t) / quant).round() * quant));
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out
}

#[derive(Clone, Copy)]
enum Recon {
    Raw,
    Pt3 { cutoff_hz: f64 },
    /// The firmware's [`SetpointEstimator`].
    /// `jerk` ≤ 0: slew limit only.
    Estimator { lambda: f32, slew: f32, jerk: f32 },
}

impl Recon {
    fn name(&self) -> String {
        match self {
            Recon::Raw => "raw staircase, r+hΔr".into(),
            Recon::Pt3 { cutoff_hz } => format!("PT3 {cutoff_hz:.0} Hz, r+hΔr (BF)"),
            Recon::Estimator { lambda, slew, .. } if slew.is_infinite() => format!("α-β λ={lambda}, no slew limit"),
            Recon::Estimator { lambda, slew, jerk } if *jerk > 0.0 => {
                format!("α-β λ={lambda}, slew {slew} rot/s², jerk {jerk} rot/s³")
            }
            Recon::Estimator { lambda, slew, .. } => format!("α-β λ={lambda}, slew {slew} rot/s²"),
        }
    }
}

/// Per-tick setpoint r(t) and optional feedforward target.
struct Reconstructor {
    kind: Recon,
    axis: usize,
    dt: f64,
    latest: f64,
    pt3: [f64; 3],
    estimator: SetpointEstimator,
}

fn to_us(t: f64) -> u32 {
    (t * 1e6) as u32
}

impl Reconstructor {
    fn new(kind: Recon, axis: usize, dt: f64) -> Self {
        let mut estimator = SetpointEstimator::new();
        if let Recon::Estimator { lambda, slew, jerk } = kind {
            estimator.configure(lambda, slew, jerk);
        }
        Self { kind, axis, dt, latest: 0.0, pt3: [0.0; 3], estimator }
    }

    fn packet(&mut self, t: f64, v: f64) {
        self.latest = v;
        let mut rate = [0.0f32; 3];
        rate[self.axis] = v as f32;
        self.estimator.packet(rate, to_us(t), (1.0 / F_RX) as f32);
    }

    fn tick(&mut self, t: f64) -> (f64, Option<f64>) {
        match self.kind {
            Recon::Raw => (self.latest, None),
            Recon::Pt3 { cutoff_hz } => {
                let w = 2.0 * core::f64::consts::PI * cutoff_hz * 1.961459177 * self.dt;
                let k = w / (w + 1.0);
                self.pt3[0] += k * (self.latest - self.pt3[0]);
                self.pt3[1] += k * (self.pt3[0] - self.pt3[1]);
                self.pt3[2] += k * (self.pt3[1] - self.pt3[2]);
                (self.pt3[2], None)
            }
            Recon::Estimator { .. } => {
                let (rate, target) = self.estimator.tick(to_us(t), self.dt as f32);
                (rate[self.axis] as f64, Some(target[self.axis] as f64))
            }
        }
    }
}

struct Metrics {
    track_rms: f64,
    overshoot: f64,
    volt_rough: f64,
    sat_frac: f64,
}

/// Fly `stick` on BF `axis` (offset by the settle time) with reconstruction `kind`.
fn fly(gains: &GainStorage, axis: usize, stick: &dyn Fn(f64) -> f64, dur: f64, kind: Recon) -> Metrics {
    let dt = gains.header.dt as f64;
    let substeps = 4;
    let t_end = SETTLE_S + dur;
    let shifted = |t: f64| stick(t - SETTLE_S);
    let pkts = packets(&shifted, t_end);
    let mut recon = Reconstructor::new(kind, axis, dt);
    let mut ctrl = ControllerState::new();
    let mut plant = Plant::hovering(300.0 / 270.0);
    let mut volts = [0.0f64; 4];
    let mut next = 0;
    let (mut se, mut n, mut rough, mut sat) = (0.0, 0usize, 0.0, 0usize);
    let mut peak = f64::MIN;
    let final_value = stick(dur);
    let ticks = (t_end / dt) as usize;
    for k in 0..ticks {
        let t = k as f64 * dt;
        while next < pkts.len() && pkts[next].0 <= t {
            recon.packet(pkts[next].0, pkts[next].1);
            next += 1;
        }
        let (r, target) = recon.tick(t);
        let mut rate = [0.0f32; 3];
        rate[axis] = r as f32;
        let sp = SetpointData { rate_rps: rate, throttle: throttle() };
        let sensors = SensorData {
            gyro_rps: plant.gyro_bf(),
            motor_rps: plant.omega.map(|w| w as f32),
            battery_voltage: battery_v(),
            accel_g: [0.0; 3],
        };
        let out = match target {
            Some(tg) => {
                let mut tgt = [0.0f32; 3];
                tgt[axis] = tg as f32;
                ctrl.update_with_target(&sensors, &sp, tgt, gains)
            }
            None => ctrl.update(&sensors, &sp, gains),
        };
        let new_volts = out.motor_duty.map(|d| (d * battery_v()) as f64);
        if t >= SETTLE_S {
            let y = sensors.gyro_rps[axis] as f64;
            se += (y - shifted(t)).powi(2);
            peak = peak.max(y);
            rough += (0..4).map(|i| (new_volts[i] - volts[i]).powi(2)).sum::<f64>() / 4.0;
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

fn flick(rise: f64) -> impl Fn(f64) -> f64 {
    move |t: f64| {
        let x = ((t - 0.05) / rise).clamp(0.0, 1.0);
        0.5 * (1.0 - (core::f64::consts::PI * x).cos())
    }
}

#[test]
#[ignore]
fn compare_setpoint_reconstructions() {
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(TEST_WFG), &mut gains), GainLoadResult::Ok);
    let (slew, jerk) = target_limits(&gains);

    let candidates = [
        Recon::Raw,
        Recon::Pt3 { cutoff_hz: 0.375 * F_RX },
        Recon::Estimator { lambda: 8.0, slew: f32::INFINITY, jerk: 0.0 },
        Recon::Estimator { lambda: 8.0, slew: 100.0, jerk: 0.0 },
        Recon::Estimator { lambda: 8.0, slew: 50.0, jerk: 0.0 },
        Recon::Estimator { lambda: 8.0, slew: 35.0, jerk: 0.0 },
        Recon::Estimator { lambda: 8.0, slew, jerk },
    ];
    let slow = flick(0.030);
    let medium = flick(0.015);
    let snap = flick(0.008);
    let sine = |t: f64| if t < 0.0 { 0.0 } else { (2.0 * core::f64::consts::PI * 6.0 * t).sin() };
    let scenarios: [(&str, usize, &dyn Fn(f64) -> f64, f64); 7] = [
        ("pitch flick 0→1 rot/s, 30 ms", 1, &slow, 0.3),
        ("pitch flick 0→1 rot/s, 15 ms", 1, &medium, 0.3),
        ("pitch snap  0→1 rot/s, 8 ms", 1, &snap, 0.3),
        ("roll flick  0→1 rot/s, 30 ms", 0, &slow, 0.3),
        ("roll flick  0→1 rot/s, 15 ms", 0, &medium, 0.3),
        ("roll snap   0→1 rot/s, 8 ms", 0, &snap, 0.3),
        ("pitch 6 Hz ±1 rot/s", 1, &sine, 0.5),
    ];
    for (label, axis, stick, dur) in scenarios {
        println!("\n{label}");
        println!("  {:30} {:>10} {:>10} {:>12} {:>8}", "reconstruction", "track_rms", "overshoot", "ΔV_rms/tick", "sat");
        for kind in candidates {
            let m = fly(&gains, axis, stick, dur, kind);
            println!("  {:30} {:10.4} {:+10.4} {:12.4} {:7.1}%",
                     kind.name(), m.track_rms, m.overshoot, m.volt_rough, 100.0 * m.sat_frac);
        }
    }
}
