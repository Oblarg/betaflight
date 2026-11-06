//! Integrator windup: ground/landing gating (no accumulation unless
//! airborne) and the integral voltage bound (enough for trim, small enough
//! that a collision's windup is recoverable), on test_gains.wfg and the
//! plant it was generated from.

mod common;

use common::plant::{duty_volts, model, Plant};
use common::units::{AngAccel, Rate, Time};
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage, MotorDrag};
use whippyflight_control::types::*;
use whippyunits::value;

static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn load() -> GainStorage {
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(TEST_WFG), &mut gains), GainLoadResult::Ok);
    assert!(gains.hover_rps > RotPerSec::new(0.0), "blob has no hover speed");
    gains
}

/// Open-loop sensors: every motor at `speed` × the blob's hover speed.
fn sensors(gains: &GainStorage, speed: f32, gyro: [RotPerSec; 3], accel_g: [f32; 3]) -> SensorData {
    SensorData {
        gyro_rps: gyro,
        motor_rps: [speed * gains.hover_rps; 4],
        battery_voltage: model().battery_v(),
        accel_g,
    }
}

/// The throttle that holds the blob's thrust-balanced hover speeds: the
/// collective is net of the affine linearizer's G·ω|ω| (see `allocation`),
/// so it is the mean of V(ωᵢ) − (G·ω|ω|)ᵢ there.
fn hover_throttle(gains: &GainStorage) -> f32 {
    let MotorDrag { k_w, k_w2 } = gains.motor_drag;
    let w = gains.thrust_balance.map(|b| b * gains.hover_rps);
    let affine = |i: usize| (0..4).fold(Volt::new(0.0), |acc, j| acc + gains.affine[i * 4 + j] * w[j] * w[j]);
    let net = (0..4).fold(Volt::new(0.0), |acc, i| acc + (k_w * w[i] + k_w2 * w[i] * w[i] - affine(i)));
    f32::from(0.25 * net / model().battery_v())
}

const STILL: [RotPerSec; 3] = [RotPerSec::new(0.0); 3];
const PITCHING: [RotPerSec; 3] = [RotPerSec::new(0.0), RotPerSec::new(0.3), RotPerSec::new(0.0)];

/// A small roll-rate error the integral would chase (never saturating).
fn stick() -> SetpointData {
    SetpointData { rate_rps: [RotPerSec::new(0.02), RotPerSec::new(0.0), RotPerSec::new(0.0)], throttle: model().throttle_at(1.0) }
}

fn ticks(gains: &GainStorage, duration: Seconds) -> usize {
    f32::from(duration / gains.header.dt) as usize
}

/// Run `duration` of ticks; returns |integral| (roll) after each.
fn run(ctrl: &mut ControllerState, gains: &GainStorage, s: &SensorData, duration: Seconds) -> Vec<Rotation> {
    (0..ticks(gains, duration))
        .map(|_| {
            ctrl.update(s, &stick(), gains);
            assert_eq!(ctrl.telemetry().command_scale, 1.0, "saturated: integral gated by that instead");
            ctrl.telemetry().error_integral[0].abs()
        })
        .collect()
}

const ONE_G: [f32; 3] = [0.0, 0.0, 1.0];
const NO_ACCEL: [f32; 3] = [0.0; 3];

fn growing(i: &[Rotation]) -> bool {
    *i.last().unwrap() > i[0] + Rotation::new(1e-12)
}

/// Each tick only the leak: e_int ← (1 − dt/τ_int)·e_int.
fn only_leaking(gains: &GainStorage, i: &[Rotation]) -> bool {
    let leak = 1.0 - f32::from(gains.header.dt / gains.header.integral_tau);
    i.windows(2).all(|w| (w[1] - w[0] * leak).abs() <= 1e-6 * w[0] + Rotation::new(1e-15))
}

#[test]
fn no_windup_before_takeoff() {
    let gains = load();
    let mut ctrl = ControllerState::new();
    // Armed on the ground, spinning at idle, stick held off center.
    for speed in [0.3, 0.6, 0.8] {
        let i = run(&mut ctrl, &gains, &sensors(&gains, speed, STILL, ONE_G), Seconds::new(2.0));
        assert!(i.iter().all(|&x| x == Rotation::new(0.0)), "speed {speed}×hover: integral {:?}", i.last().unwrap());
    }
    // Takeoff: accumulates once the motors reach 0.85× hover.
    let i = run(&mut ctrl, &gains, &sensors(&gains, 0.9, STILL, NO_ACCEL), Seconds::new(0.1));
    assert!(growing(&i), "no accumulation after takeoff");
}

#[test]
fn landing_stops_accumulation() {
    let gains = load();
    for (accel, hold) in [(ONE_G, 0.5f32), (NO_ACCEL, 1.0)] {
        let hold = Seconds::new(hold);
        let mut ctrl = ControllerState::new();
        let flying = run(&mut ctrl, &gains, &sensors(&gains, 1.0, STILL, NO_ACCEL), Seconds::new(0.5));
        assert!(growing(&flying));
        // Touchdown: motors spun down, still, resting on the ground.
        let landed = sensors(&gains, 0.3, STILL, accel);
        let before = run(&mut ctrl, &gains, &landed, hold - Seconds::new(0.05));
        assert!(growing(&before), "{accel:?}: stopped before the {hold:?} hold");
        let after = run(&mut ctrl, &gains, &landed, Seconds::new(1.0));
        assert!(only_leaking(&gains, &after[ticks(&gains, Seconds::new(0.1))..]),
                "{accel:?}: still accumulating after landing");
        // Takeoff again.
        let again = run(&mut ctrl, &gains, &sensors(&gains, 1.0, STILL, NO_ACCEL), Seconds::new(0.1));
        assert!(growing(&again), "{accel:?}: no accumulation after re-takeoff");
    }
}

#[test]
fn slow_motors_in_the_air_are_not_a_landing() {
    let gains = load();
    // Low thrust but falling (specific force ≪ 1 g), or tumbling: airborne.
    for (gyro, accel) in [(STILL, [0.0, 0.0, 0.3]), (PITCHING, ONE_G), (PITCHING, NO_ACCEL)] {
        let mut ctrl = ControllerState::new();
        run(&mut ctrl, &gains, &sensors(&gains, 1.0, STILL, NO_ACCEL), Seconds::new(0.2));
        let i = run(&mut ctrl, &gains, &sensors(&gains, 0.3, gyro, accel), Seconds::new(2.0));
        assert!(!only_leaking(&gains, &i[i.len() - 100..]), "gyro {gyro:?} accel {accel:?}: counted as landed");
    }
}

// ── Closed loop ─────────────────────────────────────────────────────────

/// Pitch angular acceleration (BF axes) of the hover thrust acting `offset`
/// of the (unit) front-back arm off center.
fn cg_torque(offset: f64) -> AngAccel {
    let m = model();
    offset * m.pitch_prop_vel_sq * 4.0 * (m.hover_rps * m.hover_rps)
}

/// What acts on the hovering airframe at a time.
#[derive(Clone, Copy, Default)]
struct Phase {
    /// External pitch angular acceleration (BF).
    torque: AngAccel,
    /// Pitch stick.
    stick: RotPerSec,
    /// Rotation held at zero (wedged against something).
    blocked: bool,
}

fn plant_dt(gains: &GainStorage) -> Time {
    gains.header.dt.lossy_into()
}

/// Hover the plant through `phase(t)`; returns the BF pitch rate per tick.
fn fly(gains: &GainStorage, duration: Time, phase: impl Fn(Time) -> Phase) -> Vec<Rate> {
    let dt = plant_dt(gains);
    let mut ctrl = ControllerState::new();
    let mut plant = Plant::hovering(1.0);
    let mut out = Vec::new();
    for k in 0..f64::from(duration / dt) as usize {
        let p = phase(k as f64 * dt);
        let s = SensorData {
            gyro_rps: plant.gyro_bf(),
            motor_rps: plant.motor_rps(),
            battery_voltage: model().battery_v(),
            accel_g: [0.0; 3],
        };
        let sp = SetpointData {
            rate_rps: [RotPerSec::new(0.0), p.stick, RotPerSec::new(0.0)],
            throttle: hover_throttle(gains),
        };
        let volts = duty_volts(ctrl.update(&s, &sp, gains).motor_duty);
        for _ in 0..4 {
            plant.rk4_with_torque(&volts, dt / 4.0, [AngAccel::new(0.0), p.torque, AngAccel::new(0.0)]);
        }
        if p.blocked {
            plant.rate = [Rate::new(0.0); 3];
        }
        out.push(plant.gyro_bf()[1].lossy_into());
    }
    out
}

fn torque_from(start: Time, torque: AngAccel) -> impl Fn(Time) -> Phase {
    move |t| Phase { torque: if t >= start { torque } else { AngAccel::new(0.0) }, ..Phase::default() }
}

fn peak(x: &[Rate]) -> Rate {
    x.iter().fold(Rate::new(0.0), |m, r| m.max(r.abs()))
}

fn mean(x: &[Rate]) -> Rate {
    x.iter().fold(Rate::new(0.0), |acc, &r| acc + r) / x.len() as f64
}

/// For messages.
fn rps(x: Rate) -> f64 {
    value!(x, rot / s)
}

/// Residual pitch rate over the last 0.5 s of 10 s with a CG offset applied
/// at 0.5 s (the trim settles in a few seconds).
fn trim_residual(gains: &GainStorage, offset: f64) -> Rate {
    let rate = fly(gains, Time::new(10.0), torque_from(Time::new(0.5), cg_torque(offset)));
    mean(&rate[rate.len() - f64::from(Time::new(0.5) / plant_dt(gains)) as usize..])
}

/// The blob without its integral bound: only the 0.2·Vbat cap.
fn cap_only() -> GainStorage {
    let mut gains = load();
    gains.integral_limit_v = Volt::new(0.0);
    gains
}

/// gen_wfg sizes the bound to trim a 25%-of-arm CG offset: up to there the
/// trim is as good as with only the battery cap, beyond it the integral is
/// pinned and the residual rate grows.
#[test]
fn trim_holds_inside_the_bound_and_pins_beyond_it() {
    let (gains, cap) = (load(), cap_only());
    for offset in [0.05, 0.15, 0.25, -0.25] {
        let (bounded, free) = (trim_residual(&gains, offset), trim_residual(&cap, offset));
        assert!((bounded - free).abs() <= 0.01 * free.abs(),
                "{offset}: residual {:.5} rot/s, {:.5} with only the cap", rps(bounded), rps(free));
    }
    let (bounded, free) = (trim_residual(&gains, 0.4), trim_residual(&cap, 0.4));
    assert!(bounded.abs() > 3.0 * free.abs(),
            "40%: bound not engaged ({:.5} vs {:.5} rot/s)", rps(bounded), rps(free));
}

/// Wedged with the pilot holding a pitch rate for 2.5 s, stick released,
/// then freed 0.5 s later: only the integral, wound to its bound, acts on
/// release — like a step torque of the bound's 25% offset — so the
/// excursion is no worse than that gust, smaller than with only the battery
/// cap, and it settles.
#[test]
fn collision_windup_is_recoverable() {
    let dt = plant_dt(&load());
    let (stuck, let_go, free) = (Time::new(0.5), Time::new(3.0), Time::new(3.5));
    let collision = |t: Time| Phase {
        stick: if (stuck..let_go).contains(&t) { RotPerSec::new(0.5) } else { RotPerSec::new(0.0) },
        blocked: (stuck..free).contains(&t),
        ..Phase::default()
    };
    let release = |gains: &GainStorage| {
        fly(gains, free + Time::new(3.0), collision).split_off(f64::from(free / dt) as usize)
    };
    let (after, after_cap) = (release(&load()), release(&cap_only()));
    let gust = fly(&load(), Time::new(3.5), torque_from(Time::new(0.5), -cg_torque(0.25)));
    println!("after release: peak {:.3} rot/s (cap only {:.3}); 25% gust {:.3}",
             rps(peak(&after)), rps(peak(&after_cap)), rps(peak(&gust)));
    assert!(peak(&after) <= 1.05 * peak(&gust),
            "release {:.3} vs gust {:.3} rot/s", rps(peak(&after)), rps(peak(&gust)));
    assert!(peak(&after) < 0.8 * peak(&after_cap),
            "bound doesn't help: {:.3} vs {:.3}", rps(peak(&after)), rps(peak(&after_cap)));
    let tail = &after[after.len() - f64::from(Time::new(0.5) / dt) as usize..];
    assert!(peak(tail) < Rate::new(0.02), "still {:.4} rot/s 2.5 s after release", rps(peak(tail)));
}
