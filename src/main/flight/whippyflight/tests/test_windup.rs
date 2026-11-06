//! Integrator windup: ground/landing gating (no accumulation unless
//! airborne) and the integral voltage bound (enough for trim, small enough
//! that a collision's windup is recoverable), on test_gains.wfg and the
//! plant it was generated from.

mod common;

use common::plant::{model, Plant};
use whippyflight_control::controller::ControllerState;
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::types::*;

static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn load() -> GainStorage {
    let mut gains = GainStorage::new();
    assert_eq!(load_gains(common::static_blob(TEST_WFG), &mut gains), GainLoadResult::Ok);
    assert!(gains.hover_rps > 0.0, "blob has no hover speed");
    gains
}

/// Open-loop sensors: every motor at `speed` × the blob's hover speed.
fn sensors(gains: &GainStorage, speed: f32, gyro: [f32; 3], accel_g: [f32; 3]) -> SensorData {
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
    let [k_w, k_w2] = gains.motor_drag;
    let w = gains.thrust_balance.map(|b| b * gains.hover_rps);
    let net: f32 = (0..4)
        .map(|i| k_w * w[i] + k_w2 * w[i] * w[i] - (0..4).map(|j| gains.affine[i * 4 + j] * w[j] * w[j]).sum::<f32>())
        .sum();
    0.25 * net / model().battery_v()
}

/// A small roll-rate error the integral would chase (never saturating).
fn stick() -> SetpointData {
    SetpointData { rate_rps: [0.02, 0.0, 0.0], throttle: model().throttle_at(1.0) }
}

/// Run `seconds` of ticks; returns |integral| (roll) after each.
fn run(ctrl: &mut ControllerState, gains: &GainStorage, s: &SensorData, seconds: f32) -> Vec<f32> {
    let n = (seconds / gains.header.dt) as usize;
    (0..n)
        .map(|_| {
            ctrl.update(s, &stick(), gains);
            assert_eq!(ctrl.telemetry().command_scale, 1.0, "saturated: integral gated by that instead");
            ctrl.telemetry().error_integral[0].abs()
        })
        .collect()
}

const ONE_G: [f32; 3] = [0.0, 0.0, 1.0];
const NO_ACCEL: [f32; 3] = [0.0; 3];

fn growing(i: &[f32]) -> bool {
    i.last().unwrap() > &(i[0] + 1e-12)
}

/// Each tick only the leak: e_int ← (1 − dt/τ_int)·e_int.
fn only_leaking(gains: &GainStorage, i: &[f32]) -> bool {
    let leak = 1.0 - gains.header.dt / gains.header.integral_tau;
    i.windows(2).all(|w| (w[1] - w[0] * leak).abs() <= 1e-6 * w[0] + 1e-15)
}

#[test]
fn no_windup_before_takeoff() {
    let gains = load();
    let mut ctrl = ControllerState::new();
    // Armed on the ground, spinning at idle, stick held off center.
    for speed in [0.3, 0.6, 0.8] {
        let i = run(&mut ctrl, &gains, &sensors(&gains, speed, [0.0; 3], ONE_G), 2.0);
        assert!(i.iter().all(|&x| x == 0.0), "speed {speed}×hover: integral {}", i.last().unwrap());
    }
    // Takeoff: accumulates once the motors reach 0.85× hover.
    let i = run(&mut ctrl, &gains, &sensors(&gains, 0.9, [0.0; 3], NO_ACCEL), 0.1);
    assert!(growing(&i), "no accumulation after takeoff");
}

#[test]
fn landing_stops_accumulation() {
    let gains = load();
    for (accel, hold_s) in [(ONE_G, 0.5f32), (NO_ACCEL, 1.0)] {
        let mut ctrl = ControllerState::new();
        let flying = run(&mut ctrl, &gains, &sensors(&gains, 1.0, [0.0; 3], NO_ACCEL), 0.5);
        assert!(growing(&flying));
        // Touchdown: motors spun down, still, resting on the ground.
        let landed = sensors(&gains, 0.3, [0.0; 3], accel);
        let before = run(&mut ctrl, &gains, &landed, hold_s - 0.05);
        assert!(growing(&before), "{accel:?}: stopped before the {hold_s} s hold");
        let after = run(&mut ctrl, &gains, &landed, 1.0);
        assert!(only_leaking(&gains, &after[(0.1 / gains.header.dt) as usize..]),
                "{accel:?}: still accumulating after landing");
        // Takeoff again.
        let again = run(&mut ctrl, &gains, &sensors(&gains, 1.0, [0.0; 3], NO_ACCEL), 0.1);
        assert!(growing(&again), "{accel:?}: no accumulation after re-takeoff");
    }
}

#[test]
fn slow_motors_in_the_air_are_not_a_landing() {
    let gains = load();
    // Low thrust but falling (specific force ≪ 1 g), or tumbling: airborne.
    for (gyro, accel) in [([0.0; 3], [0.0, 0.0, 0.3]), ([0.0, 0.3, 0.0], ONE_G), ([0.0, 0.3, 0.0], NO_ACCEL)] {
        let mut ctrl = ControllerState::new();
        run(&mut ctrl, &gains, &sensors(&gains, 1.0, [0.0; 3], NO_ACCEL), 0.2);
        let i = run(&mut ctrl, &gains, &sensors(&gains, 0.3, gyro, accel), 2.0);
        assert!(!only_leaking(&gains, &i[i.len() - 100..]), "gyro {gyro:?} accel {accel:?}: counted as landed");
    }
}

// ── Closed loop ─────────────────────────────────────────────────────────

/// Pitch angular acceleration (BF axes, rot/s²) of the hover thrust acting
/// `offset` of the (unit) front-back arm off center.
fn cg_torque(offset: f64) -> f64 {
    let m = model();
    offset * m.pitch_prop_vel_sq * 4.0 * m.hover_rps.powi(2)
}

/// What acts on the hovering airframe at a time.
#[derive(Clone, Copy, Default)]
struct Phase {
    /// External pitch angular acceleration (BF, rot/s²).
    torque: f64,
    /// Pitch stick (rot/s).
    stick: f32,
    /// Rotation held at zero (wedged against something).
    blocked: bool,
}

/// Hover the plant through `phase(t)`; returns the BF pitch rate per tick.
fn fly(gains: &GainStorage, seconds: f64, phase: impl Fn(f64) -> Phase) -> Vec<f64> {
    let dt = gains.header.dt as f64;
    let mut ctrl = ControllerState::new();
    let mut plant = Plant::hovering(1.0);
    let mut out = Vec::new();
    for k in 0..(seconds / dt) as usize {
        let p = phase(k as f64 * dt);
        let s = SensorData {
            gyro_rps: plant.gyro_bf(),
            motor_rps: plant.omega.map(|w| w as f32),
            battery_voltage: model().battery_v(),
            accel_g: [0.0; 3],
        };
        let sp = SetpointData { rate_rps: [0.0, p.stick, 0.0], throttle: hover_throttle(gains) };
        let volts = ctrl.update(&s, &sp, gains).motor_duty.map(|d| (d * model().battery_v()) as f64);
        for _ in 0..4 {
            plant.rk4_with_torque(&volts, dt / 4.0, [0.0, p.torque, 0.0]);
        }
        if p.blocked {
            plant.rate = [0.0; 3];
        }
        out.push(plant.gyro_bf()[1] as f64);
    }
    out
}

fn torque_from(start: f64, torque: f64) -> impl Fn(f64) -> Phase {
    move |t| Phase { torque: if t >= start { torque } else { 0.0 }, ..Phase::default() }
}

fn peak(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |m, r| m.max(r.abs()))
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Residual pitch rate over the last 0.5 s of 10 s with a CG offset applied
/// at 0.5 s (the trim settles in a few seconds).
fn trim_residual(gains: &GainStorage, offset: f64) -> f64 {
    let rate = fly(gains, 10.0, torque_from(0.5, cg_torque(offset)));
    mean(&rate[rate.len() - (0.5 / gains.header.dt as f64) as usize..])
}

/// The blob without its integral bound: only the 0.2·Vbat cap.
fn cap_only() -> GainStorage {
    let mut gains = load();
    gains.integral_limit_v = 0.0;
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
                "{offset}: residual {bounded:.5} rot/s, {free:.5} with only the cap");
    }
    let (bounded, free) = (trim_residual(&gains, 0.4), trim_residual(&cap, 0.4));
    assert!(bounded.abs() > 3.0 * free.abs(), "40%: bound not engaged ({bounded:.5} vs {free:.5} rot/s)");
}

/// Wedged with the pilot holding a pitch rate for 2.5 s, stick released,
/// then freed 0.5 s later: only the integral, wound to its bound, acts on
/// release — like a step torque of the bound's 25% offset — so the
/// excursion is no worse than that gust, smaller than with only the battery
/// cap, and it settles.
#[test]
fn collision_windup_is_recoverable() {
    let dt = load().header.dt as f64;
    let (stuck, let_go, free) = (0.5, 3.0, 3.5);
    let collision = |t: f64| Phase { stick: if (stuck..let_go).contains(&t) { 0.5 } else { 0.0 },
                                     blocked: (stuck..free).contains(&t), ..Phase::default() };
    let release = |gains: &GainStorage| fly(gains, free + 3.0, collision).split_off((free / dt) as usize);
    let (after, after_cap) = (release(&load()), release(&cap_only()));
    let gust = fly(&load(), 3.5, torque_from(0.5, -cg_torque(0.25)));
    println!("after release: peak {:.3} rot/s (cap only {:.3}); 25% gust {:.3}",
             peak(&after), peak(&after_cap), peak(&gust));
    assert!(peak(&after) <= 1.05 * peak(&gust), "release {:.3} vs gust {:.3} rot/s", peak(&after), peak(&gust));
    assert!(peak(&after) < 0.8 * peak(&after_cap), "bound doesn't help: {:.3} vs {:.3}", peak(&after), peak(&after_cap));
    let tail = &after[after.len() - (0.5 / dt) as usize..];
    assert!(peak(tail) < 0.02, "still {:.4} rot/s 2.5 s after release", peak(tail));
}
