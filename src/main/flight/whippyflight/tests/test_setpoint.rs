//! RX setpoint estimator (α-β tracker + slew/jerk-limited feedforward target).

use whippyflight_control::setpoint::{kalata_gains, SetpointEstimator};

const PERIOD_S: f32 = 0.002; // 500 Hz link
const DT: f32 = 0.00025; // 4 kHz loop

fn approx(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

/// Feed `packets` of `stick(t)` on roll and tick through them; returns the
/// last (rate, target) and the largest per-tick target change.
fn run(est: &mut SetpointEstimator, stick: impl Fn(f32) -> f32, packets: u32) -> ([f32; 3], [f32; 3], f32) {
    let ticks_per_packet = (PERIOD_S / DT).round() as u32;
    let mut last = ([0.0; 3], [0.0; 3]);
    let mut max_step = 0.0f32;
    let mut prev_target: Option<f32> = None;
    for n in 0..packets {
        let t_packet = n as f32 * PERIOD_S;
        est.packet([stick(t_packet), 0.0, 0.0], (t_packet * 1e6) as u32, PERIOD_S);
        for k in 0..ticks_per_packet {
            let t = t_packet + k as f32 * DT;
            last = est.tick((t * 1e6) as u32, DT);
            if let Some(p) = prev_target {
                max_step = max_step.max((last.1[0] - p).abs());
            }
            prev_target = Some(last.1[0]);
        }
    }
    (last.0, last.1, max_step)
}

#[test]
fn kalata_gains_match_closed_form() {
    let (alpha, beta) = kalata_gains(8.0);
    assert!(approx(alpha, 0.9706, 1e-3), "alpha={alpha}");
    assert!(approx(beta, 1.3726, 1e-3), "beta={beta}");
    let (alpha, beta) = kalata_gains(0.5);
    assert!(alpha > 0.0 && alpha < 1.0 && beta > 0.0 && beta < alpha, "{alpha} {beta}");
}

#[test]
fn tracks_a_ramp_without_projecting_it() {
    let mut est = SetpointEstimator::new();
    // The jerk limit trails a ramp by slope²/(2J); this checks the estimate.
    est.configure(8.0, 1e9, 0.0);
    let slope = 10.0; // rot/s²
    let (rate, target, _) = run(&mut est, |t| slope * t, 200);
    let t_last = 199.0 * PERIOD_S + (PERIOD_S / DT - 1.0) * DT;
    assert!(approx(rate[0], slope * t_last, 1e-3), "rate={}", rate[0]);
    assert!(approx(target[0], rate[0], 1e-5), "target={}", target[0]);
}

/// A flick that stops: the target barely passes where the stick stopped.
/// What's left is v̂ extrapolating within the last packet; projecting the
/// horizon ahead made the 18 deg/s, 10 ms flick's +35%.
#[test]
fn flick_target_does_not_overshoot() {
    let (slew, jerk) = (127.0, 2315.0);
    for (amp, rise, bound) in [(0.05f32, 0.010f32, 1.06f32), (0.2, 0.030, 1.015), (1.0, 0.010, 1.015)] {
        let mut est = SetpointEstimator::new();
        est.configure(8.0, slew, jerk);
        let s_curve = |t: f32| amp * 0.5 * (1.0 - (core::f32::consts::PI * (t / rise).clamp(0.0, 1.0)).cos());
        let (mut peak, ticks_per_packet) = (0.0f32, (PERIOD_S / DT).round() as u32);
        for n in 0..200u32 {
            let t_packet = n as f32 * PERIOD_S;
            est.packet([s_curve(t_packet), 0.0, 0.0], (t_packet * 1e6) as u32, PERIOD_S);
            for k in 0..ticks_per_packet {
                let (_, target) = est.tick(((t_packet + k as f32 * DT) * 1e6) as u32, DT);
                peak = peak.max(target[0]);
            }
        }
        assert!(peak <= bound * amp, "{amp} rot/s over {rise} s: target peak {}", peak / amp);
    }
}

#[test]
fn target_slew_is_limited() {
    let mut est = SetpointEstimator::new();
    est.configure(8.0, 100.0, 0.0);
    run(&mut est, |_| 0.0, 10);
    let (_, target, max_step) = run(&mut est, |_| 1.0, 1);
    assert!(max_step <= 100.0 * DT + 1e-6, "max_step={max_step}");
    assert!(target[0] < 1.0);
}

/// A stick step: in transit the target's slope changes by at most J·dt per
/// tick, and it stops on the new value without passing it (landing snaps it
/// onto the goal, so the bound is checked away from it).
#[test]
fn target_jerk_is_limited_and_lands_without_overshoot() {
    let (slew, jerk) = (100.0, 5000.0);
    let mut est = SetpointEstimator::new();
    est.configure(8.0, slew, jerk);
    run(&mut est, |_| 0.0, 10);
    let ticks_per_packet = (PERIOD_S / DT).round() as u32;
    let (mut prev, mut prev_slope, mut max_dslope, mut peak) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for n in 10..200u32 {
        let t_packet = n as f32 * PERIOD_S;
        est.packet([1.0, 0.0, 0.0], (t_packet * 1e6) as u32, PERIOD_S);
        for k in 0..ticks_per_packet {
            let (rate, target) = est.tick(((t_packet + k as f32 * DT) * 1e6) as u32, DT);
            // Only while the goal (the estimate) has settled onto the step;
            // before that it overshoots by itself.
            if (rate[0] - 1.0).abs() < 1e-3 {
                peak = peak.max(target[0]);
            }
            let slope = (target[0] - prev) / DT;
            if (target[0] - rate[0]).abs() > 1e-3 {
                max_dslope = max_dslope.max((slope - prev_slope).abs());
            }
            assert!(slope.abs() <= slew + 1e-2, "slope={slope}");
            (prev, prev_slope) = (target[0], slope);
        }
    }
    assert!(max_dslope <= jerk * DT * 1.01 + 1e-2, "max slope change {max_dslope}");
    assert!(peak <= 1.0 + 1e-4, "peak={peak}");
    assert!((prev - 1.0).abs() < 1e-4, "settles on the step: {prev}");
}

#[test]
fn holds_without_extrapolating_after_link_loss() {
    let mut est = SetpointEstimator::new();
    est.configure(8.0, 1e9, 0.0);
    run(&mut est, |t| 10.0 * t, 50);
    let last_packet_us = (49.0 * PERIOD_S * 1e6) as u32;
    let (rate_a, target_a) = est.tick(last_packet_us + 10_000, DT);
    let (rate_b, target_b) = est.tick(last_packet_us + 50_000, DT);
    assert_eq!(rate_a, rate_b);
    assert_eq!(target_a, rate_a);
    assert_eq!(target_b, rate_b);
}

#[test]
fn reseeds_after_a_long_gap() {
    let mut est = SetpointEstimator::new();
    run(&mut est, |t| 10.0 * t, 50);
    // Next packet 100 ms later at a very different value: no velocity kick.
    est.packet([-0.5, 0.0, 0.0], 200_000, PERIOD_S);
    let (rate, _) = est.tick(200_000, DT);
    assert_eq!(rate[0], -0.5);
}

#[test]
fn lambda_is_rescaled_with_link_period() {
    // At 250 Hz, λ_500 = 8 must act as λ = 32: one step of size 1 should be
    // absorbed with α(32).
    let mut est = SetpointEstimator::new();
    est.configure(8.0, 1e9, 0.0);
    let period = 0.004;
    est.packet([0.0; 3], 0, period);
    est.packet([0.0; 3], 4_000, period);
    est.packet([1.0, 0.0, 0.0], 8_000, period);
    let (rate, _) = est.tick(8_000, DT);
    let (alpha, _) = kalata_gains(32.0);
    assert!(approx(rate[0], alpha, 1e-5), "rate={} alpha={alpha}", rate[0]);
}

/// A slow ramp of 1 µs-quantized stick (SITL's 1500 → 1800 over 0.8 s,
/// 216 deg/s at full), at the derived airframes' target limits: the target
/// stays within 1 deg/s of the estimate on average.  The estimate follows
/// the 1 µs staircase (±0.6 deg/s per packet), which the jerk limit can't
/// follow, and the nonlinear limiter loses some of the mean on it.  Not
/// jerk-limiting small goal jumps turned every RX frame of a flick into a
/// slope step the feedforward inverts into a voltage spike (saturation,
/// overshoot).
#[test]
fn quantized_ramp_is_followed() {
    let per_us = 216.0 / 300.0 / 360.0; // rot/s per µs of stick
    for (slew, jerk) in [(127.0, 2315.0), (154.0, 3853.0), (100.0, 5000.0)] {
        let mut est = SetpointEstimator::new();
        est.configure(8.0, slew, jerk);
        let (mut sum, mut n) = (0.0f32, 0);
        let ticks_per_packet = (PERIOD_S / DT).round() as u32;
        for k in 0..(0.8 / DT) as u32 {
            let t = k as f32 * DT;
            if k % ticks_per_packet == 0 {
                est.packet([(300.0 * t / 0.8).round() * per_us, 0.0, 0.0], (t * 1e6) as u32, PERIOD_S);
            }
            let (rate, target) = est.tick((t * 1e6) as u32, DT);
            if t > 0.1 {
                sum += target[0] - rate[0];
                n += 1;
            }
        }
        let shortfall_dps = -sum / n as f32 * 360.0;
        assert!(shortfall_dps.abs() <= 1.0, "slew {slew} jerk {jerk}: target trails by {shortfall_dps} deg/s");
    }
}
