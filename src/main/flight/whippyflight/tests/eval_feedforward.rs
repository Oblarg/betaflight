//! Stick-flick tracking on the nonlinear plant, from gentle to saturating.
//! Not a regression test; run with
//!
//!   cargo test --release --target aarch64-apple-darwin --features std \
//!       --test eval_feedforward -- --ignored --nocapture
//!
//! Flies test_gains.wfg, or the blob at $WFG_PATH (zero its feedback gains
//! to see the feedforward + affine linearizer alone), through roll and pitch
//! flicks of 0.05–4 rot/s rising over 30 ms and 10 ms.  Each flick is flown
//! twice — with and without the stick — and the metrics are taken on the
//! difference.  `flick_summary` prints one SUMMARY line for
//! tools/sweep_qr.py.
//!
//! Overrides: FF_SLEW / FF_JERK (target slew rot/s² and jerk rot/s³ limits).

mod common;

use common::units::{Rate, Time};
use whippyflight_control::gains::{load_gains, GainStorage};
use whippyflight_control::sim::flick::{self, Limits, Metrics};
use whippyflight_control::types::*;
use whippyunits::value;

static TEST_WFG: &[u8] = include_bytes!("../tools/test_gains.wfg");

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().map(|v| v.parse().unwrap_or_else(|_| panic!("{name}")))
}

fn ms(t: Time) -> f64 {
    value!(t, ms)
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

fn all_flicks(gains: &GainStorage) -> Vec<(usize, Time, Rate, Metrics)> {
    let limits = Limits { slew: env_f32("FF_SLEW"), jerk: env_f32("FF_JERK") };
    flick::all_flicks(gains, common::plant::model(), limits)
        .into_iter()
        .map(|(axis, rise, amp, f)| (axis, rise, amp, f.metrics))
        .collect()
}

#[test]
#[ignore]
fn feedforward_tracking() {
    let gains = load();
    println!("h = {} ticks, vs nonlinear plant", gains.header.ff_horizon);
    let mut last = (usize::MAX, Time::new(0.0));
    for (axis, rise, amp, m) in all_flicks(&gains) {
        if (axis, rise) != last {
            println!("\n{} flick over {:.0} ms", ["roll", "pitch"][axis], ms(rise));
            println!("  {:>6} {:>8} {:>8} {:>8} {:>8} {:>10} {:>9} {:>10} {:>9} {:>6}",
                     "amp", "tgt ovs", "tgt lag", "gyro lag", "rise90", "overshoot", "hold err", "track rms", "residual", "sat");
            last = (axis, rise);
        }
        println!("  {:6.2} {:+7.1}% {:6.2}ms {:6.2}ms {:6.1}ms {:+9.1}% {:+8.1}% {:9.1}% {:8.1}% {:5.1}%",
                 value!(amp, rot / s), 100.0 * m.target_ovs, ms(m.target_lag), ms(m.gyro_lag),
                 ms(m.rise90), 100.0 * m.overshoot, 100.0 * m.hold_err,
                 100.0 * m.track_rms, 100.0 * m.residual, 100.0 * m.sat_frac);
    }
}

/// Worst cases over roll/pitch and both rise times: big flicks are ≥ 1 rot/s
/// (saturating), small ones below.  `track_small` is the 0.2 rot/s, 30 ms
/// flick's RMS tracking error.
#[test]
#[ignore]
fn flick_summary() {
    let gains = load();
    let (mut big_ovs, mut big_rise, mut small_ovs, mut track_small) =
        (f64::MIN, Time::new(f64::MIN), f64::MIN, f64::MIN);
    for (_, rise, amp, m) in all_flicks(&gains) {
        if amp >= Rate::new(1.0) {
            big_ovs = big_ovs.max(m.overshoot);
            big_rise = big_rise.max(m.rise90);
        } else {
            small_ovs = small_ovs.max(m.overshoot);
        }
        if amp == Rate::new(0.2) && rise == Time::new(0.030) {
            track_small = track_small.max(m.track_rms);
        }
    }
    println!("SUMMARY flick_big_overshoot={big_ovs:.3} flick_big_rise90_ms={:.1} \
              flick_small_overshoot={small_ovs:.3} flick_track_small={track_small:.3}", ms(big_rise));
}
