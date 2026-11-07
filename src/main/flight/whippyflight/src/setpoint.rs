//! RX setpoint estimator for the reference feedforward.
//!
//! The feedforward needs a per-tick rate setpoint, but the stick is only
//! sampled at the RX packet rate.  Each axis runs a constant-velocity
//! Kalman filter on packet timestamps, at its steady-state (α-β) gains:
//!
//! ```text
//!   packet n at t_n:  x⁻ = x̂ + v̂·(t_n − t_{n−1}),   ε = r_n − x⁻
//!                     x̂ = x⁻ + α·ε,                  v̂ += (β/T)·ε
//!   tick at t:        r(t)     = x̂ + v̂·(t − t_n)
//!                     r_target = r(t)                (slew- and jerk-limited)
//! ```
//!
//! The target is not projected the feedforward horizon (h ticks) ahead, so
//! the airframe trails the stick by about h·dt.  Projecting with v̂ buys that
//! back on ramps, but v̂ lags the stick, so when a flick stops the projection
//! runs past it: +4% on small flicks, +34% on a 10 ms flick of 18 deg/s, and
//! the plant follows the target faithfully.  Holding x̂ between packets
//! instead (no extrapolation at all) removes the last of it but makes every
//! packet a goal jump for the jerk limit: 60% more voltage roughness on
//! continuous stick motion.
//!
//! α, β follow from the tracking index λ = σ_a·T²/σ_n (pilot stick
//! acceleration vs stick noise) by Kalata's closed form.  λ is configured
//! at a 500 Hz reference link and rescaled by (T / 2 ms)², so the same
//! setting holds across link rates.
//!
//! The target follows its goal with bounded slope (angular acceleration)
//! and bounded change of slope (jerk), braking so it stops on the goal
//! instead of past it.  Both bounds sit inside what the airframe can do:
//! the feedforward only inverts the next h ticks, so a target that needs
//! more acceleration — or a faster stop — than the motors have saturates
//! them, and the motors' spin-down lag then carries the airframe past it.

use crate::gains::GainStorage;
use crate::types::*;

use whippyunits::api::rescale_f32;
use whippyunits::{quantity, value};

/// λ at a 500 Hz link; see the closed-loop study in
/// `tests/eval_setpoint_reconstruction.rs`.
pub const DEFAULT_LAMBDA_500HZ: f32 = 8.0;
/// Target slew limit (an angular-acceleration bound), used when the gain
/// blob carries none.  `gen_wfg` sets the blob's from the airframe:
/// 0.83 × the weakest pitch/roll direction's peak angular acceleration
/// (opposing motors at full battery and idle, from hover), so it only clips
/// targets the airframe cannot follow.  This is that value for the legacy
/// 1S regression airframe (weakest 120 rot/s²).
pub const DEFAULT_TARGET_SLEW_RPS2: RotPerSecSq = RotPerSecSq::new(100.0);
/// Target jerk limit, used when the gain blob carries none.  The blob's is
/// 0.63 × the weakest pitch/roll jerk authority at hover (motor slew ×
/// thrust slope); the fraction was tuned with
/// `tests/eval_feedforward.rs` (1–4 rot/s flicks within 3% overshoot) on the
/// legacy airframe (8000 rot/s³), which this default is.
pub const DEFAULT_TARGET_JERK_RPS3: RotPerSecCu = RotPerSecCu::new(5000.0);

const REFERENCE_PERIOD_S: Seconds = Seconds::new(1.0 / 500.0);
/// Past this many RX periods without a packet the estimate stops
/// extrapolating, and the next packet reseeds it (link loss, not a drop).
const MAX_HOLD_PERIODS: f32 = 3.0;

pub struct SetpointEstimator {
    lambda_ref: f32,
    max_slew: RotPerSecSq,
    max_jerk: RotPerSecCu,
    period_s: Seconds,
    alpha: f32,
    beta: f32,
    x: [RotPerSec; 3],
    v: [RotPerSecSq; 3],
    packet_us: Micros,
    /// The last packet's rate, as received.
    raw: [RotPerSec; 3],
    seeded: bool,
    target: [RotPerSec; 3],
    target_valid: bool,
    /// d(target)/dt.
    target_v: [RotPerSecSq; 3],
}

impl SetpointEstimator {
    pub const fn new() -> Self {
        Self {
            lambda_ref: DEFAULT_LAMBDA_500HZ,
            max_slew: DEFAULT_TARGET_SLEW_RPS2,
            max_jerk: DEFAULT_TARGET_JERK_RPS3,
            period_s: Seconds::new(0.0),
            alpha: 1.0,
            beta: 0.0,
            x: [RotPerSec::new(0.0); 3],
            v: [RotPerSecSq::new(0.0); 3],
            packet_us: Micros::new(0),
            raw: [RotPerSec::new(0.0); 3],
            seeded: false,
            target: [RotPerSec::new(0.0); 3],
            target_valid: false,
            target_v: [RotPerSecSq::new(0.0); 3],
        }
    }

    /// λ at a 500 Hz link, and the target's slew and jerk limits.  A jerk
    /// limit ≤ 0 leaves only the slew limit.
    pub fn configure(&mut self, lambda_500hz: f32, max_target_slew: RotPerSecSq, max_target_jerk: RotPerSecCu) {
        self.lambda_ref = lambda_500hz;
        self.max_slew = max_target_slew;
        self.max_jerk = max_target_jerk;
        self.period_s = Seconds::new(0.0);
    }

    /// The default λ with the gain blob's target limits, falling back to
    /// the defaults for any the blob doesn't set.
    pub fn configure_for(&mut self, gains: &GainStorage) {
        let (slew, jerk) = target_limits(gains);
        self.configure(DEFAULT_LAMBDA_500HZ, slew, jerk);
    }

    /// A new RX packet: raw (unsmoothed) setpoint [roll, pitch, yaw], its
    /// arrival time, and the current RX period estimate.
    pub fn packet(&mut self, rate: [RotPerSec; 3], time: Micros, rx_period: Seconds) {
        self.raw = rate;
        if rx_period <= Seconds::new(0.0) {
            return;
        }
        let change = rx_period - self.period_s;
        if change > 0.01 * self.period_s || change < -0.01 * self.period_s {
            self.set_period(rx_period);
        }
        let gap = elapsed(self.packet_us, time);
        if !self.seeded || gap > MAX_HOLD_PERIODS * self.period_s {
            self.x = rate;
            self.v = [RotPerSecSq::new(0.0); 3];
            self.seeded = true;
        } else {
            for i in 0..3 {
                let predicted = self.x[i] + self.v[i] * gap;
                let innovation = rate[i] - predicted;
                self.x[i] = predicted + self.alpha * innovation;
                self.v[i] += self.beta / self.period_s * innovation;
            }
        }
        self.packet_us = time;
    }

    /// The last packet's rate setpoint, as received.
    pub fn raw(&self) -> [RotPerSec; 3] {
        self.raw
    }

    /// Per-tick setpoint r(t) and slew/jerk-limited feedforward target.
    pub fn tick(&mut self, now: Micros, dt: Seconds) -> ([RotPerSec; 3], [RotPerSec; 3]) {
        if !self.seeded {
            return Default::default();
        }
        let hold = MAX_HOLD_PERIODS * self.period_s;
        let since = elapsed(self.packet_us, now).min(hold);
        let mut rate = [RotPerSec::new(0.0); 3];
        let mut target = [RotPerSec::new(0.0); 3];
        let step = self.max_slew * dt;
        for i in 0..3 {
            rate[i] = self.x[i] + self.v[i] * since;
            let wanted = rate[i];
            target[i] = if !self.target_valid {
                wanted
            } else if self.max_jerk > RotPerSecCu::new(0.0) {
                // Head for `wanted` at the slope from which the jerk limit
                // still brakes to a stop on it; land on it once reached.
                // Every jump of the goal goes through this, small ones too:
                // the feedforward inverts the target over h ticks, so a
                // target whose slope steps turns into a voltage spike.
                let err = wanted - self.target[i];
                let brake = (2.0 * self.max_jerk * err.abs()).sqrt().clamp(RotPerSecSq::new(0.0), self.max_slew);
                let dv = self.max_jerk * dt;
                let toward = if err < RotPerSec::new(0.0) { -brake } else { brake };
                self.target_v[i] += (toward - self.target_v[i]).clamp(-dv, dv);
                let next = self.target[i] + self.target_v[i] * dt;
                if (next - wanted) * (self.target[i] - wanted) <= RotSqPerSecSq::new(0.0) {
                    self.target_v[i] = self.v[i].clamp(-self.target_v[i].abs(), self.target_v[i].abs());
                    wanted
                } else {
                    next
                }
            } else {
                self.target[i] + (wanted - self.target[i]).clamp(-step, step)
            };
        }
        self.target = target;
        self.target_valid = true;
        (rate, target)
    }

    fn set_period(&mut self, period: Seconds) {
        self.period_s = period;
        let ratio: f32 = (period / REFERENCE_PERIOD_S).into();
        let (alpha, beta) = kalata_gains(self.lambda_ref * ratio * ratio);
        self.alpha = alpha;
        self.beta = beta;
    }
}

/// Steady-state gains of a constant-velocity Kalman filter with white
/// acceleration noise, for tracking index λ.
pub fn kalata_gains(lambda: f32) -> (f32, f32) {
    let r = (4.0 + lambda - sqrt(8.0 * lambda + lambda * lambda)) / 4.0;
    let alpha = 1.0 - r * r;
    (alpha, 2.0 * (2.0 - alpha) - 4.0 * sqrt(1.0 - alpha))
}

/// The target `(slew, jerk)` limits a gain blob selects.
pub fn target_limits(gains: &GainStorage) -> (RotPerSecSq, RotPerSecCu) {
    let limits = gains.target_limits;
    (
        if limits.slew > RotPerSecSq::new(0.0) { limits.slew } else { DEFAULT_TARGET_SLEW_RPS2 },
        if limits.jerk > RotPerSecCu::new(0.0) { limits.jerk } else { DEFAULT_TARGET_JERK_RPS3 },
    )
}

/// Time between two readings of the wrapping microsecond clock.
fn elapsed(from: Micros, to: Micros) -> Seconds {
    let us = value!(to, us, u32).wrapping_sub(value!(from, us, u32));
    rescale_f32(quantity!(us as f32, us, f32))
}

pub(crate) fn sqrt(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { libm::sqrtf(x) }
}

impl Default for SetpointEstimator {
    fn default() -> Self {
        Self::new()
    }
}
