//! Motor allocation: fitting the controller's voltage command into the
//! actuator range.
//!
//! The pilot's collective lives on a fixed thrust-balanced direction n: the
//! per-motor hover voltages that put every arm's torque in balance at that
//! collective (see [`balanced_direction`]).  Every adjustment made here
//! moves along n too:
//!
//! ```text
//!   v = s·u + c·n,     v_lo ≤ vᵢ ≤ v_hi
//! ```
//!
//! `u` is the controller's per-motor command and `c` the collective.  `c`
//! tracks the pilot's throttle but is pushed up when a motor would drop
//! below `v_lo` (idle) and down when one would exceed `v_hi` (headroom).
//! Only when no `c` fits does the command shrink, `s < 1`, keeping its
//! direction.
//!
//! `v_lo` is the voltage that holds a motor at the controller's
//! linearization floor in steady state, so the airframe idles where the
//! gains were solved instead of somewhere they overstate authority.  Motors
//! may still be slower transiently (spin-up, after crashflip); nothing here
//! regulates speed.
//!
//! n depends only on the airframe and the collective, never on measured
//! motor speeds: a direction that moved with them would turn the ~2 V of
//! collective into feedback on the speed differential the attitude
//! command just created.  The interval logic below handles either sign of
//! nᵢ; if no `(s, c)` exists at all, each motor is clipped to the range.

use crate::setpoint::sqrt;

/// Floor on |nᵢ| so the per-motor interval stays finite.
const MIN_DIRECTION: f32 = 1e-3;

/// Result of one allocation.
#[derive(Clone, Copy, Debug)]
pub struct Allocation {
    /// Per-motor voltage, within `[v_lo, v_hi]`.
    pub voltage: [f32; 4],
    /// Collective `c` along the balanced direction (V, mean over motors).
    pub collective_v: f32,
    /// Command scale `s` ∈ [0, 1]; 1 unless the command didn't fit.
    pub command_scale: f32,
    /// Whether the realized attitude command differs from the request
    /// (`s < 1` or per-motor clipping).
    pub saturated: bool,
}

/// Thrust-balanced collective direction at mean collective `collective_v`.
///
/// `balance` holds each motor's hover speed relative to the others: speeds
/// ωᵢ = bᵢ·ω̄ put the lever-arm-weighted thrusts in balance (gen_wfg's
/// 1/√(d_fb·d_lr)).  The voltage that holds them in steady state, net of
/// the affine linearizer's own G·ω|ω| (which the command `u` carries), is
///
/// ```text
///   Vᵢ(ω̄) = k_ω·bᵢ·ω̄ + qᵢ·ω̄²,    qᵢ = k_ω²·bᵢ² − Σⱼ Gᵢⱼ·bⱼ²
/// ```
///
/// ω̄ is solved so that mean(V) = `collective_v`, and n = V / mean(V).
/// Without motor drag (pre-v4 blobs) n falls back to n ∝ b, duty ∝ speed.
pub fn balanced_direction(
    balance: [f32; 4],
    motor_drag: [f32; 2],
    affine: &[f32; 16],
    collective_v: f32,
) -> [f32; 4] {
    let [k_w, k_w2] = motor_drag;
    let b2 = balance.map(|b| b * b);
    let q = [0, 1, 2, 3].map(|i| {
        k_w2 * b2[i] - (0..4).map(|j| affine[i * 4 + j] * b2[j]).sum::<f32>()
    });
    let mean = |x: [f32; 4]| 0.25 * (x[0] + x[1] + x[2] + x[3]);
    let (a, lin, c) = (mean(q), k_w * mean(balance), max(collective_v, 0.0));
    let w_bar = if a > 0.0 {
        2.0 * c / (lin + sqrt(lin * lin + 4.0 * a * c))
    } else if lin > 0.0 {
        c / lin
    } else {
        0.0
    };
    let v = [0, 1, 2, 3].map(|i| k_w * balance[i] + q[i] * w_bar);
    let (v, total) = if mean(v) > 0.0 { (v, mean(v)) } else { (balance, mean(balance)) };
    if !(total > 0.0) {
        return [1.0; 4];
    }
    v.map(|x| {
        let x = x / total;
        if abs(x) < MIN_DIRECTION { MIN_DIRECTION } else { x }
    })
}

/// Fit `command` into `[v_lo, v_hi]` per motor (see module docs).
pub fn allocate(
    command: [f32; 4],
    neutral: [f32; 4],
    pilot_collective: f32,
    v_lo: f32,
    v_hi: f32,
) -> Allocation {
    // Motor i admits c ∈ [lᵢ − s·pᵢ, hᵢ − s·pᵢ].
    let p = [0, 1, 2, 3].map(|i| command[i] / neutral[i]);
    let bounds = [0, 1, 2, 3].map(|i| {
        let (a, b) = (v_lo / neutral[i], v_hi / neutral[i]);
        if a < b { (a, b) } else { (b, a) }
    });

    // Feasible iff every lower bound is under every upper bound:
    // (lᵢ − hⱼ) + s·(pⱼ − pᵢ) ≤ 0 for all i, j.  Take the largest s.
    let mut s_max = 1.0f32;
    let mut s_min = 0.0f32;
    let mut feasible = true;
    for i in 0..4 {
        for j in 0..4 {
            let alpha = bounds[i].0 - bounds[j].1;
            let beta = p[j] - p[i];
            if beta > 0.0 {
                s_max = min(s_max, -alpha / beta);
            } else if beta < 0.0 {
                s_min = max(s_min, -alpha / beta);
            } else if alpha > 0.0 {
                feasible = false;
            }
        }
    }

    if !feasible || s_min > s_max {
        let voltage = [0, 1, 2, 3].map(|i| {
            clamp(command[i] + pilot_collective * neutral[i], v_lo, v_hi)
        });
        return Allocation {
            voltage,
            collective_v: pilot_collective,
            command_scale: 1.0,
            saturated: true,
        };
    }

    let s = s_max;
    let lower = (0..4).fold(f32::NEG_INFINITY, |acc, i| max(acc, bounds[i].0 - s * p[i]));
    let upper = (0..4).fold(f32::INFINITY, |acc, i| min(acc, bounds[i].1 - s * p[i]));
    let c = if lower <= upper {
        clamp(pilot_collective, lower, upper)
    } else {
        0.5 * (lower + upper)
    };

    Allocation {
        voltage: [0, 1, 2, 3].map(|i| clamp(s * command[i] + c * neutral[i], v_lo, v_hi)),
        collective_v: c,
        command_scale: s,
        saturated: s < 1.0 - 1e-6,
    }
}

fn abs(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    max(lo, min(x, hi))
}
