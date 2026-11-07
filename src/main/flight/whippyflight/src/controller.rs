//! The core LQR attitude-rate controller.
//!
//! Every matrix-vector operation flows through whippyalgebra's
//! [`MixedUnitMatrix`], ensuring that the dimensional algebra is correct at
//! compile time:
//!
//! - `Kp (V / rot·s⁻¹) × e_prop (rot/s)  →  voltage (V)` ✓
//! - `Ki (V / rot)      × e_int  (rot)    →  voltage (V)` ✓
//! - `Kd (V / rot·s⁻²) × e_deriv(rot/s²) →  voltage (V)` ✓
//! - `G_h⁺(V / rot·s⁻¹) × e_ff  (rot/s)  →  voltage (V)` ✓
//! - `G  (V / rot²·s⁻²) × ω|ω|  (rot²/s²) →  voltage (V)` ✓
//!
//! Passing the wrong error vector to the wrong gain matrix is a compile error.
//!
//! # Feedforward
//!
//! With a v3 blob the feedforward runs an internal reference model in
//! deviation coordinates, δx = [r_ref; δω_ref], relinearized every tick at
//! the measured (floored) motor speeds.  It holds u constant over an h-tick
//! horizon (ZOH) and inverts the attitude response over that horizon:
//!
//! ```text
//!   r_target = r + h·Δr                       (extrapolate the smoothed setpoint)
//!   u_ff     = G_h⁺ · (r_target − F_h·δx)     (F_h·δx: h-tick free response)
//!   δx      ← Ad·δx + Bd·ū_ff                  (one-tick ZOH propagation)
//! ```
//!
//! ū_ff is the part of u_ff the allocator actually delivered, so under
//! saturation the model follows what the motors flew rather than what was
//! asked.  The feedback regulates the measured rate onto r_ref, the model's
//! own rate, not onto the setpoint.
//!
//! The reference model sees no measurements, so gyro/eRPM noise never
//! passes through G_h⁺; measured errors are left to the LQR feedback.
//! Older blobs fall back to the one-tick `u_ff = B⁻¹·Δr`.
//!
//! # Airframe drag
//!
//! With the `airframe-drag` feature and a v7 blob that carries a drag model,
//! the rotor-drag torque of a tall frame, α_d = C·ā (ā: low-passed body x/y
//! accelerometer), enters the reference model as a known disturbance held
//! over the horizon.  The rate block of Ad is diagonal and rates don't drive
//! motor speeds, so its h-tick response is Σ_{j<h} ad_rr^j ⊙ α_d·dt:
//!
//! ```text
//!   u_ff  = G_h⁺ · (r_target − F_h·δx − Σ_{j<h} ad_rr^j ⊙ α_d·dt)
//!   r_ref ← (one-tick propagation) + α_d·dt
//! ```
//!
//! so the feedforward cancels the torque and r_ref stays on the setpoint.

use crate::allocation::{allocate, balanced_direction};
use crate::gains::{GainStorage, ReferenceModel};
use crate::interpolation::interpolate_gains;
use crate::types::*;

use whippyalgebra::nalgebra::zeros;
use whippyunits::quantity;

/// Internal controller state.
pub struct ControllerState {
    /// Leaky integral error: [roll, pitch, yaw].
    integral_error: IntegralErrorVec,
    /// Filtered derivative error: [roll, pitch, yaw].
    derivative_error: DerivErrorVec,
    /// Previous proportional error (for derivative).
    prev_prop_error: PropErrorVec,
    /// Previous setpoint (for feedforward diff).
    prev_setpoint: PropErrorVec,
    /// Minimum linearization speed (model units); see [`Self::set_min_rps`].
    min_rps: RotPerSec,
    /// Feedforward reference-model rate state r_ref.
    ref_rate: PropErrorVec,
    /// Feedforward reference-model motor-speed deviation δω_ref.
    ref_speed: MotorSpeedVec,
    /// Whether the reference model has been seeded since the last reset.
    ref_seeded: bool,
    /// Whether the previous command was scaled or clipped by allocation.
    saturated: bool,
    /// Flying, as opposed to on the ground (armed, before takeoff or after
    /// landing); see [`Self::update_flight_state`].
    airborne: bool,
    /// How long the landing condition has held.
    landing_s: Seconds,
    /// Low-passed body x/y specific force for the airframe drag model;
    /// `None` until the first sample.
    #[cfg(feature = "airframe-drag")]
    accel_lp: Option<[MetersPerSecSq; 2]>,
    /// Latest telemetry.
    telemetry: LqrTelemetry,
}

impl ControllerState {
    pub fn new() -> Self {
        Self {
            integral_error: zeros![IntegralAxes, Scalar as f32],
            derivative_error: zeros![DerivativeAxes, Scalar as f32],
            prev_prop_error: zeros![AttitudeAxes, Scalar as f32],
            prev_setpoint: zeros![AttitudeAxes, Scalar as f32],
            min_rps: RotPerSec::new(0.0),
            ref_rate: zeros![AttitudeAxes, Scalar as f32],
            ref_speed: zeros![MotorSpeedAxes, Scalar as f32],
            ref_seeded: false,
            saturated: false,
            airborne: false,
            landing_s: Seconds::new(0.0),
            #[cfg(feature = "airframe-drag")]
            accel_lp: None,
            telemetry: LqrTelemetry {
                feedback_v: Default::default(),
                feedforward_v: Default::default(),
                affine_v: Default::default(),
                error_prop: Default::default(),
                error_integral: Default::default(),
                error_derivative: Default::default(),
                kp_motor0: Default::default(),
                collective_v: Volt::new(0.0),
                command_scale: 1.0,
                setpoint_rps: Default::default(),
                ff_target_rps: Default::default(),
                status: 0,
            },
        }
    }

    pub fn reset(&mut self) {
        self.integral_error = zeros![IntegralAxes, Scalar as f32];
        self.derivative_error = zeros![DerivativeAxes, Scalar as f32];
        self.prev_prop_error = zeros![AttitudeAxes, Scalar as f32];
        self.prev_setpoint = zeros![AttitudeAxes, Scalar as f32];
        self.ref_rate = zeros![AttitudeAxes, Scalar as f32];
        self.ref_speed = zeros![MotorSpeedAxes, Scalar as f32];
        self.ref_seeded = false;
        self.saturated = false;
        self.airborne = false;
        self.landing_s = Seconds::new(0.0);
        #[cfg(feature = "airframe-drag")]
        {
            self.accel_lp = None;
        }
    }

    /// Set the minimum linearization speed, in model units (eRPM/600).
    ///
    /// Gains grow without bound as any pitch/roll motor's speed → 0 (thrust
    /// authority ∝ ω), so reported speeds are clamped up to
    /// `max(min_rps, grid min_rps)`.  The same floor sets the idle voltage
    /// the allocator keeps every motor above, so the airframe idles where
    /// the gains were solved.
    pub fn set_min_rps(&mut self, min_rps: RotPerSec) {
        self.min_rps = min_rps;
    }

    pub fn telemetry(&self) -> &LqrTelemetry {
        &self.telemetry
    }

    /// Run one control iteration.
    ///
    /// The matrix multiplications here are the core of the control law.  Each
    /// one is unit-checked at compile time:
    ///
    /// ```text
    ///   Kp [V / (rot/s)]  ×  e_prop [rot/s]   =  [V]
    ///   Ki [V / rot]       ×  e_int  [rot]      =  [V]
    ///   Kd [V / (rot/s²)] ×  e_deriv [rot/s²]  =  [V]
    ///   G_h⁺[V / (rot/s)] ×  e_ff    [rot/s]   =  [V]
    /// ```
    ///
    /// If any of these pairings were wrong (e.g. passing `e_prop` to `Ki`),
    /// the code would fail to compile because `dims![rot/s]` ≠ `dims![rot]`.
    pub fn update(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        gains: &GainStorage,
    ) -> ControlOutput {
        self.step(sensors, setpoint, None, gains)
    }

    /// [`Self::update`] with the feedforward's h-tick-ahead target supplied
    /// by the caller ([roll, pitch, yaw]) instead of extrapolated as
    /// r + h·Δr.  Only used with a reference-model blob.
    pub fn update_with_target(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        ff_target: [RotPerSec; 3],
        gains: &GainStorage,
    ) -> ControlOutput {
        self.step(sensors, setpoint, Some(ff_target), gains)
    }

    fn step(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        ff_target: Option<[RotPerSec; 3]>,
        gains: &GainStorage,
    ) -> ControlOutput {
        if !gains.loaded {
            self.telemetry.status = 0;
            return ControlOutput { motor_duty: [0.0; 4] };
        }

        let dt = gains.header.dt;
        let alpha_int: f32 = (dt / gains.header.integral_tau).into();
        let alpha_deriv: f32 = (dt / gains.header.derivative_tau).into();

        let measurement: PropErrorVec = from_array(sensors.gyro_rps);
        let setpoint_vec: SetpointDiffVec = from_array(setpoint.rate_rps);

        // Seed the reference model at the current setpoint so the first tick
        // after a reset produces no feedforward transient.
        if gains.has_ref_model && !self.ref_seeded {
            self.ref_rate = setpoint_vec;
            self.ref_speed = zeros![MotorSpeedAxes, Scalar as f32];
            self.prev_setpoint = setpoint_vec;
            self.ref_seeded = true;
        }

        // Proportional error: e = r - y  (both in rot/s → result in rot/s).
        // With a reference model, r is the rate the model predicts the
        // feedforward has produced by now, so the feedback corrects only
        // departures from that trajectory instead of pushing toward the same
        // goal (which, while the command saturates, adds up to overshoot).
        let prop_error: PropErrorVec = if gains.has_ref_model {
            self.ref_rate - measurement
        } else {
            setpoint_vec - measurement
        };

        // Setpoint diff for feedforward: Δr = r[k] - r[k-1]  (rot/s)
        let setpoint_diff: SetpointDiffVec = setpoint_vec - self.prev_setpoint;

        // Leaky integrator: e_int[k] = (1-α)·e_int[k-1] + α·dt·e[k]  (rot)
        //
        // Must match gen_wfg.py's augmented model (A_aug rows 0–2), which Ki
        // was solved against.  The clamp bounds the low-passed error e_int/dt
        // at INTEGRAL_ERROR_RATE_LIMIT.  While the last command was
        // saturated, or on the ground, the error isn't correctable, so the
        // state only leaks (anti-windup).
        self.update_flight_state(sensors, gains, dt);
        let int_gain = if self.saturated || !self.airborne { Seconds::new(0.0) } else { dt * alpha_int };
        let int_raw: IntegralErrorVec =
            self.integral_error.scale(1.0 - alpha_int) + prop_error * int_gain;
        let int_limit = INTEGRAL_ERROR_RATE_LIMIT * dt;
        self.integral_error =
            from_array(to_array(&int_raw).map(|x| x.clamp(-int_limit, int_limit)));

        // Derivative filter: ė[k] = (1-α_d)·ė[k-1] + α_d·(e[k]-e[k-1])/dt
        self.derivative_error = self.derivative_error.scale(1.0 - alpha_deriv)
            + (prop_error - self.prev_prop_error) * (alpha_deriv / dt);

        // --- Gain interpolation (returns typed matrices) ---
        let header = &gains.header;
        let floor = self.min_rps.max(header.min_rps);
        let motor_rps = sensors.motor_rps.map(|w| w.clamp(floor, header.max_rps));
        let interp = interpolate_gains(&motor_rps, gains);

        // --- Feedback law: u_fb = Kp·e + Ki·e_int + Kd·ė ---
        //
        // Each of these multiplications is compile-time unit-checked:
        //   KpMatrix [MotorAxes × AttitudeAxes]   ×  PropErrorVec [AttitudeAxes × Scalar]    → MotorVoltageVec
        //   KiMatrix [MotorAxes × IntegralAxes]   ×  IntegralErrorVec [IntegralAxes × Scalar] → MotorVoltageVec
        //   KdMatrix [MotorAxes × DerivativeAxes] ×  DerivErrorVec [DerivativeAxes × Scalar]  → MotorVoltageVec
        let battery_v = sensors.battery_voltage.max(MIN_BATTERY_V);

        // Bound the integral's voltage on every motor by scaling its state:
        // enough to trim a generous imbalance (see TRIM_CG_ARM_FRACTION),
        // little enough that windup from a collision stays recoverable.
        let mut u_i: MotorVoltageVec = interp.ki * self.integral_error;
        let u_i_max = u_i.as_uniform().amax();
        let u_i_limit = integral_voltage_limit(gains, battery_v);
        if u_i_max > u_i_limit {
            let s: f32 = (u_i_limit / u_i_max).into();
            self.integral_error = self.integral_error.scale(s);
            u_i = u_i.scale(s);
        }

        let u_p: MotorVoltageVec = interp.kp * prop_error;
        let u_d: MotorVoltageVec = interp.kd * self.derivative_error;
        let feedback: MotorVoltageVec = u_p + u_i + u_d;

        // --- Feedforward ---
        // r_target = r + h·Δr unless the caller predicted it.
        let target: PropErrorVec = match ff_target {
            Some(t) => from_array(t),
            None => setpoint_vec + setpoint_diff.scale(gains.header.ff_horizon as f32),
        };
        let drag = if gains.has_ref_model { self.airframe_drag(sensors, gains, dt) } else { None };
        let feedforward: MotorVoltageVec = if gains.has_ref_model {
            self.reference_feedforward(&interp.ref_model, &interp.binv, target, drag,
                                       gains.header.ff_horizon, battery_v)
        } else {
            //   BInvMatrix [MotorAxes × AttitudeAxes] × SetpointDiffVec → MotorVoltageVec
            interp.binv * setpoint_diff
        };

        // --- Affine linearizer: u_aff = −B⁺·C(ω̂) = G · ω̂|ω̂| ---
        //   AffineMatrix [MotorAxes × MotorSpeedSqAxes] × MotorSpeedSqVec → MotorVoltageVec
        // ω̂ is the same clamped operating point the gains were interpolated at.
        let speed_sq: MotorSpeedSqVec = from_array(motor_rps.map(|w| w * w.abs()));
        let affine: MotorVoltageVec = gains.affine_matrix() * speed_sq;

        // --- Total control voltage ---
        let total: MotorVoltageVec = feedback + feedforward + affine;

        // --- Allocation: fit the command into [v_idle, Vbat] along the
        // thrust-balanced direction (see `allocation`), then voltage → duty.
        let drag_fit = gains.motor_drag;
        let idle_v = (drag_fit.k_w * floor + drag_fit.k_w2 * floor * floor)
            .clamp(Volt::new(0.0), 0.5 * battery_v);
        let command = to_array(&total);
        let pilot_collective = setpoint.throttle.clamp(0.0, 1.0) * battery_v;
        let direction = balanced_direction(gains.thrust_balance, gains.motor_drag, &gains.affine,
                                           pilot_collective.max(idle_v));
        let alloc = allocate(command, direction, pilot_collective, idle_v, battery_v);
        let output = ControlOutput {
            motor_duty: alloc.voltage.map(|v| f32::from(v / battery_v).clamp(0.0, 1.0)),
        };

        self.saturated = alloc.saturated;

        // Propagate the reference model with the feedforward the motors
        // actually got: each motor's share of its command that survived
        // scaling/clipping (the collective c·n is not part of the command).
        if gains.has_ref_model {
            let realized: MotorVoltageVec = if alloc.saturated {
                let ff = to_array(&feedforward);
                from_array(core::array::from_fn(|m| {
                    let want = command[m];
                    let got = alloc.voltage[m] - alloc.collective_v * direction[m];
                    let kept = if want > KEPT_MIN_COMMAND || want < -KEPT_MIN_COMMAND {
                        f32::from(got / want).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    kept * ff[m]
                }))
            } else {
                feedforward
            };
            self.propagate_reference(&interp.ref_model, realized, drag);
        }

        // --- Populate telemetry ---
        let kp = interp.kp.as_uniform();
        self.telemetry = LqrTelemetry {
            feedback_v: to_array(&feedback),
            feedforward_v: to_array(&feedforward),
            affine_v: to_array(&affine),
            error_prop: to_array(&prop_error),
            error_integral: to_array(&self.integral_error),
            error_derivative: to_array(&self.derivative_error),
            kp_motor0: [kp.get(0, 0), kp.get(0, 1), kp.get(0, 2)],
            collective_v: alloc.collective_v,
            command_scale: alloc.command_scale,
            setpoint_rps: to_array(&setpoint_vec),
            ff_target_rps: to_array(&target),
            status: 1,
        };

        // --- Save state ---
        self.prev_prop_error = prop_error;
        self.prev_setpoint = setpoint_vec;

        output
    }
}

impl ControllerState {
    /// Track takeoff and landing, which gate integral accumulation.  Latches
    /// airborne at TAKEOFF_SPEED_FRACTION of hover; lands after the slow,
    /// quiet (and, with an accelerometer, 1 g) condition holds.  Without a
    /// hover speed in the blob the controller counts as always airborne.
    fn update_flight_state(&mut self, sensors: &SensorData, gains: &GainStorage, dt: Seconds) {
        let hover = gains.hover_rps;
        if hover <= RotPerSec::new(0.0) {
            self.airborne = true;
            return;
        }
        let w = sensors.motor_rps;
        let mean_rps = 0.25 * (w[0] + w[1] + w[2] + w[3]);
        if !self.airborne {
            if mean_rps >= TAKEOFF_SPEED_FRACTION * hover {
                self.airborne = true;
                self.landing_s = Seconds::new(0.0);
            }
            return;
        }
        let quiet = sensors.gyro_rps.iter().all(|r| r.abs() < LANDED_RATE);
        let a = sensors.accel;
        let accel_mag: MetersPerSecSq = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let has_accel = accel_mag > mps2(ACCEL_PRESENT);
        let supported =
            !has_accel || (accel_mag - mps2(STANDARD_GRAVITY)).abs() < mps2(LANDED_ACCEL_TOLERANCE);
        if mean_rps < LANDED_SPEED_FRACTION * hover && quiet && supported {
            self.landing_s += dt;
            let hold = if has_accel { LANDED_HOLD } else { LANDED_HOLD_NO_ACCEL };
            if self.landing_s >= hold {
                self.airborne = false;
                self.landing_s = Seconds::new(0.0);
            }
        } else {
            self.landing_s = Seconds::new(0.0);
        }
    }

    /// This tick's airframe drag disturbance α_d·dt (BF axes), or `None`
    /// when the blob has no drag model or the feature is off.
    #[cfg(feature = "airframe-drag")]
    fn airframe_drag(&mut self, sensors: &SensorData, gains: &GainStorage, dt: Seconds) -> Option<PropErrorVec> {
        let drag = gains.airframe_drag;
        if drag.lp <= Hertz::new(0.0) {
            return None;
        }
        let limit = mps2(DRAG_ACCEL_LIMIT);
        let a = [0, 1].map(|i| sensors.accel[i].clamp(-limit, limit));
        // One-pole low-pass; 2πf·dt/(1 + 2πf·dt) ≈ 1 − e^{−2πf·dt} at f ≪ 1/dt.
        let x: f32 = (2.0 * core::f32::consts::PI * drag.lp * dt).into();
        let k = x / (1.0 + x);
        let lp = match self.accel_lp {
            Some(p) => [p[0] + k * (a[0] - p[0]), p[1] + k * (a[1] - p[1])],
            None => a,
        };
        self.accel_lp = Some(lp);
        Some(from_array([
            (drag.roll[0] * lp[0] + drag.roll[1] * lp[1]) * dt,
            (drag.pitch[0] * lp[0] + drag.pitch[1] * lp[1]) * dt,
            RotPerSec::new(0.0),
        ]))
    }

    #[cfg(not(feature = "airframe-drag"))]
    fn airframe_drag(&mut self, _: &SensorData, _: &GainStorage, _: Seconds) -> Option<PropErrorVec> {
        None
    }

    /// The reference-model feedforward command for this tick (see module docs).
    ///
    /// `ref_model` is the interpolated reference model at the current
    /// operating point; `binv` is the interpolated horizon inverse G_h⁺.
    fn reference_feedforward(
        &self,
        ref_model: &ReferenceModel,
        binv: &BInvMatrix,
        target: PropErrorVec,
        drag: Option<PropErrorVec>,
        horizon: u8,
        battery_v: Volt,
    ) -> MotorVoltageVec {
        // Free response over the horizon: F_h·δx.  The rate→rate blocks are
        // diagonal (dimensionless decay), applied elementwise.
        let rate = to_array(&self.ref_rate);
        let from_speed = to_array(&(ref_model.fh_rw * self.ref_speed));
        let free: PropErrorVec = from_array(core::array::from_fn(|i| {
            ref_model.fh_rr[i] * rate[i] + from_speed[i]
        }));

        let mut request: SetpointDiffVec = target - free;
        if let Some(d) = drag {
            // Horizon response to the disturbance held constant: Σ_{j<h} ad_rr^j ⊙ d.
            let mut decay = [1.0f32; 3];
            let mut sum = [0.0f32; 3];
            for _ in 0..horizon {
                for i in 0..3 {
                    sum[i] += decay[i];
                    decay[i] *= ref_model.ad_rr[i];
                }
            }
            let (r, d) = (to_array(&request), to_array(&d));
            request = from_array(core::array::from_fn(|i| r[i] - sum[i] * d[i]));
        }
        let u_ff: MotorVoltageVec = *binv * request;
        from_array(to_array(&u_ff).map(|v| v.clamp(-battery_v, battery_v)))
    }

    /// One-tick ZOH propagation of the reference model with the voltage
    /// it actually got, plus the airframe drag disturbance:
    /// δx ← Ad·δx + Bd·u + [d; 0].
    fn propagate_reference(
        &mut self,
        ref_model: &ReferenceModel,
        u_ff: MotorVoltageVec,
        drag: Option<PropErrorVec>,
    ) {
        let rate = to_array(&self.ref_rate);
        let speed = to_array(&self.ref_speed);
        let u = to_array(&u_ff);
        let from_speed = to_array(&(ref_model.ad_rw * self.ref_speed));
        let from_volt = to_array(&(ref_model.bd_r * u_ff));
        let disturbance = drag.map_or([RotPerSec::new(0.0); 3], |d| to_array(&d));
        self.ref_rate = from_array(core::array::from_fn(|i| {
            ref_model.ad_rr[i] * rate[i] + from_speed[i] + from_volt[i] + disturbance[i]
        }));
        self.ref_speed = from_array(core::array::from_fn(|m| {
            ref_model.ad_ww[m] * speed[m] + ref_model.bd_w[m] * u[m]
        }));
    }
}

/// Bound on each accelerometer axis fed to the drag model, so impacts
/// don't charge its low-pass.
#[cfg(feature = "airframe-drag")]
const DRAG_ACCEL_LIMIT: DecaMetersPerSecSq = quantity!(3.0, g_n, f32);

/// Bound on the integral state's low-passed error e_int/dt.
const INTEGRAL_ERROR_RATE_LIMIT: RotPerSec = RotPerSec::new(0.25);

/// Battery reading below which the controller assumes this voltage (a
/// missing or implausible reading), so voltage → duty stays finite.
const MIN_BATTERY_V: Volt = Volt::new(1.0);

/// Commands smaller than this count as fully delivered when sharing out a
/// saturated command, so the ratio stays finite.
const KEPT_MIN_COMMAND: Volt = Volt::new(1e-6);

/// Largest integral voltage on any motor, as a fraction of battery voltage.
const INTEGRAL_VOLTAGE_FRACTION: f32 = 0.2;

/// Per-motor integral voltage bound: the blob's (what trimming a generous
/// CG offset at hover takes — 25% of the arm, the steady part of wind
/// included — so trim never runs out, while a collision's windup leaves a
/// recoverable torque), capped at INTEGRAL_VOLTAGE_FRACTION of the battery;
/// that cap alone for blobs without one.
fn integral_voltage_limit(gains: &GainStorage, battery_v: Volt) -> Volt {
    let cap = INTEGRAL_VOLTAGE_FRACTION * battery_v;
    let trim = gains.integral_limit_v;
    if trim > Volt::new(0.0) && trim < cap { trim } else { cap }
}

/// Takeoff: mean motor speed reaches this fraction of hover (≈70% of the
/// hover thrust).
const TAKEOFF_SPEED_FRACTION: f32 = 0.85;
/// Landing candidates: mean motor speed below this fraction of hover
/// (thrust under ≈⅓ of weight) with body rates under LANDED_RATE.
const LANDED_SPEED_FRACTION: f32 = 0.6;
const LANDED_RATE: RotPerSec = RotPerSec::new(0.05);
/// With an accelerometer, ground contact also reads a steady 1 g₀ (±this)
/// although the motors make well under it; in the air at that thrust the
/// specific force is the low thrust itself (falling reads ≪ 1 g₀).
const LANDED_ACCEL_TOLERANCE: DecaMetersPerSecSq = quantity!(0.15, g_n, f32);
/// How long the landing condition must hold: with / without accelerometer.
const LANDED_HOLD: Seconds = Seconds::new(0.5);
const LANDED_HOLD_NO_ACCEL: Seconds = Seconds::new(1.0);
/// Below this magnitude the accelerometer is taken as absent (not fitted,
/// or USE_ACC off: the glue leaves it zero).
const ACCEL_PRESENT: DecaMetersPerSecSq = quantity!(0.2, g_n, f32);
