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
use crate::gains::{GainStorage, REF_FLOATS};
use crate::interpolation::interpolate_gains;
use crate::setpoint::sqrt;
use crate::types::*;

use nalgebra::SMatrix;
use whippyalgebra::nalgebra::{mixed_unit_matrix, zeros};
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
    min_rps: f32,
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
    /// How long the landing condition has held, s.
    landing_s: f32,
    /// Low-passed body x/y specific force (g) for the airframe drag model;
    /// `None` until the first sample.
    #[cfg(feature = "airframe-drag")]
    accel_lp: Option<[f32; 2]>,
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
            min_rps: 0.0,
            ref_rate: zeros![AttitudeAxes, Scalar as f32],
            ref_speed: zeros![MotorSpeedAxes, Scalar as f32],
            ref_seeded: false,
            saturated: false,
            airborne: false,
            landing_s: 0.0,
            #[cfg(feature = "airframe-drag")]
            accel_lp: None,
            telemetry: LqrTelemetry {
                feedback_v: [0.0; 4],
                feedforward_v: [0.0; 4],
                affine_v: [0.0; 4],
                error_prop: [0.0; 3],
                error_integral: [0.0; 3],
                error_derivative: [0.0; 3],
                kp_motor0: [0.0; 3],
                collective_v: 0.0,
                command_scale: 1.0,
                setpoint_rps: [0.0; 3],
                ff_target_rps: [0.0; 3],
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
        self.landing_s = 0.0;
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
    pub fn set_min_rps(&mut self, min_rps: f32) {
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
    /// by the caller (rot/s, [roll, pitch, yaw]) instead of extrapolated as
    /// r + h·Δr.  Only used with a reference-model blob.
    pub fn update_with_target(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        ff_target_rps: [f32; 3],
        gains: &GainStorage,
    ) -> ControlOutput {
        self.step(sensors, setpoint, Some(ff_target_rps), gains)
    }

    fn step(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        ff_target_rps: Option<[f32; 3]>,
        gains: &GainStorage,
    ) -> ControlOutput {
        if !gains.loaded {
            self.telemetry.status = 0;
            return ControlOutput { motor_duty: [0.0; 4] };
        }

        let dt = gains.header.dt;
        let alpha_int = dt / gains.header.integral_tau;
        let alpha_deriv = dt / gains.header.derivative_tau;

        // --- Construct typed state vectors from FFI data ---
        //
        // SensorData arrives as raw f32 arrays from BF's C code.  We lift them
        // into typed vectors at the FFI boundary — from here on, everything is
        // unit-checked.

        let measurement: PropErrorVec = mixed_unit_matrix![AttitudeAxes, Scalar as f32;
            [quantity!(sensors.gyro_rps[0], rot / s, f32)],
            [quantity!(sensors.gyro_rps[1], rot / s, f32)],
            [quantity!(sensors.gyro_rps[2], rot / s, f32)],
        ];

        let setpoint_vec: SetpointDiffVec = mixed_unit_matrix![AttitudeAxes, Scalar as f32;
            [quantity!(setpoint.rate_rps[0], rot / s, f32)],
            [quantity!(setpoint.rate_rps[1], rot / s, f32)],
            [quantity!(setpoint.rate_rps[2], rot / s, f32)],
        ];

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
        // at 0.25 rot/s.  We drop to raw nalgebra for the scalar multiply +
        // clamp, then re-wrap.  While the last command was saturated, or on
        // the ground, the error isn't correctable, so the state only leaks
        // (anti-windup).
        self.update_flight_state(sensors, gains, dt);
        let int_gain = if self.saturated || !self.airborne { 0.0 } else { alpha_int * dt };
        let int_raw = self.integral_error.nalgebra() * (1.0 - alpha_int)
            + prop_error.nalgebra() * int_gain;
        let int_limit = 0.25 * dt;
        let int_clamped = int_raw.map(|x| clampf(x, -int_limit, int_limit));
        self.integral_error = IntegralErrorVec::from_nalgebra(int_clamped);

        // Derivative filter: ė[k] = (1-α_d)·ė[k-1] + α_d·(e[k]-e[k-1])/dt
        let deriv_raw = self.derivative_error.nalgebra() * (1.0 - alpha_deriv)
            + (prop_error.nalgebra() - self.prev_prop_error.nalgebra()) * (alpha_deriv / dt);
        self.derivative_error = DerivErrorVec::from_nalgebra(deriv_raw);

        // --- Gain interpolation (returns typed matrices) ---
        let header = &gains.header;
        let floor = if self.min_rps > header.min_rps { self.min_rps } else { header.min_rps };
        let motor_rps = sensors.motor_rps.map(|w| clampf(w, floor, header.max_rps));
        let interp = interpolate_gains(&motor_rps, gains);

        // --- Feedback law: u_fb = Kp·e + Ki·e_int + Kd·ė ---
        //
        // Each of these multiplications is compile-time unit-checked:
        //   KpMatrix [MotorAxes × AttitudeAxes]   ×  PropErrorVec [AttitudeAxes × Scalar]    → MotorVoltageVec
        //   KiMatrix [MotorAxes × IntegralAxes]   ×  IntegralErrorVec [IntegralAxes × Scalar] → MotorVoltageVec
        //   KdMatrix [MotorAxes × DerivativeAxes] ×  DerivErrorVec [DerivativeAxes × Scalar]  → MotorVoltageVec
        let battery_v = if sensors.battery_voltage > 1.0 {
            sensors.battery_voltage
        } else {
            1.0
        };

        // Bound the integral's voltage on every motor by scaling its state:
        // enough to trim a generous imbalance (see TRIM_CG_ARM_FRACTION),
        // little enough that windup from a collision stays recoverable.
        let mut u_i: MotorVoltageVec = interp.ki * self.integral_error;
        let u_i_max = u_i.nalgebra().amax();
        let u_i_limit = integral_voltage_limit(gains, battery_v);
        if u_i_max > u_i_limit {
            let s = u_i_limit / u_i_max;
            self.integral_error = IntegralErrorVec::from_nalgebra(self.integral_error.nalgebra() * s);
            u_i = MotorVoltageVec::from_nalgebra(u_i.nalgebra() * s);
        }

        let u_p: MotorVoltageVec = interp.kp * prop_error;
        let u_d: MotorVoltageVec = interp.kd * self.derivative_error;
        let feedback: MotorVoltageVec = u_p + u_i + u_d;

        // --- Feedforward ---
        // r_target = r + h·Δr unless the caller predicted it.
        let target: PropErrorVec = match ff_target_rps {
            Some(t) => mixed_unit_matrix![AttitudeAxes, Scalar as f32;
                [quantity!(t[0], rot / s, f32)],
                [quantity!(t[1], rot / s, f32)],
                [quantity!(t[2], rot / s, f32)],
            ],
            None => PropErrorVec::from_nalgebra(
                setpoint_vec.nalgebra()
                    + setpoint_diff.nalgebra() * gains.header.ff_horizon as f32),
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
        let omega_sq = |m: usize| motor_rps[m] * motor_rps[m].abs();
        let speed_sq: MotorSpeedSqVec = mixed_unit_matrix![MotorSpeedSqAxes, Scalar as f32;
            [quantity!(omega_sq(0), rot ^ 2 / s ^ 2, f32)],
            [quantity!(omega_sq(1), rot ^ 2 / s ^ 2, f32)],
            [quantity!(omega_sq(2), rot ^ 2 / s ^ 2, f32)],
            [quantity!(omega_sq(3), rot ^ 2 / s ^ 2, f32)],
        ];
        let affine_g = AffineMatrix::from_nalgebra(SMatrix::<f32, 4, 4>::from_row_slice(&gains.affine));
        let affine: MotorVoltageVec = affine_g * speed_sq;

        // --- Total control voltage ---
        let total: MotorVoltageVec = feedback + feedforward + affine;

        // --- Allocation: fit the command into [v_idle, Vbat] along the
        // thrust-balanced direction (see `allocation`), then voltage → duty.
        // Drop back to raw f32 at the FFI boundary.
        let [k_w, k_w2] = gains.motor_drag;
        let idle_v = clampf(k_w * floor + k_w2 * floor * floor, 0.0, 0.5 * battery_v);
        let total_raw = total.nalgebra();
        let pilot_collective = clampf(setpoint.throttle, 0.0, 1.0) * battery_v;
        let direction = balanced_direction(gains.thrust_balance, gains.motor_drag, &gains.affine,
                                           if pilot_collective > idle_v { pilot_collective } else { idle_v });
        let alloc = allocate(
            [0, 1, 2, 3].map(|m| total_raw[(m, 0)]),
            direction,
            pilot_collective,
            idle_v,
            battery_v,
        );
        let output = ControlOutput {
            motor_duty: alloc.voltage.map(|v| clampf(v / battery_v, 0.0, 1.0)),
        };

        self.saturated = alloc.saturated;

        // Propagate the reference model with the feedforward the motors
        // actually got: each motor's share of its command that survived
        // scaling/clipping (the collective c·n is not part of the command).
        if gains.has_ref_model {
            let realized = if alloc.saturated {
                let ff_raw = feedforward.nalgebra();
                MotorVoltageVec::from_nalgebra(SMatrix::<f32, 4, 1>::from_fn(|m, _| {
                    let want = total_raw[(m, 0)];
                    let got = alloc.voltage[m] - alloc.collective_v * direction[m];
                    let kept = if want > 1e-6 || want < -1e-6 { clampf(got / want, 0.0, 1.0) } else { 1.0 };
                    kept * ff_raw[(m, 0)]
                }))
            } else {
                feedforward
            };
            self.propagate_reference(&interp.ref_model, realized, drag);
        }

        // --- Populate telemetry (drop to raw at the boundary) ---
        let fb_raw = feedback.nalgebra();
        let ff_raw = feedforward.nalgebra();
        let af_raw = affine.nalgebra();
        let pe_raw = prop_error.nalgebra();
        let ie_raw = self.integral_error.nalgebra();
        let de_raw = self.derivative_error.nalgebra();
        let kp_raw = interp.kp.nalgebra();
        let sp_raw = setpoint_vec.nalgebra();
        let tg_raw = target.nalgebra();

        self.telemetry = LqrTelemetry {
            feedback_v: [fb_raw[(0,0)], fb_raw[(1,0)], fb_raw[(2,0)], fb_raw[(3,0)]],
            feedforward_v: [ff_raw[(0,0)], ff_raw[(1,0)], ff_raw[(2,0)], ff_raw[(3,0)]],
            affine_v: [af_raw[(0,0)], af_raw[(1,0)], af_raw[(2,0)], af_raw[(3,0)]],
            error_prop: [pe_raw[(0,0)], pe_raw[(1,0)], pe_raw[(2,0)]],
            error_integral: [ie_raw[(0,0)], ie_raw[(1,0)], ie_raw[(2,0)]],
            error_derivative: [de_raw[(0,0)], de_raw[(1,0)], de_raw[(2,0)]],
            kp_motor0: [kp_raw[(0,0)], kp_raw[(0,1)], kp_raw[(0,2)]],
            collective_v: alloc.collective_v,
            command_scale: alloc.command_scale,
            setpoint_rps: [sp_raw[(0,0)], sp_raw[(1,0)], sp_raw[(2,0)]],
            ff_target_rps: [tg_raw[(0,0)], tg_raw[(1,0)], tg_raw[(2,0)]],
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
    fn update_flight_state(&mut self, sensors: &SensorData, gains: &GainStorage, dt: f32) {
        let hover = gains.hover_rps;
        if hover <= 0.0 {
            self.airborne = true;
            return;
        }
        let w = sensors.motor_rps;
        let mean_rps = 0.25 * (w[0] + w[1] + w[2] + w[3]);
        if !self.airborne {
            if mean_rps >= TAKEOFF_SPEED_FRACTION * hover {
                self.airborne = true;
                self.landing_s = 0.0;
            }
            return;
        }
        let quiet = sensors.gyro_rps.iter().all(|r| r.abs() < LANDED_RATE_RPS);
        let a = sensors.accel_g;
        let accel_mag = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        let has_accel = accel_mag > ACCEL_PRESENT_G;
        let supported = !has_accel || (accel_mag - 1.0).abs() < LANDED_ACCEL_TOLERANCE_G;
        if mean_rps < LANDED_SPEED_FRACTION * hover && quiet && supported {
            self.landing_s += dt;
            let hold = if has_accel { LANDED_HOLD_S } else { LANDED_HOLD_NO_ACCEL_S };
            if self.landing_s >= hold {
                self.airborne = false;
                self.landing_s = 0.0;
            }
        } else {
            self.landing_s = 0.0;
        }
    }

    /// This tick's airframe drag disturbance α_d·dt (rot/s, BF axes), or
    /// `None` when the blob has no drag model or the feature is off.
    #[cfg(feature = "airframe-drag")]
    fn airframe_drag(&mut self, sensors: &SensorData, gains: &GainStorage, dt: f32) -> Option<PropErrorVec> {
        let [lp_hz, roll_x, roll_y, pitch_x, pitch_y] = gains.airframe_drag;
        if lp_hz <= 0.0 {
            return None;
        }
        let a = [0, 1].map(|i| clampf(sensors.accel_g[i], -DRAG_ACCEL_LIMIT_G, DRAG_ACCEL_LIMIT_G));
        // One-pole low-pass; 2πf·dt/(1 + 2πf·dt) ≈ 1 − e^{−2πf·dt} at f ≪ 1/dt.
        let x = 2.0 * core::f32::consts::PI * lp_hz * dt;
        let k = x / (1.0 + x);
        let lp = match self.accel_lp {
            Some(p) => [p[0] + k * (a[0] - p[0]), p[1] + k * (a[1] - p[1])],
            None => a,
        };
        self.accel_lp = Some(lp);
        Some(PropErrorVec::from_nalgebra(SMatrix::<f32, 3, 1>::new(
            (roll_x * lp[0] + roll_y * lp[1]) * dt,
            (pitch_x * lp[0] + pitch_y * lp[1]) * dt,
            0.0,
        )))
    }

    #[cfg(not(feature = "airframe-drag"))]
    fn airframe_drag(&mut self, _: &SensorData, _: &GainStorage, _: f32) -> Option<PropErrorVec> {
        None
    }

    /// The reference-model feedforward command for this tick (see module docs).
    ///
    /// `ref_model` is the interpolated [`REF_FLOATS`] block at the current
    /// operating point; `binv` is the interpolated horizon inverse G_h⁺.
    fn reference_feedforward(
        &self,
        ref_model: &[f32; REF_FLOATS],
        binv: &BInvMatrix,
        target: PropErrorVec,
        drag: Option<PropErrorVec>,
        horizon: u8,
        battery_v: f32,
    ) -> MotorVoltageVec {
        let diag3 = |o: usize| SMatrix::<f32, 3, 1>::from_column_slice(&ref_model[o..o + 3]);
        let rows3x4 = |o: usize| SMatrix::<f32, 3, 4>::from_row_slice(&ref_model[o..o + 12]);
        let fh_rr = diag3(35);
        let fh_rw = RateFromSpeedMatrix::from_nalgebra(rows3x4(38));

        // Free response over the horizon: F_h·δx.  The rate→rate blocks are
        // diagonal (dimensionless decay), applied elementwise.
        let free_from_speed: PropErrorVec = fh_rw * self.ref_speed;
        let free = PropErrorVec::from_nalgebra(
            fh_rr.component_mul(self.ref_rate.nalgebra()) + free_from_speed.nalgebra());

        let mut request: SetpointDiffVec = target - free;
        if let Some(d) = drag {
            // Horizon response to the disturbance held constant: Σ_{j<h} ad_rr^j ⊙ d.
            let ad_rr = diag3(0);
            let mut decay = SMatrix::<f32, 3, 1>::repeat(1.0);
            let mut sum = SMatrix::<f32, 3, 1>::zeros();
            for _ in 0..horizon {
                sum += decay;
                decay = decay.component_mul(&ad_rr);
            }
            request = SetpointDiffVec::from_nalgebra(
                request.nalgebra() - sum.component_mul(d.nalgebra()));
        }
        let u_ff: MotorVoltageVec = *binv * request;
        MotorVoltageVec::from_nalgebra(u_ff.nalgebra().map(|v| clampf(v, -battery_v, battery_v)))
    }

    /// One-tick ZOH propagation of the reference model with the voltage
    /// it actually got, plus the airframe drag disturbance:
    /// δx ← Ad·δx + Bd·u + [d; 0].
    fn propagate_reference(
        &mut self,
        ref_model: &[f32; REF_FLOATS],
        u_ff: MotorVoltageVec,
        drag: Option<PropErrorVec>,
    ) {
        let diag3 = |o: usize| SMatrix::<f32, 3, 1>::from_column_slice(&ref_model[o..o + 3]);
        let diag4 = |o: usize| SMatrix::<f32, 4, 1>::from_column_slice(&ref_model[o..o + 4]);
        let rows3x4 = |o: usize| SMatrix::<f32, 3, 4>::from_row_slice(&ref_model[o..o + 12]);
        let ad_rr = diag3(0);
        let ad_rw = RateFromSpeedMatrix::from_nalgebra(rows3x4(3));
        let ad_ww = diag4(15);
        let bd_r = RateFromVoltMatrix::from_nalgebra(rows3x4(19));
        let bd_w = diag4(31);
        let rate_from_speed: PropErrorVec = ad_rw * self.ref_speed;
        let rate_from_volt: PropErrorVec = bd_r * u_ff;
        let disturbance = drag.map_or(SMatrix::<f32, 3, 1>::zeros(), |d| *d.nalgebra());
        self.ref_rate = PropErrorVec::from_nalgebra(
            ad_rr.component_mul(self.ref_rate.nalgebra())
                + rate_from_speed.nalgebra()
                + rate_from_volt.nalgebra()
                + disturbance);
        self.ref_speed = MotorSpeedVec::from_nalgebra(
            ad_ww.component_mul(self.ref_speed.nalgebra())
                + bd_w.component_mul(u_ff.nalgebra()));
    }
}

/// Bound on each accelerometer axis fed to the drag model, so impacts don't
/// charge its low-pass.
#[cfg(feature = "airframe-drag")]
const DRAG_ACCEL_LIMIT_G: f32 = 3.0;

/// Largest integral voltage on any motor, as a fraction of battery voltage.
const INTEGRAL_VOLTAGE_FRACTION: f32 = 0.2;

/// Per-motor integral voltage bound: the blob's (what trimming a generous
/// CG offset at hover takes — 25% of the arm, the steady part of wind
/// included — so trim never runs out, while a collision's windup leaves a
/// recoverable torque), capped at INTEGRAL_VOLTAGE_FRACTION of the battery;
/// that cap alone for blobs without one.
fn integral_voltage_limit(gains: &GainStorage, battery_v: f32) -> f32 {
    let cap = INTEGRAL_VOLTAGE_FRACTION * battery_v;
    let trim = gains.integral_limit_v;
    if trim > 0.0 && trim < cap { trim } else { cap }
}

/// Takeoff: mean motor speed reaches this fraction of hover (≈70% of the
/// hover thrust).
const TAKEOFF_SPEED_FRACTION: f32 = 0.85;
/// Landing candidates: mean motor speed below this fraction of hover
/// (thrust under ≈⅓ of weight) with body rates under LANDED_RATE_RPS.
const LANDED_SPEED_FRACTION: f32 = 0.6;
const LANDED_RATE_RPS: f32 = 0.05;
/// With an accelerometer, ground contact also reads a steady 1 g (±this)
/// although the motors make well under it; in the air at that thrust the
/// specific force is the low thrust itself (falling reads ≪ 1 g).
const LANDED_ACCEL_TOLERANCE_G: f32 = 0.15;
/// How long the landing condition must hold: with / without accelerometer.
const LANDED_HOLD_S: f32 = 0.5;
const LANDED_HOLD_NO_ACCEL_S: f32 = 1.0;
/// Below this magnitude the accelerometer is taken as absent (not fitted,
/// or USE_ACC off: the glue leaves it zero).
const ACCEL_PRESENT_G: f32 = 0.2;

fn clampf(x: f32, min: f32, max: f32) -> f32 {
    if x < min { min } else if x > max { max } else { x }
}
