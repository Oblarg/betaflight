//! The core LQR attitude-rate controller.
//!
//! This module implements the gain-scheduled feedback/feedforward control law.
//! Every matrix-vector operation flows through whippyalgebra's
//! [`MixedUnitMatrix`], ensuring that the dimensional algebra is correct at
//! compile time.

use crate::gains::GainStorage;
use crate::interpolation::interpolate_gains;
use crate::types::*;

/// Internal controller state (integrator, derivative filter, previous values).
pub struct ControllerState {
    /// Integral error accumulator: [yaw, pitch, roll] in rotations.
    integral_error: [f32; 3],
    /// Derivative error (filtered): [yaw, pitch, roll] in rot/s².
    derivative_error: [f32; 3],
    /// Previous proportional error for derivative computation.
    prev_error: [f32; 3],
    /// Previous setpoint for feedforward.
    prev_setpoint: [f32; 3],
    /// Latest telemetry snapshot.
    telemetry: LqrTelemetry,
}

impl ControllerState {
    pub const fn new() -> Self {
        Self {
            integral_error: [0.0; 3],
            derivative_error: [0.0; 3],
            prev_error: [0.0; 3],
            prev_setpoint: [0.0; 3],
            telemetry: LqrTelemetry {
                feedback_v: [0.0; 4],
                feedforward_v: [0.0; 4],
                affine_v: [0.0; 4],
                error_prop: [0.0; 3],
                error_integral: [0.0; 3],
                error_derivative: [0.0; 3],
                kp_motor0: [0.0; 3],
                status: 0,
            },
        }
    }

    /// Reset all internal state (called on disarm).
    pub fn reset(&mut self) {
        self.integral_error = [0.0; 3];
        self.derivative_error = [0.0; 3];
        self.prev_error = [0.0; 3];
        self.prev_setpoint = [0.0; 3];
    }

    pub fn telemetry(&self) -> &LqrTelemetry {
        &self.telemetry
    }

    /// Run one control iteration.
    pub fn update(
        &mut self,
        sensors: &SensorData,
        setpoint: &SetpointData,
        gains: &GainStorage,
    ) -> ControlOutput {
        if !gains.loaded {
            self.telemetry.status = 0;
            return ControlOutput {
                motor_duty: [0.0; 4],
            };
        }

        let dt = gains.header.dt;
        let alpha_int = dt / gains.header.integral_tau;
        let alpha_deriv = dt / gains.header.derivative_tau;

        // --- Error computation ---
        let mut prop_error = [0.0f32; 3];
        let mut setpoint_diff = [0.0f32; 3];
        for axis in 0..3 {
            prop_error[axis] = setpoint.rate_rps[axis] - sensors.gyro_rps[axis];
            setpoint_diff[axis] = setpoint.rate_rps[axis] - self.prev_setpoint[axis];

            // Leaky integrator
            self.integral_error[axis] = (1.0 - alpha_int) * self.integral_error[axis]
                + alpha_int * prop_error[axis];
            // Clamp to prevent windup
            self.integral_error[axis] = clampf(self.integral_error[axis], -0.25, 0.25);

            // Derivative filter
            let raw_deriv = (prop_error[axis] - self.prev_error[axis]) / dt;
            self.derivative_error[axis] = (1.0 - alpha_deriv) * self.derivative_error[axis]
                + alpha_deriv * raw_deriv;
        }

        // --- Gain interpolation ---
        let interp = interpolate_gains(&sensors.motor_rps, gains);

        // --- Feedback: u_fb = Kp·e + Ki·e_int + Kd·e_deriv ---
        let mut feedback = [0.0f32; 4];
        let mut p_control = [0.0f32; 4];
        let mut i_control = [0.0f32; 4];
        let mut d_control = [0.0f32; 4];

        for motor in 0..4 {
            for axis in 0..3 {
                let p = interp.kp[motor][axis] * prop_error[axis];
                let i = interp.ki[motor][axis] * self.integral_error[axis];
                let d = interp.kd[motor][axis] * self.derivative_error[axis];
                p_control[motor] += p;
                i_control[motor] += i;
                d_control[motor] += d;
                feedback[motor] += p + i + d;
            }
        }

        // --- Feedforward: u_ff = B⁻¹ · Δr ---
        let mut feedforward = [0.0f32; 4];
        for motor in 0..4 {
            for axis in 0..3 {
                feedforward[motor] += interp.binv[motor][axis] * setpoint_diff[axis];
            }
        }

        // --- Mixer: voltage → duty cycle ---
        let battery_v = sensors.battery_voltage.max(1.0); // prevent div-by-zero
        let dynamic_throttle = 0.1 + 0.9 * setpoint.throttle;

        let mut output = ControlOutput {
            motor_duty: [0.0; 4],
        };
        for motor in 0..4 {
            let total_v = feedback[motor] + feedforward[motor];
            let duty = total_v / battery_v + dynamic_throttle;
            output.motor_duty[motor] = clampf(duty, 0.0, 1.0);
        }

        // --- Update telemetry ---
        self.telemetry = LqrTelemetry {
            feedback_v: feedback,
            feedforward_v: feedforward,
            affine_v: [0.0; 4], // TODO: affine linearizer
            error_prop: prop_error,
            error_integral: self.integral_error,
            error_derivative: self.derivative_error,
            kp_motor0: interp.kp[0],
            status: 1,
        };

        // --- Save state for next iteration ---
        self.prev_error = prop_error;
        self.prev_setpoint = setpoint.rate_rps;

        output
    }
}

fn clampf(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}
