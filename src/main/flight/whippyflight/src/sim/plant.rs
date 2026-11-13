//! The nonlinear plant that gen_wfg.py linearizes, from a fit_model.py
//! coefficient JSON (the same model as tools/sim_plant.py).

use super::units::*;
use crate::controller::ControllerState;
use crate::setpoint::{DEFAULT_LAMBDA_500HZ, SetpointEstimator};
use crate::types::{RotPerSec, Volt};

/// fit_model.py's `motor_speed`: the unit its motor speeds are in.  Older fits
/// (without it) are in eRPM/600, which this can't convert without the pole
/// count they were logged with.
pub const MOTOR_SPEED_UNIT: &str = "shaft rev/s";

/// Battery voltage and hover motor speed when the JSON has no fitted
/// `airframe` block: the Meteor65 Pro (1S) test airframe.
const DEFAULT_BATTERY_V: Voltage = Voltage::new(4.2);
const DEFAULT_HOVER_RPS: Rate = Rate::new(450.0);
/// Its gyro low-pass stages (BF type, cutoff), for JSONs without `gyro_lpf`.
const DEFAULT_GYRO_LPF: [(&str, Freq); 2] = [("PT1", Freq::new(275.0)), ("PT1", Freq::new(550.0))];

/// Plant coefficients, in gen_wfg's conventions (model axes, motors by BF index).
pub struct Model {
    pub yaw_rate: PerTime,
    pub yaw_prop_vel_sq: PerRot,
    /// rot/s² of yaw per rot/s² of motor acceleration.
    pub yaw_prop_accel: f64,
    pub pitch_rate: PerTime,
    pub pitch_prop_vel_sq: PerRot,
    pub roll_rate: PerTime,
    pub roll_prop_vel_sq: PerRot,
    pub motor_vel: MotorVel,
    pub motor_vel_sq: MotorVelSq,
    pub motor_accel: MotorAccel,
    /// gen_wfg arm_weights: 2·d/(d_F + d_B) and 2·d/(d_L + d_R).
    pub w_front: f64,
    pub w_back: f64,
    pub w_left: f64,
    pub w_right: f64,
    /// Flight battery voltage and median hover motor speed (shaft rev/s).
    pub battery_v: Voltage,
    pub hover_rps: Rate,
    /// BF's gyro low-pass stages on the logged airframe (type, cutoff);
    /// a dynamic stage is at its minimum (lowest, most lag) cutoff.
    pub gyro_lpf: Vec<(String, Freq)>,
    /// Cutoff of the controller's low-pass on reported motor speeds
    /// (`ControllerState::set_motor_lpf`); 0, off, without `rpm_lpf_hz`.
    pub motor_lpf: Freq,
    /// The firmware's minimum linearization speed, `idle_rps`
    /// (`ControllerState::set_min_rps`); 0, the gain grid's own floor,
    /// without it.
    pub min_rps: RotPerSec,
    /// Stick setpoint tracking index λ at 500 Hz
    /// (`SetpointEstimator::set_lambda`), from `stick_tracking`.
    pub stick_tracking: f32,
    /// Where the coefficients came from.
    pub source: String,
}

impl Model {
    pub fn from_json(text: &str, source: &str) -> Result<Model, String> {
        let c: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("{source}: {e}"))?;
        if c["motor_speed"].as_str() != Some(MOTOR_SPEED_UNIT) {
            return Err(format!(
                "{source}: fitted before motor speeds were in {MOTOR_SPEED_UNIT}; fit it again"
            ));
        }
        let get = |axis: &str, key: &str| c[axis][key].as_f64().ok_or_else(|| format!("{source}: no {axis}.{key}"));
        let opt = |axis: &str, key: &str, default: f64| c[axis][key].as_f64().unwrap_or(default);
        // gen_wfg.py's CLI defaults when the JSON doesn't carry fitted arms.
        let (d_front, d_back) = (opt("pitch", "dFront", 1.0), opt("pitch", "dBack", 1.7));
        let (d_left, d_right) = (opt("roll", "dLeft", 1.0), opt("roll", "dRight", 1.0));
        let gyro_lpf = match c["airframe"]["gyro_lpf"].as_array() {
            Some(stages) => stages
                .iter()
                .map(|s| {
                    let kind = s["type"].as_str().ok_or_else(|| format!("{source}: gyro_lpf type"))?;
                    let hz = s["hz"].as_f64().ok_or_else(|| format!("{source}: gyro_lpf hz"))?;
                    Ok((kind.to_string(), Freq::new(hz)))
                })
                .collect::<Result<_, String>>()?,
            None => DEFAULT_GYRO_LPF.iter().map(|&(k, f)| (k.to_string(), f)).collect(),
        };
        Ok(Model {
            yaw_rate: PerTime::new(get("yaw", "yawRate")?),
            yaw_prop_vel_sq: PerRot::new(get("yaw", "propVelSumSq")?),
            yaw_prop_accel: get("yaw", "propAccelSum")?,
            pitch_rate: PerTime::new(get("pitch", "pitchRate")?),
            pitch_prop_vel_sq: PerRot::new(get("pitch", "propVelSumSq")?),
            roll_rate: PerTime::new(get("roll", "rollRate")?),
            roll_prop_vel_sq: PerRot::new(get("roll", "propVelSumSq")?),
            motor_vel: MotorVel::new(get("motors", "motorVel")?),
            motor_vel_sq: MotorVelSq::new(get("motors", "motorVelSq")?),
            motor_accel: MotorAccel::new(get("motors", "motorAccel")?),
            w_front: 2.0 * d_front / (d_front + d_back),
            w_back: 2.0 * d_back / (d_front + d_back),
            w_left: 2.0 * d_left / (d_left + d_right),
            w_right: 2.0 * d_right / (d_left + d_right),
            battery_v: c["airframe"]["vbat_v"].as_f64().map_or(DEFAULT_BATTERY_V, Voltage::new),
            hover_rps: c["airframe"]["hover_rps"].as_f64().map_or(DEFAULT_HOVER_RPS, Rate::new),
            gyro_lpf,
            motor_lpf: Freq::new(c["airframe"]["rpm_lpf_hz"].as_f64().unwrap_or(0.0)),
            min_rps: RotPerSec::new(c["airframe"]["idle_rps"].as_f64().unwrap_or(0.0) as f32),
            stick_tracking: c["airframe"]["stick_tracking"]
                .as_f64()
                .map_or(DEFAULT_LAMBDA_500HZ, |l| l as f32),
            source: source.to_string(),
        })
    }

    /// Set up a controller and setpoint estimator as the firmware would on
    /// the logged airframe's settings.
    pub fn configure(&self, ctrl: &mut ControllerState, sp: &mut SetpointEstimator) {
        ctrl.set_motor_lpf(self.motor_lpf.lossy_into());
        ctrl.set_min_rps(self.min_rps);
        sp.set_lambda(self.stick_tracking);
    }

    /// Steady-state voltage holding a motor at `omega`.
    pub fn motor_volts(&self, omega: Rate) -> Voltage {
        self.motor_vel * omega + self.motor_vel_sq * omega * omega.abs()
    }

    /// Throttle (fraction of battery) that holds every motor at `scale` × hover speed.
    pub fn throttle_at(&self, scale: f64) -> f32 {
        f64::from(self.motor_volts(scale * self.hover_rps) / self.battery_v) as f32
    }

    pub fn battery_v(&self) -> Volt {
        self.battery_v.lossy_into()
    }

    /// Motor voltages for the plant from the controller's duty output.
    pub fn duty_volts(&self, duty: [f32; 4]) -> [Voltage; 4] {
        duty.map(|d| (d * self.battery_v()).lossy_into())
    }
}

/// Model-axis state: rates [yaw, pitch, roll], motor shaft speeds.
#[derive(Clone, Copy)]
pub struct Plant<'m> {
    pub rate: [Rate; 3],
    pub omega: [Rate; 4],
    pub model: &'m Model,
}

/// Time derivative of a [`Plant`].
#[derive(Clone, Copy)]
struct PlantDeriv {
    rate: [AngAccel; 3],
    omega: [AngAccel; 4],
}

impl<'m> Plant<'m> {
    /// At rest with every motor at `scale` × the model's hover speed.
    pub fn hovering(model: &'m Model, scale: f64) -> Self {
        Plant { rate: [Rate::new(0.0); 3], omega: [scale * model.hover_rps; 4], model }
    }

    /// `torque`: external angular acceleration, model axes.
    fn deriv(&self, volts: &[Voltage; 4], torque: &[AngAccel; 3]) -> PlantDeriv {
        let m = self.model;
        let sq = self.omega.map(|w| w * w.abs());
        let mut wdot = [AngAccel::new(0.0); 4];
        for i in 0..4 {
            wdot[i] = (volts[i] - m.motor_vel * self.omega[i] - m.motor_vel_sq * sq[i]) / m.motor_accel;
        }
        let s_yaw = [1.0, -1.0, -1.0, 1.0];
        let s_pitch = [m.w_front, -m.w_back, m.w_front, -m.w_back];
        let s_roll = [m.w_left, m.w_left, -m.w_right, -m.w_right];
        let dot_sq = |s: &[f64; 4]| (0..4).fold(SpeedSq::new(0.0), |acc, i| acc + s[i] * sq[i]);
        let dot_wdot = |s: &[f64; 4]| (0..4).fold(AngAccel::new(0.0), |acc, i| acc + s[i] * wdot[i]);
        PlantDeriv {
            rate: [
                m.yaw_rate * self.rate[0]
                    + m.yaw_prop_vel_sq * dot_sq(&s_yaw)
                    + m.yaw_prop_accel * dot_wdot(&s_yaw)
                    + torque[0],
                m.pitch_rate * self.rate[1] + m.pitch_prop_vel_sq * dot_sq(&s_pitch) + torque[1],
                m.roll_rate * self.rate[2] + m.roll_prop_vel_sq * dot_sq(&s_roll) + torque[2],
            ],
            omega: wdot,
        }
    }

    fn axpy(&self, k: &PlantDeriv, h: Time) -> Self {
        Plant {
            rate: core::array::from_fn(|i| self.rate[i] + h * k.rate[i]),
            omega: core::array::from_fn(|i| self.omega[i] + h * k.omega[i]),
            model: self.model,
        }
    }

    pub fn rk4(&mut self, volts: &[Voltage; 4], h: Time) {
        self.rk4_with_torque(volts, h, [AngAccel::new(0.0); 3]);
    }

    /// [`Self::rk4`] with an external angular acceleration in Betaflight
    /// axes [roll, pitch, yaw], held over the step.
    pub fn rk4_with_torque(&mut self, volts: &[Voltage; 4], h: Time, torque_bf: [AngAccel; 3]) {
        let torque = [-torque_bf[2], torque_bf[1], -torque_bf[0]];
        let k1 = self.deriv(volts, &torque);
        let k2 = self.axpy(&k1, h / 2.0).deriv(volts, &torque);
        let k3 = self.axpy(&k2, h / 2.0).deriv(volts, &torque);
        let k4 = self.axpy(&k3, h).deriv(volts, &torque);
        self.rate = core::array::from_fn(|i| {
            self.rate[i] + h / 6.0 * (k1.rate[i] + 2.0 * k2.rate[i] + 2.0 * k3.rate[i] + k4.rate[i])
        });
        self.omega = core::array::from_fn(|i| {
            self.omega[i] + h / 6.0 * (k1.omega[i] + 2.0 * k2.omega[i] + 2.0 * k3.omega[i] + k4.omega[i])
        });
    }

    /// Betaflight gyro axes [roll, pitch, yaw] (gen_wfg MODEL_AXIS_FROM_BF).
    pub fn gyro_bf(&self) -> [RotPerSec; 3] {
        [-self.rate[2], self.rate[1], -self.rate[0]].map(|r| r.lossy_into())
    }

    /// Motor speed telemetry as the controller reads it.
    pub fn motor_rps(&self) -> [RotPerSec; 4] {
        self.omega.map(|w| w.lossy_into())
    }
}
