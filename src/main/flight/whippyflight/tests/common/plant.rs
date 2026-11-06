//! The nonlinear plant that gen_wfg.py linearizes, from a coefficient JSON:
//! $PLANT_JSON (a fit_model.py fit), else tools/airframes/meteor65.json,
//! which test_gains.wfg was generated from (and SITL flies via
//! tools/sim_plant.py).  Evals that fly a $WFG_PATH blob should point
//! $PLANT_JSON at the JSON that blob was generated from.
#![allow(dead_code)]

use std::sync::OnceLock;

static DEFAULT_JSON: &str = include_str!("../../tools/airframes/meteor65.json");

/// Battery voltage and hover motor speed when the JSON has no fitted
/// `airframe` block: the Meteor65 Pro (1S) test airframe.
const DEFAULT_BATTERY_V: f64 = 4.2;
const DEFAULT_HOVER_RPS: f64 = 270.0;
/// Its gyro low-pass stages (BF type, Hz), for JSONs without `gyro_lpf`.
const DEFAULT_GYRO_LPF: [(&str, f64); 2] = [("PT1", 275.0), ("PT1", 550.0)];

/// Plant coefficients, in gen_wfg's conventions (model axes, motors by BF index).
pub struct Model {
    pub yaw_rate: f64,
    pub yaw_prop_vel_sq: f64,
    pub yaw_prop_accel: f64,
    pub pitch_rate: f64,
    pub pitch_prop_vel_sq: f64,
    pub roll_rate: f64,
    pub roll_prop_vel_sq: f64,
    pub motor_vel: f64,
    pub motor_vel_sq: f64,
    pub motor_accel: f64,
    /// gen_wfg arm_weights: 2·d/(d_F + d_B) and 2·d/(d_L + d_R).
    pub w_front: f64,
    pub w_back: f64,
    pub w_left: f64,
    pub w_right: f64,
    /// Flight battery voltage (V) and median hover motor speed (model rps).
    pub battery_v: f64,
    pub hover_rps: f64,
    /// BF's gyro low-pass stages on the logged airframe (type, cutoff Hz);
    /// a dynamic stage is at its minimum (lowest, most lag) cutoff.
    pub gyro_lpf: Vec<(String, f64)>,
    /// Where the coefficients came from.
    pub source: String,
}

impl Model {
    pub fn from_json(text: &str, source: &str) -> Model {
        let c: serde_json::Value = serde_json::from_str(text).expect("coefficient JSON");
        let get = |axis: &str, key: &str| {
            c[axis][key].as_f64().unwrap_or_else(|| panic!("{source}: no {axis}.{key}"))
        };
        let opt = |axis: &str, key: &str, default: f64| c[axis][key].as_f64().unwrap_or(default);
        // gen_wfg.py's CLI defaults when the JSON doesn't carry fitted arms.
        let (d_front, d_back) = (opt("pitch", "dFront", 1.0), opt("pitch", "dBack", 1.7));
        let (d_left, d_right) = (opt("roll", "dLeft", 1.0), opt("roll", "dRight", 1.0));
        let gyro_lpf = match c["airframe"]["gyro_lpf"].as_array() {
            Some(stages) => stages
                .iter()
                .map(|s| {
                    let kind = s["type"].as_str().unwrap_or_else(|| panic!("{source}: gyro_lpf type"));
                    let hz = s["hz"].as_f64().unwrap_or_else(|| panic!("{source}: gyro_lpf hz"));
                    (kind.to_string(), hz)
                })
                .collect(),
            None => DEFAULT_GYRO_LPF.iter().map(|&(k, hz)| (k.to_string(), hz)).collect(),
        };
        Model {
            yaw_rate: get("yaw", "yawRate"),
            yaw_prop_vel_sq: get("yaw", "propVelSumSq"),
            yaw_prop_accel: get("yaw", "propAccelSum"),
            pitch_rate: get("pitch", "pitchRate"),
            pitch_prop_vel_sq: get("pitch", "propVelSumSq"),
            roll_rate: get("roll", "rollRate"),
            roll_prop_vel_sq: get("roll", "propVelSumSq"),
            motor_vel: get("motors", "motorVel"),
            motor_vel_sq: get("motors", "motorVelSq"),
            motor_accel: get("motors", "motorAccel"),
            w_front: 2.0 * d_front / (d_front + d_back),
            w_back: 2.0 * d_back / (d_front + d_back),
            w_left: 2.0 * d_left / (d_left + d_right),
            w_right: 2.0 * d_right / (d_left + d_right),
            battery_v: opt("airframe", "vbat_v", DEFAULT_BATTERY_V),
            hover_rps: opt("airframe", "hover_rps", DEFAULT_HOVER_RPS),
            gyro_lpf,
            source: source.to_string(),
        }
    }

    /// Steady-state voltage holding a motor at `omega` (model rps).
    pub fn motor_volts(&self, omega: f64) -> f64 {
        self.motor_vel * omega + self.motor_vel_sq * omega * omega.abs()
    }

    /// Throttle (fraction of battery) that holds every motor at `scale` × hover speed.
    pub fn throttle_at(&self, scale: f64) -> f32 {
        (self.motor_volts(scale * self.hover_rps) / self.battery_v) as f32
    }

    pub fn battery_v(&self) -> f32 {
        self.battery_v as f32
    }
}

/// The plant model for this test process ($PLANT_JSON or the test JSON).
pub fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| match std::env::var("PLANT_JSON") {
        Ok(path) => {
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            Model::from_json(&text, &path)
        }
        Err(_) => Model::from_json(DEFAULT_JSON, "tools/airframes/meteor65.json"),
    })
}

/// Model-axis state: rates [yaw, pitch, roll] (rot/s), motor speeds (eRPM/600).
#[derive(Clone, Copy)]
pub struct Plant {
    pub rate: [f64; 3],
    pub omega: [f64; 4],
}

impl Plant {
    /// At rest with every motor at `scale` × the model's hover speed.
    pub fn hovering(scale: f64) -> Plant {
        Plant { rate: [0.0; 3], omega: [scale * model().hover_rps; 4] }
    }

    /// `torque`: external angular acceleration, model axes (rot/s²).
    fn deriv(&self, volts: &[f64; 4], torque: &[f64; 3]) -> Plant {
        let m = model();
        let sq = self.omega.map(|w| w * w.abs());
        let mut wdot = [0.0; 4];
        for i in 0..4 {
            wdot[i] = (volts[i] - m.motor_vel * self.omega[i] - m.motor_vel_sq * sq[i]) / m.motor_accel;
        }
        let s_yaw = [1.0, -1.0, -1.0, 1.0];
        let s_pitch = [m.w_front, -m.w_back, m.w_front, -m.w_back];
        let s_roll = [m.w_left, m.w_left, -m.w_right, -m.w_right];
        let dot = |s: &[f64; 4], v: &[f64; 4]| (0..4).map(|i| s[i] * v[i]).sum::<f64>();
        Plant {
            rate: [
                m.yaw_rate * self.rate[0]
                    + m.yaw_prop_vel_sq * dot(&s_yaw, &sq)
                    + m.yaw_prop_accel * dot(&s_yaw, &wdot)
                    + torque[0],
                m.pitch_rate * self.rate[1] + m.pitch_prop_vel_sq * dot(&s_pitch, &sq) + torque[1],
                m.roll_rate * self.rate[2] + m.roll_prop_vel_sq * dot(&s_roll, &sq) + torque[2],
            ],
            omega: wdot,
        }
    }

    fn axpy(&self, k: &Plant, h: f64) -> Plant {
        Plant {
            rate: core::array::from_fn(|i| self.rate[i] + h * k.rate[i]),
            omega: core::array::from_fn(|i| self.omega[i] + h * k.omega[i]),
        }
    }

    pub fn rk4(&mut self, volts: &[f64; 4], h: f64) {
        self.rk4_with_torque(volts, h, [0.0; 3]);
    }

    /// [`Self::rk4`] with an external angular acceleration in Betaflight
    /// axes [roll, pitch, yaw] (rot/s²), held over the step.
    pub fn rk4_with_torque(&mut self, volts: &[f64; 4], h: f64, torque_bf: [f64; 3]) {
        let torque = [-torque_bf[2], torque_bf[1], -torque_bf[0]];
        let k1 = self.deriv(volts, &torque);
        let k2 = self.axpy(&k1, h / 2.0).deriv(volts, &torque);
        let k3 = self.axpy(&k2, h / 2.0).deriv(volts, &torque);
        let k4 = self.axpy(&k3, h).deriv(volts, &torque);
        *self = Plant {
            rate: core::array::from_fn(|i| {
                self.rate[i] + h / 6.0 * (k1.rate[i] + 2.0 * k2.rate[i] + 2.0 * k3.rate[i] + k4.rate[i])
            }),
            omega: core::array::from_fn(|i| {
                self.omega[i] + h / 6.0 * (k1.omega[i] + 2.0 * k2.omega[i] + 2.0 * k3.omega[i] + k4.omega[i])
            }),
        };
    }

    /// Betaflight gyro axes [roll, pitch, yaw] (gen_wfg MODEL_AXIS_FROM_BF).
    pub fn gyro_bf(&self) -> [f32; 3] {
        [-self.rate[2] as f32, self.rate[1] as f32, -self.rate[0] as f32]
    }
}
