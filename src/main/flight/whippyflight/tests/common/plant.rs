//! The crate's sim plant, on a per-process model: $PLANT_JSON (a
//! fit_model.py fit), else tools/airframes/meteor65.json, which
//! test_gains.wfg was generated from (and SITL flies via
//! tools/sim_plant.py).  Evals that fly a $WFG_PATH blob should point
//! $PLANT_JSON at the JSON that blob was generated from.
#![allow(dead_code)]

use std::ops::{Deref, DerefMut};
use std::sync::OnceLock;

use super::units::*;
use whippyflight_control::sim::plant::Plant as SimPlant;
pub use whippyflight_control::sim::plant::Model;

static DEFAULT_JSON: &str = include_str!("../../tools/airframes/meteor65.json");

/// The plant model for this test process ($PLANT_JSON or the test JSON).
pub fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| {
        match std::env::var("PLANT_JSON") {
            Ok(path) => {
                let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
                Model::from_json(&text, &path)
            }
            Err(_) => Model::from_json(DEFAULT_JSON, "tools/airframes/meteor65.json"),
        }
        .unwrap_or_else(|e| panic!("{e}"))
    })
}

/// [`SimPlant`] on [`model`].
#[derive(Clone, Copy)]
pub struct Plant(SimPlant<'static>);

impl Plant {
    /// At rest with every motor at `scale` × the model's hover speed.
    pub fn hovering(scale: f64) -> Plant {
        Plant(SimPlant::hovering(model(), scale))
    }
}

impl Deref for Plant {
    type Target = SimPlant<'static>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Plant {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Motor voltages for the plant from the controller's duty output.
pub fn duty_volts(duty: [f32; 4]) -> [Voltage; 4] {
    model().duty_volts(duty)
}
