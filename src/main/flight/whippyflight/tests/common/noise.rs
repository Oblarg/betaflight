//! The evals' sensor noise: $NOISE_PATH, else
//! tools/airframes/meteor65_noise.bin (the default plant's airframe), both
//! written by tools/extract_noise.py from a blackbox log of the airframe the
//! plant models.  The gyro noise was taken after BF's filters, so it belongs on top
//! of that airframe's logged gyro path, not the unfiltered one.
#![allow(dead_code)]

use std::sync::OnceLock;

use super::units::Time;
pub use whippyflight_control::sim::noise::Noise;

static DEFAULT_FIXTURE: &[u8] = include_bytes!("../../tools/airframes/meteor65_noise.bin");

fn fixture() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| match std::env::var("NOISE_PATH") {
        Ok(path) => std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")),
        Err(_) => DEFAULT_FIXTURE.to_vec(),
    })
}

/// The fixture's noise from `offset` on, gyro and motor-speed noise scaled separately.
pub fn fixture_noise(offset: Time, gyro_scale: f32, motor_scale: f32) -> Noise {
    Noise::from_bytes(fixture(), offset, gyro_scale, motor_scale).unwrap()
}
