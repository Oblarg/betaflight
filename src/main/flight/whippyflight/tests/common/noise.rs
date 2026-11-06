//! Replay of measured sensor noise: $NOISE_PATH, else
//! tools/airframes/meteor65_noise.bin (the default plant's airframe), both
//! written by tools/extract_noise.py from a blackbox log of the airframe the
//! plant models.  The gyro noise was taken after BF's filters, so it belongs on top
//! of that airframe's logged gyro path, not the unfiltered one.
#![allow(dead_code)]

use std::sync::OnceLock;

static DEFAULT_FIXTURE: &[u8] = include_bytes!("../../tools/airframes/meteor65_noise.bin");

fn fixture() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| match std::env::var("NOISE_PATH") {
        Ok(path) => std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}")),
        Err(_) => DEFAULT_FIXTURE.to_vec(),
    })
}
const GYRO_LSB: f32 = 1e-5;
const ERPM_LSB: f32 = 1e-3;
const CHANNELS: usize = 7;

pub struct Noise {
    fs: f64,
    frames: Vec<[f32; CHANNELS]>,
    offset_s: f64,
    gyro_scale: f32,
    motor_scale: f32,
}

pub struct NoiseSample {
    /// BF gyro axes [roll, pitch, yaw], rot/s.
    pub gyro: [f32; 3],
    /// Motor speed telemetry, model rps.
    pub motor: [f32; 4],
}

impl Noise {
    /// Start `offset_s` into the recording (it loops), with the gyro and
    /// motor-speed noise scaled separately.
    pub fn new(offset_s: f64, gyro_scale: f32, motor_scale: f32) -> Self {
        let fixture = fixture();
        let fs = f32::from_le_bytes(fixture[0..4].try_into().unwrap()) as f64;
        let n = u32::from_le_bytes(fixture[4..8].try_into().unwrap()) as usize;
        let frames = (0..n)
            .map(|k| {
                core::array::from_fn(|c| {
                    let at = 8 + 2 * (k * CHANNELS + c);
                    let lsb = if c < 3 { GYRO_LSB } else { ERPM_LSB };
                    i16::from_le_bytes([fixture[at], fixture[at + 1]]) as f32 * lsb
                })
            })
            .collect();
        Self { fs, frames, offset_s, gyro_scale, motor_scale }
    }

    pub fn duration_s(&self) -> f64 {
        self.frames.len() as f64 / self.fs
    }

    /// Noise at time `t` s, linearly interpolated from the log rate.
    pub fn at(&self, t: f64) -> NoiseSample {
        let n = self.frames.len();
        let x = ((t + self.offset_s) * self.fs).rem_euclid(n as f64);
        let (i, f) = (x as usize % n, (x - x.floor()) as f32);
        let (a, b) = (self.frames[i], self.frames[(i + 1) % n]);
        let v: [f32; CHANNELS] = core::array::from_fn(|c| {
            (if c < 3 { self.gyro_scale } else { self.motor_scale }) * (a[c] + f * (b[c] - a[c]))
        });
        NoiseSample { gyro: [v[0], v[1], v[2]], motor: [v[3], v[4], v[5], v[6]] }
    }
}
