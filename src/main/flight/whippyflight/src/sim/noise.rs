//! Replay of measured sensor noise, as tools/extract_noise.py writes it from
//! a blackbox log: BF's filtered gyro and the eRPM telemetry (as shaft
//! speeds, by the log's `motor_poles`), high-passed to
//! remove the body's own motion, looped.  The gyro noise was taken after BF's
//! filters, so it belongs on top of the gyro the controller is handed.

use super::units::{Freq, Time};
use crate::types::RotPerSec;

const GYRO_LSB: RotPerSec = RotPerSec::new(1e-5);
const ERPM_LSB: RotPerSec = RotPerSec::new(1e-3);
const CHANNELS: usize = 7;
const HEADER: usize = 8;

pub struct Noise {
    fs: Freq,
    /// Gyro [roll, pitch, yaw] then motor speeds [0..4].
    frames: Vec<[RotPerSec; CHANNELS]>,
    offset: Time,
    gyro_scale: f32,
    motor_scale: f32,
}

pub struct NoiseSample {
    /// BF gyro axes [roll, pitch, yaw].
    pub gyro: [RotPerSec; 3],
    /// Motor speed telemetry, shaft rev/s.
    pub motor: [RotPerSec; 4],
}

impl Noise {
    /// Parse extract_noise.py's output, starting `offset` into the recording
    /// (it loops), with the gyro and motor-speed noise scaled separately.
    pub fn from_bytes(
        bytes: &[u8],
        offset: Time,
        gyro_scale: f32,
        motor_scale: f32,
    ) -> Result<Self, String> {
        if bytes.len() < HEADER {
            return Err("noise file too short".into());
        }
        let rate = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        if !rate.is_finite() || rate <= 0.0 || n == 0 || bytes.len() != HEADER + 2 * CHANNELS * n {
            return Err(format!(
                "not a noise file: {rate} Hz, {n} frames in {} bytes",
                bytes.len()
            ));
        }
        let frames = (0..n)
            .map(|k| {
                core::array::from_fn(|c| {
                    let at = HEADER + 2 * (k * CHANNELS + c);
                    let lsb = if c < 3 { GYRO_LSB } else { ERPM_LSB };
                    i16::from_le_bytes([bytes[at], bytes[at + 1]]) as f32 * lsb
                })
            })
            .collect();
        Ok(Self {
            fs: Freq::new(rate as f64),
            frames,
            offset,
            gyro_scale,
            motor_scale,
        })
    }

    pub fn duration(&self) -> Time {
        self.frames.len() as f64 / self.fs
    }

    /// Noise at time `t`, linearly interpolated from the log rate.
    pub fn at(&self, t: Time) -> NoiseSample {
        let n = self.frames.len();
        let x = f64::from((t + self.offset) * self.fs).rem_euclid(n as f64);
        let (i, f) = (x as usize % n, (x - x.floor()) as f32);
        let (a, b) = (self.frames[i], self.frames[(i + 1) % n]);
        let v: [RotPerSec; CHANNELS] = core::array::from_fn(|c| {
            (if c < 3 {
                self.gyro_scale
            } else {
                self.motor_scale
            }) * (a[c] + f * (b[c] - a[c]))
        });
        NoiseSample {
            gyro: [v[0], v[1], v[2]],
            motor: [v[3], v[4], v[5], v[6]],
        }
    }
}
