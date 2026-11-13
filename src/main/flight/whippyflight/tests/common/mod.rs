//! Shared test helpers.
#![allow(dead_code)]

pub mod noise;
pub mod plant;
pub use whippyflight_control::sim::units;

#[allow(unused_imports)]
pub use whippyflight_control::sim::{flick::micros as micros64, static_blob};
use whippyflight_control::types::{Hertz, Micros, RotPerMeter, Seconds};
use whippyunits::value;

/// Write `value` into a blob at byte offset `at`, in its storage unit (as
/// `load_gains` reads it back).
pub fn put<Q: bytemuck::Pod>(buf: &mut [u8], at: usize, value: Q) {
    let bytes = bytemuck::bytes_of(&value);
    buf[at..at + bytes.len()].copy_from_slice(bytes);
}

/// A copy of the v8 blob `blob` with its airframe drag block replaced by
/// low-pass `lp` and coefficients `c` (`[roll←ax, roll←ay, pitch←ax,
/// pitch←ay]`), CRC updated.
pub fn with_airframe_drag(blob: &[u8], lp: Hertz, c: [RotPerMeter; 4]) -> Vec<u8> {
    assert_eq!(u16::from_le_bytes([blob[4], blob[5]]), 8, "expected a v8 blob");
    // Header, then quant lo/step for 4 gain tables (12), the ref table's
    // speed-free values (6) and per-motor lo/step (44), affine (16), motor
    // (2), balance (4).
    let at = 0x20 + 4 * (2 * 4 * 12 + 6 + 2 * 44 + 16 + 2 + 4);
    let mut out = blob[..blob.len() - 4].to_vec();
    put(&mut out, at, lp);
    for (i, c) in c.into_iter().enumerate() {
        put(&mut out, at + 4 * (1 + i), c);
    }
    let crc = crc32(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// The controller's microsecond clock at time `t` (truncated).
pub fn micros(t: Seconds) -> Micros {
    Micros::new(value!(t, us, f32) as u32)
}