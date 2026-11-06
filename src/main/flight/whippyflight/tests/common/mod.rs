//! Shared test helpers.
#![allow(dead_code)]

pub mod noise;
pub mod plant;
pub mod units;

use whippyflight_control::gains::AirframeDrag;
use whippyflight_control::types::{Micros, Seconds};
use whippyunits::value;

/// Copy `bytes` into leaked, 4-byte-aligned memory.  `load_gains` reads the
/// tables in place, so blobs must be `'static` and aligned like the
/// firmware's RAM copy of the gain partition.
pub fn static_blob(bytes: &[u8]) -> &'static [u8] {
    let words: &'static mut [u32] = vec![0u32; bytes.len().div_ceil(4)].leak();
    // SAFETY: the u32 buffer covers at least `bytes.len()` bytes and is leaked.
    let buf = unsafe {
        core::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, bytes.len())
    };
    buf.copy_from_slice(bytes);
    buf
}

/// Write `value` into a blob at byte offset `at`, in its storage unit (as
/// `load_gains` reads it back).
pub fn put<Q: bytemuck::Pod>(buf: &mut [u8], at: usize, value: Q) {
    let bytes = bytemuck::bytes_of(&value);
    buf[at..at + bytes.len()].copy_from_slice(bytes);
}

/// A copy of the v7 blob `v7` with its airframe drag block replaced by
/// `drag`, CRC updated.
pub fn with_airframe_drag(v7: &[u8], drag: AirframeDrag) -> Vec<u8> {
    assert_eq!(u16::from_le_bytes([v7[4], v7[5]]), 7, "expected a v7 blob");
    // Header, then quant lo/step for 4 gain tables (12) and the ref table
    // (50), affine (16), motor (2), balance (4).
    let at = 0x20 + 4 * (2 * (4 * 12 + 50) + 16 + 2 + 4);
    let mut out = v7[..v7.len() - 4].to_vec();
    put(&mut out, at, drag.lp);
    for (i, c) in drag.roll.into_iter().chain(drag.pitch).enumerate() {
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

/// [`micros`] for the f64 plant clock.
pub fn micros64(t: units::Time) -> Micros {
    Micros::new(value!(t, us) as u32)
}
