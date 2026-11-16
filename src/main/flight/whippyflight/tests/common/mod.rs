//! Shared test helpers.
#![allow(dead_code, unused_imports)]

pub use whippyflight_control::gains::static_blob;
use whippyflight_control::types::{Micros, Seconds};
use whippyunits::value;

/// Write `value` into a blob at byte offset `at`, in its storage unit (as
/// `load_gains` reads it back).
pub fn put<Q: bytemuck::Pod>(buf: &mut [u8], at: usize, value: Q) {
    let bytes = bytemuck::bytes_of(&value);
    buf[at..at + bytes.len()].copy_from_slice(bytes);
}

/// The controller's microsecond clock at time `t` (truncated).
pub fn micros(t: Seconds) -> Micros {
    Micros::new(value!(t, us, f32) as u32)
}
