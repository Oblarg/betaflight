//! WhippyFlight for Betaflight: the C ABI the firmware glue
//! (`whippyflight_glue.c`) calls, over [`whippyflight_control`].
//!
//! Compiled as a static library and linked into the Betaflight image.
//! Betaflight is the HAL: it reads sensors, drives motors, handles
//! blackbox / MSP / OSD, and stores the gain blob.  This crate holds the
//! controller's state for the firmware's single control loop and converts
//! Betaflight's units and conventions to and from the controller's
//! ([`bf`]).

#![cfg_attr(not(feature = "std"), no_std)]

pub mod bf;
pub mod ffi;

// Panic handler for embedded — required by no_std staticlib builds.
// In release builds panics are compiled out (panic = "abort" + LTO).
#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
