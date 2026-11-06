//! WhippyFlight — LQR flight control stack for Betaflight.
//!
//! This crate implements a gain-scheduled LQR controller for multirotor
//! attitude rate control.  It is compiled as a C-compatible static library
//! (`staticlib`) and linked into the Betaflight firmware image.
//!
//! # Architecture
//!
//! Betaflight is the HAL: it reads sensors, drives motors, handles blackbox /
//! MSP / OSD.  This crate is a pure control law — it takes sensor data in and
//! produces motor voltage commands out.  It owns no hardware.
//!
//! The control law uses precomputed gain matrices (generated offline by solving
//! the discrete-time algebraic Riccati equation at a grid of operating points)
//! that are uploaded to onboard flash and loaded at boot.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod allocation;
pub mod controller;
mod ffi;
pub mod gains;
pub mod interpolation;
pub mod setpoint;
pub mod types;

// Panic handler for embedded — required by no_std staticlib builds.
// In release builds panics are compiled out (panic = "abort" + LTO).
#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
