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

#![no_std]

mod controller;
mod ffi;
mod gains;
mod interpolation;
mod types;

// Panic handler for embedded — required by no_std staticlib builds.
// In release builds panics are compiled out (panic = "abort" + LTO).
#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
