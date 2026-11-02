//! Core types for the WhippyFlight control stack.
//!
//! All FFI-boundary types are `#[repr(C)]` so BF's C code can construct and
//! read them directly.  Internal types use whippyunits quantities for
//! compile-time dimensional safety.

use whippyalgebra::dims;
use whippyalgebra::nalgebra::MixedUnitMatrix;
use nalgebra::SMatrix;

// Re-export nalgebra for internal use (whippyalgebra re-exports it)
use whippyalgebra::nalgebra as nalgebra;

// ---------------------------------------------------------------------------
// Dimension lists for the control spaces
// ---------------------------------------------------------------------------

/// Attitude rate axes: [yaw, pitch, roll] in rot/s.
pub type AttitudeAxes = dims![rot / s, rot / s, rot / s];

/// Integrated attitude error axes: [yaw, pitch, roll] in rot (rotations).
pub type IntegralAxes = dims![rot, rot, rot];

/// Derivative error axes: [yaw, pitch, roll] in rot/s².
pub type DerivativeAxes = dims![rot / s ^ 2, rot / s ^ 2, rot / s ^ 2];

/// Motor voltage outputs: [fl, bl, fr, br] in V.
pub type MotorAxes = dims![V, V, V, V];

// ---------------------------------------------------------------------------
// Gain matrix types — row dims / col dims lock the entry units
// ---------------------------------------------------------------------------

/// Proportional gain: V / (rot/s) = V·s/rot per entry.
pub type KpMatrix = MixedUnitMatrix<MotorAxes, AttitudeAxes, SMatrix<f32, 4, 3>>;

/// Integral gain: V / rot per entry.
pub type KiMatrix = MixedUnitMatrix<MotorAxes, IntegralAxes, SMatrix<f32, 4, 3>>;

/// Derivative gain: V / (rot/s²) = V·s²/rot per entry.
pub type KdMatrix = MixedUnitMatrix<MotorAxes, DerivativeAxes, SMatrix<f32, 4, 3>>;

/// Feedforward gain (B⁻¹ partial): V·s/rot per entry.
pub type BInvMatrix = MixedUnitMatrix<MotorAxes, AttitudeAxes, SMatrix<f32, 4, 3>>;

// ---------------------------------------------------------------------------
// FFI-boundary types (C-compatible)
// ---------------------------------------------------------------------------

/// Sensor data passed from BF to the Rust control stack every loop iteration.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SensorData {
    /// Gyro rates in rotations/s: [yaw, pitch, roll].
    pub gyro_rps: [f32; 3],
    /// Motor speeds from RPM telemetry, in rotations/s: [fl, bl, fr, br].
    pub motor_rps: [f32; 4],
    /// Battery voltage in volts.
    pub battery_voltage: f32,
}

/// RC setpoint data passed from BF.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SetpointData {
    /// Desired attitude rates in rotations/s: [yaw, pitch, roll].
    pub rate_rps: [f32; 3],
    /// Throttle as a fraction [0, 1].
    pub throttle: f32,
}

/// Motor output returned from the Rust control stack to BF.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ControlOutput {
    /// Motor duty cycle fractions [0, 1]: [fl, bl, fr, br].
    pub motor_duty: [f32; 4],
}

/// Telemetry data for blackbox logging.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LqrTelemetry {
    /// Per-motor feedback voltage: [fl, bl, fr, br].
    pub feedback_v: [f32; 4],
    /// Per-motor feedforward voltage: [fl, bl, fr, br].
    pub feedforward_v: [f32; 4],
    /// Per-motor affine linearizer voltage: [fl, bl, fr, br].
    pub affine_v: [f32; 4],
    /// Proportional error: [yaw, pitch, roll] in rot/s.
    pub error_prop: [f32; 3],
    /// Integral error state: [yaw, pitch, roll] in rot.
    pub error_integral: [f32; 3],
    /// Derivative error state: [yaw, pitch, roll] in rot/s².
    pub error_derivative: [f32; 3],
    /// Interpolated Kp diagonal (motor 0): [yaw, pitch, roll].
    pub kp_motor0: [f32; 3],
    /// Status: 0 = no gains, 1 = gains loaded, 2 = gains corrupt.
    pub status: u8,
}

/// Result of gain loading.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainLoadResult {
    /// Gains loaded and validated successfully.
    Ok = 0,
    /// Gain data too short or truncated.
    TooShort = 1,
    /// Magic number mismatch.
    BadMagic = 2,
    /// CRC32 checksum mismatch.
    BadChecksum = 3,
    /// Unsupported format version.
    BadVersion = 4,
    /// Resolution too large for static buffers.
    ResolutionTooLarge = 5,
}
