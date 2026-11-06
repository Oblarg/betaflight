//! Core types for the WhippyFlight control stack.
//!
//! All FFI-boundary types are `#[repr(C)]` so BF's C code can construct and
//! read them directly.  Internal types use whippyunits quantities and
//! whippyalgebra matrices for compile-time dimensional safety.
//!
//! # Units
//!
//! The Rust layer uses clean SI-adjacent units throughout:
//!
//! | Quantity       | Rust unit    | BF native unit          |
//! |----------------|-------------|-------------------------|
//! | Angular rate   | rot/s       | deg/s                   |
//! | Angle          | rot         | decidegrees             |
//! | Voltage        | V (f32)     | centivolts (u16)        |
//! | Motor speed    | rot/s       | eRPM/100 (i16)          |
//! | Motor output   | duty [0,1]  | DShot 48–2047           |
//! | Throttle       | [0, 1]      | [0, 1] (same)           |
//!
//! Conversion from BF's native units to ours happens in the C glue layer
//! that calls the Rust FFI functions.  See [`BF_DPS_TO_RPS`] and friends
//! for the conversion constants (defined here so they can be exported via
//! a C header).
//!
//! # Axis ordering
//!
//! All 3-axis arrays follow Betaflight's axis convention:
//!
//!   **\[roll=0, pitch=1, yaw=2\]**
//!
//! This matches BF's `FD_ROLL=0, FD_PITCH=1, FD_YAW=2` from `axis.h`.
//!
//! # Motor ordering
//!
//! All 4-motor arrays follow Betaflight's QUADX mixer convention:
//!
//!   **\[motor0=REAR_R, motor1=FRONT_R, motor2=REAR_L, motor3=FRONT_L\]**
//!
//! This matches `mixerQuadX[]` in `mixer_init.c`.
//!
//! Key layout guarantee: `Quantity<U, f32>` is `#[repr(transparent)]` over
//! `f32`, and `MixedUnitMatrix` carries only `PhantomData` alongside its inner
//! `SMatrix`.  Everything erases to raw floats at compile time, so the
//! `#[repr(C)]` structs below have the same layout as their C declarations
//! (plain `float` arrays) even though every field is a typed quantity.
//!
//! Dimensionless values (duty, throttle, scales, ratios, filter
//! coefficients) stay plain `f32`.  So does the accelerometer reading: it is
//! specific force as a multiple of standard gravity g₀, a ratio, because
//! g₀ = 9.80665 m/s² is not a power-of-2·3·5·π multiple of m/s² and so has
//! no storage unit.

use whippyalgebra::dims;
use whippyalgebra::nalgebra::{MixedUnitMatrix, UniformUnitMatrix};
use nalgebra::SMatrix;
use nalgebra::{Const, Matrix, U1};
use whippyalgebra::nalgebra::__backend::RawStorage;
use whippyalgebra::{CollapseUniform, DimList, FromRaw, GaugeReproduces, Uniform};
use whippyunits::{qty, DivUnit, Quantity, UnitDiv};

// Re-export for use in other modules
pub use whippyalgebra::nalgebra as nalgebra;

// ---------------------------------------------------------------------------
// BF ↔ Rust conversion constants
//
// These are provided so the C glue layer (and unit tests) can convert from
// Betaflight's native units to WhippyFlight's clean units.  Each is the
// value of one BF count: multiply the BF value by it.
// ---------------------------------------------------------------------------

/// One BF gyro/setpoint count (1 deg/s), in rot/s.
pub const BF_DPS_TO_RPS: RotPerSec = RotPerSec::new(1.0 / 360.0);

/// One BF battery count (1 centivolt), in volts.
pub const BF_CENTIVOLTS_TO_V: Volt = Volt::new(0.01);

/// One BF eRPM/100 telemetry count, in eRPM.  Mechanical rot/s is
/// `erpm_lsb * BF_ERPM_LSB * 2.0 / pole_count / 60.0`.
pub const BF_ERPM_LSB: f32 = 100.0;

/// One BF eRPM/100 telemetry count, in model motor speed (eRPM/600).
///
/// The regression (main.R) defines motor speed as the blackbox eRPM field
/// divided by 6, so all fitted coefficients and the gain grid are in this
/// unit regardless of motor pole count.
pub const BF_ERPM_LSB_TO_MODEL_RPS: RotPerSec = RotPerSec::new(1.0 / 6.0);

// ---------------------------------------------------------------------------
// Dimension lists
// ---------------------------------------------------------------------------

/// Attitude rate axes: [roll, pitch, yaw] in rot/s.
pub type AttitudeAxes = dims![rot / s, rot / s, rot / s];

/// Integrated attitude error axes: [roll, pitch, yaw] in rot.
pub type IntegralAxes = dims![rot, rot, rot];

/// Derivative error axes: [roll, pitch, yaw] in rot/s².
pub type DerivativeAxes = dims![rot / s ^ 2, rot / s ^ 2, rot / s ^ 2];

/// Motor voltage outputs: [REAR_R, FRONT_R, REAR_L, FRONT_L] in V.
pub type MotorAxes = dims![V, V, V, V];

/// Motor speeds: one per motor, in rot/s (model units).
pub type MotorSpeedAxes = dims![rot / s, rot / s, rot / s, rot / s];

/// Signed squared motor speeds ω|ω|: one per motor, in rot²/s².
pub type MotorSpeedSqAxes = dims![rot ^ 2 / s ^ 2, rot ^ 2 / s ^ 2, rot ^ 2 / s ^ 2, rot ^ 2 / s ^ 2];

/// Dimensionless column (single column vector tag).
pub type Scalar = dims![1];

// ---------------------------------------------------------------------------
// Gain matrix types
// ---------------------------------------------------------------------------

/// Proportional gain Kp: 4 motors × 3 axes.
/// Entry (i,j): V / (rot/s) = V·s/rot.
pub type KpMatrix = MixedUnitMatrix<MotorAxes, AttitudeAxes, SMatrix<f32, 4, 3>>;

/// Integral gain Ki: 4 motors × 3 axes.
/// Entry (i,j): V / rot.
pub type KiMatrix = MixedUnitMatrix<MotorAxes, IntegralAxes, SMatrix<f32, 4, 3>>;

/// Derivative gain Kd: 4 motors × 3 axes.
/// Entry (i,j): V / (rot/s²) = V·s²/rot.
pub type KdMatrix = MixedUnitMatrix<MotorAxes, DerivativeAxes, SMatrix<f32, 4, 3>>;

/// Feedforward gain B⁻¹: 4 motors × 3 axes.
/// Entry (i,j): V / (rot/s) = V·s/rot.
pub type BInvMatrix = MixedUnitMatrix<MotorAxes, AttitudeAxes, SMatrix<f32, 4, 3>>;

/// Reference-model rate ← motor-speed coupling: 3 axes × 4 motors.
/// Entry (i,j): (rot/s) / (rot/s) — dimensionless.
pub type RateFromSpeedMatrix = MixedUnitMatrix<AttitudeAxes, MotorSpeedAxes, SMatrix<f32, 3, 4>>;

/// Reference-model rate ← voltage over one tick: 3 axes × 4 motors.
/// Entry (i,j): (rot/s) / V.
pub type RateFromVoltMatrix = MixedUnitMatrix<AttitudeAxes, MotorAxes, SMatrix<f32, 3, 4>>;

/// Affine linearizer G = −B⁺·P: 4 motors × 4 signed squared motor speeds.
/// Entry (i,j): V / (rot²/s²) = V·s²/rot².
pub type AffineMatrix = MixedUnitMatrix<MotorAxes, MotorSpeedSqAxes, SMatrix<f32, 4, 4>>;

// ---------------------------------------------------------------------------
// State vector types
// ---------------------------------------------------------------------------

/// Proportional error vector: 3 attitude rates in rot/s.
pub type PropErrorVec = MixedUnitMatrix<AttitudeAxes, Scalar, SMatrix<f32, 3, 1>>;

/// Integral error vector: 3 integrated errors in rot.
pub type IntegralErrorVec = MixedUnitMatrix<IntegralAxes, Scalar, SMatrix<f32, 3, 1>>;

/// Derivative error vector: 3 rate derivatives in rot/s².
pub type DerivErrorVec = MixedUnitMatrix<DerivativeAxes, Scalar, SMatrix<f32, 3, 1>>;

/// Setpoint diff vector: 3 rate changes in rot/s.
pub type SetpointDiffVec = MixedUnitMatrix<AttitudeAxes, Scalar, SMatrix<f32, 3, 1>>;

/// Motor voltage vector: 4 motor voltages in V.
pub type MotorVoltageVec = MixedUnitMatrix<MotorAxes, Scalar, SMatrix<f32, 4, 1>>;

/// Motor speed deviation vector in rot/s (model units).
pub type MotorSpeedVec = MixedUnitMatrix<MotorSpeedAxes, Scalar, SMatrix<f32, 4, 1>>;

/// Signed squared motor speed vector ω|ω| in rot²/s².
pub type MotorSpeedSqVec = MixedUnitMatrix<MotorSpeedSqAxes, Scalar, SMatrix<f32, 4, 1>>;

// ---------------------------------------------------------------------------
// Scalar quantity aliases
// ---------------------------------------------------------------------------

/// A rotation rate in rot/s (also motor speed, in model units).
pub type RotPerSec = qty!(rot / s, f32);
/// A rotation (integrated angle) in rot.
pub type Rotation = qty!(rot, f32);
/// Angular acceleration in rot/s².
pub type RotPerSecSq = qty!(rot / s ^ 2, f32);
/// Angular jerk in rot/s³.
pub type RotPerSecCu = qty!(rot / s ^ 3, f32);
/// A signed squared motor speed ω|ω| in rot²/s².
pub type RotSqPerSecSq = qty!(rot ^ 2 / s ^ 2, f32);
/// A voltage in V.
pub type Volt = qty!(V, f32);
/// A time duration in s.
pub type Seconds = qty!(s, f32);
/// A timestamp on BF's free-running microsecond clock (wraps).
pub type Micros = qty!(us, u32);
/// A frequency in Hz.
pub type Hertz = qty!(Hz, f32);
/// Voltage per rotation rate, V·s/rot (Kp, G_h⁺, motor drag k_ω).
pub type VoltPerRps = qty!(V * s / rot, f32);
/// Voltage per signed squared speed, V·s²/rot² (G, motor drag k_ω²).
pub type VoltPerRpsSq = qty!(V * s ^ 2 / rot ^ 2, f32);
/// Voltage per rotation, V/rot (Ki).
pub type VoltPerRot = qty!(V / rot, f32);
/// Voltage per angular acceleration, V·s²/rot (Kd).
pub type VoltPerRpsPerSec = qty!(V * s ^ 2 / rot, f32);
/// Rotation rate per volt over one tick, rot/(s·V) (the reference model's Bd).
pub type RpsPerVolt = qty!(rot / s / V, f32);

// ---------------------------------------------------------------------------
// Vector ↔ array conversion
// ---------------------------------------------------------------------------

/// A uniform column vector (e.g. [`PropErrorVec`], [`MotorVoltageVec`]) from
/// its entries; the expected type picks the dimension lists, whose entry
/// unit must be the entries'.
pub fn from_array<U, Rows, Cols, const N: usize>(
    entries: [Quantity<U, f32>; N],
) -> MixedUnitMatrix<Rows, Cols, SMatrix<f32, N, 1>>
where
    U: Clone,
    Rows: DimList + CollapseUniform,
    Cols: DimList + CollapseUniform,
    Uniform<Rows>: UnitDiv<Uniform<Cols>>,
    DivUnit<Uniform<Rows>, Uniform<Cols>>: GaugeReproduces<U>,
{
    UniformUnitMatrix::<U, SMatrix<f32, N, 1>>::from_fn(|i, _| entries[i]).gauge()
}

/// The entries of a uniform column vector.
pub fn to_array<Rows, Cols, S, const N: usize>(
    v: &MixedUnitMatrix<Rows, Cols, Matrix<f32, Const<N>, U1, S>>,
) -> [Quantity<DivUnit<Uniform<Rows>, Uniform<Cols>>, f32>; N]
where
    Rows: CollapseUniform,
    Cols: CollapseUniform,
    Uniform<Rows>: UnitDiv<Uniform<Cols>>,
    S: RawStorage<f32, Const<N>, U1>,
    Quantity<DivUnit<Uniform<Rows>, Uniform<Cols>>, f32>: FromRaw<f32>,
{
    let u = v.as_uniform();
    core::array::from_fn(|i| u.get(i, 0))
}

// ---------------------------------------------------------------------------
// FFI-boundary types (C-compatible)
// ---------------------------------------------------------------------------

/// Sensor data passed from BF to the Rust control stack every loop iteration.
///
/// **All values must be in WhippyFlight units (rot/s, V) — NOT BF native
/// units.**  The C glue layer is responsible for converting from BF's
/// deg/s, centivolts, eRPM etc. before filling this struct.
///
/// Axis order: `[roll, pitch, yaw]` (matches BF's `FD_ROLL=0, FD_PITCH=1, FD_YAW=2`).
/// Motor order: `[REAR_R, FRONT_R, REAR_L, FRONT_L]` (matches BF's QUADX mixer).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct SensorData {
    /// Gyro rates in rot/s: [roll, pitch, yaw].
    ///
    /// C glue converts from BF's `gyro.gyroADCf[axis]` (deg/s):
    /// `gyro_rps[i] = gyro.gyroADCf[i] * BF_DPS_TO_RPS`
    pub gyro_rps: [RotPerSec; 3],
    /// Motor speeds in the regression's speed unit (eRPM/600), by BF motor
    /// index.  This equals mechanical rot/s only for 20-pole motors.
    ///
    /// C glue converts from BF's DShot eRPM telemetry (eRPM/100 per LSB):
    /// `motor_rps[i] = getDshotErpm(i) * BF_ERPM_LSB_TO_MODEL_RPS`
    pub motor_rps: [RotPerSec; 4],
    /// Battery voltage in volts.
    ///
    /// C glue converts from BF's `getBatteryVoltage()` (centivolts):
    /// `battery_voltage = getBatteryVoltage() * BF_CENTIVOLTS_TO_V`
    pub battery_voltage: Volt,
    /// Body-frame specific force in g₀ (a ratio, see the module docs):
    /// [x, y, z] (BF accelerometer axes).
    /// Only read by the airframe drag mode; zero without an accelerometer.
    ///
    /// C glue converts from BF's aligned `acc.accADC[axis]` (LSB):
    /// `accel_g[i] = acc.accADC[i] / acc.dev.acc_1G`
    pub accel_g: [f32; 3],
}

/// Per-tick setpoint for [`crate::controller::ControllerState`].
///
/// **All values must be in WhippyFlight units — NOT BF native units.**
///
/// Axis order: `[roll, pitch, yaw]`.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct SetpointData {
    /// Desired attitude rates in rot/s: [roll, pitch, yaw].
    ///
    /// On the firmware path this is the [`crate::setpoint`] estimate r(t),
    /// built from BF's raw setpoint packets (`getRawSetpoint(axis)` deg/s
    /// × `BF_DPS_TO_RPS`).
    pub rate_rps: [RotPerSec; 3],
    /// Throttle as a fraction [0, 1].
    ///
    /// Direct from `rcCommand[THROTTLE]` (already [0,1] in BF).
    pub throttle: f32,
}

/// Motor output returned from the Rust control stack to BF.
///
/// Motor order: `[REAR_R, FRONT_R, REAR_L, FRONT_L]` (matches BF's QUADX mixer).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct ControlOutput {
    /// Motor duty cycle fractions [0, 1]: [REAR_R, FRONT_R, REAR_L, FRONT_L].
    ///
    /// C glue maps to DShot range:
    /// `motor[i] = motorOutputLow + motor_duty[i] * (motorOutputHigh - motorOutputLow)`
    pub motor_duty: [f32; 4],
}

/// Telemetry data for blackbox logging.
///
/// Axis order: `[roll, pitch, yaw]`.
/// Motor order: `[REAR_R, FRONT_R, REAR_L, FRONT_L]`.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct LqrTelemetry {
    /// Per-motor feedback voltage: [REAR_R, FRONT_R, REAR_L, FRONT_L].
    pub feedback_v: [Volt; 4],
    /// Per-motor feedforward voltage: [REAR_R, FRONT_R, REAR_L, FRONT_L].
    pub feedforward_v: [Volt; 4],
    /// Per-motor affine linearizer voltage: [REAR_R, FRONT_R, REAR_L, FRONT_L].
    pub affine_v: [Volt; 4],
    /// Proportional error: [roll, pitch, yaw] in rot/s.
    pub error_prop: [RotPerSec; 3],
    /// Integral error state: [roll, pitch, yaw] in rot.
    pub error_integral: [Rotation; 3],
    /// Derivative error state: [roll, pitch, yaw] in rot/s².
    pub error_derivative: [RotPerSecSq; 3],
    /// Interpolated Kp diagonal (motor 0 / REAR_R): [roll, pitch, yaw].
    pub kp_motor0: [VoltPerRps; 3],
    /// Collective along the thrust-balanced direction (V, motor mean).
    pub collective_v: Volt,
    /// Attitude command scale applied by the allocator; 1 unless saturated.
    pub command_scale: f32,
    /// Rate setpoint the controller tracked: [roll, pitch, yaw] in rot/s.
    pub setpoint_rps: [RotPerSec; 3],
    /// Feedforward's h-tick-ahead target: [roll, pitch, yaw] in rot/s.
    pub ff_target_rps: [RotPerSec; 3],
    /// Status: 0 = no gains, 1 = gains loaded, 2 = gains corrupt.
    pub status: u8,
}

/// Result of gain loading.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainLoadResult {
    Ok = 0,
    TooShort = 1,
    BadMagic = 2,
    BadChecksum = 3,
    BadVersion = 4,
    ResolutionTooLarge = 5,
    /// Blob is not 4-byte aligned; tables are read from it in place.
    Misaligned = 6,
    /// Blob is larger than the target's RAM buffer (reported by the C glue).
    TooLarge = 7,
    /// No dataflash or gain partition on this target (reported by the C glue).
    NoStore = 8,
}
