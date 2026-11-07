//! Core types for the WhippyFlight control stack.
//!
//! FFI-boundary types are `#[repr(C)]` so BF's C code can construct and read
//! them directly.  Internal types use whippyunits quantities and
//! whippyalgebra matrices for compile-time dimensional safety.
//!
//! # Units
//!
//! The Rust layer uses clean SI-adjacent units throughout:
//!
//! | Quantity       | Rust unit    | BF native unit                  |
//! |----------------|-------------|---------------------------------|
//! | Angular rate   | rot/s       | deg/s ([`DegPerSec`])           |
//! | Voltage        | V (f32)     | centivolts ([`CentiVolt`])      |
//! | Motor speed    | model rot/s | eRPM/100 ([`HundredErpm`]), RPM/100 ([`HundredRpm`]) |
//! | RX rate        | period (s)  | Hz ([`Hertz`])                  |
//! | Motor output   | duty [0,1]  | DShot 48–2047                   |
//! | Throttle       | [0, 1]      | [0, 1] (same)                   |
//! | Specific force | m/s²        | g₀ (`accADC / acc_1G`, a count of `g_n`) |
//!
//! The FFI takes BF's values in BF's own units (see [`BfSensorData`]), typed,
//! and hands telemetry back in the units BF logs it in ([`BfTelemetry`]), so
//! the C glue copies values through unconverted and every conversion happens
//! here.  Electrical and mechanical motor speeds carry different brands
//! ([`Electrical`], [`Mechanical`]), and neither mixes with the model's.
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
//! coefficients) stay plain `f32`.  BF reads the accelerometer in standard
//! gravities g₀; g₀ = 9.80665 m/s² is not a power-of-2·3·5·π multiple of
//! m/s², so it is a non-storage unit (`g_n`, stored as 0.980665 dam/s²), and
//! the reading is declared in it on the way in ([`accel_from_g`]).

use whippyalgebra::dims;
use whippyalgebra::nalgebra::{MixedUnitMatrix, UniformUnitMatrix};
use nalgebra::SMatrix;
use nalgebra::{Const, Matrix, U1};
use whippyalgebra::nalgebra::__backend::RawStorage;
use whippyalgebra::{CollapseUniform, DimList, FromRaw, GaugeReproduces, Uniform};
use whippyunits::api::rescale_f32;
use whippyunits::{qty, quantity, value, DivUnit, Quantity, UnitDiv};
use bytemuck::TransparentWrapper;

// Re-export for use in other modules
pub use whippyalgebra::nalgebra as nalgebra;

// ---------------------------------------------------------------------------
// BF-native units at the FFI boundary
//
// The C glue hands over BF's values as BF stores them; these types say what
// unit that is, and the conversions below turn them into ours.
// ---------------------------------------------------------------------------

/// BF's angular rate unit (`gyro.gyroADCf`, `getRawSetpoint`).
pub type DegPerSec = qty!(deg / s, f32);

/// BF's battery voltage unit (`getBatteryVoltage()`).
pub type CentiVolt = qty!(cV, f32);

/// Brand of an electrical motor speed: the commutation rate, pole pairs
/// times the shaft's.
#[derive(Clone, Copy)]
pub struct Electrical;

/// Brand of a mechanical (shaft) motor speed.
#[derive(Clone, Copy)]
pub struct Mechanical;

/// BF's DShot telemetry speed, eRPM/100 (`getDshotErpm`).
pub type HundredErpm = qty!(100 * rot / min, f32, Electrical);

/// BF's mechanical speed count, RPM/100 (`dyn_idle_min_rpm`).
pub type HundredRpm = qty!(100 * rot / min, f32, Mechanical);

/// Pole pairs of the motor whose mechanical rot/s is the model's motor speed
/// unit.  The regression (main.R) defines motor speed as the blackbox eRPM/100
/// field divided by 6, i.e. eRPM/600 = (electrical rot/s) / 10, whatever the
/// actual pole count; the fitted coefficients and gain grid are in this unit.
pub const MODEL_POLE_PAIRS: f32 = 10.0;

/// Model motor speed from electrical (eRPM) telemetry.
pub fn model_speed_from_erpm(erpm: HundredErpm) -> RotPerSec {
    RotPerSec::new(value!(erpm, rot / s, f32, Electrical) / MODEL_POLE_PAIRS)
}

/// Electrical speed of a `pole_count`-pole motor turning at `rpm`.
pub fn erpm_from_mechanical(rpm: HundredRpm, pole_count: u8) -> HundredErpm {
    HundredErpm::new(value!(rpm, 100 * rot / min, f32, Mechanical) * (pole_count as f32 / 2.0))
}

/// Model motor speed from a mechanical speed on a `pole_count`-pole motor.
pub fn model_speed_from_mechanical(rpm: HundredRpm, pole_count: u8) -> RotPerSec {
    model_speed_from_erpm(erpm_from_mechanical(rpm, pole_count))
}

// ---------------------------------------------------------------------------
// BF-native units of the telemetry going back out
//
// Blackbox logs `pidData[axis]` as `lrintf` of the float, so those fields
// are floats in the unit of one logged LSB.  `debug[]` is `int16_t`, so the
// debug fields are rounded here ([`to_debug`]); build/debug.h annotates each
// with the same `[unit:…]`.
// ---------------------------------------------------------------------------

/// `pidData` P (rate error): 0.1 deg/s per logged count.
pub type DeciDegPerSec = qty!(0.1 * deg / s, f32);
/// `pidData` I (integrated rate error): 0.1 deg per logged count.
pub type DeciDeg = qty!(0.1 * deg, f32);
/// `pidData` D (filtered rate-error derivative): 1 deg/s² per logged count.
pub type DegPerSecSq = qty!(deg / s ^ 2, f32);

/// `debug[]` rate, `[unit:0.1dps]`.
pub type DebugDeciDps = qty!(0.1 * deg / s, i16);
/// `debug[]` voltage, `[unit:0.001V]`.
pub type DebugMilliVolt = qty!(mV, i16);
/// `debug[]` voltage, `[unit:0.01V]`.
pub type DebugCentiVolt = qty!(cV, i16);
/// `debug[]` ratio, `[unit:0.001]`.
pub type DebugPermille = qty!(0.001, i16);

/// Round to the nearest LSB of an `int16_t` debug field, saturating; NaN
/// logs as 0.  The unit is unchanged, only the storage.
pub fn to_debug<U, B>(q: Quantity<U, f32, B>) -> Quantity<U, i16, B> {
    Quantity::wrap(libm::roundf(Quantity::peel(q)) as i16)
}

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
/// Specific force (what an accelerometer measures) in m/s².
pub type MetersPerSecSq = qty!(m / s ^ 2, f32);
/// What a `g_n` declaration stores: 1 g₀ is 0.980665 dam/s².
pub type DecaMetersPerSecSq = qty!(dam / s ^ 2, f32);
/// Angular acceleration per specific force, rot/m (airframe drag).
pub type RotPerMeter = qty!(rot / m, f32);

/// Standard gravity g₀.
pub const STANDARD_GRAVITY: MetersPerSecSq = mps2(quantity!(1.0, g_n, f32));

/// m/s² of a quantity declared in g₀.
pub const fn mps2(a: DecaMetersPerSecSq) -> MetersPerSecSq {
    rescale_f32(a)
}

/// Specific force from an accelerometer reading in g₀.
pub const fn accel_from_g(g: f32) -> MetersPerSecSq {
    mps2(quantity!(g, g_n, f32))
}

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
// Controller I/O; the `#[repr(C)]` ones cross the FFI
// ---------------------------------------------------------------------------

/// Sensor data as BF hands it over every loop iteration, in BF's units.
///
/// Axis order: `[roll, pitch, yaw]` (matches BF's `FD_ROLL=0, FD_PITCH=1, FD_YAW=2`).
/// Motor order: `[REAR_R, FRONT_R, REAR_L, FRONT_L]` (matches BF's QUADX mixer).
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct BfSensorData {
    /// `gyro.gyroADCf[axis]`: [roll, pitch, yaw].
    pub gyro: [DegPerSec; 3],
    /// `getDshotErpm(motor)`, electrical, by BF motor index.
    pub motor_erpm: [HundredErpm; 4],
    /// `getBatteryVoltage()`.
    pub battery: CentiVolt,
    /// Body-frame specific force, a count of g₀ (`g_n`, see the module
    /// docs): [x, y, z] (BF accelerometer axes), `acc.accADC[i] /
    /// acc.dev.acc_1G`.  Zero without an accelerometer.
    pub accel_g: [f32; 3],
}

impl BfSensorData {
    /// The same readings in WhippyFlight's units.
    pub fn to_sensor_data(&self) -> SensorData {
        SensorData {
            gyro_rps: self.gyro.map(rescale_f32),
            motor_rps: self.motor_erpm.map(model_speed_from_erpm),
            battery_voltage: rescale_f32(self.battery),
            accel: self.accel_g.map(accel_from_g),
        }
    }
}

/// Sensor data in WhippyFlight's units, as the controller consumes it.
///
/// Axis order: `[roll, pitch, yaw]`; motor order: `[REAR_R, FRONT_R, REAR_L,
/// FRONT_L]` (see [`BfSensorData`]).
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct SensorData {
    /// Gyro rates: [roll, pitch, yaw].
    pub gyro_rps: [RotPerSec; 3],
    /// Motor speeds in the model's unit (see [`MODEL_POLE_PAIRS`]), by BF
    /// motor index.
    pub motor_rps: [RotPerSec; 4],
    /// Battery voltage.
    pub battery_voltage: Volt,
    /// Body-frame specific force: [x, y, z] (BF accelerometer axes).
    pub accel: [MetersPerSecSq; 3],
}

/// Per-tick setpoint for [`crate::controller::ControllerState`].
///
/// Axis order: `[roll, pitch, yaw]`.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct SetpointData {
    /// Desired attitude rates in rot/s: [roll, pitch, yaw].
    ///
    /// On the firmware path this is the [`crate::setpoint`] estimate r(t),
    /// built from BF's raw setpoint packets.
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

/// The controller's internal state for one tick, in WhippyFlight's units.
/// [`LqrTelemetry::to_bf`] encodes what BF logs of it.
///
/// Axis order: `[roll, pitch, yaw]`.
/// Motor order: `[REAR_R, FRONT_R, REAR_L, FRONT_L]`.
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

impl LqrTelemetry {
    /// The logged fields, encoded in BF's units; `raw_setpoint` is the last
    /// RX packet's rate setpoint.
    pub fn to_bf(&self, raw_setpoint: [RotPerSec; 3]) -> BfTelemetry {
        let debug_rate = |r: RotPerSec| -> DebugDeciDps { to_debug(rescale_f32(r)) };
        BfTelemetry {
            pid_p: self.error_prop.map(rescale_f32),
            pid_i: self.error_integral.map(rescale_f32),
            pid_d: self.error_derivative.map(rescale_f32),
            feedback: self.feedback_v.map(|v| to_debug(rescale_f32(v))),
            raw_setpoint: raw_setpoint.map(debug_rate),
            setpoint: self.setpoint_rps.map(debug_rate),
            ff_target: self.ff_target_rps.map(debug_rate),
            feedforward_motor0: to_debug(rescale_f32(self.feedforward_v[0])),
            command_scale: to_debug(rescale_f32(<qty!(1, f32)>::new(self.command_scale))),
        }
    }
}

/// What BF logs of [`LqrTelemetry`], in the units it logs it in; the C glue
/// copies each field into its blackbox or debug slot unconverted.
///
/// Axis order: `[roll, pitch, yaw]`.
/// Motor order: `[REAR_R, FRONT_R, REAR_L, FRONT_L]`.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct BfTelemetry {
    /// `pidData[axis].P` and `.Sum`: rate error.
    pub pid_p: [DeciDegPerSec; 3],
    /// `pidData[axis].I`: integral of the rate error.
    pub pid_i: [DeciDeg; 3],
    /// `pidData[axis].D`: filtered derivative of the rate error.
    pub pid_d: [DegPerSecSq; 3],
    /// `DEBUG_NONE` 0–3: per-motor feedback voltage.
    pub feedback: [DebugMilliVolt; 4],
    /// `DEBUG_WHIPPYFLIGHT`: the raw RX setpoint, the estimate the controller
    /// tracked, and the feedforward target.
    pub raw_setpoint: [DebugDeciDps; 3],
    pub setpoint: [DebugDeciDps; 3],
    pub ff_target: [DebugDeciDps; 3],
    /// `DEBUG_WHIPPYFLIGHT`: motor 0 feedforward voltage.
    pub feedforward_motor0: DebugCentiVolt,
    /// `DEBUG_WHIPPYFLIGHT`: allocator attitude command scale.
    pub command_scale: DebugPermille,
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
