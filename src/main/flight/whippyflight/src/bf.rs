//! Betaflight's units and conventions at the FFI boundary.
//!
//! The C glue hands over BF's values as BF stores them, typed in BF's
//! units ([`BfSensorData`]), and copies the telemetry coming back into its
//! blackbox and debug slots unconverted ([`BfTelemetry`]), so every
//! conversion happens here.
//!
//! | Quantity       | WhippyFlight | BF native unit                           |
//! |----------------|--------------|------------------------------------------|
//! | Angular rate   | rot/s        | deg/s ([`DegPerSec`])                    |
//! | Voltage        | V            | centivolts ([`CentiVolt`])               |
//! | Motor speed    | shaft rot/s  | eRPM/100 ([`HundredErpm`]) and `motor_poles` |
//! | RX rate        | period (s)   | Hz                                       |
//! | Motor output   | duty [0,1]   | DShot 48–2047                            |
//! | Specific force | m/s²         | g₀ (`accADC / acc_1G`)                   |
//!
//! BF's body axes are FLU (x forward, y left, z up: positive pitch is nose
//! down, positive yaw nose left); WhippyFlight's are FRD.  Every per-axis
//! value crossing the boundary goes through [`flu_frd`], both ways.  Motors
//! keep BF's motor index; the gain blob's layout must be the one BF's mixer
//! drives ([`quadx_layout`]).
//!
//! Quantity fields are `#[repr(transparent)]` over their storage, so the
//! `#[repr(C)]` structs here have the layout of their C declarations
//! (plain `float` arrays).  Electrical and mechanical motor speeds carry
//! different brands ([`Electrical`]), so eRPM doesn't mix with the shaft
//! speeds the controller uses.

use core::ops::Neg;

use bytemuck::TransparentWrapper;
use whippyflight_control::gains::{MotorLayout, MotorPosition, Spin};
use whippyflight_control::types::*;
use whippyunits::api::rescale_f32;
use whippyunits::{qty, value, Quantity};

/// BF's angular rate unit (`gyro.gyroADCf`, `getRawSetpoint`).
pub type DegPerSec = qty!(deg / s, f32);

/// BF's battery voltage unit (`getBatteryVoltage()`).
pub type CentiVolt = qty!(cV, f32);

/// Brand of an electrical motor speed: the commutation rate, pole pairs
/// times the shaft's.
#[derive(Clone, Copy)]
pub struct Electrical;

/// BF's DShot telemetry speed, eRPM/100 (`getDshotErpm`).
pub type HundredErpm = qty!(100 * rot / min, f32, Electrical);

/// Shaft speed of a `pole_count`-pole motor (`motor_poles`) from its
/// electrical (eRPM) telemetry.  The model's motor speeds, the fitted
/// coefficients and the gain grid are all in shaft rot/s.  A pole count
/// below 2 (unset) reads as 2: electrical speed itself.
pub fn mechanical_from_erpm(erpm: HundredErpm, pole_count: u8) -> RotPerSec {
    let pole_pairs = pole_count.max(2) as f32 / 2.0;
    RotPerSec::new(value!(erpm, rot / s, f32, Electrical) / pole_pairs)
}

/// A BF (FLU) axis triple in FRD, or an FRD one in FLU: the same flip of y
/// and z either way.
pub fn flu_frd<T: Neg<Output = T>>([x, y, z]: [T; 3]) -> [T; 3] {
    [x, -y, -z]
}

/// `mixerConfig()->mixerMode` of BF's quad X mixer (`MIXER_QUADX`).
pub const MIXER_QUADX: u8 = 3;

/// The motors BF's quad X mixer drives, by motor index (`mixerQuadX`):
/// props in, or props out with `yaw_motors_reversed`.
pub fn quadx_layout(yaw_motors_reversed: bool) -> MotorLayout {
    use MotorPosition::*;
    use Spin::*;
    let spin = if yaw_motors_reversed {
        [CounterClockwise, Clockwise, Clockwise, CounterClockwise]
    } else {
        [Clockwise, CounterClockwise, CounterClockwise, Clockwise]
    };
    MotorLayout { position: [RearRight, FrontRight, RearLeft, FrontLeft], spin }
}

// ---------------------------------------------------------------------------
// BF-native units of the telemetry going back out
//
// Blackbox logs `pidData[axis]` as `lrintf` of the float, so those fields
// are floats in the unit of one logged LSB.  `debug[]` is `int16_t`, so the
// debug fields are rounded here ([`to_debug`]); build/debug.h annotates each
// with the same `[unit:…]`.
// ---------------------------------------------------------------------------

/// `pidData` P (rate error) and I (integral state over dt): 0.1 deg/s per logged count.
pub type DeciDegPerSec = qty!(0.1 * deg / s, f32);
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

/// Sensor data as BF hands it over every loop iteration, in BF's units.
///
/// Axis order: `[roll, pitch, yaw]` (BF's `FD_ROLL=0, FD_PITCH=1, FD_YAW=2`),
/// FLU.  Motor order: BF's motor index (`[REAR_R, FRONT_R, REAR_L, FRONT_L]`
/// on QUADX).
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
    /// Body-frame specific force, a count of g₀: [x, y, z] (BF
    /// accelerometer axes), `acc.accADC[i] / acc.dev.acc_1G`.  Zero without
    /// an accelerometer.
    pub accel_g: [f32; 3],
    /// `motorConfig()->motorPoleCount`, which turns `motor_erpm` into shaft speeds.
    pub motor_poles: u8,
}

impl BfSensorData {
    /// The same readings in WhippyFlight's units and FRD axes.
    pub fn to_sensor_data(&self) -> SensorData {
        SensorData {
            gyro_rps: flu_frd(self.gyro.map(rescale_f32)),
            motor_rps: self.motor_erpm.map(|e| mechanical_from_erpm(e, self.motor_poles)),
            battery_voltage: rescale_f32(self.battery),
            accel: flu_frd(self.accel_g).map(accel_from_g),
        }
    }
}

/// What BF logs of [`LqrTelemetry`], in the units it logs it in; the C glue
/// copies each field into its blackbox or debug slot unconverted.
///
/// Axis order: `[roll, pitch, yaw]`, FLU.  Motor order: BF's motor index.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "std", derive(Debug))]
pub struct BfTelemetry {
    /// `pidData[axis].P` and `.Sum`: rate error.
    pub pid_p: [DeciDegPerSec; 3],
    /// `pidData[axis].I`: the integral state over the loop period (the
    /// state itself, a fraction of a degree, would log as 0).
    pub pid_i: [DeciDegPerSec; 3],
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

impl BfTelemetry {
    /// The logged fields of `t`, encoded in BF's units and axes;
    /// `raw_setpoint` is the last RX packet's rate setpoint (FRD).
    pub fn encode(t: &LqrTelemetry, raw_setpoint: [RotPerSec; 3]) -> Self {
        let debug_rate = |r: [RotPerSec; 3]| -> [DebugDeciDps; 3] {
            flu_frd(r).map(|r| to_debug(rescale_f32(r)))
        };
        BfTelemetry {
            pid_p: flu_frd(t.error_prop).map(rescale_f32),
            pid_i: flu_frd(t.error_integral_rate).map(rescale_f32),
            pid_d: flu_frd(t.error_derivative).map(rescale_f32),
            feedback: t.feedback_v.map(|v| to_debug(rescale_f32(v))),
            raw_setpoint: debug_rate(raw_setpoint),
            setpoint: debug_rate(t.setpoint_rps),
            ff_target: debug_rate(t.ff_target_rps),
            feedforward_motor0: to_debug(rescale_f32(t.feedforward_v[0])),
            command_scale: to_debug(rescale_f32(<qty!(1, f32)>::new(t.command_scale))),
        }
    }
}
