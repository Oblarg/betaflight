//! f64 quantities for the test plant and evals (the controller's own types
//! are f32; convert with `lossy_into`).
#![allow(dead_code)]

use whippyunits::qty;

/// Rotation rate (and model motor speed), rot/s.
pub type Rate = qty!(rot / s);
/// Angular acceleration, rot/s².
pub type AngAccel = qty!(rot / s ^ 2);
/// Signed squared motor speed ω|ω|, rot²/s².
pub type SpeedSq = qty!(rot ^ 2 / s ^ 2);
pub type Voltage = qty!(V);
pub type VoltageSq = qty!(V ^ 2);
pub type Time = qty!(s);
pub type TimeSq = qty!(s ^ 2);
pub type Millis = qty!(ms);
pub type Angle = qty!(rot);
pub type Freq = qty!(Hz);
/// Specific force, m/s².
pub type Accel = qty!(m / s ^ 2);
/// Airframe drag, rot/s² per m/s².
pub type PerMeter = qty!(rot / m);
/// Rate damping, rot/s² per rot/s.
pub type PerTime = qty!(1 / s);
/// Torque per signed squared speed, rot/s² per rot²/s².
pub type PerRot = qty!(1 / rot);
/// Motor drag k_ω, V·s/rot.
pub type MotorVel = qty!(V * s / rot);
/// Motor drag k_ω², V·s²/rot².
pub type MotorVelSq = qty!(V * s ^ 2 / rot ^ 2);
/// Motor inertia term k_α, V·s²/rot.
pub type MotorAccel = qty!(V * s ^ 2 / rot);
