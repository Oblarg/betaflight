//! C FFI boundary.
//!
//! These `extern "C"` functions are the only entry points BF calls.
//! The Rust side owns no hardware — it is a pure control law.
//!
//! # Units
//!
//! Every argument is in BF's own unit (deg/s, centivolts, eRPM/100, Hz, µs),
//! typed as such, and telemetry comes back in the units BF logs it in
//! ([`BfTelemetry`]), so the C glue copies values through unconverted and
//! every conversion to and from WhippyFlight's units happens here.
//!
//! Quantity arguments are `#[repr(transparent)]` over their storage, so they
//! cross the C ABI exactly as the `float`/`uint32_t` the glue declares.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use crate::controller::ControllerState;
use crate::gains::{self, GainStorage};
use crate::setpoint::SetpointEstimator;
use crate::types::*;
use whippyunits::api::rescale_f32;

// ---------------------------------------------------------------------------
// Global state — single-threaded access from BF's main loop.
// ---------------------------------------------------------------------------

struct SingleThreadCell<T>(UnsafeCell<T>);

unsafe impl<T> Sync for SingleThreadCell<T> {}

impl<T> SingleThreadCell<T> {
    const fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }

    #[inline]
    unsafe fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }

    #[inline]
    unsafe fn get_ref(&self) -> &T {
        unsafe { &*self.0.get() }
    }
}

static GAINS: SingleThreadCell<GainStorage> = SingleThreadCell::new(GainStorage::new());
static SETPOINT: SingleThreadCell<SetpointEstimator> = SingleThreadCell::new(SetpointEstimator::new());

// ControllerState uses whippyalgebra zeros! which isn't const, so we
// defer initialization to whippyflight_init().
static CONTROLLER: SingleThreadCell<MaybeUninit<ControllerState>> =
    SingleThreadCell::new(MaybeUninit::uninit());
static INITIALIZED: SingleThreadCell<bool> = SingleThreadCell::new(false);
static BF_TELEMETRY: SingleThreadCell<MaybeUninit<BfTelemetry>> =
    SingleThreadCell::new(MaybeUninit::uninit());

/// Ensure the controller is initialized; returns a mutable reference.
///
/// # Safety
/// Must be called from the single-threaded FC main loop context.
unsafe fn controller_mut() -> &'static mut ControllerState {
    let init = unsafe { INITIALIZED.get_ref() };
    if !*init {
        let slot = unsafe { CONTROLLER.get_mut() };
        slot.write(ControllerState::new());
        *unsafe { INITIALIZED.get_mut() } = true;
    }
    unsafe { (*CONTROLLER.0.get()).assume_init_mut() }
}

/// # Safety
/// Must be called from the single-threaded FC main loop context.
unsafe fn controller_ref() -> &'static ControllerState {
    unsafe { (*CONTROLLER.0.get()).assume_init_ref() }
}

/// Load gain data from a RAM buffer.  Called once at boot.
///
/// The gain tables are read in place for the rest of the program, so
/// `gains_buf` must point at static, 4-byte-aligned memory that is never
/// rewritten (the glue's RAM copy of the dataflash gain partition).
///
/// Returns 0 on success, nonzero on error (see [`GainLoadResult`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_init(
    gains_buf: *const u8,
    gains_len: u32,
) -> GainLoadResult {
    if gains_buf.is_null() || gains_len == 0 {
        return GainLoadResult::TooShort;
    }

    // SAFETY: the caller guarantees the buffer is static (see above).
    let data: &'static [u8] = unsafe { core::slice::from_raw_parts(gains_buf, gains_len as usize) };
    let storage = unsafe { GAINS.get_mut() };
    let result = gains::load_gains(data, storage);

    if result == GainLoadResult::Ok {
        unsafe { controller_mut() }.reset();
        unsafe { SETPOINT.get_mut() }.configure_for(storage);
    }

    result
}

/// Feed one RX packet to the setpoint estimator: the raw (unsmoothed) rate
/// setpoint [roll, pitch, yaw] (`getRawSetpoint`), the packet's arrival time,
/// and the current RX rate estimate (`getCurrentRxRateHz`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_setpoint_packet(
    rate: *const DegPerSec,
    time_us: Micros,
    rx_rate: Hertz,
) {
    let rate = unsafe { core::slice::from_raw_parts(rate, 3) };
    let rate = [rate[0], rate[1], rate[2]].map(rescale_f32);
    unsafe { SETPOINT.get_mut() }.packet(rate, time_us, 1.0 / rx_rate);
}

/// Run one control loop iteration.  Called from BF's main loop at the PID
/// rate (typically 4–8 kHz), on the same microsecond clock as the packet
/// timestamps.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_update(
    sensors: *const BfSensorData,
    throttle: f32,
    now_us: Micros,
) -> ControlOutput {
    let sensors = &unsafe { &*sensors }.to_sensor_data();
    let ctrl = unsafe { controller_mut() };
    let gains = unsafe { GAINS.get_ref() };

    let dt = gains.header.dt;
    let (rate_rps, target) = unsafe { SETPOINT.get_mut() }
        .tick(now_us, dt);
    ctrl.update_with_target(sensors, &SetpointData { rate_rps, throttle }, target, gains)
}

/// Set the minimum linearization speed from BF's dynamic idle floor
/// (`dyn_idle_min_rpm`, mechanical) on a `pole_count`-pole motor.  Reported
/// motor speeds are clamped up to `max(speed, grid min_rps)` before gain
/// lookup and the affine linearizer; gains diverge as ω → 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_set_min_speed(dyn_idle_min: HundredRpm, pole_count: u8) {
    unsafe { controller_mut() }.set_min_rps(model_speed_from_mechanical(dyn_idle_min, pole_count));
}

/// Reset all internal state.  Called on disarm.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_reset() {
    unsafe { controller_mut() }.reset();
}

/// Encode the current tick's telemetry for blackbox logging, in BF's units.
/// Returns null before the controller has run; otherwise the pointer is
/// valid until the next call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_telemetry() -> *const BfTelemetry {
    if !*unsafe { INITIALIZED.get_ref() } {
        return core::ptr::null();
    }
    let raw = unsafe { SETPOINT.get_ref() }.raw();
    let telemetry = unsafe { BF_TELEMETRY.get_mut() };
    telemetry.write(unsafe { controller_ref() }.telemetry().to_bf(raw));
    telemetry.as_ptr()
}
