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
//! every conversion to and from WhippyFlight's units and axes happens here.
//!
//! Quantity arguments are `#[repr(transparent)]` over their storage, so they
//! cross the C ABI exactly as the `float`/`uint32_t` the glue declares.
//!
//! # Safety
//!
//! The entry points share unsynchronized global state: each must be called
//! from BF's single-threaded main loop, never concurrently or reentrantly.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use crate::bf::*;
use whippyflight_control::controller::ControllerState;
use whippyflight_control::dropout::DropoutRamp;
use whippyflight_control::gains::{self, GainStorage};
use whippyflight_control::setpoint::SetpointEstimator;
use whippyflight_control::types::*;
use whippyunits::api::rescale_f32;
use whippyunits::qty;

// ---------------------------------------------------------------------------
// Global state — single-threaded access from BF's main loop.
// ---------------------------------------------------------------------------

struct SingleThreadCell<T>(UnsafeCell<T>);

unsafe impl<T> Sync for SingleThreadCell<T> {}

impl<T> SingleThreadCell<T> {
    const fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }

    /// # Safety
    /// No other reference to the value may be live while this one is.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }

    /// # Safety
    /// No mutable reference to the value may be live while this one is.
    #[inline]
    unsafe fn get_ref(&self) -> &T {
        unsafe { &*self.0.get() }
    }
}

static GAINS: SingleThreadCell<GainStorage> = SingleThreadCell::new(GainStorage::new());
static SETPOINT: SingleThreadCell<SetpointEstimator> = SingleThreadCell::new(SetpointEstimator::new());
static DROPOUT: SingleThreadCell<DropoutRamp> = SingleThreadCell::new(DropoutRamp::new());

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

/// Load gain data from a RAM buffer.  Called at boot, and again after a
/// gain upload.
///
/// The gain tables are read in place until the next load, so `gains_buf`
/// must point at static, 4-byte-aligned memory (the glue's RAM copy of the
/// gain store) that is only rewritten after [`whippyflight_unload`].
///
/// `mixer_mode` and `yaw_motors_reversed` are `mixerConfig()`'s: the blob
/// is rejected ([`GainLoadResult::LayoutMismatch`]) unless its motor layout
/// is the one BF's mixer drives.
///
/// Returns 0 on success, nonzero on error (see [`GainLoadResult`]).
///
/// # Safety
/// BF's main loop only (see the module docs); `gains_buf` points at
/// `gains_len` readable bytes as above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_init(
    gains_buf: *const u8,
    gains_len: u32,
    mixer_mode: u8,
    yaw_motors_reversed: bool,
) -> GainLoadResult {
    if gains_buf.is_null() || gains_len == 0 {
        return GainLoadResult::TooShort;
    }

    // SAFETY: the caller guarantees the buffer is static (see above).
    let data: &'static [u8] = unsafe { core::slice::from_raw_parts(gains_buf, gains_len as usize) };
    let storage = unsafe { GAINS.get_mut() };
    let mut result = gains::load_gains(data, storage);
    if result == GainLoadResult::Ok
        && (mixer_mode != MIXER_QUADX || storage.layout != quadx_layout(yaw_motors_reversed))
    {
        storage.loaded = false;
        result = GainLoadResult::LayoutMismatch;
    }

    if result == GainLoadResult::Ok {
        unsafe { controller_mut() }.reset();
        unsafe { SETPOINT.get_mut() }.configure_for(storage);
    }

    result
}

/// Stop reading the gain buffer, so the glue may rewrite it; the
/// controller outputs zero duty until the next successful
/// [`whippyflight_init`].
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_unload() {
    unsafe { GAINS.get_mut() }.loaded = false;
}

/// Feed one RX packet to the setpoint estimator, through the dropout ramp:
/// the raw (unsmoothed) rate setpoint [roll, pitch, yaw], FLU, the packet's
/// arrival time, and the current RX rate estimate (`getCurrentRxRateHz`).
///
/// # Safety
/// BF's main loop only (see the module docs); `rate` points at 3 floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_setpoint_packet(
    rate: *const DegPerSec,
    time_us: Micros,
    rx_rate: Hertz,
) {
    let rate = unsafe { core::slice::from_raw_parts(rate, 3) };
    let rate = flu_frd([rate[0], rate[1], rate[2]].map(rescale_f32));
    let period = 1.0 / rx_rate;
    let rate = unsafe { DROPOUT.get_mut() }.packet(rate, time_us, period);
    unsafe { SETPOINT.get_mut() }.packet(rate, time_us, period);
}

/// Run one control loop iteration.  Called from BF's main loop at the PID
/// rate (typically 4–8 kHz), on the same microsecond clock as the packet
/// timestamps.
///
/// # Safety
/// BF's main loop only (see the module docs); `sensors` points at a
/// `wf_sensor_data_t`.
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
    let setpoint = unsafe { SETPOINT.get_mut() };
    let (rate_rps, target) = setpoint.tick(now_us, dt);
    let out = ctrl.update_with_target(sensors, &SetpointData { rate_rps, throttle }, target, gains);
    setpoint.condition(ctrl.reachable_target());
    out
}

/// Set the minimum linearization speed from the idle speed
/// (`whippyflight_idle_rps`, shaft rev/s).  Reported motor speeds are
/// clamped up to `max(speed, grid min_rps)` before gain lookup and the
/// affine linearizer; gains diverge as ω → 0.
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_set_min_speed(idle: RotPerSec) {
    unsafe { controller_mut() }.set_min_rps(idle);
}

/// Set the lowest battery voltage the controller compensates for
/// (`whippyflight_critical_cell_voltage` × cells, or BF's warning voltage
/// without it); 0 compensates down to any reading.
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_set_min_battery(floor: CentiVolt) {
    unsafe { controller_mut() }.set_min_battery(rescale_f32(floor));
}

/// Set the cutoff of the low-pass on reported motor speeds
/// (`whippyflight_rpm_lpf_hz`), Hz; 0 turns it off.
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_set_motor_lpf(cutoff: Hertz) {
    unsafe { controller_mut() }.set_motor_lpf(cutoff);
}

/// Set the cutoff of the low-pass on the battery reading
/// (`whippyflight_vbat_lpf_hz`), Hz; 0 turns it off.
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_set_battery_lpf(cutoff: Hertz) {
    unsafe { controller_mut() }.set_battery_lpf(cutoff);
}

/// Set the share of the jerk limit the stick target holds back when easing
/// onto its goal (`whippyflight_brake_margin`, percent).
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_set_brake_margin(margin: Hundredths) {
    let margin: qty!(1, f32) = rescale_f32(margin);
    unsafe { SETPOINT.get_mut() }.set_brake_margin(margin.into());
}

/// Set the longest ramp over an RX dropout (`whippyflight_dropout_ramp_ms`);
/// 0 steps to the packet after it.
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_set_dropout_ramp(cap: Millis) {
    unsafe { DROPOUT.get_mut() }.set_cap(rescale_f32(cap));
}

/// Trace the feedback's schedule jitter into the telemetry's effort
/// (`debug_mode = WHIPPYFLIGHT`), at the cost of a second
/// interpolation of the feedback gains every loop.
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_set_effort_trace(on: bool) {
    unsafe { controller_mut() }.set_effort_trace(on);
}

/// Reset all internal state.  Called on disarm.
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_reset() {
    unsafe { controller_mut() }.reset();
}

/// Encode the current tick's telemetry for blackbox logging, in BF's units.
/// Returns null before the controller has run; otherwise the pointer is
/// valid until the next call.
///
/// # Safety
/// BF's main loop only (see the module docs).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_telemetry() -> *const BfTelemetry {
    if !*unsafe { INITIALIZED.get_ref() } {
        return core::ptr::null();
    }
    let raw = unsafe { SETPOINT.get_ref() }.raw();
    let telemetry = unsafe { BF_TELEMETRY.get_mut() };
    telemetry.write(BfTelemetry::encode(unsafe { controller_ref() }.telemetry(), raw));
    telemetry.as_ptr()
}
