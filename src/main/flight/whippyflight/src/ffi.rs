//! C FFI boundary.
//!
//! These `extern "C"` functions are the only entry points BF calls.
//! The Rust side owns no hardware — it is a pure control law.

use core::cell::UnsafeCell;

use crate::controller::ControllerState;
use crate::gains::{self, GainStorage};
use crate::types::*;

// ---------------------------------------------------------------------------
// Global state — single-threaded access from BF's main loop.
//
// We use UnsafeCell in a wrapper that implements Sync (the FC is
// single-threaded; there is no preemption of the control loop).
// ---------------------------------------------------------------------------

struct SingleThreadCell<T>(UnsafeCell<T>);

// Safety: the flight controller is single-threaded. The main loop calls
// whippyflight_update() exactly once per iteration from one task context.
unsafe impl<T> Sync for SingleThreadCell<T> {}

impl<T> SingleThreadCell<T> {
    const fn new(val: T) -> Self {
        Self(UnsafeCell::new(val))
    }

    /// # Safety
    /// Caller must ensure exclusive access (single-threaded context).
    #[inline]
    unsafe fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }

    /// # Safety
    /// Caller must ensure no concurrent mutation.
    #[inline]
    unsafe fn get_ref(&self) -> &T {
        unsafe { &*self.0.get() }
    }
}

static GAINS: SingleThreadCell<GainStorage> = SingleThreadCell::new(GainStorage::new());
static CONTROLLER: SingleThreadCell<ControllerState> =
    SingleThreadCell::new(ControllerState::new());

/// Load gain data from a flash buffer.  Called once at boot after BF reads
/// the LQR flash partition.
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

    let data = unsafe { core::slice::from_raw_parts(gains_buf, gains_len as usize) };
    let storage = unsafe { GAINS.get_mut() };
    let result = gains::load_gains(data, storage);

    if result == GainLoadResult::Ok {
        unsafe { CONTROLLER.get_mut() }.reset();
    }

    result
}

/// Run one control loop iteration.  Called from BF's main loop at the PID
/// rate (typically 4–8 kHz).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_update(
    sensors: *const SensorData,
    setpoint: *const SetpointData,
) -> ControlOutput {
    let sensors = unsafe { &*sensors };
    let setpoint = unsafe { &*setpoint };
    let controller = unsafe { CONTROLLER.get_mut() };
    let gains = unsafe { GAINS.get_ref() };

    controller.update(sensors, setpoint, gains)
}

/// Reset all internal state.  Called on disarm.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_reset() {
    unsafe { CONTROLLER.get_mut() }.reset();
}

/// Get a pointer to the current telemetry snapshot for blackbox logging.
/// The returned pointer is valid until the next call to `whippyflight_update`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn whippyflight_telemetry() -> *const LqrTelemetry {
    unsafe { CONTROLLER.get_ref() }.telemetry() as *const LqrTelemetry
}
