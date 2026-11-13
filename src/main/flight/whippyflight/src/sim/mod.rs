//! Host-side simulation (feature `sim`): the nonlinear plant gen_wfg.py
//! linearizes, and the stick flicks the evals and the Configurator's
//! WhippyFlight tab fly the controller through on it.

pub mod flick;
pub mod margin;
pub mod noise;
pub mod plant;
pub mod units;

/// Copy `bytes` into leaked, 4-byte-aligned memory.  `load_gains` reads the
/// tables in place, so blobs must be `'static` and aligned like the
/// firmware's RAM copy of the gain partition.
pub fn static_blob(bytes: &[u8]) -> &'static [u8] {
    let words: &'static mut [u32] = vec![0u32; bytes.len().div_ceil(4)].leak();
    // SAFETY: the u32 buffer covers at least `bytes.len()` bytes and is leaked.
    let buf = unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, bytes.len()) };
    buf.copy_from_slice(bytes);
    buf
}
