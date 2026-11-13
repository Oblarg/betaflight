/*
 * WhippyFlight LQR controller — C glue layer.
 *
 * This header declares the Rust FFI types and the glue function that
 * replaces BF's pidController() + mixTable() when USE_WHIPPYFLIGHT is
 * enabled.
 *
 * The Rust staticlib (whippyflight-control) is linked via EXTRA_LD_FLAGS.
 * Its inputs are in BF's own units (deg/s, centivolts, eRPM/100, Hz, µs) and
 * its telemetry comes back in the units BF logs, so the glue copies values
 * through; the Rust side converts them.
 */

#pragma once

#ifdef USE_WHIPPYFLIGHT

#include <stdbool.h>
#include <stdint.h>

#include "common/time.h"

// ---------------------------------------------------------------------------
// Rust FFI types — must match #[repr(C)] structs in types.rs exactly.
// ---------------------------------------------------------------------------

typedef struct {
    float gyro[3];           // deg/s (gyro.gyroADCf), [roll, pitch, yaw]
    float motor_erpm[4];     // eRPM/100 (getDshotErpm), by BF motor index
    float battery;           // centivolts (getBatteryVoltage)
    float accel_g[3];        // g, body [x, y, z]; zero without an accelerometer
    uint8_t motor_poles;     // motorConfig()->motorPoleCount: motor_erpm to shaft speed
} wf_sensor_data_t;

typedef struct {
    float motor_duty[4];     // [0, 1], [REAR_R, FRONT_R, REAR_L, FRONT_L]
} wf_control_output_t;

// Telemetry, already in the units BF logs it in: copy each field into its
// slot as is.  The pidData fields are floats in the unit of one logged count
// (blackbox lrintf()s them); the debug fields are rounded to int16_t.
typedef struct {
    float pid_p[3];               // pidData P, Sum: rate error, 0.1 deg/s
    float pid_i[3];               // pidData I: integral state / dt (low-passed rate error), 0.1 deg/s
    float pid_d[3];               // pidData D: rate-error derivative, deg/s²
    int16_t feedback[4];          // per-motor feedback voltage, mV
    int16_t raw_setpoint[3];      // last RX rate setpoint, 0.1 deg/s
    int16_t setpoint[3];          // tracked rate setpoint, 0.1 deg/s
    int16_t ff_target[3];         // feedforward target, 0.1 deg/s
    int16_t feedforward_motor0;   // motor 0 feedforward voltage, cV
    int16_t command_scale;        // allocator attitude scale, ‰ (1000 unless saturated)
} wf_bf_telemetry_t;

// GainLoadResult — matches repr(C) enum in types.rs.
typedef enum {
    WF_GAIN_OK             = 0,
    WF_GAIN_TOO_SHORT      = 1,
    WF_GAIN_BAD_MAGIC      = 2,
    WF_GAIN_BAD_CHECKSUM   = 3,
    WF_GAIN_BAD_VERSION    = 4,
    WF_GAIN_RES_TOO_LARGE  = 5,
    WF_GAIN_MISALIGNED     = 6,
    WF_GAIN_TOO_LARGE      = 7,   // blob exceeds WHIPPYFLIGHT_GAIN_STORE_SIZE (glue only)
    WF_GAIN_NO_STORE       = 8,   // no gain file on SD and no gain partition (glue only)
} wf_gain_load_result_t;

// ---------------------------------------------------------------------------
// Rust FFI functions (symbols from the staticlib).
// ---------------------------------------------------------------------------

extern wf_gain_load_result_t whippyflight_init(
    const uint8_t *gains_buf, uint32_t gains_len);

// One RX frame: raw rate setpoint (deg/s, [roll, pitch, yaw]), its arrival
// time on the micros() clock, and the current RX rate in Hz.
extern void whippyflight_setpoint_packet(
    const float rate_dps[3], uint32_t time_us, float rx_rate_hz);

// throttle in [0, 1]; now_us on the same clock as the packet timestamps.
extern wf_control_output_t whippyflight_update(
    const wf_sensor_data_t *sensors, float throttle, uint32_t now_us);

extern void whippyflight_reset(void);

// Stop reading the gain buffer (before the glue rewrites it); zero duty
// until the next successful whippyflight_init().
extern void whippyflight_unload(void);

// Idle speed, shaft rev/s.
extern void whippyflight_set_min_speed(float idle_rps);

// Cutoff of the low-pass on reported motor speeds, Hz; 0 is off.
extern void whippyflight_set_motor_lpf(float cutoff_hz);

// Stick setpoint tracking index λ at a 500 Hz link (higher follows the
// stick more closely, lower smooths more).
extern void whippyflight_set_stick_tracking(float lambda_500hz);

// NULL before the controller has run; valid until the next call.
extern const wf_bf_telemetry_t *whippyflight_telemetry(void);

// ---------------------------------------------------------------------------
// C glue entry points.
// ---------------------------------------------------------------------------

// Replaces subTaskPidController() + subTaskMotorUpdate() in taskMainPidLoop.
void whippyflightPidAndMotorUpdate(timeUs_t currentTimeUs);

// Called once at boot, after flash and SD card init: latches
// whippyflight_enabled, then loads the gains (whippyflightLoadGains), which
// load either way so they can be inspected and replaced.
void whippyflightInit(void);

// Whether this boot flies WhippyFlight; otherwise BF's PID controller and
// mixer run as in upstream builds and WhippyFlight never blocks arming.
bool whippyflightIsActive(void);

// Copies the gain blob from the SD card (WHIPPYFLIGHT_GAIN_FILENAME) or,
// failing that, the dataflash WHIPPYFLIGHT partition into RAM and hands it
// to the controller.
// Blocks for up to a few seconds while the SD card mounts.
wf_gain_load_result_t whippyflightLoadGains(void);

// Conditions the controller needs before it will fly; each set bit of
// whippyflightArmingChecks() is one that fails.  Reported over MSP, so the
// values are protocol.
typedef enum {
    WF_CHECK_UPLOADING      = 1 << 0,   // gain upload in progress
    WF_CHECK_GAINS          = 1 << 1,   // no valid gains loaded
    WF_CHECK_LOOP_RATE      = 1 << 2,   // PID loop rate differs from the gains' dt
    WF_CHECK_RPM_TELEMETRY  = 1 << 3,   // no ESC RPM telemetry (dshot_bidir off)
    WF_CHECK_SELF_LEVEL     = 1 << 4,   // a self-levelling mode is configured
    WF_CHECK_FAILSAFE       = 1 << 5,   // failsafe_procedure is not DROP
} wf_arming_check_e;

uint16_t whippyflightArmingChecks(void);
// False while the controller must not fly: any arming check fails.
bool whippyflightIsReadyToArm(void);
// Why arming is blocked (ARMING_DISABLED_WHIPPYFLIGHT): the first failing
// check, or NULL if ready.
const char *whippyflightArmingBlockReason(void);

// ---------------------------------------------------------------------------
// Gain upload to the dataflash partition (MSP2_WHIPPYFLIGHT_*).
//
// The blob is staged in the gain RAM, validated by the controller's own
// loader, and only then written to the partition and read back, so a
// rejected upload never replaces the stored gains.  Every step is refused
// while armed, and arming is blocked while an upload is in progress.
// ---------------------------------------------------------------------------

typedef enum {
    WF_GAIN_SOURCE_NONE      = 0,   // nothing loaded yet
    WF_GAIN_SOURCE_SDCARD    = 1,
    WF_GAIN_SOURCE_DATAFLASH = 2,
} wf_gain_source_t;

typedef enum {
    WF_UPLOAD_OK              = 0,
    WF_UPLOAD_ARMED           = 1,   // refused: the craft is armed
    WF_UPLOAD_NO_STORE        = 2,   // no dataflash gain partition
    WF_UPLOAD_TOO_LARGE       = 3,   // larger than the partition or the gain RAM
    WF_UPLOAD_BUSY            = 4,   // blackbox is using the dataflash
    WF_UPLOAD_NOT_STARTED     = 5,   // data or commit without a begin
    WF_UPLOAD_OUT_OF_ORDER    = 6,   // chunk offset is not the next byte, or overruns
    WF_UPLOAD_INCOMPLETE      = 7,   // commit before every byte arrived
    WF_UPLOAD_REJECTED        = 8,   // the controller rejected the blob (see gain result)
    WF_UPLOAD_VERIFY_FAILED   = 9,   // the partition does not read back as written
} wf_upload_result_t;

// Largest data chunk per MSP2_WHIPPYFLIGHT_UPLOAD_DATA; fits MSP's 192-byte
// inbound frame with its offset and framing.
#define WF_UPLOAD_CHUNK_MAX 128

typedef struct {
    wf_gain_load_result_t result;   // of the last load (or WF_GAIN_NO_STORE)
    wf_gain_source_t source;        // where the last load read from
    uint32_t blobSize;              // loaded blob, 0 if none
    uint32_t blobCrc;               // its trailing CRC32
    float blobDt;                   // its discretisation dt (s)
    uint32_t partitionSize;         // usable dataflash capacity, 0 if none
    bool uploading;
    uint32_t uploadSize;
    uint32_t uploadReceived;
} wf_gain_status_t;

void whippyflightGainStatus(wf_gain_status_t *status);

// Start staging a blob of `size` bytes; unloads the current gains.
wf_upload_result_t whippyflightUploadBegin(uint32_t size);
// Stage the next chunk; offset must equal the bytes received so far.
wf_upload_result_t whippyflightUploadData(uint32_t offset, const uint8_t *data, uint32_t length);
// Validate and load the staged blob, then write it to the partition and
// read it back.  Blocks while the partition erases (up to a few seconds).
// On rejection the stored gains are reloaded and *gainResult says why.
wf_upload_result_t whippyflightUploadCommit(wf_gain_load_result_t *gainResult);
// Abandon an upload and reload the stored gains.
void whippyflightUploadAbort(void);

#endif // USE_WHIPPYFLIGHT
