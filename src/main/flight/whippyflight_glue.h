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
} wf_sensor_data_t;

typedef struct {
    float motor_duty[4];     // [0, 1], [REAR_R, FRONT_R, REAR_L, FRONT_L]
} wf_control_output_t;

// Telemetry, already in the units BF logs it in: copy each field into its
// slot as is.  The pidData fields are floats in the unit of one logged count
// (blackbox lrintf()s them); the debug fields are rounded to int16_t.
typedef struct {
    float pid_p[3];               // pidData P, Sum: rate error, 0.1 deg/s
    float pid_i[3];               // pidData I: integrated rate error, 0.1 deg
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

// Dynamic idle floor: dyn_idle_min_rpm (mechanical RPM/100) and motorPoleCount.
extern void whippyflight_set_min_speed(float dyn_idle_min_rpm, uint8_t pole_count);

// NULL before the controller has run; valid until the next call.
extern const wf_bf_telemetry_t *whippyflight_telemetry(void);

// ---------------------------------------------------------------------------
// C glue entry points.
// ---------------------------------------------------------------------------

// Replaces subTaskPidController() + subTaskMotorUpdate() in taskMainPidLoop.
void whippyflightPidAndMotorUpdate(timeUs_t currentTimeUs);

// Called once at boot, after flash and SD card init: copies the gain blob
// from the SD card (WHIPPYFLIGHT_GAIN_FILENAME) or, failing that, the
// dataflash WHIPPYFLIGHT partition into RAM and hands it to the controller.
// Blocks for up to a few seconds while the SD card mounts.
wf_gain_load_result_t whippyflightLoadGains(void);

// False while the controller must not fly: no valid gains, gains
// discretised for a different PID loop rate, or no RPM telemetry.
bool whippyflightIsReadyToArm(void);
// Why arming is blocked (ARMING_DISABLED_WHIPPYFLIGHT), or NULL if ready.
const char *whippyflightArmingBlockReason(void);

#endif // USE_WHIPPYFLIGHT
