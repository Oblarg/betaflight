/*
 * WhippyFlight LQR controller — C glue layer.
 *
 * This header declares the Rust FFI types and the glue function that
 * replaces BF's pidController() + mixTable() when USE_WHIPPYFLIGHT is
 * enabled.
 *
 * The Rust staticlib (whippyflight-control) is linked via EXTRA_LD_FLAGS.
 * All unit conversion (deg/s → rot/s, centivolts → V, eRPM → rot/s,
 * duty → DShot) happens in the C glue — the Rust side uses clean SI
 * units exclusively.
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
    float gyro_rps[3];       // rot/s, [roll, pitch, yaw]
    float motor_rps[4];      // eRPM/600 (regression unit), by BF motor index
    float battery_voltage;   // volts
    float accel_g[3];        // g, body [x, y, z]; zero without an accelerometer
} wf_sensor_data_t;

typedef struct {
    float motor_duty[4];     // [0, 1], [REAR_R, FRONT_R, REAR_L, FRONT_L]
} wf_control_output_t;

typedef struct {
    float feedback_v[4];
    float feedforward_v[4];
    float affine_v[4];
    float error_prop[3];
    float error_integral[3];
    float error_derivative[3];
    float kp_motor0[3];
    float collective_v;      // V, along the thrust-balanced direction
    float command_scale;     // allocator attitude scale, 1 unless saturated
    float setpoint_rps[3];   // tracked rate setpoint, rot/s
    float ff_target_rps[3];  // feedforward h-tick-ahead target, rot/s
    uint8_t status;
} wf_lqr_telemetry_t;

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

// One RX frame: raw rate setpoint (rot/s, [roll, pitch, yaw]), its arrival
// time on the micros() clock, and the current RX period in seconds.
extern void whippyflight_setpoint_packet(
    const float rate_rps[3], uint32_t time_us, float rx_period_s);

// throttle in [0, 1]; now_us on the same clock as the packet timestamps.
extern wf_control_output_t whippyflight_update(
    const wf_sensor_data_t *sensors, float throttle, uint32_t now_us);

extern void whippyflight_reset(void);

extern void whippyflight_set_min_rps(float rps);

extern const wf_lqr_telemetry_t *whippyflight_telemetry(void);

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
