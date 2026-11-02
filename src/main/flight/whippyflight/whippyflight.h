/**
 * WhippyFlight — Rust LQR control stack FFI header.
 *
 * This header declares the C-callable functions exported by the Rust
 * whippyflight-control static library.  Include it from BF's C code
 * and link against libwhippyflight_control.a.
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float gyro_rps[3];      // [yaw, pitch, roll] in rotations/s
    float motor_rps[4];     // [fl, bl, fr, br] in rotations/s
    float battery_voltage;  // volts
} WhippyflightSensorData;

typedef struct {
    float rate_rps[3];      // [yaw, pitch, roll] desired rates in rotations/s
    float throttle;         // [0, 1]
} WhippyflightSetpointData;

typedef struct {
    float motor_duty[4];    // [fl, bl, fr, br] duty cycle [0, 1]
} WhippyflightControlOutput;

typedef struct {
    float feedback_v[4];
    float feedforward_v[4];
    float affine_v[4];
    float error_prop[3];
    float error_integral[3];
    float error_derivative[3];
    float kp_motor0[3];
    uint8_t status;         // 0 = no gains, 1 = loaded, 2 = corrupt
} WhippyflightLqrTelemetry;

typedef enum {
    WHIPPYFLIGHT_OK = 0,
    WHIPPYFLIGHT_TOO_SHORT = 1,
    WHIPPYFLIGHT_BAD_MAGIC = 2,
    WHIPPYFLIGHT_BAD_CHECKSUM = 3,
    WHIPPYFLIGHT_BAD_VERSION = 4,
    WHIPPYFLIGHT_RESOLUTION_TOO_LARGE = 5,
} WhippyflightGainLoadResult;

/**
 * Load gain data from a flash buffer.  Call once at boot.
 * Returns WHIPPYFLIGHT_OK (0) on success.
 */
WhippyflightGainLoadResult whippyflight_init(
    const uint8_t *gains_buf,
    uint32_t gains_len
);

/**
 * Run one control iteration.  Call from the PID loop.
 */
WhippyflightControlOutput whippyflight_update(
    const WhippyflightSensorData *sensors,
    const WhippyflightSetpointData *setpoint
);

/**
 * Reset internal state (call on disarm).
 */
void whippyflight_reset(void);

/**
 * Get telemetry snapshot for blackbox logging.
 * Valid until the next whippyflight_update() call.
 */
const WhippyflightLqrTelemetry *whippyflight_telemetry(void);

#ifdef __cplusplus
}
#endif
