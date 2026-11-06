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
 * Feed one RX frame to the setpoint estimator: raw rate setpoint in rot/s,
 * arrival time (µs), and the current RX period (s).
 */
void whippyflight_setpoint_packet(
    const float rate_rps[3],
    uint32_t time_us,
    float rx_period_s
);

/**
 * Run one control iteration.  Call from the PID loop; now_us shares the
 * packet timestamps' clock.
 */
WhippyflightControlOutput whippyflight_update(
    const WhippyflightSensorData *sensors,
    float throttle,
    uint32_t now_us
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
