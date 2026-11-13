/*
 * This file is part of Cleanflight and Betaflight.
 *
 * Cleanflight and Betaflight are free software. You can redistribute
 * this software and/or modify this software under the terms of the
 * GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * Cleanflight and Betaflight are distributed in the hope that they
 * will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software.
 *
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include <stdint.h>

#include "pg/pg.h"

#define WHIPPYFLIGHT_RPM_LPF_HZ_MAX 500
#define WHIPPYFLIGHT_STICK_TRACKING_MIN 1       // lambda 0.01
#define WHIPPYFLIGHT_STICK_TRACKING_MAX 10000   // lambda 100
#define WHIPPYFLIGHT_IDLE_RPS_MAX 500

typedef struct whippyflightConfig_s {
    // Cutoff of the WhippyFlight controller's low-pass on reported motor
    // speeds, Hz; 0 is off.  Separate from rpm_filter_lpf_hz: telemetry
    // jitter reaches the motors through the gain schedule, and the cutoff
    // that best stops it is below what the RPM notches can track with.
    uint16_t rpm_lpf_hz;
    // Fly the WhippyFlight controller rather than BF's PID controller and
    // mixer.  Read once at boot (whippyflightIsActive).
    uint8_t enabled;
    // Tracking index (lambda) of the stick setpoint estimator, in hundredths,
    // referenced to a 500 Hz link: higher trusts each RX packet more (less
    // smoothing of radio jitter, more overshoot on stick steps); below 1
    // (100) smooths over several packets, for cinematic flying.  The
    // stick-driven target is separately slew- and jerk-limited from the model.
    uint16_t stick_tracking;
    // Idle speed, mechanical rev/s: no motor is commanded below the voltage
    // that holds it, and reported speeds are clamped up to it for the gain
    // lookup.  The gain table's lowest speed applies when it is higher; 0
    // leaves only that.  BF's dynamic idle controller does not run under
    // WhippyFlight (it lives in the mixer), so dyn_idle_min_rpm is unused.
    uint16_t idle_rps;
} whippyflightConfig_t;

PG_DECLARE(whippyflightConfig_t, whippyflightConfig);
