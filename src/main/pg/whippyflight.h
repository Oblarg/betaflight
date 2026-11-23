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
#define WHIPPYFLIGHT_IDLE_RPS_MAX 500
#define WHIPPYFLIGHT_DEBUG_MOTOR_MAX 3
#define WHIPPYFLIGHT_DROPOUT_RAMP_MS_MAX 100
// Below the Nyquist rate of the voltage task, which runs at
// FAST_VOLTAGE_TASK_FREQ_HZ under WhippyFlight.
#define WHIPPYFLIGHT_VBAT_LPF_HZ_MAX 100
#define WHIPPYFLIGHT_BRAKE_MARGIN_MAX 90

typedef struct whippyflightConfig_s {
    // Cutoff of the WhippyFlight controller's low-pass on reported motor
    // speeds, Hz; 0 is off.  Separate from rpm_filter_lpf_hz: telemetry
    // jitter reaches the motors through the gain schedule, and the cutoff
    // that best stops it is below what the RPM notches can track with.
    uint16_t rpm_lpf_hz;
    // Fly the WhippyFlight controller rather than BF's PID controller and
    // mixer.  Read once at boot (whippyflightIsActive).
    uint8_t enabled;
    // Unused: keeps the stored layout of the fields after it.
    uint16_t reserved0;
    // Idle speed, mechanical rev/s: no motor is commanded below the voltage
    // that holds it, and reported speeds are clamped up to it for the gain
    // lookup.  The gain table's lowest speed applies when it is higher; 0
    // leaves only that.  BF's dynamic idle controller does not run under
    // WhippyFlight (it lives in the mixer), so dyn_idle_min_rpm is unused.
    uint16_t idle_rps;
    // BF motor index whose control effort debug_mode = WHIPPYFLIGHT logs.
    uint8_t debug_motor;
    // Lowest battery voltage per cell, 0.01 V, the controller compensates
    // for; below it the motors weaken with the pack as under BF.  0 uses
    // vbat_warning_cell_voltage.
    uint16_t critical_cell_voltage;
    // Longest ramp, ms, from the last stick setpoint to the packet after an
    // RX dropout, which otherwise steps there; the ramp spans the missed
    // frames up to this.  0 steps.
    uint8_t dropout_ramp_ms;
    // Cutoff of the controller's low-pass on the battery reading, Hz; 0 is
    // off.  The reading converts commanded volts to duty, so it has to
    // follow the sag within a flip.
    uint8_t vbat_lpf_hz;
    // Share of the stick target's jerk limit, percent, held back when it
    // eases onto its goal, for airframes that brake worse than their fit;
    // 0 eases at the full limit.
    uint8_t brake_margin;
} whippyflightConfig_t;

PG_DECLARE(whippyflightConfig_t, whippyflightConfig);
