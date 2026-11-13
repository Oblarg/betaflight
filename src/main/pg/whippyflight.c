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


#include "platform.h"

#ifdef USE_WHIPPYFLIGHT

#include "pg/pg.h"
#include "pg/pg_ids.h"

#include "whippyflight.h"

PG_REGISTER_WITH_RESET_TEMPLATE(whippyflightConfig_t, whippyflightConfig, PG_WHIPPYFLIGHT_CONFIG, 4);

PG_RESET_TEMPLATE(whippyflightConfig_t, whippyflightConfig,
    .rpm_lpf_hz = 50,
    .enabled = 1,
    .stick_tracking = 800,
    .idle_rps = 0,
);

#endif // USE_WHIPPYFLIGHT
