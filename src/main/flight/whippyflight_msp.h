/*
 * WhippyFlight MSP commands: gain store status and upload.
 */

#pragma once

#ifdef USE_WHIPPYFLIGHT

#include "msp/msp.h"

// Handle an MSP2_WHIPPYFLIGHT_* command; MSP_RESULT_CMD_UNKNOWN for others.
mspResult_e whippyflightMspProcess(int16_t cmd, sbuf_t *src, sbuf_t *dst);

#endif // USE_WHIPPYFLIGHT
