/*
 * WhippyFlight MSP commands: gain store status and upload.
 *
 * All values little-endian, floats as IEEE-754 single.
 *
 * MSP2_WHIPPYFLIGHT_STATUS (no payload) replies:
 *   u8  protocol version (1)
 *   u8  gain load result (wf_gain_load_result_t)
 *   u8  gain source (wf_gain_source_t)
 *   u8  1 while an upload is in progress
 *   u32 loaded blob size (0 if none)
 *   u32 loaded blob CRC32 (its trailing 4 bytes)
 *   f32 loaded blob dt (s)
 *   f32 PID loop dt (s)
 *   u32 dataflash gain partition size (0 if none)
 *   u32 upload size, u32 upload bytes received
 *   u16 largest upload chunk
 *   u8  1 if this boot flies WhippyFlight (whippyflight_enabled at boot)
 *   u16 failing arming checks (wf_arming_check_e bits)
 *   u8  length, then that many chars: why arming is blocked ("" if ready)
 *
 * MSP2_WHIPPYFLIGHT_UPLOAD_BEGIN   u32 blob size
 * MSP2_WHIPPYFLIGHT_UPLOAD_DATA    u32 offset, then up to WF_UPLOAD_CHUNK_MAX bytes
 * MSP2_WHIPPYFLIGHT_UPLOAD_COMMIT  (no payload; blocks while the partition erases)
 * MSP2_WHIPPYFLIGHT_UPLOAD_ABORT   (no payload)
 *   each reply: u8 upload result (wf_upload_result_t), u8 gain load result
 *
 * The whippyflight_* settings are ordinary named settings
 * (MSP2_CLI_SETTING).
 */

#include "platform.h"

#ifdef USE_WHIPPYFLIGHT

#include <string.h>

#include "common/streambuf.h"

#include "flight/pid.h"
#include "flight/whippyflight_glue.h"
#include "flight/whippyflight_msp.h"

#include "msp/msp_protocol_v2_betaflight.h"

#define WF_MSP_PROTOCOL_VERSION 2

static void writeFloat(sbuf_t *dst, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    sbufWriteU32(dst, bits);
}

static void writeStatus(sbuf_t *dst)
{
    wf_gain_status_t status;
    whippyflightGainStatus(&status);
    const char *reason = whippyflightArmingBlockReason();
    const size_t reasonLength = reason ? strlen(reason) : 0;

    sbufWriteU8(dst, WF_MSP_PROTOCOL_VERSION);
    sbufWriteU8(dst, status.result);
    sbufWriteU8(dst, status.source);
    sbufWriteU8(dst, status.uploading);
    sbufWriteU32(dst, status.blobSize);
    sbufWriteU32(dst, status.blobCrc);
    writeFloat(dst, status.blobDt);
    writeFloat(dst, pidGetDT());
    sbufWriteU32(dst, status.partitionSize);
    sbufWriteU32(dst, status.uploadSize);
    sbufWriteU32(dst, status.uploadReceived);
    sbufWriteU16(dst, WF_UPLOAD_CHUNK_MAX);
    sbufWriteU8(dst, whippyflightIsActive());
    sbufWriteU16(dst, whippyflightArmingChecks());
    sbufWriteU8(dst, reasonLength);
    sbufWriteData(dst, reason, reasonLength);
}

static void writeUploadReply(sbuf_t *dst, wf_upload_result_t upload, wf_gain_load_result_t gain)
{
    sbufWriteU8(dst, upload);
    sbufWriteU8(dst, gain);
}

mspResult_e whippyflightMspProcess(int16_t cmd, sbuf_t *src, sbuf_t *dst)
{
    wf_gain_status_t status;
    switch (cmd) {
    case MSP2_WHIPPYFLIGHT_STATUS:
        writeStatus(dst);
        return MSP_RESULT_ACK;

    case MSP2_WHIPPYFLIGHT_UPLOAD_BEGIN: {
        if (sbufBytesRemaining(src) < 4) {
            return MSP_RESULT_ERROR;
        }
        const wf_upload_result_t result = whippyflightUploadBegin(sbufReadU32(src));
        whippyflightGainStatus(&status);
        writeUploadReply(dst, result, status.result);
        return MSP_RESULT_ACK;
    }

    case MSP2_WHIPPYFLIGHT_UPLOAD_DATA: {
        if (sbufBytesRemaining(src) < 4) {
            return MSP_RESULT_ERROR;
        }
        const uint32_t offset = sbufReadU32(src);
        const int length = sbufBytesRemaining(src);
        const wf_upload_result_t result = length > WF_UPLOAD_CHUNK_MAX
            ? WF_UPLOAD_OUT_OF_ORDER
            : whippyflightUploadData(offset, sbufPtr(src), length);
        whippyflightGainStatus(&status);
        writeUploadReply(dst, result, status.result);
        return MSP_RESULT_ACK;
    }

    case MSP2_WHIPPYFLIGHT_UPLOAD_COMMIT: {
        wf_gain_load_result_t gain;
        const wf_upload_result_t result = whippyflightUploadCommit(&gain);
        writeUploadReply(dst, result, gain);
        return MSP_RESULT_ACK;
    }

    case MSP2_WHIPPYFLIGHT_UPLOAD_ABORT:
        whippyflightUploadAbort();
        whippyflightGainStatus(&status);
        writeUploadReply(dst, WF_UPLOAD_OK, status.result);
        return MSP_RESULT_ACK;

    default:
        return MSP_RESULT_CMD_UNKNOWN;
    }
}

#endif // USE_WHIPPYFLIGHT
