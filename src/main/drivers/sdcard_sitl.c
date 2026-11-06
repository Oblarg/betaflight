/*
 * This file is part of Betaflight.
 *
 * Betaflight is free software. You can redistribute this software
 * and/or modify this software under the terms of the GNU General
 * Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later
 * version.
 *
 * Betaflight is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this software.
 *
 * If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * SD card backed by a raw disk image file (SITL_SDCARD_FILENAME, an MBR plus
 * a FAT16/FAT32 partition), so asyncfatfs runs unmodified in SITL.  Like the
 * real drivers, each read or write is accepted, then completed (callback
 * fired) on the next sdcard_poll().  A missing image is a missing card.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "platform.h"

#if defined(USE_SDCARD) && defined(USE_SDCARD_SITL)

#include "common/utils.h"

#include "pg/bus_spi.h"

#include "drivers/sdcard.h"
#include "drivers/sdcard_impl.h"
#include "drivers/sdcard_sitl.h"

static FILE *imageFile;

static bool sdcardSitl_seek(uint32_t blockIndex)
{
    return imageFile && blockIndex < sdcard.metadata.numBlocks
        && fseek(imageFile, (long)blockIndex * SDCARD_BLOCK_SIZE, SEEK_SET) == 0;
}

static void sdcardSitl_preinit(const sdcardConfig_t *config)
{
    UNUSED(config);
}

static void sdcardSitl_init(const sdcardConfig_t *config, const spiPinConfig_t *spiConfig)
{
    UNUSED(config);
    UNUSED(spiConfig);

    sdcard.state = SDCARD_STATE_NOT_PRESENT;
    imageFile = fopen(SITL_SDCARD_FILENAME, "r+b");
    if (!imageFile) {
        printf("[SDCARD_SITL] no '%s', card not present\n", SITL_SDCARD_FILENAME);
        return;
    }
    fseek(imageFile, 0, SEEK_END);
    memset(&sdcard.metadata, 0, sizeof(sdcard.metadata));
    sdcard.metadata.numBlocks = (uint32_t)(ftell(imageFile) / SDCARD_BLOCK_SIZE);
    sdcard.enabled = true;
    sdcard.state = SDCARD_STATE_READY;
    printf("[SDCARD_SITL] loaded '%s', %u blocks\n", SITL_SDCARD_FILENAME, (unsigned)sdcard.metadata.numBlocks);
}

static void sdcardSitl_complete(sdcardBlockOperation_e operation, bool success)
{
    sdcard.state = SDCARD_STATE_READY;
    if (sdcard.pendingOperation.callback) {
        sdcard.pendingOperation.callback(operation, sdcard.pendingOperation.blockIndex,
            success ? sdcard.pendingOperation.buffer : NULL, sdcard.pendingOperation.callbackData);
    }
}

static bool sdcardSitl_poll(void)
{
    switch (sdcard.state) {
    case SDCARD_STATE_READING: {
        const bool ok = sdcardSitl_seek(sdcard.pendingOperation.blockIndex)
            && fread(sdcard.pendingOperation.buffer, SDCARD_BLOCK_SIZE, 1, imageFile) == 1;
        sdcardSitl_complete(SDCARD_BLOCK_OPERATION_READ, ok);
        break;
    }
    case SDCARD_STATE_SENDING_WRITE: {
        const bool ok = sdcardSitl_seek(sdcard.pendingOperation.blockIndex)
            && fwrite(sdcard.pendingOperation.buffer, SDCARD_BLOCK_SIZE, 1, imageFile) == 1;
        fflush(imageFile);
        sdcardSitl_complete(SDCARD_BLOCK_OPERATION_WRITE, ok);
        break;
    }
    default:
        break;
    }
    return sdcard.state == SDCARD_STATE_READY;
}

static void sdcardSitl_queue(uint32_t blockIndex, uint8_t *buffer, sdcard_operationCompleteCallback_c callback,
    uint32_t callbackData, sdcardState_e state)
{
    sdcard.pendingOperation.buffer = buffer;
    sdcard.pendingOperation.blockIndex = blockIndex;
    sdcard.pendingOperation.callback = callback;
    sdcard.pendingOperation.callbackData = callbackData;
    sdcard.state = state;
}

static bool sdcardSitl_readBlock(uint32_t blockIndex, uint8_t *buffer, sdcard_operationCompleteCallback_c callback, uint32_t callbackData)
{
    if (sdcard.state != SDCARD_STATE_READY) {
        return false;
    }
    sdcardSitl_queue(blockIndex, buffer, callback, callbackData, SDCARD_STATE_READING);
    return true;
}

static sdcardOperationStatus_e sdcardSitl_beginWriteBlocks(uint32_t blockIndex, uint32_t blockCount)
{
    UNUSED(blockIndex);
    UNUSED(blockCount);
    return sdcard.state == SDCARD_STATE_READY ? SDCARD_OPERATION_SUCCESS : SDCARD_OPERATION_BUSY;
}

static sdcardOperationStatus_e sdcardSitl_writeBlock(uint32_t blockIndex, uint8_t *buffer, sdcard_operationCompleteCallback_c callback, uint32_t callbackData)
{
    if (sdcard.state != SDCARD_STATE_READY) {
        return SDCARD_OPERATION_BUSY;
    }
    sdcardSitl_queue(blockIndex, buffer, callback, callbackData, SDCARD_STATE_SENDING_WRITE);
    return SDCARD_OPERATION_IN_PROGRESS;
}

static bool sdcardSitl_isFunctional(void)
{
    return sdcard.state != SDCARD_STATE_NOT_PRESENT;
}

static bool sdcardSitl_isInitialized(void)
{
    return sdcard.state >= SDCARD_STATE_READY;
}

static const sdcardMetadata_t *sdcardSitl_getMetadata(void)
{
    return &sdcard.metadata;
}

sdcardVTable_t sdcardSitlVTable = {
    sdcardSitl_preinit,
    sdcardSitl_init,
    sdcardSitl_readBlock,
    sdcardSitl_beginWriteBlocks,
    sdcardSitl_writeBlock,
    sdcardSitl_poll,
    sdcardSitl_isFunctional,
    sdcardSitl_isInitialized,
    sdcardSitl_getMetadata,
#ifdef SDCARD_PROFILING
    NULL,
#endif
};

#endif
