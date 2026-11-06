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

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "platform.h"

#if defined(USE_FLASH_CHIP) && defined(USE_FLASH_SITL)

#include "common/maths.h"
#include "common/utils.h"

#include "drivers/flash/flash.h"
#include "drivers/flash/flash_impl.h"
#include "drivers/flash/flash_sitl.h"

#define SITL_FLASH_SIZE (SITL_FLASH_SECTOR_SIZE * SITL_FLASH_SECTORS)

static FILE *flashFile;

static bool sitlFlashSeek(uint32_t address)
{
    return flashFile && address < SITL_FLASH_SIZE && fseek(flashFile, address, SEEK_SET) == 0;
}

static void sitlFlashFill(uint32_t address, uint32_t length)
{
    uint8_t erased[SITL_FLASH_PAGE_SIZE];
    memset(erased, 0xFF, sizeof(erased));
    if (!sitlFlashSeek(address)) {
        return;
    }
    while (length > 0) {
        const uint32_t chunk = MIN(length, (uint32_t)sizeof(erased));
        fwrite(erased, 1, chunk, flashFile);
        length -= chunk;
    }
    fflush(flashFile);
}

static void sitlFlashConfigure(flashDevice_t *fdevice, uint32_t configurationFlags)
{
    UNUSED(fdevice);
    UNUSED(configurationFlags);
}

static bool sitlFlashIsReady(flashDevice_t *fdevice)
{
    UNUSED(fdevice);
    return true;
}

static bool sitlFlashWaitForReady(flashDevice_t *fdevice)
{
    UNUSED(fdevice);
    return true;
}

static void sitlFlashEraseSector(flashDevice_t *fdevice, uint32_t address)
{
    UNUSED(fdevice);
    sitlFlashFill(address - (address % SITL_FLASH_SECTOR_SIZE), SITL_FLASH_SECTOR_SIZE);
}

static void sitlFlashEraseCompletely(flashDevice_t *fdevice)
{
    UNUSED(fdevice);
    sitlFlashFill(0, SITL_FLASH_SIZE);
}

static void sitlFlashPageProgramBegin(flashDevice_t *fdevice, uint32_t address, void (*callback)(uintptr_t arg))
{
    fdevice->callback = callback;
    fdevice->currentWriteAddress = address;
}

static uint32_t sitlFlashPageProgramContinue(flashDevice_t *fdevice, uint8_t const **buffers, const uint32_t *bufferSizes, uint32_t bufferCount)
{
    uint32_t written = 0;
    for (uint32_t i = 0; i < bufferCount; i++) {
        for (uint32_t j = 0; j < bufferSizes[i]; j++) {
            const uint32_t address = fdevice->currentWriteAddress + written;
            uint8_t current;
            if (!sitlFlashSeek(address) || fread(&current, 1, 1, flashFile) != 1) {
                break;
            }
            // NOR programming clears bits; it never sets them.
            const uint8_t programmed = current & buffers[i][j];
            sitlFlashSeek(address);
            fwrite(&programmed, 1, 1, flashFile);
            written++;
        }
    }
    fflush(flashFile);

    fdevice->currentWriteAddress += written;
    fdevice->bytesWritten = written;
    if (fdevice->callback) {
        fdevice->callback(written);
    }
    return written;
}

static void sitlFlashPageProgramFinish(flashDevice_t *fdevice)
{
    UNUSED(fdevice);
}

static void sitlFlashPageProgram(flashDevice_t *fdevice, uint32_t address, const uint8_t *data, uint32_t length, void (*callback)(uintptr_t arg))
{
    sitlFlashPageProgramBegin(fdevice, address, callback);
    sitlFlashPageProgramContinue(fdevice, &data, &length, 1);
    sitlFlashPageProgramFinish(fdevice);
}

static int sitlFlashReadBytes(flashDevice_t *fdevice, uint32_t address, uint8_t *buffer, uint32_t length)
{
    UNUSED(fdevice);
    if (!sitlFlashSeek(address)) {
        return 0;
    }
    length = MIN(length, SITL_FLASH_SIZE - address);
    return (int)fread(buffer, 1, length, flashFile);
}

static const flashGeometry_t *sitlFlashGetGeometry(flashDevice_t *fdevice)
{
    return &fdevice->geometry;
}

static const flashVTable_t sitlFlashVTable = {
    .configure = sitlFlashConfigure,
    .isReady = sitlFlashIsReady,
    .waitForReady = sitlFlashWaitForReady,
    .eraseSector = sitlFlashEraseSector,
    .eraseCompletely = sitlFlashEraseCompletely,
    .pageProgramBegin = sitlFlashPageProgramBegin,
    .pageProgramContinue = sitlFlashPageProgramContinue,
    .pageProgramFinish = sitlFlashPageProgramFinish,
    .pageProgram = sitlFlashPageProgram,
    .readBytes = sitlFlashReadBytes,
    .getGeometry = sitlFlashGetGeometry,
};

bool sitlFlashInit(flashDevice_t *fdevice)
{
    flashFile = fopen(SITL_FLASH_FILENAME, "r+b");
    if (!flashFile) {
        flashFile = fopen(SITL_FLASH_FILENAME, "w+b");
        if (!flashFile) {
            fprintf(stderr, "[FLASH_SITL] failed to create '%s'\n", SITL_FLASH_FILENAME);
            return false;
        }
        sitlFlashEraseCompletely(fdevice);
        printf("[FLASH_SITL] created erased '%s', size = %u\n", SITL_FLASH_FILENAME, SITL_FLASH_SIZE);
    } else {
        fseek(flashFile, 0, SEEK_END);
        const long size = ftell(flashFile);
        if (size < (long)SITL_FLASH_SIZE) {
            sitlFlashFill((uint32_t)size, SITL_FLASH_SIZE - (uint32_t)size);
        }
        printf("[FLASH_SITL] loaded '%s', size = %u\n", SITL_FLASH_FILENAME, SITL_FLASH_SIZE);
    }

    fdevice->vTable = &sitlFlashVTable;
    fdevice->geometry.flashType = FLASH_TYPE_NOR;
    fdevice->geometry.pageSize = SITL_FLASH_PAGE_SIZE;
    fdevice->geometry.sectorSize = SITL_FLASH_SECTOR_SIZE;
    fdevice->geometry.sectors = SITL_FLASH_SECTORS;
    fdevice->geometry.pagesPerSector = SITL_FLASH_SECTOR_SIZE / SITL_FLASH_PAGE_SIZE;
    fdevice->geometry.totalSize = SITL_FLASH_SIZE;
    fdevice->isLargeFlash = false;
    fdevice->couldBeBusy = false;
    return true;
}

#endif // USE_FLASH_CHIP && USE_FLASH_SITL
