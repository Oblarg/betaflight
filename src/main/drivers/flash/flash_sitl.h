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

#pragma once

#include "flash_impl.h"

// File-backed NOR flash for SITL: a 4 MiB part (like a 32 Mbit W25Q32)
// with 64 KiB erase sectors and 256 B pages, stored in SITL_FLASH_FILENAME
// in the working directory.  Erased bytes are 0xFF and programming can
// only clear bits, as on real NOR.
#define SITL_FLASH_FILENAME     "dataflash.bin"
#define SITL_FLASH_PAGE_SIZE    256
#define SITL_FLASH_SECTOR_SIZE  (64 * 1024)
#define SITL_FLASH_SECTORS      64

bool sitlFlashInit(flashDevice_t *fdevice);
