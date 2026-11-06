#!/usr/bin/env python3
"""
Build a raw SD card image: an MBR with one FAT16 partition holding the given
files in its root directory.  Used to seed the SITL file-backed SD card
(drivers/sdcard_sitl.c), which asyncfatfs then mounts like a real card.

Usage:
    python3 fat_image.py <image> [NAME.EXT=path ...]
"""

import struct
import sys

SECTOR = 512
IMAGE_SECTORS = 64 * 1024          # 32 MiB
PARTITION_START = 2048
SECTORS_PER_CLUSTER = 4
RESERVED_SECTORS = 4
NUM_FATS = 2
ROOT_ENTRIES = 512
MBR_TYPE_FAT16_LBA = 0x0E


def _geometry():
    part_sectors = IMAGE_SECTORS - PARTITION_START
    root_sectors = ROOT_ENTRIES * 32 // SECTOR
    fat_sectors = 1
    while True:
        data_sectors = part_sectors - RESERVED_SECTORS - NUM_FATS * fat_sectors - root_sectors
        clusters = data_sectors // SECTORS_PER_CLUSTER
        needed = -(-(clusters + 2) * 2 // SECTOR)
        if needed <= fat_sectors:
            break
        fat_sectors = needed
    assert 4085 <= clusters <= 65524, "geometry is not FAT16"
    return part_sectors, fat_sectors, root_sectors, clusters


def _short_name(name):
    base, _, ext = name.upper().partition(".")
    assert 1 <= len(base) <= 8 and len(ext) <= 3, f"not an 8.3 name: {name}"
    return base.ljust(8).encode() + ext.ljust(3).encode()


def build_image(path, files):
    """Write the image to `path`; `files` maps 8.3 names to contents."""
    part_sectors, fat_sectors, root_sectors, clusters = _geometry()
    cluster_bytes = SECTORS_PER_CLUSTER * SECTOR
    fat_start = PARTITION_START + RESERVED_SECTORS
    root_start = fat_start + NUM_FATS * fat_sectors
    data_start = root_start + root_sectors

    image = bytearray(IMAGE_SECTORS * SECTOR)

    mbr = bytearray(SECTOR)
    mbr[446:462] = struct.pack("<B3sB3sII", 0x00, b"\xfe\xff\xff", MBR_TYPE_FAT16_LBA,
                               b"\xfe\xff\xff", PARTITION_START, part_sectors)
    mbr[510:512] = b"\x55\xaa"
    image[0:SECTOR] = mbr

    boot = bytearray(SECTOR)
    boot[0:3] = b"\xeb\x3c\x90"
    boot[3:11] = b"WHIPPY  "
    struct.pack_into("<HBHBHHBHHHII", boot, 11,
                     SECTOR, SECTORS_PER_CLUSTER, RESERVED_SECTORS, NUM_FATS, ROOT_ENTRIES,
                     0, 0xF8, fat_sectors, 63, 255, PARTITION_START, part_sectors)
    struct.pack_into("<BBBI11s8s", boot, 36, 0x80, 0, 0x29, 0x57464731,
                     b"WHIPPYFLT  ", b"FAT16   ")
    boot[510:512] = b"\x55\xaa"
    image[PARTITION_START * SECTOR:(PARTITION_START + 1) * SECTOR] = boot

    fat = [0] * (clusters + 2)
    fat[0], fat[1] = 0xFFF8, 0xFFFF
    root = bytearray(root_sectors * SECTOR)
    next_cluster = 2
    for index, (name, data) in enumerate(files.items()):
        count = max(1, -(-len(data) // cluster_bytes))
        first = next_cluster
        for c in range(first, first + count):
            fat[c] = c + 1
        fat[first + count - 1] = 0xFFFF
        next_cluster += count
        offset = (data_start + (first - 2) * SECTORS_PER_CLUSTER) * SECTOR
        image[offset:offset + len(data)] = data
        root[index * 32:(index + 1) * 32] = struct.pack(
            "<11sBBBHHHHHHHI", _short_name(name), 0x20, 0, 0, 0, 0x5B21, 0x5B21,
            0, 0, 0x5B21, first, len(data))

    fat_bytes = struct.pack(f"<{len(fat)}H", *fat)
    for i in range(NUM_FATS):
        offset = (fat_start + i * fat_sectors) * SECTOR
        image[offset:offset + len(fat_bytes)] = fat_bytes
    image[root_start * SECTOR:root_start * SECTOR + len(root)] = root

    with open(path, "wb") as f:
        f.write(image)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    contents = {}
    for arg in sys.argv[2:]:
        name, _, src = arg.partition("=")
        contents[name] = open(src, "rb").read()
    build_image(sys.argv[1], contents)
