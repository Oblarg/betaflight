TARGET_MCU        := SIMULATOR
TARGET_MCU_FAMILY := SITL
SIMULATOR_BUILD    = yes

TARGET_SRC = \
            drivers/accgyro/accgyro_virtual.c \
            drivers/barometer/barometer_virtual.c \
            drivers/compass/compass_virtual.c \
            drivers/serial_tcp.c \
            io/gps_virtual.c \
            blackbox/blackbox_virtual.c \
            drivers/flash/flash.c \
            drivers/flash/flash_sitl.c \
            drivers/sdcard.c \
            drivers/sdcard_sitl.c \
            io/asyncfatfs/asyncfatfs.c \
            io/asyncfatfs/fat_standard.c

SIZE_OPTIMISED_SRC += \
            drivers/serial_tcp.c
