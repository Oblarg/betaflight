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

// SITL (software in the loop) simulator

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "common/utils.h"
#include "drivers/serial_uart_types.h"

#define TARGET_BOARD_IDENTIFIER "SITL"

// SITL bridge selector. Configs that target legacy bridges (X-Plane,
// RealFlight) define ENABLE_GAZEBO_BRIDGE=0 in config.h; otherwise default
// to Gazebo so the bare TARGET=SITL build behaves like the SITL_GAZEBO config.
#ifndef ENABLE_GAZEBO_BRIDGE
#define ENABLE_GAZEBO_BRIDGE  1
#endif

#define ENABLE_SIMULATOR_MULTITHREAD 1

#define SYSTEM_HSE_MHZ 0
#define DEFAULT_CPU_OVERCLOCK 1
#define DMA_RAM
#define DMA_RW_AXI
#define DMA_RAM_R
#define DMA_RAM_W
#define DMA_RAM_RW

#define DMA_DATA_ZERO_INIT
#define DMA_DATA
#define STATIC_DMA_DATA_AUTO

// SITL runs the real attitude estimator (Mahony + mag/COG fusion) against the
// virtual gyro/acc/mag feeds. Build with -DSITL_ATTITUDE_DIRECT to bypass it
// and set attitude straight from the FDM quaternion (legacy behaviour).
#if defined(SITL_ATTITUDE_DIRECT)
#undef USE_IMU_CALC
#endif

//#define ENABLE_SIMULATOR_ACC_SYNC 1
//#define ENABLE_SIMULATOR_GYRO_SYNC 1
//#define ENABLE_SIMULATOR_IMU_SYNC 1
//#define ENABLE_SIMULATOR_GYROPID_SYNC 1

// WhippyFlight's gains are discretised at the PID loop period, so the
// simulator has to step the plant at that period with exactly one PID loop
// per step: run in lockstep, one loop per fdm_packet.
#if defined(USE_WHIPPYFLIGHT) && !defined(ENABLE_SIMULATOR_GYROPID_SYNC)
#define ENABLE_SIMULATOR_GYROPID_SYNC 1
#endif

// file name to save config
#define EEPROM_FILENAME "eeprom.bin"
#define CONFIG_IN_FILE
#define EEPROM_SIZE     32768

#define U_ID_0 0
#define U_ID_1 1
#define U_ID_2 2

#undef TASK_GYROPID_DESIRED_PERIOD
#define TASK_GYROPID_DESIRED_PERIOD     100

#undef SCHEDULER_DELAY_LIMIT
#define SCHEDULER_DELAY_LIMIT           1

// OS preemption spikes must not peak-hold task duration estimates
// (see scheduler.h) or non-realtime tasks starve to a few hertz
#define TASK_EXEC_TIME_CLAMP_US         100

#define USE_VIRTUAL_LED

#define USE_ACC
#define USE_VIRTUAL_ACC

#define USE_GYRO
#define USE_VIRTUAL_GYRO

#define USE_MAG
#define USE_VIRTUAL_MAG

#define USE_BARO
#define USE_VIRTUAL_BARO

#define USABLE_TIMER_CHANNEL_COUNT 0

#define USE_UART1
#define USE_UART2
#define USE_UART3
#define USE_UART4
#define USE_UART5
#define USE_UART6
#define USE_UART7
#define USE_UART8

#define ENABLE_RX_UDP           1

// DEFAULT_RX_FEATURE is picked by config/feature.h: FEATURE_RX_UDP when
// ENABLE_RX_UDP=1 (bare SITL / Gazebo), else FEATURE_RX_MSP. Configs that want
// MSP RX as the default (e.g. SITL_X_PLANE) override DEFAULT_RX_FEATURE.
#define DEFAULT_FEATURES        (FEATURE_GPS | FEATURE_TELEMETRY)

#define USE_GPS
#define USE_VIRTUAL_GPS
#define USE_ALTITUDE_HOLD
#define USE_POSITION_HOLD
#define USE_FLIGHT_PLAN

#define USE_PARAMETER_GROUPS

#define USE_MSP_CLI_COMMAND

#ifndef USE_PWM_OUTPUT
#define USE_PWM_OUTPUT
#endif

#define USE_BLACKBOX
#define USE_BLACKBOX_VIRTUAL

#undef USE_STACK_CHECK // I think SITL don't need this
#undef USE_DASHBOARD
#undef USE_TELEMETRY_LTM
#undef USE_ADC
#undef USE_VCP
#undef USE_OSD
#undef USE_RX_PPM
#undef USE_RX_PWM
#undef USE_SERIALRX
#undef USE_SERIALRX_CRSF
#undef USE_SERIALRX_GHST
#undef USE_SERIALRX_IBUS
#undef USE_SERIALRX_SBUS
#undef USE_SERIALRX_SPEKTRUM
#undef USE_SERIALRX_SUMD
#undef USE_SERIALRX_SUMH
#undef USE_SERIALRX_XBUS
#undef USE_LED_STRIP
#undef USE_TELEMETRY_FRSKY_HUB
#undef USE_TELEMETRY_HOTT
#undef USE_TELEMETRY_SMARTPORT
#undef USE_RESOURCE_MGMT
#undef USE_CMS
#undef USE_TELEMETRY_CRSF
#undef USE_TELEMETRY_GHST
#undef USE_TELEMETRY_IBUS
#undef USE_TELEMETRY_JETIEXBUS
#undef USE_TELEMETRY_SRXL
#undef USE_SERIALRX_JETIEXBUS
#undef USE_VTX_COMMON
#undef USE_VTX_CONTROL
#undef USE_VTX_SMARTAUDIO
#undef USE_VTX_TRAMP
#undef USE_VTX_MSP
#undef USE_CAMERA_CONTROL
#undef USE_SERIAL_4WAY_BLHELI_BOOTLOADER
#undef USE_SERIAL_4WAY_SK_BOOTLOADER

#undef USE_I2C
#undef USE_SPI

#define TARGET_FLASH_SIZE 2048

#ifdef USE_WHIPPYFLIGHT
// Gain storage emulation: file-backed dataflash (dataflash.bin) holding the
// gain partition, and a file-backed SD card (sdcard.img) for WHIPPY.WFG.
// The gain store keeps the hardware default size so oversized blobs fail
// here too.
#define USE_FLASH_CHIP
#define USE_FLASH_SITL
#define USE_SDCARD
#define USE_SDCARD_SITL
// Gain blobs are discretised for the PID loop rate and arming is refused on
// a mismatch; run the virtual gyro (and PID loop) at the 4 kHz the blobs
// target on hardware.
#define VIRTUAL_GYRO_SAMPLE_RATE_HZ 4000
#endif

#define DEFIO_NO_PORTS   // suppress 'no pins defined' warning

#define FLASH_PAGE_SIZE (0x400)

// belows are internal stuff

extern uint32_t SystemCoreClock;

typedef enum
{
    Mode_TEST = 0x0,
    Mode_Out_PP = 0x10
} GPIO_Mode;

typedef enum {RESET = 0, SET = !RESET} FlagStatus, ITStatus;
typedef enum {DISABLE = 0, ENABLE = !DISABLE} FunctionalState;
typedef enum {TEST_IRQ = 0 } IRQn_Type;

typedef struct {
    void* test;
} DMA_TypeDef;

typedef struct {
    void* test;
} DMA_Channel_TypeDef;

typedef struct {
    void* test;
} DMA_InitTypeDef;

uint8_t DMA_GetFlagStatus(void *);
void DMA_Cmd(DMA_Channel_TypeDef*, FunctionalState );
void DMA_ClearFlag(uint32_t);

struct spiResource_s;
struct quadSpiResource_s;
struct octoSpiResource_s;

#define USART1 ((usartResource_t *)0x0001)
#define USART2 ((usartResource_t *)0x0002)
#define USART3 ((usartResource_t *)0x0003)
#define USART4 ((usartResource_t *)0x0004)
#define USART5 ((usartResource_t *)0x0005)
#define USART6 ((usartResource_t *)0x0006)
#define USART7 ((usartResource_t *)0x0007)
#define USART8 ((usartResource_t *)0x0008)

#define UART4 ((usartResource_t *)0x0004)
#define UART5 ((usartResource_t *)0x0005)
#define UART7 ((usartResource_t *)0x0007)
#define UART8 ((usartResource_t *)0x0008)

#define SIMULATOR_MAX_RC_CHANNELS   16
#define SIMULATOR_MAX_PWM_CHANNELS  16

struct i2cResource_s;

typedef struct {
    double timestamp;                   // in seconds
    double imu_angular_velocity_rpy[3]; // rad/s -> range: +/- 8192; +/- 2000 deg/se
    double imu_linear_acceleration_xyz[3];    // m/s/s NED, body frame -> sim 1G = 9.80665, FC 1G = 256
    double imu_orientation_quat[4];     //w, x, y, z
    double velocity_xyz[3];             // m/s, earth frame. ENU (Ve, Vn, Vup) for virtual GPS mode (USE_VIRTUAL_GPS)!
    double position_xyz[3];             // meters, NED from origin. Longitude, Latitude, Altitude (ENU) for virtual GPS mode (USE_VIRTUAL_GPS)!
    double pressure;
} fdm_packet;

typedef struct {
    double timestamp;                               // in seconds
    uint16_t channels[SIMULATOR_MAX_RC_CHANNELS];   // RC channels
} rc_packet;

typedef struct {
    float motor_speed[4];   // normal: [0.0, 1.0], 3D: [-1.0, 1.0]
} servo_packet;

// Lockstep (WhippyFlight) builds answer each fdm_packet with this: the
// servo_packet plus that PID loop's debug[] values, so a simulator can
// trace them per loop.
typedef struct {
    servo_packet servo;
    int16_t debug[8];       // build/debug.h debug[]
} servo_packet_debug;

typedef struct {
    uint16_t motorCount; // Count of motor in the PWM output.
    float pwm_output_raw[SIMULATOR_MAX_PWM_CHANNELS];   // Raw PWM from 1100 to 1900
} servo_packet_raw;

// An fdm_packet may be followed by ESC telemetry, for simulators that model
// the motors and battery; plain fdm_packets leave it unavailable.
typedef struct {
    fdm_packet fdm;
    double motor_erpm[4];               // electrical RPM, by motor index
    double battery_voltage;             // V
} fdm_packet_esc;

bool sitlHasEscTelemetry(void);
float sitlGetMotorErpm(int motorIndex);
float sitlGetBatteryVoltage(void);

uint64_t nanos64_real(void);
uint64_t micros64_real(void);
uint64_t millis64_real(void);
void delayMicroseconds_real(uint32_t us);
uint64_t micros64(void);
uint64_t millis64(void);

int lockMainPID(void);

int targetParseArgs(int argc, char * argv[]);
#ifdef CONFIG_IN_FILE
const char *targetGetConfigFile(void);
#endif
