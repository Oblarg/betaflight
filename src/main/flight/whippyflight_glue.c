/*
 * WhippyFlight LQR controller — C glue layer.
 *
 * Replaces BF's pidController() + mixTable() with a call into the Rust
 * whippyflight-control staticlib.  Handles unit conversion, special motor
 * modes (crashflip, disarm, failsafe, motor stop), and blackbox telemetry.
 */

#include "platform.h"

#ifdef USE_WHIPPYFLIGHT

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#ifdef SIMULATOR_BUILD
#include <stdio.h>
#endif

#include "blackbox/blackbox.h"

#include "build/debug.h"

#include "common/axis.h"
#include "common/maths.h"
#include "common/time.h"

#include "config/config.h"
#include "config/feature.h"

#include "drivers/dshot.h"
#include "drivers/flash/flash.h"
#include "drivers/sdcard.h"
#include "drivers/time.h"

#include "fc/core.h"
#include "fc/rc.h"
#include "fc/rc_controls.h"
#include "fc/rc_modes.h"
#include "fc/runtime_config.h"

#include "flight/autopilot.h"
#include "flight/failsafe.h"
#include "flight/mixer.h"
#include "flight/mixer_init.h"
#include "flight/pid.h"
#include "flight/whippyflight_glue.h"

#include "io/asyncfatfs/asyncfatfs.h"

#include "pg/motor.h"
#include "pg/rx.h"
#include "pg/sdcard.h"
#include "pg/whippyflight.h"

#include "rx/crsf.h"
#include "rx/rx.h"

#include "sensors/acceleration.h"
#include "sensors/battery.h"
#include "sensors/gyro.h"
#include "sensors/sensors.h"

// ---------------------------------------------------------------------------
// Sensor helpers.  The controller takes BF's values in BF's units (see
// whippyflight_glue.h); SITL has neither bidirectional DShot nor a voltage
// meter, and the simulator's ESC telemetry stands in for both in the same
// units as the getters it replaces.
// ---------------------------------------------------------------------------

// Motor speed as getDshotErpm() reports it: eRPM/100.
static float getMotorErpm(int motorIndex)
{
#if defined(SIMULATOR_BUILD)
    return sitlGetMotorErpm(motorIndex);
#elif defined(USE_DSHOT_TELEMETRY)
    return getDshotErpm(motorIndex);
#else
    UNUSED(motorIndex);
    return 0.0f;
#endif
}

// The latest battery reading, centivolts: BF's display filter lags a sag by
// seconds, and the controller filters it itself (whippyflight_vbat_lpf_hz).
static float getBatteryCentivolts(void)
{
#ifdef SIMULATOR_BUILD
    return sitlGetBatteryVoltage();
#else
    return getBatteryVoltageLatest();
#endif
}

// Aligned, low-passed accelerometer (what blackbox logs as accSmooth) in g;
// the airframe drag fit regresses on the same signal.
static void getAccelG(float out[3])
{
#ifdef USE_ACC
    if (sensors(SENSOR_ACC) && acc.dev.acc_1G) {
        for (int axis = 0; axis < XYZ_AXIS_COUNT; axis++) {
            out[axis] = acc.accADC.v[axis] / acc.dev.acc_1G;
        }
        return;
    }
#endif
    out[0] = out[1] = out[2] = 0.0f;
}

// ---------------------------------------------------------------------------
// Main entry point: replaces subTaskPidController + subTaskMotorUpdate.
// ---------------------------------------------------------------------------

static const wf_bf_telemetry_t *lastTelemetry;

#if defined(USE_SERIALRX_CRSF)
// Whether the stick setpoint is fed from CRSF's frame queue, so no frame
// waits on the RX task.  Not while BF rewrites the sticks between frame and
// setpoint: failsafe, MSP override, autopilot yaw, the camera-angle mix.
static bool stickFramesDirect(void)
{
    return rxRuntimeState.rxProvider == RX_PROVIDER_SERIAL
        && rxRuntimeState.serialrxProvider == SERIALRX_CRSF
        && !failsafeIsActive()
        && !IS_RC_MODE_ACTIVE(BOXFAILSAFE)
#ifdef USE_RX_MSP_OVERRIDE
        && !(rxConfig()->msp_override_channels_mask && IS_RC_MODE_ACTIVE(BOXMSPOVERRIDE))
#endif
        && !autopilotYawControlActive()
        && !(rxConfig()->fpvCamAngleDegrees && IS_RC_MODE_ACTIVE(BOXFPVANGLEMIX) && !FLIGHT_MODE(HEADFREE_MODE));
}

// A queued frame's raw rate setpoint; false when a stick channel is invalid.
static bool stickFrameSetpoint(const float channelsUs[CRSF_MAX_CHANNEL], float rate[3])
{
    for (int axis = FD_ROLL; axis <= FD_YAW; axis++) {
        float channel;
        if (!rxStickChannelNorm(axis, channelsUs[rxConfig()->rcmap[axis]], &channel)) {
            return false;
        }
        rate[axis] = rcStickSetpoint(axis, channel);
    }
    return true;
}
#endif

const wf_bf_telemetry_t *whippyflightTelemetry(void)
{
    return lastTelemetry;
}

void whippyflightPidAndMotorUpdate(timeUs_t currentTimeUs)
{
    uint32_t startTime = 0;
    if (debugMode == DEBUG_PIDLOOP) {
        startTime = micros();
    }

    // -----------------------------------------------------------------------
    // 1. Crashflip — bypasses everything, writes motor[] directly.
    //    Uses BF's existing crashflip logic (needs motor output endpoints).
    // -----------------------------------------------------------------------

    if (mixerApplyCrashFlipIfActive(currentTimeUs)) {
        writeMotors();
        DEBUG_SET(DEBUG_PIDLOOP, 1, micros() - startTime);  //!< PID Controller Time [unit:us]
        return;
    }

    // -----------------------------------------------------------------------
    // 2. Gather sensor data, in BF's units.
    // -----------------------------------------------------------------------

    wf_sensor_data_t sensors = {
        .gyro = {
            gyro.gyroADCf[FD_ROLL],
            gyro.gyroADCf[FD_PITCH],
            gyro.gyroADCf[FD_YAW],
        },
        .motor_erpm = {
            getMotorErpm(0),
            getMotorErpm(1),
            getMotorErpm(2),
            getMotorErpm(3),
        },
        .battery = getBatteryCentivolts(),
        .motor_poles = motorConfig()->motorPoleCount,
    };
    getAccelG(sensors.accel_g);

    // The controller reconstructs the rate setpoint from raw RX frames itself,
    // so BF's rc_smoothing output is not used here.
    static uint32_t lastRcFrameCount;
    const uint32_t rcFrameCount = getRcFrameCount();
    const bool rcFrameNew = rcFrameCount != lastRcFrameCount;
    lastRcFrameCount = rcFrameCount;
    bool direct = false;
#if defined(USE_SERIALRX_CRSF)
    direct = stickFramesDirect();
    float channelsUs[CRSF_MAX_CHANNEL];
    timeUs_t frameTimeUs;
    while (crsfPopRcFrame(channelsUs, &frameTimeUs)) {
        float rate[3];
        if (direct && stickFrameSetpoint(channelsUs, rate)) {
            whippyflight_setpoint_packet(rate, frameTimeUs, getCurrentRxRateHz());
        }
    }
#endif
    if (rcFrameNew && !direct) {
        const float rawRate[3] = {
            getRawSetpoint(FD_ROLL),
            getRawSetpoint(FD_PITCH),
            getRawSetpoint(FD_YAW),
        };
        whippyflight_setpoint_packet(rawRate, getRcFrameTimeUs(), getCurrentRxRateHz());
    }

    // -----------------------------------------------------------------------
    // 3. Run the Rust LQR controller.
    // -----------------------------------------------------------------------

    whippyflight_set_min_speed(whippyflightConfig()->idle_rps);
    const uint16_t criticalCell = whippyflightConfig()->critical_cell_voltage;
    whippyflight_set_min_battery(getBatteryCellCount()
        * (criticalCell ? criticalCell : currentBatteryProfile->vbatwarningcellvoltage));
    whippyflight_set_motor_lpf(whippyflightConfig()->rpm_lpf_hz);
    whippyflight_set_battery_lpf(whippyflightConfig()->vbat_lpf_hz);
    whippyflight_set_brake_margin(whippyflightConfig()->brake_margin);
    whippyflight_set_dropout_ramp(whippyflightConfig()->dropout_ramp_ms);
    whippyflight_set_schedule_every_tick(whippyflightConfig()->schedule_every_tick);
    whippyflight_set_effort_trace(debugMode == DEBUG_WHIPPYFLIGHT);
    const wf_control_output_t output = whippyflight_update(&sensors, rcCommand[THROTTLE], currentTimeUs);

    // -----------------------------------------------------------------------
    // 4. Convert duty cycle [0,1] → motor[] in DShot protocol range.
    // -----------------------------------------------------------------------

    const float motorLow  = getMotorOutputLow();
    const float motorHigh = getMotorOutputHigh();
    const float motorRange = motorHigh - motorLow;

    const int motorCount = getMotorCount();
    for (int i = 0; i < motorCount; i++) {
        float duty = (i < 4) ? output.motor_duty[i] : 0.0f;
        motor[i] = motorLow + constrainf(duty, 0.0f, 1.0f) * motorRange;
    }

    // -----------------------------------------------------------------------
    // 5. Special mode overrides (applied AFTER the controller output).
    //    Priority order matches BF's mixer.c conventions.
    // -----------------------------------------------------------------------

    // 5a. Failsafe output clamping.
    if (failsafeIsActive()) {
        const float disarmOutput = getMotorDisarmOutput();
        for (int i = 0; i < motorCount; i++) {
#ifdef USE_DSHOT
            if (isMotorProtocolDshot()) {
                motor[i] = (motor[i] < motorLow)
                    ? disarmOutput : motor[i];
            }
#endif
            motor[i] = constrainf(motor[i], disarmOutput, motorHigh);
        }
    }

    // 5b. Motor stop: if FEATURE_MOTOR_STOP is enabled, no airmode, and
    //     throttle is below mincheck → stop all motors.
    if (featureIsEnabled(FEATURE_MOTOR_STOP)
        && !isAirmodeEnabled()
        && (calculateThrottleStatus() == THROTTLE_LOW)
        && ARMING_FLAG(ARMED)) {
        const float disarmOutput = getMotorDisarmOutput();
        for (int i = 0; i < motorCount; i++) {
            motor[i] = disarmOutput;
        }
    }

    // 5c. Disarm override — must be last, always wins.
    if (!ARMING_FLAG(ARMED)) {
        for (int i = 0; i < motorCount; i++) {
            motor[i] = motor_disarmed[i];
        }
    }

    // -----------------------------------------------------------------------
    // 6. Populate blackbox-compatible telemetry.
    //
    //    We write our LQR telemetry into pidData[] so the existing blackbox
    //    infrastructure logs it without modification.  The semantics differ
    //    from BF's PID, but the fields are still useful for tuning:
    //      P = per-axis rate error (0.1 deg/s)
    //      I = per-axis integral state / dt: the low-passed rate error (0.1 deg/s)
    //      D = per-axis rate-error derivative (deg/s²)
    //      F = per-axis setpoint estimate before shaping (0.1 deg/s)
    //    The setpoint blackbox logs (pidRuntime.previousPidSetpoint) is the
    //    shaped target the feedforward inverts; blackbox itself logs the raw
    //    RX setpoint as rcCommand[0..2] and the command scale as setpoint[3]
    //    (whippyflightTelemetry).
    //    Motor[] is already populated above.  The Rust side encodes every
    //    field in the unit it is logged in, so nothing is scaled here.
    // -----------------------------------------------------------------------

    const wf_bf_telemetry_t *telem = whippyflight_telemetry();
    lastTelemetry = telem;
    if (telem != NULL) {
        for (int axis = 0; axis < XYZ_AXIS_COUNT; axis++) {
            pidData[axis].P = telem->pid_p[axis];
            pidData[axis].I = telem->pid_i[axis];
            pidData[axis].D = telem->pid_d[axis];
            pidData[axis].F = telem->pid_f[axis];
            pidData[axis].Sum = telem->pid_p[axis]; // best single summary
            pidRuntime.previousPidSetpoint[axis] = telem->pid_setpoint[axis];
        }
    } else {
        for (int axis = 0; axis < XYZ_AXIS_COUNT; axis++) {
            pidData[axis].P = 0.0f;
            pidData[axis].I = 0.0f;
            pidData[axis].D = 0.0f;
            pidData[axis].F = 0.0f;
            pidData[axis].Sum = 0.0f;
            pidRuntime.previousPidSetpoint[axis] = 0.0f;
        }
    }

    // LQR-specific telemetry via debug channels.
    if (telem != NULL && debugMode == DEBUG_NONE) {
        // Repurpose debug slots for LQR feedback voltage (motor 0–3).
        DEBUG_SET(DEBUG_NONE, 0, telem->feedback[0]);  //!< Feedback Voltage Motor 0 [unit:0.001V]
        DEBUG_SET(DEBUG_NONE, 1, telem->feedback[1]);  //!< Feedback Voltage Motor 1 [unit:0.001V]
        DEBUG_SET(DEBUG_NONE, 2, telem->feedback[2]);  //!< Feedback Voltage Motor 2 [unit:0.001V]
        DEBUG_SET(DEBUG_NONE, 3, telem->feedback[3]);  //!< Feedback Voltage Motor 3 [unit:0.001V]
    }

    // One motor's control effort (whippyflight_debug_motor): the command is
    // slots 0–4 summed, and slot 5 is the part of 0–2 the gain schedule's
    // fast motion adds.
    if (telem != NULL && debugMode == DEBUG_WHIPPYFLIGHT) {
        const wf_bf_effort_t *effort = &telem->effort[whippyflightConfig()->debug_motor & 3];
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 0, effort->p);               //!< Rate Error Feedback Voltage [unit:0.001V]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 1, effort->i);               //!< Integral Feedback Voltage [unit:0.001V]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 2, effort->d);               //!< Derivative Feedback Voltage [unit:0.001V]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 3, effort->feedforward);     //!< Feedforward Voltage [unit:0.001V]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 4, effort->affine);          //!< Affine Linearizer Voltage [unit:0.001V]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 5, effort->schedule_jitter); //!< Gain Schedule Jitter Voltage [unit:0.001V]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 6, effort->applied);         //!< Applied Motor Voltage [unit:0.001V]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 7, effort->filtered_speed);  //!< Filtered Motor Speed [unit:0.1rev/s]
    }

    // -----------------------------------------------------------------------
    // 7. Write motors to hardware.
    // -----------------------------------------------------------------------

    writeMotors();

    DEBUG_SET(DEBUG_PIDLOOP, 1, micros() - startTime);  //!< PID Controller Time [unit:us]
}

// ---------------------------------------------------------------------------
// Gain loading → RAM.
//
// Neither store is fast enough to serve the tables every loop, and the MCU's
// own flash has no wear levelling, so the blob is copied once at boot into
// RAM; the Rust side reads its tables in place from this buffer for the rest
// of the run.  Sources, in priority order:
//   1. WHIPPYFLIGHT_GAIN_FILENAME in the SD card's root directory;
//   2. the dataflash WHIPPYFLIGHT partition.
// A gain file that is present but invalid is reported as such rather than
// falling back, so stale gains are never flown by accident.
// ---------------------------------------------------------------------------

#define WF_GAIN_MAGIC         0x4C515247u   // "LQRG"
#define WF_GAIN_HEADER_SIZE   0x20
#define WF_GAIN_DT_OFFSET     0x10
#define WF_GAIN_SIZE_OFFSET   0x1C
#define WF_GAIN_READ_CHUNK    256u

// Allowed mismatch between the blob's discretisation dt and the PID loop.
#define WF_GAIN_DT_TOLERANCE  0.01f

static uint8_t wfGainRam[WHIPPYFLIGHT_GAIN_STORE_SIZE] __attribute__((aligned(4)));
static wf_gain_load_result_t wfGainResult = WF_GAIN_NO_STORE;
static wf_gain_source_t wfGainSource = WF_GAIN_SOURCE_NONE;
static uint32_t wfGainSize;
static float wfGainDt;

// Validate the header at the start of wfGainRam, given how many bytes the
// store can hold.  On success *size is the blob's total size.
static wf_gain_load_result_t checkGainHeader(uint32_t capacity, uint32_t *size)
{
    uint32_t magic;
    memcpy(&magic, wfGainRam, sizeof(magic));
    memcpy(size, wfGainRam + WF_GAIN_SIZE_OFFSET, sizeof(*size));

    // An erased partition reads 0xFF: no blob installed.
    if (magic != WF_GAIN_MAGIC) {
        return WF_GAIN_BAD_MAGIC;
    }
    if (*size > capacity) {
        return WF_GAIN_TOO_LARGE;
    }
    if (*size < WF_GAIN_HEADER_SIZE + 4) {
        return WF_GAIN_TOO_SHORT;
    }
    return WF_GAIN_OK;
}

#ifdef USE_SDCARD
#define WF_SD_TIMEOUT_MS      3000

static afatfsFilePtr_t wfSdFile;
static bool wfSdOpened;

static void gainFileOpened(afatfsFilePtr_t file)
{
    wfSdFile = file;
    wfSdOpened = true;
}

static void gainFileClosed(void)
{
}

// Poll asyncfatfs until done() or the boot deadline; false on timeout or a
// dead card.
static bool sdPollUntil(bool (*done)(void), timeMs_t deadline)
{
    while (!done()) {
        if (!sdcard_isFunctional() || afatfs_getFilesystemState() == AFATFS_FILESYSTEM_STATE_FATAL
            || cmp32(millis(), deadline) > 0) {
            return false;
        }
        afatfs_poll();
    }
    return true;
}

static bool sdFilesystemReady(void)
{
    return afatfs_getFilesystemState() == AFATFS_FILESYSTEM_STATE_READY;
}

static bool sdFileOpened(void)
{
    return wfSdOpened;
}

static wf_gain_load_result_t loadGainsFromSdCard(uint32_t *blobSize)
{
    if (sdcardConfig()->mode == SDCARD_MODE_NONE || !sdcard_isInserted()) {
        return WF_GAIN_NO_STORE;
    }

    const timeMs_t deadline = millis() + WF_SD_TIMEOUT_MS;
    wfSdFile = NULL;
    wfSdOpened = false;
    if (!sdPollUntil(sdFilesystemReady, deadline)) {
        return WF_GAIN_NO_STORE;
    }
    // Read-only mode never creates the file; a missing file opens as NULL.
    afatfs_fopen(WHIPPYFLIGHT_GAIN_FILENAME, "r", gainFileOpened);
    if (!sdPollUntil(sdFileOpened, deadline) || !wfSdFile) {
        return WF_GAIN_NO_STORE;
    }

    uint32_t length = 0;
    while (length < sizeof(wfGainRam) && !afatfs_feof(wfSdFile) && cmp32(millis(), deadline) <= 0) {
        const uint32_t read = afatfs_fread(wfSdFile, wfGainRam + length, sizeof(wfGainRam) - length);
        length += read;
        if (read == 0) {
            afatfs_poll();
        }
    }
    while (!afatfs_fclose(wfSdFile, gainFileClosed) && cmp32(millis(), deadline) <= 0) {
        afatfs_poll();
    }

    if (length < WF_GAIN_HEADER_SIZE) {
        return WF_GAIN_TOO_SHORT;
    }
    // The header's size is authoritative; trailing bytes in the file are
    // ignored, a truncated file is too short.
    wf_gain_load_result_t result = checkGainHeader(sizeof(wfGainRam), blobSize);
    if (result == WF_GAIN_OK && *blobSize > length) {
        result = WF_GAIN_TOO_SHORT;
    }
    return result;
}
#endif

#ifdef USE_FLASH_CHIP
// The dataflash gain partition: byte address and usable size (capped at the
// gain RAM).  False if the board has none.
static bool gainPartition(uint32_t *base, uint32_t *capacity)
{
    const flashPartition_t *partition = flashPartitionFindByType(FLASH_PARTITION_TYPE_WHIPPYFLIGHT);
    const flashGeometry_t *geometry = flashGetGeometry();
    if (!partition || geometry->totalSize == 0) {
        return false;
    }
    *base = partition->startSector * geometry->sectorSize;
    *capacity = MIN(FLASH_PARTITION_SECTOR_COUNT(partition) * geometry->sectorSize,
                    (uint32_t)sizeof(wfGainRam));
    return true;
}

static bool readGainFlash(uint32_t address, uint8_t *buffer, uint32_t length)
{
    while (length > 0) {
        const int read = flashReadBytes(address, buffer, MIN(length, WF_GAIN_READ_CHUNK));
        if (read <= 0) {
            return false;
        }
        address += read;
        buffer += read;
        length -= read;
    }
    return true;
}
#endif

static wf_gain_load_result_t loadGainsFromFlash(uint32_t *blobSize)
{
#ifdef USE_FLASH_CHIP
    uint32_t base, capacity;
    if (!gainPartition(&base, &capacity)) {
        return WF_GAIN_NO_STORE;
    }

    if (!readGainFlash(base, wfGainRam, WF_GAIN_HEADER_SIZE)) {
        return WF_GAIN_TOO_SHORT;
    }
    const wf_gain_load_result_t result = checkGainHeader(capacity, blobSize);
    if (result != WF_GAIN_OK) {
        return result;
    }
    if (!readGainFlash(base + WF_GAIN_HEADER_SIZE, wfGainRam + WF_GAIN_HEADER_SIZE,
                       *blobSize - WF_GAIN_HEADER_SIZE)) {
        return WF_GAIN_TOO_SHORT;
    }
    return WF_GAIN_OK;
#else
    UNUSED(blobSize);
    return WF_GAIN_NO_STORE;
#endif
}

// Hand the blob staged in wfGainRam to the controller and record the outcome.
static wf_gain_load_result_t initGains(wf_gain_load_result_t result, uint32_t blobSize, wf_gain_source_t source)
{
    if (result == WF_GAIN_OK) {
        result = whippyflight_init(wfGainRam, blobSize,
                                   mixerConfig()->mixerMode, mixerConfig()->yaw_motors_reversed);
    }
    if (result == WF_GAIN_OK) {
        memcpy(&wfGainDt, wfGainRam + WF_GAIN_DT_OFFSET, sizeof(wfGainDt));
        wfGainSize = blobSize;
    } else {
        wfGainSize = 0;
    }
    wfGainResult = result;
    wfGainSource = source;
    return result;
}

static bool wfActive;

void whippyflightInit(void)
{
    wfActive = whippyflightConfig()->enabled;
    whippyflightLoadGains();
}

bool whippyflightIsActive(void)
{
    return wfActive;
}

wf_gain_load_result_t whippyflightLoadGains(void)
{
    whippyflight_unload();
    uint32_t blobSize = 0;
    wf_gain_source_t source = WF_GAIN_SOURCE_SDCARD;
    wf_gain_load_result_t result = WF_GAIN_NO_STORE;
#ifdef USE_SDCARD
    result = loadGainsFromSdCard(&blobSize);
#endif
    if (result == WF_GAIN_NO_STORE) {
        source = WF_GAIN_SOURCE_DATAFLASH;
        result = loadGainsFromFlash(&blobSize);
    }
    result = initGains(result, blobSize, source);
#ifdef SIMULATOR_BUILD
    printf("[WHIPPYFLIGHT] gain load from %s: result=%d, %u bytes\n",
           source == WF_GAIN_SOURCE_SDCARD ? "sdcard" : "dataflash", (int)result, (unsigned)wfGainSize);
#endif
    return result;
}

// ---------------------------------------------------------------------------
// Gain upload.
// ---------------------------------------------------------------------------

static bool wfUploading;
static uint32_t wfUploadSize;
static uint32_t wfUploadReceived;

void whippyflightGainStatus(wf_gain_status_t *status)
{
    status->result = wfGainResult;
    status->source = wfGainSource;
    status->blobSize = wfGainSize;
    status->blobCrc = 0;
    status->blobDt = 0.0f;
    if (wfGainSize > 0) {
        memcpy(&status->blobCrc, wfGainRam + wfGainSize - 4, sizeof(status->blobCrc));
        status->blobDt = wfGainDt;
    }
    status->partitionSize = 0;
#ifdef USE_FLASH_CHIP
    uint32_t base;
    if (!gainPartition(&base, &status->partitionSize)) {
        status->partitionSize = 0;
    }
#endif
    status->uploading = wfUploading;
    status->uploadSize = wfUploadSize;
    status->uploadReceived = wfUploadReceived;
}

wf_upload_result_t whippyflightUploadBegin(uint32_t size)
{
    if (ARMING_FLAG(ARMED)) {
        return WF_UPLOAD_ARMED;
    }
#ifdef USE_FLASH_CHIP
    uint32_t base, capacity;
    if (!gainPartition(&base, &capacity)) {
        return WF_UPLOAD_NO_STORE;
    }
    if (size > capacity) {
        return WF_UPLOAD_TOO_LARGE;
    }
    if (size < WF_GAIN_HEADER_SIZE + 4) {
        return WF_UPLOAD_INCOMPLETE;
    }
    // The staged blob overwrites the tables the controller reads in place.
    whippyflight_unload();
    wfGainResult = WF_GAIN_NO_STORE;
    wfGainSize = 0;
    wfUploading = true;
    wfUploadSize = size;
    wfUploadReceived = 0;
    return WF_UPLOAD_OK;
#else
    UNUSED(size);
    return WF_UPLOAD_NO_STORE;
#endif
}

wf_upload_result_t whippyflightUploadData(uint32_t offset, const uint8_t *data, uint32_t length)
{
    if (ARMING_FLAG(ARMED)) {
        return WF_UPLOAD_ARMED;
    }
    if (!wfUploading) {
        return WF_UPLOAD_NOT_STARTED;
    }
    if (offset != wfUploadReceived || length > wfUploadSize - wfUploadReceived) {
        return WF_UPLOAD_OUT_OF_ORDER;
    }
    memcpy(wfGainRam + offset, data, length);
    wfUploadReceived += length;
    return WF_UPLOAD_OK;
}

void whippyflightUploadAbort(void)
{
    if (wfUploading) {
        wfUploading = false;
        whippyflightLoadGains();
    }
}

#ifdef USE_FLASH_CHIP
// Erase the sectors the blob occupies, program it page by page, and read it
// back against the RAM copy.
static bool writeGainFlash(uint32_t base, uint32_t size)
{
    const flashGeometry_t *geometry = flashGetGeometry();
    for (uint32_t offset = 0; offset < size; offset += geometry->sectorSize) {
        flashEraseSector(base + offset);
    }
    for (uint32_t offset = 0; offset < size; offset += geometry->pageSize) {
        flashPageProgram(base + offset, wfGainRam + offset, MIN(geometry->pageSize, size - offset), NULL);
        if (!flashWaitForReady()) {
            return false;
        }
    }
    flashFlush();

    uint8_t check[WF_GAIN_READ_CHUNK];
    for (uint32_t offset = 0; offset < size; offset += sizeof(check)) {
        const uint32_t length = MIN((uint32_t)sizeof(check), size - offset);
        if (!readGainFlash(base + offset, check, length) || memcmp(check, wfGainRam + offset, length) != 0) {
            return false;
        }
    }
    return true;
}
#endif

wf_upload_result_t whippyflightUploadCommit(wf_gain_load_result_t *gainResult)
{
    *gainResult = wfGainResult;
    if (ARMING_FLAG(ARMED)) {
        return WF_UPLOAD_ARMED;
    }
    if (!wfUploading) {
        return WF_UPLOAD_NOT_STARTED;
    }
    if (wfUploadReceived != wfUploadSize) {
        return WF_UPLOAD_INCOMPLETE;
    }
#ifdef USE_FLASH_CHIP
#ifdef USE_BLACKBOX
    // Blackbox shares the dataflash (blackbox_mode ALWAYS logs disarmed).
    if (!blackboxMayEditConfig()) {
        return WF_UPLOAD_BUSY;
    }
#endif
    uint32_t base, capacity;
    if (!gainPartition(&base, &capacity)) {
        return WF_UPLOAD_NO_STORE;
    }
    wfUploading = false;
    *gainResult = initGains(WF_GAIN_OK, wfUploadSize, WF_GAIN_SOURCE_DATAFLASH);
    if (*gainResult != WF_GAIN_OK) {
        // The stored gains were never touched; fly those instead.
        whippyflightLoadGains();
        return WF_UPLOAD_REJECTED;
    }
    if (!writeGainFlash(base, wfUploadSize)) {
        return WF_UPLOAD_VERIFY_FAILED;
    }
#ifdef SIMULATOR_BUILD
    printf("[WHIPPYFLIGHT] gain upload: %u bytes written to dataflash\n", (unsigned)wfUploadSize);
#endif
    return WF_UPLOAD_OK;
#else
    return WF_UPLOAD_NO_STORE;
#endif
}

uint16_t whippyflightArmingChecks(void)
{
    uint16_t failing = 0;
    if (wfUploading) {
        failing |= WF_CHECK_UPLOADING;
    }
    if (wfGainResult != WF_GAIN_OK) {
        failing |= WF_CHECK_GAINS;
    } else {
        // The gains are discretised at the blob's dt.
        const float pidDt = pidGetDT();
        if (fabsf(pidDt - wfGainDt) > WF_GAIN_DT_TOLERANCE * wfGainDt) {
            failing |= WF_CHECK_LOOP_RATE;
        }
    }
    // The gains are scheduled on measured motor speed.
#if defined(SIMULATOR_BUILD)
    if (!sitlHasEscTelemetry()) {
        failing |= WF_CHECK_RPM_TELEMETRY;
    }
#elif defined(USE_DSHOT_TELEMETRY)
    if (!useDshotTelemetry) {
        failing |= WF_CHECK_RPM_TELEMETRY;
    }
#endif
    // Self-levelling setpoints come from pidController(), which this
    // controller replaces; these modes would silently fly as acro.
    if (isModeActivationConditionPresent(BOXANGLE)
        || isModeActivationConditionPresent(BOXHORIZON)
        || isModeActivationConditionPresent(BOXALTHOLD)
        || isModeActivationConditionPresent(BOXPOSHOLD)
        || isModeActivationConditionPresent(BOXGPSRESCUE)
        || isModeActivationConditionPresent(BOXACROTRAINER)) {
        failing |= WF_CHECK_SELF_LEVEL;
    }
    // Every other failsafe procedure flies in ANGLE_MODE.
    if (failsafeConfig()->failsafe_procedure != FAILSAFE_PROCEDURE_DROP_IT) {
        failing |= WF_CHECK_FAILSAFE;
    }
    return failing;
}

const char *whippyflightArmingBlockReason(void)
{
    const uint16_t failing = whippyflightArmingChecks();
    if (failing & WF_CHECK_UPLOADING) {
        return "gain upload in progress";
    }
    if (failing & WF_CHECK_GAINS) {
        return "gains not loaded";
    }
    if (failing & WF_CHECK_LOOP_RATE) {
        return "PID loop rate does not match gain blob dt";
    }
    if (failing & WF_CHECK_RPM_TELEMETRY) {
#if defined(SIMULATOR_BUILD)
        return "simulator sends no ESC telemetry";
#else
        return "dshot_bidir off";
#endif
    }
    if (failing & WF_CHECK_SELF_LEVEL) {
        return "self-level mode configured (angle/horizon/althold/poshold/rescue/trainer)";
    }
    if (failing & WF_CHECK_FAILSAFE) {
        return "failsafe_procedure is not DROP";
    }
    return NULL;
}

bool whippyflightIsReadyToArm(void)
{
    return whippyflightArmingChecks() == 0;
}

#endif // USE_WHIPPYFLIGHT
