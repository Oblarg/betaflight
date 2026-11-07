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

#include "flight/failsafe.h"
#include "flight/mixer.h"
#include "flight/mixer_init.h"
#include "flight/pid.h"
#include "flight/whippyflight_glue.h"

#include "io/asyncfatfs/asyncfatfs.h"

#include "pg/motor.h"
#include "pg/sdcard.h"

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

// Battery voltage as getBatteryVoltage() reports it: centivolts.
static float getBatteryCentivolts(void)
{
#ifdef SIMULATOR_BUILD
    return sitlGetBatteryVoltage();
#else
    return getBatteryVoltage();
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
    };
    getAccelG(sensors.accel_g);

    // The controller reconstructs the rate setpoint from raw RX frames itself,
    // so BF's rc_smoothing output is not used here.
    static uint32_t lastRcFrameCount;
    const uint32_t rcFrameCount = getRcFrameCount();
    if (rcFrameCount != lastRcFrameCount) {
        lastRcFrameCount = rcFrameCount;
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

    whippyflight_set_min_speed(currentPidProfile->dyn_idle_min_rpm, motorConfig()->motorPoleCount);
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
    //      I = per-axis integrated rate error (0.1 deg)
    //      D = per-axis rate-error derivative (deg/s²)
    //      F = 0 (feedforward is per-motor in our system)
    //    Motor[] is already populated above.  The Rust side encodes every
    //    field in the unit it is logged in, so nothing is scaled here.
    // -----------------------------------------------------------------------

    const wf_bf_telemetry_t *telem = whippyflight_telemetry();
    if (telem != NULL) {
        for (int axis = 0; axis < XYZ_AXIS_COUNT; axis++) {
            pidData[axis].P = telem->pid_p[axis];
            pidData[axis].I = telem->pid_i[axis];
            pidData[axis].D = telem->pid_d[axis];
            pidData[axis].F = 0.0f;
            pidData[axis].Sum = telem->pid_p[axis]; // best single summary
        }
    } else {
        for (int axis = 0; axis < XYZ_AXIS_COUNT; axis++) {
            pidData[axis].P = 0.0f;
            pidData[axis].I = 0.0f;
            pidData[axis].D = 0.0f;
            pidData[axis].F = 0.0f;
            pidData[axis].Sum = 0.0f;
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

    // Setpoint reconstruction, in 0.1 deg/s (the feedforward lead is only a
    // few deg/s): raw RX setpoint, the estimate the controller tracked, and
    // the feedforward target, for roll and pitch.
    if (telem != NULL && debugMode == DEBUG_WHIPPYFLIGHT) {
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 0, telem->raw_setpoint[FD_ROLL]);  //!< Raw Setpoint Roll [unit:0.1dps]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 1, telem->setpoint[FD_ROLL]);      //!< Setpoint Estimate Roll [unit:0.1dps]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 2, telem->ff_target[FD_ROLL]);     //!< Feedforward Target Roll [unit:0.1dps]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 3, telem->raw_setpoint[FD_PITCH]); //!< Raw Setpoint Pitch [unit:0.1dps]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 4, telem->setpoint[FD_PITCH]);     //!< Setpoint Estimate Pitch [unit:0.1dps]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 5, telem->ff_target[FD_PITCH]);    //!< Feedforward Target Pitch [unit:0.1dps]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 6, telem->feedforward_motor0);     //!< Feedforward Voltage Motor 0 [unit:0.01V]
        DEBUG_SET(DEBUG_WHIPPYFLIGHT, 7, telem->command_scale);          //!< Attitude Command Scale [unit:0.001]
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
    const flashPartition_t *partition = flashPartitionFindByType(FLASH_PARTITION_TYPE_WHIPPYFLIGHT);
    const flashGeometry_t *geometry = flashGetGeometry();
    if (!partition || geometry->totalSize == 0) {
        return WF_GAIN_NO_STORE;
    }

    const uint32_t base = partition->startSector * geometry->sectorSize;
    const uint32_t capacity = MIN(FLASH_PARTITION_SECTOR_COUNT(partition) * geometry->sectorSize,
                                  (uint32_t)sizeof(wfGainRam));

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

wf_gain_load_result_t whippyflightLoadGains(void)
{
    uint32_t blobSize = 0;
    const char *source = "sdcard";
    wf_gain_load_result_t result = WF_GAIN_NO_STORE;
#ifdef USE_SDCARD
    result = loadGainsFromSdCard(&blobSize);
#endif
    if (result == WF_GAIN_NO_STORE) {
        source = "dataflash";
        result = loadGainsFromFlash(&blobSize);
    }
    if (result == WF_GAIN_OK) {
        result = whippyflight_init(wfGainRam, blobSize);
    }
    if (result == WF_GAIN_OK) {
        memcpy(&wfGainDt, wfGainRam + WF_GAIN_DT_OFFSET, sizeof(wfGainDt));
    } else {
        blobSize = 0;
    }
    wfGainResult = result;
#ifdef SIMULATOR_BUILD
    printf("[WHIPPYFLIGHT] gain load from %s: result=%d, %u bytes\n", source, (int)result, (unsigned)blobSize);
#else
    UNUSED(source);
#endif
    return result;
}

const char *whippyflightArmingBlockReason(void)
{
    if (wfGainResult != WF_GAIN_OK) {
        return "gains not loaded";
    }
    // The gains are discretised at the blob's dt.
    const float pidDt = pidGetDT();
    if (fabsf(pidDt - wfGainDt) > WF_GAIN_DT_TOLERANCE * wfGainDt) {
        return "PID loop rate does not match gain blob dt";
    }
    // The gains are scheduled on measured motor speed.
#if defined(SIMULATOR_BUILD)
    if (!sitlHasEscTelemetry()) {
        return "simulator sends no ESC telemetry";
    }
#elif defined(USE_DSHOT_TELEMETRY)
    if (!useDshotTelemetry) {
        return "dshot_bidir off";
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
        return "self-level mode configured (angle/horizon/althold/poshold/rescue/trainer)";
    }
    // Every other failsafe procedure flies in ANGLE_MODE.
    if (failsafeConfig()->failsafe_procedure != FAILSAFE_PROCEDURE_DROP_IT) {
        return "failsafe_procedure is not DROP";
    }
    return NULL;
}

bool whippyflightIsReadyToArm(void)
{
    return whippyflightArmingBlockReason() == NULL;
}

#endif // USE_WHIPPYFLIGHT
