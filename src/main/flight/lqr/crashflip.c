#include <math.h>

#include "crashflip.h"
#include "lqr_constants.h"
#include "../../fc/rc.h"
#include "../../common/axis.h"
#include "../../common/maths.h"

#define CRASHFLIP_MOTOR_DEADBAND 0.02f
#define CRASHFLIP_STICK_DEADBAND 0.15f // 15%

MotorOutputsFractional mixCrashFlipOutputs(void) {
    MotorOutputsFractional motorOutputs = {{0, 0, 0, 0}};

    const float stickDeflectionPitchAbs = getRcDeflectionAbs(FD_PITCH);
    const float stickDeflectionRollAbs = getRcDeflectionAbs(FD_ROLL);
    const float stickDeflectionYawAbs = getRcDeflectionAbs(FD_YAW);

    float signPitch = getRcDeflection(FD_PITCH) < 0 ? 1 : -1;
    float signRoll = getRcDeflection(FD_ROLL) < 0 ? 1 : -1;
    float signYaw = (getRcDeflection(FD_YAW) < 0 ? 1 : -1);

    float stickDeflectionLength = sqrtf(sq(stickDeflectionPitchAbs) + sq(stickDeflectionRollAbs));

    if (stickDeflectionYawAbs > MAX(stickDeflectionPitchAbs, stickDeflectionRollAbs)) {
        // If yaw is the dominant, disable pitch and roll
        stickDeflectionLength = stickDeflectionYawAbs;
        signRoll = 0;
        signPitch = 0;
    } else {
        // If pitch/roll dominant, disable yaw
        signYaw = 0;
    }

    const float cosPhi = (stickDeflectionLength > 0) ? (stickDeflectionPitchAbs + stickDeflectionRollAbs) / (sqrtf(2.0f) * stickDeflectionLength) : 0;
    const float cosThreshold = sqrtf(3.0f) / 2.0f; // cos(30 deg)

    if (cosPhi < cosThreshold) {
        // Enforce either roll or pitch exclusively, if not on diagonal
        if (stickDeflectionRollAbs > stickDeflectionPitchAbs) {
            signPitch = 0;
        } else {
            signRoll = 0;
        }
    }

    // Calculate crashflipPower from stick deflection with a reasonable amount of stick deadband
    float crashflipPower = stickDeflectionLength > CRASHFLIP_STICK_DEADBAND ? stickDeflectionLength : 0.0f;

    for (int motor = 0; motor < 4; motor++) {
        float motorOutputNormalised =
                signYaw * motorMix.weight[motor][0] +
                signPitch * motorMix.weight[motor][1] +
                signRoll * motorMix.weight[motor][2];

        float motorOutput = MIN(1.0f, crashflipPower * motorOutputNormalised);

        motorOutputs.frac[motor] = motorOutput;
    }

    return motorOutputs;
}