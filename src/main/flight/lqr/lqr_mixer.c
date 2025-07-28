#include "../../drivers/motor.h"
#include "../../sensors/battery.h"
#include "lqr_controller.h"
#include "lqr_mixer.h"
#include "lqr_constants.h"
#include "common/maths.h"
#include <math.h>

MotorOutputsFractional mixMotorOutputs(MotorOutputs attitudeControl, float throttleFrac) {
    MotorOutputsFractional motorOutputs = {{0, 0, 0, 0}};
    // TODO: what the actual fuck is this contract?
    float batteryVoltage = getBatteryVoltage() / 100.;

    // Thrust balancing: scale throttle duty cycles to compensate for geometric asymmetries
    // Assumption: duty cycle ∝ motor velocity ∝ √thrust
    // So we need to scale duty cycles by 1/√d_coefficient to balance thrust
    // Make robust to projective rescaling by normalizing relative to geometric mean
    float dFront = attitudeModel.dFront;
    float dBack = attitudeModel.dBack;
    float dLeft = attitudeModel.dLeft;
    float dRight = attitudeModel.dRight;
    
    // Calculate geometric mean of all d coefficients
    float geometricMean = sqrtf(sqrtf(dFront * dBack * dLeft * dRight));
    
    // Normalize d coefficients relative to geometric mean
    float dFrontNorm = dFront / geometricMean;
    float dBackNorm = dBack / geometricMean;
    float dLeftNorm = dLeft / geometricMean;
    float dRightNorm = dRight / geometricMean;
    
    // Calculate throttle scaling factors using normalized coefficients
    float frontThrottleScale = 1.0f / sqrtf(dFrontNorm);
    float backThrottleScale = 1.0f / sqrtf(dBackNorm);
    float leftThrottleScale = 1.0f / sqrtf(dLeftNorm);
    float rightThrottleScale = 1.0f / sqrtf(dRightNorm);

    // Dynamic idle: scale throttle from 0.1 to 1.0 to maintain control authority
    // This ensures motors never go below 10% throttle, preventing prop stall
    float dynamicThrottleFrac = 0.1f + 0.9f * throttleFrac;
    
    // Apply thrust balancing to throttle duty cycles
    float balancedThrottle[4] = {
        dynamicThrottleFrac * frontThrottleScale * leftThrottleScale,   // Front-Left
        dynamicThrottleFrac * frontThrottleScale * rightThrottleScale,  // Front-Right  
        dynamicThrottleFrac * backThrottleScale * leftThrottleScale,    // Back-Left
        dynamicThrottleFrac * backThrottleScale * rightThrottleScale    // Back-Right
    };

    for (int motor = 0; motor < 4; motor++) {
        motorOutputs.frac[motor] = constrainf(
            attitudeControl.volts[motor] / batteryVoltage + balancedThrottle[motor],
            0,
            1
        );
    }

    return motorOutputs;
}
