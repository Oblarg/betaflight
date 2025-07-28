#pragma once

typedef struct {
    float BContinuousRotationPerVoltSecondSquared[7][4];
    float BContinuousInvVoltSecondSquaredPerRotation[4][7];
    float affineInterceptPartial[7];
    float dFront;
    float dBack;
    float dLeft;
    float dRight;
} AttitudeModel;

extern const AttitudeModel attitudeModel;

typedef struct {
    // Integral gains: volts per rotation (since we've integrated with time constant)
    float KiVoltPerRotation[5][5][5][5][4][3];
    // Rate gains: volts per rotation per second
    float KpVoltPerRotationPerSecond[5][5][5][5][4][3];
    float BDiscreteInvVoltSecondPerRotation[5][5][5][5][4][3];
} ModelLookup;

extern const ModelLookup modelLookup;

typedef struct {
    float weight[4][3];
} MotorMix;

extern const MotorMix motorMix;

extern const float lqrLookupStepSize;

extern const float lqrLookupMinRps;

extern const float integralTimeConstant;

// Integral windup protection constant (in rotations)
extern const float integralWindupLimitRotations;

// Resolution constants for dynamic bounds checking
extern const int lqrLookupResolution;
extern const int lqrLookupMaxIndex; 