<%
import numpy as np

def c_array(arr, indent=8):
    # Recursively format numpy arrays/lists as C arrays with newlines and indentation
    if isinstance(arr, (list, np.ndarray)):
        if len(arr) == 0:
            return '{}'
        inner = ',\n'.join([c_array(x, indent + 4) for x in arr])
        return '{\n' + ' ' * indent + inner.replace('\n', '\n' + ' ' * indent) + '\n' + ' ' * (indent - 4) + '}'
    else:
        # Format floats with 6 decimal places, always with 'f' suffix
        return f'{float(arr):.6f}f'
%>
#include "lqr_constants.h"

const AttitudeModel attitudeModel = {
        .BContinuousRotationPerVoltSecondSquared = ${c_array(B_continuous)},
        .BContinuousInvVoltSecondSquaredPerRotation = ${c_array(B_continuous_inverse)},
        .affineInterceptPartial = ${c_array(affine_intercept_partial)},
        .dLeft = ${c_array(d_left)},
        .dRight = ${c_array(d_right)},
        .dFront = ${c_array(d_front)},
        .dBack = ${c_array(d_back)}
};

const MotorMix motorMix = {
        .weight = ${c_array(motor_mix)}
};

const float lqrLookupStepSize = ${c_array(lookup_step_size)};
const float lqrLookupMinRps = ${c_array(lookup_min_rps)};
const float integralTimeConstant = 1.0f;

// Integral windup protection constant (in rotations)
// Limit integral error to ±0.25 rotations (90 degrees) to prevent excessive windup
const float integralWindupLimitRotations = 0.250000f;

// Resolution constants for dynamic bounds checking
const int lqrLookupResolution = ${resolution};
const int lqrLookupMaxIndex = ${resolution - 1};

const ModelLookup modelLookup = {
        .KiVoltPerRotation = ${c_array(ki_lookup)},
        .KpVoltPerRotationPerSecond = ${c_array(kp_lookup)},
        .BDiscreteInvVoltSecondPerRotation = ${c_array(B_discrete_inv_lookup)}
};