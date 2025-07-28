#pragma once

#include "../../drivers/motor.h"
#include "lqr_controller.h"

MotorOutputsFractional mixMotorOutputs(MotorOutputs attitudeControl, float throttleFrac);