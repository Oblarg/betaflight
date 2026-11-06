#!/usr/bin/env python3
"""
Float64 reference for test_real_gains.rs's stick-flick feedforward tests.

Runs the controller's reference-model feedforward (u_ff = G_h⁺·(r + h·Δr −
F_h·δx), δx ← Ad·δx + Bd·u_ff, |u| ≤ Vbat) at a uniform 275 rps operating
point straight from gen_wfg's model, for the tests' FLICK_RPS flick (linear
over 30 ms, PT3-smoothed at 190 Hz), and prints the peak and settled
feedforward voltages.  The tests compare the blob (quantized, interpolated,
f32) against these.

    python3 ff_reference.py [airframes/meteor65.json]
"""

import json
import math
import os
import sys

import control as ct
import numpy as np

import gen_wfg

HERE = os.path.dirname(os.path.abspath(__file__))
OMEGA = 275.0
DT = 0.00025
HORIZON = 8
BATTERY_V = 16.8
TICKS = 1200
# Small enough that hover collective + feedforward stays off the rails.
FLICK_RPS = 0.4


def flick(coeffs, bf_axis):
    w = gen_wfg.arm_weights(coeffs, 1.0, 1.7, 1.0, 1.0)
    A = gen_wfg.build_effective_A((OMEGA, OMEGA, OMEGA, OMEGA), coeffs, w)
    B = gen_wfg.build_B_matrix(coeffs['yaw']['propAccelSum'], coeffs['motors']['motorAccel'])
    sys_d = ct.ss(A, B, np.eye(7), np.zeros((7, 4))).sample(DT)
    Ad, Bd = sys_d.A, sys_d.B
    G = sum(np.linalg.matrix_power(Ad, i) @ Bd for i in range(HORIZON))[:3]
    F = np.linalg.matrix_power(Ad, HORIZON)[:3]
    Ginv = np.linalg.pinv(G)
    model_axis, sign = gen_wfg.MODEL_AXIS_FROM_BF[bf_axis]

    a = 1.0 - math.exp(-2.0 * math.pi * 190.0 * 1.96 * DT)
    s = [0.0, 0.0, 0.0]
    x = np.zeros(7)
    prev = np.zeros(3)
    out = []
    for k in range(TICKS):
        raw = FLICK_RPS * min(k * DT / 0.03, 1.0)
        s[0] += a * (raw - s[0])
        s[1] += a * (s[0] - s[1])
        s[2] += a * (s[1] - s[2])
        r = np.zeros(3)
        r[model_axis] = sign * s[2]
        target = r + HORIZON * (r - prev)
        u = np.clip(Ginv @ (target - F @ x), -BATTERY_V, BATTERY_V)
        x = Ad @ x + Bd @ u
        prev = r
        out.append(u)
    return np.array(out)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "airframes", "meteor65.json")
    with open(path) as f:
        coeffs = json.load(f)
    for name, axis in (("pitch", 1), ("yaw", 2)):
        u = flick(coeffs, axis)
        print(f"{name}: peak |u_ff| {np.abs(u).max():.3f} V, settled |u_ff| {np.abs(u[-1]).round(4)} V")


if __name__ == "__main__":
    main()
