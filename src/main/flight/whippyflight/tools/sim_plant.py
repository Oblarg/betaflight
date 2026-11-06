"""
Nonlinear rate/motor plant that gen_wfg.py linearizes, for closing the loop
around SITL.  Same model as tests/eval_setpoint_reconstruction.rs:

  ω̇_i    = (V_i − kω·ω_i − kω²·ω_i|ω_i|) / kα
  yaw'   = ρ_y·yaw   + kτ·Σ s_y·ω|ω| + k_I·Σ s_y·ω̇     s_y = [+, −, −, +]
  pitch' = ρ_p·pitch + kT_p·Σ s_p·ω|ω|                  s_p = [w_F, −w_B, w_F, −w_B]
  roll'  = ρ_r·roll  + kT_r·Σ s_r·ω|ω|                  s_r = [w_L, w_L, −w_R, −w_R]

Rates are model axes [yaw, pitch, roll] in rot/s; motor speeds are by BF
motor index in the regression's unit, eRPM/600; V is duty × pack voltage.
"""

import json
import math

TWO_PI = 2.0 * math.pi


def arm_weights(coeffs, d_front=1.0, d_back=1.7, d_left=1.0, d_right=1.0):
    """main.R's lever-arm weights, 2·d/(d_1 + d_2); as gen_wfg.arm_weights."""
    d_front = coeffs["pitch"].get("dFront", d_front)
    d_back = coeffs["pitch"].get("dBack", d_back)
    d_left = coeffs["roll"].get("dLeft", d_left)
    d_right = coeffs["roll"].get("dRight", d_right)
    return (2 * d_front / (d_front + d_back), 2 * d_back / (d_front + d_back),
            2 * d_left / (d_left + d_right), 2 * d_right / (d_left + d_right))


class Plant:
    def __init__(self, coeffs_path, omega0=0.0):
        with open(coeffs_path) as f:
            c = json.load(f)
        w_f, w_b, w_l, w_r = arm_weights(c)
        self.k_alpha = c["motors"]["motorAccel"]
        self.k_w = c["motors"]["motorVel"]
        self.k_w2 = c["motors"]["motorVelSq"]
        self.rho = (c["yaw"]["yawRate"], c["pitch"]["pitchRate"], c["roll"]["rollRate"])
        self.k_tau = c["yaw"]["propVelSumSq"]
        self.k_i = c["yaw"]["propAccelSum"]
        self.k_tp = c["pitch"]["propVelSumSq"]
        self.k_tr = c["roll"]["propVelSumSq"]
        self.s_yaw = (1.0, -1.0, -1.0, 1.0)
        self.s_pitch = (w_f, -w_b, w_f, -w_b)
        self.s_roll = (w_l, w_l, -w_r, -w_r)
        self.rate = [0.0, 0.0, 0.0]
        self.omega = [omega0] * 4

    def _deriv(self, rate, omega, volts):
        sq = [w * abs(w) for w in omega]
        wdot = [(volts[i] - self.k_w * omega[i] - self.k_w2 * sq[i]) / self.k_alpha
                for i in range(4)]
        sy, sp, sr = self.s_yaw, self.s_pitch, self.s_roll
        yaw = (self.rho[0] * rate[0]
               + self.k_tau * sum(sy[i] * sq[i] for i in range(4))
               + self.k_i * sum(sy[i] * wdot[i] for i in range(4)))
        pitch = self.rho[1] * rate[1] + self.k_tp * sum(sp[i] * sq[i] for i in range(4))
        roll = self.rho[2] * rate[2] + self.k_tr * sum(sr[i] * sq[i] for i in range(4))
        return [yaw, pitch, roll], wdot

    def step(self, volts, h):
        """Advance h seconds with the motor voltages held (RK4)."""
        r0, w0 = self.rate, self.omega
        k1r, k1w = self._deriv(r0, w0, volts)
        k2r, k2w = self._deriv([r0[i] + h / 2 * k1r[i] for i in range(3)],
                               [w0[i] + h / 2 * k1w[i] for i in range(4)], volts)
        k3r, k3w = self._deriv([r0[i] + h / 2 * k2r[i] for i in range(3)],
                               [w0[i] + h / 2 * k2w[i] for i in range(4)], volts)
        k4r, k4w = self._deriv([r0[i] + h * k3r[i] for i in range(3)],
                               [w0[i] + h * k3w[i] for i in range(4)], volts)
        self.rate = [r0[i] + h / 6 * (k1r[i] + 2 * k2r[i] + 2 * k3r[i] + k4r[i]) for i in range(3)]
        self.omega = [w0[i] + h / 6 * (k1w[i] + 2 * k2w[i] + 2 * k3w[i] + k4w[i]) for i in range(4)]

    def gyro_bf(self):
        """Betaflight gyro [roll, pitch, yaw] in rot/s (gen_wfg MODEL_AXIS_FROM_BF)."""
        return (-self.rate[2], self.rate[1], -self.rate[0])

    def imu_rpy(self):
        """fdm_packet imu_angular_velocity_rpy (rad/s) that SITL maps onto
        gyro_bf(): BF roll = rpy[0], pitch = −rpy[1], yaw = rpy[2]."""
        roll, pitch, yaw = self.gyro_bf()
        return (TWO_PI * roll, -TWO_PI * pitch, TWO_PI * yaw)

    def erpm(self):
        return [600.0 * w for w in self.omega]
