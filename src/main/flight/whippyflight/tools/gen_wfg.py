#!/usr/bin/env python3
"""
Generate a WhippyFlight .wfg gain blob from motor/aerodynamic regression
coefficients.

Solves the discrete-time LQR (DARE) at each point on a 4D motor-speed grid,
extracts Ki/Kd/Kp/B⁻¹ gain matrices, remaps from the model's axis
conventions to BF, computes the affine linearizer matrix, and packs
everything into the .wfg binary format consumed by the Rust
whippyflight-control crate.

Usage:
    python3 gen_wfg.py <coefficients.json> <output.wfg> [--resolution N]
                       [--min-rps F] [--max-rps F] [--dt F]
                       [--tau-int F] [--tau-deriv F]
                       [--d-front F] [--d-back F] [--d-left F] [--d-right F]
"""

import argparse
import binascii
import json
import math
import struct
import sys

import numpy as np
import scipy as sc
import control as ct
import frccontrol as frcct


# ── .wfg format constants ────────────────────────────────────────────────
MAGIC       = 0x4C515247   # "LQRG" little-endian
VERSION     = 7
HEADER_SIZE = 0x20

# Reference-model table, per grid point (BF axes, motors by index):
REF_LAYOUT = [          # (name, rows, cols); diagonal blocks stored as vectors
    ('ad_rr', 3, 1),    # one-tick rate decay  e^{ρ·dt}
    ('ad_rw', 3, 4),    # one-tick rate ← motor-speed coupling
    ('ad_ww', 4, 1),    # one-tick motor-speed decay
    ('bd_r',  3, 4),    # one-tick rate ← voltage
    ('bd_w',  4, 1),    # one-tick motor speed ← voltage
    ('fh_rr', 3, 1),    # h-tick free response, rate ← rate
    ('fh_rw', 3, 4),    # h-tick free response, rate ← motor speed
]
REF_FLOATS = sum(r * c for _, r, c in REF_LAYOUT)

# ── Motor / axis conventions ─────────────────────────────────────────────
#
# The regression (whippyflight/lqr-prep/main.R) fits blackbox columns
# directly: motor[i]/eRPM[i] are labelled [FL, BL, FR, BR] for i = 0..3, so
# model motor i IS Betaflight motor index i — no motor remapping.  The
# labels describe the logged quad's wiring, not BF's QUADX names.
#
# The model's sign patterns relative to main.R's regressors (which use raw
# gyroADC) put the model axes at:
#   yaw_model   = -gyro[FD_YAW]
#   pitch_model = +gyro[FD_PITCH]
#   roll_model  = -gyro[FD_ROLL]
# matching lqr/original_c/lqr_controller.c.  Gains are exported in BF axis
# order [roll, pitch, yaw] with these signs folded in.
#
# Model motor speed unit is eRPM/600 (main.R: blackbox eRPM field / 6),
# which the C glue reproduces from getDshotErpm().

# BF axis index → (model axis index, sign)
MODEL_AXIS_FROM_BF = [(2, -1.0), (1, +1.0), (0, -1.0)]


# ── DARE solver ──────────────────────────────────────────────────────────

def solve_dare(A, B, Q_elems, R_elems):
    """Solve discrete-time LQR.  Returns gain matrix K."""
    Q = frcct.make_cost_matrix(Q_elems)
    R = frcct.make_cost_matrix(R_elems)
    S = sc.linalg.solve_discrete_are(a=A, b=B, q=Q, r=R)
    return np.linalg.solve(B.T @ S @ B + R, B.T @ S @ A)


# Augmented-state columns of K the firmware applies: integral, derivative and
# rate errors.  The delay buffer and motor speeds get no feedback (the motor
# speeds are the operating point the gains are scheduled at, so δω ≡ 0).
APPLIED_COLUMNS = list(range(0, 6)) + list(range(9, 12))


def solve_gains(A, B, Q_elems, R_elems, structured):
    return solve_structured_lqr(A, B, Q_elems, R_elems) if structured else solve_dare(A, B, Q_elems, R_elems)


def _closed_loop_cost(A, B, Q, R, K, W):
    """J = tr(P·W) for x ← (A − BK)x + w, w ~ (0, W); None if unstable."""
    A_cl = A - B @ K
    if np.abs(np.linalg.eigvals(A_cl)).max() >= 1.0:
        return None
    P = sc.linalg.solve_discrete_lyapunov(A_cl.T, Q + K.T @ R @ K)
    L = sc.linalg.solve_discrete_lyapunov(A_cl, W)
    return np.trace(P @ W), P, L


def solve_structured_lqr(A, B, Q_elems, R_elems, iters=200, tol=1e-9):
    """
    LQR with K restricted to APPLIED_COLUMNS (zero elsewhere).

    The full LQR feeds back every state, including unpenalized ones that
    predict the cost (motor speeds lead the rates).  Dropping those columns
    after the fact leaves the rest tuned for feedback that isn't there.
    This minimizes the same quadratic cost over the restricted K instead
    (Levine–Athans: the stationarity condition, iterated with a backtracking
    step), warm-started from the truncated full solution.

    Unlike the full LQR, the optimum depends on what excites the loop: here
    unit disturbances on the three rates (gusts, setpoint steps, as seen by
    the rate error).
    """
    Q = frcct.make_cost_matrix(Q_elems)
    R = frcct.make_cost_matrix(R_elems)
    n = A.shape[0]
    W = np.zeros((n, n))
    W[9:12, 9:12] = np.eye(3)
    S = np.eye(n)[APPLIED_COLUMNS]
    K = solve_dare(A, B, Q_elems, R_elems) @ S.T @ S
    first = _closed_loop_cost(A, B, Q, R, K, W)
    assert first is not None, "truncated LQR gains are unstable"
    J, P, L = first
    for _ in range(iters):
        F = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A @ L @ S.T) @ np.linalg.inv(S @ L @ S.T)
        step, trial = 1.0, None
        while step > 1e-4:
            trial = _closed_loop_cost(A, B, Q, R, K + step * (F @ S - K), W)
            if trial is not None and trial[0] < J:
                break
            step /= 2
            trial = None
        if trial is None:
            break
        done = (J - trial[0]) < tol * J
        K = K + step * (F @ S - K)
        J, P, L = trial
        if done:
            break
    return K


# ── Plant model ──────────────────────────────────────────────────────────

def build_B_matrix(k_I, k_alpha):
    """Control input matrix (7×4): voltage → [yaw, pitch, roll, ω_FL..ω_BR]."""
    return np.array([
        [k_I / k_alpha, -k_I / k_alpha, -k_I / k_alpha,  k_I / k_alpha],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1 / k_alpha, 0, 0, 0],
        [0, 1 / k_alpha, 0, 0],
        [0, 0, 1 / k_alpha, 0],
        [0, 0, 0, 1 / k_alpha],
    ])


def arm_weights(coeffs, d_front, d_back, d_left, d_right):
    """
    Per-motor lever-arm weights as they enter main.R's pitch/roll regressors.

    main.R fits k_T against (2·d_F·Σω_front² − 2·d_B·Σω_back²)/(d_F + d_B), so
    the effective weights are 2·d/(d_F + d_B).  The JSON does not record the
    d values used for the fit; values in it override the CLI defaults.
    """
    d_front = coeffs['pitch'].get('dFront', d_front)
    d_back  = coeffs['pitch'].get('dBack', d_back)
    d_left  = coeffs['roll'].get('dLeft', d_left)
    d_right = coeffs['roll'].get('dRight', d_right)
    return {
        'front': 2 * d_front / (d_front + d_back),
        'back':  2 * d_back / (d_front + d_back),
        'left':  2 * d_left / (d_left + d_right),
        'right': 2 * d_right / (d_left + d_right),
    }


def build_A_matrix(omega_fl, omega_bl, omega_fr, omega_br, coeffs, w):
    """Linearised attitude dynamics (7×7) at one operating point."""
    k_alpha    = coeffs['motors']['motorAccel']
    k_omega    = coeffs['motors']['motorVel']
    k_omega_sq = coeffs['motors']['motorVelSq']
    k_I        = coeffs['yaw']['propAccelSum']
    k_tau      = coeffs['yaw']['propVelSumSq']
    k_rho_yaw  = coeffs['yaw']['yawRate']
    k_T_pitch  = coeffs['pitch']['propVelSumSq']
    k_rho_pitch = coeffs['pitch']['pitchRate']
    d_front    = w['front']
    d_back     = w['back']
    k_T_roll   = coeffs['roll']['propVelSumSq']
    k_rho_roll = coeffs['roll']['rollRate']
    d_left     = w['left']
    d_right    = w['right']

    def k_beta(w):
        return -(k_omega + 2 * k_omega_sq * w) / k_alpha

    def k_gamma(w):
        return 2 * k_tau * w + k_I * k_beta(w)

    def k_phi(w, d=1):
        return 2 * k_T_pitch * d * w

    def k_theta(w, d=1):
        return 2 * k_T_roll * d * w

    return np.array([
        [k_rho_yaw,  0,           0,          k_gamma(omega_fl), -k_gamma(omega_bl), -k_gamma(omega_fr),  k_gamma(omega_br)],
        [0,          k_rho_pitch, 0,          k_phi(omega_fl, d_front), -k_phi(omega_bl, d_back), k_phi(omega_fr, d_front), -k_phi(omega_br, d_back)],
        [0,          0,           k_rho_roll, k_theta(omega_fl, d_left), k_theta(omega_bl, d_left), -k_theta(omega_fr, d_right), -k_theta(omega_br, d_right)],
        [0, 0, 0, k_beta(omega_fl), 0, 0, 0],
        [0, 0, 0, 0, k_beta(omega_bl), 0, 0],
        [0, 0, 0, 0, 0, k_beta(omega_fr), 0],
        [0, 0, 0, 0, 0, 0, k_beta(omega_br)],
    ])


def build_affine_P(coeffs, w):
    """
    Affine term of the Taylor expansion as a linear map of signed squares:
    C(ω̂) = P · [ω̂ᵢ|ω̂ᵢ|]  (7×4, model motor order [FL, BL, FR, BR]).

    Each row is (true nonlinear dynamics) − (A·x̂) evaluated at x̂:
      yaw:    (k_I·k_ω²/k_α − k_τ) · (+FL − BL − FR + BR)
      pitch:  −k_T · (d_F·FL − d_B·BL + d_F·FR − d_B·BR)
      roll:   −k_T · (d_L·FL + d_L·BL − d_R·FR − d_R·BR)
      motor:  k_ω²/k_α
    """
    k_alpha    = coeffs['motors']['motorAccel']
    k_omega_sq = coeffs['motors']['motorVelSq']
    k_I        = coeffs['yaw']['propAccelSum']
    k_tau      = coeffs['yaw']['propVelSumSq']
    k_T_pitch  = coeffs['pitch']['propVelSumSq']
    k_T_roll   = coeffs['roll']['propVelSumSq']

    c_yaw = k_I * k_omega_sq / k_alpha - k_tau
    P = np.zeros((7, 4))
    P[0] = c_yaw * np.array([1, -1, -1, 1])
    P[1] = -k_T_pitch * np.array([w['front'], -w['back'], w['front'], -w['back']])
    P[2] = -k_T_roll * np.array([w['left'], w['left'], -w['right'], -w['right']])
    P[3:7] = np.eye(4) * (k_omega_sq / k_alpha)
    return P


def build_affine_linearizer(coeffs, w):
    """
    Linearizer gain G (4×4, V per (model rps)²): u_aff = G · [ω̂ᵢ|ω̂ᵢ|].

    The plant is Ẋ = AX + BU + C(ω̂); the LQR is designed for Ẋ = AX + BU.
    U_aff = −B⁺C cancels the component of C in range(B) — the yaw row and
    the motor rows.  B has no direct pitch/roll path, so those rows of C
    are unmatched (B⁺ has zero columns there) and are left to the
    integrator.
    """
    k_I     = coeffs['yaw']['propAccelSum']
    k_alpha = coeffs['motors']['motorAccel']
    B_pinv = np.linalg.pinv(build_B_matrix(k_I, k_alpha))
    return -B_pinv @ build_affine_P(coeffs, w)


def build_effective_A(omegas, coeffs, w):
    """
    Linearised dynamics (7×7) of the plant as the controller drives it.

    The controller evaluates the affine linearizer at the measured motor
    speeds every tick, so u_aff = G·ω|ω| is itself feedback through ω: its
    slope B·G·diag(2|ω̂|) acts on the motor rows as extra drag (and on yaw,
    via G's yaw coupling).  A frozen-point linearization (build_A_matrix)
    misses it — first order in δω, ≈15% of the steady motor gain near
    hover — so the gains and the reference model are solved on A plus it.
    The motor rows keep only its diagonal: G's motor-to-motor coupling (via
    yaw) is ~1e-5 of the diagonal, and the reference model stores the motor
    block as a diagonal.
    """
    om = np.asarray(omegas, dtype=float)
    k_I     = coeffs['yaw']['propAccelSum']
    k_alpha = coeffs['motors']['motorAccel']
    A = build_A_matrix(*om, coeffs, w)
    G = build_affine_linearizer(coeffs, w)
    feedback = build_B_matrix(k_I, k_alpha) @ G @ np.diag(2.0 * np.abs(om))
    feedback[3:7] = np.diag(np.diag(feedback[3:7]))
    A[:, 3:7] += feedback
    return A


def build_motor_drag(coeffs):
    """
    Steady-state motor drag [k_ω, k_ω²]: main.R fits
    V = k_ω·ω + k_ω²·ω|ω| + k_α·ω̇, so holding ω takes k_ω·ω + k_ω²·ω|ω| volts.
    The controller uses this for the idle voltage floor.
    """
    return [coeffs['motors']['motorVel'], coeffs['motors']['motorVelSq']]


def build_thrust_balance(w):
    """
    Per-motor hover speed ratio [FL, BL, FR, BR] for a torque-free hover:
    ωᵢ ∝ 1/√(d_fbᵢ·d_lrᵢ) balances the lever-arm-weighted thrusts d·ω² on
    pitch and roll (as the POC mixer's throttle scaling did), normalized to
    geometric mean 1.  The controller turns it into voltages with the motor
    drag fit.
    """
    d_fb = np.array([w['front'], w['back'], w['front'], w['back']])
    d_lr = np.array([w['left'], w['left'], w['right'], w['right']])
    b = 1.0 / np.sqrt(d_fb * d_lr)
    return list(b / np.exp(np.log(b).mean()))


def build_airframe_drag(coeffs):
    """
    Airframe drag block [lp_hz, roll←accX, roll←accY, pitch←accX, pitch←accY]
    from fit_model.py's drag terms: angular acceleration per specific force
    (rot/s² per m/s², i.e. rot/m) from the low-passed body x/y accelerometer.  The fit regresses BF gyro axes on
    BF accelerometer axes, so the coefficients go in as fitted.  All zero
    (mode off) unless the fit kept drag terms on some axis.
    """
    drag = coeffs.get('drag', {})
    axes = drag.get('axes', [])
    if not axes:
        return [0.0] * 5
    term = lambda axis, name: coeffs[axis].get(name, 0.0) if axis in axes else 0.0
    return [drag['lp_hz'], term('roll', 'accX'), term('roll', 'accY'),
            term('pitch', 'accX'), term('pitch', 'accY')]


# ── Airframe operating point ─────────────────────────────────────────────

FULL_CELL_V = 4.2
# Without an airframe block (pre-fit_model JSONs): the legacy 1S regression
# airframe.
LEGACY_CELLS = 1
LEGACY_HOVER_RPS = 270.0
LEGACY_LOOP_HZ = 4000.0
# Fractions of the weakest pitch/roll authority the setpoint target may use
# (see src/setpoint.rs); they give 100 rot/s² and 5000 rot/s³ on the legacy
# airframe, where the limits were tuned.
TARGET_SLEW_FRACTION = 0.83
TARGET_JERK_FRACTION = 0.63


def operating_point(coeffs):
    """(cells, hover rps, loop Hz) from fit_model's airframe block."""
    a = coeffs.get('airframe', {})
    return (a.get('cells') or LEGACY_CELLS, a.get('hover_rps') or LEGACY_HOVER_RPS,
            a.get('loop_hz') or LEGACY_LOOP_HZ)


def steady_speed(coeffs, volts):
    """Motor speed (model rps) the motor fit settles at on `volts`."""
    kw, kq = coeffs['motors']['motorVel'], coeffs['motors']['motorVelSq']
    return (-kw + math.sqrt(kw * kw + 4 * kq * volts)) / (2 * kq)


def build_target_limits(coeffs, w, vbat, hover_rps, idle_rps):
    """
    Setpoint target limits [slew rot/s², jerk rot/s³] from the weakest
    pitch/roll direction.  From a torque-free hover, the motors that push
    that way go to full speed on `vbat` and the others to `idle_rps`; the
    resulting angular acceleration is the slew authority.  The jerk
    authority is the initial rate of change of that torque: thrust slope
    2·k_T·d·ω at hover times each motor's acceleration from the voltage
    step, (vbat − V_hover)/k_α up or −V_hover/k_α down.
    """
    k = coeffs['motors']
    hover = hover_rps * np.array(build_thrust_balance(w))
    v_hover = k['motorVel'] * hover + k['motorVelSq'] * hover ** 2
    full = steady_speed(coeffs, vbat)
    patterns = [
        coeffs['pitch']['propVelSumSq'] * np.array([w['front'], -w['back'], w['front'], -w['back']]),
        coeffs['roll']['propVelSumSq'] * np.array([w['left'], w['left'], -w['right'], -w['right']]),
    ]
    slew, jerk = [], []
    for s in [d * p for p in patterns for d in (1, -1)]:
        up = s > 0
        slew.append(s @ (np.where(up, full, idle_rps) ** 2 - hover ** 2))
        jerk.append(s @ (2 * hover * np.where(up, vbat - v_hover, -v_hover) / k['motorAccel']))
    return [TARGET_SLEW_FRACTION * min(slew), TARGET_JERK_FRACTION * min(jerk)]


# ── Grid solver ──────────────────────────────────────────────────────────

def feedforward_model(Ad, Bd, horizon):
    """
    Reference-model feedforward terms at one operating point (model order).

    The feedforward holds u constant for `horizon` ticks (ZOH) and inverts
    the attitude response over that horizon:
        u_ff = G_h⁺ · (r_target − F_h · δx),
        G_h = [Σ_{i<h} Ad^i Bd]_att,   F_h = [Ad^h]_att,
    where δx = [r_ref; δω_ref] is the reference model's state, propagated
    one tick at a time by δx ← Ad·δx + Bd·u_ff.  A one-tick horizon leaves a
    sampling zero near −1 (Nyquist chatter in u); h ≥ 2 removes it.

    Returns (binv [4,3], ref [REF_FLOATS]) with ref packed as REF_LAYOUT.
    """
    G = sum(np.linalg.matrix_power(Ad, i) @ Bd for i in range(horizon))[:3]
    F = np.linalg.matrix_power(Ad, horizon)[:3]
    for M in (Ad[:3, :3], F[:, :3], Ad[3:, 3:], Bd[3:]):
        assert np.allclose(M, np.diag(np.diag(M)), atol=1e-12 * np.abs(M).max())
    ref = np.concatenate([
        np.diag(Ad[:3, :3]),            # ad_rr  [3]
        Ad[:3, 3:].ravel(),             # ad_rw  [3×4]
        np.diag(Ad[3:, 3:]),            # ad_ww  [4]
        Bd[:3].ravel(),                 # bd_r   [3×4]
        np.diag(Bd[3:]),                # bd_w   [4]
        np.diag(F[:, :3]),              # fh_rr  [3]
        F[:, 3:].ravel(),               # fh_rw  [3×4]
    ])
    assert ref.size == REF_FLOATS
    return np.linalg.pinv(G), ref


def augmented_system(omegas, coeffs, w, dt, tau_int, tau_deriv):
    """
    Discrete plant at motor speeds `omegas` [FL, BL, FR, BR] augmented with
    the controller's filter states, 16 states:
      [int_yaw, int_pitch, int_roll,
       deriv_yaw, deriv_pitch, deriv_roll,
       buf_yaw, buf_pitch, buf_roll,
       yaw_rate, pitch_rate, roll_rate,
       omega_fl, omega_bl, omega_fr, omega_br]
    Returns (A_aug, B_aug, sys_d).
    """
    alpha   = dt / tau_int
    alpha_d = dt / tau_deriv
    k_I     = coeffs['yaw']['propAccelSum']
    k_alpha = coeffs['motors']['motorAccel']
    A = build_effective_A(omegas, coeffs, w)
    sys_d = ct.ss(A, build_B_matrix(k_I, k_alpha),
                  np.array([[1, 0, 0, 0, 0, 0, 0]]), np.array([[0, 0, 0, 0]])).sample(dt)

    A_aug_top = np.zeros((9, 16))
    # Leaky integrators (rows 0–2)
    A_aug_top[0, 0] = 1 - alpha
    A_aug_top[1, 1] = 1 - alpha
    A_aug_top[2, 2] = 1 - alpha
    A_aug_top[0, 9]  = alpha * dt
    A_aug_top[1, 10] = alpha * dt
    A_aug_top[2, 11] = alpha * dt
    # Derivative filters (rows 3–5)
    A_aug_top[3, 3] = 1 - alpha_d
    A_aug_top[4, 4] = 1 - alpha_d
    A_aug_top[5, 5] = 1 - alpha_d
    A_aug_top[3, 6] = -alpha_d / dt
    A_aug_top[4, 7] = -alpha_d / dt
    A_aug_top[5, 8] = -alpha_d / dt
    A_aug_top[3, 9]  = alpha_d / dt
    A_aug_top[4, 10] = alpha_d / dt
    A_aug_top[5, 11] = alpha_d / dt
    # Delay buffers (rows 6–8): copy current rate
    A_aug_top[6, 9]  = 1
    A_aug_top[7, 10] = 1
    A_aug_top[8, 11] = 1

    A_aug = np.block([
        [A_aug_top],
        [np.zeros((7, 9)), sys_d.A],
    ])
    B_aug = np.block([
        [np.zeros((9, 4))],
        [sys_d.B],
    ])
    return A_aug, B_aug, sys_d


def lqr_weights(sigma_int, sigma_deriv, sigma_rate, sigma_volt):
    """LQR cost weights (Bryson): sigma_int in rot, sigma_deriv in rot/s²,
    sigma_rate in rot/s, sigma_volt in V."""
    Q_elems = [
        sigma_int, sigma_int, sigma_int,       # integral states
        sigma_deriv, sigma_deriv, sigma_deriv,  # derivative states
        math.inf, math.inf, math.inf,          # delay buffer (no cost)
        sigma_rate, sigma_rate, sigma_rate,     # attitude rates
        math.inf, math.inf, math.inf, math.inf # motor speeds (no cost)
    ]
    return Q_elems, [sigma_volt] * 4


# The integral may cancel a pitch/roll imbalance as large as the hover
# thrust acting this fraction of the arm off center: a CG offset, or the
# steady part of wind.  Generous (5× the evals' 5% CG shift) so trim never
# runs out; bounded so a collision's windup leaves a recoverable torque.
TRIM_CG_ARM_FRACTION = 0.25


def build_integral_limit(coeffs, w, hover_rps, dt, ki_py, kp_py, min_rps, max_rps):
    """
    Per-motor integral voltage bound (V): the largest feedback voltage on
    any motor once the controller has trimmed a TRIM_CG_ARM_FRACTION CG
    offset on pitch or on roll, from a torque-free hover.

    At that steady state the leaky integral holds z = dt·e for a residual
    rate error e, and the feedback is (Ki·dt + Kp)·e; with the affine
    linearizer and the hover collective it holds motor speeds ω whose
    torques cancel the offset's k_T·e·Σω_h² on that axis and nothing on the
    others:  k_ω·ω + k_ω²·ω|ω| = (Ki·dt + Kp)·e + G·ω|ω| + v_c.  Solved
    for (ω, e); the bound is the largest |Ki·dt·e|.
    The lever arms, hover speeds and the LQR's split across motors make
    this airframe-specific (the back motors of a long-tailed frame carry
    most of a pitch trim).

    Ki and Kp are the grid's (solve_gain_grid), multilinearly interpolated
    as the firmware does: on a coarse grid they differ from the gains solved
    at the trimmed speeds by enough (2–3%) to pin a bound sized from those.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import fsolve
    hover = hover_rps * np.array(build_thrust_balance(w))
    axis = np.linspace(min_rps, max_rps, ki_py.shape[0])
    ki_grid = RegularGridInterpolator((axis,) * 4, ki_py)
    kp_grid = RegularGridInterpolator((axis,) * 4, kp_py)

    def gains_at(om):
        p = np.clip(om, min_rps, max_rps)[None, :]
        return ki_grid(p)[0] * dt, kp_grid(p)[0]

    G = build_affine_linearizer(coeffs, w)
    k_w, k_w2 = build_motor_drag(coeffs)
    volts = lambda om: k_w * om + k_w2 * om * np.abs(om)
    v_c = volts(hover) - G @ (hover * np.abs(hover))
    # Model-axis [yaw, pitch, roll] torques of the speeds (yaw's k_I·ω̇ term
    # is zero at steady state).
    patterns = np.array([
        coeffs['yaw']['propVelSumSq'] * np.array([1, -1, -1, 1]),
        coeffs['pitch']['propVelSumSq'] * np.array([w['front'], -w['back'], w['front'], -w['back']]),
        coeffs['roll']['propVelSumSq'] * np.array([w['left'], w['left'], -w['right'], -w['right']]),
    ])
    torque = lambda om: patterns @ (om * np.abs(om))
    worst = 0.0
    for axis, k_T in [(1, coeffs['pitch']['propVelSumSq']), (2, coeffs['roll']['propVelSumSq'])]:
        for sign in (1.0, -1.0):
            want = np.zeros(3)
            want[axis] = -sign * k_T * TRIM_CG_ARM_FRACTION * np.sum(hover ** 2)

            # The controller's gains are the ones at the trimmed speeds, not
            # at hover: re-solve with them there until the speeds settle.
            # (Interpolated gains are only piecewise smooth in ω, so they're
            # held fixed within each solve.)
            x = np.concatenate([hover, np.zeros(3)])
            for _ in range(5):
                ki_dt, kp = gains_at(x[0:4])

                def residual(x):
                    om, e = x[0:4], x[4:7]
                    return np.concatenate([volts(om) - (ki_dt + kp) @ e - G @ (om * np.abs(om)) - v_c,
                                           (torque(om) - want) / (k_T * np.sum(hover ** 2))])

                x, info, ok, msg = fsolve(residual, x, full_output=True)
                assert ok == 1, f"integral limit: trim solve failed: {msg}"
            worst = max(worst, np.abs(ki_dt @ x[4:7]).max())
    return worst


def solve_gain_grid(coeffs, w, resolution, min_rps, max_rps, dt, tau_int, tau_deriv,
                    ff_horizon, sigma_int=1e-4, sigma_deriv=20.0, sigma_rate=1.0, sigma_volt=1.0,
                    structured=False):
    """
    Solve DARE at each point on the N⁴ motor-speed grid.

    Returns numpy arrays in *Python* ordering [fl, bl, fr, br, ...]:
      ki_py  [N,N,N,N, 4, 3]   integral gains (V / rot)
      kd_py  [N,N,N,N, 4, 3]   derivative gains (V / (rot/s²))
      kp_py  [N,N,N,N, 4, 3]   proportional gains (V / (rot/s))
      binv_py[N,N,N,N, 4, 3]   horizon feedforward inverse G_h⁺ (V / (rot/s))
      ref_py [N,N,N,N, REF_FLOATS]  reference-model terms (REF_LAYOUT)
    """
    N = resolution
    rps_step = (max_rps - min_rps) / (N - 1)

    ki_py   = np.empty([N, N, N, N, 4, 3])
    kd_py   = np.empty([N, N, N, N, 4, 3])
    kp_py   = np.empty([N, N, N, N, 4, 3])
    binv_py = np.empty([N, N, N, N, 4, 3])
    ref_py  = np.empty([N, N, N, N, REF_FLOATS])

    total = N ** 4
    done = 0

    Q_elems, R_elems = lqr_weights(sigma_int, sigma_deriv, sigma_rate, sigma_volt)

    for fl_i in range(N):
        for bl_i in range(N):
            for fr_i in range(N):
                for br_i in range(N):
                    omega_fl = fl_i * rps_step + min_rps
                    omega_bl = bl_i * rps_step + min_rps
                    omega_fr = fr_i * rps_step + min_rps
                    omega_br = br_i * rps_step + min_rps

                    A_aug, B_aug, sys_d = augmented_system(
                        (omega_fl, omega_bl, omega_fr, omega_br), coeffs, w, dt, tau_int, tau_deriv)

                    K_full = solve_gains(A_aug, B_aug, Q_elems, R_elems, structured)
                    # K_full is [4, 16].  Extract:
                    #   Ki  = K[:, 0:3]   (integral gains)
                    #   Kd  = K[:, 3:6]   (derivative gains)
                    #   Kp  = K[:, 9:12]  (proportional / rate gains)
                    ki_py[fl_i, bl_i, fr_i, br_i, :, :] = K_full[:, 0:3]
                    kd_py[fl_i, bl_i, fr_i, br_i, :, :] = K_full[:, 3:6]
                    kp_py[fl_i, bl_i, fr_i, br_i, :, :] = K_full[:, 9:12]

                    binv_py[fl_i, bl_i, fr_i, br_i], ref_py[fl_i, bl_i, fr_i, br_i] = \
                        feedforward_model(sys_d.A, sys_d.B, ff_horizon)

                    done += 1
                    if done % max(1, total // 10) == 0:
                        print(f"  [{done}/{total}] operating points solved", file=sys.stderr)

    print(f"  [{total}/{total}] done", file=sys.stderr)
    return ki_py, kd_py, kp_py, binv_py, ref_py


# ── Remap + flatten ─────────────────────────────────────────────────────

def remap_and_flatten(table_py, resolution):
    """
    Convert a model-ordered [N,N,N,N, 4, 3] gain table (axes [yaw, pitch,
    roll]) to the flat BF-axis array expected by the Rust parser.

    Rust flat layout: for motor grid indices (m0, m1, m2, m3):
        flat_offset = ((m0*N + m1)*N + m2)*N + m3
        table[flat_offset*12 + motor*3 + bf_axis]

    Motor indices are identical in both conventions.  Since
    e_model = sign · e_bf, the BF-axis gain column is sign · K_model[:, axis].
    """
    N = resolution
    flat = np.empty(N**4 * 4 * 3, dtype=np.float32)
    for bf_axis, (model_axis, sign) in enumerate(MODEL_AXIS_FROM_BF):
        flat_view = flat.reshape(N, N, N, N, 4, 3)
        flat_view[..., bf_axis] = sign * table_py[..., model_axis]
    return flat


def remap_ref_and_flatten(ref_py, resolution):
    """
    Convert the model-axis reference-model table [N,N,N,N, REF_FLOATS] to BF
    axes.  With r_bf[a] = sign_a · r_model[axis_a], a rate-row block becomes
    sign_a · row(axis_a), and a diagonal rate→rate block just reorders.
    """
    N = resolution
    src = ref_py.reshape(N**4, REF_FLOATS)
    out = np.empty_like(src)
    offset = 0
    for name, rows, cols in REF_LAYOUT:
        block = src[:, offset:offset + rows * cols].reshape(-1, rows, cols)
        if rows == 3:
            mapped = np.empty_like(block)
            for bf_axis, (model_axis, sign) in enumerate(MODEL_AXIS_FROM_BF):
                row_sign = 1.0 if cols == 1 else sign
                mapped[:, bf_axis] = row_sign * block[:, model_axis]
            block = mapped
        out[:, offset:offset + rows * cols] = block.reshape(-1, rows * cols)
        offset += rows * cols
    return out.reshape(-1).astype(np.float32)


# ── CRC32 ────────────────────────────────────────────────────────────────

def crc32(data: bytes) -> int:
    """ISO 3309 CRC32, same as the Rust parser."""
    return binascii.crc32(data) & 0xFFFFFFFF


# ── Pack .wfg ────────────────────────────────────────────────────────────

def quantize_table(flat, width):
    """Quantize a table to u16, each of its `width` slots over its own range
    across the grid: value = lo + step·q.  Returns (lo, step, q)."""
    table = np.asarray(flat, dtype=np.float64).reshape(-1, width)
    lo = table.min(axis=0).astype(np.float32)
    span = table.max(axis=0) - lo
    step = (span / 65535.0).astype(np.float32)
    # Quantize against the stored f32 lo/step so decoding reproduces the
    # nearest code exactly.  Constant slots get step 0 and decode to lo.
    safe = np.where(step > 0, step, 1.0).astype(np.float64)
    q = np.where(step > 0, np.rint((table - lo) / safe), 0.0)
    q = np.clip(q, 0, 65535).astype('<u2')
    return lo, step, q.reshape(-1)


def pack_wfg(ki_flat, kd_flat, kp_flat, binv_flat, ref_flat, affine_g, motor_drag,
             thrust_balance, resolution, min_rps, max_rps, dt, tau_int, tau_deriv, ff_horizon,
             airframe_drag=(0.0,) * 5, target_limits=(0.0, 0.0), hover_rps=0.0,
             integral_limit_v=0.0):
    """Pack gain tables, reference model, affine linearizer, motor drag,
    thrust balance, airframe drag, target limits, hover speed and integral
    voltage bound into .wfg."""
    widths = [12, 12, 12, 12, REF_FLOATS]
    quantized = [quantize_table(flat, w)
                 for flat, w in zip([ki_flat, kd_flat, kp_flat, binv_flat, ref_flat], widths)]
    tables_bytes = b''
    for lo, step, _ in quantized:
        tables_bytes += lo.astype('<f4').tobytes() + step.astype('<f4').tobytes()
    affine_flat = np.asarray(affine_g, dtype=np.float32).reshape(16)
    tables_bytes += struct.pack('<16f', *affine_flat)
    tables_bytes += struct.pack('<2f', *motor_drag)
    tables_bytes += struct.pack('<4f', *thrust_balance)
    tables_bytes += struct.pack('<5f', *airframe_drag)
    tables_bytes += struct.pack('<2f', *target_limits)
    tables_bytes += struct.pack('<2f', hover_rps, integral_limit_v)
    for _, _, q in quantized:
        tables_bytes += q.tobytes()

    total_size = HEADER_SIZE + len(tables_bytes) + 4  # +4 for CRC

    header = struct.pack('<I',  MAGIC)           # 0x00: magic
    header += struct.pack('<H', VERSION)          # 0x04: version
    header += struct.pack('<B', resolution)       # 0x06: resolution
    header += struct.pack('<B', ff_horizon)       # 0x07: feedforward horizon (ticks)
    header += struct.pack('<f', min_rps)          # 0x08: min_rps
    header += struct.pack('<f', max_rps)          # 0x0C: max_rps
    header += struct.pack('<f', dt)               # 0x10: dt
    header += struct.pack('<f', tau_int)          # 0x14: integral_tau
    header += struct.pack('<f', tau_deriv)        # 0x18: derivative_tau
    header += struct.pack('<I', total_size)       # 0x1C: total_size

    assert len(header) == HEADER_SIZE

    payload = header + tables_bytes
    checksum = crc32(payload)
    return payload + struct.pack('<I', checksum)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a WhippyFlight .wfg gain blob from regression coefficients.")
    parser.add_argument("coefficients", help="Path to regression_coefficients.json")
    parser.add_argument("output", help="Output .wfg file path")
    parser.add_argument("--resolution", "-n", type=int, default=3,
                        help="Grid resolution N (default: 3, max: 5)")
    parser.add_argument("--min-rps", type=float, default=50.0,
                        help="Minimum motor speed in rot/s (default: 50)")
    parser.add_argument("--max-rps", type=float, default=None,
                        help="Maximum motor speed in rot/s (default: the motor fit's steady "
                             "speed on a full battery, rounded up to 10; 500 without an "
                             "airframe block)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Loop period in seconds (default: 1/airframe.loop_hz, else "
                             "0.00025 = 4 kHz)")
    parser.add_argument("--tau-int", type=float, default=10.0,
                        help="Integral time constant (default: 10.0)")
    parser.add_argument("--tau-deriv", type=float, default=0.02,
                        help="Derivative smoothing time constant (default: 0.02)")
    parser.add_argument("--d-front", type=float, default=1.0,
                        help="Front lever-arm weight used by main.R (default: 1.0)")
    parser.add_argument("--d-back", type=float, default=1.7,
                        help="Back lever-arm weight used by main.R (default: 1.7)")
    parser.add_argument("--d-left", type=float, default=1.0,
                        help="Left lever-arm weight used by main.R (default: 1.0)")
    parser.add_argument("--d-right", type=float, default=1.0,
                        help="Right lever-arm weight used by main.R (default: 1.0)")
    parser.add_argument("--sigma-int", type=float, default=0.012,
                        help="LQR: allowed low-passed rate error the integral holds, rot/s "
                             "(the state is dt × that, so its weight is σ·dt; default: 0.012)")
    parser.add_argument("--sigma-deriv", type=float, default=15.0,
                        help="LQR: allowed derivative-state excursion, rot/s² (default: 15.0)")
    parser.add_argument("--sigma-rate", type=float, default=5.0,
                        help="LQR: allowed rate excursion, rot/s (default: 5.0)")
    parser.add_argument("--sigma-volt", type=float, default=None,
                        help="LQR: allowed motor voltage excursion, V (default: 1 V per cell)")
    parser.add_argument("--ff-horizon", type=int, default=8,
                        help="Feedforward inversion horizon in loop ticks (default: 8; ≥2)")
    parser.add_argument("--structured", action="store_true",
                        help="Optimize only the gains the firmware applies (Ki, Kd, Kp) "
                             "instead of truncating the full LQR")
    parser.add_argument("--feedback-scale", type=float, default=1.0,
                        help="Scale Ki, Kd and Kp (0: feedforward only, for comparison "
                             "flights; the integral bound keeps the unscaled design's)")
    args = parser.parse_args()

    if args.resolution < 2 or args.resolution > 5:
        print("Error: resolution must be 2–5", file=sys.stderr)
        sys.exit(1)
    if args.ff_horizon < 2 or args.ff_horizon > 255:
        print("Error: ff-horizon must be 2–255 (h=1 has a sampling zero near −1)",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading coefficients from {args.coefficients}", file=sys.stderr)
    with open(args.coefficients, 'r') as f:
        coeffs = json.load(f)

    cells, hover_rps, loop_hz = operating_point(coeffs)
    vbat_full = cells * FULL_CELL_V
    if args.max_rps is None:
        args.max_rps = (10 * math.ceil(steady_speed(coeffs, vbat_full) / 10)
                        if 'airframe' in coeffs else 500.0)
    if args.dt is None:
        args.dt = 1.0 / loop_hz
    if args.sigma_volt is None:
        args.sigma_volt = 1.0 * cells
    print(f"Airframe: {cells}S ({vbat_full:.1f} V full), hover {hover_rps:.0f} rps; "
          f"grid {args.min_rps:.0f}–{args.max_rps:.0f} rps, dt {args.dt * 1e6:.1f} µs, "
          f"σ_volt {args.sigma_volt:.2f} V", file=sys.stderr)

    print(f"Solving {'structured LQR' if args.structured else 'DARE'} on {args.resolution}⁴ = {args.resolution**4} grid points...",
          file=sys.stderr)
    w = arm_weights(coeffs, args.d_front, args.d_back, args.d_left, args.d_right)
    print(f"Lever-arm weights: {w}", file=sys.stderr)

    ki_py, kd_py, kp_py, binv_py, ref_py = solve_gain_grid(
        coeffs, w, args.resolution, args.min_rps, args.max_rps,
        args.dt, args.tau_int, args.tau_deriv, args.ff_horizon,
        args.sigma_int * args.dt, args.sigma_deriv, args.sigma_rate, args.sigma_volt,
        args.structured)

    print("Remapping to BF axis ordering...", file=sys.stderr)
    if args.feedback_scale != 1.0:
        print(f"Feedback gains scaled by {args.feedback_scale}", file=sys.stderr)
    s = args.feedback_scale
    ki_flat   = remap_and_flatten(ki_py * s, args.resolution)
    kd_flat   = remap_and_flatten(kd_py * s, args.resolution)
    kp_flat   = remap_and_flatten(kp_py * s, args.resolution)
    binv_flat = remap_and_flatten(binv_py, args.resolution)
    ref_flat  = remap_ref_and_flatten(ref_py, args.resolution)

    affine_g = build_affine_linearizer(coeffs, w)
    motor_drag = build_motor_drag(coeffs)
    thrust_balance = build_thrust_balance(w)
    airframe_drag = build_airframe_drag(coeffs)
    if airframe_drag[0]:
        print(f"Airframe drag: {airframe_drag}", file=sys.stderr)
    target_limits = build_target_limits(coeffs, w, vbat_full, hover_rps, args.min_rps)
    print(f"Target limits: slew {target_limits[0]:.1f} rot/s², jerk {target_limits[1]:.0f} rot/s³",
          file=sys.stderr)
    integral_limit_v = build_integral_limit(
        coeffs, w, hover_rps, args.dt, ki_py, kp_py, args.min_rps, args.max_rps)
    print(f"Integral voltage bound: {integral_limit_v:.3f} V "
          f"({TRIM_CG_ARM_FRACTION:.0%}-of-arm CG offset)", file=sys.stderr)

    print("Packing .wfg binary...", file=sys.stderr)
    blob = pack_wfg(ki_flat, kd_flat, kp_flat, binv_flat, ref_flat, affine_g, motor_drag,
                    thrust_balance,
                    args.resolution, args.min_rps, args.max_rps,
                    args.dt, args.tau_int, args.tau_deriv, args.ff_horizon,
                    airframe_drag, target_limits, hover_rps, integral_limit_v)

    with open(args.output, 'wb') as f:
        f.write(blob)
    print(f"Wrote {len(blob)} bytes to {args.output}", file=sys.stderr)

    # Sanity print: gain magnitudes at the centre of the grid
    mid = args.resolution // 2
    print(f"\nSanity check — gains at grid centre [{mid},{mid},{mid},{mid}]:",
          file=sys.stderr)
    for name, table in [("Ki", ki_py), ("Kd", kd_py), ("Kp", kp_py), ("G_h⁺", binv_py)]:
        vals = table[mid, mid, mid, mid]
        print(f"  {name} (Python ordering, [motor, axis]):", file=sys.stderr)
        for m in range(4):
            print(f"    motor {m}: [{vals[m,0]:.4f}, {vals[m,1]:.4f}, {vals[m,2]:.4f}]",
                  file=sys.stderr)
    print("  Affine linearizer G (V per rps², motor × signed-square):", file=sys.stderr)
    for m in range(4):
        print("    motor {}: [{}]".format(m, ", ".join(f"{g:.3e}" for g in affine_g[m])),
              file=sys.stderr)


if __name__ == "__main__":
    main()
