"""
Fit the WhippyFlight plant model from Betaflight blackbox logs.

A port of the lqr-prep/main.R regressions (same regressors and arm weighting)
that reads .bbl/.bfl files directly via bbl.py, so the bbl -> csv -> R -> json
chain becomes one step.  The output JSON is what gen_wfg.py consumes, plus an
"airframe" block taken from the log header.

    python3 fit_model.py fit LOG [LOG ...] -o coeffs.json [--dump CLI.txt]
    python3 fit_model.py validate coeffs.json LOG [LOG ...]
    python3 fit_model.py crossval LOG [LOG ...] [--dump CLI.txt] [-o pooled.json]

LOG is a blackbox file (every log in it is used), FILE#N for only its N-th
log (1-based), or a Blackbox Explorer CSV export.

The front/back and left/right arm ratios are fitted, not assumed: the thrust
regressor with a free ratio spans the same space as regressing the two motor
pairs separately, so that fit gives the R²-optimal ratio in closed form (what
sweeping dBack/dLeft by hand converges to).  --d-front/--d-back or
--d-left/--d-right pin an axis to given values instead.

Data selection (--select robust, the default) works on uncurated flights:
  * ground time is cut where motor speed sits near idle (takeoff, landing,
    sitting armed after a crash) instead of trimming a fixed tail;
  * impacts are masked from the accelerometer and gyro saturation, padded;
  * excitation gating is relative to a high percentile of |rate|, not the
    single largest sample (which on a crash is the impact);
  * finite-difference windows are durations, so logs at different rates
    difference over the same time span.
--select r reproduces main.R exactly (fixed 5 s tail trim, max-relative
gating, windows in samples).

Airframe drag (--drag, default auto): tall frames carry rotor drag well
above the CG, so airspeed torques pitch and roll.  The fit can add the body
x/y accelerometer (accSmooth, one-pole low-passed at DRAG_LP_HZ like the
firmware) as pitch/roll regressors; auto keeps them on an axis only if they
raise its held-out R² by DRAG_MIN_GAIN to at least DRAG_MIN_HELDOUT_R2.
Held-out R² is pooled over folds that stay within one flight (a quarter
held out, the rest of that flight trained on): props and motors change
between flights, and across flights the drag terms absorb that instead.
The JSON's "drag" block records the axes kept; gen_wfg packs it into the
v7 blob, and firmware built with the airframe-drag feature compensates it.

`validate` reports per log and axis the R² of given coefficients on that
log's rows next to the R² of a fit to that log alone.  `crossval` is
leave-one-out within one airframe: each flight is scored by a fit to the
others, then per-flight coefficient spread and a motor-pairing check (each
axis should be best explained by its own QUADX motor grouping).  --dump checks
the CLI dump's mixer is the QUADX layout the regressors assume.

Units follow main.R: rates in rot/s, motor speed in eRPM/600 (the unit the
glue feeds the controller), volts from motor command × vbat.  --volts glue
maps the motor command the way the glue drives it, (motor − 48)/1999, instead
of main.R's motor/2047.
"""

import argparse
import csv
import functools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bbl import read_logs  # noqa: E402

AXES = ("yaw", "pitch", "roll", "motors")

# main.R's windows, in samples of the Meteor65 log it was tuned on (1609 Hz).
R_WINDOW = {"yaw": 50, "pitch": 20, "roll": 20, "motors": 16}
R_LOG_HZ = 1609.0
THRESHOLD = {"yaw": 0.3, "pitch": 0.3, "roll": 0.3, "motors": 0.1}
TRIM_TAIL_S = 5.0
MIN_DURATION_S = 10.0

EXCITATION_PERCENTILE = 99.5
IMPACT_G = 8.0
GYRO_SATURATION_DPS = 1950.0
IMPACT_PAD_S = (0.25, 1.0)
GROUND_FRACTION = 0.3
GROUND_MIN_S = 0.5

MIN_MOTOR_ERPM_CORR = 0.6

# Airframe drag: on tall frames rotor drag acts well above the CG, so airspeed
# torques pitch and roll.  In flight the accelerometer reads thrust (body z)
# plus drag, so its x/y components measure the drag force directly; the fit
# adds them, one-pole low-passed at DRAG_LP_HZ (the firmware's filter), as
# regressors on pitch and roll, in m/s² so the coefficients are rot/m, the
# blob's unit.  --drag auto keeps them on an axis only if
# they improve its held-out R² by DRAG_MIN_GAIN and the axis then reaches
# DRAG_MIN_HELDOUT_R2 (below that the model doesn't predict the axis at all,
# and the gain is noise).
DRAG_LP_HZ = 5.0
DRAG_MIN_GAIN = 0.02
DRAG_MIN_HELDOUT_R2 = 0.3
DRAG_FOLDS = 4
DRAG_TERMS = ["accX", "accY"]
STANDARD_GRAVITY = 9.80665

DSHOT_MIN = 48
DSHOT_MAX = 2047


def _dilate(mask, t, before, after):
    """Samples within [event − before, event + after] of any True sample."""
    events = t[mask]
    if not len(events):
        return np.zeros_like(mask)
    lo = np.searchsorted(events, t - after, "left")
    hi = np.searchsorted(events, t + before, "right")
    return hi > lo


def _runs(mask):
    """(start, stop) index pairs of the True runs in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


class Flight:
    """One log's signals in model units, the usable-sample mask, and header metadata."""

    def __init__(self, label, cols, header, selection="robust"):
        self.label = label
        self.header = header
        self.selection = selection
        t = (cols["time"] - cols["time"][0]) * 1e-6
        if selection == "r":
            keep = t < t.max() - TRIM_TAIL_S
            t = t[keep]
            cols = {k: v[keep] for k, v in cols.items()}
        self.t = t
        self.cols = cols
        self.log_hz = (len(t) - 1) / (t[-1] - t[0])
        self.gating = {}
        self.good = np.ones(len(t), dtype=bool) if selection == "r" else self._gate()

    def window(self, axis):
        if self.selection == "r":
            return R_WINDOW[axis]
        return max(2, 2 * int(round(R_WINDOW[axis] / R_LOG_HZ * self.log_hz / 2)))

    def _gate(self):
        t = self.t
        erpm = np.array([self.cols[f"eRPM[{i}]"] for i in range(4)])
        good = erpm.min(axis=0) > 0

        speed = erpm.mean(axis=0)
        idle, typical = np.percentile(speed, 2), np.median(speed)
        ground = speed < idle + GROUND_FRACTION * (typical - idle)
        ground_mask = np.zeros_like(ground)
        for a, b in _runs(ground):
            if a == 0 or b == len(t) or t[b - 1] - t[a] > GROUND_MIN_S:
                ground_mask[a:b] = True
        good &= ~ground_mask

        impact = np.abs([self.cols[f"gyroADC[{i}]"] for i in range(3)]).max(axis=0) > GYRO_SATURATION_DPS
        if "accSmooth[0]" in self.cols:
            one_g = float(self.header.get("acc_1G", 2048))
            acc = np.sqrt(sum(self.cols[f"accSmooth[{i}]"] ** 2 for i in range(3))) / one_g
            impact |= acc > IMPACT_G
        impact_mask = _dilate(impact, t, *IMPACT_PAD_S)
        good &= ~impact_mask

        self.gating = {"ground_s": float(ground_mask.sum() / self.log_hz),
                       "impacts": int(sum(1 for _ in _runs(impact))),
                       "impact_s": float(impact_mask.sum() / self.log_hz)}
        return good

    def rate(self, axis):
        return self.cols[f"gyroADC[{axis}]"] / 360.0

    def motors(self):
        """Motor speeds in main.R's naming: FL, BL, FR, BR = eRPM[0..3]."""
        fl, bl, fr, br = (self.cols[f"eRPM[{i}]"] / 6.0 for i in range(4))
        return fl, bl, fr, br

    def volts(self, mapping):
        vbat = self.cols["vbatLatest"] / 100.0
        m = [self.cols[f"motor[{i}]"] for i in range(4)]
        if mapping == "glue":
            duty = [np.clip((x - DSHOT_MIN) / (DSHOT_MAX - DSHOT_MIN), 0.0, 1.0) for x in m]
        else:
            duty = [x / DSHOT_MAX for x in m]
        fl, bl, fr, br = (d * vbat for d in duty)
        return fl, bl, fr, br

    def kept_s(self):
        return float(self.good.sum() / self.log_hz)

    def has_accel(self):
        return "accSmooth[0]" in self.cols

    def accel_lp(self):
        """Body x/y specific force in m/s², one-pole low-passed at DRAG_LP_HZ (causal, as on the FC)."""
        if getattr(self, "_accel_lp", None) is None or len(self._accel_lp[0]) != len(self.t):
            one_g = float(self.header.get("acc_1G", 2048))
            k = 1.0 - np.exp(-2.0 * np.pi * DRAG_LP_HZ / self.log_hz)
            from scipy.signal import lfilter
            out = []
            for i in range(2):
                a = STANDARD_GRAVITY * self.cols[f"accSmooth[{i}]"] / one_g
                out.append(lfilter([k], [1.0, k - 1.0], a, zi=[(1.0 - k) * a[0]])[0])
            self._accel_lp = out
        return self._accel_lp


def _cols_from_csv(path):
    with open(path) as f:
        reader = csv.reader(f)
        names = [h.strip().strip('"') for h in next(reader)]
        rows = [r for r in reader if r]
    scale = {}
    for i, n in enumerate(names):
        base, _, unit = n.partition(" (")
        names[i] = base
        if unit.startswith("V)"):
            scale[base] = 100.0
    want = ["time", "vbatLatest"] + [f"gyroADC[{i}]" for i in range(3)] + \
           [f"motor[{i}]" for i in range(4)] + [f"eRPM[{i}]" for i in range(4)]
    idx = {n: names.index(n) for n in want}
    return {n: np.array([float(r[i]) for r in rows]) * scale.get(n, 1.0) for n, i in idx.items()}


def unusable(flight):
    """Why a flight can't be fitted, or None."""
    erpm = [flight.cols[f"eRPM[{i}]"] for i in range(4)]
    if (np.min(erpm, axis=0) > 0).mean() < 0.5:
        return "no eRPM telemetry for most of the flight"
    corr = [np.corrcoef(flight.cols[f"motor[{i}]"], erpm[i])[0, 1] for i in range(4)]
    if np.median(corr) < MIN_MOTOR_ERPM_CORR:
        return (f"logged motor[] doesn't track eRPM (corr {np.round(corr, 2).tolist()}); "
                "motor output was not what the ESCs received")
    if flight.kept_s() < MIN_DURATION_S:
        return f"only {flight.kept_s():.1f} s airborne and impact-free"
    return None


def split(flight, parts):
    """Cut a flight into `parts` contiguous, equal-duration Flights."""
    edges = np.linspace(flight.t[0], flight.t[-1], parts + 1)
    out = []
    for k in range(parts):
        keep = (flight.t >= edges[k]) & (flight.t <= edges[k + 1])
        f = Flight.__new__(Flight)
        f.__dict__.update(flight.__dict__)
        f.label = f"{flight.label}/{k + 1}"
        f.t = flight.t[keep]
        f.cols = {c: v[keep] for c, v in flight.cols.items()}
        f.good = flight.good[keep]
        out.append(f)
    return out


def load_flights(spec, selection="robust"):
    """Expand a LOG argument into gated Flights."""
    path, _, which = spec.partition("#")
    if path.lower().endswith(".csv"):
        candidates = [Flight(os.path.basename(path), _cols_from_csv(path), {}, selection)]
    else:
        candidates = []
        short = 0
        for n, log in enumerate(read_logs(path), 1):
            if which and int(which) != n:
                continue
            t = log["time"]
            if (t[-1] - t[0]) * 1e-6 < MIN_DURATION_S + (TRIM_TAIL_S if selection == "r" else 0.0):
                short += 1
                continue
            cols = {name: log[name].astype(np.float64) for name in log.names}
            label = os.path.join(os.path.basename(os.path.dirname(os.path.abspath(path))),
                                 os.path.basename(path))
            candidates.append(Flight(f"{label}#{n}", cols, log.header, selection))
        if short:
            print(f"skipping {short} log(s) in {os.path.basename(path)} shorter than {MIN_DURATION_S:g} s",
                  file=sys.stderr)
    flights = []
    for flight in candidates:
        reason = unusable(flight)
        if reason:
            print(f"skipping {flight.label}: {reason}", file=sys.stderr)
            continue
        flights.append(flight)
    return flights


def _sq(x):
    return np.sign(x) * x * x


def _trunc(x, n):
    return x[n // 2 - 1: len(x) - n // 2 - 1]


def _cfd(x, t, n):
    return (x[n:] - x[:-n]) / (t[n:] - t[:-n])


def _rows_ok(good, n):
    """For each finite-difference row k (samples k..k+n): every sample usable."""
    bad = np.concatenate([[0], np.cumsum(~good)])
    return (bad[n + 1:] - bad[:-n - 1]) == 0


def _gate(flight, gate, threshold, n):
    """Row mask: window usable and |gate| above threshold × excitation scale."""
    ok = _rows_ok(flight.good, n)
    if not ok.any():
        return ok
    if flight.selection == "r":
        scale = np.abs(gate).max()
    else:
        scale = np.percentile(np.abs(gate[ok]), EXCITATION_PERCENTILE)
    return ok & (np.abs(gate) > scale * threshold)


def design(flight, axis, arms, volts_map, drag=()):
    """Regression rows (X, y), regressor names, and each row's sample index."""
    n = flight.window(axis)
    thr = THRESHOLD[axis]
    t = flight.t
    fl, bl, fr, br = flight.motors()
    center = np.arange(len(t) - n) + n // 2 - 1

    if axis == "yaw":
        rate = flight.rate(2)
        X = np.column_stack([
            _trunc(rate, n),
            _trunc(-_sq(fl) + _sq(fr) + _sq(bl) - _sq(br), n),
            -_cfd(fl, t, n) + _cfd(fr, t, n) + _cfd(bl, t, n) - _cfd(br, t, n),
        ])
        y = _cfd(rate, t, n)
        keep = _gate(flight, X[:, 0], thr, n)
        return X[keep], y[keep], ["yawRate", "propVelSumSq", "propAccelSum"], center[keep]

    if axis in ("pitch", "roll"):
        X, y, names, idx = design_split(flight, axis, axis in drag)
        pos, neg = ARM_KEYS[axis]
        X = np.column_stack([X[:, 0], arm_regressor(X, arms[pos], arms[neg]), X[:, 3:]])
        return X, y, names[:1] + ["propVelSumSq"] + names[3:], idx

    vfl, vbl, vfr, vbr = flight.volts(volts_map)
    vel = np.concatenate([_trunc(m, n) for m in (fl, fr, bl, br)])
    acc = np.concatenate([_cfd(m, t, n) for m in (fl, fr, bl, br)])
    v = np.concatenate([_trunc(m, n) for m in (vfl, vfr, vbl, vbr)])
    X = np.column_stack([vel, _sq(vel), acc])
    ok = np.tile(_rows_ok(flight.good, n), 4)
    if flight.selection == "r":
        scale = np.abs(vel).max()
    else:
        scale = np.percentile(np.abs(vel[ok]), EXCITATION_PERCENTILE) if ok.any() else 0.0
    keep = ok & (np.abs(vel) > scale * thr)
    return X[keep], v[keep], ["motorVel", "motorVelSq", "motorAccel"], np.tile(center, 4)[keep]


# (positive-side arm, negative-side arm) of each axis's thrust regressor.
ARM_KEYS = {"pitch": ("dFront", "dBack"), "roll": ("dRight", "dLeft")}


def design_split(flight, axis, drag=False):
    """
    Pitch/roll rows with the two motor pairs as separate regressors:
    columns [rate, Σω²(positive pair), Σω²(negative pair)], then with `drag`
    the low-passed body x/y specific force (m/s²).
    """
    n = flight.window(axis)
    t = flight.t
    fl, bl, fr, br = flight.motors()
    if axis == "pitch":
        rate = flight.rate(1)
        pos, neg = _sq(fl) + _sq(fr), _sq(bl) + _sq(br)
    else:
        rate = flight.rate(0)
        pos, neg = _sq(fr) + _sq(br), _sq(fl) + _sq(bl)
    X = np.column_stack([_trunc(rate, n), _trunc(pos, n), _trunc(neg, n)])
    names = [f"{axis}Rate", "pos", "neg"]
    if drag:
        X = np.column_stack([X] + [_trunc(a, n) for a in flight.accel_lp()])
        names += DRAG_TERMS
    y = _cfd(rate, t, n)
    keep = _gate(flight, X[:, 0], THRESHOLD[axis], n)
    center = np.arange(len(t) - n) + n // 2 - 1
    return X[keep], y[keep], names, center[keep]


def arm_regressor(X_split, d_pos, d_neg):
    """main.R's thrust regressor 2·(d₊Σ₊ − d₋Σ₋)/(d₊ + d₋) from split columns."""
    return 2 * (d_pos * X_split[:, 1] - d_neg * X_split[:, 2]) / (d_pos + d_neg)


def fit_arm_ratio(X, y):
    """
    Best-fit arm ratio d₋/d₊ for a pitch/roll axis.

    k·2(d₊Σ₊ − d₋Σ₋)/(d₊ + d₋) spans the same space as c₊Σ₊ + c₋Σ₋, so the
    unconstrained least-squares fit is the R²-optimal ratio: d₋/d₊ = −c₋/c₊.
    Returns (ratio, its standard error by the delta method).
    """
    beta, cov = ols_cov(X, y)
    cp, cn = beta[1], beta[2]
    ratio = -cn / cp
    grad = np.array([cn / cp ** 2, -1.0 / cp])
    se = float(np.sqrt(grad @ cov[1:3, 1:3] @ grad))
    return float(ratio), se


def ols_cov(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    return beta, (resid @ resid / dof) * np.linalg.inv(X.T @ X)


def ols(X, y):
    beta, cov = ols_cov(X, y)
    return beta, np.sqrt(np.diag(cov)), r2(X, y, beta)


def r2(X, y, beta):
    resid = y - X @ beta
    return 1.0 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))


def stack(flights, axis, arms, volts_map, drag=()):
    parts = [design(f, axis, arms, volts_map, drag) for f in flights]
    names = parts[0][2]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), names


def airframe(flights):
    h = next((f.header for f in flights if f.header), {})
    if not h:
        return {}
    looptime = int(h.get("looptime", 0))
    denom = int(h.get("pid_process_denom", 1))
    # Operating point for the evals, over the rows the fit kept (airborne).
    vbat = np.concatenate([f.cols["vbatLatest"][f.good] / 100.0 for f in flights])
    speed = np.concatenate([np.mean(f.motors(), axis=0)[f.good] for f in flights])
    cellv = [int(x) for x in h.get("vbatcellvoltage", "0").split(",")]
    # BF's cell detection: ceil(pack voltage / max cell voltage).
    vmax = max(float(f.cols["vbatLatest"].max()) / 100.0 for f in flights)
    cells = math.ceil(vmax / (cellv[-1] / 100.0)) if cellv[-1] > 0 else None
    return {
        "vbat_v": round(float(np.median(vbat)), 3),
        "hover_rps": round(float(np.median(speed)), 1),
        "cells": cells,
        "craft": h.get("Craft name", ""),
        "board": h.get("Board information", ""),
        "firmware": h.get("Firmware revision", ""),
        "motor_poles": int(h.get("motor_poles", 14)),
        "loop_hz": round(1e6 / (looptime * denom), 1) if looptime else None,
        "vbatcellvoltage": cellv,
        "gyro_lpf": gyro_lpf(h),
    }


LPF_TYPES = {0: "PT1", 1: "BIQUAD", 2: "PT2", 3: "PT3"}


def gyro_lpf(h):
    """
    BF's gyro low-pass stages from the log header, in order, as
    {type, hz[, max_hz]}.  LPF1 is dynamic (cutoff rising with throttle from
    hz to max_hz) when its dynamic minimum is set, else static; a 0 Hz stage
    is off.  Notches and the RPM filter are not recorded.
    """
    def num(key, default="0"):
        return [int(x) for x in h.get(key, default).split(",")]

    stages = []
    dyn_lo, dyn_hi = (num("gyro_lpf1_dyn_hz", "0,0") + [0, 0])[:2]
    lpf1 = num("gyro_lpf1_static_hz")[0]
    kind = LPF_TYPES.get(num("gyro_lpf1_type")[0], "PT1")
    if dyn_lo > 0:
        stages.append({"type": kind, "hz": dyn_lo, "max_hz": dyn_hi})
    elif lpf1 > 0:
        stages.append({"type": kind, "hz": lpf1})
    lpf2 = num("gyro_lpf2_static_hz")[0]
    if lpf2 > 0:
        stages.append({"type": LPF_TYPES.get(num("gyro_lpf2_type")[0], "PT1"), "hz": lpf2})
    return stages


def read_dump(path):
    """Motor-layout facts from a Betaflight CLI `diff all` / `dump`."""
    out = {"mixer": None, "mmix": [], "yaw_motors_reversed": False, "motor_poles": None, "name": None}
    with open(path, errors="replace") as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            if words[0] == "mixer" and len(words) > 1:
                out["mixer"] = words[1]
            elif words[0] == "mmix" and len(words) == 6:
                out["mmix"].append([int(words[1])] + [float(x) for x in words[2:]])
            elif words[:2] == ["#", "name:"]:
                out["name"] = " ".join(words[2:])
            elif words[0] == "set" and len(words) >= 4:
                if words[1] == "yaw_motors_reversed":
                    out["yaw_motors_reversed"] = words[3] == "ON"
                elif words[1] == "motor_poles":
                    out["motor_poles"] = int(words[3])
    return out


def check_layout(dump):
    """
    The regressors assume Betaflight's QUADX motor indices (0 rear-right,
    1 front-right, 2 rear-left, 3 front-left).  Prop direction and gyro or
    board alignment don't matter: BF aligns gyroADC before logging, and spin
    direction only flips signs the fit absorbs.  A different mixer or a custom
    mmix reorders which indices pair up, which the model can't absorb.
    """
    if dump["mixer"] != "QUADX" or dump["mmix"]:
        sys.exit(f"unsupported motor layout: mixer {dump['mixer']}, mmix {dump['mmix'] or 'reset'}; "
                 "the regressors assume QUADX motor indices")


# The three ways to split four motors into two pairs; for QUADX these are
# the yaw (diagonals), pitch and roll groupings of eRPM[0..3].
PAIRINGS = {"yaw": ((0, 3), (1, 2)), "pitch": ((0, 2), (1, 3)), "roll": ((2, 3), (0, 1))}


def pairing_check(flight):
    """
    For each rate axis, R² of accel ~ rate + Σω²(pair A) + Σω²(pair B) under
    each of the three pairings.  On a correctly indexed QUADX each axis is
    best explained by its own grouping; a wrong layout shows up as a mismatch.
    """
    n = flight.window("pitch")
    t = flight.t
    w = [flight.cols[f"eRPM[{i}]"] / 6.0 for i in range(4)]
    result = {}
    for axis, gyro in (("yaw", 2), ("pitch", 1), ("roll", 0)):
        rate = flight.rate(gyro)
        y = _cfd(rate, t, n)
        keep = _gate(flight, _trunc(rate, n), 0.3, n)
        scores = {}
        for name, (a, b) in PAIRINGS.items():
            X = np.column_stack([_trunc(rate, n),
                                 _trunc(_sq(w[a[0]]) + _sq(w[a[1]]), n),
                                 _trunc(_sq(w[b[0]]) + _sq(w[b[1]]), n)])
            scores[name] = float(r2(X[keep], y[keep], ols_cov(X[keep], y[keep])[0]))
        result[axis] = scores
    return result


def fit_arms(flights, arms, verbose=False, drag=()):
    """Fill in arm lengths not pinned in `arms` with the best-fit ratio."""
    arms = dict(arms)
    stats = {}
    for axis, (pos, neg) in ARM_KEYS.items():
        if arms.get(pos) is not None and arms.get(neg) is not None:
            stats[axis] = {"ratio": arms[neg] / arms[pos], "source": "given"}
            continue
        parts = [design_split(f, axis, axis in drag) for f in flights]
        Xs = np.vstack([p[0] for p in parts])
        ys = np.concatenate([p[1] for p in parts])
        ratio, se = fit_arm_ratio(Xs, ys)
        arms[pos], arms[neg] = 1.0, ratio
        profile = {}
        for scale in (0.8, 0.9, 1.0, 1.1, 1.25):
            X = np.column_stack([Xs[:, 0], arm_regressor(Xs, 1.0, ratio * scale), Xs[:, 3:]])
            profile[scale] = float(r2(X, ys, ols(X, ys)[0]))
        stats[axis] = {"ratio": ratio, "se": se, "source": "fit",
                       "r2_profile": {f"{s:g}x": round(v, 5) for s, v in profile.items()}}
        if verbose:
            print(f"{axis:7s} {neg}/{pos} = {ratio:.4f} ± {se:.4f}   R² at ratio × " +
                  "  ".join(f"{s:g}: {v:.4f}" for s, v in profile.items()))
    return arms, stats


def fit_coeffs(flights, arms, volts_map, verbose=False, drag=()):
    """Pooled fit over `flights`; returns the gen_wfg JSON dict (sans airframe)."""
    arms, arm_stats = fit_arms(flights, arms, verbose, drag)
    out = {}
    fit_stats = {}
    for axis in AXES:
        X, y, names = stack(flights, axis, arms, volts_map, drag)
        beta, se, rsq = ols(X, y)
        out[axis] = {n: float(b) for n, b in zip(names, beta)}
        fit_stats[axis] = {"r2": round(float(rsq), 4), "rows": int(len(y)),
                           "rel_se": {n: round(float(s / abs(b)), 5) for n, b, s in zip(names, beta, se)}}
        if verbose:
            print(f"{axis:7s} R²={rsq:.4f} n={len(y):7d}  " +
                  "  ".join(f"{n}={b:.6g}±{s / abs(b) * 100:.2f}%" for n, b, s in zip(names, beta, se)))
    out["pitch"].update(dFront=arms["dFront"], dBack=arms["dBack"])
    out["roll"].update(dLeft=arms["dLeft"], dRight=arms["dRight"])
    out["drag"] = {"axes": list(drag), "lp_hz": DRAG_LP_HZ}
    out["fit"] = {"sources": [f.label for f in flights], "volts": volts_map,
                  "select": flights[0].selection, "arms": arm_stats, **fit_stats}
    return out


def stored_arms(coeffs):
    return {"dFront": coeffs["pitch"]["dFront"], "dBack": coeffs["pitch"]["dBack"],
            "dLeft": coeffs["roll"]["dLeft"], "dRight": coeffs["roll"]["dRight"]}


def evaluate(coeffs, flight, arms, volts_map):
    """Per-axis R² of `coeffs` on one flight's regression rows."""
    drag = drag_axes(coeffs)
    out = {}
    for axis in AXES:
        X, y, names, _ = design(flight, axis, arms, volts_map, drag)
        out[axis] = float(r2(X, y, np.array([coeffs[axis][n] for n in names])))
    return out


def drag_axes(coeffs):
    return tuple(coeffs.get("drag", {}).get("axes", ()))


def choose_drag(flights, mode, arms, volts_map):
    """
    Resolve --drag {auto,on,off} to the axes that get drag terms, with the
    evidence for auto: per axis, the held-out R² gain from the terms
    (row-weighted mean), leaving out one flight at a time (or one quarter of
    a lone flight).
    """
    both = tuple(ARM_KEYS)
    if mode == "off":
        return (), None
    if not all(f.has_accel() for f in flights):
        if mode == "on":
            sys.exit("--drag on needs accSmooth in every log")
        return (), {"reason": "no accSmooth"}
    if mode == "on":
        return both, None
    # Props and motors get swapped between flights (and after crashes, within
    # a day), which moves the thrust coefficients; across flights the drag
    # terms soak that up.  So each fold holds out a quarter of one flight and
    # trains on the rest of that flight only.
    tasks = []
    for f in flights:
        chunks = split(f, DRAG_FOLDS)
        tasks += [(held, chunks[:i] + chunks[i + 1:]) for i, held in enumerate(chunks)]
    # Pooled over folds as 1 − ΣSSE/ΣSST, so quiet folds (tiny SST) can't
    # swing the result the way a mean of per-fold R² does.
    sse = {(axis, on): 0.0 for axis in both for on in (False, True)}
    sst = {axis: 0.0 for axis in both}
    for held, train in tasks:
        for drag in ((), both):
            arms_k, _ = fit_arms(train, arms, drag=drag)
            for axis in both:
                X, y, _ = stack(train, axis, arms_k, volts_map, drag)
                Xh, yh, _, _ = design(held, axis, arms_k, volts_map, drag)
                resid = yh - Xh @ ols(X, y)[0]
                sse[axis, bool(drag)] += float(resid @ resid)
                if not drag:
                    sst[axis] += float((yh - yh.mean()) @ (yh - yh.mean())) if len(yh) else 0.0
    heldout = {a: {on: 1.0 - sse[a, on] / sst[a] if sst[a] else 0.0 for on in (False, True)} for a in both}
    axes = tuple(a for a in both if heldout[a][True] - heldout[a][False] > DRAG_MIN_GAIN
                 and heldout[a][True] >= DRAG_MIN_HELDOUT_R2)
    return axes, {"heldout_r2": {a: {"off": round(h[False], 4), "on": round(h[True], 4)}
                                 for a, h in heldout.items()},
                      "folds": f"{DRAG_FOLDS} per flight × {len(flights)}"}


def load_all(specs, jobs, selection="robust"):
    """Decode LOG arguments (in parallel across files) into Flights, in order."""
    load = functools.partial(load_flights, selection=selection)
    if jobs > 1 and len(specs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            groups = list(pool.map(load, specs))
    else:
        groups = [load(s) for s in specs]
    flights = [f for g in groups for f in g]
    for f in flights:
        if f.gating:
            g = f.gating
            print(f"{f.label}: {f.kept_s():.0f}/{f.t[-1] - f.t[0]:.0f} s kept; ground {g['ground_s']:.1f} s, "
                  f"{g['impacts']} impact(s) masking {g['impact_s']:.1f} s",
                  file=sys.stderr)
    return flights


def given_arms(args):
    return {"dFront": args.d_front, "dBack": args.d_back, "dLeft": args.d_left, "dRight": args.d_right}


def resolve_drag(flights, args):
    drag, evidence = choose_drag(flights, args.drag, given_arms(args), args.volts)
    print(f"airframe drag: {', '.join(drag) or 'off'} (--drag {args.drag}"
          + (f"; {evidence}" if evidence else "") + ")", file=sys.stderr)
    return drag, evidence


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"wrote {path}", file=sys.stderr)


def cmd_fit(args):
    flights = load_all(args.logs, args.jobs, args.select)
    if not flights:
        sys.exit("no usable flights")
    dump = read_dump(args.dump) if args.dump else None
    if dump:
        check_layout(dump)
    drag, evidence = resolve_drag(flights, args)
    out = fit_coeffs(flights, given_arms(args), args.volts, verbose=True, drag=drag)
    if evidence:
        out["drag"]["selection"] = evidence
    out["airframe"] = airframe(flights)
    if dump:
        out["airframe"].update(mixer=dump["mixer"], yaw_motors_reversed=dump["yaw_motors_reversed"],
                               dump=os.path.basename(args.dump))
    if args.output:
        write_json(args.output, out)


def cmd_validate(args):
    with open(args.coefficients) as f:
        coeffs = json.load(f)

    def arm(axis, key, override):
        value = override if override is not None else coeffs[axis].get(key, coeffs.get(key))
        if value is None:
            sys.exit(f"{args.coefficients} has no {key}; pass --{key[0]}-{key[1:].lower()}")
        return value

    arms = {"dFront": arm("pitch", "dFront", args.d_front),
            "dBack": arm("pitch", "dBack", args.d_back),
            "dLeft": arm("roll", "dLeft", args.d_left),
            "dRight": arm("roll", "dRight", args.d_right)}
    volts_map = coeffs.get("fit", {}).get("volts", args.volts)
    drag = drag_axes(coeffs)
    print_r2_table([(f, evaluate(coeffs, f, arms, volts_map), own_r2(f, volts_map, drag))
                    for f in load_all(args.logs, args.jobs, args.select)], "given")


def own_r2(flight, volts_map, drag=()):
    own = fit_coeffs([flight], {}, volts_map, drag=drag)
    return evaluate(own, flight, stored_arms(own), volts_map)


def print_r2_table(rows, label):
    print(f"{'flight':30s} " + " ".join(f"{a:>15s}" for a in AXES))
    print(f"{'':30s} " + " ".join(f"{label + ' / own':>15s}" for _ in AXES))
    for flight, given, own in rows:
        print(f"{flight.label:30s} " + " ".join(f"{given[a]:7.3f} /{own[a]:6.3f}" for a in AXES))


def cmd_crossval(args):
    """
    Leave-one-out within an airframe: each flight is scored by a fit to all
    the others, next to its own best fit; then per-flight coefficient spread
    and the motor-pairing check.
    """
    flights = load_all(args.logs, args.jobs, args.select)
    if args.split > 1:
        flights = [part for f in flights for part in split(f, args.split)]
    if len(flights) < 2:
        sys.exit("crossval needs at least two usable flights (try --split)")
    if args.dump:
        check_layout(read_dump(args.dump))
    given = given_arms(args)
    drag, evidence = resolve_drag(flights, args)
    rows = []
    singles = []
    for i, flight in enumerate(flights):
        loo = fit_coeffs(flights[:i] + flights[i + 1:], given, args.volts, drag=drag)
        own = fit_coeffs([flight], given, args.volts, drag=drag)
        singles.append(own)
        rows.append((flight, evaluate(loo, flight, stored_arms(loo), args.volts),
                     evaluate(own, flight, stored_arms(own), args.volts)))
    print_r2_table(rows, "loo")

    print("\nper-flight fits: median, and spread as (p90 − p10)/|median|")
    keys = [(a, n) for a in AXES for n in singles[0][a]]
    for axis, name in keys:
        vals = np.array([s[axis][name] for s in singles])
        med = np.median(vals)
        spread = (np.percentile(vals, 90) - np.percentile(vals, 10)) / abs(med) if med else float("nan")
        print(f"  {axis:7s} {name:13s} {med:+.6g}  spread {spread * 100:6.1f}%")

    print("\npairing check (R² of each axis under each grouping; own grouping should win):")
    worst = {}
    for flight in flights:
        for axis, scores in pairing_check(flight).items():
            best = max(scores, key=scores.get)
            if best != axis:
                worst.setdefault(axis, []).append((flight.label, best, scores))
    for axis in ("yaw", "pitch", "roll"):
        bad = worst.get(axis, [])
        print(f"  {axis:5s}: {len(flights) - len(bad)}/{len(flights)} flights best fit by their own grouping"
              + "".join(f"\n      {lbl}: best {b} {({k: round(v, 3) for k, v in s.items()})}" for lbl, b, s in bad))

    print("\npooled:")
    pooled = fit_coeffs(flights, given, args.volts, verbose=True, drag=drag)
    if evidence:
        pooled["drag"]["selection"] = evidence
    if args.output:
        pooled["airframe"] = airframe(flights)
        write_json(args.output, pooled)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "validate", "crossval"):
        p = sub.add_parser(name)
        if name == "validate":
            p.add_argument("coefficients")
        p.add_argument("logs", nargs="+")
        for side in ("front", "back", "left", "right"):
            p.add_argument(f"--d-{side}", type=float, default=None,
                           help="override the arm length stored in the JSON" if name == "validate"
                           else "fix this arm length (with its opposite) instead of fitting the ratio")
        p.add_argument("--volts", choices=("r", "glue"), default="r")
        p.add_argument("--select", choices=("robust", "r"), default="robust",
                       help="data selection: ground/impact gating, or main.R's exactly")
        p.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1,
                       help="parallel log decoding")
        if name != "validate":
            p.add_argument("-o", "--output")
            p.add_argument("--dump", help="BF CLI diff/dump; checks the motor layout and records it")
            p.add_argument("--drag", choices=("auto", "on", "off"), default="auto",
                           help="airframe drag terms (accelerometer x/y on pitch/roll); "
                                "auto keeps them if they improve held-out R²")
        if name == "crossval":
            p.add_argument("--split", type=int, default=1,
                           help="cut each flight into N contiguous chunks and cross-validate those")
    args = parser.parse_args()
    {"fit": cmd_fit, "validate": cmd_validate, "crossval": cmd_crossval}[args.cmd](args)


if __name__ == "__main__":
    main()
