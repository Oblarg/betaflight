#!/usr/bin/env python3
"""
Extract sensor noise from a blackbox log (.bbl/.bfl, LOG#n for one flight
of a multi-flight file, or a blackbox_decode CSV) into the fixture the
pure-Rust evals replay (tests/common/noise.rs, $NOISE_PATH).

  python3 extract_noise.py LOG OUT.bin [--seconds 3.0] [--json INFO.json]

Noise is what the controller would see on top of the true motion:
  - gyro: BF's filtered gyro (gyroADC, after its LPFs/RPM/dynamic notches),
    zero-phase high-passed at --gyro-hp Hz to remove body motion;
  - motor speed: eRPM telemetry in shaft rev/s (from the log's motor_poles, or
    --motor-poles; the glue converts the same way), high-passed at --erpm-hp Hz (the motors' own time constant is ~18 ms, so
    real speed ripple above that is negligible next to telemetry jitter).
Content below the cut-offs is not represented.

The window is the stretch of flight (sticks within --max-stick deg/s, all
motors above idle) whose mean motor speed is closest to --hover-rps
(default: the median of those samples).

Also prints BF's own motor-voltage roughness over that window, computed like
the evals' `roughness_v` (one-pole high-pass at ROUGH_HP_HZ), as a budget
reference: that much ripple is known to fly on this airframe.  --json also
writes it, with the window, for the Configurator.

Output format (little-endian):
  f32 sample_rate_hz, u32 n, then n frames of i16[7]:
  gyro roll/pitch/yaw (GYRO_LSB rot/s), motor 0..3 (ERPM_LSB shaft rev/s).
"""

import argparse
import csv
import json
import struct

import numpy as np
from scipy import signal

GYRO_LSB = 1e-5
ERPM_LSB = 1e-3
ROUGH_HP_HZ = 50.0


COLS = ["time", "vbatLatest"] + [f"gyroADC[{i}]" for i in range(3)] + \
       [f"motor[{i}]" for i in range(4)] + [f"eRPM[{i}]" for i in range(4)] + \
       [f"setpoint[{i}]" for i in range(3)]


def load(spec, motor_poles=None):
    """A blackbox_decode CSV, or a .bbl/.bfl (first usable flight, or LOG#n)."""
    from fit_model import load_flights, shaft_rps
    if spec.lower().endswith(".csv"):
        if not motor_poles:
            raise SystemExit("a CSV doesn't record motor_poles; pass --motor-poles")
        with open(spec) as f:
            r = csv.reader(f)
            hdr = next(r)
            idx = [hdr.index(c) for c in COLS]
            rows = []
            for row in r:
                try:
                    rows.append([float(row[i]) for i in idx])
                except (ValueError, IndexError):
                    pass
        d = np.array(rows)
    else:
        flights = load_flights(spec, motor_poles=motor_poles)
        if not flights:
            raise SystemExit(f"no usable flight in {spec}")
        d = np.column_stack([flights[0].cols[c] for c in COLS])
        motor_poles = flights[0].motor_poles
    t = (d[:, 0] - d[0, 0]) * 1e-6
    return dict(t=t, fs=1.0 / np.median(np.diff(t)), vbat=d[:, 1] / 100.0,
                gyro=d[:, 2:5] / 360.0, motor=d[:, 5:9] / 2047.0, erpm=shaft_rps(d[:, 9:13], motor_poles),
                setpoint=d[:, 13:16])


def one_pole_hp(x, fc, fs):
    k = 2 * np.pi * fc / fs
    k = k / (k + 1)
    return x - signal.lfilter([k], [1, k - 1], x, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--hover-rps", type=float, default=None,
                    help="target mean motor speed, shaft rev/s (default: the median over "
                         "qualifying samples)")
    ap.add_argument("--max-stick", type=float, default=200.0)
    ap.add_argument("--gyro-hp", type=float, default=80.0)
    ap.add_argument("--erpm-hp", type=float, default=40.0)
    ap.add_argument("--motor-poles", type=int, default=None,
                    help="the motors' pole count (default: the log's motor_poles)")
    ap.add_argument("--json", help="also write the window and BF's roughness_v here")
    args = ap.parse_args()

    d = load(args.log, args.motor_poles)
    fs, n = d["fs"], int(args.seconds * d["fs"])
    spinning = d["erpm"].mean(1)
    idle = 0.4 * np.median(spinning[spinning > 0])
    ok = (d["erpm"].min(1) > idle) & (np.abs(d["setpoint"]).max(1) < args.max_stick)
    run = np.convolve(ok, np.ones(n), "valid") >= n
    starts = np.flatnonzero(run)
    if len(starts) == 0:
        raise SystemExit(f"no {args.seconds:.1f} s window meets the criteria")
    mean_w = np.convolve(d["erpm"].mean(1), np.ones(n) / n, "valid")
    hover = args.hover_rps if args.hover_rps is not None else float(np.median(d["erpm"][ok].mean(1)))
    a = starts[np.argmin(np.abs(mean_w[starts] - hover))]
    pad = int(0.2 * fs)
    lo, hi = max(a - pad, 0), min(a + n + pad, len(d["t"]))

    def hp(x, fc):
        sos = signal.butter(4, fc, "highpass", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, x[lo:hi], axis=0)[a - lo:a - lo + n]

    gyro = hp(d["gyro"], args.gyro_hp)
    erpm = hp(d["erpm"], args.erpm_hp)
    frames = np.hstack([gyro / GYRO_LSB, erpm / ERPM_LSB])
    if np.abs(frames).max() > 32767:
        raise SystemExit("noise exceeds the i16 range; raise GYRO_LSB/ERPM_LSB")
    with open(args.out, "wb") as f:
        f.write(struct.pack("<fI", fs, n))
        f.write(np.round(frames).astype("<i2").tobytes())

    volts = d["motor"][lo:hi] * d["vbat"][lo:hi, None]
    rough = one_pole_hp(volts, ROUGH_HP_HZ, fs)[a - lo:a - lo + n]
    print(f"window {d['t'][a]:.2f}–{d['t'][a + n - 1]:.2f} s at {fs:.0f} Hz, "
          f"mean motor speed {mean_w[a]:.0f} rev/s, vbat {d['vbat'][a:a + n].mean():.2f} V")
    print("gyro noise rms (rot/s) roll/pitch/yaw: " + " ".join(f"{x:.5f}" for x in gyro.std(0)))
    print("motor speed noise rms (rev/s) motors 0-3: " + " ".join(f"{x:.3f}" for x in erpm.std(0)))
    print(f"BF roughness_v (rms of {ROUGH_HP_HZ:.0f} Hz high-passed motor volts, mean over motors): "
          f"{np.sqrt((rough ** 2).mean()):.4f} V")
    print(f"wrote {args.out} ({8 + frames.size * 2} bytes)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"log": args.log, "window_s": [float(d["t"][a]), float(d["t"][a + n - 1])],
                       "bf_rough_v": float(np.sqrt((rough ** 2).mean()))}, f)


if __name__ == "__main__":
    main()
