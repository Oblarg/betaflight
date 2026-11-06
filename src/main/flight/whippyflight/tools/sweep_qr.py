#!/usr/bin/env python3
"""
Sweep the LQR cost weights and integral/derivative time constants in
gen_wfg.py and score each blob with
tests/eval_gyro_filter.rs::lag_margin_summary and
tests/eval_feedforward.rs::flick_summary.

  python3 sweep_qr.py [--out /tmp/qr_sweep.csv] [--jobs 4]
                      [--airframe NAME | --coefficients FIT.json --noise NOISE.bin]

Needs gen_wfg.py's dependencies (numpy, scipy, control, frccontrol).
"""

import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
COEFFS = os.path.join(HERE, "airframes", "meteor65.json")

SIGMA_RATE = [1.0, 2.0, 5.0, 20.0]
SIGMA_DERIV = [5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
SIGMA_INT = [0.004, 0.008, 0.012, 0.02, 0.04]  # rot/s (gen_wfg --sigma-int)

EVALS = {"eval_gyro_filter": "lag_margin_summary", "eval_feedforward": "flick_summary"}
CARGO = ["cargo", "test", "--release", "--target", "aarch64-apple-darwin", "--features", "std",
         "--no-run", "--message-format=json"] + [a for name in EVALS for a in ("--test", name)]


def build_evals():
    out = subprocess.run(CARGO, cwd=CRATE, check=True, capture_output=True, text=True).stdout
    exes = {}
    for line in out.splitlines():
        msg = json.loads(line)
        if msg.get("reason") == "compiler-artifact" and msg["target"]["name"] in EVALS:
            exes[msg["target"]["name"]] = msg["executable"]
    missing = set(EVALS) - set(exes)
    if missing:
        raise RuntimeError(f"eval executables not found: {sorted(missing)}")
    return [(exes[name], test) for name, test in EVALS.items()]


def score(point, tmp, evals, gen_args, coeffs, noise):
    rate, deriv, integ, tau_int, tau_deriv = point
    blob = os.path.join(tmp, f"r{rate}_d{deriv}_i{integ}_ti{tau_int}_td{tau_deriv}.wfg")
    gen = subprocess.run([sys.executable, os.path.join(HERE, "gen_wfg.py"), coeffs, blob,
                          "--sigma-rate", str(rate), "--sigma-deriv", str(deriv), "--sigma-int", str(integ),
                          "--tau-int", str(tau_int), "--tau-deriv", str(tau_deriv)] + gen_args,
                         capture_output=True, text=True)
    if gen.returncode != 0:
        print(f"  gen_wfg failed at {point}: {gen.stderr.strip().splitlines()[-1]}", flush=True)
        return None
    row = {"sigma_rate": rate, "sigma_deriv": deriv, "sigma_int": integ,
           "tau_int": tau_int, "tau_deriv": tau_deriv}
    for exe, test in evals:
        out = subprocess.run([exe, test, "--exact", "--ignored", "--nocapture"],
                             env={**os.environ, "WFG_PATH": blob, "PLANT_JSON": coeffs,
                                  **({"NOISE_PATH": noise} if noise else {})},
                             check=True, capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("SUMMARY"))
        row.update(kv.split("=") for kv in line.split()[1:])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/qr_sweep.csv")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--coefficients", default=COEFFS,
                    help="fit JSON: generates the blobs and is the evals' plant")
    ap.add_argument("--noise", default=None,
                    help="sensor-noise fixture (tools/extract_noise.py) from a log of the "
                         "--coefficients airframe; default: the evals' default fixture "
                         "(tools/airframes/meteor65_noise.bin)")
    ap.add_argument("--sigma-rate", type=float, nargs="+", default=SIGMA_RATE)
    ap.add_argument("--sigma-deriv", type=float, nargs="+", default=SIGMA_DERIV)
    ap.add_argument("--sigma-int", type=float, nargs="+", default=SIGMA_INT)
    ap.add_argument("--tau-int", type=float, nargs="+", default=[10.0])
    ap.add_argument("--tau-deriv", type=float, nargs="+", default=[0.02])
    ap.add_argument("--gen-args", default="",
                    help="extra gen_wfg.py arguments, e.g. '--ff-horizon 16'")
    ap.add_argument("--airframe", default=None,
                    help="an airframe of airframes/airframes.json: its fit, noise fixture and "
                         "gen_wfg arguments (before --gen-args)")
    args = ap.parse_args()
    if args.airframe:
        from pipeline import airframes
        a = airframes()[args.airframe]
        args.coefficients, args.noise = a["fit"], a["noise"]
        args.gen_args = shlex.join(a["gen_args"]) + " " + args.gen_args

    evals = build_evals()
    points = list(itertools.product(args.sigma_rate, args.sigma_deriv, args.sigma_int,
                                    args.tau_int, args.tau_deriv))
    rows = []
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(args.jobs) as pool:
        gen_args = shlex.split(args.gen_args)
        for i, row in enumerate(pool.map(lambda p: score(p, tmp, evals, gen_args, args.coefficients,
                                                                   args.noise), points)):
            if row is None:
                continue
            rows.append(row)
            print(f"[{i + 1}/{len(points)}] " + " ".join(f"{k}={v}" for k, v in row.items()),
                  flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
