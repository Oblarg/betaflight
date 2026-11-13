#!/usr/bin/env python3
"""
Generate gain blobs over a grid of the two LQR weights that set the
controller's tradeoffs, for the Configurator to score and draw as a Pareto
front (see sweep_qr.py for what they trade).  σ_deriv trades disturbance
rejection against motor roughness and delay margin; σ_int trades attitude
hold against delay margin.  σ_rate barely matters and is held fixed.

  python3 front_wfg.py FIT.json OUT_DIR [--sigma-deriv ...] [--sigma-int ...]
                       [--jobs N] [gen_wfg.py arguments, e.g. --dt 1.25e-4]

Writes OUT_DIR/<n>.wfg and OUT_DIR/front.json:
  {"dt": s, "sigma_rate": rot/s, "resolution": N, "idle_rps": rev/s (0: none),
   "points": [{"sigma_deriv", "sigma_int", "wfg": "<n>.wfg"}, ...]}
"""

import argparse
import contextlib
import io
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

SIGMA_DERIV = [5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
SIGMA_INT = [0.004, 0.006, 0.008, 0.012, 0.02, 0.03, 0.05]
FRONT = "front.json"


def solve(argv):
    """gen_wfg on `argv`, quietly: (blob, its arguments with the defaults filled in)."""
    import gen_wfg
    args = gen_wfg.parse_args(argv)
    with contextlib.redirect_stderr(io.StringIO()):
        blob, _ = gen_wfg.generate(args)
    return blob, args


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coefficients", help="the fit's coefficients.json")
    ap.add_argument("out", help="directory for the blobs and front.json")
    ap.add_argument("--sigma-deriv", type=float, nargs="+", default=SIGMA_DERIV)
    ap.add_argument("--sigma-int", type=float, nargs="+", default=SIGMA_INT)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args, gen_args = ap.parse_known_args()
    if any(a in gen_args for a in ("--sigma-deriv", "--sigma-int")):
        ap.error("the grid's weights can't be passed through to gen_wfg.py")

    os.makedirs(args.out, exist_ok=True)
    grid = list(itertools.product(args.sigma_deriv, args.sigma_int))
    print(f"Solving gains at {len(grid)} weightings ({len(args.sigma_deriv)} σ_deriv × "
          f"{len(args.sigma_int)} σ_int) on {args.jobs} processes...", flush=True)
    points, gen = [None] * len(grid), None
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {
            pool.submit(solve, [args.coefficients, os.devnull, *gen_args,
                                "--sigma-deriv", repr(deriv), "--sigma-int", repr(integ)]): i
            for i, (deriv, integ) in enumerate(grid)
        }
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            deriv, integ = grid[i]
            try:
                blob, gen = future.result()
            except (Exception, SystemExit) as e:
                print(f"  [{done}/{len(grid)}] σ_deriv {deriv:g}, σ_int {integ:g}: failed: {e}",
                      file=sys.stderr, flush=True)
                continue
            name = f"{i}.wfg"
            with open(os.path.join(args.out, name), "wb") as f:
                f.write(blob)
            points[i] = {"sigma_deriv": deriv, "sigma_int": integ, "wfg": name}
            print(f"  [{done}/{len(grid)}] σ_deriv {deriv:g}, σ_int {integ:g}", flush=True)
    points = [p for p in points if p is not None]
    if not points:
        sys.exit("no gains could be computed")
    with open(os.path.join(args.out, FRONT), "w") as f:
        json.dump({"dt": gen.dt, "sigma_rate": gen.sigma_rate, "resolution": gen.resolution,
                   "idle_rps": gen.idle_rps, "points": points}, f, indent=1)
    print(f"Wrote {len(points)} blobs and {FRONT} to {args.out}", flush=True)


if __name__ == "__main__":
    main()
