#!/usr/bin/env python3
"""
The crate's test pipeline over the airframes in airframes/airframes.json.

Each airframe is a fit_model.py fit, a sensor-noise fixture from the same
airframe's logs (extract_noise.py), and the gen_wfg.py arguments its blob is
generated with.  One of them names a committed blob (test_gains.wfg, which
the Rust tests embed and SITL flies); the others' blobs are generated into
target/airframes/.

    python3 pipeline.py gen               regenerate every airframe's blob
    python3 pipeline.py check             committed blob up to date with its fit
    python3 pipeline.py test              cargo test, the whole suite
    python3 pipeline.py eval [--save]     SUMMARY evals per airframe, against
                                          airframes/eval_baseline.json
    python3 pipeline.py sitl [--rc-phase] rebuild SITL, run sitl_test.py
                                          (and sitl_rc_phase.py); flick plots
                                          with and without feedback in
                                          out/sitl/<airframe>/
    python3 pipeline.py all               check, test, eval, sitl

Run with the interpreter that has gen_wfg.py's dependencies (numpy, scipy,
control, frccontrol); gen_wfg.py and the SITL scripts run under it too.
"""

import argparse
import filecmp
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from sweep_qr import CARGO, build_evals

HERE = os.path.dirname(os.path.abspath(__file__))
CRATE = os.path.dirname(HERE)
REPO = os.path.realpath(os.path.join(CRATE, "..", "..", "..", ".."))
AIRFRAMES = os.path.join(HERE, "airframes")
MANIFEST = os.path.join(AIRFRAMES, "airframes.json")
BASELINE = os.path.join(AIRFRAMES, "eval_baseline.json")
GENERATED = os.path.join(CRATE, "target", "airframes")
OUT = os.path.join(CRATE, "out")
HOST_TARGET = CARGO[CARGO.index("--target") + 1]
SITL_BINARY = os.path.join(REPO, "obj", "betaflight_2026.12.0_SITL")


def airframes():
    with open(MANIFEST) as f:
        frames = json.load(f)
    for name, a in frames.items():
        a["name"] = name
        a["fit"] = os.path.join(AIRFRAMES, a["fit"])
        a["noise"] = os.path.join(AIRFRAMES, a["noise"])
        a["blob"] = os.path.normpath(os.path.join(AIRFRAMES, a["blob"])) if "blob" in a \
            else os.path.join(GENERATED, f"{name}.wfg")
    return frames


def committed(frames):
    (a,) = [a for a in frames.values() if not a["blob"].startswith(GENERATED)]
    return a


def generate(a, out, extra_args=()):
    r = subprocess.run([sys.executable, os.path.join(HERE, "gen_wfg.py"), a["fit"], out]
                       + a["gen_args"] + list(extra_args),
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"gen_wfg failed for {a['name']}:\n{r.stderr}")


def cmd_gen(frames, _args):
    os.makedirs(GENERATED, exist_ok=True)
    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda a: generate(a, a["blob"]), frames.values()))
    for a in frames.values():
        print(f"  {a['name']:12s} → {os.path.relpath(a['blob'], CRATE)}")


def cmd_check(frames, _args):
    a = committed(frames)
    with tempfile.TemporaryDirectory() as tmp:
        fresh = os.path.join(tmp, "fresh.wfg")
        generate(a, fresh)
        if not filecmp.cmp(fresh, a["blob"], shallow=False):
            sys.exit(f"{os.path.relpath(a['blob'], CRATE)} is stale: regenerate it from "
                     f"{os.path.relpath(a['fit'], CRATE)} (pipeline.py gen)")
    print(f"  {os.path.relpath(a['blob'], CRATE)} matches {os.path.relpath(a['fit'], CRATE)}")


def cmd_test(_frames, _args):
    subprocess.run(["cargo", "test", "--no-fail-fast", "--release", "--target", HOST_TARGET,
                    "--features", "std"], cwd=CRATE, check=True)


def summaries(a, evals):
    out = {}
    for exe, test in evals:
        stdout = subprocess.run([exe, test, "--exact", "--ignored", "--nocapture"], check=True,
                                capture_output=True, text=True,
                                env={**os.environ, "WFG_PATH": a["blob"], "PLANT_JSON": a["fit"],
                                     "NOISE_PATH": a["noise"]}).stdout
        for line in stdout.splitlines():
            if line.startswith("SUMMARY"):
                out.update((k, float(v)) for k, v in (kv.split("=") for kv in line.split()[1:]))
    return out


def cmd_eval(frames, args):
    cmd_gen(frames, args)
    evals = build_evals()
    with ThreadPoolExecutor() as pool:
        results = dict(zip(frames, pool.map(lambda a: summaries(a, evals), frames.values())))
    baseline = {}
    if os.path.exists(BASELINE) and not args.save:
        with open(BASELINE) as f:
            baseline = json.load(f)
    names = list(results)
    metrics = list(dict.fromkeys(k for r in results.values() for k in r))
    print(f"\n{'':24s}" + "".join(f"{n:>24s}" for n in names))
    changed = 0
    for m in metrics:
        cells = []
        for n in names:
            v, b = results[n].get(m), baseline.get(n, {}).get(m)
            if v is None:
                cells.append(f"{'—':>24s}")
            elif b is None or b == v:
                cells.append(f"{v:24.5g}")
            else:
                changed += 1
                rel = f"{100 * (v - b) / abs(b):+.1f}%" if b else "new"
                cells.append(f"{f'{v:.5g} ({rel})':>24s}")
        print(f"{m:24s}" + "".join(cells))
    if args.save:
        with open(BASELINE, "w") as f:
            json.dump(results, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"\nsaved {os.path.relpath(BASELINE, CRATE)}")
    elif baseline:
        print(f"\n{changed} value(s) differ from {os.path.relpath(BASELINE, CRATE)} (relative change in parentheses)")


def cmd_sitl(frames, args):
    a = frames[args.airframe] if args.airframe else committed(frames)
    os.makedirs(GENERATED, exist_ok=True)
    if a["blob"].startswith(GENERATED):
        generate(a, a["blob"])
    no_feedback = os.path.join(GENERATED, f"{a['name']}_no_feedback.wfg")
    generate(a, no_feedback, ["--feedback-scale", "0"])
    subprocess.run(["cargo", "build", "-q", "--release", "--target", HOST_TARGET], cwd=CRATE, check=True)
    # make doesn't see the Rust static library as a dependency of the binary.
    if os.path.exists(SITL_BINARY):
        os.remove(SITL_BINARY)
    subprocess.run(["make", "TARGET=SITL", "OPTIONS=USE_WHIPPYFLIGHT", f"-j{os.cpu_count()}"],
                   cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    env = {**os.environ, "WFG_PATH": a["blob"], "PLANT_JSON": a["fit"],
           "WFG_NO_FEEDBACK_PATH": no_feedback, "SITL_OUT": os.path.join(OUT, "sitl", a["name"])}
    subprocess.run([sys.executable, os.path.join(HERE, "sitl_test.py")], env=env, check=True)
    if args.rc_phase:
        subprocess.run([sys.executable, os.path.join(HERE, "sitl_rc_phase.py")], env=env, check=True)


def cmd_all(frames, args):
    for step in (cmd_check, cmd_test, cmd_eval, cmd_sitl):
        print(f"\n=== {step.__name__[4:]} ===", flush=True)
        step(frames, args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("gen", "check", "test"):
        sub.add_parser(name)
    for name in ("eval", "all"):
        p = sub.add_parser(name)
        p.add_argument("--save", action="store_true", help="write the results as the new baseline")
    for name in ("sitl", "all"):
        p = sub.choices.get(name) or sub.add_parser(name)
        p.add_argument("--airframe", help="fly this airframe's blob (its loop rate must be one SITL "
                                          "supports); default: the committed blob")
        p.add_argument("--rc-phase", action="store_true", help="also run sitl_rc_phase.py")
    args = ap.parse_args()
    frames = airframes()
    {"gen": cmd_gen, "check": cmd_check, "test": cmd_test, "eval": cmd_eval,
     "sitl": cmd_sitl, "all": cmd_all}[args.cmd](frames, args)


if __name__ == "__main__":
    main()
