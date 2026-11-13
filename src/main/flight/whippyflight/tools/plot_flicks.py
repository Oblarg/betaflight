#!/usr/bin/env python3
"""
Plot the stick flicks the Configurator's WhippyFlight tab flies computed
gains through (src/sim/flick.rs, written as gains/flicks.json): one
full-stick flick per axis, the fastest raised cosine within the setpoint
target's slew and jerk limits.

    python3 plot_flicks.py flicks.json OUT_DIR

Writes OUT_DIR/flicks.png: per axis, the whole flick and its onset and
release up close (stick, feedforward target and response in rot/s, motor
saturation shaded), and under it the four motor voltages over the same
spans.  Needs matplotlib.
"""

import json
import os
import sys


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def times(report, f):
    """Trace times, ms from the stick's start."""
    return [1e3 * i * report["trace_dt"] for i in range(len(f["response"]))]


def shade_saturation(ax, t, saturated):
    start = None
    for ti, s in zip(t + [None], saturated + [0.0]):
        if s > 0 and start is None:
            start = ti
        elif s == 0 and start is not None:
            ax.axvspan(start, ti if ti is not None else t[-1], color="red", alpha=0.12, lw=0)
            start = None


def describe(m):
    """The controller's part: how far the gyro trails the target, and how it lands."""
    behind = m["gyro_lag_ms"] - m["target_lag_ms"]
    rough = "" if m.get("rough_v") is None else f", roughness {1e3 * m['rough_v']:.0f} mV"
    return (f"gyro {behind:.1f} ms behind target, overshoot {100 * m['overshoot']:+.1f}%, "
            f"sat {100 * m['sat_frac']:.0f}%{rough}")


def windows(report, f):
    """(title, ms span) of the whole flick, its onset and its release."""
    rise = f["rise_ms"]
    release = rise + 1e3 * report["hold_s"]
    near = 1.6 * rise + 30
    return [("", None), ("onset", (-5, near)), ("release", (release - 5, release + near))]


def plot(report, out_dir):
    """Write the PNG; returns its paths."""
    plt = pyplot()
    os.makedirs(out_dir, exist_ok=True)
    flicks = report["flicks"]
    fig, axes = plt.subplots(2 * len(flicks), 3, figsize=(15, 6.4 * len(flicks)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 1, 1]})
    for row, motor_row, f in zip(axes[::2], axes[1::2], flicks):
        t = times(report, f)
        name = f"{f['axis']} {360 * f['amp']:.0f}°/s"
        for ax, (part, span) in zip(row, windows(report, f)):
            shade_saturation(ax, t, f["saturated"])
            ax.plot(t, f["stick"], color="0.5", ls="--", lw=1, label="stick")
            ax.plot(t, f["target"], color="tab:orange", lw=1, label="target")
            ax.plot(t, f["response"], color="tab:blue", lw=1.2, label="response")
            if span:
                ax.set_xlim(*span)
                ax.set_title(f"{name} {part}", fontsize=9)
            else:
                ax.set_title(f"{name}, {f['rise_ms']:.0f} ms stick: {describe(f['metrics'])}", fontsize=9)
            ax.set_xlabel("ms from stick start")
            ax.set_ylabel("rot/s")
            ax.grid(alpha=0.3)
        for ax, (part, span) in zip(motor_row, windows(report, f)):
            shade_saturation(ax, t, f["saturated"])
            for m, volts in enumerate(f["volts"]):
                ax.plot(t, volts, lw=1, label=f"motor {m + 1}")
            if span:
                ax.set_xlim(*span)
            ax.set_title(f"{name} motors {part}".rstrip(), fontsize=9)
            ax.set_xlabel("ms from stick start")
            ax.set_ylabel("V")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    rates = {"fc": "the FC's max rates", "log": "the log's max rates"}.get(report["rates_from"],
                                                                         "Betaflight's default max rates")
    hover, bf = report.get("hover_rough_v"), report.get("bf_rough_v")
    bf = "" if hover is None else (
        f"; noise roughness at hover {1e3 * hover:.1f} mV"
        + ("" if bf is None else f" (BF's total ripple at hover on the log {1e3 * bf:.0f} mV)"))
    robust = []
    margin = report.get("delay_margin_ms")
    if margin is not None:
        robust.append("delay margin none" if margin < 0 else f"delay margin {margin:.1f} ms")
    if report.get("trim_deg") is not None:
        robust.append(f"5% CG shift {report['trim_deg']:.1f}°")
    robust = "" if not robust else "; " + ", ".join(robust)
    fig.suptitle(f"Full-stick flicks at {rates}, on the {report['fit']} plant{robust}\n"
                 f"target limits {report['slew']:.0f} rot/s², {report['jerk']:.0f} rot/s³; red: motors saturated{bf}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    path = os.path.join(out_dir, "flicks.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        report = json.load(f)
    for path in plot(report, sys.argv[2]):
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
