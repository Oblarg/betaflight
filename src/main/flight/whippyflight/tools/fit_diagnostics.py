"""
Fit-quality diagnostics for fit_model.py, for judging whether test flights
excited the airframe enough and whether the model explains them.

`diagnostics()` returns a JSON-ready dict (what the Configurator's
WhippyFlight tab draws); `plot()` renders the same data as matplotlib PNGs.

Per axis (yaw, pitch, roll: angular acceleration in rot/s²; motors: volts):
  * R² of the pooled fit on its own rows, and held out: leave one flight
    out with two or more flights, else each quarter of the lone flight
    scored by a fit to the other three;
  * coefficients with relative standard errors;
  * measured vs predicted on a random sample of rows, and the residual
    histogram;
  * a time-series excerpt: the most heavily excited window of the longest
    flight, measured and predicted.
Per flight: kept time, gating, the pooled fit's R² on that flight alone, and
the motor-pairing check.  `warnings` lists what looks wrong, as stable codes
with parameters, so a UI can phrase them.

    python3 fit_diagnostics.py diagnostics.json OUT_DIR

re-plots a saved fit's diagnostics without refitting (needs matplotlib).
"""

import numpy as np

import fit_model as fm

UNITS = {"yaw": "rot/s²", "pitch": "rot/s²", "roll": "rot/s²", "motors": "V"}
MAX_POINTS = 1500
HIST_BINS = 41
EXCERPT_S = 6.0
HELDOUT_FOLDS = 4

# Below these the fit is suspect enough to flag.
MIN_R2 = {"yaw": 0.5, "pitch": 0.5, "roll": 0.5, "motors": 0.9}
MAX_HELDOUT_DROP = 0.1
MIN_TOTAL_KEPT_S = 30.0
MAX_REL_SE = 0.05


def _rows(flights, axis, arms, volts_map, drag):
    parts = [fm.design(f, axis, arms, volts_map, drag) for f in flights]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _sse_sst(y, pred):
    resid = y - pred
    return float(resid @ resid), float((y - y.mean()) @ (y - y.mean())) if len(y) else 0.0


def heldout_r2(flights, arms, volts_map, drag):
    """
    Per-axis held-out R², pooled over folds as 1 − ΣSSE/ΣSST, and the method.
    Folds partition the pooled fit's own rows (by flight, or a lone flight's
    rows into contiguous blocks of equal count) rather than re-gating pieces
    of a flight, so every fold is scored on rows the deployed fit was made
    from.  Blocks are equal in rows, not duration: excitation comes in bursts,
    and equal-duration folds can leave a fold trained on a handful of rows.
    """
    method = "leave one flight out" if len(flights) >= 2 else f"{HELDOUT_FOLDS} blocks of one flight"
    out = {}
    for axis in fm.AXES:
        X, y, fold = [], [], []
        for k, f in enumerate(flights):
            Xf, yf, _, idx = fm.design(f, axis, arms, volts_map, drag)
            X.append(Xf)
            y.append(yf)
            if len(flights) >= 2:
                fold.append(np.full(len(yf), k))
            else:
                fold.append(np.arange(len(yf)) * HELDOUT_FOLDS // max(len(yf), 1))
        X, y, fold = np.vstack(X), np.concatenate(y), np.concatenate(fold)
        sse = sst = 0.0
        for k in np.unique(fold):
            held = fold == k
            if held.all():
                continue
            a, b = _sse_sst(y[held], X[held] @ fm.ols(X[~held], y[~held])[0])
            sse, sst = sse + a, sst + b
        out[axis] = round(1.0 - sse / sst, 4) if sst else None
    return out, method


def _sample(n, k, rng):
    return np.sort(rng.choice(n, size=k, replace=False)) if n > k else np.arange(n)


def _excerpt(flight, axis, beta, arms, volts_map, drag):
    """Measured and predicted over the EXCERPT_S window with the most rows."""
    X, y, _, idx = fm.design(flight, axis, arms, volts_map, drag)
    if axis == "motors" and len(idx) > 1:
        # Rows are the four motors' in turn; the first motor's run ends where idx wraps.
        wraps = np.flatnonzero(np.diff(idx) < 0)
        end = wraps[0] + 1 if len(wraps) else len(idx)
        X, y, idx = X[:end], y[:end], idx[:end]
    if not len(idx):
        return None
    t = flight.t[idx]
    counts = np.searchsorted(t, t + EXCERPT_S) - np.arange(len(t))
    start = int(np.argmax(counts))
    stop = start + int(counts[start])
    pick = start + _sample(stop - start, MAX_POINTS, np.random.default_rng(0))
    return {"flight": flight.label,
            "window_s": EXCERPT_S,
            "t": np.round(t[pick] - t[start], 4).tolist(),
            "measured": np.round(y[pick], 5).tolist(),
            "predicted": np.round(X[pick] @ beta, 5).tolist()}


def diagnostics(flights, coeffs, volts_map):
    arms = fm.stored_arms(coeffs)
    drag = fm.drag_axes(coeffs)
    heldout, method = heldout_r2(flights, arms, volts_map, drag)
    rng = np.random.default_rng(0)
    longest = max(flights, key=lambda f: f.kept_s())
    axes = {}
    warnings = []
    for axis in fm.AXES:
        names = fm.design(flights[0], axis, arms, volts_map, drag)[2]
        beta = np.array([coeffs[axis][n] for n in names])
        X, y = _rows(flights, axis, arms, volts_map, drag)
        pred = X @ beta
        resid = y - pred
        sse, sst = _sse_sst(y, pred)
        r2 = 1.0 - sse / sst if sst else 0.0
        pick = _sample(len(y), MAX_POINTS, rng)
        lim = float(np.percentile(np.abs(resid), 99.5)) if len(resid) else 1.0
        counts, edges = np.histogram(resid, bins=HIST_BINS, range=(-lim, lim))
        rel_se = coeffs.get("fit", {}).get(axis, {}).get("rel_se", {})
        axes[axis] = {
            "unit": UNITS[axis],
            "rows": int(len(y)),
            "r2": round(r2, 4),
            "heldout_r2": heldout[axis],
            "measured_std": round(float(y.std()), 5),
            "residual_std": round(float(resid.std()), 5),
            "coefficients": [{"name": n, "value": float(b), "rel_se": rel_se.get(n)}
                             for n, b in zip(names, beta)],
            "scatter": {"measured": np.round(y[pick], 5).tolist(),
                        "predicted": np.round(pred[pick], 5).tolist()},
            "residual_hist": {"edges": np.round(edges, 5).tolist(), "counts": counts.tolist()},
            "excerpt": _excerpt(longest, axis, beta, arms, volts_map, drag),
        }
        if r2 < MIN_R2[axis]:
            warnings.append({"code": "low_r2", "axis": axis, "value": round(r2, 3), "limit": MIN_R2[axis]})
        if heldout[axis] is not None and r2 - heldout[axis] > MAX_HELDOUT_DROP:
            warnings.append({"code": "overfit", "axis": axis, "r2": round(r2, 3),
                             "heldout_r2": heldout[axis]})
        for c in axes[axis]["coefficients"]:
            if c["rel_se"] is not None and c["rel_se"] > MAX_REL_SE:
                warnings.append({"code": "uncertain_coefficient", "axis": axis, "name": c["name"],
                                 "rel_se": c["rel_se"], "limit": MAX_REL_SE})

    per_flight = []
    for f in flights:
        pairing = fm.pairing_check(f)
        mismatched = [a for a, s in pairing.items() if max(s, key=s.get) != a]
        per_flight.append({"label": f.label,
                           "duration_s": round(float(f.t[-1] - f.t[0]), 1),
                           "kept_s": round(f.kept_s(), 1),
                           "log_hz": round(f.log_hz, 1),
                           "gating": f.gating,
                           "r2": {a: round(v, 4) for a, v in fm.evaluate(coeffs, f, arms, volts_map).items()},
                           "pairing": {a: {k: round(v, 4) for k, v in s.items()} for a, s in pairing.items()},
                           "pairing_mismatch": mismatched})
        if mismatched:
            warnings.append({"code": "pairing_mismatch", "flight": f.label, "axes": mismatched})
    kept = sum(f.kept_s() for f in flights)
    if kept < MIN_TOTAL_KEPT_S:
        warnings.append({"code": "little_data", "kept_s": round(kept, 1), "limit": MIN_TOTAL_KEPT_S})

    return {"version": 1,
            "heldout_method": method,
            "kept_s": round(kept, 1),
            "drag_axes": list(drag),
            "axes": axes,
            "flights": per_flight,
            "warnings": warnings}


def plot(diag, out_dir):
    """One PNG per axis (excerpt, measured vs predicted, residuals) and flights.png."""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for axis, a in diag["axes"].items():
        fig, (ax_t, ax_s, ax_h) = plt.subplots(1, 3, figsize=(15, 4.2),
                                               gridspec_kw={"width_ratios": [2.2, 1, 1]})
        title = f"{axis}: R² {a['r2']:.3f}"
        if a["heldout_r2"] is not None:
            title += f", held out {a['heldout_r2']:.3f}"
        fig.suptitle(f"{title} ({a['rows']} rows)")
        ex = a["excerpt"]
        if ex:
            ax_t.plot(ex["t"], ex["measured"], ".", ms=2, label="measured")
            ax_t.plot(ex["t"], ex["predicted"], ".", ms=2, label="predicted")
            ax_t.set_title(ex["flight"], fontsize=9)
            ax_t.set_xlim(0, ex.get("window_s", EXCERPT_S))
            ax_t.set_xlabel("s (gaps: samples left out of the fit)")
            ax_t.set_ylabel(a["unit"])
            ax_t.legend(markerscale=4)
        s = a["scatter"]
        ax_s.plot(s["measured"], s["predicted"], ".", ms=2, alpha=0.5)
        if s["measured"]:
            lo, hi = min(s["measured"]), max(s["measured"])
            ax_s.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax_s.set_xlabel(f"measured ({a['unit']})")
        ax_s.set_ylabel("predicted")
        h = a["residual_hist"]
        ax_h.stairs(h["counts"], h["edges"], fill=True)
        ax_h.set_xlabel(f"residual ({a['unit']})")
        ax_h.set_title(f"σ {a['residual_std']:.3g} of {a['measured_std']:.3g}", fontsize=9)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{axis}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)

    flights = diag["flights"]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(flights) + 3), 4))
    width = 0.8 / len(fm.AXES)
    for k, axis in enumerate(fm.AXES):
        ax.bar([i + k * width for i in range(len(flights))], [f["r2"][axis] for f in flights],
               width, label=axis)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(flights))])
    ax.set_xticklabels([f["label"] for f in flights], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R² of the pooled fit")
    ax.set_ylim(min(0.0, *(f["r2"][a] for f in flights for a in fm.AXES)), 1.0)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "flights.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path)
    return written


def main():
    import json
    import sys

    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        diag = json.load(f)
    for path in plot(diag, sys.argv[2]):
        print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
