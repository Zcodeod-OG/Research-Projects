#!/usr/bin/env python3
"""Which Rail-VIVID channel and frequency band actually carries track position?

The impact score in diagnose_vivid.py showed no joints and almost no agreement between
runs. Before tuning a detector, this checks every channel x frequency band:

1. Repeatability: per 1 m of track, each feature is z-scored per run; r is how well a
   run's profile matches the mean of the other runs. Anything tied to the track
   (curves, joints, switches) repeats; noise and vehicle-only effects don't.
2. Alignment: the slow (< 2 Hz) signal follows the track's curves and grades, so it is a
   positional fingerprint. Each run is shifted to best match the others on the most
   repeatable slow channel, which corrects GPS offsets run by run.
3. Anomaly contrast: after alignment, how much higher each feature is within --tol of
   the 9 known anomalies than at random sets of 9 positions (p = share of random sets
   that do at least as well; small p = the feature sees the anomalies).

Dead stretches (a channel stuck at a constant value) are masked out.

    python sweep_vivid.py --data /content/drive/MyDrive/rail-v_dataset --out $RUNS/vivid_sweep
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from anomaly_vivid import CHANNELS, Track, find_runs

warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "All-NaN slice")
warnings.filterwarnings("ignore", "Degrees of freedom")
warnings.filterwarnings("ignore", "invalid value encountered")

BANDS = [(2, 10), (10, 30), (30, 80), (80, 200), (200, 500), (500, 990)]
FEATURES = ["slow<2Hz"] + [f"{lo}-{hi}Hz" for lo, hi in BANDS] + ["impact>30Hz"]


def load(csv_path, track):
    df = pd.read_csv(csv_path, usecols=["Timestamp"] + CHANNELS + ["Latitude", "Longitude"]).dropna()
    t = pd.to_datetime(df["Timestamp"]).astype("int64").to_numpy() / 1e9
    fs = 1.0 / np.median(np.diff(t))
    dist = track.distance(df["Latitude"].to_numpy(), df["Longitude"].to_numpy())
    return df[CHANNELS].to_numpy(np.float64), fs, dist


def per_metre(acc, fs, dist, length):
    """(length, channels, features) array of per-metre features, NaN where missing or dead."""
    b = np.floor(dist).astype(int)
    ok = (b >= 0) & (b < length)
    sigs = [sosfiltfilt(butter(2, 2, "lowpass", fs=fs, output="sos"), acc, axis=0)]
    sigs += [sosfiltfilt(butter(4, [lo, min(hi, 0.45 * fs)], "bandpass", fs=fs, output="sos"), acc, axis=0)
             for lo, hi in BANDS]
    hp = sosfiltfilt(butter(4, 30, "highpass", fs=fs, output="sos"), acc, axis=0)
    order = np.flatnonzero(ok)[np.argsort(b[ok], kind="stable")]
    bs = b[order]
    edges = np.flatnonzero(np.r_[True, np.diff(bs) != 0, True])
    out = np.full((length, acc.shape[1], len(FEATURES)), np.nan)
    raw_std = np.array([acc[order[s:e]].std(0) for s, e in zip(edges[:-1], edges[1:])])
    alive = raw_std > 0.02 * np.nanmedian(raw_std, 0)  # a stuck channel has ~zero spread
    for k, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        if e - s < 16:
            continue
        idx = order[s:e]
        f = [sigs[0][idx].mean(0)]
        f += [np.log(np.sqrt((sig[idx] ** 2).mean(0)) + 1e-12) for sig in sigs[1:]]
        h = np.abs(hp[idx])
        f.append(np.log(h.max(0) / (np.sqrt((h ** 2).mean(0)) + 1e-12)))
        row = np.stack(f, 1)
        row[~alive[k]] = np.nan
        out[bs[s]] = row
    med = np.nanmedian(out, 0)
    mad = np.nanmedian(np.abs(out - med), 0) * 1.4826 + 1e-9
    return (out - med) / mad


def shifted(p, s):
    out = np.full_like(p, np.nan)
    if s >= 0:
        out[s:] = p[:len(p) - s]
    else:
        out[:s] = p[-s:]
    return out


def corr(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 100 else np.nan


def repeatability(F):
    """F: (runs, length). Mean over runs of corr(run, mean of the others)."""
    return float(np.nanmean([corr(F[i], np.nanmean(np.delete(F, i, 0), 0)) for i in range(len(F))]))


def contrast(mean, truth, tol, rng, n_null=500):
    """Mean of the window max around each anomaly, and the share of random 9-sets doing as well."""
    valid = np.flatnonzero(~np.isnan(mean))
    lo, hi = valid[0] + tol, valid[-1] - tol

    def stat(pos):
        return np.mean([np.nanmax(mean[int(p - tol):int(p + tol) + 1]) for p in pos])

    obs = stat(truth)
    null = [stat(rng.uniform(lo, hi, len(truth))) for _ in range(n_null)]
    return obs, float(np.mean(np.array(null) >= obs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/vivid_sweep"))
    ap.add_argument("--anomalies", type=Path, default=Path(__file__).with_name("vivid_anomalies.csv"))
    ap.add_argument("--max-shift", type=int, default=30)
    ap.add_argument("--trim-m", type=int, default=20)
    ap.add_argument("--tol", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    runs = find_runs(args.data)
    ref = pd.read_csv(runs[0], usecols=["Latitude", "Longitude"]).dropna()
    track = Track(ref["Latitude"].to_numpy(), ref["Longitude"].to_numpy())
    length = int(np.ceil(track.s[-1])) + 1
    t = pd.read_csv(args.anomalies)
    truth = track.distance(t["lat"].to_numpy(), t["lon"].to_numpy())

    names, feats = [], []
    for p in runs:
        acc, fs, dist = load(p, track)
        F = per_metre(acc, fs, dist, length)
        have = np.flatnonzero(np.any(~np.isnan(F), (1, 2)))
        F[:have[0] + args.trim_m] = np.nan
        F[have[-1] - args.trim_m + 1:] = np.nan
        dead = float(np.mean(np.isnan(F[have[0] + args.trim_m:have[-1] - args.trim_m, :, 1])))
        names.append(p.parent.name)
        feats.append(F)
        print(f"  {p.parent.name}: {dead:.1%} of channel-metres dead or missing")
    X = np.array(feats)  # (runs, length, channels, features)
    nch = X.shape[2]

    def table(title, fn, fmt="{:6.2f}"):
        print(f"\n{title}")
        print("channel  " + "".join(f"{f:>12s}" for f in FEATURES))
        vals = np.zeros((nch, len(FEATURES)))
        for c in range(nch):
            vals[c] = [fn(c, k) for k in range(len(FEATURES))]
            print(f"{CHANNELS[c]:9s}" + "".join(f"{fmt.format(v):>12s}" for v in vals[c]))
        return vals

    rep0 = table("repeatability r before alignment (1 = identical profile every run)",
                 lambda c, k: repeatability(X[:, :, c, k]))

    # Align every run on the most repeatable slow channel.
    c_ref = int(np.nanargmax(rep0[:, 0]))
    slow = X[:, :, c_ref, 0]
    shifts = np.zeros(len(X), int)
    for _ in range(3):
        A = np.array([shifted(p, s) for p, s in zip(slow, shifts)])
        for i in range(len(X)):
            others = np.nanmean(np.delete(A, i, 0), 0)
            scores = [corr(shifted(slow[i], s), others) for s in range(-args.max_shift, args.max_shift + 1)]
            shifts[i] = int(np.nanargmax(scores)) - args.max_shift
        shifts -= int(np.median(shifts))
    print(f"\nalignment on {CHANNELS[c_ref]} slow signal (r before {rep0[c_ref, 0]:.2f}); shift per run (m):")
    print("  " + ", ".join(f"{n} {s:+d}" for n, s in zip(names, shifts)))
    XA = np.array([np.stack([np.stack([shifted(X[i, :, c, k], s) for k in range(len(FEATURES))], 1)
                             for c in range(nch)], 1) for i, s in enumerate(shifts)])

    rep1 = table("repeatability r after alignment", lambda c, k: repeatability(XA[:, :, c, k]))
    means = np.nanmean(XA, 0)  # (length, channels, features)
    res = {}

    def p_of(c, k):
        obs, p = contrast(means[:, c, k], truth, args.tol, rng, 300)
        res[(c, k)] = (obs, p)
        return p

    pv = table(f"anomaly contrast p after alignment (share of random 9-sets scoring as high "
               f"within {args.tol} m; < 0.05 = sees the anomalies)", p_of, "{:6.3f}")
    # lowest p; ties broken by the larger contrast
    c_best, k_best = min(res, key=lambda ck: (res[ck][1], -res[ck][0]))
    print(f"\nbest: {CHANNELS[c_best]} {FEATURES[k_best]} (p={pv[c_best, k_best]:.3f}, "
          f"r={rep1[c_best, k_best]:.2f})")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep.json").write_text(json.dumps({
        "features": FEATURES, "channels": CHANNELS, "repeat_before": rep0.round(3).tolist(),
        "repeat_after": rep1.round(3).tolist(), "anomaly_p": pv.round(3).tolist(),
        "align_channel": CHANNELS[c_ref], "shifts_m": dict(zip(names, shifts.tolist()))}, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.3, 1.3]})
        axes[0].imshow(np.clip(XA[:, :, c_best, k_best], -2, 4), aspect="auto", cmap="magma",
                       interpolation="nearest", extent=[0, length, len(names) - 0.5, -0.5])
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=6)
        axes[0].set_title(f"{CHANNELS[c_best]} {FEATURES[k_best]} per metre, aligned runs", fontsize=9)
        axes[1].plot(means[:, c_best, k_best], lw=0.8, color="#1e5a78")
        axes[1].set_ylabel("run mean", fontsize=8)
        axes[2].plot(means[:, c_ref, 0], lw=0.8, color="#7a4e9c")
        axes[2].set_ylabel(f"{CHANNELS[c_ref]} slow", fontsize=8)
        axes[2].set_xlabel("distance from Point A end (m)")
        for ax in axes:
            for m in truth:
                ax.axvline(m, color="#2bb3c0", lw=0.7, alpha=0.8)
        fig.tight_layout()
        fig.savefig(args.out / "sweep_best.png", dpi=110)
        print(f"plot: {args.out / 'sweep_best.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
