#!/usr/bin/env python3
"""Why does (or doesn't) the Rail-VIVID anomaly detector see the known anomalies?

A joint or a switch is a short impact, so this looks at the simplest possible signal for
it, how impulsive each 1 m piece of track is, and checks three things the full detector
depends on:

1. Timing and GPS: real sample rate, how often the GPS position updates, run speed.
2. Alignment: whether the runs line up by position. GPS lag and the antenna-to-sensor
   offset push the two travel directions apart, so the averaged impact profiles of
   the A->B and B->A runs are cross-correlated to find that offset; each direction is
   moved half of it. Per-run shifts against the rest are printed too, for information.
3. Visibility: after aligning, the average impact score around each known anomaly
   versus ordinary track, and a simple detector (peaks of the run-averaged score).

It prints a compact summary and saves impact_heatmap.png (runs x distance, known
anomalies as red lines): if the joints are visible, they show as vertical stripes.

    python diagnose_vivid.py --data /content/drive/MyDrive/rail-v_dataset --out $RUNS/vivid_diag
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from anomaly_vivid import CHANNELS, Track, find_runs


def load_run(csv_path, track, hp_hz):
    df = pd.read_csv(csv_path, usecols=["Timestamp"] + CHANNELS + ["Latitude", "Longitude"]).dropna()
    t = pd.to_datetime(df["Timestamp"]).astype("int64").to_numpy() / 1e9
    t = t - t[0]
    fs = 1.0 / np.median(np.diff(t))
    lat, lon = df["Latitude"].to_numpy(), df["Longitude"].to_numpy()
    change = np.flatnonzero(np.r_[True, (np.diff(lat) != 0) | (np.diff(lon) != 0)])
    gps_dt = float(np.median(np.diff(t[change]))) if len(change) > 1 else float("nan")
    dist = track.distance(lat, lon)
    if len(change) > 1:  # interpolate between GPS fixes in time
        dist = np.interp(t, t[change], dist[change])
    acc = df[CHANNELS].to_numpy(np.float64)
    acc = sosfiltfilt(butter(4, hp_hz, "highpass", fs=fs, output="sos"), acc, axis=0)
    info = {"fs_hz": round(fs, 1), "duration_s": round(float(t[-1]), 1), "gps_update_s": round(gps_dt, 3),
            "gps_fixes": int(len(change)), "time_gaps_over_10ms": int(np.sum(np.diff(t) > 0.01)),
            "speed_m_s": round(float(np.ptp(dist) / max(t[-1], 1e-9)), 2)}
    info["direction"] = "AtoB" if dist[-1] > dist[0] else "BtoA"
    return acc, dist, info


def impact_profile(acc, dist, length):
    """Per 1 m: log(peak |a| / run median rms) per channel, z-scored per run, averaged
    over channels. NaN where the run has no samples."""
    b = np.floor(dist).astype(int)
    ok = (b >= 0) & (b < length)
    b, acc = b[ok], acc[ok]
    order = np.argsort(b, kind="stable")
    b, acc = b[order], np.abs(acc[order])
    edges = np.flatnonzero(np.r_[True, np.diff(b) != 0, True])
    prof = np.full((length, acc.shape[1]), np.nan)
    for s, e in zip(edges[:-1], edges[1:]):
        if e - s >= 16:
            seg = acc[s:e]
            prof[b[s]] = np.log(seg.max(0) / (np.sqrt((seg ** 2).mean(0)) + 1e-12))
    med = np.nanmedian(prof, 0)
    mad = np.nanmedian(np.abs(prof - med), 0) * 1.4826 + 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean((prof - med) / mad, 1)


def best_shift(p, ref, max_shift):
    """Shift (m) to add to p's positions that best matches ref, and the correlation there."""
    best = (0, -np.inf)
    for s in range(-max_shift, max_shift + 1):
        a = np.roll(p, s)
        ok = ~np.isnan(a) & ~np.isnan(ref)
        if s > 0:
            ok[:s] = False
        elif s < 0:
            ok[s:] = False
        if ok.sum() > 200:
            c = np.corrcoef(a[ok], ref[ok])[0, 1]
            if c > best[1]:
                best = (s, c)
    return best


warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "All-NaN slice")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/vivid_diag"))
    ap.add_argument("--anomalies", type=Path, default=Path(__file__).with_name("vivid_anomalies.csv"))
    ap.add_argument("--hp", type=float, default=50.0, help="high-pass (Hz) before the impact score")
    ap.add_argument("--max-shift", type=int, default=40, help="largest alignment shift tried (m)")
    ap.add_argument("--trim-m", type=float, default=20.0)
    ap.add_argument("--tol", type=float, default=15.0)
    args = ap.parse_args()

    runs = find_runs(args.data)
    ref = pd.read_csv(runs[0], usecols=["Latitude", "Longitude"]).dropna()
    track = Track(ref["Latitude"].to_numpy(), ref["Longitude"].to_numpy())
    length = int(np.ceil(track.s[-1])) + 1
    t = pd.read_csv(args.anomalies)
    truth = track.distance(t["lat"].to_numpy(), t["lon"].to_numpy())
    print(f"track {length} m; known anomalies at " + ", ".join(f"{m:.0f}" for m in truth))

    names, profs, infos = [], [], {}
    print("\nrun            fs(Hz)  dur(s)  gps_every(s)  fixes  gaps  speed(m/s)")
    for p in runs:
        acc, dist, info = load_run(p, track, args.hp)
        prof = impact_profile(acc, dist, length)
        lo, hi = np.flatnonzero(~np.isnan(prof))[[0, -1]]
        prof[:int(lo + args.trim_m)] = np.nan
        prof[int(hi - args.trim_m) + 1:] = np.nan
        names.append(p.parent.name)
        profs.append(prof)
        infos[p.parent.name] = info
        print(f"{p.parent.name:14s} {info['fs_hz']:7.1f} {info['duration_s']:7.1f} {info['gps_update_s']:12.3f} "
              f"{info['gps_fixes']:6d} {info['time_gaps_over_10ms']:5d} {info['speed_m_s']:10.2f}")
    P = np.array(profs)

    def shifted(M, shift):
        out = np.full_like(M, np.nan)
        if shift >= 0:
            out[..., shift:] = M[..., :M.shape[-1] - shift]
        else:
            out[..., :shift] = M[..., -shift:]
        return out

    def smooth(p):  # emphasise the impacts and tolerate a metre or two of jitter
        q = np.clip(np.nan_to_num(p, nan=0.0), 0, None)
        q = np.convolve(q, np.ones(3) / 3, "same")
        q[np.isnan(p)] = np.nan
        return q

    fwd = np.array([infos[n]["direction"] == "AtoB" for n in names])
    if fwd.any() and (~fwd).any():
        d, r = best_shift(smooth(np.nanmean(P[~fwd], 0)), smooth(np.nanmean(P[fwd], 0)), args.max_shift)
    else:
        d, r = 0, float("nan")
    a_shift, b_shift = -(d // 2), d - d // 2  # move A->B back and B->A forward by half each
    print(f"\nB->A runs sit {-d:+d} m from A->B runs (match r={r:.2f}); "
          f"moving A->B by {a_shift:+d} m and B->A by {b_shift:+d} m")
    A = np.array([shifted(p, a_shift if f else b_shift) for p, f in zip(P, fwd)])
    run_shift = []
    print("per-run residual shift against the other aligned runs (information only):")
    for i, n in enumerate(names):
        s_i, c_i = best_shift(smooth(A[i]), smooth(np.nanmean(np.delete(A, i, 0), 0)), 15)
        run_shift.append(int(s_i))
        print(f"  {n:14s} {infos[n]['direction']}  {s_i:+3d} m  r={c_i:.2f}")

    def visibility(M, label):
        mean = np.nanmean(M, 0)
        bg = np.nanpercentile(mean, [50, 95, 99])
        print(f"\n{label}: run-averaged impact score, ordinary track median {bg[0]:.2f}, "
              f"95th pct {bg[1]:.2f}, 99th pct {bg[2]:.2f}")
        for m in truth:
            w = slice(max(0, int(m - args.tol)), int(m + args.tol) + 1)
            seg = mean[w]
            if np.all(np.isnan(seg)):
                continue
            k = int(np.nanargmax(seg))
            print(f"  anomaly at {m:6.0f} m: peak {seg[k]:5.2f} at {w.start + k - m:+3.0f} m")
        return mean

    visibility(P, "before alignment")
    mean = visibility(A, "after alignment")
    mean_gps = mean
    ok = ~np.isnan(mean_gps)
    thr = np.nanpercentile(mean_gps, 99)
    peaks, _ = find_peaks(np.where(ok, mean_gps, -np.inf), height=thr, distance=int(args.tol))
    found = sum(any(abs(p - m) <= args.tol for p in peaks) for m in truth)
    false = [int(p) for p in peaks if all(abs(p - m) > args.tol for m in truth)]
    print(f"\nsimple detector (peaks above the 99th pct of the aligned average): "
          f"{len(peaks)} sites, found {found}/{len(truth)}, {len(false)} false: {false}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnosis.json").write_text(json.dumps(
        {"runs": infos, "direction_offset_m": int(-d), "direction_match_r": round(float(r), 3),
         "residual_shift_m": dict(zip(names, run_shift)),
         "simple_detector": {"sites_m": peaks.tolist(), "found": int(found), "false": false}}, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 3, 1.4]})
        for ax, M, title in ((axes[0], P, "before alignment"), (axes[1], A, "after aligning the two directions")):
            ax.imshow(np.clip(M, -1, 4), aspect="auto", cmap="magma", interpolation="nearest",
                      extent=[0, length, len(M) - 0.5, -0.5])
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=6)
            ax.set_title(f"impact score per metre, {title}", fontsize=9)
        axes[2].plot(np.arange(length), mean_gps, lw=0.8, color="#1e5a78")
        axes[2].axhline(thr, ls="--", lw=0.8, color="#888")
        axes[2].set_xlabel("distance from Point A end (m)")
        for ax in axes:
            for m in truth:
                ax.axvline(m, color="#2bb3c0", lw=0.7, alpha=0.8)
        fig.tight_layout()
        fig.savefig(args.out / "impact_heatmap.png", dpi=110)
        print(f"plot: {args.out / 'impact_heatmap.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
