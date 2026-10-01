#!/usr/bin/env python3
"""Rail-VIVID track-feature detector (v2): find where the track's vibration changes.

The channel sweep (sweep_vivid.py) showed that the small joints don't show up as
impacts on these sensors. What does repeat in every run are stretches of track,
typically 15-30 m long, with distinctly higher 500-990 Hz vibration on Channels 1 and 3
and a shifted slow signal on Channel 2. On the real runs, each of the 9 known anomalies
lies inside one of these stretches. So this detector finds the stretches and their ends:

1. Per 1 m of track, per-run z-scored features (as in sweep_vivid.py), averaged over runs.
2. Keep only features that repeat between runs (r >= --min-r). This uses no labels.
3. Step strength at each metre = how different the next --win metres are from the
   previous --win metres (medians), in units of that feature's noise, averaged over
   the kept features.
4. Candidate sites = step peaks above --min-step, at least --min-sep metres apart.
5. Stretches = where the combined feature (each kept feature signed to agree with the
   most repeatable one, then averaged) stays above --level, gaps up to --gap-m closed,
   at least --min-len metres long. Each stretch is one site.

With the known anomalies (vivid_anomalies.csv by default) it reports how many lie in or
within --near-m of a stretch, and how many stretches hold none. Those are not
necessarily wrong: they are real, repeatable features of the track that just aren't in
Table 2. The step sites are also scored within --tol metres, for comparison.

    python detect_vivid.py --data /content/drive/MyDrive/rail-v_dataset --out $RUNS/vivid_detect
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from anomaly_vivid import CHANNELS, K_LAT, K_LON, POINT_A, Track, evaluate, find_runs
from sweep_vivid import FEATURES, load, per_metre, repeatability


def step_strength(m, win):
    """|median of the next win metres - median of the previous win metres| in noise units."""
    d = np.diff(m)
    noise = np.nanmedian(np.abs(d - np.nanmedian(d))) * 1.4826 / np.sqrt(2) + 1e-9
    out = np.full(len(m), np.nan)
    for x in range(win, len(m) - win):
        a, b = m[x - win:x], m[x:x + win]
        if np.sum(~np.isnan(a)) >= win // 2 and np.sum(~np.isnan(b)) >= win // 2:
            out[x] = abs(np.nanmedian(b) - np.nanmedian(a)) / noise
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/vivid_detect"))
    ap.add_argument("--anomalies", type=Path, default=Path(__file__).with_name("vivid_anomalies.csv"),
                    help="known positions (lat,lon); pass an empty string to skip scoring")
    ap.add_argument("--min-r", type=float, default=0.3, help="keep features at least this repeatable")
    ap.add_argument("--win", type=int, default=6, help="metres compared on each side of a step")
    ap.add_argument("--min-step", type=float, default=4.0, help="step threshold, in noise units")
    ap.add_argument("--min-sep", type=int, default=10, help="metres between candidate sites")
    ap.add_argument("--level", type=float, default=0.6, help="combined z above which track is in a stretch")
    ap.add_argument("--gap-m", type=int, default=5, help="close dips in a stretch up to this long")
    ap.add_argument("--min-len", type=int, default=8, help="shortest stretch kept (m)")
    ap.add_argument("--near-m", type=float, default=5.0, help="an anomaly this close to a stretch counts")
    ap.add_argument("--trim-m", type=int, default=20)
    ap.add_argument("--tol", type=float, default=15.0)
    args = ap.parse_args()

    runs = find_runs(args.data)
    ref = pd.read_csv(runs[0], usecols=["Latitude", "Longitude"]).dropna()
    track = Track(ref["Latitude"].to_numpy(), ref["Longitude"].to_numpy())
    length = int(np.ceil(track.s[-1])) + 1

    feats = []
    for p in runs:
        acc, fs, dist = load(p, track)
        F = per_metre(acc, fs, dist, length)
        have = np.flatnonzero(np.any(~np.isnan(F), (1, 2)))
        F[:have[0] + args.trim_m] = np.nan
        F[have[-1] - args.trim_m + 1:] = np.nan
        feats.append(F)
        print(f"  {p.parent.name}: features done")
    X = np.array(feats)  # (runs, length, channels, features)

    kept = []
    for c in range(X.shape[2]):
        for k in range(X.shape[3]):
            r = repeatability(X[:, :, c, k])
            if r >= args.min_r:
                kept.append((c, k, r))
    if not kept:
        raise SystemExit(f"no feature repeats with r >= {args.min_r}; nothing tied to the track to detect")
    print("features used (repeat r): " + ", ".join(f"{CHANNELS[c]} {FEATURES[k]} ({r:.2f})" for c, k, r in kept))

    means = np.nanmean(X, 0)
    steps = np.array([step_strength(means[:, c, k], args.win) for c, k, _ in kept])
    strength = np.nanmean(steps, 0)
    peaks, _ = find_peaks(np.nan_to_num(strength, nan=0.0), height=args.min_step, distance=args.min_sep)

    def latlon(pos):
        xy = track.xy[np.searchsorted(track.s, pos).clip(0, len(track.s) - 1)]
        return POINT_A[0] + xy[1] / K_LAT, POINT_A[1] + xy[0] / K_LON

    c0, k0, _ = max(kept, key=lambda t: t[2])
    rows = []
    for p in peaks:
        lat, lon = latlon(p)
        before = np.nanmedian(means[max(0, p - args.win):p, c0, k0])
        after = np.nanmedian(means[p:p + args.win, c0, k0])
        rows.append({"pos_m": int(p), "lat": round(float(lat), 6), "lon": round(float(lon), 6),
                     "strength": round(float(strength[p]), 2), "change": "up" if after > before else "down"})
    cands = pd.DataFrame(rows, columns=["pos_m", "lat", "lon", "strength", "change"])
    args.out.mkdir(parents=True, exist_ok=True)
    cands.to_csv(args.out / "candidates.csv", index=False)
    print(f"\n{len(cands)} candidate sites (step > {args.min_step}):")
    for _, c in cands.iterrows():
        print(f"  {c['pos_m']:5d} m  ({c['lat']:.6f}, {c['lon']:.6f})  strength {c['strength']:5.1f}  "
              f"{CHANNELS[c0]} {FEATURES[k0]} {c['change']}")

    # Stretches from the combined, consistently signed feature.
    ref_mean = means[:, c0, k0]
    signs = []
    for c, k, _ in kept:
        ok = ~np.isnan(ref_mean) & ~np.isnan(means[:, c, k])
        signs.append(np.sign(np.corrcoef(ref_mean[ok], means[ok, c, k])[0, 1]) or 1.0)
    combined = np.nanmean([sg * means[:, c, k] for sg, (c, k, _) in zip(signs, kept)], 0)
    # Stretches are the minority of the track, so they pull the mean away from the median:
    # flip the sign so they always read high.
    if np.nanmean(combined) < np.nanmedian(combined):
        combined = -combined
    combined -= np.nanmedian(combined)
    inside = np.nan_to_num(combined, nan=-np.inf) > args.level
    stretches, x = [], 0
    while x < length:
        if inside[x]:
            y = x
            while y + 1 < length and (inside[y + 1] or inside[y + 1:y + 2 + args.gap_m].any()):
                y += 1
            if y - x + 1 >= args.min_len:
                stretches.append((x, y + 1))
            x = y + 1
        else:
            x += 1
    srows = []
    for a, b in stretches:
        lat, lon = latlon((a + b) / 2)
        srows.append({"start_m": a, "end_m": b, "length_m": b - a, "centre_lat": round(float(lat), 6),
                      "centre_lon": round(float(lon), 6), "level": round(float(np.nanmedian(combined[a:b])), 2)})
    sdf = pd.DataFrame(srows, columns=["start_m", "end_m", "length_m", "centre_lat", "centre_lon", "level"])

    result = {"candidates": len(cands), "stretches": len(sdf), "features": [f"{CHANNELS[c]} {FEATURES[k]}" for c, k, _ in kept],
              "config": {k: str(v) for k, v in vars(args).items()}}
    truth = None
    if str(args.anomalies) and args.anomalies.exists():
        t = pd.read_csv(args.anomalies)
        truth = track.distance(t["lat"].to_numpy(), t["lon"].to_numpy())
        e = evaluate(cands["pos_m"].tolist(), list(truth), args.tol)
        result["evaluation_steps"] = e
        near = lambda m, a, b: a - args.near_m <= m <= b + args.near_m  # noqa: E731
        sdf["known_anomaly_m"] = [", ".join(f"{m:.0f}" for m in truth if near(m, a, b))
                                  for a, b in zip(sdf["start_m"], sdf["end_m"])]
        hit = [any(near(m, a, b) for a, b in stretches) for m in truth]
        covered = sum(b - a for a, b in stretches) + 2 * args.near_m * len(stretches)
        span = float(np.sum(~np.isnan(combined)))
        result["evaluation_stretches"] = {
            "found": int(sum(hit)), "of": len(truth), "stretches": len(stretches),
            "stretches_without_known": int(sum(sdf["known_anomaly_m"] == "")),
            "missed_m": [round(float(m)) for m, h in zip(truth, hit) if not h],
            "chance_per_anomaly": round(covered / span, 3)}
        es = result["evaluation_stretches"]
        print("\nknown anomalies at " + ", ".join(f"{m:.0f}" for m in truth) + " m")
        print(f"step sites: found {e['found']}/{e['of']} within {args.tol:.0f} m; {e['false_alarms']} other sites")
        print(f"stretches:  {es['found']}/{es['of']} known anomalies lie in one; "
              f"{es['stretches_without_known']} of {es['stretches']} stretches hold none "
              f"(a random spot would land in one {es['chance_per_anomaly']:.0%} of the time)")
    sdf.to_csv(args.out / "stretches.csv", index=False)
    print(f"\n{len(sdf)} stretches:")
    for _, r in sdf.iterrows():
        tag = f"  known anomaly at {r['known_anomaly_m']} m" if r.get("known_anomaly_m") else ""
        print(f"  {r['start_m']:5d}-{r['end_m']:<5d} m ({r['length_m']:3d} m)  "
              f"centre ({r['centre_lat']:.6f}, {r['centre_lon']:.6f}){tag}")
    (args.out / "results.json").write_text(json.dumps(result, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
        for c, k, _ in kept:
            axes[0].plot(means[:, c, k], lw=0.8, label=f"{CHANNELS[c]} {FEATURES[k]}")
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].set_ylabel("run mean (z)", fontsize=8)
        axes[1].plot(strength, lw=0.8, color="#1e5a78")
        axes[1].axhline(args.min_step, ls="--", lw=0.8, color="#888")
        axes[1].plot(peaks, strength[peaks], "v", color="#e0a030", ms=5)
        for a, b in stretches:
            axes[0].axvspan(a, b, color="#e0a030", alpha=0.15, lw=0)
        axes[1].set_ylabel("step strength", fontsize=8)
        axes[1].set_xlabel("distance from Point A end (m)")
        if truth is not None:
            for ax in axes:
                for m in truth:
                    ax.axvline(m, color="#2bb3c0", lw=0.7, alpha=0.8)
        fig.tight_layout()
        fig.savefig(args.out / "detect.png", dpi=110)
        print(f"plot: {args.out / 'detect.png'}")
    except ImportError:
        pass
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
