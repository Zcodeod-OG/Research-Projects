#!/usr/bin/env python3
"""Unsupervised track-anomaly detector on the Rail-VIVID accelerometer runs.

Rail-VIVID has 20 runs over the same 1.4 km, in both directions and at four speeds,
and 9 ground-truth anomalies (rail joints, divergence / convergence points). The CSVs
carry no labels, so this learns what ordinary track looks like and flags what isn't:

1. Each run is cut into 1 m pieces of track by GPS distance along the A->B line (not
   by time, so the four speeds line up), and each piece gets per-channel vibration
   features: level, kurtosis, crest factor and band energies.
2. Leave-one-run-out: an Isolation Forest fits the other 19 runs and scores the held-out
   run. Anomalies are a tiny share of the track, so "mostly healthy" training is fine.
3. Scores are pooled in 5 m bins across all runs. A real anomaly sits at the same place
   in every pass, so the evidence is how many independent runs flag a bin, not one
   high score. Candidate sites are the bins most runs agree on.

With --anomalies (a CSV of the paper's Table 2 positions: columns lat,lon or
distance_m), it also reports how many of the 9 were found within --tol metres and
how many candidates were false alarms.

    python anomaly_vivid.py --data /content/data --out /content/drive/MyDrive/rail-v-runs/vivid_anomaly
    python anomaly_vivid.py ... --anomalies vivid_anomalies.csv

Needs numpy, pandas, scikit-learn; matplotlib for the plot.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FS = 2000
CHANNELS = [f"Channel{i}" for i in range(1, 7)]
# Survey points from the Rail-VIVID README.
POINT_A = (40.23951169072744, -77.89729802853955)
POINT_B = (40.23184123709821, -77.87387005111827)
BANDS_HZ = [(1, 20), (20, 60), (60, 150), (150, 300), (300, 500)]


def along_track(lat, lon):
    """Metres from Point A, projected onto the straight A->B line (local flat-earth)."""
    k_lat = 110_540.0
    k_lon = 111_320.0 * math.cos(math.radians(POINT_A[0]))
    bx, by = (POINT_B[1] - POINT_A[1]) * k_lon, (POINT_B[0] - POINT_A[0]) * k_lat
    x, y = (lon - POINT_A[1]) * k_lon, (lat - POINT_A[0]) * k_lat
    return (x * bx + y * by) / math.hypot(bx, by)


def segment_features(seg):
    """seg: (n, 6) accelerations for one 1 m piece of track."""
    feats = []
    spec = np.abs(np.fft.rfft(seg - seg.mean(0), axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / FS)
    for c in range(seg.shape[1]):
        x = seg[:, c] - seg[:, c].mean()
        rms = math.sqrt(float(np.mean(x ** 2))) + 1e-9
        kurt = float(np.mean(x ** 4)) / rms ** 4
        crest = float(np.max(np.abs(x))) / rms
        bands = [np.log(spec[(freqs >= lo) & (freqs < hi), c].sum() + 1e-12) for lo, hi in BANDS_HZ]
        feats += [math.log(rms), math.log(kurt), math.log(crest)] + bands
    return feats


def run_features(csv_path, seg_m, trim_m):
    df = pd.read_csv(csv_path, usecols=CHANNELS + ["Latitude", "Longitude"])
    df = df.dropna()
    acc = df[CHANNELS].to_numpy(np.float32)
    dist = along_track(df["Latitude"].to_numpy(), df["Longitude"].to_numpy())
    # GPS updates far slower than 2 kHz; smooth the distance so pieces aren't empty or doubled.
    win = 2 * FS
    dist = np.convolve(np.pad(dist, (win // 2, win - win // 2 - 1), mode="edge"),
                       np.ones(win) / win, mode="valid")
    bins = np.floor(dist / seg_m).astype(int)
    rows, pos, lat, lon = [], [], [], []
    lats, lons = df["Latitude"].to_numpy(), df["Longitude"].to_numpy()
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        if len(idx) < 64:  # too few samples (e.g. GPS jump or standing still at the ends)
            continue
        rows.append(segment_features(acc[idx]))
        pos.append((b + 0.5) * seg_m)
        lat.append(lats[idx].mean())
        lon.append(lons[idx].mean())
    X, pos, lat, lon = np.asarray(rows, np.float32), np.asarray(pos), np.asarray(lat), np.asarray(lon)
    # Drop the ends of each run: starting, stopping and the distance smoothing make them
    # look unusual in every run, which would otherwise show up as two false sites.
    keep = (pos > pos.min() + trim_m) & (pos < pos.max() - trim_m)
    X, pos, lat, lon = X[keep], pos[keep], lat[keep], lon[keep]
    # Per-run robust scaling removes run-level differences (speed, temperature, mounting).
    med = np.median(X, 0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0)) + 1e-6
    return (X - med) / iqr, pos, lat, lon


def find_runs(data):
    runs = sorted((data / "rail_vivid_vibration").glob("*/*.csv"))
    if not runs:
        raise SystemExit(f"no run CSVs under {data / 'rail_vivid_vibration'}")
    return runs


def evaluate(cands, truth, tol):
    hits = [any(abs(c - t) <= tol for c in cands) for t in truth]
    false = [c for c in cands if all(abs(c - t) > tol for t in truth)]
    return {"found": int(sum(hits)), "of": len(truth), "recall": round(sum(hits) / len(truth), 3),
            "false_alarms": len(false), "false_alarm_positions_m": [round(float(f), 1) for f in false]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder holding rail_vivid_vibration/")
    ap.add_argument("--out", type=Path, default=Path("runs/vivid_anomaly"))
    ap.add_argument("--seg-m", type=float, default=1.0, help="metres per feature piece")
    ap.add_argument("--trim-m", type=float, default=20.0, help="metres dropped at each end of a run")
    ap.add_argument("--bin-m", type=float, default=5.0, help="metres per pooled bin")
    ap.add_argument("--flag-pct", type=float, default=98.0,
                    help="a piece is 'flagged' when its score is above this percentile of its run")
    ap.add_argument("--min-runs", type=float, default=0.5, help="share of runs that must flag a bin")
    ap.add_argument("--anomalies", type=Path, help="CSV of true positions: lat,lon or distance_m")
    ap.add_argument("--tol", type=float, default=15.0, help="match tolerance in metres (GPS drift)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = find_runs(args.data)
    print(f"{len(runs)} runs; extracting 1 m features ...")
    feats = []
    for p in runs:
        X, pos, lat, lon = run_features(p, args.seg_m, args.trim_m)
        feats.append((p.parent.name, X, pos, lat, lon))
        print(f"  {p.parent.name}: {len(X)} pieces, {pos.min():.0f}-{pos.max():.0f} m")

    print("scoring each run with a model fit on the other runs ...")
    rows = []
    for i, (name, X, pos, lat, lon) in enumerate(feats):
        train = np.concatenate([f[1] for j, f in enumerate(feats) if j != i])
        model = IsolationForest(n_estimators=300, random_state=args.seed).fit(train)
        score = -model.score_samples(X)  # higher = more unusual
        flagged = score >= np.percentile(score, args.flag_pct)
        rows += [{"run": name, "pos_m": p, "lat": a, "lon": o, "score": s, "flagged": f}
                 for p, a, o, s, f in zip(pos, lat, lon, score, flagged)]
    df = pd.DataFrame(rows)
    df["bin"] = np.floor(df["pos_m"] / args.bin_m).astype(int)
    n_runs = df["run"].nunique()

    per_bin = (df.groupby("bin")
                 .agg(pos_m=("pos_m", "mean"), lat=("lat", "mean"), lon=("lon", "mean"),
                      median_score=("score", "median"),
                      runs_flagged=("run", lambda r: r[df.loc[r.index, "flagged"]].nunique()))
                 .reset_index())
    per_bin["share_flagged"] = per_bin["runs_flagged"] / n_runs

    # Candidates: bins most runs flag, merged so one site within --tol counts once.
    cand_bins = per_bin[per_bin["share_flagged"] >= args.min_runs].sort_values("share_flagged", ascending=False)
    sites = []
    for _, b in cand_bins.iterrows():
        if all(abs(b["pos_m"] - s["pos_m"]) > args.tol for s in sites):
            sites.append(b.to_dict())
    cands = pd.DataFrame(sites).sort_values("pos_m") if sites else pd.DataFrame(columns=per_bin.columns)

    args.out.mkdir(parents=True, exist_ok=True)
    per_bin.to_csv(args.out / "per_bin.csv", index=False)
    cands.to_csv(args.out / "candidates.csv", index=False)
    print(f"\n{len(cands)} candidate sites flagged by at least {args.min_runs:.0%} of {n_runs} runs:")
    for _, c in cands.iterrows():
        print(f"  {c['pos_m']:7.1f} m from A  ({c['lat']:.6f}, {c['lon']:.6f})  "
              f"flagged in {int(c['runs_flagged'])}/{n_runs} runs")

    result = {"runs": n_runs, "candidates": len(cands), "config": {k: str(v) for k, v in vars(args).items()}}
    truth_m = None
    if args.anomalies:
        t = pd.read_csv(args.anomalies)
        truth_m = (t["distance_m"].to_numpy() if "distance_m" in t
                   else along_track(t["lat"].to_numpy(), t["lon"].to_numpy()))
        result["evaluation"] = evaluate(cands["pos_m"].tolist(), list(truth_m), args.tol)
        e = result["evaluation"]
        print(f"\nfound {e['found']}/{e['of']} known anomalies within {args.tol:.0f} m; "
              f"{e['false_alarms']} false-alarm sites")
    (args.out / "results.json").write_text(json.dumps(result, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(per_bin["pos_m"], per_bin["share_flagged"], lw=1, color="#1e5a78")
        ax.axhline(args.min_runs, ls="--", lw=0.8, color="#888")
        if truth_m is not None:
            for m in truth_m:
                ax.axvline(m, color="#a23b2a", lw=0.8, alpha=0.7)
        ax.set_xlabel("distance from Point A (m)")
        ax.set_ylabel("share of runs flagging")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(args.out / "anomaly_profile.png", dpi=120)
        print(f"plot: {args.out / 'anomaly_profile.png'}")
    except ImportError:
        pass
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
