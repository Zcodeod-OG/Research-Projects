#!/usr/bin/env python3
"""Cut the Rail-VIVID runs into labelled clips: joint / switch stretch vs plain track.

The runs have no labels, but the detector work showed that each of the paper's 9
anomalies (Table 2: joints, a divergence and a convergence point) lies inside a
~25 m stretch of track whose vibration differs from the rest in every run. So:

- `joint`  clip centre within --pos-m metres of a known anomaly (overlapping clips,
           --pos-hop-m apart, for more views of the 9 short stretches)
- `normal` clip at least --buffer metres from every known anomaly and from every
           stretch the detector found that holds no listed anomaly (those may be
           unlisted joints, so they are left out rather than guessed)

Each run is resampled onto a distance grid, as if driven at --ref-speed, so a clip
is the same length of track in every run whatever the speed. Channels 1 and 3 (the
ones whose vibration repeats between runs) are written as separate mono clips in the
same group, so the model never averages two sensors together. Stretches where a
channel is stuck at a constant value are skipped.

Output: DATA/rail_vivid_vibration/clips/<label>/<run>_<pos>m_ch<k>.wav (32-bit float,
scaled by each run's typical level so louder stretches stay louder)
build_manifest.py picks these up as source vivid_joint, split by run.

    python prepare_vivid.py --data /content/drive/MyDrive/rail-v_dataset
    python prepare_vivid.py --data ... --stretches $RUNS/vivid_detect/stretches.csv
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from anomaly_vivid import CHANNELS, Track, find_runs
from common import resample

# Stretches detect_vivid.py found on the real runs that hold no listed anomaly (metres
# from the Point A end, reference track from the first run). Used when --stretches
# isn't given.
UNLISTED_STRETCHES_M = [(255, 279), (372, 397), (611, 643), (882, 906), (1054, 1079)]


def load_run(path, track):
    df = pd.read_csv(path, usecols=["Timestamp"] + CHANNELS + ["Latitude", "Longitude"]).dropna()
    t = pd.to_datetime(df["Timestamp"]).astype("int64").to_numpy() / 1e9
    fs = float(round(1.0 / np.median(np.diff(t))))
    dist = track.distance(df["Latitude"].to_numpy(), df["Longitude"].to_numpy())
    return df[CHANNELS].to_numpy(np.float32), fs, dist


def to_distance_grid(acc, fs, dist, fs_out, ref_speed):
    """Uniform distance grid at fs_out / ref_speed samples per metre. Returns (signal, metres)."""
    if dist[-1] < dist[0]:
        acc, dist = acc[::-1], dist[::-1]
    dist = np.maximum.accumulate(dist)
    rate = fs_out / ref_speed
    v = (dist[-1] - dist[0]) / (len(dist) / fs)
    fs_mid = max(100, int(round(rate * v / 100.0)) * 100)  # anti-aliased resample first
    mid = np.stack([resample(acc[:, c], fs, fs_mid) for c in range(acc.shape[1])], 1)
    d_mid = np.interp(np.arange(len(mid)) * (fs / fs_mid), np.arange(len(dist)), dist)
    grid = np.arange(dist[0], dist[-1], 1.0 / rate)
    return np.stack([np.interp(grid, d_mid, mid[:, c]) for c in range(mid.shape[1])], 1).astype(np.float32), grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--anomalies", type=Path, default=Path(__file__).with_name("vivid_anomalies.csv"))
    ap.add_argument("--stretches", type=Path, help="stretches.csv from detect_vivid.py (else built-in list)")
    ap.add_argument("--channels", default="1,3", help="which of Channel1-6 to export, as separate clips")
    ap.add_argument("--ref-speed", type=float, default=6.5, help="m/s every run is normalised to")
    ap.add_argument("--fs-out", type=int, default=2000)
    ap.add_argument("--clip-m", type=float, default=20.0)
    ap.add_argument("--pos-m", type=float, default=6.0, help="clip centre this close to an anomaly = joint")
    ap.add_argument("--pos-hop-m", type=float, default=2.0)
    ap.add_argument("--buffer", type=float, default=25.0, help="metres kept clear around stretches")
    ap.add_argument("--trim-m", type=float, default=20.0)
    args = ap.parse_args()

    runs = find_runs(args.data)
    ref = pd.read_csv(runs[0], usecols=["Latitude", "Longitude"]).dropna()
    track = Track(ref["Latitude"].to_numpy(), ref["Longitude"].to_numpy())
    t = pd.read_csv(args.anomalies)
    truth = track.distance(t["lat"].to_numpy(), t["lon"].to_numpy())
    if args.stretches:
        s = pd.read_csv(args.stretches)
        unlisted = [(a, b) for a, b, k in zip(s["start_m"], s["end_m"], s.get("known_anomaly_m", [""] * len(s)))
                    if not isinstance(k, str) or not k.strip()]
    else:
        unlisted = UNLISTED_STRETCHES_M
    print("known anomalies at " + ", ".join(f"{m:.0f}" for m in truth) + " m; excluded stretches "
          + ", ".join(f"{a:.0f}-{b:.0f}" for a, b in unlisted))

    chans = [int(c) - 1 for c in args.channels.split(",")]
    out = args.data / "rail_vivid_vibration" / "clips"
    if out.exists():
        shutil.rmtree(out)
    rate = args.fs_out / args.ref_speed
    n = int(round(args.clip_m * rate))
    total = {"joint": 0, "normal": 0}
    for p in runs:
        acc, fs, dist = load_run(p, track)
        y, grid = to_distance_grid(acc[:, chans], fs, dist, args.fs_out, args.ref_speed)
        lo, hi = grid[0] + args.trim_m, grid[-1] - args.trim_m
        spread = np.array([y[s:s + n].std(0) for s in range(0, len(y) - n + 1, n)])
        scale = np.median(spread, 0) + 1e-12  # per-run, per-channel level, kept across clips
        alive_level = 0.02 * scale
        counts = {"joint": 0, "normal": 0, "dead": 0}
        hop_pos = int(round(args.pos_hop_m * rate))
        next_normal = 0  # normal clips don't overlap
        for s in range(0, len(y) - n + 1, hop_pos):
            a, b = grid[s], grid[s + n - 1]
            c = (a + b) / 2
            if a < lo or b > hi:
                continue
            if np.min(np.abs(truth - c)) <= args.pos_m:
                label = "joint"
            elif s >= next_normal and all(b < m - args.buffer or a > m + args.buffer for m in truth) and \
                    all(b < u0 - args.buffer or a > u1 + args.buffer for u0, u1 in unlisted):
                label = "normal"
            else:
                continue
            if label == "normal":
                next_normal = s + n
            seg = y[s:s + n]
            for j, ch in enumerate(chans):
                if seg[:, j].std() < alive_level[j]:
                    counts["dead"] += 1
                    continue
                x = seg[:, j] - seg[:, j].mean()
                d = out / label
                d.mkdir(parents=True, exist_ok=True)
                # Scaled by the run's typical level, not per clip, so a louder stretch stays louder.
                sf.write(d / f"{p.parent.name}_{c:07.1f}m_ch{ch + 1}.wav", 0.05 * x / scale[j],
                         args.fs_out, subtype="FLOAT")
                counts[label] += 1
        total["joint"] += counts["joint"]
        total["normal"] += counts["normal"]
        print(f"  {p.parent.name}: {counts}")
    print(f"total {total}; clips under {out}")


if __name__ == "__main__":
    main()
