#!/usr/bin/env python3
"""Cut the UPM corrugation runs into labelled 1 s WAV clips.

The .mat files hold no labels. Each sample has a kilometric position (So_pks for
sound, Vi_pks for vibration), and the dataset's paper (Soto-Ocampo et al.,
Mathematics 2025, 13(17), 2815) reports two sections with severe corrugation,
confirmed by visual inspection:

    Case A  km 4.855 - 4.914
    Case B  km 5.239 - 5.326

A clip is labelled `corrugated` when at least half of it lies inside one of these
sections and `normal` when it lies entirely outside them and at least --buffer
metres away (the track next to a corrugated patch may be partly worn too); clips in
between are dropped. The rest of the 3.798-6.246 km stretch wasn't reported as
corrugated, which is not the same as inspected and clean, so treat `normal` as
"not reported corrugated".

Two signals are exported as separate sources, so the models never mix sensors:
    mic   in-cabin microphone, 48 kHz, resampled to 16 kHz
    axle  vertical axlebox acceleration, 4 kHz, both axles as 2 channels

Output: DATA/corrugation/segments/<mic|axle>/<label>/<run>_<km>.wav
build_manifest.py picks these up as sources corrugation_mic and corrugation_axle.

The corrugated track is only ~146 m, so each run yields ~7-13 s of corrugated
signal: a few dozen clips across all four runs. It's real, but small.

    python prepare_corrugation.py --data /content/drive/MyDrive/rail-v_dataset
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from common import resample

SECTIONS_KM = [(4.855, 4.914), (5.239, 5.326)]


def load_run(path, signal):
    from scipy.io import loadmat
    if signal == "mic":
        m = loadmat(path, variable_names=["So_y", "So_t", "So_pks"])
        y, t, km = m["So_y"], m["So_t"].ravel(), m["So_pks"].ravel()
    else:
        m = loadmat(path, variable_names=["Vi_y", "Vi_t", "Vi_pks"])
        y, t, km = m["Vi_y"], m["Vi_t"].ravel(), m["Vi_pks"].ravel()
    fs = 1.0 / float(np.median(np.abs(np.diff(t[:100000]))))
    return y.astype(np.float32).reshape(len(t), -1), float(round(fs)), km


def label_clip(km_lo, km_hi, buffer_km):
    length = km_hi - km_lo
    inside = sum(max(0.0, min(km_hi, b) - max(km_lo, a)) for a, b in SECTIONS_KM)
    if length > 0 and inside / length >= 0.5:
        return "corrugated"
    near = any(km_hi > a - buffer_km and km_lo < b + buffer_km for a, b in SECTIONS_KM)
    return None if near else "normal"


def export(path, signal, out_root, clip_s, buffer_m, target_fs):
    y, fs, km = load_run(path, signal)
    if target_fs and fs > target_fs:
        y = np.stack([resample(y[:, c], fs, target_fs) for c in range(y.shape[1])], axis=1)
        km = np.interp(np.linspace(0, len(km) - 1, len(y)), np.arange(len(km)), km)
        fs = target_fs
    y = y / (np.abs(y).max() + 1e-9) * 0.9  # per-run scaling so PCM_16 doesn't clip
    n = int(clip_s * fs)
    counts = {"corrugated": 0, "normal": 0, "dropped": 0}
    for s in range(0, len(y) - n + 1, n):
        seg_km = km[s:s + n]
        label = label_clip(float(seg_km.min()), float(seg_km.max()), buffer_m / 1000)
        if label is None:
            counts["dropped"] += 1
            continue
        d = out_root / signal / label
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{path.stem}_{float(np.median(seg_km)):.4f}.wav", y[s:s + n], int(fs), subtype="PCM_16")
        counts[label] += 1
    print(f"  {path.name} [{signal}] fs {fs:.0f} Hz, km {km.min():.3f}-{km.max():.3f}: {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder download.py wrote into")
    ap.add_argument("--clip", type=float, default=1.0, help="clip length in seconds")
    ap.add_argument("--buffer", type=float, default=20.0, help="metres around a section to leave out")
    ap.add_argument("--signals", default="mic,axle")
    args = ap.parse_args()

    src = args.data / "corrugation"
    runs = sorted(src.glob("Run_*.mat"))
    if not runs:
        raise SystemExit(f"no Run_*.mat files in {src}")
    out = src / "segments"
    for signal in args.signals.split(","):
        for p in runs:
            export(p, signal, out, args.clip, args.buffer, 16000 if signal == "mic" else None)
    print(f"clips written under {out}")


if __name__ == "__main__":
    main()
