#!/usr/bin/env python3
"""Cut the UPM corrugation runs into labelled 1 s WAV clips.

The .mat files hold no labels. Each sample has a kilometric position (So_pks for
sound, Vi_pks for vibration), and the dataset's paper (Soto-Ocampo et al.,
Mathematics 2025, 13(17), 2815) reports two sections with severe corrugation,
confirmed by visual inspection:

    Case A  km 4.855 - 4.914
    Case B  km 5.239 - 5.326

Each run is first resampled onto a uniform distance grid, as if the train had
always run at 72 km/h, so the corrugation frequency is the same in every run (see
to_distance_grid). Clips are 20 m long (1 s at that reference speed).

A clip is labelled `corrugated` when at least 75% of it lies inside one of these
sections and `normal` when it lies entirely outside them and at least --buffer
metres away (the track next to a corrugated patch may be partly worn too); clips in
between are dropped. The rest of the 3.798-6.246 km stretch wasn't reported as
corrugated, which is not the same as inspected and clean, so treat `normal` as
"not reported corrugated".

Two signals are exported as separate sources, so the models never mix sensors:
    mic   in-cabin microphone, 48 kHz -> 16 kHz at the reference speed
    axle  vertical axlebox acceleration, 4 kHz, both axles as 2 channels

Output: DATA/corrugation/segments/<mic|axle>/<label>/<run>_<km>.wav
build_manifest.py picks these up as sources corrugation_mic and corrugation_axle.

The corrugated track is only ~146 m. Corrugated clips overlap (2 m hop) to give the
model more views of it, but they still come from two short stretches of rail.

    python prepare_corrugation.py --data /content/drive/MyDrive/rail-v_dataset
"""

import argparse
import shutil
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


def label_clip(km_lo, km_hi, buffer_km, min_inside):
    length = km_hi - km_lo
    inside = sum(max(0.0, min(km_hi, b) - max(km_lo, a)) for a, b in SECTIONS_KM)
    if length > 0 and inside / length >= min_inside:
        return "corrugated"
    near = any(km_hi > a - buffer_km and km_lo < b + buffer_km for a, b in SECTIONS_KM)
    return None if near else "normal"


def to_distance_grid(y, fs, km, fs_out, ref_speed):
    """Resample a run onto a uniform distance grid, as if the train had run at a
    constant ref_speed (m/s) and been recorded at fs_out.

    Corrugation is a fixed wavelength (~28.5 mm here), so in time its frequency is
    speed / wavelength: ~390 Hz at 40 km/h but ~730 Hz at 75 km/h. On a distance
    grid it sits at the same frequency in every run, which is what lets a model
    trained at 40-50 km/h recognise it at 75 km/h. Returns (signal, km per sample)."""
    d = km * 1000.0
    if d[-1] < d[0]:
        y, d = y[::-1], d[::-1]
    d = np.maximum.accumulate(d)
    rate = fs_out / ref_speed  # samples per metre
    v_mean = (d[-1] - d[0]) / (len(d) / fs)
    # Anti-aliased time-domain resample to roughly the target rate first (rounded to
    # 100 Hz to keep the polyphase factors small), then interpolate onto the grid.
    fs_mid = max(100, int(round(rate * v_mean / 100.0)) * 100)
    y_mid = np.stack([resample(y[:, c], fs, fs_mid) for c in range(y.shape[1])], axis=1)
    d_mid = np.interp(np.arange(len(y_mid)) * (fs / fs_mid), np.arange(len(d)), d)
    grid = np.arange(d[0], d[-1], 1.0 / rate)
    y_u = np.stack([np.interp(grid, d_mid, y_mid[:, c]) for c in range(y.shape[1])], axis=1)
    return y_u.astype(np.float32), grid / 1000.0


def export(path, signal, out_root, fs_out, args):
    y, fs, km = load_run(path, signal)
    y, km = to_distance_grid(y, fs, km, fs_out, args.ref_speed)
    y = y / (np.abs(y).max() + 1e-9) * 0.9  # per-run scaling so PCM_16 doesn't clip
    rate = fs_out / args.ref_speed
    n = int(round(args.clip_m * rate))
    hop_pos, hop_neg = int(round(args.pos_hop_m * rate)), int(round(args.clip_m * rate))
    counts = {"corrugated": 0, "normal": 0, "dropped": 0}
    for s in range(0, len(y) - n + 1, hop_pos):
        seg_km = km[s:s + n]
        label = label_clip(float(seg_km[0]), float(seg_km[-1]), args.buffer / 1000, args.min_inside)
        # Corrugated clips overlap (hop pos_hop_m) to get more training views of the
        # two short sections; normal clips don't (hop clip_m).
        if label == "normal" and s % hop_neg:
            continue
        if label is None:
            counts["dropped"] += 1
            continue
        d = out_root / signal / label
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / f"{path.stem}_{float(np.median(seg_km)):.4f}.wav", y[s:s + n], int(fs_out),
                 subtype="PCM_16")
        counts[label] += 1
    print(f"  {path.name} [{signal}] {rate:.0f} samples/m, km {km.min():.3f}-{km.max():.3f}: {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder download.py wrote into")
    ap.add_argument("--ref-speed", type=float, default=20.0,
                    help="m/s every run is normalised to (default 20 = 72 km/h)")
    ap.add_argument("--clip-m", type=float, default=20.0,
                    help="clip length in metres (20 m = 1 s at the reference speed)")
    ap.add_argument("--pos-hop-m", type=float, default=2.0, help="hop between corrugated clips, metres")
    ap.add_argument("--min-inside", type=float, default=0.75,
                    help="fraction of a clip inside a section to call it corrugated")
    ap.add_argument("--buffer", type=float, default=20.0, help="metres around a section to leave out")
    ap.add_argument("--signals", default="mic,axle")
    args = ap.parse_args()

    src = args.data / "corrugation"
    runs = sorted(src.glob("Run_*.mat"))
    if not runs:
        raise SystemExit(f"no Run_*.mat files in {src}")
    out = src / "segments"
    for signal in args.signals.split(","):
        if (out / signal).exists():  # clips from an earlier run of this script
            shutil.rmtree(out / signal)
        for p in runs:
            export(p, signal, out, 16000 if signal == "mic" else 4000, args)
    print(f"clips written under {out}")


if __name__ == "__main__":
    main()
