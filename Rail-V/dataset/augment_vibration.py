#!/usr/bin/env python3
"""Turn a small set of vibration / acoustic recordings into many training windows.

Two steps, both cheap and label-preserving:
1. Windowing: cut each recording into short overlapping windows (e.g. a 17 s
   clip at 1 s windows / 0.5 s hop gives 33 samples instead of 1).
2. Augmentation: make N perturbed copies of each window with
   - speed perturbation: resample by a factor, which mimics the same defect
     passed at a different train speed (defect frequency = speed / wavelength),
   - gain: random amplitude scaling (sensor mounting, coupling),
   - noise: additive Gaussian noise at a random SNR,
   - time shift: circular shift so the event is not always centred.
   With --background, also mix fault windows onto healthy-track windows.

Usage:
    python augment_vibration.py raw/acoustic_track_pk/wheel_burnt/*.wav \\
        --label wheel_burnt --window 1.0 --hop 0.5 --copies 4 -o windows/wheel_burnt.npz
    python augment_vibration.py run.csv --columns Channel_1,Channel_2 --fs 2000 --label normal -o n.npz

Output .npz holds `x` (n, channels, samples), `y` (n,) labels, `fs`, and `source`.
Split train/test by *recording* (the `source` array) before training, never by
window, or overlapping windows leak between the splits.

Needs numpy.
"""

import argparse
import csv
import wave
from pathlib import Path

import numpy as np


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        fs, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.reshape(-1, ch).T.astype(np.float32) / 32768.0, fs


def load_csv(path, columns):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [[float(r[c]) for c in columns] for r in reader]
    return np.asarray(rows, dtype=np.float32).T


def load(path, columns, fs):
    path = Path(path)
    if path.suffix.lower() == ".wav":
        return load_wav(path)
    if fs is None:
        raise SystemExit(f"{path}: --fs is required for non-WAV input")
    if path.suffix.lower() == ".npy":
        x = np.load(path).astype(np.float32)
        return np.atleast_2d(x), fs
    if not columns:
        raise SystemExit(f"{path}: --columns is required for CSV input")
    return load_csv(path, columns), fs


def windows(x, fs, win_s, hop_s):
    win, hop = int(win_s * fs), int(hop_s * fs)
    n = x.shape[1]
    if n < win:
        return np.empty((0, x.shape[0], win), dtype=x.dtype)
    starts = range(0, n - win + 1, hop)
    return np.stack([x[:, s:s + win] for s in starts])


def speed_perturb(w, factor):
    """Resample by `factor` (>1 = faster pass: shorter event, higher frequencies)
    and crop/pad back to the original length."""
    n = w.shape[-1]
    src = np.arange(0, n, factor)
    out = np.stack([np.interp(src, np.arange(n), ch) for ch in w])
    if out.shape[-1] >= n:
        return out[:, :n]
    return np.pad(out, ((0, 0), (0, n - out.shape[-1])))


def add_noise(w, snr_db, rng):
    power = np.mean(w ** 2) + 1e-12
    return w + rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), w.shape).astype(w.dtype)


def augment(w, rng, speed=(0.85, 1.15), gain=(0.7, 1.3), snr=(10, 30)):
    w = speed_perturb(w, rng.uniform(*speed))
    w = w * rng.uniform(*gain)
    w = add_noise(w, rng.uniform(*snr), rng)
    return np.roll(w, rng.integers(0, w.shape[-1]), axis=-1).astype(np.float32)


def mix(fault, background, rng, ratio=(0.5, 1.0)):
    """Overlay a fault window on a healthy-track window (same length)."""
    return (rng.uniform(*ratio) * fault + background).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="WAV, CSV or NPY recordings")
    p.add_argument("--label", required=True)
    p.add_argument("-o", "--output", required=True, help="output .npz")
    p.add_argument("--fs", type=float, help="sample rate (Hz); read from WAV headers otherwise")
    p.add_argument("--columns", type=lambda s: s.split(","), help="CSV columns to use as channels")
    p.add_argument("--window", type=float, default=1.0, help="window length in seconds")
    p.add_argument("--hop", type=float, default=0.5, help="hop between windows in seconds")
    p.add_argument("--copies", type=int, default=4, help="augmented copies per window (0 = window only)")
    p.add_argument("--background", help="optional .npz of healthy windows to mix fault windows onto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    bg = np.load(args.background)["x"] if args.background else None
    xs, sources, fs_seen = [], [], set()
    for path in args.inputs:
        x, fs = load(path, args.columns, args.fs)
        if fs_seen and fs not in fs_seen:
            raise SystemExit(f"{path}: sample rate {fs} differs from {fs_seen.pop()}; resample first")
        fs_seen.add(fs)
        ws = windows(x, fs, args.window, args.hop)
        out = [ws]
        for _ in range(args.copies):
            aug = np.stack([augment(w, rng) for w in ws]) if len(ws) else ws
            if bg is not None and len(ws):
                aug = np.stack([mix(a, bg[rng.integers(len(bg))], rng) for a in aug])
            out.append(aug)
        ws = np.concatenate(out)
        xs.append(ws)
        sources += [str(path)] * len(ws)
        print(f"{path}: {len(ws)} windows")

    x = np.concatenate(xs)
    np.savez_compressed(args.output, x=x, y=np.array([args.label] * len(x)),
                        fs=fs_seen.pop(), source=np.array(sources))
    print(f"wrote {len(x)} windows, shape {x.shape[1:]}, to {args.output}")


if __name__ == "__main__":
    main()
