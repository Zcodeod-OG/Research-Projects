"""Shared helpers: loading recordings, resampling, windowing, augmentation, MFCC features.

Kept dependency-light (numpy, scipy, soundfile) so the baseline runs anywhere; only
train_ast.py needs torch + transformers.
"""

import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly

MANIFEST_COLUMNS = ["path", "source", "sensor", "fs", "channels", "duration_s",
                    "label", "group", "split", "fold"]


def read_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["fs"] = float(r["fs"])
        r["channels"] = int(r["channels"])
        r["duration_s"] = float(r["duration_s"])
        r["fold"] = int(r["fold"]) if r["fold"] not in ("", None) else -1
    return rows


def load_recording(path, target_fs=None):
    """Return (mono float32 signal, fs). Multi-channel audio is averaged to mono."""
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if target_fs and fs != target_fs:
        x = resample(x, fs, target_fs)
        fs = target_fs
    return x.astype(np.float32), fs


def resample(x, fs_in, fs_out):
    fs_in, fs_out = int(round(fs_in)), int(round(fs_out))
    g = math.gcd(fs_in, fs_out)
    return resample_poly(x, fs_out // g, fs_in // g).astype(np.float32)


def windows(x, fs, win_s, hop_s):
    """Cut a 1-D signal into overlapping windows; a short signal is zero-padded to one window."""
    win, hop = int(round(win_s * fs)), int(round(hop_s * fs))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    starts = range(0, len(x) - win + 1, hop)
    return np.stack([x[s:s + win] for s in starts])


# ---- augmentation (waveform level, applied on the fly during training) ----

def random_crop(x, n, rng):
    if len(x) <= n:
        return np.pad(x, (0, n - len(x)))
    s = rng.integers(0, len(x) - n + 1)
    return x[s:s + n]


def speed_crop(x, n, rng, speed=(0.85, 1.15)):
    """Random crop of n samples at a random playback speed. Mimics the same defect
    passed at a different train speed: faster pass = shorter event, higher frequencies."""
    factor = rng.uniform(*speed)
    seg = random_crop(x, int(math.ceil(n * factor)) + 1, rng)
    src = np.arange(n) * factor
    return np.interp(src, np.arange(len(seg)), seg).astype(np.float32)


def augment(w, rng, gain=(0.7, 1.3), snr_db=(10, 30)):
    w = w * rng.uniform(*gain)
    power = float(np.mean(w ** 2)) + 1e-12
    noise_power = power / 10 ** (rng.uniform(*snr_db) / 10)
    w = w + rng.normal(0, math.sqrt(noise_power), w.shape)
    return w.astype(np.float32)


# ---- MFCC features for the classical baseline ----

def _mel_filterbank(fs, n_fft, n_mels, fmin=20.0, fmax=None):
    fmax = fmax or fs / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b > a:
            fb[i, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[i, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


_FB_CACHE = {}


def mfcc_features(w, fs, n_mfcc=20, n_mels=64, frame_s=0.025, hop_s=0.010):
    """Fixed-length feature vector for one window: MFCC mean/std, delta mean/std,
    plus RMS, spectral centroid, bandwidth and roll-off statistics."""
    n_fft = 1 << int(math.ceil(math.log2(frame_s * fs)))
    hop = max(1, int(hop_s * fs))
    if len(w) < n_fft:
        w = np.pad(w, (0, n_fft - len(w)))
    frames = np.lib.stride_tricks.sliding_window_view(w, n_fft)[::hop] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    key = (fs, n_fft, n_mels)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = _mel_filterbank(fs, n_fft, n_mels)
    logmel = np.log(spec @ _FB_CACHE[key].T + 1e-10)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    delta = np.diff(mfcc, axis=0) if len(mfcc) > 1 else np.zeros_like(mfcc)

    freqs = np.fft.rfftfreq(n_fft, 1 / fs)
    total = spec.sum(axis=1) + 1e-12
    centroid = (spec * freqs).sum(axis=1) / total
    bandwidth = np.sqrt((spec * (freqs - centroid[:, None]) ** 2).sum(axis=1) / total)
    rolloff = freqs[np.minimum((np.cumsum(spec, axis=1) >= 0.85 * total[:, None]).argmax(axis=1),
                               len(freqs) - 1)]
    rms = np.sqrt((frames ** 2).mean(axis=1))
    scalars = np.stack([rms, centroid, bandwidth, rolloff], axis=1)
    return np.concatenate([mfcc.mean(0), mfcc.std(0), delta.mean(0), delta.std(0),
                           scalars.mean(0), scalars.std(0)]).astype(np.float32)


def macro_report(y_true, y_pred, labels):
    """Accuracy, macro-F1 and a per-class recall dict, without needing sklearn."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s, recall = [], {}
    for c in labels:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        recall[c] = round(float(r), 4)
    return {"accuracy": round(float(np.mean(y_true == y_pred)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4), "recall": recall}


def confusion(y_true, y_pred, labels):
    idx = {c: i for i, c in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    width = max(len(c) for c in labels) + 2
    lines = ["true \\ pred".ljust(width) + "".join(c[:10].rjust(12) for c in labels)]
    lines += [c.ljust(width) + "".join(str(v).rjust(12) for v in row) for c, row in zip(labels, m)]
    return "\n".join(lines)


def data_root_default():
    return Path("/content/drive/MyDrive/rail-v-data")
