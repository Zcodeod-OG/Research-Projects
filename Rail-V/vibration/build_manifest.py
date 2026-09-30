#!/usr/bin/env python3
"""Scan the downloaded data and write one manifest.csv with a row per recording.

Columns: path, source, sensor, fs, channels, duration_s, label, group, split, fold.
Every later script reads only the manifest, so the split is fixed once here.

Splits are grouped: all windows of a recording, and recordings that probably come
from the same stretch of track, stay on one side. For the Pakistan clips, files
whose numbers fall in the same block of --group-block (default 10) share a group,
because consecutive clips are likely cut from one drive. The two microphones'
files share a group when they share a number. Set --group-block 1 to group by
file only (optimistic; test scores will be higher and less trustworthy).

    python build_manifest.py --data /content/drive/MyDrive/rail-v-data --out manifest.csv
"""

import argparse
import csv
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from common import MANIFEST_COLUMNS

PK_LABELS = [("superelevation", ("super",)), ("wheel_burnt", ("burn", "wheel")),
             ("normal", ("normal", "healthy"))]


def extract_zips(d):
    """Extract any .zip in d next to itself (once), so loaders only see plain files."""
    for z in sorted(d.rglob("*.zip")):
        target = z.with_suffix("")
        if target.exists():
            continue
        print(f"  extracting {z.relative_to(d)} -> {target.name}/")
        with zipfile.ZipFile(z) as f:
            f.extractall(target)


def pk_label(rel):
    s = str(rel).lower()
    for label, keys in PK_LABELS:
        if any(k in s for k in keys):
            return label
    return None


def number_in(stem):
    nums = re.findall(r"\d+", stem)
    return int(nums[-1]) if nums else None


def wav_row(p, source, sensor, label, group):
    info = sf.info(str(p))
    return {"path": str(p), "source": source, "sensor": sensor, "fs": info.samplerate,
            "channels": info.channels, "duration_s": round(info.duration, 3),
            "label": label, "group": group}


def scan_pk(d, block):
    rows, skipped = [], []
    for p in sorted(d.rglob("*.wav")):
        rel = p.relative_to(d)
        label = pk_label(rel)
        if label is None:
            skipped.append(str(rel))
            continue
        n = number_in(p.stem)
        group = f"pk/{label}/{n // block}" if n is not None else f"pk/{label}/{p.stem}"
        rows.append(wav_row(p, "acoustic_track_pk", "mic", label, group))
    if skipped:
        print(f"  WARNING: {len(skipped)} WAVs with no recognisable class in their path, e.g. {skipped[:3]}")
    return rows


def scan_draisine(d):
    # Healthy-track passes; not used by the classifiers yet (different sensor), kept for
    # the healthy-only anomaly detector.
    return [wav_row(p, "draisine_vibration", "accel", "healthy", f"draisine/{p.stem}")
            for p in sorted(d.rglob("*.wav"))]


def scan_rail_vivid(d):
    # Unlabelled until rows are matched to the anomaly GPS positions (later step).
    # Duration is left at 0: counting rows of ~2 GB of CSV on Drive is slow.
    return [{"path": str(p), "source": "rail_vivid_vibration", "sensor": "accel", "fs": 2000,
             "channels": 6, "duration_s": 0, "label": "unlabelled", "group": f"vivid/{p.parent.name}"}
            for p in sorted(d.rglob("*.csv"))]


def assign_splits(rows, val=0.15, test=0.15, folds=5, seed=0):
    """Grouped, per-label split into train/val/test, plus a CV fold id for train+val.
    Unlabelled / healthy-only sources go to split 'pool'."""
    rng = np.random.default_rng(seed)
    by_label = defaultdict(set)
    for r in rows:
        if r["source"] == "acoustic_track_pk":
            by_label[r["label"]].add(r["group"])
    split_of, fold_of = {}, {}
    for label, groups in sorted(by_label.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        n_test = max(1, round(len(groups) * test))
        n_val = max(1, round(len(groups) * val))
        for i, g in enumerate(groups):
            split_of[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            if split_of[g] != "test":
                fold_of[g] = i % folds
    for r in rows:
        r["split"] = split_of.get(r["group"], "pool")
        r["fold"] = fold_of.get(r["group"], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder download.py wrote into")
    ap.add_argument("--out", type=Path, default=Path("manifest.csv"))
    ap.add_argument("--group-block", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-extract", action="store_true", help="don't unzip archives found in the data")
    args = ap.parse_args()

    rows = []
    scanners = {"acoustic_track_pk": lambda d: scan_pk(d, args.group_block),
                "draisine_vibration": scan_draisine, "rail_vivid_vibration": scan_rail_vivid}
    for name, scan in scanners.items():
        d = args.data / name
        print(f"[{name}]")
        if not d.exists():
            print("  not found, skipped")
            continue
        if not args.no_extract:
            extract_zips(d)
        found = scan(d)
        print(f"  {len(found)} recordings")
        rows += found
    if (args.data / "corrugation").exists():
        print("[corrugation] found, but its .mat layout isn't known yet; run inspect_data.py and "
              "share the output so a loader can be added")

    assign_splits(rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    labelled = [r for r in rows if r["split"] != "pool"]
    print(f"\nwrote {len(rows)} rows to {args.out}")
    if labelled:
        table = Counter((r["label"], r["split"]) for r in labelled)
        labels = sorted({r["label"] for r in labelled})
        print(f"{'label':16}{'train':>8}{'val':>8}{'test':>8}{'groups':>8}")
        for lab in labels:
            groups = len({r['group'] for r in labelled if r['label'] == lab})
            print(f"{lab:16}" + "".join(f"{table[(lab, s)]:>8}" for s in ("train", "val", "test"))
                  + f"{groups:>8}")
        rates = Counter((r["fs"], r["channels"]) for r in labelled)
        print("sample rate / channels:", dict(rates))


if __name__ == "__main__":
    main()
