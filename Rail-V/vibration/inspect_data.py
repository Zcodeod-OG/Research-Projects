#!/usr/bin/env python3
"""Print a compact summary of the downloaded vibration / acoustic data.

Run this once after downloading and paste the output back into the project thread:
it shows the folder layout, file types, WAV sample rates, CSV headers and the
variables inside the UPM corrugation .mat files, which the loaders depend on.

    python inspect_data.py /content/drive/MyDrive/rail-v-data
"""

import argparse
import csv
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

VIBRATION_SETS = ["acoustic_track_pk", "corrugation", "draisine_vibration",
                  "rail_vivid_vibration", "kaggle_bogie_synthetic", "madrid_bcn_onboard"]


def human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def describe_wav(p):
    import soundfile as sf
    info = sf.info(str(p))
    return f"{info.samplerate} Hz, {info.channels} ch, {info.duration:.1f} s, {info.subtype}"


def describe_mat(p):
    try:
        from scipy.io import whosmat
        return "; ".join(f"{name} {shape} {cls}" for name, shape, cls in whosmat(str(p)))
    except (NotImplementedError, ValueError):  # MATLAB v7.3 files are HDF5
        import h5py
        out = []
        with h5py.File(p, "r") as f:
            f.visititems(lambda name, obj: out.append(f"{name} {getattr(obj, 'shape', '')}")
                         if hasattr(obj, "shape") else None)
        return "(v7.3) " + "; ".join(out[:40])


def describe_csv(p):
    with open(p, newline="", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        first = next(reader, [])
    return f"columns={header[:12]} first_row={first[:12]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", type=Path)
    ap.add_argument("--samples", type=int, default=6, help="example paths to print per dataset")
    args = ap.parse_args()

    root = args.data_root
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    print(f"data root: {root}")
    print("top-level folders:", sorted(p.name for p in root.iterdir() if p.is_dir()))

    for name in VIBRATION_SETS:
        d = root / name
        print(f"\n=== {name} ===")
        if not d.exists():
            print("  (not downloaded)")
            continue
        files = [p for p in d.rglob("*") if p.is_file() and ".cache" not in p.parts]
        exts = Counter(p.suffix.lower() or "<none>" for p in files)
        print(f"  {len(files)} files, {human(sum(p.stat().st_size for p in files))}; types {dict(exts)}")

        per_dir = defaultdict(int)
        for p in files:
            per_dir[str(p.parent.relative_to(d))] += 1
        for sub, n in sorted(per_dir.items())[:25]:
            print(f"  dir {sub or '.'}: {n} files")
        if len(per_dir) > 25:
            print(f"  ... {len(per_dir) - 25} more folders")

        shown = defaultdict(int)
        for p in sorted(files):
            ext = p.suffix.lower()
            if shown[ext] >= args.samples:
                continue
            shown[ext] += 1
            rel = p.relative_to(d)
            try:
                if ext == ".wav":
                    detail = describe_wav(p)
                elif ext == ".mat":
                    detail = describe_mat(p)
                elif ext == ".csv":
                    detail = describe_csv(p)
                elif ext == ".zip":
                    with zipfile.ZipFile(p) as z:
                        names = z.namelist()
                    detail = f"zip with {len(names)} entries, e.g. {names[:5]}"
                else:
                    detail = human(p.stat().st_size)
            except Exception as exc:  # keep going; a bad file shouldn't hide the rest
                detail = f"could not read: {exc}"
            print(f"  {rel}: {detail}")


if __name__ == "__main__":
    main()
