#!/usr/bin/env python3
"""Print the layout of the rail image sets on Drive, so loaders can be written for them.

For each source folder: folders with their image counts, file types, a few image sizes,
and the contents of any zip (counts per top-level folder and type, without extracting).
Reads only a handful of images per folder, so it is quick even on the Drive mount.

    python inspect_images.py --data /content/drive/MyDrive/rail-v_dataset
    python inspect_images.py --data ... --sources rsdds kaggle_track_faults
"""

import argparse
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SOURCES = ["rsdds", "kaggle_track_faults", "kaggle_bangladesh_track", "mendeley_track_surface",
                   "faultseg", "rfdd"]


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return f"{im.size[0]}x{im.size[1]} {im.mode}"
    except Exception as e:  # noqa: BLE001 - report and keep going
        return f"unreadable ({type(e).__name__})"


def describe_zip(path, max_dirs=15):
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
    except (zipfile.BadZipFile, OSError) as e:
        print(f"    zip unreadable: {e} (split zips need joining first)")
        return
    by_dir = defaultdict(Counter)
    for n in names:
        parts = n.split("/")
        key = "/".join(parts[:-1][:3]) or "."
        by_dir[key][Path(n).suffix.lower()] += 1
    print(f"    {len(names)} files inside")
    for d, c in sorted(by_dir.items())[:max_dirs]:
        print(f"      {d}: " + ", ".join(f"{k or '(none)'} {v}" for k, v in c.most_common()))
    if len(by_dir) > max_dirs:
        print(f"      ... {len(by_dir) - max_dirs} more folders")


def describe_source(root, max_dirs=40, samples=2):
    files = [p for p in root.rglob("*") if p.is_file()]
    types = Counter(p.suffix.lower() for p in files)
    print(f"  {len(files)} files; types " + ", ".join(f"{k or '(none)'} {v}" for k, v in types.most_common(8)))
    by_dir = defaultdict(list)
    for p in files:
        by_dir[p.parent].append(p)
    for i, (d, ps) in enumerate(sorted(by_dir.items())):
        if i >= max_dirs:
            print(f"  ... {len(by_dir) - max_dirs} more folders")
            break
        imgs = sorted(p for p in ps if p.suffix.lower() in IMAGE_EXT)
        other = Counter(p.suffix.lower() for p in ps if p.suffix.lower() not in IMAGE_EXT)
        rel = d.relative_to(root) if d != root else Path(".")
        line = f"  {rel}/: {len(imgs)} images"
        if other:
            line += "; other " + ", ".join(f"{k or '(none)'} {v}" for k, v in other.most_common(4))
        print(line)
        for p in imgs[:samples]:
            print(f"    {p.name}: {image_size(p)}")
        for p in ps:
            if p.suffix.lower() == ".zip":
                print(f"    {p.name}:")
                describe_zip(p)
            elif p.suffix.lower() in {".csv", ".txt", ".json", ".xml"} and len(ps) < 50:
                try:
                    head = p.read_text(errors="replace").splitlines()[:2]
                    print(f"    {p.name}: " + " | ".join(h[:120] for h in head))
                except OSError as e:
                    print(f"    {p.name}: could not read ({e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sources", nargs="*", help=f"folders under --data (default: {' '.join(DEFAULT_SOURCES)})")
    args = ap.parse_args()
    print(f"data root: {args.data}")
    print("top-level folders:", sorted(p.name for p in args.data.iterdir() if p.is_dir()))
    for s in args.sources or DEFAULT_SOURCES:
        root = args.data / s
        print(f"\n=== {s} ===")
        if not root.exists():
            print("  (not downloaded)")
            continue
        describe_source(root)


if __name__ == "__main__":
    main()
