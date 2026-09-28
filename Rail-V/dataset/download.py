#!/usr/bin/env python3
"""Download the public Rail-V datasets into Rail-V/dataset/raw/.

Raw data is large and is never committed; raw/ is git-ignored.

Usage:
    python download.py --list
    python download.py rail_vivid_vibration corrugation
    python download.py all --dry-run

Optional dependencies, only needed for the sources that use them:
    pip install huggingface_hub   # rail_vivid*
    pip install kaggle            # kaggle_track_faults (needs ~/.kaggle/kaggle.json)
    pip install gdown             # acoustic_track_pk
"""

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"
MAX_GB = 20  # guard against accidentally pulling huge records; see --max-gb


def zenodo(record_id):
    def fetch(dest, dry_run):
        with urllib.request.urlopen(f"https://zenodo.org/api/records/{record_id}") as resp:
            record = json.load(resp)
        total_gb = sum(f["size"] for f in record["files"]) / 1e9
        print(f"  zenodo record {record_id}: {total_gb:.1f} GB")
        if total_gb > MAX_GB and not dry_run:
            raise RuntimeError(f"{total_gb:.0f} GB exceeds --max-gb {MAX_GB}; raise it to proceed")
        for f in record["files"]:
            url = f["links"]["self"]
            target = dest / f["key"]
            print(f"  {f['key']} ({f['size'] / 1e6:.1f} MB)")
            if dry_run or target.exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url) as resp, open(target, "wb") as out:
                shutil.copyfileobj(resp, out)
    return fetch


def huggingface(repo_id, allow_patterns=None):
    def fetch(dest, dry_run):
        print(f"  hf dataset {repo_id} patterns={allow_patterns or 'all'}")
        if dry_run:
            return
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dest,
                          allow_patterns=allow_patterns)
    return fetch


def git_branch(url, branch):
    def fetch(dest, dry_run):
        print(f"  git clone -b {branch} {url}")
        if dry_run or dest.exists():
            return
        subprocess.run(["git", "clone", "--depth", "1", "-b", branch, url, str(dest)], check=True)
    return fetch


def kaggle(slug):
    def fetch(dest, dry_run):
        print(f"  kaggle datasets download {slug}")
        if dry_run:
            return
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
                       check=True)
    return fetch


def gdrive_folder(url):
    def fetch(dest, dry_run):
        print(f"  gdown --folder {url}")
        if dry_run:
            return
        import gdown
        gdown.download_folder(url, output=str(dest), quiet=False)
    return fetch


def manual(url):
    def fetch(dest, dry_run):
        print(f"  manual download: open {url} and extract into {dest}")
    return fetch


# name -> (modality, description, fetcher). Details and licences are in README.md.
SOURCES = {
    # Rail-VIVID runs are folders like AtoB_20_1/ holding one CSV (6 accelerometer channels
    # at 2 kHz + GPS, 75-120 MB) and a frames subfolder of JPGs, which is most of the 114 GB.
    "rail_vivid_vibration": ("vibration", "Rail-VIVID accelerometer CSVs only, all 20 runs (~2 GB)",
                             huggingface("saluslab/Rail-VIVID", ["README.md", "*.py", "*.csv"])),
    "rail_vivid_sample": ("vibration+image", "Rail-VIVID, one run with its frames",
                          huggingface("saluslab/Rail-VIVID", ["README.md", "*.py", "AtoB_20_1/*"])),
    "rail_vivid": ("vibration+image", "Rail-VIVID, full (~114 GB)",
                   huggingface("saluslab/Rail-VIVID")),
    "corrugation": ("acoustic+vibration", "UPM onboard corrugation database (~607 MB)",
                    zenodo("16569018")),
    "draisine_vibration": ("vibration", "Rail-mounted triaxial accel. at 1.6 kHz, 93 passes w/ speed (12 MB)",
                           zenodo("19851718")),
    "kaggle_bogie_synthetic": ("vibration", "SYNTHETIC train-bogie vibration, 7/15 classes (~200 MB)",
                              kaggle("ziya07/high-speed-train-bogie-vibration-and-fault-diagnosis")),
    "madrid_bcn_onboard": ("vibration", "Onboard IMU + GPS, Madrid-Barcelona, 40 Hz, unlabelled",
                           zenodo("17607068")),
    "faultseg": ("image", "FaultSeg train-wheel defects (v1 ~8.5 GB)", zenodo("12957455")),
    "rsdds": ("image", "NEU RSDDS rail surface, 2D + depth + masks (zip password: neurail)",
              git_branch("https://github.com/neu-rail-rsdds/rsdds.git", "dataset_link")),
    "mendeley_track_surface": ("image", "MUET railway track surface faults, 7 classes",
                               manual("https://data.mendeley.com/datasets/8hxtgyyxrw/2")),
    "kaggle_track_faults": ("image", "Kaggle defective / non-defective rail + fastener images (2.1 GB)",
                            kaggle("salmaneunus/railway-track-fault-detection")),
    "kaggle_bangladesh_track": ("image", "Bangladesh Railway track fault images",
                                kaggle("ashikadnan/railway-track-fault-detectionbangladesh-railway")),
    "rfdd": ("image", "RFDD fastener defects, 1,350 images, 6 classes (CC BY-NC-ND)",
             manual("https://doi.org/10.57760/sciencedb.msdc.00071")),
    "acoustic_track_pk": ("acoustic", "Pakistan Railways cart audio: normal/superelevation/wheel burnt",
                          gdrive_folder("https://drive.google.com/drive/folders/"
                                        "1fgty5Ek_fTLLfnggFQy2KdWuktDp6usi")),
}


def main():
    global MAX_GB
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="source names, or 'all'")
    parser.add_argument("--list", action="store_true", help="list available sources")
    parser.add_argument("--dry-run", action="store_true", help="print what would be downloaded")
    parser.add_argument("--max-gb", type=float, default=MAX_GB,
                        help="refuse Zenodo records larger than this (default %(default)s)")
    args = parser.parse_args()
    MAX_GB = args.max_gb

    if args.list or not args.names:
        for name, (modality, desc, _) in SOURCES.items():
            print(f"{name:24} {modality:20} {desc}")
        return

    names = list(SOURCES) if args.names == ["all"] else args.names
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        sys.exit(f"unknown source(s): {', '.join(unknown)}; see --list")

    failed = []
    for name in names:
        print(f"[{name}]")
        try:
            SOURCES[name][2](RAW_DIR / name, args.dry_run)
        except Exception as exc:  # keep going so one dead link doesn't block the rest
            print(f"  FAILED: {exc}")
            failed.append(name)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
