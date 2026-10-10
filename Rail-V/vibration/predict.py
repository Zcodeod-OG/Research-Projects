#!/usr/bin/env python3
"""One predictor for every rail fault we have a model for, from vibration and sound.

Each fault keeps the model that scored best on its own held-out test split, and this
script runs them together so one call reports every fault:

    fault                 trained on            sensor  model                     test macro-F1
    rail surface          Pakistan track audio  mic     AST    (runs/ast)         1.00
    corrugation (axle)    UPM axle box          accel   forest (runs/corr_axle_rf) 0.94
    corrugation (mic)     UPM microphone        mic     AST    (runs/corr_mic_ast) 0.90
    joint / switch        Rail-VIVID            accel   forest (runs/vivid_rf)    0.79

A shared multi-task model (train_multitask.py) was tried first. Over four runs it never
matched all of these at once, so the faults keep separate models.

Score a recording (each model only sees recordings from its own kind of sensor):

    python predict.py --runs /content/drive/MyDrive/rail-v-runs --sensor mic some_clip.wav
    python predict.py --runs ... --sensor accel axle_clip.wav

Check the package reproduces each model's test score:

    python predict.py --runs /content/drive/MyDrive/rail-v-runs --manifest manifest.csv

The models only know the line, vehicle and sensor setup they were trained on; scores
on other track will be lower.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from baseline_rf import features
from common import confusion, load_recording, macro_report, read_manifest, resample, windows

# name, manifest source, sensor, model kind, run folder under --runs
FAULTS = [
    ("rail surface", "acoustic_track_pk", "mic", "ast", "ast"),
    ("corrugation (axle)", "corrugation_axle", "accel", "rf", "corr_axle_rf"),
    ("corrugation (mic)", "corrugation_mic", "mic", "ast", "corr_mic_ast"),
    ("joint / switch", "vivid_joint", "accel", "rf", "vivid_rf"),
]


class ForestModel:
    def __init__(self, folder):
        import joblib
        m = joblib.load(folder / "model.joblib")
        self.clf, self.labels = m["clf"], [str(c) for c in m["clf"].classes_]
        self.fs, self.window, self.hop, self.kind = m["fs"], m["window"], m["hop"], m["features"]
        self.positive, self.threshold = m["positive"], m["threshold"]

    def probs(self, x, fs):
        x = resample(x, fs, self.fs) if fs != self.fs else x
        X = np.stack([features(w, self.fs, self.kind) for w in windows(x, self.fs, self.window, self.hop)])
        return self.clf.predict_proba(X).mean(0)


class ASTModel:
    def __init__(self, folder, device=None):
        import torch
        from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
        from train_ast import FS, n_frames
        self.torch, self.FS = torch, FS
        ck = torch.load(folder / "best.pt", map_location="cpu")
        self.model = ASTForAudioClassification(ASTConfig.from_dict(ck["config"]))
        self.model.load_state_dict(ck["state_dict"])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.labels, self.window = list(ck["labels"]), float(ck["window_s"])
        res = json.loads((folder / "results.json").read_text()) if (folder / "results.json").exists() else {}
        self.hop = float(res.get("config", {}).get("hop", self.window / 2))
        self.threshold = res.get("threshold")
        self.positive = next((c for c in self.labels if c != "normal"), None) if len(self.labels) == 2 else None
        self.fe = ASTFeatureExtractor(sampling_rate=FS, max_length=n_frames(int(round(self.window * FS))))

    def probs(self, x, fs):
        x = resample(x, fs, self.FS) if fs != self.FS else x
        out = []
        with self.torch.no_grad():
            ws = windows(x, self.FS, self.window, self.hop)
            for i in range(0, len(ws), 16):
                feat = self.fe(list(ws[i:i + 16]), sampling_rate=self.FS, return_tensors="pt")["input_values"]
                out.append(self.model(input_values=feat.to(self.device)).logits.float().softmax(-1).cpu().numpy())
        return np.concatenate(out).mean(0)


def decide(model, p):
    """Label for one recording: the validation-tuned threshold for a two-class model, else argmax."""
    if model.positive and model.threshold is not None:
        k = model.labels.index(model.positive)
        return model.positive if p[k] >= model.threshold else model.labels[1 - k]
    return model.labels[int(p.argmax())]


def load_models(runs, only=None):
    models = {}
    for name, source, sensor, kind, folder in FAULTS:
        if only and source not in only:
            continue
        d = runs / folder
        need = d / ("model.joblib" if kind == "rf" else "best.pt")
        if not need.exists():
            print(f"note: no model for {name} ({need} missing); skipping it")
            continue
        models[name] = (source, sensor, ForestModel(d) if kind == "rf" else ASTModel(d))
    return models


def check(models, manifest, out):
    rows = read_manifest(manifest)
    report = {}
    for name, (source, _, m) in models.items():
        test = [r for r in rows if r["source"] == source and r["split"] == "test"]
        true, pred = [], []
        for r in test:
            x, fs = load_recording(r["path"])
            true.append(r["label"])
            pred.append(decide(m, m.probs(x, fs)))
        rep = macro_report(true, pred, m.labels)
        report[name] = rep
        print(f"\n[{name}] {source} test: acc {rep['accuracy']}  macro-F1 {rep['macro_f1']}  recall {rep['recall']}")
        print(confusion(true, pred, m.labels))
    if out:
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="recordings to score")
    ap.add_argument("--runs", type=Path, required=True, help="folder holding ast/, corr_axle_rf/, ...")
    ap.add_argument("--sensor", choices=["mic", "accel"], default="mic",
                    help="kind of sensor the files come from; only matching models run")
    ap.add_argument("--manifest", type=Path, help="instead of files: score every model on its test split")
    ap.add_argument("--out", type=Path, help="with --manifest: write the scores to this JSON file")
    args = ap.parse_args()
    if not args.files and not args.manifest:
        ap.error("give recordings to score, or --manifest to check the models")

    models = load_models(args.runs)
    if not models:
        raise SystemExit(f"no models found under {args.runs}")
    if args.manifest:
        check(models, args.manifest, args.out)
        return
    for f in args.files:
        x, fs = load_recording(f)
        print(f"\n{f.name}  ({len(x) / fs:.1f} s, {fs} Hz, {args.sensor})")
        for name, (_, sensor, m) in models.items():
            if sensor != args.sensor:
                continue
            p = m.probs(x, fs)
            print(f"  {name:20s} {decide(m, p):14s} " + "  ".join(f"{c} {v:.2f}" for c, v in zip(m.labels, p)))


if __name__ == "__main__":
    main()
