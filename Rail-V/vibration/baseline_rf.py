#!/usr/bin/env python3
"""Classical baseline: MFCC + spectral statistics per window, random forest classifier.

The Pakistan papers already reach high accuracy with features like these, so the deep
model has to beat this number on the same grouped split to be worth its cost.

Reports window-level and clip-level scores (clip = average of its windows'
probabilities) on the held-out test split, and optionally grouped 5-fold CV.
The final forest (trained on train+val, with the validation-tuned threshold for a
two-class task) is saved as model.joblib for predict.py.

    python baseline_rf.py --manifest manifest.csv --out runs/baseline_rf
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from common import (confusion, load_recording, macro_report, mfcc_features, positive_label, psd_features,
                    read_manifest, tune_threshold, windows)


def features(w, fs, kind):
    parts = []
    if kind in ("mfcc", "both"):
        parts.append(mfcc_features(w, fs))
    if kind in ("psd", "both"):
        parts.append(psd_features(w, fs))
    return np.concatenate(parts)


def featurise(rows, fs, win_s, hop_s, kind):
    X, y, rec = [], [], []
    for i, r in enumerate(rows):
        x, _ = load_recording(r["path"], target_fs=fs)
        for w in windows(x, fs, win_s, hop_s):
            X.append(features(w, fs, kind))
            y.append(r["label"])
            rec.append(i)
        if (i + 1) % 100 == 0:
            print(f"  featurised {i + 1}/{len(rows)} recordings")
    return np.stack(X), np.array(y), np.array(rec)


def clip_probs(clf, X, y, rec):
    """Mean class probabilities per clip, and each clip's true label."""
    proba = clf.predict_proba(X)
    clips = np.unique(rec)
    return (np.stack([proba[rec == r].mean(0) for r in clips]),
            np.array([y[rec == r][0] for r in clips]), proba)


def evaluate(clf, X, y, rec, labels, pos=None, threshold=None):
    """Argmax decisions, or for a two-class task with a tuned threshold, 'positive'
    whenever its probability reaches the threshold."""
    classes = list(clf.classes_)
    cp, clip_true, proba = clip_probs(clf, X, y, rec)
    if threshold is None:
        win_pred = np.array(classes)[proba.argmax(1)]
        clip_pred = np.array(classes)[cp.argmax(1)]
    else:
        neg = [c for c in classes if c != pos][0]
        k = classes.index(pos)
        win_pred = np.where(proba[:, k] >= threshold, pos, neg)
        clip_pred = np.where(cp[:, k] >= threshold, pos, neg)
    return {"window": macro_report(y, win_pred, labels),
            "clip": macro_report(clip_true, clip_pred, labels),
            "clip_confusion": confusion(clip_true, clip_pred, labels)}


def fit(X, y, seed):
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=seed)
    return clf.fit(X, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/baseline_rf"))
    ap.add_argument("--source", default="acoustic_track_pk")
    ap.add_argument("--fs", type=int, default=16000, help="resample everything to this rate")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--features", choices=["mfcc", "psd", "both"], default="both")
    ap.add_argument("--cv", action="store_true", help="also run grouped 5-fold CV on train+val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [r for r in read_manifest(args.manifest)
            if r["source"] == args.source and r["split"] in ("train", "val", "test")]
    if not rows:
        raise SystemExit(f"no labelled rows for source {args.source} in {args.manifest}")
    labels = sorted({r["label"] for r in rows})
    print(f"{len(rows)} recordings, classes {labels}")

    X, y, rec = featurise(rows, args.fs, args.window, args.hop, args.features)
    split = np.array([rows[i]["split"] for i in rec])
    fold = np.array([rows[i]["fold"] for i in rec])
    print(f"{len(X)} windows, {X.shape[1]} features each")

    results = {"config": vars(args) | {"manifest": str(args.manifest), "out": str(args.out)}}
    if args.cv:
        cv = []
        dev = split != "test"
        for k in sorted(set(fold[dev])):
            tr, va = dev & (fold != k), dev & (fold == k)
            res = evaluate(fit(X[tr], y[tr], args.seed), X[va], y[va], rec[va], labels)
            cv.append(res["clip"]["macro_f1"])
            print(f"  fold {k}: clip macro-F1 {res['clip']['macro_f1']}")
        results["cv_clip_macro_f1"] = {"folds": cv, "mean": round(float(np.mean(cv)), 4),
                                       "std": round(float(np.std(cv)), 4)}

    train = split != "test"
    pos = positive_label(labels)
    if pos:  # two-class task: pick the decision threshold on validation, never on test
        tr, va = split == "train", split == "val"
        clf = fit(X[tr], y[tr], args.seed)
        cp, true, _ = clip_probs(clf, X[va], y[va], rec[va])
        results["threshold"] = tune_threshold(cp[:, list(clf.classes_).index(pos)], true == pos)
        print(f"decision threshold for '{pos}' tuned on val: {results['threshold']:.3f}")
    clf = fit(X[train], y[train], args.seed)
    results["test"] = evaluate(clf, X[~train], y[~train], rec[~train], labels)
    if pos:
        results["test_tuned"] = evaluate(clf, X[~train], y[~train], rec[~train], labels, pos,
                                         results["threshold"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    joblib.dump({"clf": clf, "labels": labels, "fs": args.fs, "window": args.window, "hop": args.hop,
                 "features": args.features, "positive": pos, "threshold": results.get("threshold")},
                args.out / "model.joblib")
    t = results["test"]
    print(f"\nTEST  window acc {t['window']['accuracy']}  macro-F1 {t['window']['macro_f1']}")
    print(f"TEST  clip   acc {t['clip']['accuracy']}  macro-F1 {t['clip']['macro_f1']}  "
          f"recall {t['clip']['recall']}")
    print(t["clip_confusion"])
    if "test_tuned" in results:
        tt = results["test_tuned"]
        print(f"TEST  clip, tuned threshold: acc {tt['clip']['accuracy']}  macro-F1 {tt['clip']['macro_f1']}  "
              f"recall {tt['clip']['recall']}")
        print(tt["clip_confusion"])
    if "cv_clip_macro_f1" in results:
        print(f"CV clip macro-F1 {results['cv_clip_macro_f1']['mean']} ± {results['cv_clip_macro_f1']['std']}")
    print(f"saved {args.out / 'results.json'} and model.joblib")


if __name__ == "__main__":
    main()
