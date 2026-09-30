#!/usr/bin/env python3
"""Fine-tune a pretrained Audio Spectrogram Transformer (AST, AudioSet weights) on the
labelled rail audio.

AST is used rather than BEATs because its AudioSet checkpoint loads straight from
Hugging Face in Colab; BEATs needs a manual checkpoint download.

What it does:
- loads every labelled recording once (resampled to 16 kHz mono),
- trains on random windows cut on the fly, with speed perturbation, gain, noise
  and SpecAugment masks, so each epoch sees fresh variants,
- shortens AST's position embeddings to the window length (the original AST
  recipe for short clips), which makes 5 s windows ~2x cheaper than padding to 10 s,
- scores each clip by averaging its windows, keeps the checkpoint with the best
  validation clip macro-F1, then reports the held-out test split.

    python train_ast.py --manifest manifest.csv --out /content/drive/MyDrive/rail-v-runs/ast
    python train_ast.py --manifest manifest.csv --fold 0 ...   # grouped CV fold instead of val split
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (ASTConfig, ASTFeatureExtractor, ASTForAudioClassification,
                          get_cosine_schedule_with_warmup)

from common import (augment, confusion, load_recording, macro_report, positive_label, random_crop,
                    read_manifest, speed_crop, tune_threshold, windows)

FS = 16000
PRETRAINED = "MIT/ast-finetuned-audioset-10-10-0.4593"
POS_KEY = "audio_spectrogram_transformer.embeddings.position_embeddings"


def n_frames(n_samples):
    """Frames the AST (Kaldi fbank, 25 ms / 10 ms) extractor produces for n samples."""
    return 1 + (n_samples - 400) // 160


def build_model(labels, max_length, smoke=False):
    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}
    if smoke:  # tiny random model for testing the pipeline without downloading weights
        cfg = ASTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=128, max_length=max_length, num_labels=len(labels),
                        id2label=id2label, label2id=label2id)
        return ASTForAudioClassification(cfg)

    pre = ASTForAudioClassification.from_pretrained(PRETRAINED)
    cfg = ASTConfig.from_dict(pre.config.to_dict())
    cfg.max_length, cfg.num_labels, cfg.id2label, cfg.label2id = max_length, len(labels), id2label, label2id
    model = ASTForAudioClassification(cfg)

    state = {k: v for k, v in pre.state_dict().items() if not k.startswith("classifier.")}
    emb = pre.audio_spectrogram_transformer.embeddings
    f_old, t_old = emb.get_shape(pre.config)
    f_new, t_new = model.audio_spectrogram_transformer.embeddings.get_shape(cfg)
    pe = state[POS_KEY]
    grid = pe[:, 2:].reshape(1, f_old, t_old, -1)
    if t_new <= t_old:  # keep the first t_new time positions
        grid = grid[:, :, :t_new]
    else:  # longer than 10 s: interpolate along time
        grid = torch.nn.functional.interpolate(grid.permute(0, 3, 1, 2), size=(f_new, t_new),
                                               mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    state[POS_KEY] = torch.cat([pe[:, :2], grid.reshape(1, f_new * t_new, -1)], dim=1)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert all(k.startswith("classifier.") for k in missing), missing
    assert not unexpected, unexpected
    return model


def spec_augment(feat, valid_frames, rng, time_masks=2, time_w=40, freq_masks=2, freq_w=20):
    feat = feat.copy()
    for _ in range(time_masks):
        w = rng.integers(0, time_w + 1)
        s = rng.integers(0, max(1, valid_frames - w))
        feat[s:s + w] = 0.0  # 0 == dataset mean after the extractor's normalisation
    for _ in range(freq_masks):
        w = rng.integers(0, freq_w + 1)
        s = rng.integers(0, max(1, feat.shape[1] - w))
        feat[:, s:s + w] = 0.0
    return feat


class TrainWindows(Dataset):
    def __init__(self, signals, targets, n_samples, extractor, crops_per_rec, aug, balance=True):
        self.signals, self.targets = signals, targets
        self.n, self.fe, self.crops, self.aug = n_samples, extractor, crops_per_rec, aug
        # Class-balanced sampling: pick a class uniformly, then one of its recordings.
        # Matters for corrugation, where corrugated clips are a few percent of the run.
        self.by_class = [[k for k, t in enumerate(targets) if t == c] for c in sorted(set(targets))]
        self.balance = balance

    def __len__(self):
        return len(self.signals) * self.crops

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))
        if self.balance:
            members = self.by_class[rng.integers(len(self.by_class))]
            k = members[rng.integers(len(members))]
        else:
            k = i % len(self.signals)
        x = self.signals[k]
        if self.aug:
            w = augment(speed_crop(x, self.n, rng), rng)
        else:
            w = random_crop(x, self.n, rng)
        feat = self.fe(w, sampling_rate=FS, return_tensors="np")["input_values"][0]
        if self.aug:
            feat = spec_augment(feat, n_frames(self.n), rng)
        return torch.from_numpy(feat), self.targets[k]


class EvalWindows(Dataset):
    """Every window of every recording, tagged with its recording index."""

    def __init__(self, signals, targets, n_samples, hop_samples, extractor):
        self.items = []
        for k, x in enumerate(signals):
            for w in windows(x, FS, n_samples / FS, hop_samples / FS):
                self.items.append((w, targets[k], k))
        self.fe = extractor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        w, t, k = self.items[i]
        feat = self.fe(w, sampling_rate=FS, return_tensors="np")["input_values"][0]
        return torch.from_numpy(feat), t, k


@torch.no_grad()
def evaluate(model, loader, device, labels, n_recs, threshold=None):
    """Argmax decisions, or for a two-class task with a tuned threshold, the positive
    class whenever its clip probability reaches it. Also returns per-clip
    probabilities and labels so the threshold can be tuned on validation."""
    model.eval()
    logits_sum = torch.zeros(n_recs, len(labels))
    counts = torch.zeros(n_recs)
    true = [None] * n_recs
    win_true, win_pred = [], []
    for feat, t, k in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(input_values=feat.to(device)).logits.float().cpu()
        prob = logits.softmax(-1)
        logits_sum.index_add_(0, k, prob)
        counts.index_add_(0, k, torch.ones(len(k)))
        for ti, ki in zip(t.tolist(), k.tolist()):
            true[ki] = ti
        win_true += t.tolist()
        win_pred += logits.argmax(-1).tolist()
    seen = counts > 0
    clip_prob = logits_sum[seen] / counts[seen, None]
    clip_true = [true[i] for i in range(n_recs) if seen[i]]
    pos = positive_label(labels)
    if threshold is not None and pos:
        k = labels.index(pos)
        clip_pred = [k if p >= threshold else 1 - k for p in clip_prob[:, k].tolist()]
    else:
        clip_pred = clip_prob.argmax(-1).tolist()
    name = lambda ids: [labels[i] for i in ids]
    return {"window": macro_report(name(win_true), name(win_pred), labels),
            "clip": macro_report(name(clip_true), name(clip_pred), labels),
            "clip_confusion": confusion(name(clip_true), name(clip_pred), labels),
            "_clip_prob": clip_prob.numpy(), "_clip_true": clip_true}


def evaluate_threshold_f1(res, labels, pos, t):
    k = labels.index(pos)
    pred = [labels[k] if p >= t else labels[1 - k] for p in res["_clip_prob"][:, k].tolist()]
    return macro_report([labels[i] for i in res["_clip_true"]], pred, labels)["macro_f1"]


def public(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/ast"))
    ap.add_argument("--source", default="acoustic_track_pk")
    ap.add_argument("--fold", type=int, default=None,
                    help="use CV fold k (of train+val) as validation instead of the val split")
    ap.add_argument("--window", type=float, default=5.0, help="seconds per window")
    ap.add_argument("--hop", type=float, default=2.5, help="eval hop in seconds")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs without val gain")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crops-per-rec", type=int, default=8, help="random windows per recording per epoch")
    ap.add_argument("--lr", type=float, default=2e-5, help="encoder learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-balance", action="store_true", help="sample recordings uniformly, not per class")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny random model, for testing the code only")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rows = [r for r in read_manifest(args.manifest)
            if r["source"] == args.source and r["split"] in ("train", "val", "test")]
    if not rows:
        raise SystemExit(f"no labelled rows for source {args.source} in {args.manifest}")
    labels = sorted({r["label"] for r in rows})
    pos = positive_label(labels)
    if args.fold is None:
        part = {"train": "train", "val": "val", "test": "test"}
        role = [part[r["split"]] for r in rows]
    else:
        role = ["test" if r["split"] == "test" else "val" if r["fold"] == args.fold else "train"
                for r in rows]

    print(f"loading {len(rows)} recordings at {FS} Hz ...")
    t0 = time.time()
    signals = [load_recording(r["path"], target_fs=FS)[0] for r in rows]
    targets = [labels.index(r["label"]) for r in rows]
    print(f"  done in {time.time() - t0:.0f} s; classes {labels}")

    def subset(name):
        idx = [i for i, ro in enumerate(role) if ro == name]
        return [signals[i] for i in idx], [targets[i] for i in idx]

    n = int(round(args.window * FS))
    max_len = n_frames(n)
    extractor = ASTFeatureExtractor(sampling_rate=FS, max_length=max_len)
    tr_x, tr_y = subset("train")
    va_x, va_y = subset("val")
    te_x, te_y = subset("test")
    print(f"recordings: train {len(tr_x)}, val {len(va_x)}, test {len(te_x)}; "
          f"window {args.window} s = {max_len} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.smoke:
        print("WARNING: no GPU found; in Colab use Runtime > Change runtime type > T4 GPU")
    model = build_model(labels, max_len, smoke=args.smoke).to(device)

    train_dl = DataLoader(TrainWindows(tr_x, tr_y, n, extractor, args.crops_per_rec, not args.no_aug,
                                       not args.no_balance),
                          batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    hop = int(round(args.hop * FS))
    val_dl = DataLoader(EvalWindows(va_x, va_y, n, hop, extractor), batch_size=args.batch * 2,
                        num_workers=args.workers)
    test_dl = DataLoader(EvalWindows(te_x, te_y, n, hop, extractor), batch_size=args.batch * 2,
                         num_workers=args.workers)

    head = [p for name, p in model.named_parameters() if name.startswith("classifier.")]
    body = [p for name, p in model.named_parameters() if not name.startswith("classifier.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                            weight_decay=args.weight_decay)
    steps = args.epochs * len(train_dl)
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    args.out.mkdir(parents=True, exist_ok=True)
    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for feat, y in train_dl:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(input_values=feat.to(device)).logits.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val = evaluate(model, val_dl, device, labels, len(va_x))
        f1 = val["clip"]["macro_f1"]
        if pos:  # rare positive class: select on macro-F1 at the best val threshold
            t = tune_threshold(val["_clip_prob"][:, labels.index(pos)],
                               [labels[i] == pos for i in val["_clip_true"]])
            f1 = evaluate_threshold_f1(val, labels, pos, t)
        history.append({"epoch": epoch, "train_loss": round(total / max(1, len(train_dl)), 4),
                        "val_clip_macro_f1": f1, "val_window_macro_f1": val["window"]["macro_f1"]})
        print(f"epoch {epoch}: loss {history[-1]['train_loss']}  val clip macro-F1 {f1}  "
              f"window macro-F1 {val['window']['macro_f1']}  ({time.time() - t0:.0f} s)")
        if f1 > best:
            best, best_epoch = f1, epoch
            torch.save({"state_dict": model.state_dict(), "labels": labels,
                        "config": model.config.to_dict(), "window_s": args.window}, args.out / "best.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"no val improvement for {args.patience} epochs; stopping")
            break

    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device)["state_dict"])
    test = evaluate(model, test_dl, device, labels, len(te_x))
    results = {"config": {k: str(v) for k, v in vars(args).items()}, "labels": labels,
               "best_epoch": best_epoch, "best_val_clip_macro_f1": best, "history": history,
               "test": public(test)}
    if pos:  # threshold chosen on validation with the best checkpoint, then applied to test
        val = evaluate(model, val_dl, device, labels, len(va_x))
        results["threshold"] = tune_threshold(val["_clip_prob"][:, labels.index(pos)],
                                              [labels[i] == pos for i in val["_clip_true"]])
        results["test_tuned"] = public(evaluate(model, test_dl, device, labels, len(te_x),
                                                threshold=results["threshold"]))
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nbest epoch {best_epoch} (val clip macro-F1 {best})")
    print(f"TEST  window acc {test['window']['accuracy']}  macro-F1 {test['window']['macro_f1']}")
    print(f"TEST  clip   acc {test['clip']['accuracy']}  macro-F1 {test['clip']['macro_f1']}  "
          f"recall {test['clip']['recall']}")
    print(test["clip_confusion"])
    if pos:
        tt = results["test_tuned"]
        print(f"TEST  clip, threshold {results['threshold']:.3f} tuned on val: acc {tt['clip']['accuracy']}  "
              f"macro-F1 {tt['clip']['macro_f1']}  recall {tt['clip']['recall']}")
        print(tt["clip_confusion"])
    print(f"saved {args.out / 'best.pt'} and results.json")


if __name__ == "__main__":
    main()
