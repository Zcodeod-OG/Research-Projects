#!/usr/bin/env python3
"""One model for every rail fault we have labels for, from vibration and sound only.

A single pretrained AST encoder (AudioSet weights) is shared by all datasets, with one
small output head per fault family:

    surface      acoustic_track_pk            normal / superelevation / wheel burn
    corrugation  corrugation_axle             normal / corrugated
    corr_mic     corrugation_mic              normal / corrugated
    joint        vivid_joint                  normal / joint (joint or switch stretch)

Why one head per family instead of one big output: each fault type comes from a
different dataset, sensor and line. A single "which fault?" output would learn to
recognise the dataset (microphone, vehicle, noise floor) rather than the fault. Here
every head only ever compares a fault with normal track from the same recordings,
while the shared encoder learns from all of them. Corrugation's microphone and axle
clips get separate heads: sharing one made the microphone score fall as the axle
score rose (0.75 vs 0.90 on its own), so the two sensors are kept apart.

Sensors: microphones are resampled to 16 kHz. Accelerometers are "played back
faster": their samples are treated as if recorded at --accel-as Hz (default 4000), so
a 2 kHz sensor's 0-1 kHz content covers the lower half of AST's mel bands while a
20 m Rail-VIVID clip still lasts 1.5 s. A faster speed-up made clips shorter than the
window.

Loading ~5000 small files from Google Drive can take over an hour; --cache keeps
every decoded signal in one file so later runs load in a minute or two.

Each batch picks a head (equal shares by default, --head-weights to change), then a
class (balanced), then a recording, and cuts a random window. Each clip trains only
its own head. Every source is evaluated on its own held-out split, so the numbers
compare directly with the single-task results (train_ast.py, baseline_rf.py). The
checkpoint kept is the one whose weakest source scores best on validation (--select
min); picking by the mean let one head sit at chance while the others carried it.
The surface head gets twice the batches by default: its wheel-burn class was the
first to collapse when it had to share training with three other heads.
Per-clip test predictions go to predictions.csv, to see which clips each head misses.

    python train_multitask.py --manifest manifest.csv --out /content/drive/MyDrive/rail-v-runs/multitask
"""

import argparse
import csv
import json
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, get_cosine_schedule_with_warmup

from common import (augment, confusion, load_recording, macro_report, positive_label, random_crop,
                    read_manifest, resample, speed_crop, tune_threshold, windows)
from train_ast import FS, build_model, n_frames, spec_augment

HEAD_OF = {"acoustic_track_pk": "surface", "corrugation_mic": "corr_mic",
           "corrugation_axle": "corrugation", "vivid_joint": "joint"}


def load_signal(row, accel_as):
    """Mono float32 at 16 kHz. Accelerometers are reinterpreted at accel_as Hz first."""
    if row["sensor"] == "accel":
        x, _ = sf.read(row["path"], dtype="float32", always_2d=True)
        return resample(x.mean(1), accel_as, FS)
    return load_recording(row["path"], target_fs=FS)[0]


class Encoder(torch.nn.Module):
    """Shared AST encoder with one LayerNorm + Linear head per fault family."""

    def __init__(self, heads, max_len, smoke=False):
        super().__init__()
        ast = build_model(["x", "y"], max_len, smoke=smoke)
        self.body = ast.audio_spectrogram_transformer
        dim = ast.config.hidden_size
        self.heads = torch.nn.ModuleDict({h: torch.nn.Sequential(torch.nn.LayerNorm(dim),
                                                                  torch.nn.Linear(dim, len(labels)))
                                          for h, labels in heads.items()})

    def embed(self, x):
        return self.body(input_values=x).pooler_output

    def forward(self, x, head):
        return self.heads[head](self.embed(x))


class MixedWindows(Dataset):
    def __init__(self, recs, n, extractor, steps, weights, aug):
        # recs[head] = list of (signal, target)
        self.recs, self.n, self.fe, self.steps, self.aug = recs, n, extractor, steps, aug
        self.heads = sorted(recs)
        w = np.array([weights.get(h, 1.0) for h in self.heads], float)
        self.p = w / w.sum()
        self.by_class = {h: defaultdict(list) for h in self.heads}
        for h in self.heads:
            for k, (_, t) in enumerate(recs[h]):
                self.by_class[h][t].append(k)

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,))))
        h = self.heads[rng.choice(len(self.heads), p=self.p)]
        classes = sorted(self.by_class[h])
        members = self.by_class[h][classes[rng.integers(len(classes))]]
        x, t = self.recs[h][members[rng.integers(len(members))]]
        w = augment(speed_crop(x, self.n, rng), rng) if self.aug else random_crop(x, self.n, rng)
        feat = self.fe(w, sampling_rate=FS, return_tensors="np")["input_values"][0]
        if self.aug:
            feat = spec_augment(feat, n_frames(self.n), rng)
        return torch.from_numpy(feat), self.heads.index(h), t


class EvalWindows(Dataset):
    def __init__(self, signals, targets, n, hop, extractor):
        self.items = [(w, targets[k], k) for k, x in enumerate(signals)
                      for w in windows(x, FS, n / FS, hop / FS)]
        self.fe = extractor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        w, t, k = self.items[i]
        return torch.from_numpy(self.fe(w, sampling_rate=FS, return_tensors="np")["input_values"][0]), t, k


@torch.no_grad()
def evaluate(model, loader, head, labels, n_recs, device, threshold=None):
    model.eval()
    prob_sum, counts, true = torch.zeros(n_recs, len(labels)), torch.zeros(n_recs), [None] * n_recs
    for feat, t, k in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            prob = model(feat.to(device), head).float().softmax(-1).cpu()
        prob_sum.index_add_(0, k, prob)
        counts.index_add_(0, k, torch.ones(len(k)))
        for ti, ki in zip(t.tolist(), k.tolist()):
            true[ki] = ti
    seen = counts > 0
    clip_prob = (prob_sum[seen] / counts[seen, None]).numpy()
    clip_true = [true[i] for i in range(n_recs) if seen[i]]
    pos = positive_label(labels)
    if threshold is not None and pos:
        kp = labels.index(pos)
        pred = [kp if p >= threshold else 1 - kp for p in clip_prob[:, kp]]
    else:
        pred = clip_prob.argmax(-1).tolist()
    names = lambda ids: [labels[i] for i in ids]  # noqa: E731
    return {"clip": macro_report(names(clip_true), names(pred), labels),
            "clip_confusion": confusion(names(clip_true), names(pred), labels),
            "_prob": clip_prob, "_true": clip_true}


def tuned(res, labels):
    """Validation threshold for a two-class head (None otherwise) and the macro-F1 there."""
    pos = positive_label(labels)
    if not pos:
        return None, res["clip"]["macro_f1"]
    kp = labels.index(pos)
    t = tune_threshold(res["_prob"][:, kp], [labels[i] == pos for i in res["_true"]])
    pred = [pos if p >= t else labels[1 - kp] for p in res["_prob"][:, kp]]
    return t, macro_report([labels[i] for i in res["_true"]], pred, labels)["macro_f1"]


def public(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/multitask"))
    ap.add_argument("--sources", default=",".join(HEAD_OF), help="comma-separated manifest sources")
    ap.add_argument("--head-weights", default="surface=2", help="batch share per head, e.g. surface=2,joint=1 (others 1)")
    ap.add_argument("--select", choices=["min", "mean"], default="min",
                    help="keep the checkpoint with the best weakest-source (min) or mean val score")
    ap.add_argument("--accel-as", type=int, default=4000, help="treat accelerometer samples as this rate (Hz)")
    ap.add_argument("--window", type=float, default=2.0, help="seconds per window (after accel speed-up)")
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=8000, help="training windows per epoch")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--cache", type=Path, help="file holding the decoded signals (made on the first run)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny random model, for testing the code only")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    wanted = [s for s in args.sources.split(",") if s]
    unknown = [s for s in wanted if s not in HEAD_OF]
    if unknown:
        raise SystemExit(f"no head defined for {unknown}; known: {list(HEAD_OF)}")
    rows = [r for r in read_manifest(args.manifest)
            if r["source"] in wanted and r["split"] in ("train", "val", "test")]
    sources = sorted({r["source"] for r in rows})
    missing = sorted(set(wanted) - set(sources))
    if missing:
        print(f"note: no labelled rows for {missing}; training without them")
    if not rows:
        raise SystemExit("no labelled rows for any requested source")
    heads = {}
    for s in sources:
        heads.setdefault(HEAD_OF[s], set()).update(r["label"] for r in rows if r["source"] == s)
    heads = {h: sorted(v) for h, v in heads.items()}
    print("heads: " + "; ".join(f"{h} {labels}" for h, labels in heads.items()))

    def key(r):
        return f"{r['path']}|{args.accel_as}" if r["sensor"] == "accel" else r["path"]

    cached = {}
    if args.cache and args.cache.exists():
        with open(args.cache, "rb") as f:
            cached = pickle.load(f)
    todo = [r for r in rows if key(r) not in cached]
    print(f"loading {len(rows)} recordings ({len(rows) - len(todo)} from cache) ...")
    t0 = time.time()
    for r in todo:
        cached[key(r)] = load_signal(r, args.accel_as).astype(np.float16)
    print(f"  done in {time.time() - t0:.0f} s")
    if args.cache and todo:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, "wb") as f:
            pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  saved signals to {args.cache}")
    data = defaultdict(lambda: defaultdict(list))  # data[source][split] = [(signal, target)]
    paths = defaultdict(lambda: defaultdict(list))
    for r in rows:
        h = HEAD_OF[r["source"]]
        data[r["source"]][r["split"]].append((cached[key(r)].astype(np.float32), heads[h].index(r["label"])))
        paths[r["source"]][r["split"]].append(r["path"])
    del cached
    for s in sources:
        print(f"  {s}: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(data[s].items())))

    n = int(round(args.window * FS))
    max_len = n_frames(n)
    fe = ASTFeatureExtractor(sampling_rate=FS, max_length=max_len)
    train_recs = defaultdict(list)
    for s in sources:
        train_recs[HEAD_OF[s]] += data[s]["train"]
    weights = {k: float(v) for k, v in (kv.split("=") for kv in args.head_weights.split(",") if kv)}
    head_names = sorted(train_recs)
    train_dl = DataLoader(MixedWindows(train_recs, n, fe, args.steps_per_epoch, weights, not args.no_aug),
                          batch_size=args.batch, shuffle=False, num_workers=args.workers, drop_last=True)
    hop = int(round(args.hop * FS))

    def loader(s, split):
        recs = data[s][split]
        return DataLoader(EvalWindows([x for x, _ in recs], [t for _, t in recs], n, hop, fe),
                          batch_size=args.batch * 2, num_workers=args.workers), len(recs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.smoke:
        print("WARNING: no GPU found; in Colab use Runtime > Change runtime type > T4 GPU")
    model = Encoder(heads, max_len, smoke=args.smoke).to(device)
    opt = torch.optim.AdamW([{"params": model.body.parameters(), "lr": args.lr},
                             {"params": model.heads.parameters(), "lr": args.head_lr}],
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
        for feat, hidx, y in train_dl:
            feat, y = feat.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                emb = model.embed(feat)
                loss = 0.0
                for j, h in enumerate(head_names):
                    m = (hidx == j).to(device)
                    if m.any():
                        loss = loss + loss_fn(model.heads[h](emb[m]).float(), y[m]) * m.float().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.detach())
        scores = {}
        for s in sources:
            if not data[s]["val"]:
                continue
            dl, nr = loader(s, "val")
            h = HEAD_OF[s]
            scores[s] = tuned(evaluate(model, dl, h, heads[h], nr, device), heads[h])[1]
        mean = float(np.mean(list(scores.values()))) if scores else 0.0
        worst = float(min(scores.values())) if scores else 0.0
        score = worst if args.select == "min" else mean
        history.append({"epoch": epoch, "train_loss": round(total / max(1, len(train_dl)), 4),
                        "val": {k: round(v, 4) for k, v in scores.items()}, "val_mean": round(mean, 4),
                        "val_min": round(worst, 4)})
        print(f"epoch {epoch}: loss {history[-1]['train_loss']}  val " +
              "  ".join(f"{k} {v:.3f}" for k, v in scores.items()) + f"  mean {mean:.3f}  min {worst:.3f}  ({time.time() - t0:.0f} s)")
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(), "heads": heads, "head_of": HEAD_OF,
                        "window_s": args.window, "accel_as": args.accel_as}, args.out / "best.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"no val improvement for {args.patience} epochs; stopping")
            break

    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device)["state_dict"])
    pred_rows = []
    results = {"config": {k: str(v) for k, v in vars(args).items()}, "heads": heads,
               "best_epoch": best_epoch, "select": args.select, "best_val_score": best, "history": history, "test": {}}
    print(f"\nbest epoch {best_epoch} ({args.select} val {best:.3f})")
    for s in sources:
        h = HEAD_OF[s]
        dl, nr = loader(s, "test")
        if not nr:
            continue
        res = evaluate(model, dl, h, heads[h], nr, device)
        entry = {"test": public(res)}
        if positive_label(heads[h]) and data[s]["val"]:
            vdl, vnr = loader(s, "val")
            t, _ = tuned(evaluate(model, vdl, h, heads[h], vnr, device), heads[h])
            entry["threshold"] = t
            entry["test_tuned"] = public(evaluate(model, dl, h, heads[h], nr, device, threshold=t))
        results["test"][s] = entry
        labels = heads[h]
        kp = labels.index(positive_label(labels)) if "threshold" in entry else None
        for path, t, p in zip(paths[s]["test"], res["_true"], res["_prob"]):
            pred = (kp if p[kp] >= entry["threshold"] else 1 - kp) if kp is not None else int(p.argmax())
            pred_rows.append({"source": s, "path": path, "label": labels[t], "pred": labels[pred],
                              **{f"p_{c}": round(float(v), 4) for c, v in zip(labels, p)}})
        main_res = entry.get("test_tuned", entry["test"])
        tag = " (val-tuned threshold)" if "test_tuned" in entry else ""
        print(f"\n[{s}] head {h}{tag}: TEST clip acc {main_res['clip']['accuracy']}  "
              f"macro-F1 {main_res['clip']['macro_f1']}  recall {main_res['clip']['recall']}")
        print(main_res["clip_confusion"])
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    fields = ["source", "path", "label", "pred"] + sorted({k for r in pred_rows for k in r if k.startswith("p_")})
    with open(args.out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pred_rows)
    print(f"\nsaved {args.out / 'best.pt'}, results.json and predictions.csv")


if __name__ == "__main__":
    main()
