#!/usr/bin/env python3
"""Everything the paper needs, from one Colab cell: repeated runs, tables and figures.

1. Repeats each final model with several seeds (default 3), so every score in the
   paper is a mean and spread, not one lucky run. Runs that already finished are
   skipped, so the cell can be re-run after a Colab disconnect.
2. Collects those results with the multi-task runs and the Rail-VIVID detector
   into tables (LaTeX) and figures (PDF for the paper, PNG to look at).
3. Zips the lot into RUNS/paper/paper_results.zip.

    python paper_results.py --runs /content/drive/MyDrive/rail-v-runs --manifest $RUNS/manifest.csv
    python paper_results.py --runs ... --manifest ... --no-train   # tables and figures only

Output in RUNS/paper/: runs/<job>_s<seed>/, figures/, tables/, summary.json.
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import read_manifest

# job, script, extra args. One forest and, where it is the better model, one AST per task.
JOBS = [
    ("surface_rf", "baseline_rf.py", ["--source", "acoustic_track_pk"]),
    ("surface_ast", "train_ast.py", ["--source", "acoustic_track_pk"]),
    ("axle_rf", "baseline_rf.py", ["--source", "corrugation_axle", "--fs", "4000"]),
    ("mic_rf", "baseline_rf.py", ["--source", "corrugation_mic"]),
    ("mic_ast", "train_ast.py", ["--source", "corrugation_mic", "--window", "1", "--hop", "0.5"]),
    ("joint_rf", "baseline_rf.py", ["--source", "vivid_joint", "--fs", "2000", "--window", "1", "--hop", "0.5"]),
]

# fault -> (paper name, manifest source, forest job, AST job or None, model the predictor uses)
FAULTS = [
    ("surface", "Rail surface", "acoustic_track_pk", "surface_rf", "surface_ast", "surface_ast"),
    ("corr_axle", "Corrugation (axle)", "corrugation_axle", "axle_rf", None, "axle_rf"),
    ("corr_mic", "Corrugation (mic)", "corrugation_mic", "mic_rf", "mic_ast", "mic_ast"),
    ("joint", "Joint / switch", "vivid_joint", "joint_rf", None, "joint_rf"),
]

# multi-task runs as the notebook named them, in order
MULTITASK = [("Run 1", "multitask"), ("Run 2", "multitask_v2"), ("Run 3", "multitask_v3"), ("Run 4", "multitask_v4")]

# reference categorical palette, slots in fixed order (validated: CVD-safe on white)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def run_jobs(runs, manifest, seeds, smoke=False):
    out = runs / "paper" / "runs"
    for seed in seeds:
        for job, script, extra in JOBS:
            d = out / f"{job}_s{seed}"
            if (d / "results.json").exists():
                print(f"skip {d.name} (done)")
                continue
            cmd = [sys.executable, script, "--manifest", str(manifest), "--out", str(d), "--seed", str(seed)] + extra
            if smoke and script == "train_ast.py":
                cmd += ["--smoke", "--epochs", "1", "--workers", "0"]
            print(f"\n=== {d.name}: {' '.join(cmd[1:])}", flush=True)
            subprocess.run(cmd, check=True)


def main_clip(res):
    """The clip-level result the paper reports: val-tuned threshold for two-class tasks."""
    return (res.get("test_tuned") or res["test"])["clip"], (res.get("test_tuned") or res["test"])["clip_confusion"]


def parse_confusion(text, labels):
    rows = text.strip().splitlines()[1:]
    return np.array([[int(v) for v in r.split()[-len(labels):]] for r in rows])


def collect(runs):
    """scores[job] = list of per-seed dicts (macro_f1, accuracy, recall, confusion, history)."""
    scores = defaultdict(list)
    for d in sorted((runs / "paper" / "runs").glob("*_s*")):
        f = d / "results.json"
        if not f.exists():
            continue
        job = d.name.rsplit("_s", 1)[0]
        res = json.loads(f.read_text())
        clip, conf = main_clip(res)
        labels = list(clip["recall"])
        scores[job].append({"macro_f1": clip["macro_f1"], "accuracy": clip["accuracy"], "recall": clip["recall"],
                            "labels": labels, "confusion": parse_confusion(conf, labels).tolist(),
                            "history": res.get("history", [])})
    return scores


def stat(vals):
    v = np.array(vals, float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
            "n": int(len(v)), "values": [round(float(x), 4) for x in v]}


PM = r"$\pm$"


def fmt(s):
    return f"{s['mean']:.3f} {PM} {s['std']:.3f}" if s["n"] > 1 else f"{s['mean']:.3f}"


def multitask_scores(runs):
    out = {}
    for name, folder in MULTITASK:
        f = runs / folder / "results.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        out[name] = {s: main_clip(e)[0]["macro_f1"] for s, e in res.get("test", {}).items()}
    return out


def dataset_table(manifest):
    rows = [r for r in read_manifest(manifest) if r["split"] in ("train", "val", "test")]
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[(r["source"], r["label"])][r["split"]] += 1
    groups = defaultdict(set)
    for r in rows:
        groups[r["source"]].add(r["group"])
    fs = {r["source"]: int(r["fs"]) for r in rows}
    return counts, groups, fs


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
    return plt


def save(fig, figdir, name):
    fig.savefig(figdir / f"{name}.pdf")
    fig.savefig(figdir / f"{name}.png", dpi=200)


def fig_scores(plt, summary, figdir):
    """Forest vs AST per fault, mean with min-max of the seeds."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.arange(len(FAULTS))
    w = 0.36
    for k, (kind, color, label) in enumerate((("forest", BLUE, "Random forest"), ("ast", ORANGE, "AST"))):
        labelled = False
        for i, f in enumerate(FAULTS):
            s = summary["faults"][f[0]].get(kind)
            if not s:
                continue
            xi = x[i] + (k - 0.5) * w
            ax.bar(xi, s["mean"], w - 0.04, color=color, label=None if labelled else label)
            labelled = True
            if s["n"] > 1:
                ax.errorbar(xi, s["mean"], yerr=[[s["mean"] - min(s["values"])], [max(s["values"]) - s["mean"]]],
                            color=INK, lw=0.8, capsize=2)
            ax.text(xi, max(s["values"]) + 0.015, f"{s['mean']:.2f}", ha="center", va="bottom", fontsize=6.5,
                    color=INK)
    ax.set_xticks(x, [f[1].replace(" (", "\n(") for f in FAULTS])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Test macro-F1")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7)
    save(fig, figdir, "scores")
    plt.close(fig)


def fig_confusions(plt, summary, figdir):
    """Row-normalised confusion matrix of the predictor's model per fault, summed over seeds."""
    fig, axes = plt.subplots(1, len(FAULTS), figsize=(7.2, 2.2), layout="constrained")
    for ax, f in zip(axes, FAULTS):
        c = summary["faults"][f[0]].get("chosen_confusion")
        ax.set_title(f[1], fontsize=8, color=INK)
        ax.grid(False)
        if c is None:
            ax.axis("off")
            continue
        m, labels = np.array(c["matrix"], float), c["labels"]
        norm = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
        short = [lab.replace("superelevation", "superelev.").replace("wheel_burnt", "wheel burn") for lab in labels]
        ax.set_xticks(range(len(labels)), short, rotation=30, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(labels)), short, fontsize=6.5)
        ax.set_xlabel("predicted", fontsize=7)
    axes[0].set_ylabel("true", fontsize=7)
    save(fig, figdir, "confusion")
    plt.close(fig)


def fig_curves(plt, scores, figdir):
    """Validation macro-F1 per epoch for each AST, one line per seed."""
    jobs = [j for j in ("surface_ast", "mic_ast") if scores.get(j)]
    if not jobs:
        return
    fig, axes = plt.subplots(1, len(jobs), figsize=(3.5 * len(jobs) / 1.4, 2.0), squeeze=False)
    for ax, j in zip(axes[0], jobs):
        for s in scores[j]:
            h = s["history"]
            ax.plot([e["epoch"] for e in h], [e["val_clip_macro_f1"] for e in h], color=BLUE, lw=1.2, alpha=0.8,
                    marker="o", ms=3)
        ax.set_title({"surface_ast": "Rail surface (AST)", "mic_ast": "Corrugation, mic (AST)"}[j], fontsize=8)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)
    axes[0][0].set_ylabel("Validation macro-F1")
    save(fig, figdir, "training_curves")
    plt.close(fig)


def fig_multitask(plt, summary, figdir):
    """Each shared-model run against the separate model, one panel per fault."""
    mt = summary["multitask"]
    if not mt:
        return
    names = list(mt)
    lowest = min([v for n in names for v in mt[n].values()] + [0.45])
    fig, axes = plt.subplots(1, len(FAULTS), figsize=(7.2, 1.9), sharey=True)
    for ax, f in zip(axes, FAULTS):
        y = [mt[n].get(f[2], np.nan) for n in names]
        ax.plot(range(len(names)), y, color=BLUE, lw=2, marker="o", ms=5, label="Shared model")
        ref = summary["faults"][f[0]]["chosen"]["mean"]
        ax.axhline(ref, color=ORANGE, lw=1.5, ls="--", label="Separate model")
        ax.set_title(f[1], fontsize=8)
        ax.set_xticks(range(len(names)), [n.replace("Run ", "") for n in names])
        ax.set_xlabel("shared-model run")
        ax.set_ylim(np.floor((lowest - 0.05) * 10) / 10, 1.02)
    axes[0].set_ylabel("Test macro-F1")
    axes[0].legend(loc="lower left", fontsize=6.5)
    save(fig, figdir, "multitask_runs")
    plt.close(fig)


def write_tables(summary, counts, groups, fs, tabdir):
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Dataset & Class & Train & Val & Test & Groups & Rate (Hz) \\", r"\midrule"]
    for source in sorted({s for s, _ in counts}):
        first = True
        for (s, lab), c in sorted(counts.items()):
            if s != source:
                continue
            name = source.replace("_", r"\_") if first else ""
            lines.append(f"{name} & {lab.replace('_', ' ')} & {c['train']} & {c['val']} & {c['test']} & "
                         f"{len(groups[source]) if first else ''} & {fs[source] if first else ''} \\\\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (tabdir / "datasets.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Fault & Random forest & AST & Predictor uses \\", r"\midrule"]
    for key, name, *_ in FAULTS:
        f = summary["faults"][key]
        rf = fmt(f["forest"]) if f.get("forest") else "--"
        ast = fmt(f["ast"]) if f.get("ast") else "--"
        lines.append(f"{name} & {rf} & {ast} & {f['chosen_kind']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tabdir / "main_results.tex").write_text("\n".join(lines) + "\n")

    mt = summary["multitask"]
    if mt:
        names = list(mt)
        lines = [r"\begin{tabular}{l" + "c" * (len(names) + 1) + "}", r"\toprule",
                 "Fault & Separate & " + " & ".join(names) + r" \\", r"\midrule"]
        for key, name, source, *_ in FAULTS:
            ref = summary["faults"][key]["chosen"]["mean"]
            vals = [f"{mt[n][source]:.2f}" if source in mt[n] else "--" for n in names]
            lines.append(f"{name} & {ref:.2f} & " + " & ".join(vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (tabdir / "multitask.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-train", action="store_true", help="only build tables and figures from finished runs")
    ap.add_argument("--smoke", action="store_true", help="tiny random ASTs, for testing the code only")
    args = ap.parse_args()

    if not args.no_train:
        run_jobs(args.runs, args.manifest, range(args.seeds), args.smoke)

    paper = args.runs / "paper"
    figdir, tabdir = paper / "figures", paper / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    scores = collect(args.runs)
    summary = {"faults": {}, "multitask": multitask_scores(args.runs)}
    for key, name, source, rf_job, ast_job, chosen in FAULTS:
        f = {"name": name, "source": source}
        if scores.get(rf_job):
            f["forest"] = stat([s["macro_f1"] for s in scores[rf_job]])
        if ast_job and scores.get(ast_job):
            f["ast"] = stat([s["macro_f1"] for s in scores[ast_job]])
        f["chosen_kind"] = "AST" if chosen.endswith("_ast") else "Random forest"
        f["chosen"] = f.get("ast" if chosen.endswith("_ast") else "forest", {"mean": float("nan"), "n": 0})
        if scores.get(chosen):
            runs_ = scores[chosen]
            f["chosen_confusion"] = {"labels": runs_[0]["labels"],
                                     "matrix": np.sum([s["confusion"] for s in runs_], 0).tolist()}
            f["chosen_recall"] = {c: stat([s["recall"][c] for s in runs_]) for c in runs_[0]["labels"]}
        summary["faults"][key] = f

    det = args.runs / "vivid_detect" / "results.json"
    if det.exists():
        summary["vivid_detector"] = json.loads(det.read_text())
    for src, dst in (("vivid_detect/detect.png", "vivid_detect.png"), ("vivid_sweep/sweep_best.png", "vivid_sweep.png"),
                     ("vivid_diag/impact_heatmap.png", "vivid_impacts.png"), ("predictor_check.json", "predictor_check.json")):
        if (args.runs / src).exists():
            shutil.copy(args.runs / src, (figdir if dst.endswith(".png") else paper) / dst)

    counts, groups, fs = dataset_table(args.manifest)
    plt = setup_mpl()
    fig_scores(plt, summary, figdir)
    fig_confusions(plt, summary, figdir)
    fig_curves(plt, scores, figdir)
    fig_multitask(plt, summary, figdir)
    write_tables(summary, counts, groups, fs, tabdir)
    (paper / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\nTest macro-F1 (mean ± std over seeds)")
    for key, name, *_ in FAULTS:
        f = summary["faults"][key]
        parts = [f"{k} {fmt(f[k]).replace(PM, '±')}" for k in ("forest", "ast") if f.get(k)]
        print(f"  {name:20s} " + "   ".join(parts) + f"   -> predictor uses {f['chosen_kind']}")
    # zip everything except the model weights, which are large and not needed for writing
    tmp = Path("/tmp/paper_results")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(paper, tmp, ignore=shutil.ignore_patterns("runs", "*.zip"))
    for job_dir in (paper / "runs").glob("*_s*"):
        if (job_dir / "results.json").exists():
            (tmp / "runs" / job_dir.name).mkdir(parents=True, exist_ok=True)
            shutil.copy(job_dir / "results.json", tmp / "runs" / job_dir.name / "results.json")
    zip_path = shutil.make_archive(str(paper / "paper_results"), "zip", tmp)
    print(f"\nfigures in {figdir}, tables in {tabdir}\nupload this file: {zip_path}")


if __name__ == "__main__":
    main()
