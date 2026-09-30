# Rail-V vibration / acoustic pipeline

Training code for the vibration and acoustic branch. The plan behind it: the data is enough to
fine-tune a pretrained audio model on the labelled track faults, but test on recordings the model
never saw, never on windows cut from training clips.

Easiest route: open [`colab_vibration.ipynb`](colab_vibration.ipynb) in Colab and run it top to bottom.

| Script | What it does |
|---|---|
| `inspect_data.py DATA` | Prints the downloaded folder layout, WAV rates, CSV headers and `.mat` variables. |
| `prepare_corrugation.py --data DATA` | Cuts the UPM corrugation runs into labelled 1 s clips (microphone and axlebox), using the corrugated sections from the dataset's paper. |
| `build_manifest.py --data DATA --out manifest.csv` | One row per recording with label, group and a fixed grouped train/val/test split plus CV folds. Unzips archives it finds. |
| `baseline_rf.py --manifest manifest.csv [--cv]` | MFCC + spectral statistics per 1 s window, random forest. The number to beat. |
| `train_ast.py --manifest manifest.csv` | Fine-tunes the AudioSet-pretrained Audio Spectrogram Transformer on 5 s windows with on-the-fly augmentation. |
| `common.py` | Loading, resampling, windowing, augmentation, MFCC, metrics. |

## Splits

Scores are reported per clip (its windows' probabilities averaged) and per window, on the `test`
split, which no script trains or tunes on. Pakistan clips are grouped in blocks of 10 consecutive
file numbers (`--group-block`), since neighbouring clips probably come from the same drive; files
with the same number in different microphone folders share a group. A random window-level split
would leak and inflate the score.

## What each source is used for right now

| Source | Used by | Status |
|---|---|---|
| `acoustic_track_pk` | baseline, AST | Labelled: normal / superelevation / wheel burnt. |
| `corrugation_mic`, `corrugation_axle` | baseline, AST | Corrugated vs normal, from `prepare_corrugation.py`. Split by run: 40/50 km/h train, 60 val, 75 test. |
| `draisine_vibration` | manifest only (`pool`) | Healthy track, for the anomaly detector (next step). |
| `rail_vivid_vibration` | manifest only (`pool`) | Unlabelled until rows are matched to anomaly GPS positions. |

Different sensors (microphone vs accelerometer) are not mixed into one class: a "normal" class
drawn from another sensor would teach the model to recognise the sensor, not the fault.

## Corrugation labels

The `.mat` files have no labels: `So_pks` / `Vi_pks` are kilometric positions. The dataset's paper
(Soto-Ocampo et al., *Mathematics* 2025, 13(17), 2815) reports two sections with severe corrugation,
confirmed by visual inspection: km 4.855-4.914 and 5.239-5.326. Clips at least half inside those are
`corrugated`; clips at least 20 m clear of them are `normal` (really "not reported corrugated"); the
rest are dropped. That's ~146 m of corrugated track, so only a few dozen corrugated clips over four
runs, and every one comes from the same two sites. Scores here show whether the model generalises
across speed, not across track.

## Choices

- **AST instead of BEATs**: AST's AudioSet checkpoint loads from Hugging Face directly in Colab;
  BEATs needs a manual checkpoint download. Same role, similar accuracy on small sets.
- **5 s windows for AST, 1 s for the baseline**: AST works better with more context, and the
  position embeddings are cut to the window length so short windows stay cheap.
- **Class-balanced sampling** in `train_ast.py` (pick a class, then a recording), since corrugated
  clips are a few percent of each run. `--no-balance` turns it off.
- **Augmentation on the fly** (speed ±15 %, gain, 10-30 dB noise, SpecAugment), so each epoch sees
  new variants and nothing large is written to disk.

## Testing

Checked on synthetic WAV data laid out like the real download: manifest, splits, baseline and the
AST training loop (with a tiny random model, `--smoke`) all run end to end, and the position-embedding
cut was verified against the source weights. Not yet run on the real data or the real AST weights.
