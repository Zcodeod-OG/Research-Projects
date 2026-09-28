# Rail-V datasets

Public datasets for railroad fault detection from images and acoustic / vibration signals.
Raw data is **not** committed (several sources are tens to hundreds of GB). Run the
download script and the data lands in `raw/`, which is git-ignored.

```bash
cd Rail-V/dataset
python download.py --list                         # what's available
python download.py rail_vivid_vibration corrugation  # recommended starting set
python download.py all --dry-run                  # see everything without downloading
```

## Recommended starting set

| Priority | Source | Why |
|---|---|---|
| 1 | **Rail-VIVID (vibration only)** | ~2 GB of 6-channel accelerometer data at 2 kHz with GPS, over 20 runs of the same track. The biggest vibration source here. |
| 2 | **UPM Corrugation database** | Onboard acoustic + vibration at 4 speeds, CC-BY. Good for the audio branch and for speed-robustness. |
| 3 | **Acoustic track faults (Pakistan)** | Microphone audio with class labels (normal / superelevation / wheel burnt). Small but the closest match to "audio frequency" classification. |
| 4 | Image sets (MUET, RSDDS, Kaggle, FaultSeg) | Pre-train / fine-tune the image branch on labelled defect images. |

## Sources

### Multimodal (vibration + image)

**Rail-VIVID: A Multimodal Railway Vibration and Vision Dataset** (`rail_vivid_vibration`, `rail_vivid_sample`, `rail_vivid`)
- Layout: one folder per run (e.g. `AtoB_20_1/`, direction_speed_repeat) holding a CSV of Timestamp, Channel_1..6 (m/s²), Temperature, Humidity, Latitude, Longitude (75–120 MB each), plus a subfolder of JPG frames. The frames are nearly all of the 114 GB. `Anomaly/` holds FARO `.fls` 3D scans of the anomalies.
- `rail_vivid_vibration` pulls only the 20 CSVs (~2 GB). The CSV rows carry **no fault labels**: anomalies are marked by position, so labels come from matching each row's GPS to the anomaly locations in the paper (allow for the ~15 m drift noted below). Also well suited to self-supervised pre-training or healthy-track anomaly detection.
- 6 accelerometers (Silicon Designs 2012, ±5 g) at 2 kHz, 16-bit; 5 MP mono camera at ~30–40 FPS; GNSS.
- 20 repeated runs over a fixed 1.4 km segment at different speeds and directions.
- 9 ground-truthed track anomalies (e.g. rail joints, divergence/convergence points) with 3D point-cloud scans.
- Per-run CSVs plus JPG folders linked by Unix timestamp. ~114 GB total.
- Caveat from the card: image-to-location alignment can drift up to ~15 m.
- Licence: HF card says CC-BY-4.0; the Scientific Data article says CC BY-NC-ND 4.0. Treat as non-commercial until confirmed.
- https://huggingface.co/datasets/saluslab/Rail-VIVID · paper: https://www.nature.com/articles/s41597-026-07555-y

### Acoustic / vibration

**Corrugation Database, Polytechnic University of Madrid** (`corrugation`)
- Onboard acoustic and vibration records from an instrumented train, runs at 40/50/60/75 km/h.
- 4 `.mat` files + README.pdf, 607 MB. Licence CC-BY-4.0.
- https://zenodo.org/records/16569018

**Acoustic railway track faults, Pakistan Railways** (`acoustic_track_pk`)
- 720 WAV clips (240 per class): normal, superelevation, wheel burnt. 17 s each, 22.05 kHz, 16-bit.
- Two ECM-X7BMP microphones on a cart at ~35 km/h, Sadiq Abad / Khanpur area.
- Papers: Shafique et al. 2021 (https://pmc.ncbi.nlm.nih.gov/articles/PMC8472961/), https://www.mdpi.com/1424-8220/23/16/7018, https://www.nature.com/articles/s41598-025-14763-w.
- The papers say "available on request"; a Google Drive copy is linked from https://github.com/Arehmans/railways. No licence stated; research use only, cite Shafique et al.

**Rail vibrations from draisine passages with speed information** (not scripted yet)
- Zenodo https://zenodo.org/records/19851718. Found in search but the page could not be read (rate-limited), so contents and labels are unverified.

### Images

**Railway Track Surface Faults Dataset, MUET** (`mendeley_track_surface`, manual download)
- 7 classes: grooves, joints, cracks, flakings, shellings, spallings, squats.
- DOI 10.17632/8hxtgyyxrw.2 · https://data.mendeley.com/datasets/8hxtgyyxrw/2
- Data article: https://pmc.ncbi.nlm.nih.gov/articles/PMC10828558/

**RSDDS, Northeastern University** (`rsdds`)
- 113 samples, each with a 2D colour image, depth map and ground-truth defect mask.
- Split zips on the repo's `dataset_link` branch; zip password `neurail`. No licence stated; cite the authors.
- https://github.com/neu-rail-rsdds/rsdds (also anomaly-detection version: https://github.com/neu-rail-rsdds/rail_surface_anomaly_detection)
- Not the same as the older RSDDs Type-I/II set (https://ieee-dataport.org/documents/rsdds).

**Railway Track Fault Detection, Kaggle** (`kaggle_track_faults`)
- Binary defective / non-defective track photos, a few hundred images. Needs a Kaggle API token.
- https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection
- Related: fastener and rail subsets https://www.kaggle.com/datasets/ashikadnan/railway-track-fault-detection-dataset2fastener

**FaultSeg: train wheel defects** (`faultseg`)
- Segmentation of wheel, shelling, discoloration, cracks/scratches; COCO/VOC/YOLO/TFRecord formats.
- v1 ~8.5 GB (the full record may be far larger; the script refuses Zenodo records over `--max-gb`, default 20). CC-BY-4.0.
- https://zenodo.org/records/12957455 · paper: https://www.nature.com/articles/s41597-025-04557-0

**Rail-5k** (not scripted)
- ~5k real-world rail surface images, 13 defect types. Access is by request to the authors.
- https://arxiv.org/abs/2106.14366

## Gaps to know about

- No public dataset pairs **microphone audio** with **camera images** of the same defects. Rail-VIVID pairs *accelerometer* vibration with images, which carries the same time-frequency information and is the realistic base for a fused model. The labelled audio sets are separate and can train or pre-train the acoustic branch on its own.
- The labelled acoustic sets are small (hundreds of clips). Expect to lean on pre-trained audio models and augmentation.
- Licences differ; check each before redistributing anything derived from them.
