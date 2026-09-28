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

**Rail vibrations from draisine passages** (`draisine_vibration`)
- Rail-mounted triaxial accelerometers, 1.6 kHz, 16-bit WAV. 93 complete passes with measured speed, plus hammer-strike and braking recordings. 12 MB, CC-BY-4.0.
- No defect labels: useful as healthy-track background for mixing (see below) and for learning how speed changes the signal.
- https://zenodo.org/records/19851718

**High-Speed Train Bogie Vibration & Fault Diagnosis, Kaggle** (`kaggle_bogie_synthetic`)
- **Synthetic** (simulated) 3-axis vibration of train bogie components with added noise. CC0, ~200 MB.
- Set 1: normal + 6 faults at 80–200 km/h; set 2: normal + 14 single faults at 200 km/h; 1,000 samples per class, 486 points each.
- Faults are in the *vehicle* (air springs, dampers, wheelsets, motors), not the track. Useful only for pre-training a vibration encoder or testing the pipeline end to end; don't count results on it as track-fault accuracy.
- https://www.kaggle.com/datasets/ziya07/high-speed-train-bogie-vibration-and-fault-diagnosis

**Madrid–Barcelona onboard monitoring** (`madrid_bcn_onboard`)
- On-board accelerometer, gyroscope, magnetometer and GPS at 40 Hz along the full route. CC-BY-4.0. Unlabelled.
- 40 Hz only captures low-frequency ride / track-geometry behaviour, not rail-head defects. Use for geometry-type faults (e.g. superelevation) or pre-training.
- https://zenodo.org/records/17607068

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
- Defective / non-defective photos of rails and fasteners, collected by hand in Bangladesh. 2.14 GB. Needs a Kaggle API token.
- Licence "Data files © Original Authors"; cite Eunus et al., "ECARRNet", AI 2024, 5(2):482-503.
- https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection
- Related: fastener and rail subsets https://www.kaggle.com/datasets/ashikadnan/railway-track-fault-detection-dataset2fastener

**FaultSeg: train wheel defects** (`faultseg`)
- Segmentation of wheel, shelling, discoloration, cracks/scratches; COCO/VOC/YOLO/TFRecord formats.
- v1 ~8.5 GB (the full record may be far larger; the script refuses Zenodo records over `--max-gb`, default 20). CC-BY-4.0.
- https://zenodo.org/records/12957455 · paper: https://www.nature.com/articles/s41597-025-04557-0

**RFDD: rail fastener defects** (`rfdd`, manual download)
- 1,350 images at 2048×2021 with 8,100+ annotated fasteners: normal, missing, inverted, displaced, deformed, fractured.
- Licence CC BY-NC-ND 4.0 (non-commercial, no redistribution of modified data).
- https://doi.org/10.57760/sciencedb.msdc.00071 · paper: https://www.nature.com/articles/s41597-026-07851-7

**Railway Track Fault Detection (Bangladesh Railway), Kaggle** (`kaggle_bangladesh_track`)
- Track fault images from Bangladesh Railway; contents not verified (page could not be read). Needs a Kaggle API token.
- https://www.kaggle.com/datasets/ashikadnan/railway-track-fault-detectionbangladesh-railway

**Rail-5k** (not scripted)
- ~5k real-world rail surface images, 13 defect types. Access is by request to the authors.
- https://arxiv.org/abs/2106.14366

## Enlarging the vibration / acoustic data

No large labelled public rail vibration dataset exists (searched Kaggle, Zenodo, Mendeley, IEEE DataPort, Indian sources). In order of payoff:

1. **Window + augment what we have** with `augment_vibration.py`. A 17 s clip becomes 33 one-second windows, and 4 augmented copies each gives 165 per clip: the 720 Pakistan clips become ~24k training windows. Speed perturbation is the most physically meaningful augmentation, since the same defect passed at a different speed shifts its frequencies proportionally. Split train/test by recording, never by window.
   ```bash
   python augment_vibration.py raw/draisine_vibration/**/*.wav --label normal --copies 0 -o windows/healthy.npz
   python augment_vibration.py raw/acoustic_track_pk/<wheel_burnt dir>/*.wav --label wheel_burnt \
       --background windows/healthy.npz -o windows/wheel_burnt.npz   # needs matching sample rates
   ```
2. **Label Rail-VIVID by position.** Match each CSV row's GPS to the 9 anomaly locations in its paper; each anomaly is passed in 20 runs at 4 speeds, giving ~180 labelled passes plus over an hour of healthy track.
3. **Pre-train on unlabelled data, fine-tune on labelled.** Rail-VIVID, draisine and Madrid–Barcelona signals are unlabelled but plentiful; self-supervised pre-training (masked-spectrogram or contrastive) on them, then fine-tuning on the small labelled sets, usually beats training from scratch. Pretrained audio models (AST, BEATs, PANNs) are another starting point.
4. **Generate fault examples.** Once 1–3 are in place, train a diffusion model on fault spectrograms to synthesise extra rare-class samples. Validate that a classifier trained on synthetic + real beats real alone on a real-only test set.
5. **Record your own.** Phone accelerometer + microphone (e.g. the free phyphox app, accelerometer typically 100–500 Hz depending on the phone, audio 44.1–48 kHz) on a train over known defects, noting GPS. The only route to audio, vibration and images of the *same* faults.
6. **Simulate.** Vehicle–track dynamics models can generate axle-box acceleration for squats, joints and wheel flats (see the DLR / TU Delft axle-box work); higher effort, useful if a specific defect type stays scarce.

## Gaps to know about

- No public dataset pairs **microphone audio** with **camera images** of the same defects. Rail-VIVID pairs *accelerometer* vibration with images, which carries the same time-frequency information and is the realistic base for a fused model. The labelled audio sets are separate and can train or pre-train the acoustic branch on its own.
- The labelled acoustic sets are small (hundreds of clips). Expect to lean on pre-trained audio models and augmentation.
- Licences differ; check each before redistributing anything derived from them.
