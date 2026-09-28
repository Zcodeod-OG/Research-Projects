# Rail-V modelling approach

## The proposal

Two diffusion models, one trained on audio time-frequency images (spectrograms etc.) and one on
track images, combined in a pipeline to detect faults.

## Assessment

The two-branch idea (one model per modality, then combine) is sound. Using **diffusion models as
the detectors** is the part likely to hurt results:

- Diffusion models are generative. They learn to produce data, not to output a fault label. Using
  one as a classifier means either reconstruction-error anomaly scoring or diffusion-classifier
  tricks, both of which are slow at inference (many denoising steps) and usually less accurate
  than a discriminative model trained on the same labels.
- The public labelled rail datasets are small (hundreds to a few thousand samples). Diffusion
  models need a lot of data to train well; classifiers fine-tuned from pre-trained weights do not.
- For on-track or on-board detection, inference speed matters. A CNN/ViT forward pass is
  milliseconds; diffusion sampling is orders of magnitude slower.

## Recommended instead

1. **Acoustic / vibration branch.** Convert each window to a log-mel spectrogram (plus CQT or
   MFCC channels, which the rail acoustic papers found helpful) and fine-tune a pre-trained audio
   model such as AST, BEATs or PANNs (CNN14). For accelerometer data at 2 kHz, use an STFT or
   wavelet scalogram instead of mel.
2. **Image branch.** Fine-tune a pre-trained CNN/ViT (EfficientNet, ConvNeXt, ViT/DINOv2) for
   classification, or YOLO / a segmentation model if defect location matters.
3. **Fusion.** Start with late fusion (average or learned weights over each branch's
   probabilities). Once paired data is in place, try feature-level fusion with cross-attention
   between the two embeddings. Align the modalities by timestamp / track position.
4. **Where diffusion does help.** Use it to *augment* the scarce fault classes (generate extra
   defect images or spectrograms), or as an unsupervised anomaly detector trained only on healthy
   track, run alongside the classifiers.

## Data reality

No public dataset pairs microphone audio with camera images of the same defects. The closest is
Rail-VIVID (accelerometer vibration + camera, synchronized, with labelled anomalies), which is the
realistic base for training fusion. Labelled microphone datasets exist separately and can train
the acoustic branch. See [dataset/README.md](dataset/README.md).
