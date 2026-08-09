# skylens_model.datasets

Placeholder for datasets the SkyLens models will eventually train/evaluate on.
No data is included in this repository.

## Expected datasets

- **AI-Hub RGB+thermal disaster imagery** — paired RGB and thermal frames of
  disaster/collapse scenarios, used to train the UNet detection backbone's
  4-channel input (person instance head + danger-zone segmentation head).
- **DroneAudioset** (2025) — a drone-based SAR audio dataset with propeller
  noise removed/suppressed, human-voice-labeled. Used for the audio-based
  survivor detection extension (YAMNet environmental sound classification,
  confidence correction for the person head), not part of the main demo.

## Layout (future)

When real data is added, expect a structure roughly like:

```
datasets/
  README.md
  <dataset_name>/
    raw/          # unmodified source files (not committed)
    processed/    # preprocessed/aligned RGB+thermal pairs (not committed)
    manifest.*    # index/metadata files (may be committed)
```

Nothing under `raw/` or `processed/` should be committed — this folder
currently holds documentation only.
