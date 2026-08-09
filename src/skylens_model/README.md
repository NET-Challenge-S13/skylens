# skylens_model

Python package that will hold SkyLens's AI/reconstruction model code:

1. **Detection** — a UNet/TransUNet backbone over 4-channel input (RGB + thermal)
   with two heads: a segmentation head for danger zones ("stuff": collapse, fire)
   and an instance head for people. Combined with a Depth-Map raycasting step,
   2D detections are projected into 3D world coordinates.
2. **Reconstruction** — real-time 3D scene reconstruction via 3D Gaussian
   Splatting (3DGS), fed by pose estimation (GLOMAP) + Gaussian training
   (gsplat) + multi-drone fusion (Open3D ICP).

This is currently a **scaffold only**: dataclasses and interface stubs, no
trained weights, no tensor math, no third-party ML dependencies. Real
implementations (numpy/torch/etc.) will fill these in later.

## Folder map

```
skylens_model/
  __init__.py         package version/docstring
  models/
    detection.py       HumanDetector interface: Frame, Detection, HumanDetector
    splat.py            SplatReconstructor interface: SplatChunkSpec, SplatAlign
  datasets/
    README.md           expected datasets (AI-Hub RGB+thermal, DroneAudioset)
  utils/
    geo.py               ENU <-> GPS math, mirrors src/skylens_core/geo.ts
```

## How this plugs into the client

The TypeScript client (`src/skylens_core/`) never talks to this package
directly — it consumes results over the wire protocol defined in
`src/skylens_core/protocol.ts`. This package's job is to eventually *produce*
values that get serialized into those same shapes:

- `HumanDetector.infer()` returns `Detection` objects that map 1:1 onto
  `DetectionResult` in `protocol.ts` (`category`, `gps`, `confidence`, `label`).
- `SplatReconstructor.export_chunk()` returns a `SplatChunkSpec` that maps
  onto `SplatChunk` (`id`, `url`, `align`) in `protocol.ts`, where `align`
  mirrors `SplatAlign` (`anchor?`, `position`, `rotation`, `scale`).

Coordinates are shared via the ENU (East/North/Up) convention defined in
`src/skylens_core/geo.ts` and mirrored in `skylens_model/utils/geo.py` — GPS
in, local meters out, so detections and splat chunks from this package line
up in the same scene frame the client renders.
