"""Contract self-test -- builds synthetic dummy datasets in a temp dir and runs
every loader through ``__getitem__`` -> :class:`SkyLensCollator`.

Run with ``python -m skylens_model.tests.test_datasets`` (needs ``src`` on the
path). It asserts the exact shapes/dtypes documented in ``README.md``; no real
data is required.
"""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch

from skylens_model.datasets import (
    LLVIP,
    SARD,
    AIResQ,
    DangerClass,
    Flame3Pairs,
    FlameSegmentation,
    RescueNetSegmentation,
    VisDronePerson,
)
from skylens_model.utils.collate import SkyLensCollator

H, W, N = 64, 96, 4
rng = np.random.default_rng(0)


def _img(path: Path, h=H, w=W):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _lbl(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


def _txt(path: Path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_flame(root: Path):
    for i in range(N):
        _img(root / "Images" / f"f{i:03d}.jpg")
        m = np.zeros((H, W), np.uint8)
        m[10:30, 10:40] = 255
        _lbl(root / "Masks" / f"f{i:03d}.png", m)


def make_flame3(root: Path):
    for i in range(N):
        _img(root / "Corrected FOV Images" / "Fire" / f"q{i:03d}.jpg")
        d = root / "Thermal TIFFs" / "Fire"
        d.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(d / f"q{i:03d}.tiff"),
                         rng.uniform(-5, 400, (H, W)).astype(np.float32))


def make_rescuenet(root: Path):
    for s in ("train", "val"):
        for i in range(N):
            _img(root / s / f"{s}-org-img" / f"{s}{i:03d}.jpg")
            lab = np.zeros((H, W), np.uint8)
            lab[:20] = 2   # building no damage  -> normal
            lab[20:35] = 4  # major damage       -> collapse
            lab[35:50] = 8  # road blocked       -> road_blocked
            lab[50:] = 0    # unlabeled          -> ignore
            _lbl(root / s / f"{s}-label-img" / f"{s}{i:03d}_lab.png", lab)


def make_sard(root: Path):
    for s in ("train", "valid"):
        for i in range(N):
            _img(root / s / "images" / f"s{i:03d}.jpg")
            _txt(root / s / "labels" / f"s{i:03d}.txt",
                 "0 0.5 0.5 0.06 0.12\n0 0.25 0.75 0.04 0.09\n")


def make_visdrone(root: Path):
    base = root / "VisDrone2019-DET-train"
    for i in range(N):
        _img(base / "images" / f"v{i:03d}.jpg")
        _txt(base / "annotations" / f"v{i:03d}.txt",
             "10,10,8,16,1,1,0,0\n40,30,6,12,1,2,0,0\n"   # pedestrian, people
             "50,50,20,20,1,4,0,0\n0,0,5,5,0,1,0,0\n")     # car (drop), ignored (drop)


def make_llvip(root: Path):
    for i in range(N):
        _img(root / "visible" / "train" / f"l{i:03d}.jpg")
        _img(root / "infrared" / "train" / f"l{i:03d}.jpg")
        ann = ET.Element("annotation")
        for x in (10, 40):
            o = ET.SubElement(ann, "object")
            ET.SubElement(o, "name").text = "person"
            bb = ET.SubElement(o, "bndbox")
            for k, v in zip(("xmin", "ymin", "xmax", "ymax"), (x, 10, x + 9, 34), strict=False):
                ET.SubElement(bb, k).text = str(v)
        (root / "Annotations").mkdir(parents=True, exist_ok=True)
        ET.ElementTree(ann).write(root / "Annotations" / f"l{i:03d}.xml")


def make_airesq(root: Path):
    for i in range(N):
        _img(root / "train" / "images" / f"a{i:03d}.jpg")
        _txt(root / "train" / "labels" / f"a{i:03d}.txt", "0 0.4 0.4 0.05 0.10\n")


def check(name, ds, *, expect_mask, expect_boxes, stride=4, validity=False):
    s = ds[0]
    assert set(s) >= {"image", "has_rgb", "has_thermal", "danger_mask", "person_boxes"}
    assert s["image"].ndim == 3, s["image"].shape
    assert (s["danger_mask"] is not None) == expect_mask
    assert (s["person_boxes"] is not None) == expect_boxes
    if expect_mask:
        assert s["danger_mask"].dtype == np.uint8 and s["danger_mask"].shape == (H, W)
    if expect_boxes:
        assert s["person_boxes"].dtype == np.float32 and s["person_boxes"].shape[1] == 4

    coll = SkyLensCollator(person_head_stride=stride, validity_channel=validity)
    batch = coll([ds[i] for i in range(min(3, len(ds)))])
    b = batch["pixel_values"].shape[0]
    c = 5 if validity else 4
    assert tuple(batch["pixel_values"].shape) == (b, c, H, W), batch["pixel_values"].shape
    assert batch["pixel_values"].dtype == torch.float32
    assert tuple(batch["modality_mask"].shape) == (b, 2)
    assert batch["modality_mask"].dtype == torch.bool

    print(f"\n=== {name}  (n={len(ds)}) ===")
    print(f"  image            {s['image'].shape} {s['image'].dtype} "
          f"has_rgb={s['has_rgb']} has_thermal={s['has_thermal']}")
    if expect_mask:
        print(f"  danger_mask      {s['danger_mask'].shape} {s['danger_mask'].dtype} "
              f"unique={np.unique(s['danger_mask']).tolist()}")
    if expect_boxes:
        print(f"  person_boxes     {s['person_boxes'].shape} {s['person_boxes'].dtype}")
    for k, v in batch.items():
        print(f"  [batch] {k:16s} {tuple(v.shape)} {v.dtype}")

    if expect_mask:
        assert "danger_labels" in batch
        assert tuple(batch["danger_labels"].shape) == (b, H, W)
        assert batch["danger_labels"].dtype == torch.int64
    else:
        assert "danger_labels" not in batch, "seg key must be omitted"

    if expect_boxes:
        oh, ow = H // stride, W // stride
        assert tuple(batch["person_heatmap"].shape) == (b, 1, oh, ow)
        assert tuple(batch["person_wh"].shape) == (b, 2, oh, ow)
        assert tuple(batch["person_reg_mask"].shape) == (b, 1, oh, ow)
        for k in ("person_heatmap", "person_wh", "person_reg_mask"):
            assert batch[k].dtype == torch.float32
        hm = batch["person_heatmap"]
        print(f"  heatmap peak={hm.max().item():.4f}  "
              f"centres={int(batch['person_reg_mask'].sum().item())}")
        assert abs(hm.max().item() - 1.0) < 1e-6, "gaussian peak must be exactly 1"
        assert batch["person_reg_mask"].sum().item() > 0
    else:
        assert "person_heatmap" not in batch, "det keys must be omitted"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="skylens_ds_"))
    try:
        make_flame(tmp / "flame")
        make_flame3(tmp / "flame3")
        make_rescuenet(tmp / "rescuenet")
        make_sard(tmp / "sard")
        make_visdrone(tmp / "visdrone")
        make_llvip(tmp / "llvip")
        make_airesq(tmp / "airesq")

        check("FlameSegmentation", FlameSegmentation(tmp / "flame", "train"),
              expect_mask=True, expect_boxes=False)
        check("Flame3Pairs (RGB+thermal)", Flame3Pairs(tmp / "flame3", "train"),
              expect_mask=False, expect_boxes=False, validity=True)
        check("RescueNetSegmentation", RescueNetSegmentation(tmp / "rescuenet", "train"),
              expect_mask=True, expect_boxes=False)
        check("SARD", SARD(tmp / "sard", "train"), expect_mask=False, expect_boxes=True)
        check("VisDronePerson", VisDronePerson(tmp / "visdrone", "train"),
              expect_mask=False, expect_boxes=True)
        check("LLVIP (rgbt)", LLVIP(tmp / "llvip", "train"),
              expect_mask=False, expect_boxes=True, validity=True)
        check("AIResQ (thermal-only)", AIResQ(tmp / "airesq", "train"),
              expect_mask=False, expect_boxes=True)

        # -- class mapping ------------------------------------------------
        rn = RescueNetSegmentation(tmp / "rescuenet", "train")[0]["danger_mask"]
        assert rn[0, 0] == DangerClass.NORMAL
        assert rn[25, 0] == DangerClass.COLLAPSE
        assert rn[40, 0] == DangerClass.ROAD_BLOCKED
        assert rn[60, 0] == DangerClass.IGNORE
        print("\nRescueNet 11->4 mapping OK "
              "(no-damage->0, major-damage->2, road-blocked->3, unlabeled->255)")

        # -- VisDrone person filtering ------------------------------------
        vd = VisDronePerson(tmp / "visdrone", "train")[0]["person_boxes"]
        assert vd.shape == (2, 4), vd.shape
        print(f"VisDrone person filter OK (4 rows -> {vd.shape[0]} person boxes; "
              "car + score=0 dropped)")

        # -- thermal reserved-zero encoding -------------------------------
        px = SkyLensCollator()([Flame3Pairs(tmp / "flame3", "train")[0]])["pixel_values"]
        t = px[0, 3]
        assert float(t.min()) >= 0.1 - 1e-6 and float(t.max()) <= 1.0 + 1e-6
        rgbonly = SkyLensCollator()([SARD(tmp / "sard", "train")[0]])["pixel_values"]
        assert float(rgbonly[0, 3].abs().max()) == 0.0
        print(f"Thermal channel range [{t.min():.3f}, {t.max():.3f}] "
              "(0.0 reserved for 'absent'; RGB-only sample has exact 0 thermal)")

        # -- mixed batch: seg sample + det sample --------------------------
        mixed = SkyLensCollator()([FlameSegmentation(tmp / "flame", "train")[0],
                                   SARD(tmp / "sard", "train")[0]])
        print("Mixed seg+det batch keys:", sorted(mixed))
        assert {"danger_labels", "person_heatmap"} <= set(mixed)
        assert int((mixed["danger_labels"][1] == 255).all()) == 1, \
            "det-only sample must be all-ignore in the seg target"

        # -- manual-download error ----------------------------------------
        from skylens_model.datasets import ManualDownloadRequired

        try:
            RescueNetSegmentation(tmp / "nope", "train", download=True)
        except ManualDownloadRequired as exc:
            print("\n--- ManualDownloadRequired sample message ---")
            print(str(exc))
        else:
            raise AssertionError("expected ManualDownloadRequired")

        print("\nALL CONTRACT CHECKS PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dataset_contracts() -> None:
    """pytest 진입점 — 계약 검증 전체를 돌린다."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
