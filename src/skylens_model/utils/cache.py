"""리사이즈된 샘플을 디스크에 캐시한다.

원본은 데이터셋마다 3000x4000(RescueNet) · 1024x1280(LLVIP) 등으로 크고,
매 에폭 디코딩+리사이즈를 반복하면 GPU 가 놀고 데이터 로딩이 병목이 된다.
한 번만 리사이즈해 두고 이후에는 작은 파일만 읽는다.

저장 형식 (샘플당):
    <i>.jpg     RGB (JPEG q95)
    <i>_t.png   열화상 8bit — 있을 때만
    <i>_m.png   위험구역 마스크 — 있을 때만 (무손실)
    index.json  박스 · 모달리티 플래그 · 파일 유무

`ResizedCache` 는 항상 **float32 [0, 1]** 이미지를 돌려준다. 원본이 uint8[0,255]
이든 float32[0,1] 이든 이 형태로 수렴하며, collator 의 정규화 규칙과 일치한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from torch.utils.data import Dataset

__all__ = ["resize_sample", "build_resized_cache", "ResizedCache"]

# RGB 는 JPEG q95. 목적이 **디코딩 속도**라 무손실 포맷을 쓰지 않는다:
#   jpg95   163KB  2.8ms  평균오차 1~2.5/255
#   webp무손실 538KB  7.7ms  오차 0
#   png     580KB  8.8ms  오차 0
# 원본도 이미 JPEG 이고, 학습 시 밝기·대비 증강이 이보다 큰 섭동을 준다.
# 반면 마스크(라벨)와 열화상은 무손실이어야 하므로 PNG 로 남긴다.
_JPEG_QUALITY = 95


def resize_sample(sample: dict, size: int) -> dict:
    """이미지·마스크·박스를 `size x size` 로 맞춘다.

    마스크는 nearest(라벨 보존), 이미지는 area(축소에 적합) 보간을 쓴다.
    """
    image = np.asarray(sample["image"])
    h, w = image.shape[:2]
    out = dict(sample)

    out["image"] = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    mask = sample.get("danger_mask")
    if mask is not None:
        out["danger_mask"] = cv2.resize(
            np.asarray(mask), (size, size), interpolation=cv2.INTER_NEAREST
        )

    boxes = sample.get("person_boxes")
    if boxes is not None and len(boxes):
        b = np.asarray(boxes, dtype=np.float32).copy()
        b[:, [0, 2]] *= size / w
        b[:, [1, 3]] *= size / h
        out["person_boxes"] = b
    elif boxes is not None:
        out["person_boxes"] = np.zeros((0, 4), np.float32)
    return out


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """float [0,1] 이든 uint8 이든 uint8 [0,255] 로 만든다."""
    if image.dtype == np.uint8:
        return image
    a = np.asarray(image, dtype=np.float32)
    if a.max() <= 1.5:
        a = a * 255.0
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def build_resized_cache(
    dataset: Dataset,
    cache_dir: str | Path,
    size: int,
    *,
    overwrite: bool = False,
    log_every: int = 500,
) -> Path:
    """`dataset` 을 한 번 훑어 리사이즈본을 `cache_dir` 에 쓴다.

    `dataset` 은 **transforms 없이** 만든 원본이어야 한다(리사이즈를 여기서 한다).
    이미 완성된 캐시가 있으면 건너뛴다.
    """
    cache_dir = Path(cache_dir)
    index_path = cache_dir / "index.json"

    if index_path.is_file() and not overwrite:
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        if meta.get("size") == size and len(meta.get("samples", [])) == len(dataset):
            return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for i in range(len(dataset)):
        s = resize_sample(dataset[i], size)
        img = _to_uint8(np.asarray(s["image"]))
        if img.ndim == 2:
            img = img[:, :, None]

        rgb = img[:, :, :3] if img.shape[2] >= 3 else np.repeat(img[:, :, :1], 3, axis=2)
        # cv2 는 BGR 로 쓰므로 뒤집어 저장하고 읽을 때 되돌린다.
        cv2.imwrite(str(cache_dir / f"{i:06d}.jpg"), rgb[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])

        has_thermal = bool(s.get("has_thermal", False))
        if has_thermal:
            plane = img[:, :, 3] if img.shape[2] >= 4 else img[:, :, 0]
            cv2.imwrite(str(cache_dir / f"{i:06d}_t.png"), plane)

        mask = s.get("danger_mask")
        if mask is not None:
            cv2.imwrite(str(cache_dir / f"{i:06d}_m.png"), np.asarray(mask, np.uint8))

        boxes = s.get("person_boxes")
        records.append({
            "has_rgb": bool(s.get("has_rgb", False)),
            "has_thermal": has_thermal,
            "has_mask": mask is not None,
            "boxes": [] if boxes is None else np.asarray(boxes, np.float32).tolist(),
            "has_boxes": boxes is not None,
        })

        if log_every and (i + 1) % log_every == 0:
            print(f"    {i + 1}/{len(dataset)}", flush=True)

    index_path.write_text(
        json.dumps({"size": size, "samples": records}, ensure_ascii=False),
        encoding="utf-8",
    )
    return cache_dir


class ResizedCache(Dataset):
    """`build_resized_cache` 가 만든 캐시를 읽는다.

    `transforms` 는 **리사이즈를 제외한 증강만** 받는다(크기는 이미 맞춰져 있다).
    """

    def __init__(self, cache_dir: str | Path, transforms: Callable | None = None):
        self.root = Path(cache_dir)
        meta = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.size: int = meta["size"]
        self.records: list[dict] = meta["samples"]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        bgr = cv2.imread(str(self.root / f"{i:06d}.jpg"), cv2.IMREAD_COLOR)
        image = bgr[:, :, ::-1].astype(np.float32) / 255.0

        if r["has_thermal"]:
            t = cv2.imread(str(self.root / f"{i:06d}_t.png"), cv2.IMREAD_GRAYSCALE)
            image = np.concatenate([image, t[:, :, None].astype(np.float32) / 255.0], axis=2)

        mask = None
        if r["has_mask"]:
            mask = cv2.imread(str(self.root / f"{i:06d}_m.png"), cv2.IMREAD_GRAYSCALE)

        boxes = np.asarray(r["boxes"], np.float32).reshape(-1, 4) if r["has_boxes"] else None

        sample = {
            "image": np.ascontiguousarray(image),
            "has_rgb": r["has_rgb"],
            "has_thermal": r["has_thermal"],
            "danger_mask": mask,
            "person_boxes": boxes,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
