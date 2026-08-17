"""Binary readers/writers for COLMAP sparse models.

We parse ``images.bin`` and ``points3D.bin`` by hand instead of going through
pycolmap. Two reasons:

* gsplat's example loader pins the ``rmbrualla/pycolmap`` fork, which exposes
  ``SceneManager`` but not the ``Reconstruction`` API needed to write a model
  back out. Installing the official pycolmap alongside it clashes on the module
  name (see ``INSTALL.md``).
* Every tool here rewrites *part* of a model and copies the rest untouched.
  Round-tripping through a full reconstruction object would re-derive fields we
  deliberately want to leave alone.

Binary layout (little-endian), COLMAP ``images.bin``::

    uint64  num_images
    repeat:
        uint32  image_id
        double  qw, qx, qy, qz      world->cam rotation, Hamilton quaternion
        double  tx, ty, tz          world->cam translation
        uint32  camera_id
        char[]  name                NUL-terminated
        uint64  num_points2D
        repeat: double x, double y, uint64 point3D_id

``points3D.bin``::

    uint64  num_points
    repeat:
        uint64  point3D_id
        double  x, y, z
        uint8   r, g, b
        double  reprojection_error
        uint64  track_length
        repeat: uint32 image_id, uint32 point2D_idx

COLMAP 4.x also writes ``rigs.bin`` and ``frames.bin``. Nothing here touches
them -- they are copied verbatim, which is safe because gsplat's loader only
reads cameras, images and points.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

__all__ = [
    "Image",
    "Point3D",
    "copy_model_except",
    "qvec_to_rotmat",
    "read_images",
    "read_points3d",
    "rotmat_to_qvec",
    "write_images",
    "write_points3d",
]


class Image:
    """One registered frame. ``qvec``/``tvec`` are world->cam.

    ``point2d_blob`` keeps the 2D observations as raw bytes. No tool here needs
    to interpret them, and passing them through untouched avoids a lossy
    decode/encode round trip.
    """

    __slots__ = ("camera_id", "image_id", "name", "num_points2d", "point2d_blob", "qvec", "tvec")

    def __init__(self, image_id, qvec, tvec, camera_id, name, num_points2d, point2d_blob):
        self.image_id = image_id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.num_points2d = num_points2d
        self.point2d_blob = point2d_blob

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -qvec_to_rotmat(self.qvec).T @ self.tvec


class Point3D:
    """One triangulated point. ``track`` is a list of ``(image_id, point2d_idx)``."""

    __slots__ = ("error_blob", "point_id", "rgb", "track", "xyz")

    def __init__(self, point_id, xyz, rgb, error_blob, track):
        self.point_id = point_id
        self.xyz = xyz
        self.rgb = rgb
        self.error_blob = error_blob
        self.track = track


def qvec_to_rotmat(q) -> np.ndarray:
    """Hamilton quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> Hamilton quaternion (w, x, y, z), normalised.

    Branches on the largest diagonal term so the divisor never approaches zero.
    """
    trace = np.trace(rot)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = [0.25 * s,
             (rot[2, 1] - rot[1, 2]) / s,
             (rot[0, 2] - rot[2, 0]) / s,
             (rot[1, 0] - rot[0, 1]) / s]
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        q = [(rot[2, 1] - rot[1, 2]) / s,
             0.25 * s,
             (rot[0, 1] + rot[1, 0]) / s,
             (rot[0, 2] + rot[2, 0]) / s]
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        q = [(rot[0, 2] - rot[2, 0]) / s,
             (rot[0, 1] + rot[1, 0]) / s,
             0.25 * s,
             (rot[1, 2] + rot[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        q = [(rot[1, 0] - rot[0, 1]) / s,
             (rot[0, 2] + rot[2, 0]) / s,
             (rot[1, 2] + rot[2, 1]) / s,
             0.25 * s]
    qv = np.array(q)
    return qv / np.linalg.norm(qv)


def read_images(path: str | Path) -> list[Image]:
    """Read ``images.bin``."""
    out: list[Image] = []
    with open(path, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            (image_id,) = struct.unpack("<I", f.read(4))
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            (camera_id,) = struct.unpack("<I", f.read(4))
            name = b""
            while (char := f.read(1)) != b"\x00":
                name += char
            (num_points2d,) = struct.unpack("<Q", f.read(8))
            blob = f.read(num_points2d * 24)
            out.append(Image(image_id, qvec, tvec, camera_id, name, num_points2d, blob))
    return out


def write_images(path: str | Path, images: list[Image]) -> None:
    """Write ``images.bin``."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<I", im.image_id))
            f.write(struct.pack("<4d", *im.qvec))
            f.write(struct.pack("<3d", *im.tvec))
            f.write(struct.pack("<I", im.camera_id))
            f.write(im.name + b"\x00")
            f.write(struct.pack("<Q", im.num_points2d))
            f.write(im.point2d_blob)


def read_points3d(path: str | Path) -> list[Point3D]:
    """Read ``points3D.bin``."""
    out: list[Point3D] = []
    with open(path, "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            (point_id,) = struct.unpack("<Q", f.read(8))
            xyz = f.read(24)
            rgb = f.read(3)
            error = f.read(8)
            (track_len,) = struct.unpack("<Q", f.read(8))
            track = [struct.unpack("<II", f.read(8)) for _ in range(track_len)]
            out.append(Point3D(point_id, xyz, rgb, error, track))
    return out


def write_points3d(path: str | Path, points: list[Point3D]) -> None:
    """Write ``points3D.bin``."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pt in points:
            f.write(struct.pack("<Q", pt.point_id))
            f.write(pt.xyz)
            f.write(pt.rgb)
            f.write(pt.error_blob)
            f.write(struct.pack("<Q", len(pt.track)))
            for image_id, point2d_idx in pt.track:
                f.write(struct.pack("<II", image_id, point2d_idx))


def copy_model_except(src: str | Path, dst: str | Path, skip: set[str]) -> None:
    """Copy every model file from ``src`` to ``dst`` except the named ones.

    Used by tools that rewrite one or two files and must carry the rest
    (``cameras.bin``, and on COLMAP 4.x ``rigs.bin``/``frames.bin``) across
    unchanged, or the model will not load.
    """
    import shutil

    src_dir, dst_dir = Path(src), Path(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file() and item.name not in skip:
            shutil.copy2(item, dst_dir / item.name)
