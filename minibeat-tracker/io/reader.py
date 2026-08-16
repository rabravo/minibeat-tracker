"""Image I/O — tifffile-based reader, mirrors LoadAndResizeImage.m."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import tifffile
import cv2


def scan_tif_folder(folder: Path) -> list[Path]:
    """Return sorted list of .tif/.tiff files in folder."""
    files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    # avoid duplicates when both extensions exist
    seen, result = set(), []
    for f in sorted(set(files), key=lambda p: p.name):
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def load_and_resize(
    path: Path | str,
    scale_width: int = 0,
    mb_size: int = 16,
) -> np.ndarray:
    """Load a single grayscale frame and optionally scale to target width.

    scale_width=0 returns the image at its native resolution.
    The resized width is snapped to the nearest multiple of mb_size.
    """
    try:
        img = tifffile.imread(str(path))
    except Exception:
        img = None

    if img is None:
        # Fallback: OpenCV handles YCbCr-subsampled TIFFs and other formats tifffile rejects
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"Cannot read image: {path}")
        img = raw

    # Collapse to 2D grayscale
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGRA2GRAY)
        else:
            img = img[..., 0]
    img = img.astype(np.float64)

    if scale_width > 0:
        h, w = img.shape
        scale = scale_width / w
        new_w = int(round(w * scale / mb_size) * mb_size)
        new_h = int(round(h * scale / mb_size) * mb_size)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return img


def load_bioformats_meta(path: Path) -> dict:
    """Read pixel size and framerate from OME-TIFF metadata (best-effort)."""
    meta = {"pixel_size_um": 0.0, "framerate": 0.0, "frame_count": 0}
    try:
        with tifffile.TiffFile(str(path)) as tf:
            meta["frame_count"] = len(tf.pages)
            if tf.ome_metadata:
                import xml.etree.ElementTree as ET
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                root = ET.fromstring(tf.ome_metadata)
                px = root.find(".//ome:Pixels", ns)
                if px is not None:
                    meta["pixel_size_um"] = float(px.get("PhysicalSizeX", 0))
                    time_inc = float(px.get("TimeIncrement", 0))
                    if time_inc > 0:
                        meta["framerate"] = round(1.0 / time_inc, 4)
    except Exception:
        pass
    return meta
