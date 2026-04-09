"""Mask conversion utilities and image loading helpers for the AnyChange server."""

from __future__ import annotations

import io
import requests
from fastapi import HTTPException
from google.api_core import exceptions as gcs_exc
import numpy as np
import torch
from torchange.models.segment_any_change.segment_anything.utils.amg import rle_to_mask, MaskData


def maskdata_to_binary_union(md: MaskData, h: int, w: int) -> np.ndarray:
    """Merge all masks into a single binary union. Returns (H, W) uint8 with 0 or 255."""
    if not isinstance(md, MaskData) or len(md["rles"]) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    out = np.zeros((h, w), bool)
    for rle in md["rles"]:
        out |= rle_to_mask(rle).astype(bool)
    return out.astype(np.uint8) * 255


def maskdata_to_label_map(md: MaskData, h: int, w: int) -> np.ndarray:
    """Paint each mask as a unique integer label (1..N). Background is 0."""
    if not isinstance(md, MaskData) or len(md["rles"]) == 0:
        return np.zeros((h, w), dtype=np.int32)
    # paint larger instances first
    try:
        areas = md["areas"]
        if hasattr(areas, "detach"):
            areas = areas.detach().cpu().numpy()
        order = np.argsort(-areas)
    except Exception:
        print("[Warn] areas not found in MaskData, using default order")
        order = np.arange(len(md["rles"]))

    lab = np.zeros((h, w), dtype=np.int32)
    cur = 1
    for i in order:
        m = rle_to_mask(md["rles"][i]).astype(bool)
        paint = m & (lab == 0)
        lab[paint] = cur
        cur += 1
    return lab


def maskdata_to_instance_stack(md: MaskData, h: int, w: int, order: str = "area_desc"):
    """
    Convert MaskData -> (masks, meta)
    masks: (N, H, W) bool
    meta:  dict with areas (N,), boxes (N,4) if present, and scores if present.
    order: "area_desc" (default) sorts with this order; None keeps original order.
    """
    out_masks = []
    areas_out, boxes_out, conf_out, iou_out = [], [], [], []

    if not isinstance(md, MaskData) or len(md["rles"]) == 0:
        masks = np.zeros((0, h, w), dtype=bool)
        meta = {"areas": np.array([], dtype=np.int64)}
        return masks, meta

    # Build index order
    idxs = np.arange(len(md["rles"]))
    if order == "area_desc" and "areas" in md._stats:
        areas = md["areas"]
        if isinstance(areas, torch.Tensor):
            areas = areas.detach().cpu().numpy()
        idxs = np.argsort(-areas)

    # Extract masks/metadata
    for i in idxs:
        m = rle_to_mask(md["rles"][i]).astype(bool)
        out_masks.append(m)

        if "areas" in md._stats:
            a = md["areas"][i]
            if isinstance(a, torch.Tensor):
                a = int(a.item())
            areas_out.append(int(a))
        if "boxes" in md._stats:
            # xyxy → list[4]
            b = md["boxes"][i]
            if isinstance(b, torch.Tensor):
                b = b.detach().cpu().tolist()
            boxes_out.append(b)
        if "change_confidence" in md._stats:
            s = md["change_confidence"][i]
            if isinstance(s, torch.Tensor):
                s = float(s.item())
            conf_out.append(float(s))
        if "iou_preds" in md._stats:
            s = md["iou_preds"][i]
            if isinstance(s, torch.Tensor):
                s = float(s.item())
            iou_out.append(float(s))

    masks = np.stack(out_masks, axis=0)  # (N,H,W) bool
    meta = {}
    if areas_out:
        meta["areas"] = np.array(areas_out, dtype=np.int64)

    if boxes_out:
        meta["boxes"] = np.array(boxes_out, dtype=np.float32)

    if conf_out:
        meta["change_confidence"] = np.array(conf_out, dtype=np.float32)

    if iou_out:
        meta["iou_preds"] = np.array(iou_out, dtype=np.float32)

    return masks, meta



def load_npy_from_bytes(b: bytes) -> np.ndarray:
    """Load a .npy array from raw bytes."""
    return np.load(io.BytesIO(b), allow_pickle=False)


def load_npz_from_bytes(b: bytes, key: str | None = None) -> np.ndarray:
    """Load a single array from .npz bytes. Tries *key*, then 'arr', then first array."""
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        if key and key in z.files:
            return z[key]
        if "arr" in z.files:    # common convention
            return z["arr"]
        # fallback: first array in archive
        return z[z.files[0]]


def select_bands(a: np.ndarray, bands: list | tuple, channel_order: str) -> np.ndarray:
    """Select bands by index (e.g. [0,1,2]) or name (e.g. ['r','g','b'])."""
    if isinstance(bands, (list, tuple)):
        if len(bands) == 3 and all(isinstance(x, str) for x in bands):
            name2idx = {"r": 0, "g": 1, "b": 2}
            bands = [name2idx.get(s.lower(), i) for i, s in enumerate(["r","g","b"])]
        bands = list(map(int, bands))
        if channel_order == "chw":
            return a[bands, ...]
        else:
            return a[..., bands]
    return a


def to_rgb_uint8(a: np.ndarray, *, bands=None, channel_order="hwc") -> np.ndarray:
    """Coerce numpy array into HxWx3 uint8 (drop alpha; pick/trim bands)."""
    if a.ndim == 2:
        a = a[..., None]  # HxW -> HxWx1

    # Accept CHW or HWC
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[0] <= a.shape[-1]:
        # likely CHW; trust explicit override if provided
        if channel_order == "chw":
            a = np.transpose(a, (1, 2, 0))  # -> HWC
    elif a.ndim == 3:
        channel_order = "hwc"

    # Optionally select bands
    if bands is not None:
        a = select_bands(a, bands, channel_order="chw" if a.shape[0] in (1,3,4) and a.ndim==3 and a.shape[0] < a.shape[-1] else "hwc")
        # ensure HWC after selection
        if a.ndim == 3 and a.shape[0] in (1,3,4) and a.shape[0] <= a.shape[-1]:
            a = np.transpose(a, (1,2,0))

    # If we still have more than 3 channels, take first 3. If 4, drop alpha.
    if a.ndim == 3:
        c = a.shape[-1]
        if c == 1:
            a = np.repeat(a, 3, axis=-1)
        elif c == 2:
            a = np.concatenate([a, a[..., :1]], axis=-1)  # pad to 3 (simple fallback)
        elif c >= 4:
            a = a[..., :3]

    # Normalize dtype → uint8
    if a.dtype == np.uint8:
        pass
    elif np.issubdtype(a.dtype, np.floating):
        # assume 0..1 floats; if values >1, clip into 0..255 after scaling by 255
        if np.nanmax(a) <= 1.0:
            a = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            a = np.clip(a, 0.0, 255.0).round().astype(np.uint8)
    elif np.issubdtype(a.dtype, np.integer):
        # scale based on dtype range → 0..255
        rng = np.iinfo(a.dtype)
        scale = 255.0 / float(rng.max if rng.max > 0 else 255)
        a = (a.astype(np.float32) * scale).round()
        a = np.clip(a, 0.0, 255.0).astype(np.uint8)
    else:
        a = a.astype(np.uint8, copy=False)

    # Ensure final shape HxWx3
    if a.ndim != 3 or a.shape[-1] != 3:
        raise HTTPException(400, f"Decoded NumPy array could not be coerced to HxWx3 uint8 (got shape {a.shape}).")
    return a


def fetch_bytes_from_uri(gcs_client: object, uri: str) -> bytes:
    """Fetch raw bytes from a GCS URI or HTTP(S) URL."""
    if uri.startswith("gs://"):
        try:
            bucket_name, blob_name = uri[5:].split("/", 1)
        except ValueError:
            raise HTTPException(400, f"Bad GCS URI: {uri}")
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        try:
            return blob.download_as_bytes()
        except gcs_exc.NotFound:
            raise HTTPException(404, f"GCS object not found: {uri}")
        except gcs_exc.Forbidden:
            raise HTTPException(403, f"GCS access denied for: {uri} (check bucket IAM)")
        except Exception as e:
            raise HTTPException(502, f"GCS download error for {uri}: {e}")
    # http(s)
    r = requests.get(uri, timeout=30)
    r.raise_for_status()
    return r.content
