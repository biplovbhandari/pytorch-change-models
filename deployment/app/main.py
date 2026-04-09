"""FastAPI server for AnyChange inference. See deployment/README.md for API docs."""

from __future__ import annotations

import os, base64, io, numpy as np
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from predictor import Predictor
from config import DEFAULT_MASK_MODE, VALID_MASK_MODES
from utils import maskdata_to_instance_stack, maskdata_to_binary_union, \
                  maskdata_to_label_map, load_npy_from_bytes, load_npz_from_bytes, \
                  to_rgb_uint8, fetch_bytes_from_uri

app = FastAPI()
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

predictor = Predictor()

_gcs_client = None


def get_gcs_client():
    """Lazily initialize the GCS client on first use."""
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def _load_img(spec: dict) -> np.ndarray:
    """
    Accepts:
      - {'b64': '<base64 PNG/JPEG bytes>'}
      - {'uri': 'gs://...|https://...|http://...'}  (PNG/JPEG/NPY/NPZ)
      - {'npy_b64': '<base64 .npy>'}
      - {'npz_b64': '<base64 .npz>', 'key': 'optional-key'}
      - {'array': {'format': 'npy'|'npz', 'b64': '...', 'key': 'optional-key'}}

    Optional hints (for NP* inputs):
      - 'channel_order': 'hwc' (default) or 'chw'
      - 'bands': [r,g,b] indices or ['r','g','b'] (names -> [0,1,2])

    Returns HxWx3 uint8 RGB ndarray.
    """
    bands = spec.get("bands")
    channel_order = str(spec.get("channel_order", "hwc")).lower()

    # --- Direct base64 image (PNG/JPEG) ---
    if "b64" in spec:
        img_bytes = base64.b64decode(spec["b64"])
        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    # --- Base64 NPY/NPZ shortcuts ---
    if "npy_b64" in spec:
        arr = load_npy_from_bytes(base64.b64decode(spec["npy_b64"]))
        return to_rgb_uint8(arr, bands=bands, channel_order=channel_order)

    if "npz_b64" in spec:
        arr = load_npz_from_bytes(base64.b64decode(spec["npz_b64"]), key=spec.get("key"))
        return to_rgb_uint8(arr, bands=bands, channel_order=channel_order)

    # --- Generic 'array' envelope ---
    if "array" in spec and isinstance(spec["array"], dict):
        fmt = str(spec["array"].get("format", "")).lower()
        b64 = spec["array"].get("b64")
        if not b64:
            raise HTTPException(400, "array.b64 is required when using 'array' input.")
        raw = base64.b64decode(b64)
        if fmt == "npy":
            arr = load_npy_from_bytes(raw)
        elif fmt == "npz":
            arr = load_npz_from_bytes(raw, key=spec["array"].get("key"))
        else:
            raise HTTPException(400, "array.format must be 'npy' or 'npz'.")
        return to_rgb_uint8(arr, bands=bands, channel_order=channel_order)

    # --- URI (GCS or HTTP[S]) ---
    if "uri" in spec:
        uri = spec["uri"]
        raw = fetch_bytes_from_uri(get_gcs_client(), uri)
        lower = uri.lower()
        if lower.endswith(".npy"):
            arr = load_npy_from_bytes(raw)
            return to_rgb_uint8(arr, bands=bands, channel_order=channel_order)
        if lower.endswith(".npz"):
            arr = load_npz_from_bytes(raw, key=spec.get("key"))
            return to_rgb_uint8(arr, bands=bands, channel_order=channel_order)
        # assume raster image if not npy/npz
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

    raise HTTPException(400, "Each image must include one of: {'b64':...} | {'uri':...} | {'npy_b64':...} | {'npz_b64':...} | {'array':{format,b64}}")


def _extract_query(inst: dict) -> dict | list | None:
    """
    Accept either:
      - inst['point']  = {"xy":[x,y], "temporal":1|2}
      - inst['points'] = [ {"xy":[x,y], "temporal":1|2}, ... ] or [[x,y,t], ...]
    Returns None (auto), a dict (single), or a list (multi).
    """
    if "point" in inst and inst["point"] is not None:
        return inst["point"]
    if "points" in inst and inst["points"] is not None:
        return inst["points"]
    return None


@app.get(HEALTH_ROUTE, status_code=200)
async def health():
    return {"status": "ok"}


@app.post(PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()

    allowed_param_keys = {
        "points_per_side",
        "stability_thresh",
        "change_conf_thresh",
        "object_sim_thresh",
        "use_normalized_feature",
        "bitemporal_match",
    }

    # Global default: support Vertex ("parameters") and local (top-level)
    req_params = (body.get("parameters") or {})
    global_mode = (
        body.get("mask_mode")
        or req_params.get("mask_mode")
        or DEFAULT_MASK_MODE
    )

    def _sanitize_mode(mm: str, fallback: str) -> str:
        if isinstance(mm, str):
            m = mm.lower()
            if m in VALID_MASK_MODES:
                return m
        return fallback

    instances = body.get("instances", [])
    preds = []
    for inst in instances:
        inst_params = {k: req_params.get(k) for k in allowed_param_keys if k in req_params}
        inst_params.update({k: v for k, v in (inst.get("parameters") or {}).items() if k in allowed_param_keys})

        # Per-instance override (optional)
        inst_mode = _sanitize_mode(inst.get("mask_mode") or req_params.get("mask_mode") or global_mode, global_mode)

        img1 = _load_img(inst["img1"])
        img2 = _load_img(inst["img2"])

        query = _extract_query(inst)

        print(f"Processing instance with mode: {inst_mode}")
        print(f"Query: {query}, mask_mode: {inst_mode}, params: {inst_params}")

        mask_data = predictor.predict(
            img1, img2,
            query=query,
            params=inst_params,
        )
        count = int(getattr(predictor, "last_instances_count", 0))
        params_used = getattr(predictor, "last_used_params", None)

        h, w = img1.shape[:2]

        if inst_mode == "instances":
            # Build (N,H,W) + meta and pack
            masks, meta = maskdata_to_instance_stack(mask_data, h, w, order="area_desc")
            buf = io.BytesIO()
            np.savez_compressed(buf, masks=masks.astype(np.uint8), **meta, allow_pickle=False)
            preds.append({
                "mask_mode": "instances",
                "mask_array": {"format": "npz", "b64": base64.b64encode(buf.getvalue()).decode("utf-8")},
                "shape": list(masks.shape),     # [N,H,W]
                "dtype": "uint8",
                "mode": "auto" if query is None else ("single_point" if isinstance(query, dict) else "multi_points"),
                "instances_count": count,
                "used_parameters": params_used,
            })

        elif inst_mode == "union":
            union = maskdata_to_binary_union(mask_data, h, w)  # uint8 0/255
            buf = io.BytesIO()
            np.savez_compressed(buf, arr=union, allow_pickle=False)
            preds.append({
                "mask_mode": "union",
                "mask_array": {"format": "npz", "b64": base64.b64encode(buf.getvalue()).decode("utf-8")},
                "shape": list(union.shape),     # [H,W]
                "dtype": "uint8",
                "mode": "auto" if query is None else ("single_point" if isinstance(query, dict) else "multi_points"),
                "instances_count": count,
                "used_parameters": params_used,
            })

        else:  # "label"
            label = maskdata_to_label_map(mask_data, h, w).astype(np.int32)  # 0..N
            buf = io.BytesIO()
            np.savez_compressed(buf, arr=label, allow_pickle=False)
            preds.append({
                "mask_mode": "label",
                "mask_array": {"format": "npz", "b64": base64.b64encode(buf.getvalue()).decode("utf-8")},
                "shape": list(label.shape),     # [H,W]
                "dtype": "int32",
                "mode": "auto" if query is None else ("single_point" if isinstance(query, dict) else "multi_points"),
                "instances_count": count,
                "used_parameters": params_used,
            })

        print(f"[Predict] Parameters used: {params_used}, Instances count: {count}")

    return {"predictions": preds}
