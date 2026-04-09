"""AnyChange model wrapper with thread-safe, per-request parameter tuning."""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from google.cloud import storage
import threading
import torch
from torchange.models.segment_any_change import AnyChange
from torchange.models.segment_any_change.segment_anything.utils.amg import MaskData
from config import (
    SAM_CHECKPOINT_FILENAME, ANYCHANGE_MODEL_TYPE,
    POINTS_PER_SIDE, STABILITY_THRESH,
    CHANGE_CONF_THRESH, OBJECT_SIM_THRESH,
    USE_NORMALIZED_FEATURE, BITEMPORAL_MATCH,
)

LOCAL_DIR = Path("/tmp/model")


def _download_gcs(gcs_uri: str, dest: Path) -> Path:
    """Download a GCS object to a local path. Returns the local path."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {gcs_uri}")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: {gcs_uri}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(dest)
    return dest


def resolve_checkpoint_to_local(filename: str = SAM_CHECKPOINT_FILENAME) -> Path:
    """
    Resolve checkpoint from (in priority order):
      1) SAM_CKPT_URI (gs://file|folder OR local file|folder)
      2) AIP_STORAGE_URI (gs://file|folder OR local file|folder)
      3) local CWD (./<filename>)
    For gs:// candidates: download to /tmp/model/<filename>.
    For local candidates: return the existing local path.
    """
    def normalize_candidate(uri_or_path):
        """Return ('gcs', 'gs://...') or ('local', Path(...))."""
        if uri_or_path is None:
            return None
        s = str(uri_or_path).strip()
        if s.startswith("gs://"):
            # allow file or folder
            if s.lower().endswith((".pt", ".pth")):
                return ("gcs", s)
            return ("gcs", s.rstrip("/") + "/" + filename)
        # treat as local file/folder
        p = Path(s)
        if p.is_dir():
            p = p / filename
        return ("local", p)

    candidates = []
    for src in (os.getenv("SAM_CKPT_URI"), os.getenv("AIP_STORAGE_URI"), filename):
        cand = normalize_candidate(src)
        if cand:
            candidates.append(cand)

    dest = LOCAL_DIR / filename
    tried = []
    last_err = None

    for typ, val in candidates:
        tried.append(str(val))
        if typ == "gcs":
            try:
                print(f"[Startup] Trying GCS checkpoint: {val}")
                return _download_gcs(val, dest)
            except Exception as e:
                print(f"[Startup] GCS failed: {val} -> {e}")
                last_err = e
                continue
        else:
            # local
            p = Path(val)
            if p.exists():
                print(f"[Startup] Using local checkpoint at: {p}")
                return p.resolve()
            else:
                print(f"[Startup] Local file not found: {p}")
                last_err = FileNotFoundError(p)

    raise FileNotFoundError(f"Could not locate checkpoint. Tried {tried}. Last error: {last_err}")


class Predictor:
    """Thread-safe AnyChange inference wrapper with per-request parameter overrides."""

    def __init__(self) -> None:
        ckpt_path = resolve_checkpoint_to_local()
        print(f"[Startup] torch.cuda.is_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"[Startup] CUDA device: {torch.cuda.get_device_name(0)}")
            except Exception:
                pass

        self.model = AnyChange(ANYCHANGE_MODEL_TYPE, sam_checkpoint=str(ckpt_path))

        pps  = POINTS_PER_SIDE
        stab = STABILITY_THRESH
        cct  = CHANGE_CONF_THRESH
        ost  = OBJECT_SIM_THRESH

        print(f"[Startup] Using AnyChange model_type={ANYCHANGE_MODEL_TYPE}, checkpoint={ckpt_path}")
        print(f"[Startup] Using hyperparameters: points_per_side={pps}, stability_score_thresh={stab}, change_confidence_threshold={cct}, object_sim_thresh={ost}")

        self.model.make_mask_generator(points_per_side=pps, stability_score_thresh=stab)
        self.model.set_hyperparameters(
            change_confidence_threshold=cct,
            use_normalized_feature=USE_NORMALIZED_FEATURE,
            bitemporal_match=BITEMPORAL_MATCH,
            object_sim_thresh=ost,
        )

        self._pps  = pps
        self._stab = stab
        self._cct  = cct
        self._ost  = ost
        self._cache_defaults()

        # Serialize updates/inference to avoid cross-request parameter races
        self._lock = threading.Lock()


    @staticmethod
    def _count_from_maskdata(md: MaskData | list | np.ndarray) -> int:
        """Return the number of masks in a MaskData, list, or (N,H,W) array."""
        if isinstance(md, MaskData):
            return len(md["rles"])
        if isinstance(md, list):
            return len(md)
        if isinstance(md, np.ndarray) and md.ndim == 3:
            return md.shape[0]  # (N,H,W) stack
        return 0


    def _purge_embed_cache(self, hard: bool = False) -> None:
        """
        Reset AnyChange's cached embeddings between requests.
        - soft: remove stray 'points' if present, else set to None
        - hard: force to None (don't delattr – upstream expects the attributes to exist)
        """
        for name in ("embed_data1", "embed_data2"):
            # Ensure the attribute always exists
            if not hasattr(self.model, name):
                setattr(self.model, name, None)
                continue

            if hard:
                # Force a full recompute next call
                setattr(self.model, name, None)
                print(f"[Predictor] Purged {name} = None (hard)")
            else:
                obj = getattr(self.model, name)
                if isinstance(obj, dict):
                    # Remove transient point prompts but keep embeddings if desired
                    obj.pop("points", None)
                    setattr(self.model, name, obj)
                    print(f"[Predictor] Cleared points from {name} (soft)")
                else:
                    # If it's not a dict, just reset it
                    setattr(self.model, name, None)
                    print(f"[Predictor] Purged {name} = None (soft)")


    def _cache_defaults(self) -> None:
        """Capture the current defaults after __init__ so we can fall back per request."""
        self._defaults = {
            "points_per_side": getattr(self, "_pps", POINTS_PER_SIDE),
            "stability_thresh": getattr(self, "_stab", STABILITY_THRESH),
            "change_conf_thresh": getattr(self, "_cct", CHANGE_CONF_THRESH),
            "object_sim_thresh": getattr(self, "_ost", OBJECT_SIM_THRESH),
            "use_normalized_feature": USE_NORMALIZED_FEATURE,
            "bitemporal_match": BITEMPORAL_MATCH,
        }


    @staticmethod
    def _coerce_bool(v: object, default: bool) -> bool:
        """Coerce a value to bool, returning *default* on None or unrecognized input."""
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        s = str(v).strip().lower()
        if s in {"1","true","t","yes","y"}:
            return True
        if s in {"0","false","f","no","n"}:
            return False
        return default


    @staticmethod
    def _coerce_int(v: object, default: int) -> int:
        """Coerce a value to int, returning *default* on failure."""
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(v: object, default: float) -> float:
        """Coerce a value to float, returning *default* on failure."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return default


    def update_from_params(self, params: dict | None) -> dict:
        """Apply per-request parameter overrides. Returns dict with 'maskgen_rebuilt' flag."""
        if not hasattr(self, "_defaults"):
            self._cache_defaults()
        if not params:
            self._last_params = dict(self._defaults)
            return {"maskgen_rebuilt": False}

        mg_rebuilt = False

        # 1) Mask-generator knobs
        pps  = self._coerce_int(params.get("points_per_side"),   self._defaults["points_per_side"])
        stab = self._coerce_float(params.get("stability_thresh"), self._defaults["stability_thresh"])
        if (pps != getattr(self, "_pps", None)) or (stab != getattr(self, "_stab", None)):
            self.model.make_mask_generator(points_per_side=pps, stability_score_thresh=stab)
            self._pps, self._stab = pps, stab
            mg_rebuilt = True

        # 2) AnyChange hyperparameters
        cct = self._coerce_int(params.get("change_conf_thresh"),  self._defaults["change_conf_thresh"])
        ost = self._coerce_int(params.get("object_sim_thresh"),   self._defaults["object_sim_thresh"])
        use_norm = self._coerce_bool(params.get("use_normalized_feature"), self._defaults["use_normalized_feature"])
        bitemp   = self._coerce_bool(params.get("bitemporal_match"),       self._defaults["bitemporal_match"])

        self.model.set_hyperparameters(
            change_confidence_threshold=cct,
            use_normalized_feature=use_norm,
            bitemporal_match=bitemp,
            object_sim_thresh=ost,
        )

        self._last_params = {
            "points_per_side": pps, "stability_thresh": stab,
            "change_conf_thresh": cct, "object_sim_thresh": ost,
            "use_normalized_feature": use_norm, "bitemporal_match": bitemp,
        }
        return {"maskgen_rebuilt": mg_rebuilt}


    def predict(
        self,
        img1_array: np.ndarray,
        img2_array: np.ndarray,
        query: dict | list | None = None,
        params: dict | None = None,
    ) -> MaskData:
        """Run change detection inference. Thread-safe with per-request params.

        Args:
            img1_array: Before image (HxW or HxWxC, any dtype — coerced to uint8 RGB).
            img2_array: After image (same constraints as img1_array).
            query: None for automatic, dict for single point, list for multi point.
            params: Optional per-request parameter overrides.

        Returns:
            MaskData with detected change masks.
        """
        with self._lock:
            # Ensure 3-channel uint8
            if img1_array.ndim == 2:
                img1_array = np.repeat(img1_array[..., None], 3, axis=-1)
            if img2_array.ndim == 2:
                img2_array = np.repeat(img2_array[..., None], 3, axis=-1)
            img1_array = np.ascontiguousarray(img1_array[..., :3].astype(np.uint8))
            img2_array = np.ascontiguousarray(img2_array[..., :3].astype(np.uint8))

            # Require same HxW (AnyChange works on paired images)
            if img1_array.shape[:2] != img2_array.shape[:2]:
                raise ValueError(f"img1/img2 sizes differ: {img1_array.shape[:2]} vs {img2_array.shape[:2]}")

            # Apply request parameters
            changed = self.update_from_params(params)

            # Purge caches (hard if we rebuilt the maskgen)
            self._purge_embed_cache(hard=changed["maskgen_rebuilt"])

            with torch.inference_mode():
                if query is None:
                    mask_data, _, _ = self.model.forward(img1_array, img2_array)
                elif isinstance(query, dict):
                    xy = query.get("xy"); temporal = int(query.get("temporal", 2))
                    if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                        raise ValueError("single-point query requires {'xy':[x,y], 'temporal':1|2}")
                    mask_data = self.model.single_point_match(xy=xy, temporal=temporal,
                                                            img1=img1_array, img2=img2_array)
                elif isinstance(query, list):
                    xyts = []
                    for p in query:
                        if isinstance(p, dict):
                            xy = p.get("xy"); t = int(p.get("temporal", 2))
                            if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                                raise ValueError("each point dict needs {'xy':[x,y], 'temporal':1|2}")
                            xyts.append([int(xy[0]), int(xy[1]), t])
                        elif isinstance(p, (list, tuple)) and len(p) == 3:
                            xyts.append([int(p[0]), int(p[1]), int(p[2])])
                        else:
                            raise ValueError("points must be dicts or [x,y,t] triples")
                    xyts_arr = np.asarray(xyts, dtype=np.int64)
                    if xyts_arr.ndim != 2 or xyts_arr.shape[1] != 3:
                        raise ValueError(f"multi-points expects shape (N,3), got {xyts_arr.shape}")
                    mask_data = self.model.multi_points_match(xyts=xyts_arr, img1=img1_array, img2=img2_array)
                else:
                    raise ValueError("query must be None, a dict (single), or a list (multi)")

            n_instances = self._count_from_maskdata(mask_data)
            self.last_instances_count = int(n_instances)
            self.last_used_params = getattr(self, "_last_params", dict(self._defaults))
            print(f"[Predict] rles={n_instances}, img_shape={img1_array.shape}, params={self.last_used_params}")
            return mask_data
