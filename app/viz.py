"""Pure visualization helpers for change detection masks. No Streamlit dependency."""

import io
import base64
import numpy as np
from PIL import Image


def b64_png(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    b = io.BytesIO()
    img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode("utf-8")


def png_bytes(img: Image.Image) -> bytes:
    """Convert a PIL image (RGB or RGBA) to PNG bytes for download."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def decode_npz(b64str: str):
    """Decode a base64-encoded .npz archive into a numpy NpzFile."""
    return np.load(io.BytesIO(base64.b64decode(b64str)), allow_pickle=False)


def overlay_mask_rgba(base_img: Image.Image, mask_bool: np.ndarray, rgba=(0, 255, 0, 120)) -> Image.Image:
    """Overlay a boolean mask on an image with the given RGBA color."""
    base = base_img.convert("RGBA").copy()
    overlay = Image.new("RGBA", base.size, rgba)
    alpha = Image.fromarray((mask_bool * rgba[3]).astype(np.uint8), mode="L")
    overlay.putalpha(alpha)
    base.alpha_composite(overlay)
    return base


def random_colors(n: int, seed: int = 0) -> np.ndarray:
    """Generate n random RGB colors as (n, 3) uint8 array."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (n, 3), dtype=np.uint8)


def approx_boundary(mask_bool: np.ndarray) -> np.ndarray:
    """Extract approximate boundary pixels from a boolean mask."""
    m = mask_bool
    up, down = np.roll(m, -1, 0), np.roll(m, 1, 0)
    left, right = np.roll(m, 1, 1), np.roll(m, -1, 1)
    inner = m & up & down & left & right
    return m & (~inner)


def overlay_instances(base_img: Image.Image, masks_bool: np.ndarray, colors: np.ndarray) -> Image.Image:
    """Overlay multiple instance masks with colored fills and cyan boundaries."""
    canvas = base_img.convert("RGBA").copy()
    for i in range(masks_bool.shape[0]):
        fill = tuple(colors[i].tolist()) + (90,)
        canvas = overlay_mask_rgba(canvas, masks_bool[i], rgba=fill)
        bnd = approx_boundary(masks_bool[i])
        canvas = overlay_mask_rgba(canvas, bnd, rgba=(0, 255, 255, 180))
    return canvas


def colorize_label(label: np.ndarray, seed: int = 0) -> np.ndarray:
    """Convert an integer label map to an RGB color image. Background (0) is black."""
    k = int(label.max())
    colors = random_colors(k + 1, seed=seed)
    colors[0] = np.array([0, 0, 0], dtype=np.uint8)
    return colors[label]
