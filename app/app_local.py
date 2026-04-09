import os, io, base64, time, requests
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

LOCAL_API = os.getenv("LOCAL_API", "http://127.0.0.1:8080/predict")

DEMO_T1_URL = "https://raw.githubusercontent.com/Z-Zheng/pytorch-change-models/main/demo_images/t1_img.png"
DEMO_T2_URL = "https://raw.githubusercontent.com/Z-Zheng/pytorch-change-models/main/demo_images/t2_img.png"


# ---------- helpers ----------
def _b64_png(img: Image.Image) -> str:
    b = io.BytesIO(); img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode("utf-8")


@st.cache_data(show_spinner=False)
def _load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30); r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def _check_server_health(api_url: str) -> bool:
    """Check if the FastAPI server is reachable."""
    health_url = api_url.rsplit("/", 1)[0] + "/health"
    try:
        r = requests.get(health_url, timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def _points_from_canvas(canvas_json, temporal: int):
    pts = []
    if not canvas_json or "objects" not in canvas_json:
        return pts
    for o in canvas_json["objects"]:
        x = int(round(o.get("left", 0)))
        y = int(round(o.get("top",  0)))
        pts.append((x, y, temporal))
    return pts


def _decode_npz(b64str: str):
    return np.load(io.BytesIO(base64.b64decode(b64str)), allow_pickle=False)


def _png_bytes(img: Image.Image) -> bytes:
    """Convert a PIL image (RGB or RGBA) to PNG bytes for download."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _overlay_mask_rgba(base_img: Image.Image, mask_bool: np.ndarray, rgba=(0,255,0,120)) -> Image.Image:
    base = base_img.convert("RGBA").copy()
    overlay = Image.new("RGBA", base.size, rgba)
    alpha = Image.fromarray((mask_bool * rgba[3]).astype(np.uint8), mode="L")
    overlay.putalpha(alpha)
    base.alpha_composite(overlay)
    return base


def _random_colors(n, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, (n, 3), dtype=np.uint8)


def _approx_boundary(mask_bool: np.ndarray) -> np.ndarray:
    m = mask_bool
    up, down  = np.roll(m, -1, 0), np.roll(m, 1, 0)
    left,right= np.roll(m,  1, 1), np.roll(m,-1, 1)
    inner = m & up & down & left & right
    return m & (~inner)


def _overlay_instances(base_img: Image.Image, masks_bool: np.ndarray, colors: np.ndarray) -> Image.Image:
    canvas = base_img.convert("RGBA").copy()
    for i in range(masks_bool.shape[0]):
        fill = tuple(colors[i].tolist()) + (90,)
        canvas = _overlay_mask_rgba(canvas, masks_bool[i], rgba=fill)
        bnd = _approx_boundary(masks_bool[i])
        canvas = _overlay_mask_rgba(canvas, bnd, rgba=(0,255,255,180))
    return canvas


def _colorize_label(label: np.ndarray, seed=0):
    k = int(label.max())
    colors = _random_colors(k + 1, seed=seed)
    colors[0] = np.array([0,0,0], dtype=np.uint8)  # background=black
    return colors[label]  # (H,W,3)


# ---------- UI ----------
st.set_page_config(page_title="Segment Any Change", layout="wide")
st.title("Segment Any Change")

if not _check_server_health(LOCAL_API):
    st.error(f"Server not reachable at `{LOCAL_API}`. Start the FastAPI server first:\n\n"
             "```\ncd deployment/app\n"
             "SAM_CKPT_URI=../../sam_weights/sam_vit_h_4b8939.pth "
             "python -m uvicorn main:app --host 0.0.0.0 --port 8080\n```")
    st.stop()

with st.sidebar:
    st.markdown("### Inference mode")
    mask_mode = st.selectbox("mask_mode", ["instances", "label", "union"], index=0)

    st.markdown("### AnyChange / SAM parameters")
    points_per_side  = st.slider("points_per_side", 4, 64, 32, step=2)
    stability_thresh = st.slider("stability_thresh", 0.50, 0.99, 0.95, step=0.01)
    change_conf_thr  = st.slider("change_conf_thresh", 50, 200, 145, step=1)
    object_sim_thr   = st.slider("object_sim_thresh (for points)", 10, 120, 60, step=5)
    st.divider()
    use_demo = st.checkbox("Use demo URLs", value=True)

# Load images (demo URLs or uploads)
colL, colR = st.columns(2)
with colL:
    if use_demo:
        img1 = _load_image_from_url(DEMO_T1_URL)
        st.info("Using demo BEFORE URL")
    else:
        up1 = st.file_uploader("Upload BEFORE", type=["png","jpg","jpeg"], key="img1")
        if not up1: st.stop()
        img1 = Image.open(up1).convert("RGB")
with colR:
    if use_demo:
        img2 = _load_image_from_url(DEMO_T2_URL)
        st.info("Using demo AFTER URL")
    else:
        up2 = st.file_uploader("Upload AFTER", type=["png","jpg","jpeg"], key="img2")
        if not up2: st.stop()
        img2 = Image.open(up2).convert("RGB")

st.caption(f"Loaded. BEFORE: {img1.size}, AFTER: {img2.size}")

# Point canvases
st.write("Use the 'point' tool to add points on either image.")
c1, c2 = st.columns(2)
with c1:
    st.subheader("BEFORE (temporal=1)")
    canvas1 = st_canvas(
        fill_color="rgba(255,255,255,0)", stroke_width=8, stroke_color="#007bff",
        background_image=img1, update_streamlit=True,
        height=img1.height, width=img1.width, drawing_mode="point", key="canvas1",
    )
with c2:
    st.subheader("AFTER (temporal=2)")
    canvas2 = st_canvas(
        fill_color="rgba(255,255,255,0)", stroke_width=8, stroke_color="#ff7f0e",
        background_image=img2, update_streamlit=True,
        height=img2.height, width=img2.width, drawing_mode="point", key="canvas2",
    )

pts = _points_from_canvas(canvas1.json_data, 1) + _points_from_canvas(canvas2.json_data, 2)
with st.expander("Collected points", expanded=False):
    if pts:
        for i, (x, y, t) in enumerate(pts):
            img_label = "BEFORE" if t == 1 else "AFTER"
            st.text(f"  Point {i+1}: ({x}, {y}) on {img_label}")
    else:
        st.text("  None — automatic mode (detect all changes)")

if st.button("Run"):
    # Build request
    if use_demo:
        inst = {
            "img1": {"uri": DEMO_T1_URL},
            "img2": {"uri": DEMO_T2_URL},
        }
    else:
        inst = {"img1": {"b64": _b64_png(img1)}, "img2": {"b64": _b64_png(img2)}}

    if pts:
        inst["points"] = pts

    parameters = {
        "mask_mode": mask_mode,
        "points_per_side": int(points_per_side),
        "stability_thresh": float(stability_thresh),
        "change_conf_thresh": int(change_conf_thr),
        "object_sim_thresh": int(object_sim_thr),
        "use_normalized_feature": True,
        "bitemporal_match": True,
    }

    payload = {"parameters": parameters, "instances": [inst]}

    try:
        with st.spinner("Running change detection..."):
            t0 = time.monotonic()
            r = requests.post(LOCAL_API, json=payload, timeout=300)
            r.raise_for_status()
            elapsed = time.monotonic() - t0
            pred = r.json()["predictions"][0]
    except requests.ConnectionError:
        st.error(f"Server not reachable at `{LOCAL_API}`.")
        st.stop()
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Cache results in session state so they survive reruns (e.g. download button clicks)
    st.session_state["pred"] = pred
    st.session_state["elapsed"] = elapsed

# Show results from session state
if "pred" not in st.session_state:
    st.stop()

pred = st.session_state["pred"]
elapsed = st.session_state["elapsed"]

mode  = pred.get("mask_mode", mask_mode)
count = int(pred.get("instances_count", 0))
st.markdown(f"**Instances detected:** {count} | **Inference time:** {elapsed:.1f}s")

if count == 0:
    st.warning("No changes detected. Try different points or adjust the parameters (lower change_conf_thresh = more sensitive).")
    st.stop()

npz = _decode_npz(pred["mask_array"]["b64"])

W, H = img2.size
white_bg = Image.new("RGB", (W, H), (255,255,255))

# Results: side-by-side BEFORE | AFTER, then standalone mask below
if mode == "instances":
    masks = npz["masks"].astype(bool)
    colors = _random_colors(masks.shape[0], seed=0)
    overlay_before = _overlay_instances(img1, masks, colors)
    overlay_after = _overlay_instances(img2, masks, colors)
    mask_only = _overlay_instances(white_bg, masks, colors)

    rL, rR = st.columns(2)
    with rL:
        st.subheader("BEFORE")
        st.image(overlay_before, width="stretch")
    with rR:
        st.subheader("AFTER")
        st.image(overlay_after, width="stretch")

    st.subheader("Change Mask")
    st.image(mask_only, width="stretch")

elif mode == "label":
    label = npz["arr"].astype(np.int32)
    color_img = _colorize_label(label, seed=0)
    overlay_before = Image.blend(img1.convert("RGB"), Image.fromarray(color_img), 0.4)
    overlay_after = Image.blend(img2.convert("RGB"), Image.fromarray(color_img), 0.4)
    mask_only = Image.fromarray(color_img)

    rL, rR = st.columns(2)
    with rL:
        st.subheader("BEFORE")
        st.image(overlay_before, width="stretch")
    with rR:
        st.subheader("AFTER")
        st.image(overlay_after, width="stretch")

    st.subheader("Label Map")
    st.image(mask_only, width="stretch")

else:  # union
    ubin = npz["arr"].astype(bool)
    overlay_before = _overlay_mask_rgba(img1, ubin, rgba=(0,255,0,120))
    overlay_after = _overlay_mask_rgba(img2, ubin, rgba=(0,255,0,120))
    mask_only = _overlay_mask_rgba(white_bg, ubin, rgba=(0,255,0,120))

    rL, rR = st.columns(2)
    with rL:
        st.subheader("BEFORE")
        st.image(overlay_before, width="stretch")
    with rR:
        st.subheader("AFTER")
        st.image(overlay_after, width="stretch")

    st.subheader("Change Mask")
    st.image(mask_only, width="stretch")

# Download buttons
dL, dM, dR = st.columns(3)
with dL:
    st.download_button("Download BEFORE overlay", _png_bytes(overlay_before),
                       file_name=f"before_{mode}.png", mime="image/png")
with dM:
    st.download_button("Download AFTER overlay", _png_bytes(overlay_after),
                       file_name=f"after_{mode}.png", mime="image/png")
with dR:
    st.download_button("Download Change Mask", _png_bytes(mask_only),
                       file_name=f"mask_{mode}.png", mime="image/png")

with st.expander("Parameters used", expanded=False):
    params_used = pred.get("used_parameters")
    if params_used:
        st.json(params_used)
