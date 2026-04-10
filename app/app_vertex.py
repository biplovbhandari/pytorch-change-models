"""Segment Any Change — Vertex AI Endpoint Streamlit app."""

import io
import time
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from google.cloud import storage, aiplatform

from config import (
    DEMO_T1_URL, DEMO_T2_URL,
    VERTEX_PROJECT_ID, VERTEX_REGION, VERTEX_ENDPOINT_ID,
    GCS_BUCKET, GCS_PREFIX,
    MASK_MODES, DEFAULT_MASK_MODE_INDEX,
    POINTS_PER_SIDE_MIN, POINTS_PER_SIDE_MAX, POINTS_PER_SIDE_DEFAULT, POINTS_PER_SIDE_STEP,
    STABILITY_THRESH_MIN, STABILITY_THRESH_MAX, STABILITY_THRESH_DEFAULT, STABILITY_THRESH_STEP,
    CHANGE_CONF_THRESH_MIN, CHANGE_CONF_THRESH_MAX, CHANGE_CONF_THRESH_DEFAULT, CHANGE_CONF_THRESH_STEP,
    OBJECT_SIM_THRESH_MIN, OBJECT_SIM_THRESH_MAX, OBJECT_SIM_THRESH_DEFAULT, OBJECT_SIM_THRESH_STEP,
    API_TIMEOUT,
)
from viz import (
    png_bytes, decode_npz, overlay_mask_rgba,
    random_colors, overlay_instances, colorize_label,
)
from api_client import load_image_from_url, points_from_canvas


# ---------- Vertex AI helpers ----------
def _init_vertex():
    """Initialize Vertex AI and GCS clients. Returns (endpoint, gcs_bucket)."""
    if not VERTEX_PROJECT_ID or not VERTEX_ENDPOINT_ID:
        st.error("Set `PROJECT_ID` and `ENDPOINT_ID` in your environment or `.env`.")
        st.stop()
    aiplatform.init(project=VERTEX_PROJECT_ID, location=VERTEX_REGION)
    endpoint = aiplatform.Endpoint(VERTEX_ENDPOINT_ID)
    gcs_bucket = None
    if GCS_BUCKET:
        gcs_bucket = storage.Client().bucket(GCS_BUCKET)
    return endpoint, gcs_bucket


def _upload_pil_to_gcs(gcs_bucket, img: Image.Image, dest_path: str) -> str:
    """Upload a PIL image to GCS and return the gs:// URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    blob = gcs_bucket.blob(dest_path)
    blob.upload_from_string(buf.getvalue(), content_type="image/png")
    return f"gs://{gcs_bucket.name}/{dest_path}"


# ---------- Page setup ----------
st.set_page_config(page_title="Segment Any Change (Vertex)", layout="wide")
st.title("Segment Any Change — Vertex AI")

endpoint, gcs_bucket = _init_vertex()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Inference mode")
    mask_mode = st.selectbox("mask_mode", MASK_MODES, index=DEFAULT_MASK_MODE_INDEX)

    st.markdown("### AnyChange / SAM parameters")
    points_per_side = st.slider("points_per_side",
        POINTS_PER_SIDE_MIN, POINTS_PER_SIDE_MAX, POINTS_PER_SIDE_DEFAULT, step=POINTS_PER_SIDE_STEP)
    stability_thresh = st.slider("stability_thresh",
        STABILITY_THRESH_MIN, STABILITY_THRESH_MAX, STABILITY_THRESH_DEFAULT, step=STABILITY_THRESH_STEP)
    change_conf_thr = st.slider("change_conf_thresh",
        CHANGE_CONF_THRESH_MIN, CHANGE_CONF_THRESH_MAX, CHANGE_CONF_THRESH_DEFAULT, step=CHANGE_CONF_THRESH_STEP)
    object_sim_thr = st.slider("object_sim_thresh (for points)",
        OBJECT_SIM_THRESH_MIN, OBJECT_SIM_THRESH_MAX, OBJECT_SIM_THRESH_DEFAULT, step=OBJECT_SIM_THRESH_STEP)

# ---------- Input mode ----------
st.markdown("### Input mode")
input_mode = st.radio("Provide images by:", ["Demo URLs", "Paste URLs", "Upload to GCS"], index=0, horizontal=True)

if input_mode == "Demo URLs":
    img1 = load_image_from_url(DEMO_T1_URL)
    img2 = load_image_from_url(DEMO_T2_URL)
    inst_img1 = {"uri": DEMO_T1_URL}
    inst_img2 = {"uri": DEMO_T2_URL}
    colL, colR = st.columns(2)
    with colL: st.image(img1, caption="BEFORE (demo)", width="stretch")
    with colR: st.image(img2, caption="AFTER (demo)", width="stretch")

elif input_mode == "Paste URLs":
    url1 = st.text_input("BEFORE URL", value=DEMO_T1_URL)
    url2 = st.text_input("AFTER URL", value=DEMO_T2_URL)
    try:
        img1 = load_image_from_url(url1)
        img2 = load_image_from_url(url2)
    except Exception as e:
        st.warning(f"Could not load URLs: {e}")
        st.stop()
    inst_img1 = {"uri": url1}
    inst_img2 = {"uri": url2}
    colL, colR = st.columns(2)
    with colL: st.image(img1, caption="BEFORE", width="stretch")
    with colR: st.image(img2, caption="AFTER", width="stretch")

else:  # Upload to GCS
    if not gcs_bucket:
        st.error("Set `GCS_BUCKET` in your environment to enable uploads.")
        st.stop()
    colL, colR = st.columns(2)
    with colL: up1 = st.file_uploader("Upload BEFORE", type=["png", "jpg", "jpeg"], key="img1")
    with colR: up2 = st.file_uploader("Upload AFTER", type=["png", "jpg", "jpeg"], key="img2")
    if not up1 or not up2:
        st.info("Upload both images to continue.")
        st.stop()
    img1 = Image.open(up1).convert("RGB")
    img2 = Image.open(up2).convert("RGB")
    st.caption(f"Loaded. BEFORE: {img1.size}, AFTER: {img2.size}")
    ts = time.strftime("%Y%m%d-%H%M%S")
    with st.spinner("Uploading to GCS..."):
        uri1 = _upload_pil_to_gcs(gcs_bucket, img1, f"{GCS_PREFIX}/{ts}/t1.png")
        uri2 = _upload_pil_to_gcs(gcs_bucket, img2, f"{GCS_PREFIX}/{ts}/t2.png")
    inst_img1 = {"uri": uri1}
    inst_img2 = {"uri": uri2}

st.caption(f"Image sizes: BEFORE {img1.size}, AFTER {img2.size}")

# ---------- Point canvases ----------
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

pts = points_from_canvas(canvas1.json_data, 1) + points_from_canvas(canvas2.json_data, 2)
with st.expander("Collected points", expanded=False):
    if pts:
        for i, (x, y, t) in enumerate(pts):
            img_label = "BEFORE" if t == 1 else "AFTER"
            st.text(f"  Point {i+1}: ({x}, {y}) on {img_label}")
    else:
        st.text("  None — automatic mode (detect all changes)")

# ---------- Run inference ----------
if st.button("Run"):
    inst = {"img1": inst_img1, "img2": inst_img2}
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

    try:
        with st.spinner("Calling Vertex AI endpoint..."):
            t0 = time.monotonic()
            resp = endpoint.predict(instances=[inst], parameters=parameters, timeout=API_TIMEOUT)
            elapsed = time.monotonic() - t0
            pred = resp.predictions[0]
    except Exception as e:
        st.error(f"Vertex predict error: {e}")
        st.stop()

    st.session_state["pred"] = pred
    st.session_state["elapsed"] = elapsed

# ---------- Display results ----------
if "pred" not in st.session_state:
    st.stop()

pred = st.session_state["pred"]
elapsed = st.session_state["elapsed"]

mode = pred.get("mask_mode", mask_mode)
count = int(pred.get("instances_count", 0))
st.markdown(f"**Instances detected:** {count} | **Inference time:** {elapsed:.1f}s")

if count == 0:
    st.warning("No changes detected. Try different points or adjust the parameters (lower change_conf_thresh = more sensitive).")
    st.stop()

npz = decode_npz(pred["mask_array"]["b64"])
W, H = img2.size
white_bg = Image.new("RGB", (W, H), (255, 255, 255))

if mode == "instances":
    masks = npz["masks"].astype(bool)
    colors = random_colors(masks.shape[0], seed=0)
    overlay_before = overlay_instances(img1, masks, colors)
    overlay_after = overlay_instances(img2, masks, colors)
    mask_only = overlay_instances(white_bg, masks, colors)
elif mode == "label":
    label = npz["arr"].astype(np.int32)
    color_img = colorize_label(label, seed=0)
    overlay_before = Image.blend(img1.convert("RGB"), Image.fromarray(color_img), 0.4)
    overlay_after = Image.blend(img2.convert("RGB"), Image.fromarray(color_img), 0.4)
    mask_only = Image.fromarray(color_img)
else:  # union
    ubin = npz["arr"].astype(bool)
    overlay_before = overlay_mask_rgba(img1, ubin, rgba=(0, 255, 0, 120))
    overlay_after = overlay_mask_rgba(img2, ubin, rgba=(0, 255, 0, 120))
    mask_only = overlay_mask_rgba(white_bg, ubin, rgba=(0, 255, 0, 120))

rL, rR = st.columns(2)
with rL:
    st.subheader("BEFORE")
    st.image(overlay_before, width="stretch")
with rR:
    st.subheader("AFTER")
    st.image(overlay_after, width="stretch")

st.subheader("Change Mask" if mode != "label" else "Label Map")
st.image(mask_only, width="stretch")

# Download buttons
dL, dM, dR = st.columns(3)
with dL:
    st.download_button("Download BEFORE overlay", png_bytes(overlay_before),
                       file_name=f"before_{mode}.png", mime="image/png")
with dM:
    st.download_button("Download AFTER overlay", png_bytes(overlay_after),
                       file_name=f"after_{mode}.png", mime="image/png")
with dR:
    st.download_button("Download Change Mask", png_bytes(mask_only),
                       file_name=f"mask_{mode}.png", mime="image/png")

with st.expander("Parameters used", expanded=False):
    params_used = pred.get("used_parameters")
    if params_used:
        st.json(params_used)
