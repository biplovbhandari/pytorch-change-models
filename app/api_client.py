"""Network and utility functions for the Streamlit app."""

import io
import requests
from PIL import Image
import streamlit as st


@st.cache_data(show_spinner=False)
def load_image_from_url(url: str) -> Image.Image:
    """Fetch an image from a URL and return as PIL RGB. Cached across reruns."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def check_server_health(api_url: str) -> bool:
    """Check if the FastAPI server is reachable via its /health endpoint."""
    health_url = api_url.rsplit("/", 1)[0] + "/health"
    try:
        r = requests.get(health_url, timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def points_from_canvas(canvas_json, temporal: int) -> list[tuple]:
    """Extract (x, y, temporal) points from a streamlit-drawable-canvas JSON result."""
    pts = []
    if not canvas_json or "objects" not in canvas_json:
        return pts
    for o in canvas_json["objects"]:
        x = int(round(o.get("left", 0)))
        y = int(round(o.get("top", 0)))
        pts.append((x, y, temporal))
    return pts
