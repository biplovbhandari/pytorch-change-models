"""App-level constants for the Segment Any Change Streamlit UI."""

import os

# API
LOCAL_API = os.getenv("LOCAL_API", "http://127.0.0.1:8080/predict")

# Demo images
DEMO_T1_URL = "https://raw.githubusercontent.com/Z-Zheng/pytorch-change-models/main/demo_images/t1_img.png"
DEMO_T2_URL = "https://raw.githubusercontent.com/Z-Zheng/pytorch-change-models/main/demo_images/t2_img.png"

# Mask modes
MASK_MODES = ["instances", "label", "union"]
DEFAULT_MASK_MODE_INDEX = 0

# Slider ranges and defaults
POINTS_PER_SIDE_MIN = 4
POINTS_PER_SIDE_MAX = 64
POINTS_PER_SIDE_DEFAULT = 32
POINTS_PER_SIDE_STEP = 2

STABILITY_THRESH_MIN = 0.50
STABILITY_THRESH_MAX = 0.99
STABILITY_THRESH_DEFAULT = 0.95
STABILITY_THRESH_STEP = 0.01

CHANGE_CONF_THRESH_MIN = 50
CHANGE_CONF_THRESH_MAX = 200
CHANGE_CONF_THRESH_DEFAULT = 145
CHANGE_CONF_THRESH_STEP = 1

OBJECT_SIM_THRESH_MIN = 10
OBJECT_SIM_THRESH_MAX = 120
OBJECT_SIM_THRESH_DEFAULT = 60
OBJECT_SIM_THRESH_STEP = 5

# Request timeout
API_TIMEOUT = 300

# Vertex AI (used by app_vertex.py)
VERTEX_PROJECT_ID = os.getenv("PROJECT_ID", "")
VERTEX_REGION = os.getenv("REGION", "us-central1")
VERTEX_ENDPOINT_ID = os.getenv("ENDPOINT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
GCS_PREFIX = os.getenv("GCS_PREFIX", "ui-uploads")
