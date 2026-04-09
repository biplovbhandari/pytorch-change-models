"""Quick test: send demo images to the local FastAPI server."""

import base64
import json
import requests

SERVER_URL = "http://localhost:8080/predict"
DEMO_DIR = "../demo_images"

with open(f"{DEMO_DIR}/t1_img.png", "rb") as f:
    b64_t1 = base64.b64encode(f.read()).decode()
with open(f"{DEMO_DIR}/t2_img.png", "rb") as f:
    b64_t2 = base64.b64encode(f.read()).decode()

resp = requests.post(SERVER_URL, json={
    "instances": [{
        "img1": {"b64": b64_t1},
        "img2": {"b64": b64_t2},
    }]
})
resp.raise_for_status()

pred = resp.json()["predictions"][0]
print(json.dumps({k: v for k, v in pred.items() if k != "mask_array"}, indent=2))
