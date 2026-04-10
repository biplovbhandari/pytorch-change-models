## torchange - A Unified Change Representation Learning Benchmark Library
[![PyPI Downloads](https://static.pepy.tech/badge/torchange)](https://pepy.tech/projects/torchange)

torchange aims to provide out-of-box contemporary spatiotemporal change model implementations, standard metrics, and datasets, in pursuit of benchmarking and reproducibility.

>This project is still under development. Other repositories would be gradually merged into ```torchange```.

> The ```torchange``` API is in beta and may change in the near future.

> Note: ```torchange``` is designed to provide straightforward implementations, thus we will adopt a single file for each algorithm without any modular encapsulation.
Algorithms released before 2024 will be transferred here from our internal codebase.
If you encounter any bugs, please report them in the issue section. Please be patient with new releases and bug fixes, as this is a significant burden for a single maintainer.
Technical consultations are only accepted via email inquiry.

> Our default training engine is [ever](https://github.com/Z-Zheng/ever/).

### News

- 2024/06, we launch the project of ``torchange``.

### Features

- Out-of-box and straightforward model implementations
- Highly-optimized implementations, e.g., multi-gpu sync dice loss.
- Multi-gpu metric computation and score tracker, supporting wandb.
- Including the latest research advancements in ``Change``, not just architecture games.

### Installation


#### nightly version (master)
```bash
pip install -U --no-deps --force-reinstall git+https://github.com/Z-Zheng/pytorch-change-models
```

#### stable version (pypi)
```bash
pip install torchange
```

#### conda-forge version
```bash
conda install -c conda-forge torchange
```

### Model zoo (in progress)

This is also a tutorial for junior researchers interested in contemporary change detection.


#### 0. change modeling principle
- (PCM) Unifying Remote Sensing Change Detection via Deep Probabilistic Change Models: from Principles, Models to Applications, ISPRS P&RS 2024. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0924271624002624)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changesparse.py)]
- (GPCM) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]


#### 1.0 unified architecture
- (ChangeStar) Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery, ICCV 2021. [[`Paper`](https://arxiv.org/abs/2108.07002)], [[`Project`](https://zhuozheng.top/changestar/)], [[`Code`](https://github.com/Z-Zheng/ChangeStar)]
- (ChangeStar2) Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection, IJCV 2024. [[`Paper`](https://link.springer.com/article/10.1007/s11263-024-02141-4)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changestar2.py)]
- (ChangeSparse) Unifying Remote Sensing Change Detection via Deep Probabilistic Change Models: from Principles, Models to Applications, ISPRS P&RS 2024. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0924271624002624)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changesparse.py)]

#### 1.1 one-to-many semantic change detection
- (ChangeOS) Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: from natural disasters to man-made disasters, RSE 2021. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0034425721003564)], [[`Inference API Code`](https://github.com/Z-Zheng/ChangeOS)], [[`Trainable Model Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changeos.py)]

#### 1.2 many-to-many semantic change detection
- (ChangeMask) ChangeMask: Deep Multi-task Encoder-Transformer-Decoder Architecture for Semantic Change Detection, ISPRS P&RS 2022. [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0924271621002835)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changemask.py)]


#### 2.0 learning change representation via single-temporal supervision
- (STAR) Change is Everywhere: Single-Temporal Supervised Object Change Detection in Remote Sensing Imagery, ICCV 2021. [[`Paper`](https://arxiv.org/abs/2108.07002)], [[`Project`](https://zhuozheng.top/changestar/)], [[`Code`](https://github.com/Z-Zheng/ChangeStar)]
- (G-STAR) Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection, IJCV 2024. [[`Paper`](https://link.springer.com/article/10.1007/s11263-024-02141-4)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/changestar2.py)]
- (Changen) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]
- (Changen2) Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model, IEEE TPAMI 2024 [[`Paper`](https://arxiv.org/abs/2406.17998)]，[[`Code/Dataset/Pretrained Models`](https://github.com/Z-Zheng/pytorch-change-models/tree/main/torchange/models/changen2)]


#### 2.1 change data synthesis from single-temporal data
- (Changen) Scalable Multi-Temporal Remote Sensing Change Data Generation via Simulating Stochastic Change Process, ICCV 2023 [[`Paper`](https://arxiv.org/pdf/2309.17031)], [[`Code`](https://github.com/Z-Zheng/Changen)]
- (Changen2) Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model, IEEE TPAMI 2024 [[`Paper`](https://arxiv.org/abs/2406.17998)]，[[`Code/Dataset/Pretrained Models`](https://github.com/Z-Zheng/pytorch-change-models/tree/main/torchange/models/changen2)]

#### 2.2 change data synthesis from bi-temporal data
- (NeDS) Neural Disaster Simulation for Transferable Building Damage Assessment, RSE 2025 [[`Paper`](https://www.sciencedirect.com/science/article/pii/S0034425725003839)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/neds.py)], [[`Model`](https://huggingface.co/EVER-Z/NeDS)]

#### 3.0 zero-shot change detection
- (AnyChange) Segment Any Change, NeurIPS 2024 [[`Paper`](https://arxiv.org/abs/2402.01188)], [[`Code`](https://github.com/Z-Zheng/pytorch-change-models/blob/main/torchange/models/segment_any_change)]
- (Changen2) Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model, IEEE TPAMI 2024 [[`Paper`](https://arxiv.org/abs/2406.17998)]，[[`Code/Dataset/Pretrained Models`](https://github.com/Z-Zheng/pytorch-change-models/tree/main/torchange/models/changen2)]


---

### AnyChange Streamlit App + FastAPI Server

This fork adds a Streamlit web app and FastAPI inference server for the [AnyChange (Segment Any Change)](https://arxiv.org/abs/2402.01188) model, enabling interactive zero-shot change detection through a browser UI.

#### What's added

- `app/` — Two Streamlit apps (local FastAPI + Vertex AI) sharing visualization, config, and canvas modules. Interactive point drawing, three mask modes, parameter controls, and result downloads. See [`app/README.md`](app/README.md) for details.
- `deployment/app/` — FastAPI server wrapping AnyChange with support for automatic, single-point, and multi-point change detection. Accepts images via base64, GCS URI, or HTTP URL. Per-request parameter tuning. See [`deployment/README.md`](deployment/README.md) for API docs.
- `deployment/app/config.py` — Single source of truth for all model defaults and server configuration, overridable via environment variables.

#### Quick start

```bash
# 1. Create environment and install
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[app]"

# 2. Download SAM checkpoint (vit_h = 2.4GB, or vit_b = 375MB for a lighter option)
mkdir -p sam_weights
curl -L -o sam_weights/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 3. Start the FastAPI server
cd deployment/app
SAM_CKPT_URI=../../sam_weights/sam_vit_h_4b8939.pth \
python -m uvicorn main:app --host 0.0.0.0 --port 8080

# 4. In another terminal, run the Streamlit app (from repo root):
streamlit run app/app_local.py
```

#### Configuration

All defaults live in `deployment/app/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM_CKPT_URI` | — | Path or `gs://` URI to SAM checkpoint |
| `ANYCHANGE_MODEL_TYPE` | `vit_h` | SAM variant: `vit_b`, `vit_l`, or `vit_h` |
| `POINTS_PER_SIDE` | `32` | Grid density for automatic mask generation |
| `STABILITY_THRESH` | `0.95` | SAM mask stability score threshold |
| `CHANGE_CONF_THRESH` | `145` | Change confidence angle threshold (degrees) |
| `OBJECT_SIM_THRESH` | `60` | Object similarity threshold for point queries |
| `DEFAULT_MASK_MODE` | `instances` | Output mode: `instances`, `label`, or `union` |

#### SAM checkpoints

| Model | Size | Download |
|-------|------|----------|
| `vit_b` | 375 MB | `sam_vit_b_01ec64.pth` |
| `vit_l` | 1.2 GB | `sam_vit_l_0b3195.pth` |
| `vit_h` | 2.4 GB | `sam_vit_h_4b8939.pth` |

All available from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints).

---

### Contributing
We welcome issues and pull requests that improve model implementations, documentation, or usability.

**Before filing an issue**
- Search existing issues to avoid duplicates.
- Include your environment details (OS, Python, PyTorch), model name, and a minimal reproducible example if possible.
- Attach logs or error traces with clear steps to reproduce.

**Requesting features or new models**
- Provide the paper link, a short summary, and expected I/O behavior.
- If you have a reference implementation, link it.

### License
This project is under the Apache 2.0 License. See [LICENSE](https://github.com/Z-Zheng/pytorch-change-models/blob/main/LICENSE) for details.

If you find it useful in your work — whether in research, demos, products, or educational materials — we’d love to hear from you!

Sharing your use case helps us:

📌 Understand real-world impact

📣 Highlight your work in our future talks or papers

🚀 Improve future versions of this project

📬 Please drop us a short message at:
zhuozheng@cs.stanford.edu
Feel free to include a brief description, your institution/company, a link to your work (if available), or any suggestions.


![visitors](https://visitor-badge.laobi.icu/badge?page_id=Z-Zheng/pytorch-change-models)
