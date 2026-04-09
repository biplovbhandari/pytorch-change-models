"""AnyChange server configs."""

import os

# Model
SAM_CHECKPOINT_FILENAME = os.getenv("SAM_CHECKPOINT_FILENAME", "sam_vit_h_4b8939.pth")
ANYCHANGE_MODEL_TYPE = os.getenv("ANYCHANGE_MODEL_TYPE", "vit_h")

# Mask generator
POINTS_PER_SIDE = int(os.getenv("POINTS_PER_SIDE", "32"))
STABILITY_THRESH = float(os.getenv("STABILITY_THRESH", "0.95"))

# AnyChange hyperparameters
CHANGE_CONF_THRESH = int(os.getenv("CHANGE_CONF_THRESH", "145"))
OBJECT_SIM_THRESH = int(os.getenv("OBJECT_SIM_THRESH", "60"))
USE_NORMALIZED_FEATURE = os.getenv("USE_NORMALIZED_FEATURE", "true").lower() in ("1", "true", "yes")
BITEMPORAL_MATCH = os.getenv("BITEMPORAL_MATCH", "true").lower() in ("1", "true", "yes")

# Server
DEFAULT_MASK_MODE = os.getenv("DEFAULT_MASK_MODE", "instances")
VALID_MASK_MODES = {"union", "label", "instances"}
