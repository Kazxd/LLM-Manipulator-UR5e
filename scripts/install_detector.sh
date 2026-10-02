#!/usr/bin/env bash
# One-time: Python deps for the open-vocabulary detector (ur_perception/detect_open).
# ROS nodes use the system Python, so install there. numpy is pinned below 2 because ROS Jazzy's
# cv_bridge is built against numpy 1.x (numpy 2 makes cv_bridge crash on import).
set -e
pip install --break-system-packages "numpy<2" torch transformers pillow
python3 - <<'PY'
import numpy, torch, transformers
print("numpy", numpy.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| CUDA", torch.cuda.is_available())
from cv_bridge import CvBridge; CvBridge(); print("cv_bridge OK")
PY
echo "Optional: pre-download the model so the first request is fast:"
echo "  python3 -c \"from transformers import pipeline; pipeline('zero-shot-object-detection', model='google/owlvit-base-patch32')\""
