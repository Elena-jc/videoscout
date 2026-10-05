"""VideoScout: a tool-using agent for long-video question answering."""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ultralytics pip-installs missing packages at import time by default. Dependencies
# are declared in pyproject.toml instead, so nothing gets installed behind your back.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
# Keep model downloads inside the project (weights/ and .cache/) instead of the
# user's home directory on the system drive. Set HF_HOME yourself to share a cache.
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".cache" / "huggingface"))

__version__ = "0.2.0"
