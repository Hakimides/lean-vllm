"""benchmarks 各脚本共用的路径与约定。"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
FP8_SCALES_PATH = REPO_ROOT / "fp8_scales.json"


def resolve_model_path() -> str:
    """模型路径：先 local_settings.MODEL_PATH，再环境变量 MODEL_PATH"""
    try:
        from local_settings import MODEL_PATH
        if MODEL_PATH:
            return MODEL_PATH
    except ImportError:
        pass
    return os.environ.get("MODEL_PATH", "")


def load_run(tag: str) -> dict:
    """读一次运行落盘的 json"""
    return json.loads((RESULTS_DIR / f"{tag}.json").read_text(encoding="utf-8"))
