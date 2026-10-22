"""Environment / provenance capture."""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version


def _v(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def environment() -> dict:
    import torch

    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "packages": {p: _v(p) for p in ("transformers", "tokenizers", "accelerate", "datasets", "lm_eval",
                                        "math-verify", "sympy", "antlr4-python3-runtime", "numpy",
                                        "huggingface_hub", "triton", "vllm-rlt")},
        "gpus": smi,
    }
