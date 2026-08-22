"""Out-of-tree vLLM model registration."""

from __future__ import annotations

import os
import socket

ARCHITECTURE = "Qwen2_5_VLForConditionalGenerationWithVisionSelector"
MODEL_CLASS_PATH = (
    "vision_opd_selector.vllm_model:Qwen2_5_VLForConditionalGenerationWithVisionSelector"
)
_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm import ModelRegistry

    existing = getattr(ModelRegistry, "models", {}).get(ARCHITECTURE)
    if existing is not None:
        existing_path = f"{getattr(existing, 'module_name', '')}:{getattr(existing, 'class_name', '')}"
        if existing_path != MODEL_CLASS_PATH:
            raise RuntimeError(
                f"vLLM architecture {ARCHITECTURE} is already registered as {existing_path}, "
                f"expected {MODEL_CLASS_PATH}"
            )
    else:
        ModelRegistry.register_model(ARCHITECTURE, MODEL_CLASS_PATH)
    _REGISTERED = True
    print(
        "[VisionSelector vLLM registry] "
        f"host={socket.gethostname()} pid={os.getpid()} architecture={ARCHITECTURE} class={MODEL_CLASS_PATH}",
        flush=True,
    )
