#!/usr/bin/env python3
"""Verify plugin discovery and lazy native vLLM model registration."""

from __future__ import annotations

from importlib.metadata import entry_points


ARCHITECTURE = "Qwen2_5_VLForConditionalGenerationWithVisionSelector"
EXPECTED_CLASS_PATH = (
    "vision_opd_selector.vllm_model:"
    "Qwen2_5_VLForConditionalGenerationWithVisionSelector"
)


def main() -> None:
    discovered = {
        entry_point.name: entry_point.value
        for entry_point in entry_points(group="vllm.general_plugins")
    }
    expected_entry_point = "vision_opd_selector.vllm_plugin:register"
    actual_entry_point = discovered.get("vision_opd_selector")
    if actual_entry_point != expected_entry_point:
        raise RuntimeError(
            "editable install did not expose the expected vLLM plugin: "
            f"expected={expected_entry_point!r}, actual={actual_entry_point!r}"
        )

    from vllm.model_executor.models import ModelRegistry
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    registered = ModelRegistry.models.get(ARCHITECTURE)
    if registered is None:
        raise RuntimeError(f"{ARCHITECTURE} is absent from vLLM ModelRegistry")

    actual_class_path = f"{registered.module_name}:{registered.class_name}"
    if actual_class_path != EXPECTED_CLASS_PATH:
        raise RuntimeError(
            f"wrong class registered for {ARCHITECTURE}: {actual_class_path}"
        )

    model_class = registered.load_model_cls()
    if model_class.__module__ != "vision_opd_selector.vllm_model":
        raise RuntimeError(f"registry resolved an unexpected module: {model_class.__module__}")
    print(f"entry_point={actual_entry_point}")
    print(f"architecture={ARCHITECTURE}")
    print(f"class_path={actual_class_path}")
    print("vllm_registration=PASS")


if __name__ == "__main__":
    main()

