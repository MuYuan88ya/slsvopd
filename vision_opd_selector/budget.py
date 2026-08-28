"""Runtime budget assignment shared by rollout and FSDP training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

BUDGET_KEY = "vision_selector_budget"
MODEL_BUDGETS_KEY = "vision_selector_budgets"


def validate_budget(value: Any) -> float:
    budget = float(value)
    if not 0.0 < budget <= 1.0:
        raise ValueError(f"vision selector budget must be in (0, 1], got {budget}")
    return budget


def _as_budget_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif not isinstance(value, Sequence) or isinstance(value, str | bytes):
        value = [value]
    return [validate_budget(item) for item in value]


def resolve_rollout_budgets(
    *,
    num_prompts: int,
    rollout_n: int,
    default_budget: float,
    configured_budgets: Any = None,
    sample_budgets: Any = None,
) -> np.ndarray:
    """Resolve one budget for every interleaved rollout in a prompt group."""

    default_budget = validate_budget(default_budget)
    if num_prompts < 0 or rollout_n <= 0:
        raise ValueError(f"invalid rollout shape: num_prompts={num_prompts}, rollout_n={rollout_n}")

    configured = None if configured_budgets is None else _as_budget_list(configured_budgets)
    if configured is not None and len(configured) not in {1, rollout_n}:
        raise ValueError(
            "vision_selector.rollout_budgets must contain one value or exactly rollout.n values, "
            f"got {len(configured)} values for rollout.n={rollout_n}"
        )

    if sample_budgets is None:
        per_group = configured or [default_budget]
        if len(per_group) == 1:
            per_group *= rollout_n
        # Keep budgets as float64 metadata.  Values such as 0.7 can otherwise
        # land just below an integer boundary in floor(token_count * budget).
        return np.asarray(per_group * num_prompts, dtype=np.float64)

    if len(sample_budgets) != num_prompts:
        raise ValueError(
            f"sample budget count must match prompt count: {len(sample_budgets)} != {num_prompts}"
        )

    resolved: list[float] = []
    fallback = configured or [default_budget]
    if len(fallback) == 1:
        fallback *= rollout_n
    for sample_value in sample_budgets:
        if sample_value is None:
            resolved.extend(fallback)
            continue
        sample_values = _as_budget_list(sample_value)
        if len(sample_values) == 1:
            sample_values *= rollout_n
        elif len(sample_values) != rollout_n:
            raise ValueError(
                "each sample vision_selector_budget must be a scalar or contain exactly rollout.n values, "
                f"got {len(sample_values)} values for rollout.n={rollout_n}"
            )
        resolved.extend(sample_values)
    return np.asarray(resolved, dtype=np.float64)


def expand_sample_budgets_for_images(sample_budgets: Any, multi_modal_inputs: Any) -> list[float]:
    """Repeat each sample budget for every image owned by that sample."""

    budgets = _as_budget_list(sample_budgets)
    if len(budgets) != len(multi_modal_inputs):
        raise ValueError(
            "vision selector sample budgets and multimodal inputs must have the same batch size, "
            f"got {len(budgets)} and {len(multi_modal_inputs)}"
        )

    per_image: list[float] = []
    for budget, inputs in zip(budgets, multi_modal_inputs, strict=True):
        if inputs is None:
            continue
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            per_image.extend([budget] * len(image_grid_thw))
    return per_image
