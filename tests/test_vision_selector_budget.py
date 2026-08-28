from __future__ import annotations

import numpy as np
import pytest
import torch

from vision_opd_selector.budget import expand_sample_budgets_for_images, resolve_rollout_budgets


def test_default_budget_is_repeated_for_every_rollout():
    budgets = resolve_rollout_budgets(num_prompts=2, rollout_n=3, default_budget=0.2)
    assert np.allclose(budgets, [0.2] * 6)


def test_configured_and_per_sample_rollout_budgets():
    configured = resolve_rollout_budgets(
        num_prompts=2,
        rollout_n=3,
        default_budget=0.2,
        configured_budgets=[0.2, 0.15, 0.1],
    )
    assert np.allclose(configured, [0.2, 0.15, 0.1, 0.2, 0.15, 0.1])

    per_sample = resolve_rollout_budgets(
        num_prompts=2,
        rollout_n=3,
        default_budget=0.2,
        configured_budgets=[0.2, 0.15, 0.1],
        sample_budgets=np.array([0.3, [0.4, 0.3, 0.2]], dtype=object),
    )
    assert np.allclose(per_sample, [0.3, 0.3, 0.3, 0.4, 0.3, 0.2])


def test_rollout_budget_shape_is_explicit():
    with pytest.raises(ValueError, match="exactly rollout.n"):
        resolve_rollout_budgets(
            num_prompts=1,
            rollout_n=4,
            default_budget=0.2,
            configured_budgets=[0.2, 0.1],
        )


def test_sample_budget_expands_to_each_image_without_touching_text_samples():
    multi_modal_inputs = np.array(
        [
            {"image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 2]])},
            {},
            {"image_grid_thw": torch.tensor([[1, 4, 4]])},
        ],
        dtype=object,
    )
    budgets = expand_sample_budgets_for_images([0.2, 0.15, 0.1], multi_modal_inputs)
    assert budgets == [0.2, 0.2, 0.1]
