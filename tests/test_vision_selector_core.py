from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from verl.model_merger.base_model_merger import BaseModelMerger
from vision_opd_selector.checkpoint import SCORER_KEYS, export_selector, load_selector_weights
from vision_opd_selector.core import TransformerScorer, apply_segmented_soft_mask
from vision_opd_selector.fsdp_hook import apply_vision_selector_before_fsdp


def test_segmented_soft_mask_has_independent_budgets_and_gradients():
    scorer = TransformerScorer(8, hidden_dim=4, init_scale=1e-4)
    hidden = torch.randn(8, 8)
    masked, mask = apply_segmented_soft_mask(
        hidden,
        grid_thw=[[1, 4, 4], [1, 4, 4]],
        spatial_merge_size=2,
        scorer=scorer,
        budget=0.5,
    )
    assert masked.shape == hidden.shape
    assert torch.allclose(mask[:4].sum(), torch.tensor(2.0), atol=1e-4)
    assert torch.allclose(mask[4:].sum(), torch.tensor(2.0), atol=1e-4)
    masked.square().sum().backward()
    assert all(parameter.grad is not None for parameter in scorer.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in scorer.parameters())


class _VisionOutput:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output
        self.hidden_states = "preserved"


class _DummyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(out_hidden_size=8)
        self.spatial_merge_size = 2
        self.backbone = nn.Linear(8, 8)

    def forward(self, hidden_states, grid_thw=None):
        return _VisionOutput(self.backbone(hidden_states))


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _DummyVisual()
        self.lm_head = nn.Linear(8, 8)

    def forward(self, hidden_states, image_grid_thw=None, video_grid_thw=None):
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw
        return self.model.visual(hidden_states, grid_thw=grid_thw)


def _selector_config(checkpoint=None):
    return {
        "enabled": True,
        "budget": 0.5,
        "hidden_dim": 4,
        "init_scale": 1e-4,
        "checkpoint": checkpoint,
        "trainable_only": True,
    }


def test_pre_fsdp_hook_preserves_base_keys_and_freezes_everything_else():
    model = _DummyModel()
    before = set(model.state_dict())
    apply_vision_selector_before_fsdp(
        model=model,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(),
    )
    after = set(model.state_dict())
    added = after - before
    assert added == {f"model.visual.importance_scorer.{key}" for key in SCORER_KEYS}
    assert before <= after
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == added

    output = model.model.visual(torch.randn(4, 8), grid_thw=[[1, 4, 4]])
    assert output.hidden_states == "preserved"
    assert output.pooler_output.shape == (4, 8)
    assert torch.allclose(model.model.visual._vision_selector_last_mask.sum(), torch.tensor(2.0), atol=1e-4)


def test_non_actor_role_is_unchanged():
    model = _DummyModel()
    before = set(model.state_dict())
    apply_vision_selector_before_fsdp(
        model=model,
        role="ref",
        rank=0,
        model_config=None,
        selector_config=_selector_config(),
    )
    assert set(model.state_dict()) == before
    assert not hasattr(model.model.visual, "importance_scorer")


def test_hook_rejects_duplicate_injection_and_bypasses_video():
    model = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=model,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(),
    )
    with torch.no_grad():
        hidden = torch.randn(4, 8)
        grid = torch.tensor([[1, 4, 4]])
        expected = model.model.visual.backbone(hidden)
        output = model(hidden_states=hidden, video_grid_thw=grid)
    assert torch.equal(output.pooler_output, expected)
    assert model.model.visual._vision_selector_last_mask is None

    try:
        apply_vision_selector_before_fsdp(
            model=model,
            role="actor",
            rank=1,
            model_config=None,
            selector_config=_selector_config(),
        )
    except RuntimeError as exc:
        assert "invoked twice" in str(exc)
    else:
        raise AssertionError("duplicate VisionSelector injection was accepted")


def test_angel_slim_export_and_reload_round_trip(tmp_path):
    source = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=source,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(),
    )
    export_dir = export_selector(source, tmp_path / "selector")

    target = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=target,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(),
    )
    load_selector_weights(target, export_dir)
    source_state = source.model.visual.importance_scorer.state_dict()
    target_state = target.model.visual.importance_scorer.state_dict()
    assert source_state.keys() == target_state.keys()
    for name in source_state:
        assert torch.equal(source_state[name], target_state[name])


class _DummyMerger(BaseModelMerger):
    def merge_and_save(self):
        raise NotImplementedError

    def cleanup(self):
        return None


def test_model_merger_recreates_selector_before_key_validation():
    source = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=source,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(),
    )
    merger = _DummyMerger.__new__(_DummyMerger)
    merger.model_config = None
    target = merger.attach_vision_selector_for_state_dict(_DummyModel(), source.state_dict())
    assert hasattr(target.model.visual, "importance_scorer")
    assert set(target.state_dict()) == set(source.state_dict())
