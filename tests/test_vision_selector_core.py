from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from verl.model_merger.base_model_merger import BaseModelMerger
from verl.utils.fsdp_utils import get_fsdp_wrap_policy
from vision_opd_selector.checkpoint import (
    SCORER_KEYS,
    export_selector,
    export_selector_state,
    gather_selector_state,
    load_selector_weights,
)
from vision_opd_selector.core import TransformerScorer, apply_segmented_soft_mask, mask_constraint_loss
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


def test_selector_forward_stays_fp32_under_autocast():
    scorer = TransformerScorer(8, hidden_dim=4, init_scale=1e-4)
    hidden = torch.randn(1, 4, 8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scores = scorer(hidden)
    assert scores.dtype == torch.float32


def test_selector_requests_separate_fp32_fsdp_wrapping():
    model = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=model,
        role="actor",
        rank=0,
        model_config=None,
        selector_config=_selector_config(),
    )
    scorer = model.model.visual.importance_scorer
    assert scorer._fsdp_keep_fp32 is True
    assert scorer._fsdp_wrap_separately is True
    assert scorer._wrap_overrides == {"mixed_precision": None}
    policy = get_fsdp_wrap_policy(model, config={"min_num_params": 0})
    assert policy(scorer, recurse=False, nonwrapped_numel=sum(p.numel() for p in scorer.parameters()))


def test_segmented_soft_mask_accepts_one_budget_per_image():
    scorer = TransformerScorer(8, hidden_dim=4, init_scale=1e-4)
    hidden = torch.randn(8, 8)
    _, mask = apply_segmented_soft_mask(
        hidden,
        grid_thw=[[1, 4, 4], [1, 4, 4]],
        spatial_merge_size=2,
        scorer=scorer,
        budget=[0.25, 0.75],
    )
    assert torch.allclose(mask[:4].sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(mask[4:].sum(), torch.tensor(3.0), atol=1e-4)


def test_mask_constraint_loss_builds_hard_targets_per_image_and_backpropagates():
    mask = torch.tensor(
        [0.8, 0.1, 0.7, 0.4, 0.2, 0.9, 0.6, 0.3],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = mask_constraint_loss(
        mask,
        grid_thw=[[1, 4, 4], [1, 4, 4]],
        spatial_merge_size=2,
        budget=[0.25, 0.75],
    )
    expected_target = torch.tensor([1, 0, 0, 0, 0, 1, 1, 1], dtype=torch.float32)
    expected = torch.nn.functional.binary_cross_entropy(mask, expected_target)
    assert torch.equal(loss, expected)
    loss.backward()
    assert mask.grad is not None
    assert torch.isfinite(mask.grad).all()


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


def _selector_config(checkpoint=None, llm_train_mode="frozen"):
    return {
        "enabled": True,
        "budget": 0.5,
        "hidden_dim": 4,
        "init_scale": 1e-4,
        "checkpoint": checkpoint,
        "llm_train_mode": llm_train_mode,
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
    model.lm_head(output.pooler_output).square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.model.visual.importance_scorer.parameters()
    )
    assert model.model.visual.backbone.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_pre_fsdp_hook_consumes_runtime_budget_before_model_forward():
    model = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=model,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(),
    )
    model(
        hidden_states=torch.randn(4, 8),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        vision_selector_budgets=[0.25],
    )
    assert torch.allclose(model.model.visual._vision_selector_last_mask.sum(), torch.tensor(1.0), atol=1e-4)
    assert model.model.visual._vision_selector_runtime_budgets is None


def test_pre_fsdp_hook_exposes_differentiable_mask_loss_only_during_training():
    model = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=model,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(),
    )
    model.train()
    model(
        hidden_states=torch.randn(4, 8),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
    )
    mask_loss = model.model.visual._vision_selector_mask_loss
    assert mask_loss is not None
    assert mask_loss.requires_grad
    mask_loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.model.visual.importance_scorer.parameters()
    )

    model.eval()
    with torch.no_grad():
        model(
            hidden_states=torch.randn(4, 8),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
        )
    assert model.model.visual._vision_selector_mask_loss is None


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
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_full_llm_mode_keeps_visual_frozen_and_trains_non_visual_parameters():
    model = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=model,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(llm_train_mode="full"),
    )

    assert all(
        parameter.requires_grad
        for parameter in model.model.visual.importance_scorer.parameters()
    )
    assert not model.model.visual.backbone.weight.requires_grad
    assert not model.model.visual.backbone.bias.requires_grad
    assert model.lm_head.weight.requires_grad
    assert model.lm_head.bias.requires_grad

    pooled = model.model.visual(torch.randn(4, 8), grid_thw=[[1, 4, 4]]).pooler_output
    model.lm_head(pooled).square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in model.model.visual.importance_scorer.parameters()
    )
    assert model.model.visual.backbone.weight.grad is None
    assert model.model.visual.backbone.bias.grad is None
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.bias.grad is not None


def test_hook_rejects_unknown_llm_train_mode():
    model = _DummyModel()
    try:
        apply_vision_selector_before_fsdp(
            model=model,
            role="actor",
            rank=1,
            model_config=None,
            selector_config=_selector_config(llm_train_mode="lora"),
        )
    except ValueError as exc:
        assert "llm_train_mode" in str(exc)
    else:
        raise AssertionError("unsupported LLM train mode was accepted")


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


def test_unwrapped_selector_only_gather_contains_exactly_four_tensors():
    model = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=model,
        role="actor",
        rank=0,
        model_config=None,
        selector_config=_selector_config(),
    )
    state = gather_selector_state(model)
    assert state is not None
    assert set(state) == SCORER_KEYS


def test_angel_slim_state_export_and_reload_round_trip(tmp_path):
    source = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=source,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(),
    )
    source_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in source.model.visual.importance_scorer.state_dict().items()
    }
    export_dir = export_selector_state(source_state, tmp_path / "selector-state")

    target = _DummyModel()
    apply_vision_selector_before_fsdp(
        model=target,
        role="actor",
        rank=1,
        model_config=None,
        selector_config=_selector_config(checkpoint=str(export_dir)),
    )
    for name, tensor in target.model.visual.importance_scorer.state_dict().items():
        assert torch.equal(tensor, source_state[name])


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
