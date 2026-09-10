"""Exercise the real actor loss path with a tiny, frozen multimodal backbone."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig, PolicyLossConfig, SelfDistillationConfig
from vision_opd_selector.core import mask_constraint_loss
from vision_opd_selector.fsdp_hook import (
    apply_vision_selector_before_fsdp,
    finalize_vision_selector_trainability,
)


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(out_hidden_size=8)
        self.spatial_merge_size = 1
        self.merger = nn.Linear(8, 8)

    def forward(self, pixels, grid_thw=None):
        return self.merger(pixels)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.visual = TinyVisual()
        self.lm_head = nn.Linear(8, 5)

    def can_generate(self):
        return False

    def forward(
        self, input_ids, pixel_values=None, image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None, **kwargs
    ):
        if pixel_values is not None:
            hidden = self.visual(pixel_values, grid_thw=image_grid_thw)
            counts = image_grid_thw.prod(-1).tolist()
            pooled = torch.stack([segment.mean(0) for segment in hidden.split(counts)])
        else:
            # Match the remove-padding helper's dummy-vision behavior.
            dummy = torch.zeros(4, 8, device=input_ids.device)
            hidden = self.visual(dummy, grid_thw=torch.tensor([[1, 1, 4]], device=input_ids.device))
            pooled = hidden.mean(0, keepdim=True)
        if pixel_values_videos is not None:
            pooled = pooled + self.visual(pixel_values_videos, grid_thw=video_grid_thw).mean(0) * 0
        logits = self.lm_head(pooled)
        return CausalLMOutputWithPast(logits=logits[:, None, :].repeat(1, input_ids.shape[1], 1))


def inject(model, mode="frozen"):
    return apply_vision_selector_before_fsdp(
        model=model,
        role="actor",
        rank=1,
        model_config=None,
        selector_config={"enabled": True, "hidden_dim": 4, "budget": 0.5, "llm_train_mode": mode},
    )


def make_data():
    torch.manual_seed(71)
    inputs = [{"pixel_values": torch.randn(n, 8), "image_grid_thw": torch.tensor([[1, 1, n]])} for n in (4, 12)]
    ids = torch.tensor([[1, 2, 3], [1, 2, 4]])
    return DataProto.from_dict(
        tensors={
            "input_ids": ids,
            "responses": ids[:, -1:],
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(3).expand(2, -1),
            "response_mask": torch.ones(2, 1),
            "old_log_probs": torch.zeros(2, 1),
            "teacher_input_ids": ids.clone(),
            "teacher_attention_mask": torch.ones_like(ids),
            "teacher_position_ids": torch.arange(3).expand(2, -1),
            "teacher_response_start_idx": torch.full((2,), 2),
            "self_distillation_mask": torch.ones(2),
        },
        non_tensors={"multi_modal_inputs": inputs, "teacher_multi_modal_inputs": copy.deepcopy(inputs)},
        meta_info={"temperature": 1.0, "pad_token_id": 0, "global_steps": 1},
    )


def make_actor(model, teacher, micro_size, enabled, dynamic=False):
    cfg = FSDPActorConfig(
        rollout_n=1,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=micro_size,
        use_dynamic_bsz=dynamic,
        ppo_max_token_len_per_gpu=3,
        use_torch_compile=False,
        policy_loss=PolicyLossConfig(loss_mode="vopd"),
        self_distillation=SelfDistillationConfig(
            teacher_model_source="fixed",
            teacher_model_path="test-only",
            teacher_regularization="fixed",
            teacher_input_mode="same_input",
            distillation_topk=None,
            is_clip=None,
            selector_mask_loss_enabled=enabled,
        ),
    )
    cfg.optim.total_training_steps = 2
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-6)
    actor = DataParallelPPOActor(cfg, model, optimizer)
    actor.teacher_module = teacher
    return actor


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_vopd_actor_updates_lis_with_frozen_backbone(monkeypatch, enabled):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_name", lambda: "cpu")
    torch.manual_seed(17)
    base = TinyVLM()
    model = inject(copy.deepcopy(base))
    teacher = copy.deepcopy(base).requires_grad_(False)
    actor = make_actor(model, teacher, 1, enabled)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    teacher_before = copy.deepcopy(teacher.state_dict())
    if not enabled:

        def forbidden(*args, **kwargs):
            raise AssertionError("Disabled BCE was computed")

        monkeypatch.setattr("vision_opd_selector.fsdp_hook.mask_constraint_loss", forbidden)
    metrics = actor.update_policy(make_data())
    assert metrics["actor/grad_norm"] > 0
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all()
        if "importance_scorer" not in name:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    for name in ("k_proj.weight", "q_proj.weight"):
        key = f"visual.importance_scorer.{name}"
        assert not torch.equal(model.state_dict()[key], before[key])
    for name, parameter in teacher.state_dict().items():
        torch.testing.assert_close(parameter, teacher_before[name], rtol=0, atol=0)
    if enabled:
        assert metrics["actor/selector_mask_token_count"] == 16
        assert metrics["actor/selector_mask_loss_weight"] == 0.1


def test_actual_actor_bce_is_invariant_to_dynamic_splitting(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr("verl.workers.actor.dp_actor.get_device_name", lambda: "cpu")
    torch.manual_seed(19)
    base = TinyVLM()
    model = inject(copy.deepcopy(base))
    results = []
    for micro, dynamic in ((2, False), (1, False), (1, True)):
        student = copy.deepcopy(model)
        actor = make_actor(student, copy.deepcopy(base).requires_grad_(False), micro, True, dynamic)

        # Isolate the auxiliary term, retaining a graph-connected zero task loss.
        def zero_vopd(**kwargs):
            return kwargs["student_all_log_probs"].sum() * 0, {}

        monkeypatch.setattr("verl.workers.actor.dp_actor.compute_self_distillation_loss", zero_vopd)
        gradients = []

        def capture_step(student=student, gradients=gradients):
            gradients.append(torch.cat([p.grad.flatten() for p in student.parameters() if p.requires_grad]))
            return gradients[-1].norm()

        monkeypatch.setattr(actor, "_optimizer_step", capture_step)
        metrics = actor.update_policy(make_data())
        results.append((metrics["actor/selector_mask_loss"], gradients[0]))
    for mean, gradient in results[1:]:
        assert mean == pytest.approx(results[0][0], abs=1e-6)
        torch.testing.assert_close(gradient, results[0][1], rtol=0.02, atol=1e-8)


@pytest.mark.parametrize("mode", ["frozen", "full"])
@pytest.mark.parametrize("freeze", [False, True])
def test_final_freezing_contract(mode, freeze):
    model = inject(TinyVLM(), mode)
    if freeze:
        model.visual.requires_grad_(False)
    finalize_vision_selector_trainability(model)
    assert all(p.requires_grad for p in model.visual.importance_scorer.parameters())
    assert not any(p.requires_grad for p in model.visual.merger.parameters())
    assert all(p.requires_grad == (mode == "full") for p in model.lm_head.parameters())
    teacher = TinyVLM().requires_grad_(False)
    finalize_vision_selector_trainability(teacher)
    assert not any(p.requires_grad for p in teacher.parameters())


def test_constraint_uses_scores_when_soft_mask_is_tied():
    scores = torch.tensor([4e-8, 3e-8, 2e-8, 1e-8], requires_grad=True)
    mask = torch.full((4,), 0.5, requires_grad=True)
    loss = mask_constraint_loss(mask, [[1, 1, 4]], 1, 0.5, scores=scores)
    (gradient,) = torch.autograd.grad(loss, mask)
    assert (gradient[:2] < 0).all()
    assert (gradient[2:] > 0).all()
    assert scores.grad is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_forward_fp32_mask_and_bce(device):
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        pytest.skip("Requires a CUDA GPU with BF16 support")
    model = inject(TinyVLM()).to(device)
    scorer = model.visual.importance_scorer
    dtypes = []
    handle = scorer.register_forward_hook(lambda module, args, output: dtypes.append(output.dtype))
    with torch.autocast(device, dtype=torch.bfloat16):
        out = model(
            input_ids=torch.ones(1, 3, dtype=torch.long, device=device),
            pixel_values=torch.randn(16, 8, device=device),
            image_grid_thw=torch.tensor([[1, 1, 16]], device=device),
            vision_selector_compute_mask_loss=True,
        )
        stats = DataParallelPPOActor._take_vision_selector_mask_loss(model)
        assert stats[0].dtype == torch.float32
        assert model.visual._vision_selector_last_mask.dtype == torch.float32
        loss = out.logits.float().square().mean() + stats[0] / stats[1]
    loss.backward()
    handle.remove()
    assert dtypes == [torch.bfloat16]
    assert all(p.dtype == torch.float32 and p.grad is not None and p.grad.isfinite().all() for p in scorer.parameters())


def test_video_preserves_image_stats_and_dummy_image_is_excluded():
    model = inject(TinyVLM())
    ids = torch.ones(1, 3, dtype=torch.long)
    model(
        input_ids=ids,
        pixel_values=torch.randn(4, 8),
        image_grid_thw=torch.tensor([[1, 1, 4]]),
        pixel_values_videos=torch.randn(6, 8),
        video_grid_thw=torch.tensor([[1, 1, 6]]),
        vision_selector_compute_mask_loss=True,
    )
    assert DataParallelPPOActor._take_vision_selector_mask_loss(model)[1] == 4
    model(input_ids=ids, vision_selector_compute_mask_loss=True)
    assert DataParallelPPOActor._take_vision_selector_mask_loss(model) is None
