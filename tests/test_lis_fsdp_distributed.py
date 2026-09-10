"""Native FSDP sharding/export/resume without a dedicated selector wrapper."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_lis_training_contract import TinyVLM, inject, make_actor, make_data
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
)

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from vision_opd_selector.checkpoint import gather_selector_state
from vision_opd_selector.core import differentiable_topk, mask_constraint_loss


def _worker(rank, rendezvous, device_kind, legacy_layout):
    device = torch.device(device_kind, rank) if device_kind == "cuda" else torch.device("cpu")
    if device_kind == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if device_kind == "cuda" else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        import verl.workers.actor.dp_actor as actor_impl

        actor_impl.get_device_id = lambda: device
        actor_impl.get_device_name = lambda: device_kind
        torch.manual_seed(41)
        base = TinyVLM()
        student = inject(copy.deepcopy(base))
        initial_selector = {n: p.detach().clone() for n, p in student.visual.importance_scorer.named_parameters()}
        observed_dtypes = []
        student.visual.importance_scorer.register_forward_hook(
            lambda module, args, output: observed_dtypes.append(output.dtype)
        )

        def wrap(module):
            return FSDP(
                module,
                device_id=device,
                use_orig_params=True,
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            )

        if legacy_layout:
            # Emulate the previous dedicated scorer shard boundary, only in this migration test.
            student.visual.importance_scorer = wrap(student.visual.importance_scorer)
        model = wrap(student)
        assert sum(isinstance(m, FSDP) for m in model.modules()) == (2 if legacy_layout else 1)
        actor = make_actor(model, copy.deepcopy(base).to(device).requires_grad_(False), 1, True)
        # One image per rank, with unequal image sizes (4 and 12 tokens).
        actor.config.ppo_mini_batch_size = 1
        data = make_data()[rank : rank + 1]
        # Compare distributed BCE gradients with the unsplit global token mean.
        token_count = actor._selector_global_token_count(data)
        scalar = torch.tensor(0.3, device=device, requires_grad=True)

        def bce_sum(n):
            scores = torch.arange(n, device=device) * scalar
            mask = differentiable_topk(scores[None], n // 2)[0]
            return mask_constraint_loss(mask, [[1, 1, n]], 1, 0.5, scores=scores, reduction="sum")

        local_loss = actor._scale_selector_mask_loss(bce_sum(4 if rank == 0 else 12), token_count, 2)
        (local_grad,) = torch.autograd.grad(local_loss, scalar)
        dist.all_reduce(local_grad)
        local_grad /= 2
        (reference_grad,) = torch.autograd.grad((bce_sum(4) + bce_sum(12)) / 16, scalar)
        torch.testing.assert_close(local_grad, reference_grad, rtol=1e-6, atol=1e-7)
        scheduler = torch.optim.lr_scheduler.LambdaLR(actor.actor_optimizer, lambda step: 1 / (step + 1))
        manager = FSDPCheckpointManager(model, actor.actor_optimizer, scheduler)
        checkpoint_path = rendezvous + "_checkpoint"
        gradients = []
        for step in (1, 2):
            data.meta_info["global_steps"] = step
            metrics = actor.update_policy(data)
            scheduler.step()
            gradients.append(metrics["actor/grad_norm"])
            assert metrics["actor/selector_mask_token_count"] == 16
            exported = gather_selector_state(model)
            if rank == 0:
                assert set(exported) == set(initial_selector)
                assert all(t.dtype == torch.float32 and t.isfinite().all() for t in exported.values())
                if step == 1:
                    assert not torch.equal(exported["q_proj.weight"], initial_selector["q_proj.weight"])
            if step == 1:
                saved_selector = manager.save_checkpoint(checkpoint_path, global_step=1)
                saved_rng = torch.get_rng_state().clone()
                saved_scheduler = copy.deepcopy(scheduler.state_dict())
                if rank == 0:
                    for name in exported:
                        torch.testing.assert_close(saved_selector[name], exported[name], rtol=0, atol=0)
        assert all(g > 0 for g in gradients)
        assert observed_dtypes and set(observed_dtypes) == {torch.bfloat16}
        for param, state in actor.actor_optimizer.state.items():
            assert param.dtype == torch.float32
            assert state["exp_avg"].dtype == torch.float32

        # Restore ordinary native model/optimizer shards and replay the next update.
        torch.manual_seed(999)
        restored = wrap(inject(TinyVLM()))
        resumed = make_actor(restored, copy.deepcopy(base).to(device).requires_grad_(False), 1, True)
        resumed.config.ppo_mini_batch_size = 1
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed.actor_optimizer, lambda step: 1 / (step + 1))
        resumed_manager = FSDPCheckpointManager(restored, resumed.actor_optimizer, resumed_scheduler)
        if legacy_layout:
            with pytest.raises(ValueError, match="optimizer shard layout"):
                resumed_manager.load_checkpoint(checkpoint_path)
            # Named model shards still load correctly; only positional optimizer resume is rejected.
            restored_selector = gather_selector_state(restored)
            if rank == 0:
                for name in saved_selector:
                    torch.testing.assert_close(restored_selector[name], saved_selector[name], rtol=0, atol=0)
            return
        resumed_manager.load_checkpoint(checkpoint_path)
        assert torch.equal(torch.get_rng_state(), saved_rng)
        assert resumed_scheduler.state_dict() == saved_scheduler
        resumed.update_policy(data)
        resumed_export = gather_selector_state(restored)
        if rank == 0:
            for name in exported:
                torch.testing.assert_close(resumed_export[name], exported[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_kind,legacy_layout", [("cpu", False), ("cpu", True), ("cuda", False)])
def test_native_fsdp_two_steps_export_and_resume(tmp_path, device_kind, legacy_layout):
    if device_kind == "cuda" and (torch.cuda.device_count() < 2 or not torch.cuda.is_bf16_supported()):
        pytest.skip("Requires two CUDA GPUs with BF16 support")
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), device_kind, legacy_layout), nprocs=2, join=True)
