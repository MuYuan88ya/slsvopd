from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_same_input_teacher_reuses_student_tensors_and_multimodal_inputs():
    input_ids = torch.arange(20).reshape(2, 10)
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    position_ids = torch.arange(10).repeat(2, 1)
    responses = torch.arange(8).reshape(2, 4)
    response_mask = torch.ones(2, 4, dtype=torch.long)
    multimodal_inputs = np.array([{"pixel_values": "image-a"}, {"pixel_values": "image-b"}], dtype=object)
    batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
        },
        non_tensors={"multi_modal_inputs": multimodal_inputs},
    )
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "actor": {
                        "policy_loss": {"loss_mode": "vopd"},
                        "self_distillation": {
                            "teacher_always_on": True,
                            "teacher_input_mode": "same_input",
                        },
                    }
                }
            }
        )
    )

    teacher, metrics = RayPPOTrainer._maybe_build_self_distillation_batch(
        trainer,
        batch,
        reward_tensor=torch.zeros_like(responses, dtype=torch.float32),
    )
    assert torch.equal(teacher.batch["teacher_input_ids"], input_ids)
    assert torch.equal(teacher.batch["teacher_attention_mask"], attention_mask)
    assert torch.equal(teacher.batch["teacher_position_ids"], position_ids)
    assert torch.equal(teacher.batch["teacher_response_start_idx"], torch.tensor([6, 6]))
    assert torch.equal(teacher.batch["self_distillation_mask"], torch.ones(2))
    assert list(teacher.non_tensor_batch["teacher_multi_modal_inputs"]) == list(multimodal_inputs)
    assert metrics["self_distillation/same_input_fraction"] == 1.0
