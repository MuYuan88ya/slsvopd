from types import SimpleNamespace

import torch
from torch import nn

from verl.workers.actor.dp_actor import DataParallelPPOActor


def test_selector_mask_loss_weight_spans_the_training_run():
    actor = object.__new__(DataParallelPPOActor)
    actor.self_distillation_cfg = {
        "selector_mask_loss_weight_start": 0.1,
        "selector_mask_loss_weight_end": 2.0,
    }
    actor.config = SimpleNamespace(optim=SimpleNamespace(total_training_steps=11))

    actor._current_global_steps = 1
    assert actor._vision_selector_mask_loss_weight() == 0.1
    actor._current_global_steps = 6
    assert actor._vision_selector_mask_loss_weight() == 1.05
    actor._current_global_steps = 11
    assert actor._vision_selector_mask_loss_weight() == 2.0


def test_actor_takes_mask_loss_once_without_detaching_it():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    source = torch.tensor(0.5, requires_grad=True)
    model[1]._vision_selector_mask_loss = source.square()

    loss = DataParallelPPOActor._take_vision_selector_mask_loss(model)
    assert loss is not None
    assert loss.requires_grad
    assert model[1]._vision_selector_mask_loss is None
    assert DataParallelPPOActor._take_vision_selector_mask_loss(model) is None
    loss.backward()
    assert source.grad is not None
