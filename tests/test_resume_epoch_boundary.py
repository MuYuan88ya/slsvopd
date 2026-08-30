from verl.trainer.ppo.ray_trainer import _resume_epoch_index


def test_resume_epoch_keeps_rollover_iteration_at_epoch_boundary():
    assert _resume_epoch_index(global_steps=0, dataloader_length=10) == 0
    assert _resume_epoch_index(global_steps=5, dataloader_length=10) == 0
    assert _resume_epoch_index(global_steps=10, dataloader_length=10) == 0
    assert _resume_epoch_index(global_steps=11, dataloader_length=10) == 1
    assert _resume_epoch_index(global_steps=20, dataloader_length=10) == 1
