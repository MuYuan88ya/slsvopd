from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import tomllib
from hydra import compose, initialize_config_dir

from vision_opd_selector.vllm_plugin import ARCHITECTURE, MODEL_CLASS_PATH


def test_vopd_config_declares_native_vllm_architecture_without_worker_extension():
    config_dir = Path(__file__).parents[1] / "verl" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="vopd")

    assert config.actor_rollout_ref.model.pre_fsdp_model_hook == (
        "pkg://vision_opd_selector.fsdp_hook:apply_vision_selector_before_fsdp"
    )
    vllm = config.actor_rollout_ref.rollout.engine_kwargs.vllm
    assert "worker_extension_cls" not in vllm
    assert vllm.model_impl == "auto"
    assert vllm.hf_overrides.architectures == [ARCHITECTURE]
    assert vllm.hf_overrides.vision_selector.budget == 0.2
    assert config.actor_rollout_ref.model.vision_selector.rollout_budgets is None
    assert config.actor_rollout_ref.model.vision_selector.eval_budget is None
    assert config.actor_rollout_ref.model.vision_selector.llm_train_mode == "frozen"
    assert config.actor_rollout_ref.model.lora_rank == 0
    assert config.actor_rollout_ref.model.lora_adapter_path is None


def test_pyproject_exposes_vllm_general_plugin():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    entry_points = metadata["project"]["entry-points"]["vllm.general_plugins"]
    assert entry_points["vision_opd_selector"] == "vision_opd_selector.vllm_plugin:register"


def test_vllm_plugin_registration_is_idempotent(monkeypatch):
    class _Registry:
        models = {}

        @classmethod
        def register_model(cls, architecture, model_path):
            module_name, class_name = model_path.split(":", 1)
            cls.models[architecture] = SimpleNamespace(module_name=module_name, class_name=class_name)

    fake_vllm = ModuleType("vllm")
    fake_vllm.ModelRegistry = _Registry
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    from vision_opd_selector.vllm_plugin import register

    register()
    register()
    assert ARCHITECTURE in _Registry.models
    registered = _Registry.models[ARCHITECTURE]
    assert f"{registered.module_name}:{registered.class_name}" == MODEL_CLASS_PATH


def test_vllm_plugin_rejects_architecture_conflict(monkeypatch):
    class _Registry:
        models = {
            ARCHITECTURE: SimpleNamespace(module_name="another_package.model", class_name="WrongModel")
        }

    fake_vllm = ModuleType("vllm")
    fake_vllm.ModelRegistry = _Registry
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    import vision_opd_selector.vllm_plugin as plugin

    monkeypatch.setattr(plugin, "_REGISTERED", False)
    with pytest.raises(RuntimeError, match="already registered"):
        plugin.register()
