from __future__ import annotations

from types import SimpleNamespace

import transformers

from verl.utils.tokenizer import hf_processor


def test_qwen25_processor_binds_rope_helper(monkeypatch):
    processor = type("Qwen2_5_VLProcessor", (), {})()

    class FakeQwen25Model:
        def get_vision_position_ids(self):
            return "vision-position-ids"

        def get_rope_index(self):
            return self.get_vision_position_ids()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="qwen2_5_vl"),
    )
    monkeypatch.setattr(transformers.models.qwen2_5_vl, "Qwen2_5_VLModel", FakeQwen25Model)

    loaded = hf_processor("unused")

    assert loaded is processor
    assert loaded.get_rope_index() == "vision-position-ids"
