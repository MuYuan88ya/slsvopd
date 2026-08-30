from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verl.models.transformers.qwen2_vl import _get_vision_pooler_output


def test_vision_pooler_output_accepts_transformers5_structured_output():
    merged = torch.randn(3, 4)

    assert _get_vision_pooler_output(SimpleNamespace(pooler_output=merged)) is merged
    assert _get_vision_pooler_output(merged) is merged


def test_vision_pooler_output_rejects_unknown_contract():
    with pytest.raises(TypeError, match="Unsupported vision output type"):
        _get_vision_pooler_output(SimpleNamespace(last_hidden_state=torch.randn(3, 4)))
