from types import SimpleNamespace

from verl.utils.flops_counter import _estimate_qwen2_flops


def _text_config():
    return SimpleNamespace(
        hidden_size=32,
        vocab_size=128,
        num_hidden_layers=2,
        num_key_value_heads=1,
        num_attention_heads=2,
        intermediate_size=64,
        head_dim=16,
    )


def test_qwen2_flops_accepts_transformers_5_nested_text_config():
    nested = SimpleNamespace(text_config=_text_config())
    flat = _text_config()

    nested_result = _estimate_qwen2_flops(nested, 20, [8, 12], 0.5)
    flat_result = _estimate_qwen2_flops(flat, 20, [8, 12], 0.5)

    assert nested_result == flat_result
