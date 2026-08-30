# Kaggle dual-T4 experiment log

## 2026-08-31 01:34 CST — environment discovery

- Branch: `experiment/kaggle-t4-fsdp-vllm-smoke`
- Remote host: Kaggle container `240e2e4b18c7`
- GPUs: 2 × Tesla T4, 15,360 MiB each, compute capability 7.5
- `/kaggle/working`: 20 GiB free at session start
- Initial state: `/root/miniconda3` and `/root/Vision-OPD` were absent
- Decision: keep production Flash-Attention defaults unchanged. The dedicated
  smoke command overrides only the Hugging Face/FSDP attention implementation
  to `sdpa`; vLLM is left on its native T4 backend selection and eager mode.
- Model choice for the first control-flow test:
  `katuni4ka/tiny-random-qwen2.5-vl` (~33 MB). The real 3B model remains a
  separate memory-feasibility test after the registration path is proven.
- Status: source synchronized to `/root/Vision-OPD`; Conda bootstrap started.

## 2026-08-31 01:39 CST — environment bootstrap

- Bootstrap completed successfully with Python 3.12.14, Torch 2.10.0+cu128,
  vLLM 0.18.0, Ray 2.53.0, Transformers 5.5.0, and xFormers 0.0.34.
- CUDA detected both T4 GPUs. The bundled xFormers forward/backward test passed.
- Flash-Attention is intentionally absent.
- `uv pip check` reported five direct-pin conflicts. The relevant integration
  risk is that vLLM 0.18 declares `transformers>=4.56,<5`, while this project
  pins Transformers 5.5.0. Plugin import and engine construction must therefore
  be treated as runtime gates rather than inferred from installation success.

## 2026-08-31 01:41 CST — vLLM registration gate

- The editable installation exposed
  `vision_opd_selector = vision_opd_selector.vllm_plugin:register` in the
  `vllm.general_plugins` group.
- A clean remote Python process loaded the general plugins and resolved
  `Qwen2_5_VLForConditionalGenerationWithVisionSelector` to the lazy native
  class path
  `vision_opd_selector.vllm_model:Qwen2_5_VLForConditionalGenerationWithVisionSelector`.
- The custom class imported successfully against the installed vLLM 0.18.
- Result: plugin discovery and native registry registration PASS. Engine model
  construction and checkpoint loading remain separate gates.
