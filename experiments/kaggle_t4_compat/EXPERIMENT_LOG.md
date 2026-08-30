# Kaggle dual-T4 experiment log

## 2026-08-31 — non-interactive environment activation

- Made `run_smoke.sh` explicitly activate the remote `vision-opd` Conda
  environment. Non-interactive SSH does not source the interactive shell
  setup, so relying on the caller caused the entry-point check to run under the
  system Python even though the editable install in the Conda environment was
  correct.

## 2026-08-31 — Transformers 5 multimodal FLOPs compatibility

- The live run completed rollout, M-RoPE handling, actor/ref forward passes,
  fixed-teacher inference, VOPD backward, and optimizer update. It then failed
  while reporting MFU because VeRL's Qwen2 FLOPs estimator expected language
  dimensions on the top-level config.
- Transformers 5 stores those dimensions under `Qwen2_5_VLConfig.text_config`.
  The estimator now reads the nested text config when present; this changes
  statistics only and does not touch model execution or gradients.

## 2026-08-31 — first complete train/save pass

- The two-T4 run completed native multimodal rollout, M-RoPE, fixed-teacher
  same-input inference, VOPD backward, actor update, sharded model/optimizer/RNG
  save, and the AngelSlim-compatible four-tensor selector export at step 1.
- The run used the production warmup value of 10 with only one step, so its
  learning rate was zero. In addition, the near-zero selector initialization
  produced zero FP16 optimizer moments on T4. This pass validates control flow
  and checkpoint structure, not parameter learning.
- The compatibility launcher now exposes step count, epoch count, warmup,
  selector initialization scale, and actor learning rate as environment-only
  smoke controls. Production defaults remain unchanged.

## 2026-08-31 — epoch-boundary resume bug

- Auto-resume found and loaded `global_step_1`, including the actor,
  optimizer, scheduler/RNG extra state, and StatefulDataLoader state.
- With a one-batch dataloader, VeRL computed `current_epoch = 1` and gave the
  restored exhausted iterator no rollover iteration. The process returned zero
  at 50% progress without running step 2.
- Resume now starts the outer epoch loop from
  `max(0, (global_steps - 1) // len(dataloader))`. Mid-epoch behavior is
  unchanged, while exact epoch-boundary checkpoints retain the rollover
  iteration needed by StatefulDataLoader.

## 2026-08-31 — resume and learning validation passed

- Re-ran auto-resume with the epoch-boundary fix. The trainer found
  `global_step_1`, restored all rank-local model, optimizer, scheduler/RNG, and
  dataloader files, initialized the custom vLLM architecture, synchronized the
  restored actor, generated a fresh multimodal rollout, and completed step 2.
- The tracker advanced to 2 and `global_step_2` contains both FSDP model
  shards, both optimizer shards, both extra-state shards, dataloader state,
  Hugging Face config/processor assets, and the lightweight selector export.
- The rank-0 optimizer state advanced from step 1 to step 2. Selector deltas
  between exports were nonzero for both projection weights (approximately
  `4.32e-5` and `4.01e-5` in L1) and for the numerically non-invariant bias.
  `k_proj.bias` remained zero in this tiny control model.
- The full local test suite passes: 36 tests. This includes the vendored
  AngelSlim loader consuming the exported `config.json`, Python scorer source,
  and `model.safetensors` without conversion.

## 2026-08-31 — real Vero metadata audit

- Ran the metadata-only downloader against `muyuho/Vero-35k-VisionSelector`.
  It fetched only README/summary and the 50.2 MB annotation JSONL; no image
  archive part was downloaded.
- The JSONL has 35,400 records and the real record schema matches the smoke
  path: a VeRL `prompt` chat list containing `<image>`, an `images` list with a
  relative path, plus `data_source`, `ability`, `reward_model`, and
  `extra_info`.
- The published records do not contain `vision_selector_budget`, so training
  correctly uses the configured default 0.2 for every repeated rollout unless
  users enrich the JSONL with per-sample scalar/list budgets.

## 2026-08-31 — first real 3B attempt

- Downloaded `Qwen/Qwen2.5-VL-3B-Instruct` remotely and reused the two tiny
  local smoke images; no Vero image archive was downloaded.
- Both FSDP workers were killed by Ray's host-memory monitor near 87% of the
  initial HF load. The smoke launcher had overridden FSDP mixed precision but
  not `fsdp_config.model_dtype`, so each actor rank still constructed a full
  FP32 3B model before sharding. Two such processes exceed the Kaggle host's
  31 GiB RAM once Python/Ray overhead is included.
- The isolated T4 launcher now also sets actor/ref `model_dtype=float16` to
  test structural 3B compatibility within this host. Production scripts remain
  FP32-master-weight training and are unchanged.
- Raising Ray's node-memory threshold from 95% to 99% still failed while the
  two workers simultaneously constructed the fixed teacher: measured usage was
  29.91/30.00 GiB. The actor itself loaded, injected the selector against the
  real 2048-wide merger, and wrapped with FSDP successfully. Completing a 3B
  actor+fixed-teacher run on this particular host would require a lower-peak
  rank-staggered/meta initialization change or more host RAM; disabling the
  memory monitor would leave effectively no safety margin and was not used.
- A separate single-process load of the real checkpoint successfully applied
  the production selector contract before FSDP and reported
  `in_features=2048`, `hidden_dim=1024`, and exactly four trainable scorer
  tensors. Thus the uncompleted two-model run is a Kaggle host-RAM capacity
  limit, not a shape or registration incompatibility.

## 2026-08-31 — selector-only FSDP checkpoint gather

- Replaced the lightweight export's recursive full-actor
  `FSDP.summon_full_params` with an all-rank gather of only the four scorer
  parameter shards. Rank 0 validates the reconstructed sizes and writes the
  same AngelSlim-compatible directory after the normal full checkpoint has
  succeeded.
- This removes a checkpoint-time full-3B materialization from the selector
  path while preserving the standard VeRL model/optimizer/scheduler/RNG save.
- The revised gather completed in the real two-rank FSDP1/offload smoke run.
  The normal checkpoint, four-tensor selector directory, rollout, fixed-teacher
  forward, actor update, and save all completed successfully.

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

## 2026-08-31 01:43 CST — first data smoke attempt

- The Vero-shaped fixture and `train.verl.jsonl` were generated successfully;
  both records passed the real overlong-prompt filter.
- The validation script then expected a `multi_modal_data` field directly from
  `RLHFDataset.__getitem__` and failed with `KeyError`. This was a smoke-test
  mistake: the current VeRL dataset returns `raw_prompt`; the agent loop calls
  `process_vision_info` and the processor later.
- Resolution: update the gate to exercise the actual sequence
  `raw_prompt -> process_vision_info -> AutoProcessor -> image_grid_thw`.

## 2026-08-31 01:45 CST — data gate and rsync issue

- The corrected data gate passed with two records, including image decoding,
  chat templating, Qwen processing, and `image_grid_thw` generation.
- Hydra then failed before worker launch because `data/legacy_data` was absent.
- Root cause: the initial unanchored rsync exclusion `data/` also matched the
  source directory `verl/trainer/config/data/`.
- Resolution: add a dedicated sync script with root-anchored generated paths
  (`/data/`, `/outputs/`, `/checkpoints/`, and `/rollouts/`). This preserves
  source configuration while still preventing large local artifacts from being
  uploaded.

## 2026-08-31 01:47 CST — Hydra SDPA override

- With the complete source tree present, Hydra reached command composition but
  rejected `actor_rollout_ref.model.override_config.attn_implementation=sdpa`
  because that key is not declared in the structured empty override mapping.
- Resolution: use Hydra's explicit append form
  `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`. This
  remains confined to the T4 smoke command and does not change production
  defaults.

## 2026-08-31 01:52 CST — isolate Hub download from worker startup

- Hydra and Ray core initialized, but TaskRunner remained before GPU worker
  creation for several minutes. Both GPUs stayed at 0 MiB and the process held
  an outbound Hugging Face connection.
- Resolution: stop the inconclusive run, download all 15 tiny-model files to
  `/kaggle/working/vision-opd-t4-smoke/model` with Xet disabled, and make the
  smoke runner use that local directory with `HF_HUB_OFFLINE=1` before Ray
  starts. The download completed in about two seconds.

## 2026-08-31 01:57 CST — placement-group deadlock diagnosed

- The same wait reproduced with a fully local model, ruling out Hub download.
- An experiment-only `SIGUSR1` stack dump showed TaskRunner blocked in
  `RayResourcePool.get_placement_groups()` at `ray.get(pg.ready())`.
- VeRL creates two GPU bundles and reserves three logical CPUs for each. Kaggle
  exposes four CPUs, so the six-CPU placement group cannot be scheduled.
- Resolution: the smoke command sets `ray_kwargs.ray_init.num_cpus=8`. This
  logically oversubscribes the four physical cores only for the tiny control
  flow test; no framework or production configuration is changed.

## 2026-08-31 02:00 CST — first vLLM server construction

- Both FSDP ranks initialized. Rank 0 reported the actor selector with exactly
  four trainable tensors; the fixed teacher reused the frozen ref module.
- Both vLLM server actors discovered the custom native architecture and reached
  `EngineArgs.create_engine_config`.
- vLLM rejected the smoke configuration because `max_num_batched_tokens=384`
  was smaller than the default `max_num_seqs=1024`.
- A separate issue was observed: `vLLMHttpServer.__init__` overwrites configured
  `rollout.max_model_len` with the model's 32768 position limit. For now the
  isolated smoke command sets `max_num_seqs=4` and supplies
  `engine_kwargs.vllm.max_model_len=384`, which is applied last. Whether to fix
  the framework assignment will be decided after the complete control flow is
  proven.

## 2026-08-31 02:04 CST — T4 dtype gate

- A first retry ended in a Ray worker SIGSEGV inside libc `getenv` before the
  second `vLLMHttpServer` actor executed any Python code. Host RAM and GPU RAM
  were idle and a clean retry launched both actors, so this is recorded as a
  non-deterministic Kaggle/Ray startup failure rather than a selector defect.
- On retry, registration propagated through both HTTP servers, both EngineCore
  subprocesses, and both colocated vLLM GPU workers. Device initialization then
  correctly rejected the inherited `bfloat16` dtype because T4 has compute
  capability 7.5.
- Resolution: the isolated T4 command now selects `float16` for rollout and for
  actor/ref FSDP mixed precision. Production dtype defaults remain unchanged.

## 2026-08-31 02:08 CST — first real multimodal rollout

- Both vLLM servers became healthy and the trainer entered step 1. The first
  image request then exposed a real custom-model defect:
  `_parse_and_validate_image_input` tried item assignment on vLLM 0.18's
  `Qwen2_5_VLImagePixelInputs`.
- That native type is a `TensorSchema`: it implements keyed reads with
  `__getitem__`, but has no `__setitem__`. The budget must be attached as an
  attribute, which preserves the downstream keyed-read protocol.
- Resolution: use `setattr` for the extra per-image budget field and add a
  regression test using a TensorSchema-like object that cannot accept item
  assignment.

## 2026-08-31 02:14 CST — tiny-model attention backend

- The TensorSchema fix passed the live image encoder path. Execution proceeded
  into the tiny model's language-model attention, where vLLM's T4 fallback to
  Flex Attention rejected the synthetic model's head dimension of 8 (the
  kernel requires at least 16).
- This is a limitation of the smoke model/backend pair, not the custom visual
  architecture. vLLM 0.18 no longer recognizes the legacy
  `VLLM_ATTENTION_BACKEND` environment variable and its v1 backend enum does
  not include xFormers. The isolated command therefore passes the supported
  engine argument `attention_backend=TORCH_SDPA`; CUDA v1 subsequently reports
  that this enum is not registered as an executable backend, so that override
  is also not usable.
- Resolution: generate a tiny but structurally valid random Qwen2.5-VL model
  with text hidden size 32, two attention heads (head dimension 16), and a
  matching vision output size of 32. It reuses only tokenizer/processor assets
  from the public tiny model. vLLM backend selection remains automatic, and no
  production launcher or backend registration is changed.

## 2026-08-31 02:31 CST — Transformers 5.5 M-RoPE binding

- With the head-dimension-16 model, the first multimodal vLLM request completed
  and returned to VeRL. Agent-loop postprocessing then failed because the
  dynamically bound `Qwen2_5_VLModel.get_rope_index` calls
  `self.get_vision_position_ids`, but VeRL bound only the first method to the
  processor.
- The older ViCuR reference already uses the correct generic pattern: whenever
  a model class exposes `get_vision_position_ids`, bind it alongside
  `get_rope_index`. Current VeRL already does the same for Qwen3/Qwen3.5.
- Resolution: add the missing conditional binding for Qwen2.5-VL and a focused
  regression test. This is a four-line Transformers 5.5 compatibility fix in
  the framework utility, independent of VisionSelector.

## 2026-08-31 02:37 CST — Flash-Attention padding helper

- The M-RoPE fix passed live rollout postprocessing. The trainer reached the
  FSDP actor log-prob forward and then imported
  `flash_attn.bert_padding` because the production launcher enables
  `model.use_remove_padding=True`.
- The T4 compatibility environment intentionally does not install
  Flash-Attention. Resolution: override `use_remove_padding=False` only in the
  isolated smoke command, using the standard padded SDPA path. Production
  defaults remain unchanged.

## 2026-08-31 02:42 CST — Transformers 5.5 vision output contract

- The padded path passed rollout, actor/ref log-prob computation, and fixed
  teacher inference. During `update_actor`, VeRL's Qwen2-VL compatibility
  forward called the vision tower directly and assumed the return value was a
  tensor.
- Transformers 5.5 returns `BaseModelOutputWithPooling`; merged image tokens
  are in `pooler_output`. The selector hook was already updating that field,
  but VeRL then attempted `.shape` on the container.
- Resolution: unwrap `pooler_output` for image, video, and dummy-image calls,
  while retaining the legacy tensor return path. Added focused compatibility
  tests. This is a general Transformers 5.5 VeRL fix, not selector-specific.
