# 训练框架差异审计（2026-09-10）

> 本文记录修复前 `4e5be7b` 的调查快照，包含已被后续决定取代的建议。当前实现将 LIS 作为普通参数层管理，已删除专属 FP32 和独立 wrapping；以 README、代码和回归测试为准。

审计基线：`vopd` 的 `c8a8fdd`；实验版本：`Vision-OPD` 的 `4e5be7b`。两个工作树审计开始时均干净。本次只增加审计文档，没有修改训练代码或实验设定。

## 结论

注册已经不是主要问题。当前实现复用了原有 VOPD loss、FSDP optimizer/backward 和 rollout 全量权重同步，方向合理；但最新 mask BCE 接入存在 CUDA autocast 阻断，以及动态 micro-batch 下目标权重变化。应先修复这些问题并验证训练语义，再收缩框架改动。

本地现有测试 **44 passed**。这不等于真实 CUDA 训练通过：本次环境为 Python 3.12.13、Torch 2.13.0、Transformers 5.14.1、torchdata 0.11.0+cpu，项目锁定的是 Torch 2.10 / Transformers 5.5 / vLLM 0.18。没有运行真实 GPU rollout/FSDP 测试。

## 改了什么

`verl/` 共修改 20 个文件；新增的 selector 实现在 `vision_opd_selector/`。vendored AngelSlim 属于评估依赖，不应混入训练框架改动规模。

| 层 | 原始分支 → 当前实验分支 | 训练影响 |
| --- | --- | --- |
| 默认模型、数据 | Qwen3.5-4B / train.parquet → Qwen2.5-VL-3B / Vero JSONL | 模型、模板、数据分布均变，不能将效果差异仅归于 selector |
| 教师 | 默认 EMA、更新率 0.05、bbox_images → fixed、更新率 0、same_input | 不再是原始高信息输入教师；教师用同一图片的完整视觉特征 |
| 学生 | 原训练参数集合 → 新增四个 LIS 参数，冻结 vision，默认冻结 LLM | 默认是 selector 蒸馏；full 模式只额外训练非 vision 参数 |
| 视觉计算 | native merger → merger 后逐图片 soft Top-K gating | 保留 placeholder、序列长度和 M-RoPE；训练没有真实 token 删除 |
| loss | 原 VOPD loss → 原 VOPD loss + mask BCE | 默认 BCE 系数 0.1 → 2.0，按 trainer global step 调度 |
| 预算 | 固定模型行为 → 默认、逐 group、逐样本预算 | 预算随 trajectory 传给 rollout 和 actor；teacher 不传预算 |
| 教师构建 | 固定教师单独构建 → 复用 ref 模型 | 减少一份模型，但 ref 与 teacher 的语义被绑定 |
| FSDP | 原 wrapping → LIS 独立 FP32 wrapping、use_orig_params=True | 为混合冻结/训练参数与 FP16 梯度保留做适配，建议保留 |
| checkpoint | 原分片 checkpoint → 分片外附 selector 导出，另支持 selector-only | selector-only 不能恢复 optimizer、scheduler、RNG 或数据进度 |
| 兼容修改 | JSONL、vision output.pooler_output、processor M-RoPE、FLOPs text_config | 主要是当前模型/依赖兼容，不是新的优化算法 |
| 恢复 | global_steps // len → (global_steps-1) // len | 为 epoch 末尾 exhausted loader 留一次 rollover，需保留独立测试 |

`verl/trainer/ppo/core_algos.py` 与原始分支没有差异。当前默认 full-logit 分支实际使用学生 top-100 支持集 + tail bucket 的 JSD（alpha=0.5），以及原有 importance weighting，不是全词表逐项 loss，也不是新增 GRPO reward 目标。same_input / always_on / 无 fallback 时训练可以不计算 reward；`adv_estimator=grpo` 的名字不能理解为当前仍主要靠组内奖励学习。

## 已定位问题

### P1：mask BCE 位于 CUDA autocast 内；关闭 loss 开关也无法绕过

调用链：`verl/workers/actor/dp_actor.py:400` 打开 autocast → model forward → `vision_opd_selector/fsdp_hook.py:109` → `vision_opd_selector/core.py:157` 调用 `F.binary_cross_entropy`。

scorer 自己关闭了 autocast，但其上下文在 scorer 返回时结束。BCE 的 `.float()` 不会关闭外层 autocast。PyTorch 2.10 明确规定 CUDA autocast 内的 BCE/BCELoss 报错；CPU 的 BCE 则支持 autocast，解释了现有测试为何未检出。[PyTorch 2.10 AMP 文档](https://docs.pytorch.org/docs/2.10/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy)

`selector_mask_loss_enabled` 仅在 actor 添加 loss 时检查；hook 不读取该开关，只要 training + grad enabled 就计算 BCE。因此设置 false 仍会进入上述调用。

建议：在 mask BCE 计算区域显式关闭 autocast并用 FP32；把 enabled 配置传到产生 loss 的模块，关闭时直接跳过。不要直接对已有概率调用 BCEWithLogitsLoss，它需要 logits，会改变数学含义。

证据等级：代码路径 + 锁定版本官方文档；本机无 CUDA，未声称在 GPU 上复现。

### P1：mask BCE 的有效权重依赖 micro-batch 划分

`core.py:157` 在 micro-batch 内按全部视觉 token 求均值；`dp_actor.py:1011` 附近用 micro-batch 样本数 / mini-batch 样本数给总 loss 缩放。两个不同的归一化单位组合，使同一 batch 的梯度随动态切分或 DP 分配改变。

CPU 数值复现：两张图片分别 2 和 6 个 token，mask 分别为 `[0.9,0.1]` 与 `[0.6,0.4,0.6,0.4,0.6,0.4]`，budget=0.5。

- 一次处理：BCE = **0.40945935**。
- 拆成两个单样本 micro-batch，各乘 1/2：BCE = **0.30809307**。
- 对同一 mask 的梯度最大绝对差 = **0.13888890**。

默认启用了 dynamic batch，因而不是纯理论边界。必须先确定目标是“样本等权”还是“全 batch 视觉 token 等权”：前者需要保留每条样本的 loss，后者需要全 mini-batch / DP 的视觉 token denominator。只换一个 mean 不能同时满足两者。

原版 `VisionSelector/qwen-vl-finetune/compression_method/selector_model.py:310` 也使用 BCE mean，但直接移植该 reduction 到 VeRL 的动态梯度累积，不会自动保留相同的整体权重。

### P2：freeze_vision_tower 会再次冻结 LIS

hook 已将 vision 冻结并把 scorer 设为 trainable；但 `verl/workers/fsdp_workers.py:499` 随后对整个 vision tower 调用 `requires_grad_(False)`。若用户设置通用 `actor.freeze_vision_tower=True`：

- LIS-only 模式最终没有可训练参数，会在 optimizer 构建时报错；
- full 模式仍有 LLM 参数，可继续训练，但 LIS 被静默冻结。

默认值 false，因此不影响默认配置。建议在全部 freezing 逻辑之后验证最终 trainable 参数集合，并在冻结 vision 时排除 scorer，或拒绝冲突配置。

### P2：budget=1 的 LIS-only micro-batch 没有梯度路径

`core.py:90` 的全保留分支返回 `ones_like(scores)`，与 scorer 脱离计算图。若该 micro-batch 的所有图片都全保留，且 backbone 全冻结，则 actor 输出与 mask BCE 都没有训练图，`.backward()` 会失败。budget 校验允许 1，因此这是合法配置的边界问题。

CPU 最小复现确认：全保留 soft-mask 后的 loss.requires_grad=False，backward 报 `does not require grad and does not have a grad_fn`。混合预算也可能因动态切分而出现一个全保留 micro-batch。处理时必须保持各 DP rank backward/collective 一致，不能随意只在某个 rank 跳过 backward。

建议保留值为零的 scorer 梯度依赖，或明确限制训练预算与采样规则；评估仍应允许全保留基线。

## 条件性风险与待验证项

1. **ref/teacher 复用**：`fsdp_workers.py:890` 起，fixed teacher 强制决定 ref_model_path，覆盖独立 ref 路径。当前 use_kl_loss=False 的目标下可以接受；若以后启用 reference KL 并指定另一 ref，此优化会改变 KL 锚点。应仅在两者路径与配置一致时复用，否则单独构建，或提前拒绝。
2. **FSDP2 与 wrapping 配置**：README 限定 FSDP1，但入口没有统一阻止 FSDP2；FP32 子模块约定和导出属于 FSDP1 路径。应启动时校验，避免到最终保存才报 unsupported。
3. **image+video 混合输入**：视频 forward 在 hook 中将 `_vision_selector_mask_loss` 清空；图片先算、视频后算时会丢掉图片 mask loss。纯图片 Vero 主路径不受影响；多模态扩展前需要测试累积而非覆盖。
4. **text-only 数据**：VeRL remove-padding helper 在没有视觉输入时跑 dummy vision 以保持分布式路径。当前 hook 可能对 dummy image 产生非任务相关 mask BCE，应增加真实图片标记/有效 mask loss 数量。需要真实 actor 路径确认。
5. **训练 soft gating vs 评估 hard pruning**：训练保留所有序列位置，hard pruning 删除 token；即使 mask 很尖，两者 attention 的归一化与位置处理也不等价。应作为实验假设测量，不应把 loss 下降视为硬剪枝效果已验证。
6. **HF 导出不自包含 selector 行为**：merger 临时注入 scorer，并硬编码 budget=0.2；普通 HF architecture 的 from_pretrained 不会自动重新安装 hook。README 已提示普通 HF 导出不自动硬剪枝。若目标是可直接重载的 soft-selector 模型，需要保存 config + HF 注册类；否则应把产物明确定位为 backbone 和外部 selector。
7. **旧实验记录的覆盖范围**：08-31 的实机验证包含临时 residency 方案；09-04 明确删除该方案，最新提交又增加 BCE。不能用旧日志证明当前默认 offload + BCE 链路已在 GPU 通过。

## 更简洁的实现建议

不建议为了“可注册”重写一套 trainer，也不建议现在把已验证的原生权重同步替换为增量同步。按以下边界收缩：

1. **保留核心训练流程**：继续复用 VOPD loss、FSDP backward/optimizer、原生 full-state rollout sync、checkpoint 恢复。把 JSONL / Transformers 兼容补丁与实验功能分开记录。
2. **模型负责 selector 行为**：保留 vLLM registration。训练侧先保留现有 pre-FSDP hook 作为小改动方案；若确实需要可直接 HF 重载，再实现 HF 自定义模型类，通过已有 `external_lib` 入口注册。不要仅为移除 hook 引入另一份大型 Qwen forward 拷贝。
3. **辅助 loss 显式返回**：模型返回带梯度的 auxiliary loss 及计数，actor 统一聚合；替代遍历整个模型寻找 `_vision_selector_mask_loss` 的共享可变状态。这样更容易覆盖多图片、视频与 checkpoint recompute。
4. **预算是 trajectory 元数据**：保留当前预算随样本流动的设计。`AsyncLLMServerManager` / server 的通用 `mm_processor_kwargs` 透传值得保留；selector 专属组预算策略和 agent 行为可以放入插件，通过已有 agent-loop registry 接入。不要修改已有基础 `SingleTurnAgentLoop` 的默认行为来承载所有实验。
5. **配置收敛为实验 overlay**：保留原 vopd 配置，另建 selector preset，shell 只负责必要环境和参数。现在 shell 与 YAML 重复定义教师、loss 和预算，容易发生仅修改其中一处的漂移。
6. **统一训练契约校验**：开始训练前检查 actor 参数集合、teacher 无 LIS 且冻结、backend=FSDP1、LoRA 关闭、budget 与 rollout.n、checkpoint 类型、ref/teacher 复用条件。验证的是最终配置和最终 wrapping 结果，不只是在注入时验证一次。

另有一个可以独立优化的数学等价变换：scorer 当前构造 N×N 的 `Q @ K.T` 后沿 key 维平均，可改成 `Q @ mean(K)`，将显式二次注意力矩阵消掉。数学实数意义等价，但浮点归约顺序会改变，尤其初始化分数接近时可能影响 hard top-k 排序；训练与 rollout 要同时修改并比较输出和梯度。

## 下一轮验证顺序

1. 修复 CUDA BCE、开关、最终冻结契约；由实验负责人确认 mask loss 的归一化单位后再修 aggregation。
2. 增加不同 micro-batch 划分输出/梯度等价测试、budget=1、image+video、最终 trainable 参数检查。
3. 在锁定依赖下跑真实 Qwen2.5-VL 两步：同一图片/预算比较 HF 与 vLLM mask、response log-prob；记录 scorer 四参数的梯度和更新量、冻结 backbone/teacher 的不变性。覆盖默认 offload。
4. 完整 checkpoint 的断点续训与连续训练对齐：比较后续样本、global step、LR、BCE weight、selector 参数，而不是只比较 epoch index 函数。CPU 实测 exhausted StatefulDataLoader 首次恢复迭代为空、第二次才进入下一 epoch，支持现有 rollover 修复的动机。
5. 分别测 soft gating 和 hard pruning 的任务指标，确认训练目标确实迁移到预期评估方式。

待确认：README 中固定教师、same_input、默认 LIS-only、soft rollout、BCE 调度是否全部是最终实验意图；mask BCE 要样本等权还是全 batch 视觉 token 等权。
