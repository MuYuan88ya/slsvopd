# 项目目标与开发约束

本文件适用于 Vision-OPD 仓库。操作步骤和启动参数见 [guide.md](guide.md)，更详细的使用与评估说明见 [README.md](README.md)。

## 项目定位

本实验称为 SLSOPD：将 VisionSelector 的 LIS（Learned Importance Scorer）接入 VeRL 的 Vision-OPD on-policy self-distillation 流程，让任务蒸馏信号学习视觉 token 的重要性。

工作区中的参考关系：

- `Vision-OPD/`：当前实验实现，代码修改主要落在这里。
- `../vopd/`：原始 Vision-OPD 参考，用于对比训练框架和原版行为。
- `../VisionSelector/`：LIS、可微 Top-K 与原版辅助 BCE 的重要参考。
- `../AngelSlim/`：selector 的部署与 hard pruning 评估参考。

默认目标是训练 Qwen2.5-VL-3B 的 LIS，固定同模型教师，学生和教师读取相同文本与图片。教师使用完整视觉特征；学生在 vision merger 后施加 soft mask。默认 group size 为 8，每张图片的 token 保留预算为 0.2，辅助 BCE 关闭。

训练采用 soft gating，不真实删除视觉 token；hard pruning 的实际压缩效果由评估侧验证。不要把训练预算 0.2 表述为训练序列长度或显存减少到 20%。当前 selector 面向图片，视频绕过 selector。不支持 LoRA、训练期硬删除 token 或通用 Transformers vLLM fallback。

## 必须保持的训练语义

1. **LIS 是普通网络层。** 初始化、参数 dtype、分片、optimizer、offload、同步与恢复遵循框架默认流程。Actor 和 vLLM 均不得新增 LIS 专属 `.float()`、dtype/device 修正、mixed-precision override 或独立 FSDP wrapping。实验配置使用原生 `use_orig_params=True` 支持冻结/可训练参数混合。
2. **参数精度与局部算子精度分开。** LIS 投影和打分遵循默认 BF16 前向；VeRL 默认 FP32 主参数、更新和梯度归约照常处理。只有 Top-K 阈值、sigmoid、自定义 backward 和 BCE 使用 FP32 中间值。BCE 显式关闭 autocast，soft mask 在特征 gating 时转为特征 dtype。
3. **任务梯度必须到达 LIS。** 保持 `VOPD loss → 学生网络 → soft mask → scores → LIS` 的反向路径。冻结 LLM 参数不能用覆盖学生前向的 `no_grad` 切断输入梯度。关闭 BCE 后仍须能由真实 VOPD loss 单独更新 LIS。
4. **BCE 是可选正则。** 开启时从当前 scores 构造 detached 二值 Top-K 约束，计算 `BCE(soft_mask, hard_target)`；hard target 不替代任务路径中的 soft mask。关闭时跳过 BCE 计算，保留原有权重调度。
5. **BCE 按真实图片视觉 token 等权。** 使用带梯度的 loss sum 与有效 merger token count，在整个 optimizer mini-batch 上按 DP 汇总分母；排除视频、dummy image、SP 重复副本。独立归一化，不能再套用样本数缩放。日志使用相同 sum/count。
6. **最终冻结设置在通用冻结后、wrapping 前完成。** LIS 始终可训练，vision encoder/merger 始终冻结；非视觉参数遵循 `frozen/full`；teacher/ref 全冻结。默认教师 fixed、same_input、无 LIS，不改为 EMA 或 bbox 教师。
7. **Checkpoint 使用原生流程。** selector 导出读取原生 FSDP sharded state dict，再提取并物化四个 q/k 投影参数，不能保存分片参数视图。selector-only 只限定输出内容，不承诺峰值内存仅等于 LIS 大小。旧独立 wrapper 的 optimizer 分片可能不兼容，不能静默加载或把 selector 初始化称为完整恢复。

全保留预算的梯度断开不在当前处理范围。未经任务要求，不重写 trainer、不新增专用参数管理协议，不改变 VOPD loss、预算或教师语义。

## 文件组织

| 路径 | 职责 |
| --- | --- |
| `scripts/slsopd.sh` | 当前 LIS 实验入口，使用 `slsopd.yaml` |
| `scripts/run_vision_opd.sh` | 保留原版 VOPD 入口，使用原版 `vopd.yaml` |
| `scripts/prepare_vero35k.py` | Vero 元数据下载、图片物化、绝对路径 JSONL 构建 |
| `scripts/prepare_data.py` | 原版 Vision-OPD-6K 数据转 parquet |
| `tools/merge_checkpoint.sh` | 完整 FSDP actor 合并工具 |
| `verl/trainer/config/slsopd.yaml` | LIS 实验配置、原生 FSDP 和 vLLM architecture 配置 |
| `verl/trainer/config/vopd.yaml` | 原版 VOPD 预设，避免混入 LIS 配置 |
| `vision_opd_selector/core.py` | 普通 scorer、可微 Top-K、分图片 gating、辅助 BCE |
| `vision_opd_selector/fsdp_hook.py` | Actor 注入、最终 trainability、辅助统计与视频/dummy 处理 |
| `vision_opd_selector/budget.py` | 预算解析与分配 |
| `vision_opd_selector/vllm_plugin.py` | native vLLM architecture 注册 |
| `vision_opd_selector/vllm_model.py` | vLLM 视觉模型与正常权重加载对接 |
| `vision_opd_selector/checkpoint.py` | selector 加载、原生分片导出与兼容格式 |
| `verl/workers/fsdp_workers.py` | 模型构建、冻结、FSDP wrapping、actor 初始化 |
| `verl/workers/actor/dp_actor.py` | VOPD 更新、BCE 调度和全局 token 归一化 |
| `verl/trainer/ppo/ray_trainer.py` | rollout/预算、训练调度与 checkpoint 编排 |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | 完整 checkpoint 与 optimizer 分片兼容检查 |
| `tests/` | 数值、梯度、冻结、注册、权重同步及分布式回归 |
| `docs/` | 调查记录；带日期的审计可能描述修改前状态 |

`scripts/` 仅保留上述两个 shell 启动入口；Python 工具可继续放在其中。

## 核心执行流程

1. `slsopd.sh` 将环境变量和尾部 Hydra overrides 传给 `verl.trainer.main_ppo`，组合 `slsopd.yaml`。
2. Ray/VeRL 构建 actor/teacher/ref；pre-FSDP hook 为学生视觉模块注入 LIS。通用冻结后重新确定最终 trainability，再进行原生 FSDP wrapping。
3. vLLM 通过安装包 entry point 注册自定义 Qwen2.5-VL architecture，并通过框架正常路径接收 actor 权重。group 的每个回答默认使用相同的 0.2 预算。
4. 学生在 merger 后按图片分段，经 q/k scorer 得到 scores，产生 FP32 soft Top-K mask，再 gating 原视觉特征。教师使用同输入的完整视觉特征。
5. Actor 计算 VOPD 蒸馏目标；若启用 BCE，收集真实图片的辅助 sum/count，跨 micro-batch/DP 归一化后加入目标，反向和 optimizer 更新走默认流程。
6. Trainer 按原生机制同步、保存和恢复；selector artifact 可用于新实验初始化或 AngelSlim 评估。

## 修改与验证

对照参考仓库解释行为差异，优先局部修复。不要在未明确要求时修改参考仓库。更改启动默认值时同步 `guide.md`，保持原版入口和实验入口分离。

涉及训练正确性的修改应覆盖：关闭 BCE 时真实 VOPD 梯度与更新、开启时两项梯度、micro-batch/DP 切分一致性、冻结组合、CUDA BF16 autocast、原生 FSDP 导出/恢复。普通文档和脚本小改动只做适当检查。

本地测试与 GPU 验证分开报告。CPU/Gloo 或 mock vLLM 测试不能代替真实 Qwen/vLLM 两步训练、同步及恢复。不要把 skipped CUDA 测试视为通过，也不要把不同依赖环境的结果当成锁定环境验证。
