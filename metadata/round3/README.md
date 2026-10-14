# 第三轮候选实测：11 个新 VF 入口

本轮仅新增测试与编译日志回放工具，未修改生产 Core 或 ISA/timing 参数。源版本和 SHA256 见各 case 的 provenance.json 及候选索引。

## 结果口径

- 11 个 U1 基准均通过 CAModel 数值验证：与所选原 AscendC 入口逐位一致，并通过独立 golden 与 GM 越界 guard。
- 10 个入口另测 U2/U4/U8，共 30 个版本；26 个数值通过，4 个 VF stack 超限编译失败。失败版本不纳入性能表。
- Compressor 保留源码的 4 路 Sc 分块及真实 Reduce 回边，本次没有把 Reduce 循环当成独立数据循环做 AABB unroll。
- 29 个版本完成 VfSim 全局预测和同流局部 A/B；其余明确记录转换拒绝点，不填 0，不删 spill 指令。
- 全局/局部使用同一份真实 RV 指令流、相同算术与次序。局部结果仅在实验模型全局结果与当前主线完全一致后采用；非零访存与 CAModel UB 日志核对，align 缓冲写出按封存组核对。全零 mask store 保留名义范围作保守阻塞，并单独记录，不声称发生真实写入。RAW/WAR 按 producer done+1 放行。
- 精度 = 100 × (1 - |VfSim全局 - CCE CAModel| / CCE CAModel)，不是数值正确率。局部时间是假设局部依赖硬件的性能估计，没有该硬件的 CAModel 精度基准。
- V/P/S 分别是 vector、predicate、scalar spill 写出/读回数。V 是经地址配对确认的编译器额外访问，不把只读常量重载计为 spill。
- 测试对象是指定子 VF，不含完整算子的前置统计、GM 通信或 Cube；scalar 控制保留日志但不计入 Core 时间。
- 交叉熵使用真实 48×128 行统计，测试后 64 列的输出子 VF；loss 累加器由前 64 列的 partial 生成，不是任意旧数值。

## 基准

| 入口 | 形状 / 循环 | AscendC CAModel | CCE CAModel | VfSim全局 | 精度 | VfSim局部 | spill 写/读 |
|---|---|---:|---:|---:|---:|---:|---|
| [GroupNormSwishGrad dx (pair)](cases/groupnorm_swish_dx/cce/kernel.cce) | 6144; I48 | 1645 | 1645 | 1571 | 95.50% | 1571 | V 0/0; P 0/0; S 0/0 |
| [LogSigmoid forward](cases/logsigmoid/cce/kernel.cce) | 3072; I48 | 544 | 567 | 522 | 92.06% | 522 | V 0/0; P 0/0; S 0/0 |
| [LayerNormGrad V3 dx](cases/layernorm_dx/cce/kernel.cce) | 3072; I48 | 319 | 319 | 283 | 88.71% | 283 | V 0/0; P 0/0; S 0/0 |
| [Softplus DAG / fused CCE](cases/softplus/cce/kernel.cce) | 3072; I48 | 1869 | 515 | 541 | 94.95% | 541 | V 0/0; P 0/0; S 0/0 |
| [MoE single-expert Softplus](cases/moe_softplus/cce/kernel.cce) | 3072; I48 | 771 | 771 | 739 | 95.85% | 739 | V 0/0; P 0/0; S 0/0 |
| [FusedCausalConv1D residual](cases/causal_conv/cce/kernel.cce) | 49×192; I24 | 1266 | 1266 | 1156 | 91.31% | 1156 | V 0/0; P 0/0; S 0/0 |
| [RoPE Grad half](cases/rope_grad/cce/kernel.cce) | 1×1×48×128; I48 | 439 | 439 | 392 | 89.29% | 392 | V 0/0; P 0/0; S 0/0 |
| [RMSNormGrad dx](cases/rmsnorm_dx/cce/kernel.cce) | 3072; I48 | 218 | 218 | 227 | 95.87% | 227 | V 0/0; P 0/0; S 0/0 |
| [SoftmaxCrossEntropy loss/backprop partial](cases/softmax_ce/cce/kernel.cce) | 3072; I48 | 356 | 356 | 333 | 93.54% | 333 | V 0/0; P 0/0; S 0/0 |
| [MHC Pre partial merge](cases/mhc_partial/cce/kernel.cce) | 4×48×64; I48 | 1145 | 1145 | 613 | 53.54% | 613 | V 0/0; P 0/0; S 0/0 |
| [Compressor DND Softmax](cases/compressor_softmax/cce/kernel.cce) | 4×8×128; I1 | 1182 | 1183 | 1149 | 97.13% | 418 | V 17/42; P 0/0; S 0/0 |

## Unroll 实测

GN 的 U 是对原 pair 继续展开：U2 实际每轮处理 4 VL，U4 为 8 VL；CausalConv 每个源码 xLoop 产生四个 half-vector 输出。
RoPE、交叉熵和 MHC 只展开独立外层行，保留行内的加载及累加顺序；不是对真实 Reduce/K 回边做独立重排。RoPE 的指针/标量控制开销也随之增加。

| 入口 | U | CCE CAModel | VfSim全局 | 精度 | VfSim局部 | 周期减少 | spill 写/读 | Membar数 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| [GroupNormSwishGrad dx (pair)](cases/groupnorm_swish_dx/u2/cce/kernel.cce) | 2 | 1499 | 1370 | 91.39% | 1370 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [GroupNormSwishGrad dx (pair)](cases/groupnorm_swish_dx/u4/cce/kernel.cce) | 4 | 12519 | 12638 | 99.05% | 1430 | 88.68% | V 156/156; P 96/384; S 0/0 | 792 |
| [LogSigmoid forward](cases/logsigmoid/u2/cce/kernel.cce) | 2 | 532 | 518 | 97.37% | 518 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [LogSigmoid forward](cases/logsigmoid/u4/cce/kernel.cce) | 4 | 1173 | 1099 | 93.69% | 520 | 52.68% | V 0/0; P 24/24; S 0/0 | 48 |
| [LogSigmoid forward](cases/logsigmoid/u8/cce/kernel.cce) | 8 | 5831 | 5907 | 98.70% | 537 | 90.91% | V 49/54; P 78/174; S 0/0 | 355 |
| [LayerNormGrad V3 dx](cases/layernorm_dx/u2/cce/kernel.cce) | 2 | 314 | 283 | 90.13% | 283 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [LayerNormGrad V3 dx](cases/layernorm_dx/u4/cce/kernel.cce) | 4 | 343 | 279 | 81.34% | 279 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [LayerNormGrad V3 dx](cases/layernorm_dx/u8/cce/kernel.cce) | 8 | 2479 | 未预测 | -- | 未预测 | -- | V 60/60; P 0/0; S 133/114 | 120 |
| [Softplus DAG / fused CCE](cases/softplus/u2/cce/kernel.cce) | 2 | 542 | 529 | 97.60% | 529 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [Softplus DAG / fused CCE](cases/softplus/u4/cce/kernel.cce) | 4 | 547 | 527 | 96.34% | 527 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [Softplus DAG / fused CCE](cases/softplus/u8/cce/kernel.cce) | 8 | 3015 | 2995 | 99.34% | 532 | 82.24% | V 0/0; P 48/150; S 0/0 | 198 |
| [MoE single-expert Softplus](cases/moe_softplus/u2/cce/kernel.cce) | 2 | 756 | 743 | 98.28% | 743 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [MoE single-expert Softplus](cases/moe_softplus/u4/cce/kernel.cce) | 4 | 1586 | 1074 | 67.72% | 725 | 32.50% | V 0/0; P 24/24; S 0/0 | 48 |
| [MoE single-expert Softplus](cases/moe_softplus/u8/cce/kernel.cce) | 8 | 6137 | 未预测 | -- | 未预测 | -- | V 0/0; P 96/300; S 28/24 | 396 |
| [FusedCausalConv1D residual](cases/causal_conv/u2/cce/kernel.cce) | 2 | 4309 | 4361 | 98.79% | 1095 | 74.89% | V 50/231; P 0/0; S 0/0 | 281 |
| [RoPE Grad half](cases/rope_grad/u2/cce/kernel.cce) | 2 | 1693 | 未预测 | -- | 未预测 | -- | V 0/0; P 0/0; S 0/0 | 0 |
| [RoPE Grad half](cases/rope_grad/u4/cce/kernel.cce) | 4 | 3447 | 未预测 | -- | 未预测 | -- | V 0/0; P 0/0; S 84/84 | 0 |
| [RoPE Grad half](cases/rope_grad/u8/cce/kernel.cce) | 8 | 5891 | 未预测 | -- | 未预测 | -- | V 0/0; P 0/0; S 234/234 | 0 |
| [RMSNormGrad dx](cases/rmsnorm_dx/u2/cce/kernel.cce) | 2 | 219 | 230 | 94.98% | 230 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [RMSNormGrad dx](cases/rmsnorm_dx/u4/cce/kernel.cce) | 4 | 267 | 236 | 88.39% | 236 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [RMSNormGrad dx](cases/rmsnorm_dx/u8/cce/kernel.cce) | 8 | 1597 | 未预测 | -- | 未预测 | -- | V 0/0; P 24/72; S 28/24 | 96 |
| [SoftmaxCrossEntropy loss/backprop partial](cases/softmax_ce/u2/cce/kernel.cce) | 2 | 391 | 334 | 85.42% | 334 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [SoftmaxCrossEntropy loss/backprop partial](cases/softmax_ce/u4/cce/kernel.cce) | 4 | 1105 | 未预测 | -- | 未预测 | -- | V 0/0; P 0/0; S 52/48 | 0 |
| [SoftmaxCrossEntropy loss/backprop partial](cases/softmax_ce/u8/cce/kernel.cce) | 8 | 4312 | 未预测 | -- | 未预测 | -- | V 0/0; P 0/0; S 252/216 | 0 |
| [MHC Pre partial merge](cases/mhc_partial/u2/cce/kernel.cce) | 2 | 1291 | 611 | 47.33% | 611 | 0.00% | V 0/0; P 0/0; S 0/0 | 0 |
| [MHC Pre partial merge](cases/mhc_partial/u4/cce/kernel.cce) | 4 | 2313 | 1879 | 81.24% | 661 | 64.82% | V 48/48; P 0/0; S 0/0 | 96 |

## 限制与真实差异

- Softplus 原 DAG 在禁用自动 VF 融合时是 12 个 VF，CCE 是单个融合 VF。输出与算术一致，二者总耗时差包含 VF 融合和中间搬运差异；不能直接称为 spill 收益。
- LogSigmoid 原 SDK 把零常量生成改成 VBR，CCE 是 PSET + VDUP；Compressor 也有 VBR/VDUP 的 1-cycle 差异。VfSim精度统一与对应 CCE 日志比较，不混用 AscendC 周期。
- MHC 使用原 Reg::Load/Store 的 unaligned 序列，保留 VLDAS/VLDUS/VSTUS/VSTAS，不替换为普通 load/store。现已通过手工日志对照 JSON 补齐 U1/U2/U4，按封存状态组校验 VSTUS/VSTAS 实际写出；scalar 运算仍未计时。
- CausalConv U1/U2 的手工 JSON 保留原 VLOOP，以显式有效地址表达编译器指针重置，不继续累加旧 POST_UPDATE。全部动态指令、数值和 VF 周期经原 CCE 复跑确认一致。全零 predicate 的尾部 store 保留指令，使用保守名义范围，证据和数量单列。
- CausalConv U2 同时发生源数据重用与 Cast 消除，V 50/231 是 stack 地址的配对写/读数，不可简单等同于总 load 数相对 U1 的差值。
- RoPE U2 的 scalar compare 控制及各算子含 SSTI/SLDI 的 scalar spill 版本暂未回放；已有完整 CAModel、数值和失败诊断。
- Compressor U1 已有 17 次 vector 写出、42 次读回和 60 条 Membar。其局部收益也包含源码原有 softmax 阶段间同步被局部化，不是纯 spill 消除收益。
- 本轮没有校准未知 timing；model_warnings.json 保留 forwarding/II 等 fallback。预测精度较低的 MoE U4 等版本应先研究时序误差，不能视作硬件收益的已实测证明。

### 转换拒绝点

| 入口 | U | 拒绝原因 |
|---|---:|---|
| LayerNormGrad V3 dx | 8 | No vector/predicate destination contract for RV_SSTI |
| MoE single-expert Softplus | 8 | No vector/predicate destination contract for RV_SSTI |
| RoPE Grad half | 2 | No vector/predicate destination contract for RV_SCMP |
| RoPE Grad half | 4 | No vector/predicate destination contract for RV_SSTI |
| RoPE Grad half | 8 | No vector/predicate destination contract for RV_SSTI |
| RMSNormGrad dx | 8 | No vector/predicate destination contract for RV_SSTI |
| SoftmaxCrossEntropy loss/backprop partial | 4 | No vector/predicate destination contract for RV_SSTI |
| SoftmaxCrossEntropy loss/backprop partial | 8 | No vector/predicate destination contract for RV_SSTI |

### VF stack 超限（不计入性能结果）

| 入口 | U | 编译器要求 stack bytes | 上限 |
|---|---:|---:|---:|
| GroupNormSwishGrad dx (pair) | 8 | 16416 | 6144 |
| FusedCausalConv1D residual | 4 | 11072 | 6144 |
| FusedCausalConv1D residual | 8 | 33568 | 6144 |
| MHC Pre partial merge | 8 | 9280 | 6144 |

## 复现与证据

CANN 9.0.0-beta.1；目标 Ascend950PR_9599 / dav-c310-vec。编译使用 `-O2 --cce-simd-vf-fusion=false -mllvm -cce-aicore-vec-misched=0`。

```bash
export CLANG_FORMAT=/path/to/clang-format
python3 cce_code/spill_candidates_round3/run.py --jobs 3
python3 cce_code/spill_candidates_round3/unroll.py --factors 2 4 8 --jobs 3
python3 cce_code/spill_candidates_round3/predict.py --local --jobs 3
python3 cce_code/spill_candidates_round3/manual_json.py
python3 cce_code/spill_candidates_round3/report.py
```

CAModel 环境和工具链路径沿用 spill_candidate_alignment/run.py；局部预测依赖 `/tmp/spill-ub-model-0060620` 隔离快照。后者不是新 clone 自动存在的文件，缺少时只能运行全局或显式准备实验模型。

每个 case 包含原始头文件、provenance、布局、golden、CCE/AscendC、二进制输入输出、host、commands、compile/run 和 core0 instr/popped/IDU 日志；uN 子目录存变体。
prediction/ 保存 canonical JSON、原 RV 文本、地址/顺序审计、start/done/IDU/trace 和 warning，不保存大体积 sim_history。
prediction_provenance.json 记录当前 dirty worktree 的模型文件 SHA256，并确认测量期间未变化。数据不是只由 HEAD 提交号决定的。
