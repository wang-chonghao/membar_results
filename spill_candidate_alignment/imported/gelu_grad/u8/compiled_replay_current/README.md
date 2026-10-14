# GeLU Grad U8 完整指令流预测

A5 / FP32，4096 个元素，源码循环 64 次，unroll=8；硬件 VLOOP 重复 8 次。

| CAModel | VfSim 全局 Membar | VfSim 局部依赖 | 全局预测精度 | 局部周期减少 |
|---:|---:|---:|---:|---:|
| 2553 | 2421 | 862 | 94.83% | 64.39% |

使用原始归档 instr/popped 日志生成 Canonical JSON，包含 vector spill 写/读 40/40、predicate spill 写/读 24/24、128 条 Membar；未使用缺少 spill 的源码级时间。原 `hardware_equivalent.cce` 重编译会多两条 VMOV，未用其编译日志替代原指令流。

原 CCE 复跑仍为 2553 cycles，4096 个元素通过 golden（最大绝对误差 4.76837e-7），输出与原归档逐字节相同，原/新编译动态 PC、opcode、operand 和顺序一致。

局部模型提交 `0060620` 的全局侧与当前模型均为 2421。两侧使用相同动态身份/取指次序；320 次访存全部与 CAModel UB 日志核对；512 个历史 RAW/WAR overlap 检查满足 done+1，224 个重叠 store 完成顺序检查通过，无保守 fallback。局部周期是研究假设的预测，不是现有硬件 CAModel 实测。

Scalar 地址算术仍未计时；本例没有 scalar spill。当前模型及实验模型 SHA256 见 `provenance.json`。

- `current/canonical/vf_000.json`：正式全局输入。
- `experimental/canonical/local.json`：移除方向性 Membar 后的局部输入。
- `experimental/compiled_vf_replay.cce`：含 PC、动态 ID 的审计清单，不用于重新编译替代原二进制。
- `experimental/memory_address_audit.json`、`camodel_evidence/`：320 次实际地址核对证据。
- `original_stream_audit.json`、`prediction.json`：数值、原始指令流、周期及 A/B 审计。

复现：

```bash
python3 results/membar_results/gelu_grad/u8/camodel_case/run.py \
  --toolkit /path/to/cann-9.0.0-beta.1 \
  --out-dir /tmp/gelu-grad-u8-prediction/camodel
python3 cce_code/spill_candidate_alignment/fill_historical_predictions.py \
  --model-root /path/to/ub-model-0060620
```

CAModel 输出目录必须不存在或为空；该工具复用原用例的 source、host、输入、golden 和编译选项。汇总表只更新本行，不覆盖原历史数据源及软件优化结果。
