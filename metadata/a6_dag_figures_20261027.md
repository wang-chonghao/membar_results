# A6 Spill DAG 图

这组图沿用 A5 的指令顺序泳道画法，横轴是代码顺序，不是执行时间。
黄色为 load，绿色为 store，蓝色为计算；曲线仅表示寄存器 RAW、状态数据流和 UB RAW。
不展示地址复用的 WAR/WAW 顺序约束，但仿真仍保留这些约束。
局部依赖图中的红菱形仅标记原 Membar 位置，不参与调度或计时。

## 图片索引

| 算子 | Membar 优化 | 软件优化 | 说明 |
|---|---|---|---|
| AdamApplyOne U8 | [604 cycles](../figure/DAG/a6_adam_u8_local_instruction_order_20261027.png) | [455 cycles](../figure/DAG/a6_adam_u8_software_instruction_order_20261027.png) | 软件方案包含主动 UB 暂存和真实的全局 Membar。 |
| FlashAttention Grad U2 | [252 cycles](../figure/DAG/a6_flash_u2_local_instruction_order_20261027.png) | [207 cycles](../figure/DAG/a6_flash_u2_software_instruction_order_20261027.png) | 软件方案使用 Bank-aware 顺序，按两行分别展示。 |
| Logit Grad U3 | [229 cycles](../figure/DAG/a6_logit_u3_local_instruction_order_20261027.png) | [224 cycles](../figure/DAG/a6_logit_u3_software_instruction_order_20261027.png) | U3 内按 2+1 分组；整段为 5 组 U3 加 1 份 tail，图中仅展示首组。 |
| Logit Grad U4 | [390 cycles](../figure/DAG/a6_logit_u4_local_instruction_order_20261027.png) | [223 cycles](../figure/DAG/a6_logit_u4_software_instruction_order_20261027.png) | U4 代码体内按 2+2 分组，实际并行展开程度接近 U2。 |
| Logit Grad U8 | [297 cycles](../figure/DAG/a6_logit_u8_local_instruction_order_20261027.png) | [223 cycles](../figure/DAG/a6_logit_u8_software_instruction_order_20261027.png) | U8 代码体内按 2+2+2+2 分组，实际并行展开程度接近 U2。 |

软件分组方案属于允许任意指令重排的软件基准，不代表保持原始 U4/U8 并行调度程度的改进。
Logit 图只画首组循环体，省略循环外 VDUPS/PSET 初始化；节点数量是所展示范围的数量，cycle 是整段模型预测值。

## 时序口径

- 固定模型提交：`336844700d704bc3afdc4b190b215e736b65ebc9`，A6 / DV121。
- 所有图均不计时标量流水。Logit U8 原始日志另外含 57 条 SSTI 和 57 条 SLDI；297 cycles 明确排除了这些标量 spill 时序，不能视为完整硬件 VF 周期。
- Logit U8 按原 CAModel 日志恢复向量、谓词及 Membar 顺序，逐动态访问核对有效字节地址，不重新编译回放 CCE。本次不修改主汇总表。
- 软件方案已有 A6 VF 编译无 spill 检查；这不是 A6 simulator 运行或数值验证。
- 完整模型节点及依赖留在各文件夹的 `.json` 中；图片实际展示的节点和边在 `.view.json` 中。仅后者移除了 WAR/WAW 绘图连线。

## 复现

在结果仓根目录运行，`--model-root` 指向上述模型提交的检出目录。Python 需要安装 `matplotlib` 和 `jsonschema`，并指定可用的中文字体。

```bash
python3 -m unittest discover -s scripts -p test_a6_vector_spill_dags.py
python3 scripts/plot_a6_vector_spill_dags.py \
  --model-root /path/to/model-at-3368447 \
  --font /path/to/chinese-font.ttf \
  --cases adam_u8 flash_u2 logit_u3 logit_u4 logit_u8 \
  --render-only --replace-generated
```

核验目录：

- [Adam / Flash 图及模型数据](a6_vector_spill_dags_20261027/summary.json)
- [Logit U3 图及模型数据](a6_logit_u3_dags_20261027/summary.json)
- [Logit U4 图及模型数据](a6_logit_predicate_dags_20261027/summary.json)
- [Logit U8 图及模型数据](a6_logit_u8_dags_20261027/summary.json)
- [Logit U8 原始日志投影和地址核验](a6_logit_u8_dags_20261027/input_audit.json)
