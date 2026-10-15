# FlashAttention Grad 指令顺序图

`figure/DAG/` 只保留按编译后程序顺序绘制的图片，不再保留旧的 dataflow、hardware_group、full_vf 图片及 PDF。

| 版本 | 总览 PNG | 总览 SVG | 编号与依赖说明 |
| --- | --- | --- | --- |
| U1 | [查看](../../../figure/DAG/u1_instruction_order.png) | [查看](../../../figure/DAG/u1_instruction_order.svg) | [说明](u1_instruction_order.md) |
| U2 含 Spill | [查看](../../../figure/DAG/u2_instruction_order.png) | [查看](../../../figure/DAG/u2_instruction_order.svg) | [五张分段图及说明](u2_instruction_order.md) |

横轴为编译后循环体的程序顺序，不是执行 cycle。黄色表示 load/reload，绿色表示 store，蓝色表示计算，红色菱形表示 Membar，灰色表示标量辅助指令。曲线表示向量 RAW 和 spill UB RAW 依赖。

U1 展示首轮 49 条指令，硬件重复 8 次。U2 展示首个双行展开组的 230 条指令，硬件重复 4 次；五张细节图各包含连续 46 条指令。

U2 每组有 21 次向量 spill 保存和 21 次重载，以及 42 条 Membar。完整日志还存在 48 条 RV_SSTI 和 48 条 RV_SLDI，不能把旧表中的“仅 Vector spill”理解为没有标量保存/恢复。

原始 graph JSON、DOT、编号对照和顺序图 PDF 留在本目录，保留分析依据；本次不修改测量周期。

## 重绘顺序图

在结果仓根目录执行，需要 matplotlib 和中文字体：

```bash
python3 scripts/plot_u1_instruction_order.py --overwrite
python3 scripts/plot_u2_instruction_order.py --overwrite
```
