# A6 谓词依赖 DAG 补充

## 图片

| Case | 图片 | 完整 VF 周期 | 展示范围 |
|---|---|---:|---|
| LogSigmoidGrad U4，Membar 优化 | [DAG](../figure/DAG/a6_logsigmoid_u4_local_instruction_order_20261028.png) | 380 | 首组 U4，保留 2 条 PSTS、2 条 PLDS 和 4 个原 Membar 位置标记。 |
| LogSigmoidGrad U4，软件优化 | [DAG](../figure/DAG/a6_logsigmoid_u4_software_instruction_order_20261028.png) | 386 | 首组 U4，保持四路展开，不使用 U2 分组。 |
| LogSigmoidGrad U1 | [DAG](../figure/DAG/a6_logsigmoid_u1_original_instruction_order_20261028.png) | 499 | 首轮 12 条指令；完整 VF 为 48 轮。 |
| Logit Grad U1 | [DAG](../figure/DAG/a6_logit_u1_original_instruction_order_20261028.png) | 231 | 首轮 20 条指令；完整 VF 为 16 轮。 |

所有图使用固定 A6 / DV121 模型提交 `336844700d704bc3afdc4b190b215e736b65ebc9`，复算周期和归档一致。
横轴表示源码指令顺序，不表示实际开始时间；图中不含 WAR/WAW 地址复用约束。
循环外 PSET/VDUPS 初始化省略，U1 的每个节点标有实际指令名称。

两个 U1 均无 spill 和 Membar，仅用于比较算法本身的数据依赖，不能直接代表 U4 的 spill 效果。
两者完整 VF 的迭代次数不同，不能直接按总周期横向比较性能。
U4 局部依赖图中的 Membar 为位置参考，不参与调度和计时。
A6 软件方案仅通过 VF 编译无 spill 检查，不声称做过 A6 simulator 数值验证；标量时序不计入模型。

## 复现

在结果仓根目录执行，模型目录应检出上述固定提交；Python 需安装 matplotlib 和 jsonschema。

```bash
python3 -m unittest discover -s scripts -p test_a6_vector_spill_dags.py
python3 scripts/plot_a6_vector_spill_dags.py \
  --model-root /path/to/model-at-3368447 --font /path/to/chinese-font.ttf \
  --cases logsigmoid_u4 --render-only --replace-generated
python3 scripts/plot_a6_vector_spill_dags.py \
  --model-root /path/to/model-at-3368447 --font /path/to/chinese-font.ttf \
  --cases logsigmoid_u1 logit_u1 --variants original --render-only --replace-generated
```

去掉 `--render-only` 可以复算并核对动态发射流。配套核验数据：
[LogSigmoidGrad U4](a6_logsigmoid_u4_dags_20261028/summary.json)、
[两个 U1](a6_u1_predicate_dags_20261028/summary.json)。
`.json` 保留完整模型节点和依赖，`.view.json` 记录图片实际显示的节点、编号和数据流边。
