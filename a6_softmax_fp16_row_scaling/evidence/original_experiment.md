# A5/A6 FP16 Softmax 行数增长对照

## A5 补充结果（2026-09-16）

DV100（dav-3510），局部 UB 地址依赖模式，1024列、FP32输入及中间值、FP16输出。
使用与下方A6完全相同的VF-only源码，10组source SHA256逐项一致。
使用A5正式参数，未修改寄存器数、SHQ或时序配置。

| 行数 | 三循环 cycles | gap2 cycles | gap2增加 cycles | 增幅 |
|---|---:|---:|---:|---:|
| 8 | 548 | 618 | 70 | 12.77% |
| 12 | 792 | 874 | 82 | 10.35% |
| 16 | 1051 | 1135 | 84 | 7.99% |
| 24 | 1541 | 1645 | 104 | 6.75% |
| 32 | 2026 | 2148 | 122 | 6.02% |

A5这五组均是三循环更快，未出现A6在较大行数下的gap2收益。
这里只确认周期对比，尚未对A5差异作逐事件瓶颈归因。
指令计数与输出地址覆盖检查全部通过；未额外运行CAmodel。
下方UB容量及PK_B32范围建模限制同样适用。

结果独立保存于a5/summary.json及a5/rows*/{three_loop,gap2}/，含source与vfsim/local日志。
复现命令（仓库根目录运行）：

```bash
python3 results/ub_address_dependency_experiment/softmax_fp16_row_scaling/scripts/run.py --soc dav-3510 --output-subdir a5 --reuse-sources
```

## A6 结果

固定1024列，FP32输入及中间exp，FP16输出；DV121正式配置不变，
68物理寄存器、SHQ44，全部启用局部UB地址依赖。无新增优化。

| 行数 | 三循环 cycles | gap2 cycles | gap2节省 cycles | 降幅 |
|---|---:|---:|---:|---:|
| 8 | 568 | 570 | -2 | -0.35% |
| 12 | 833 | 784 | 49 | 5.88% |
| 16 | 1099 | 998 | 101 | 9.19% |
| 24 | 1631 | 1426 | 205 | 12.57% |
| 32 | 2162 | 1854 | 308 | 14.25% |

| 增长区间 | 三循环增量 cycles/row | gap2增量 cycles/row |
|---|---:|---:|
| 8 -> 16 | 66.375 | 53.5 |
| 16 -> 32 | 66.4375 | 53.5 |

增量显示gap2每行约节省12.9 cycles，持续处理成本降低约19.4%。
由8与32行端点得到的经验式：三循环 T≈66.4167R+36.6667；gap2 T=53.5R+142。
截距差约105 cycles，支持“融合吞吐更好，但短shape被固定成本抵消”的判断。
这些是经验拟合，不是逐事件分离出的真实head/tail耗时；不能将截距直接等同
模型固定VF头尾参数。两版模型参数相同。

## 范围与验证

- 每行实际展开16条VEXPDIF、16条VMUL、16条VCVT及1条VDIV，检查通过。
- 所有FP16输出store起始地址完整无重复，排除修改shape未生效或少算行。
- 8行结果复现568/570。
- 生成VF-only源码，删除原host wrapper，避免沿用固定UB地址造成重叠。
- 内存对象仍独立，中间依赖基于同base动态范围识别。
- 32行全量buffer需约320KiB（另三循环有256B统计量）；未验证实际SoC UB容量，
  未编译、未运行CAmodel。此点仅用于模型吞吐趋势，不承诺硬件单VF可执行。
- PK_B32最终写出span仍未实现，输出buffer无后续load；现有时序fallback未改。
- source和logs按rows8/rows16/rows32及three_loop/gap2归档。
- scripts/run.py、summary.json保存复现与完整结果，plots/row_scaling.png为总周期曲线。

## 五联 IPC 图

plots/ipc_five_rows_window10.png（另有SVG）从上到下为8、12、16、24、32行，
每个子图叠加三循环和gap2的整体计算完成IPC。窗口10 cycles，纵轴统一0到3。
各曲线以首次计算完成为零点，每个子图横轴范围按本shape单独设置，单位均为cycle。
图例是完整VF周期，而非对齐后的横轴终点。旧图未覆盖。
每个shape的CSV、事件计数/对齐起点JSON也位于plots目录。
脚本：scripts/plot_ipc_five_panels.py；新增case的源代码和日志在rows12、rows24。
