# Logit Grad U4 软件优化 DAG

依据实际编译日志和已验证 Canonical 回放重建，不是根据源码画出的理想调度。实际动态访存地址逐条与 RV 的 Sn/Sn2/Am 字段核对。
首个 U4 循环组共 80 条指令：8 VLD、24 谓词生成/计算、44 向量计算、4 VST。循环重复四次，无 PSTS、PLDS 或 Membar。
软件策略为 2+2 分组：源码先安排第0、1路的交错计算与输出，再安排第2、3路。竖虚线只标记排列分组，不表示硬件必须等待前组完成。
比较指令靠近消费者，以缩短掩码的存活区间；不是维持原始四路 AABB 排列下的局部修补，也不是完全退回 U1。
横轴保留程序顺序，曲线表示真实向量/谓词 RAW；不表示执行周期。初始化 PSET、VDUP 位于循环外，完整动态指令和依赖保存在 JSON。
既有测量：CAModel 216、VfSim 228 cycles。本次没有重跑预测。

| 范围 | PNG | SVG |
| --- | --- | --- |
| 总览 1–80 | [查看](../../../figure/DAG/logit_grad_u4_software_order.png) | [查看](../../../figure/DAG/logit_grad_u4_software_order.svg) |
| 第0、1路 1–40 | [查看](../../../figure/DAG/logit_grad_u4_software_order_part01.png) | [查看](../../../figure/DAG/logit_grad_u4_software_order_part01.svg) |
| 第2、3路 41–80 | [查看](../../../figure/DAG/logit_grad_u4_software_order_part02.png) | [查看](../../../figure/DAG/logit_grad_u4_software_order_part02.svg) |

[编号、PC、操作数与字节地址](logit_grad_u4_software_order.csv) · [依赖及来源校验值](logit_grad_u4_software_order.json)
