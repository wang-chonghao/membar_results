# SUM -> NORM 两行间距，A6 FP16 输出

保留 pair=128 元素交错、运行期行循环、FP32 输入与中间 exp，最终 VCVT -> PK_B32 输出 FP16。
新源码 cce_code/attention_softmax_giantloopfusion_fp16_gap2.cce；原版本不修改。
当前 shape (8,1024)。流水需要 rows >= 3，cols 是128的倍数，主机测试固定8行。

```text
M0
M1 + S0
M2 + S1
for i=3..rows-1: M(i) + S(i-1) + N(i-3)
S(rows-1) + N(rows-3)
N(rows-2)
N(rows-1)
```

两个倒数状态：norm_store 保存本轮 NORM 使用的广播倒数；sum_inv 保存较新行的标量倒数。
本轮 NORM 之后，先执行 vdup(norm_store,sum_inv)，再用 vdiv 重定义 sum_inv。
这把原本位于新 DIV 后面的广播移到下一轮、下一次 DIV 前，未增加广播或访存。
没有寄存器 alias 赋值。最初使用 norm_store=sum_store 的尝试被 canonical
loop-carried logical ID 校验拒绝，未运行出预测值；未修改该校验器。

| A6 局部地址依赖模式 | VF cycles |
|---|---:|
| 三循环 FP16 输出 | 568 |
| giant-loop 一行间距 | 776 |
| giant-loop 两行间距 | 570 |

相比一行间距减少206 cycles (26.55%)，比三循环多2 cycles (0.35%)。
稳态IPC低谷显著缩短，但启动、排空与部分稳态低谷仍存在，不称为全程满发射。
正式 SoC 参数、寄存器数68、SHQ44均不改。

## 验证

- 新旧 giant-loop 动态 opcode/form 数量完全一致，1321条非barrier指令，681条compute。
- 输出128次store的起始offset完整且不重复，覆盖8192个half。
- 根据真实 start 日志的 preg 依赖追踪：128条 NORM VMUL 均读取对应行倒数。
- A5 CAmodel 编译通过，VF 634 cycles；这不是A6实测时间。
- CAmodel原始FP16输出独立与FP64稳定softmax比较：8192个元素全部通过。
- 最大绝对误差9.535e-7，最大相对误差0.04672%，测试输入沿用现有确定性基准。
- 无显式membar，no_address_check目录不能当作全局同步硬件基准。
- 局部依赖模式保护中间UB；本轮0次地址依赖阻塞，不代表没有做检查。
- 最终128个PK_B32 store仍为unknown_span，输出buffer独立且无后续load。
- 时序仍使用当前参数fallback，结果是模型内实验，不是硬件收益承诺。

scripts/run.py、check_dependencies.py 和 archive_check.py 分别保存预测、依赖审计和golden检查。
summary.json 是预测结果；golden_summary.json 是独立数值校验；camodel 保存原始日志。
ipc/gap1_vs_gap2.png 和 CSV：计算完成事件，窗口10，首次计算完成对齐。
