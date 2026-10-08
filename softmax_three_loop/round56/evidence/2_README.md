# Softmax Round 56 VDup + VSTS 变体

本变体基于 `round_56_minimal_r19_mte3`，将两组 scalar 收集从
`VSTUS/VSTAS` 改为普通 `VSTS(ONEPT_B32)`。行最大值 reduction 的最低
lane 直接写入 UB，不再先执行无必要的广播：

```text
VCMAX -> VSTS(ONEPT_B32)
```

倒数 scalar 路径仍保留 `VDUP(POS_LOWEST) -> VSTS`。作为 `VDIV` 分子的
常量 `1.0f` 改由 `vec_one` 保存，并在 Phase 2 循环外只初始化一次：

```text
VDUP(vec_one, 1.0f)
for each row: VCADD -> VDIV(vec_one, sum) -> VDUP -> VSTS
```

## 结果

| 模型 | VF 周期 | 相对 CAmodel 误差 |
|---|---:|---:|
| CAmodel | 654 | - |
| VfSim 全局 Membar | 638 | -16 cycle，2.45% |
| VfSim 局部 UB 地址依赖 | 524 | -130 cycle，19.88% |

局部地址依赖相对全局 Membar 节省 114 cycle，加速 1.2176x。原始
VSTUS/VSTAS 版本的 CAmodel 时间为 639 cycle，因此该 CCE 改写在当前
硬件模型上增加 15 cycle。

Golden mismatch 为 0。独立检查的最大绝对误差约为
`3.91e-10`，最大相对误差约为 `1.79e-7`。

EXU/指令日志确认改写版本包含 8 条动态 `RV_VDUP`、1 条 `RV_VDUPS`
和对应的普通 `RV_VST` scalar store，不再包含 `VSTUS/VSTAS`。显式提升
循环不变量后，CAmodel 与两种 VfSim 模式均包含 529 个计算完成事件。

## UB 地址依赖统计

| 指标 | 结果 |
|---|---:|
| 动态非 Membar 指令 | 1201 |
| 精确地址覆盖率 | 100% |
| 动态阻塞依赖边 | 0 |
| fallback 依赖边 | 0 |

动态阻塞依赖边为 0，是因为同地址 consumer 参与发射仲裁时，对应的
producer store 已经完成；局部模式的收益来自不同 row 之间的阶段重叠。

## IPC 口径

- 只统计计算指令完成事件。
- CAmodel 使用 EXU dump 的 `retire` 周期。
- VfSim 使用 `done_by_cycle.json` 的完成周期。
- 10-cycle trailing sliding window，纵轴固定为 0 到 2 IPC。
- CAmodel 时间轴按第一条计算指令完成事件与 VfSim 对齐。

IPC 图：`ipc/compute_completion_ipc_window10.png`。
