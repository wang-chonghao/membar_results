# 显式 Spill CCE 重编译验证

`source/kernel.cce` 是完整可编译内核，保留原 host、GM/UB 布局和输入。每条 VF 指令标注原日志 PC。原用例在相邻 `camodel_case/`，未修改。

这是对应固定 shape/unroll 的硬件流回放，不是通用算法实现；标量地址效果已折入访存表达式，没有逐条重放原来的标量指令。

原版 / 显式版：2553 / 2541 cycles。完整动态数据流、地址、模式及屏障序列一致：True。输出逐字节一致：True。

核查不要求物理寄存器编号或立即数/寄存器寻址编码相同，透明 VMOV 和全真 PSET 归一化；不把周期接近或 golden 通过作为指令流一致的证明。详细差异见 `comparison.json`。

在仓库根目录配置并运行：

```bash
source ./set_env.sh
python3 gelu_grad/u8/explicit_spill_camodel_case/run.py --out-dir /tmp/gelu_grad_u8_explicit_rerun
```

新输出目录必须为空，不能覆盖归档目录。仅保留主要日志、输入输出和运行命令，不归档 VCD 等大文件。

## 单目录独立运行

可只复制本目录到新机器。完整文件清单和配置步骤见 [README.txt](README.txt)。编辑本目录 `set_env.sh` 后，在本目录执行：

```bash
bash ./run.sh --out-dir /tmp/gelu_grad_u8_explicit_rerun
```
