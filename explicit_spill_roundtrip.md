# 显式 Spill CCE 重编译验证

日期：2026-10-20。A5 / Ascend950PR_9599，CANN 9.0.0-beta.1，O2、vec-misched=0。

原 CCE 不动；在每个版本下增加 `explicit_spill_camodel_case/`，包含完整 kernel、host、运行配置、PC 对照、输出及两份主要 CAModel 日志。

| Case | 原版 cycles | 显式版 cycles | 差值 | Vector 写/读 | Predicate 写/读 | 原版/显式 Membar | 动态流核对 | 数值校验 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| [adam_apply_one u6](adam_apply_one/u6/explicit_spill_camodel_case/source/kernel.cce) | 1435 | 1438 | +3 | 32 / 32 | 0 / 0 | 64 / 64 | 一致 | golden 通过，原输出逐字节相同 |
| [gelu_poly u12](gelu_poly/u12/explicit_spill_camodel_case/source/kernel.cce) | 6037 | 6050 | +13 | 160 / 160 | 0 / 0 | 320 / 320 | 一致 | golden 通过，原输出逐字节相同 |
| [swiglu_grad u6](swiglu_grad/u6/explicit_spill_camodel_case/source/kernel.cce) | 3281 | 3283 | +2 | 80 / 80 | 0 / 0 | 160 / 160 | 一致 | golden 通过，原输出逐字节相同 |
| [swiglu_grad u8](swiglu_grad/u8/explicit_spill_camodel_case/source/kernel.cce) | 8243 | 8233 | -10 | 205 / 216 | 0 / 0 | 421 / 421 | 一致 | golden 通过，原输出逐字节相同 |
| [gelu_grad u8](gelu_grad/u8/explicit_spill_camodel_case/source/kernel.cce) | 2553 | 2541 | -12 | 40 / 40 | 24 / 24 | 128 / 128 | 一致 | golden 通过，原输出逐字节相同 |

## 未完成项

本轮尚未通过重编译验证：adam_apply_one u8。不能把这些版本视为等效实现。
AdamApplyOne U8 的重建源码会被 CCEC 再次插入 EPS spill，覆盖原显式 spill 槽，导致 golden 失败；已尝试共享基址、16 位循环、补码偏移、移除 bitcast、SSA 临时变量和寄存器分配开关，尚未解决。失败源码不作为可用等效用例发布；原 U8 及其旧结果不变。

## 核对口径

- 不是只数指令：比较整个 VF 所有动态迭代的计算 RAW、谓词依赖、VAXPY 旧 destination、标量常数位模式、实际 UB 字节地址、访存 mode 与 Membar 顺序。
- 寄存器重编号、VLD/VLDS/VLDI 与 VST/VSTS/VSTI 的寻址编码、透明 VMOV 和相同 PAT_ALL PSET 归一化；B8 spill 拷贝不改成 B32 模式，PSTS/PLDS 保留真实谓词搬运。
- 表中写/读是整个 VF 的动态 spill 指令次数，不是寄存器个数；根据本组 UB 布局的高地址 spill 区分类，不能当作任意 kernel 的通用识别规则。
- **数据流/地址等效不等于周期或机器编码逐位一致**。地址表达式的重新物化、标量初始化/标量 spill 和编译器分配仍可能变化；不改模型参数，不覆盖旧 CAModel/VfSim 记录。GeLU Poly 原版含标量 spill，该部分只保留地址效果，不逐条重放时序。
- 每个 `comparison.json` 提供原/新动态 opcode 数量、spill 数量、源码和日志 SHA256、数值检查及差异。失败试跑不作为等效结果归档。

## 复现

在结果仓库根目录修改 `set_env.sh` 的 CANN 路径后：

```bash
source ./set_env.sh
python3 adam_apply_one/u6/explicit_spill_camodel_case/run.py --out-dir /tmp/adam_u6_explicit_check
```

其他版本替换路径即可，输出目录必须为空。精度通过后，将新目录交给核查脚本：

```bash
python3 scripts/compare_explicit_spill.py --work-dir /tmp/your_runs --require-all
```

`--work-dir` 下按 `<case>/<uN>/` 放运行目录，可以重复指定多个工作目录；`--require-all` 要求本轮六组均已通过数值和动态流核对。
生成工具为 `scripts/explicit_spill_roundtrip.py`，只接受本轮已验证的单层、固定循环、仿射地址模式；未覆盖 opcode/模式直接报错。
归档不包含全量 VCD 或 VfSim history 大文件。
