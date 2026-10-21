# 显式 Spill CCE 重编译验证

## 当前默认源码

当前默认 source/kernel.cce 已注释全部 mem_bar(...)，用于无显式 Membar 实验；尚未重新运行 CAModel，周期和精度待验证。source/kernel_with_membar.cce 保留修改前源码；comparison.json、run_output/ 和下方旧周期/精度记录仅对应这个带屏障版本。pipe_barrier、set_flag/wait_flag 和其他指令不变。

以下表格保留带屏障版本的历史验证结果，不代表当前无屏障版本。

日期：2026-10-20。A5 / Ascend950PR_9599，CANN 9.0.0-beta.1，O2、vec-misched=0。

原 CCE 不动；在每个版本下增加 `explicit_spill_camodel_case/`，包含完整 kernel、host、运行配置、PC 对照、输出及两份主要 CAModel 日志。

| Case | 原版 cycles | 显式版 cycles | 差值 | Vector 写/读 | Predicate 写/读 | 原版/显式 Membar | 动态流核对 | 数值校验 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| [adam_apply_one u6](adam_apply_one/u6/explicit_spill_camodel_case/source/kernel_with_membar.cce) | 1435 | 1438 | +3 | 32 / 32 | 0 / 0 | 64 / 64 | 一致 | golden 通过，原输出逐字节相同 |
| [gelu_poly u12](gelu_poly/u12/explicit_spill_camodel_case/source/kernel_with_membar.cce) | 6037 | 6050 | +13 | 160 / 160 | 0 / 0 | 320 / 320 | 一致 | golden 通过，原输出逐字节相同 |
| [swiglu_grad u6](swiglu_grad/u6/explicit_spill_camodel_case/source/kernel_with_membar.cce) | 3281 | 3283 | +2 | 80 / 80 | 0 / 0 | 160 / 160 | 一致 | golden 通过，原输出逐字节相同 |
| [swiglu_grad u8](swiglu_grad/u8/explicit_spill_camodel_case/source/kernel_with_membar.cce) | 8243 | 8233 | -10 | 205 / 216 | 0 / 0 | 421 / 421 | 一致 | golden 通过，原输出逐字节相同 |
| [gelu_grad u8](gelu_grad/u8/explicit_spill_camodel_case/source/kernel_with_membar.cce) | 2553 | 2541 | -12 | 40 / 40 | 24 / 24 | 128 / 128 | 一致 | golden 通过，原输出逐字节相同 |

## 完整用例入口

以下五个目录均可独立重新编译并执行 CAModel；原版仍保留在同级的 `camodel_case/`，不覆盖原源码或历史结果。

| Case | 等效 CCE | 配套运行说明 |
|---|---|---|
| AdamApplyOne U6 | [kernel.cce](adam_apply_one/u6/explicit_spill_camodel_case/source/kernel.cce) | [README](adam_apply_one/u6/explicit_spill_camodel_case/README.md) |
| GeLU poly U12 | [kernel.cce](gelu_poly/u12/explicit_spill_camodel_case/source/kernel.cce) | [README](gelu_poly/u12/explicit_spill_camodel_case/README.md) |
| SwiGLU Grad U6 | [kernel.cce](swiglu_grad/u6/explicit_spill_camodel_case/source/kernel.cce) | [README](swiglu_grad/u6/explicit_spill_camodel_case/README.md) |
| SwiGLU Grad U8 | [kernel.cce](swiglu_grad/u8/explicit_spill_camodel_case/source/kernel.cce) | [README](swiglu_grad/u8/explicit_spill_camodel_case/README.md) |
| GeLU Grad U8 | [kernel.cce](gelu_grad/u8/explicit_spill_camodel_case/source/kernel.cce) | [README](gelu_grad/u8/explicit_spill_camodel_case/README.md) |

每个 `explicit_spill_camodel_case/` 的组织一致：

```text
source/kernel.cce              完整 CCE，显式写入 spill 存取和屏障
source/original_pc_mapping.json 原日志 PC 与还原代码的对应关系
host.cpp                       输入构造、运行与 golden 校验
case.json                      shape、编译选项、入口和误差容限
run.py                         重新编译、运行、检查输出
run.sh / set_env.sh             独立运行入口和本用例环境配置
README.txt                     单独复制本目录后的完整使用说明
evidence/summary.json           配置、周期和精度验证摘要
comparison.json                原版与显式版的动态流核对结果
run_output/                    已验证的源码快照、输入输出和 CAModel 日志
```

各用例已附带独立的 `set_env.sh`，初始配置来自结果仓库根目录。可以只复制一个 `explicit_spill_camodel_case/` 到新机器，编辑其中的环境配置后，在该目录执行 `bash ./run.sh --out-dir /tmp/new_explicit_run`，不依赖上层文件。它包含完整 CCE、host 及 golden，并非只有 VF 函数。也可按下文从仓库根目录加载公共环境。需要 Linux/WSL、Python 及匹配的 CANN 模拟器，不依赖 VfSimulator 安装。跨 CANN 版本重新编译可能改变指令分配，不能直接沿用本表的等效性结论。

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

其余四个用例的运行命令如下，均在同一个已加载环境的终端中执行：

```bash
python3 gelu_poly/u12/explicit_spill_camodel_case/run.py --out-dir /tmp/gelu_poly_u12_explicit_check
python3 swiglu_grad/u6/explicit_spill_camodel_case/run.py --out-dir /tmp/swiglu_grad_u6_explicit_check
python3 swiglu_grad/u8/explicit_spill_camodel_case/run.py --out-dir /tmp/swiglu_grad_u8_explicit_check
python3 gelu_grad/u8/explicit_spill_camodel_case/run.py --out-dir /tmp/gelu_grad_u8_explicit_check
```

输出目录必须为空或尚不存在，且不能位于归档用例内。终端显示 `"passed": true` 表示本次输出通过 golden；这不替代动态流核对。

不重新运行 CAModel 时，可以核验这五个已归档结果：

```bash
python3 scripts/validate_explicit_spill_archive.py
```

精度通过后，将新目录交给核查脚本：

```bash
python3 scripts/compare_explicit_spill.py --work-dir /tmp/your_runs --require-all
```

`--work-dir` 下按 `<case>/<uN>/` 放运行目录，可以重复指定多个工作目录；`--require-all` 要求本轮六组均已通过数值和动态流核对。
生成工具为 `scripts/explicit_spill_roundtrip.py`，只接受本轮已验证的单层、固定循环、仿射地址模式；未覆盖 opcode/模式直接报错。
归档不包含全量 VCD 或 VfSim history 大文件。
