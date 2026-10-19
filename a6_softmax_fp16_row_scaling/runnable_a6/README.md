# A6 Softmax 完整运行用例

本目录补齐 `microbenchmark-cce-direct/run_test.py` 所需的完整 CCE kernel：
原始 VF + GM→UB NDDMA + UB→GM scatter + 外层同步。
不替换 `microbenchmark-cce-direct/op_kernel/kernel.cce`，不修改该仓的 runner。
提供 8、12、16、24 行，列数均为 1024，每种 shape 两份独立完整 CCE。
32 行按要求不提供完整入口；历史报告中的 32 行趋势数据保留不变。

## CCE 索引

| Shape | 无 membar 融合版 | 三循环版 | 融合版 UB / B | 三循环版 UB / B |
|---|---|---|---:|---:|
| 8×1024 | [rows8/no_membar.cce](rows8/no_membar.cce) | [rows8/three_loop.cce](rows8/three_loop.cce) | 81952 | 82016 |
| 12×1024 | [rows12/no_membar.cce](rows12/no_membar.cce) | [rows12/three_loop.cce](rows12/three_loop.cce) | 122912 | 123008 |
| 16×1024 | [rows16/no_membar.cce](rows16/no_membar.cce) | [rows16/three_loop.cce](rows16/three_loop.cce) | 163872 | 164000 |
| 24×1024 | [rows24/no_membar.cce](rows24/no_membar.cce) | [rows24/three_loop.cce](rows24/three_loop.cce) | 245792 | 245984 |

- `no_membar.cce`：SUM→NORM 错开两行的 gap2 融合实现，VF 内没有 mem_bar。
- `three_loop.cce`：MAX、EXP/SUM、NORM 三个阶段，保留原来的两条循环间
  `mem_bar(VST_VLD)`，作为同步基线；不是给融合版加 barrier。
- 两版 VF 均原样取自对应 shape 的历史源码，不改指令顺序或循环展开方式。
- “完整入口”表示包含运行所需的 GM/UB 包装，不表示已经在 A6 上数值验证通过。

## 文件

| 文件 | 用途 |
|---|---|
| `rows*/no_membar.cce` | 本次四种 shape 的无 membar 完整入口 |
| `rows*/three_loop.cce` | 本次四种 shape 的三循环完整入口 |
| `run_softmax_a6.py` | 调用原 run_test.py，然后自动执行独立 FP16 golden 检查 |
| `check_softmax.py` | 读取真实输入/输出文件，按行验证稳定 softmax；只依赖 Python 标准库 |
| `test_runnable.py` | 离线检查 golden、入口标记和 runner 参数 |
| `cases.json` | 八份 CCE 的 shape、来源、SHA256 和逐段 UB 地址 |
| `source_provenance.json` | 旧 8 行对照的源码校验记录 |
| `validation/` | 本机编译与离线检查记录，不是 A6 实测数值结果 |

八份 CCE 均为独立完整文件，不依赖 `#include` 其他实验源码。文件内的
`MB_KERNEL` 与两个 `MB_TENSOR` 按函数参数顺序声明，host 只传 GM input 和 output。
没有错误的加法 `MB_GOLDEN`，也没有将 memB 暴露为 host 输入。

旧文件 `softmax_8x1024_a6.cce` 和 `softmax_8x1024_a6_membar.cce` 保留，
避免已有链接失效。后者是融合版逐行加 barrier 的旧对照，**不是三循环版**。
本次测试请使用上表 `rows*/` 内的文件。

## 数据和 UB 布局

设行数 R、元素数 E=R×1024。GM input 是 E 个 FP32，GM output 是 E 个 FP16；
memB 和标量统计量均为 kernel 内部工作区，不作为 host 参数。

| 对象 | UB 字节区间 |
|---|---|
| memA：FP32 输入 | `[0, 4E)` |
| memB：FP32 exp | `[4E, 8E)` |
| 三循环的 memMax/memSum | `[8E, 8E+S)`，三循环 S=8R；融合版 S=0 |
| memOutput：FP16 输出 | `[8E+S, 10E+S)` |
| offsets：scatter 第 0 项 | `[10E+S, 10E+S+32)` |

各区间按 32B 对齐且互不重叠。三循环不能把输出放在 exp 尾部，否则会覆盖
memMax/memSum。上表 UB 数字包含统计量和 offset 表，不含编译器额外 VF 栈。
24 行三循环距 248KiB 预算还剩 7968B，目标环境仍需确认编译器额外开销。

FP32 NDDMA descriptor 以 E 个元素为单位；FP16 scatter 长度为 2E **bytes**。
scatter 的 uint32_t 指针转换只搬运位模式，不把 FP16 数值转换为 uint32。

外层保留 MTE2→V、V→MTE3、S→MTE3 与 MTE3→S 的 set/wait_flag，以及
已验证示例中的 `pipe_barrier(PIPE_ALL)`。它们不替代 VF 内部 memB 的写后读约束。

## 环境

在已经安装 A6 runtime/simulator 的机器进入 `microbenchmark-cce-direct`：

```bash
cd /你的路径/microbenchmark-cce-direct
export CANN_HOME=/你的路径/cann
export ARCH=x86_64-linux
export SOC_VERSION=Ascend910_9691
export NPU_TYPE=Ascend910_9691
export CORE_ARCH=dav-920r1-vec
export CORE_SIM_DIR=dav_9201
# 仅在全量 simulator 与 SDK 分开安装时填写，指向 tools/simulator 根目录：
# export FULL_SIMULATOR_HOME=/你的路径/full_dump/tools/simulator
source ./set_env.sh
```

不能用 A5 `dav_3510` 或 `Ascend950PR_9599` 的库代替 A6 simulator。
启动脚本固定传入 A6 SoC/core 选择，并覆盖可能遗留的 A5 `NPU_TYPE`。
若设置 FULL_SIMULATOR_HOME，还会为 host 编译补齐该目录的 LIBRARY_PATH。
应在 SDK 本身或独立 simulator 中实际提供兼容的 A6 runtime 模型库。

## 推荐命令

设置当前目录的绝对路径。下面是 WSL 上这份归档的路径；换机器时只改 CASE：

```bash
CASE=/mnt/e/vfsimulator/results/membar_results/a6_softmax_fp16_row_scaling/runnable_a6
```

例如运行 12×1024 无内部 membar 的实验版，并自动检查数值：

```bash
python3 "$CASE/run_softmax_a6.py" \
  --runner ./run_test.py \
  --rows 12 \
  --variant no_membar \
  --output result/softmax_a6_rows12_no_membar
```

相同 shape 的三循环同步基线使用另一份输出目录：

```bash
python3 "$CASE/run_softmax_a6.py" \
  --runner ./run_test.py \
  --rows 12 \
  --variant three_loop \
  --output result/softmax_a6_rows12_three_loop
```

输出目录必须为空或尚不存在。只有运行成功且独立 golden 检查通过，脚本才返回 0
并输出 `[CHECK] PASS softmax`。数值失败返回非 0；详细结果位于
`result/softmax_a6_rows12_no_membar/run/softmax_check.json`。

把 `--rows` 换成 8、16 或 24 即可选择其他 shape，列数固定为 1024。
所有用例只启动 1 个 block；CCE 还显式限制 block 0 执行，避免重复写同一输出。

`--compile-only` 仅编译 kernel 和通用 host，不运行模型，也不会宣称数值通过。
`--check-only` 只校验指定输出目录内已有的输入/输出，不会重新仿真。

## 直接使用原 run_test.py

也可直接指定完整 CCE，但必须额外做独立检查，不能把正常退出当作精度通过：

```bash
python3 run_test.py \
  --kernel "$CASE/rows12/no_membar.cce" \
  --soc-version Ascend910_9691 \
  --core-arch dav-920r1-vec --core-sim-dir dav_9201 \
  --block-dim 1 \
  --golden none \
  --output result/softmax_a6_direct

python3 "$CASE/check_softmax.py" \
  --input result/softmax_a6_direct/run/input_input.bin \
  --output result/softmax_a6_direct/run/output_output.bin \
  --rows 12 --cols 1024 \
  --report result/softmax_a6_direct/run/softmax_check.json
```

原 run_test.py 的 MB_GOLDEN 不支持 FP16，且默认表达式机制不是按行归约接口。
这里从实际 `input_input.bin` 读取 FP32 数据，用 Python 双精度计算：

```text
e = exp(x - row_max(x))
y = e / row_sum(e)
```

没有额外 `1/sqrt(d)` 缩放。逐元素判定为绝对误差 ≤1e-6 或相对误差 ≤1e-3，
并拒绝 NaN/Inf、错误文件长度和未写出的输出。当前 generic_runner 没有额外 GM
保护区，本目录不声称检查过不存在的 guard。

## 验证状态和日志

- 各 shape 的编译结果见 `validation/shape_kernel_compile.json`，包括对应源码
  SHA256；这是 kernel 编译和设备链接记录，不是 CAModel 执行结果。
- 独立 checker 与 runner 契约离线测试已通过，覆盖行归约、稳定性、FP16、
  NaN/Inf、未写输出与错误文件长度。
- 本机 A6 SDK 缺少 `libruntime_camodel`，通用 host 链接失败，尚未执行 A6 CAModel。
  因此当前没有这些完整入口的 A6 数值通过结论，也没有 A6 实测周期。
- 之前 950PR 上的测试是 A5 目标，不能替代本目录的 A6 验证。

目标机成功运行后，还应结合 `run/log_ca/` 中的 instr_popped_log、instr_log 和
EXU 日志核对完整两级 A6 归约、实际 membar/spill 数量，以及 memB 的 store→load
顺序。无 membar 版通过某次数值测试不代表任意 shape 或重排都安全。

## 离线测试与重新生成

```bash
MICROBENCH_RUNNER=/你的路径/microbenchmark-cce-direct/run_test.py \
python3 -B -m unittest discover -s "$CASE" -p test_runnable.py -v
```

在 membar_results 根目录运行：

```bash
python3 scripts/build_a6_softmax_runnable.py --archive-root .
```

重新生成八份 CCE 和 `cases.json`；不修改原 VF-only 快照和旧 8 行对照。
