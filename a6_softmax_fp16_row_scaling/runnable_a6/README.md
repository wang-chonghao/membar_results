# A6 Softmax 完整运行用例

本目录补齐 `microbenchmark-cce-direct/run_test.py` 所需的完整 CCE kernel：
原始 8×1024 gap2 VF + GM→UB NDDMA + UB→GM scatter + 外层同步。
不替换 `microbenchmark-cce-direct/op_kernel/kernel.cce`，不修改该仓的 runner。
这里固定为 8×1024；报告中的其他 shape 仍为 VF-only 快照，不能只修改 MB_TENSOR
来运行其他 shape，需同时调整 VF、UB 布局和搬运长度。

## 文件

| 文件 | 用途 |
|---|---|
| `softmax_8x1024_a6.cce` | 完整入口，无 VF 内部 membar；作为局部依赖融合实验版本 |
| `softmax_8x1024_a6_membar.cce` | 完整入口，在每行 NORM 首次读回前增加 VST_VLD 的同步对照 |
| `run_softmax_a6.py` | 调用原 run_test.py，然后自动执行独立 FP16 golden 检查 |
| `check_softmax.py` | 读取真实输入/输出文件，按行验证稳定 softmax；只依赖 Python 标准库 |
| `test_runnable.py` | 离线检查 golden、入口标记和 runner 参数 |
| `source_provenance.json` | 原始 VF 与两份完整 CCE 的 SHA256 |
| `validation/` | 本机编译与离线检查记录，不是 A6 实测数值结果 |

两份 CCE 均为独立完整文件，不依赖 `#include` 其他实验源码。文件内的
`MB_KERNEL` 与两个 `MB_TENSOR` 按函数参数顺序声明，host 只传 GM input 和 output。
没有错误的加法 `MB_GOLDEN`，也没有将 memB 暴露为 host 输入。

## 数据和 UB 布局

| 对象 | 类型 / shape | 字节数 | UB 地址区间 |
|---|---|---:|---|
| GM input / UB memA | FP32，8×1024 | 32768 | `[0x0000, 0x8000)` |
| UB memB | FP32，8×1024 的 exp 中间结果 | 32768 | `[0x8000, 0x10000)` |
| UB memOutput / GM output | FP16，8×1024 | 16384 | `[0x10000, 0x14000)` |
| UB offsets | uint32，scatter 使用第 0 项 | 预留 32 | `[0x14000, 0x14020)` |

合计预留 81952B，约 80.03125KiB，不含编译器额外 VF 栈。
FP32 NDDMA descriptor 以 8192 个元素为单位；FP16 scatter 搬运长度为
16384 **bytes**，不是 32768B。scatter 的 uint32_t 指针转换只搬运位模式，
不会把 FP16 数值转换为 uint32。

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

运行无内部 membar 的实验版，并自动检查数值：

```bash
python3 "$CASE/run_softmax_a6.py" \
  --runner ./run_test.py \
  --variant no_membar \
  --output result/softmax_a6_gap2
```

同步对照使用另一份输出目录：

```bash
python3 "$CASE/run_softmax_a6.py" \
  --runner ./run_test.py \
  --variant membar \
  --output result/softmax_a6_gap2_membar
```

输出目录必须为空或尚不存在。只有运行成功且独立 golden 检查通过，脚本才返回 0
并输出 `[CHECK] PASS softmax`。数值失败返回非 0；详细结果位于
`result/softmax_a6_gap2/run/softmax_check.json`。

`--compile-only` 仅编译 kernel 和通用 host，不运行模型，也不会宣称数值通过。
`--check-only` 只校验指定输出目录内已有的输入/输出，不会重新仿真。

## 直接使用原 run_test.py

也可直接指定完整 CCE，但必须额外做独立检查，不能把正常退出当作精度通过：

```bash
python3 run_test.py \
  --kernel "$CASE/softmax_8x1024_a6.cce" \
  --golden none \
  --output result/softmax_a6_direct

python3 "$CASE/check_softmax.py" \
  --input result/softmax_a6_direct/run/input_input.bin \
  --output result/softmax_a6_direct/run/output_output.bin \
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

- 两份完整 CCE 均已通过本机 `dav-920r1-vec` 的 kernel 编译和设备链接。
- 独立 checker 与 runner 契约离线测试已通过，覆盖行归约、稳定性、FP16、
  NaN/Inf、未写输出与错误文件长度。
- 本机 A6 SDK 缺少 `libruntime_camodel`，通用 host 链接失败，尚未执行 A6 CAModel。
  因此当前没有这两份完整入口的 A6 数值通过结论，也没有 A6 实测周期。
- 之前 950PR 上的测试是 A5 目标，不能替代本目录的 A6 验证。

目标机成功运行后，还应结合 `run/log_ca/` 中的 instr_popped_log、instr_log 和
EXU 日志核对完整两级 A6 归约、实际 membar/spill 数量，以及 memB 的 store→load
顺序。无 membar 版通过某次数值测试不代表任意 shape 或重排都安全。

## 离线测试与重新生成

```bash
MICROBENCH_RUNNER=/你的路径/microbenchmark-cce-direct/run_test.py \
python3 -B -m unittest discover -s "$CASE" -p test_runnable.py -v
```

在 VfSimulator 仓库可用 `scripts/build_a6_softmax_runnable.py --archive-root
results/membar_results` 重新生成两份 CCE 和源码校验记录；不修改原 VF-only 快照。
