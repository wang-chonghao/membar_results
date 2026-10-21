gelu_poly / U12 显式 Spill 完整 CAModel 用例

当前无 Membar 实验版本

当前默认 source/kernel.cce 已注释全部 mem_bar(...)，用于无显式 Membar 实验；尚未重新运行 CAModel，周期和精度待验证。source/kernel_with_membar.cce 保留修改前源码；comparison.json、run_output/ 和下方旧周期/精度记录仅对应这个带屏障版本。pipe_barrier、set_flag/wait_flag 和其他指令不变。

本目录可以单独复制到新机器，不需要相邻的原版用例、上层 set_env.sh 或 VfSimulator。
依赖 Linux/WSL、Bash、Python 3.11+，以及带匹配模拟器的 CANN。
原验证环境：CANN 9.0.0-beta.1 / Ascend950PR_9599 / dav-c310-vec。

目录内容
  source/kernel.cce：完整内核，含 VF、GM/UB 搬运、同步及可调用的 kernel 入口。
  source/original_pc_mapping.json：与原硬件日志的 PC 对应关系。
  host.cpp：设备初始化、内存分配、输入构造、kernel 调用、golden 计算及精度校验。
  case.json：kernel 名称、host 参数、编译选项、输出文件和精度容限。
  set_env.sh：本用例的独立环境配置；换机器时在此修改 CANN 路径等四项配置。
  run.py / run.sh：编译 host 和 CCE、链接、执行 CAModel、检查 golden。
  comparison.json / evidence/summary.json：原版与显式版的核验结果和运行配置。
  run_output/：实际编译源码快照、命令、构建日志、输入/golden/输出、kernel.o 和主要 CAModel 日志。

运行方法（先进入本目录）
  1. 编辑本目录 set_env.sh 的 MEMBAR_CANN_ROOT、MEMBAR_HOST_ARCH、
     MEMBAR_NPU_TYPE、MEMBAR_CORE_ARCH。额外 include/lib 和工具覆盖通常留空。
  2. 加载环境并运行：
       source ./set_env.sh
       python3 run.py --dry-run
       python3 run.py --out-dir /tmp/gelu_poly_u12_explicit_rerun
     或由脚本加载本地环境后运行：
       bash ./run.sh --out-dir /tmp/gelu_poly_u12_explicit_rerun
  两种方式任选一种；每次必须换用为空或尚不存在的输出目录，且位于本归档目录之外。
  精度通过时终端显示 passed: true（JSON 格式）。

只检查归档数值（不需要 CANN，也无需 source 环境）
  python3 run.py --check-archived

原版 / 显式版 VF：6037 / 6050 cycles。
已核对完整动态计算依赖、谓词依赖、UB 字节地址、mode 和屏障顺序，输出逐字节相同。
不保证机器编码、标量指令时序或周期完全相同；重新编译后仍须校验，不能仅凭周期接近判定等效。
历史 run_output/commands.json 中的绝对路径仅用于追溯；运行入口使用当前用例位置。
不复制带历史绝对路径的 original_scripts，也不归档 VCD 或 VfSim history 大文件。
