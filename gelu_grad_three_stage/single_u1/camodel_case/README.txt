GeLU Grad 三段循环 / 单循环 U1 参考 的 CAModel 用例

历史 VF 时间：996 cycles。归档方式：从仓库保留材料恢复；原临时目录已不存在。
缺失材料：原临时目录、原编译二进制（可重新构建）。本次仅归档，没有重新运行 CAModel。

目录：host.cpp 为 host 源码；run_output 为已保存的运行日志、数据及仍存在的编译产物；
source/code 为源码；evidence 为历史命令与精度记录；original_scripts 为原实验脚本。
case.json 记录实际运行源码、入口、编译参数与材料来源。原脚本可能包含历史绝对路径。

直接复现（Linux/WSL，安装对应 CANN）：
  source ../../../set_env.sh  # 首次使用前编辑结果根目录中的环境配置
  python3 run.py --dry-run
  python3 run.py --check-archived
  python3 run.py --out-dir /tmp/gelu_grad_three_stage_single_u1_rerun

重新运行的输出目录必须为空且位于本归档目录之外；run.py 会编译并执行，然后进行 golden 校验。
没有历史 golden 的版本不支持 --check-archived，但生成式 host 会在重跑时生成并校验 golden。
报告中的 kernel.cce 可能是后续等效改写；复现始终使用 case.json 中的实际历史运行源码。
