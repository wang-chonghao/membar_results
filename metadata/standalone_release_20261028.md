# 单文件报告更新：Softmax Giant-Loop 交错说明

本次保留原有 A5/A6 预测表、Spill DAG、源码查看及附件下载能力，新增：

- Softmax giant-loop 的128元素分段交错图，包含按阶段集中排列、分段交错和20条实际指令放大。
- 第0段初始化的明确说明：两条 VLDS 初始化 MAX 累加器，不执行 VMAX；
  steady loop 首个计算指令为 VEXPDIF，VMAX 更新从 pair=1 开始。
- PNG、SVG 和逐段说明，图中的箭头表示源码排列，而非数据依赖边或硬件周期。

下载 Release 附件 `spill_candidate_alignment_standalone.html`，单独保存即可离线查看。
本版仍为精简浏览版，不内嵌完整日志、编译产物和中间寻优候选；不用于完整复现实验。
更新只涉及说明图与报告，不修改历史周期、模型代码或 CCE 实现。

Release 同时提供 `SHA256SUMS.txt`。浏览核验包括：离线打开、A5/A6 切换、
源码查看与下载、新交错图解码，以及首段初始化文字检查。
