# 单文件报告下载

GitHub 普通文件列表仅保存源码、图片和核验数据。完整单文件报告通过
[membar_results Releases](https://github.com/wang-chonghao/membar_results/releases)
发布，下载附件 `spill_candidate_alignment_standalone.html` 即可，不需要下载配套目录。

下载后在桌面浏览器打开 HTML。可以切换 A5/A6、查看汇总表、图表、源码和归档附件；
顶部的“A6 Spill DAG 对比”包含 AdamApplyOne U8、FlashAttention Grad U2 和 Logit Grad U3/U4/U8 的两种优化图。

软件分组策略和不计标量时序的限制均在报告中保留，特别是 Logit U8 的 297 cycles 不含标量 spill 时序。

每次 Release 同时提供 `SHA256SUMS.txt`。Linux 可将两个附件放在同一目录后运行：

```bash
sha256sum -c SHA256SUMS.txt
```

完整 HTML 不加入 Git，以免把大型打包文件写入版本历史。打包与验证脚本：
`scripts/build_standalone_report.py`、`scripts/check_standalone_report.py`。
发布脚本 `scripts/publish_standalone_release.py` 只在内存中使用登录凭据，先上传并校验文件大小和 SHA256，再公开 Release。
