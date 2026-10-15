# 版本周期对比图

与 [HTML 汇总页](../spill_candidate_alignment_summary.html) 默认完整数据一致，共 38 张图、92 行记录。
每个算子和 shape/参数对照组独立绘图，不跨组连接。横轴沿用 HTML 的分类版本轴，不补入未展示的无 spill U2/U4。
四条曲线分别为原 CAModel、VfSim 全局、VfSim 局部、VfSim 软件优化全局 / 保留原版本。缺失值保留断点，不设为零，也不插值。
PNG 为 2560 × 1440（16:9），PDF 为相同比例的矢量图。未重跑预测或修改周期。
软件侧仍采用全局 Membar；无 spill 对照沿用原版本。历史参数差异及限制参见 HTML 详情。
数据 SHA256：`a85ab3cb218d77a0f48c8098bd1361e5cbe4d355ff862346f8bcbfb13b0a4604`。各图的数据、行 ID 和文件校验值见 [index.json](./index.json)。

| 算子 | Shape / 参数组 | PNG | PDF |
| --- | --- | --- | --- |
| SwiGLU Group Grad | 1024 元素 / 本轮参数 | [查看](swiglu__fixed-swiglu.png) | [下载](swiglu__fixed-swiglu.pdf) |
| Sigmoid Focal Loss Grad | 1024 元素 / 本轮参数 | [查看](focal__fixed-focal.png) | [下载](focal__fixed-focal.pdf) |
| SwiGLU Backward Group Quant Dual Axis | 1024 元素 / 本轮参数 | [查看](swiglu_quant__fixed-swiglu_quant.png) | [下载](swiglu_quant__fixed-swiglu_quant.pdf) |
| BCEWithLogits V2 | 1024 元素 / 本轮参数 | [查看](bce__fixed-bce.png) | [下载](bce__fixed-bce.pdf) |
| AdamApplyOne With Decay | 1024 元素 / 本轮参数 | [查看](adam_decay__fixed-adam_decay.png) | [下载](adam_decay__fixed-adam_decay.pdf) |
| GroupNorm SiLU | 1024 元素 / 本轮参数 | [查看](groupnorm__fixed-groupnorm.png) | [下载](groupnorm__fixed-groupnorm.pdf) |
| Logit Grad | 1024 元素 / 本轮参数 | [查看](logit__fixed-logit.png) | [下载](logit__fixed-logit.pdf) |
| LSTM Block Cell Grad | 1024 元素 / 本轮参数 | [查看](lstm__fixed-lstm.png) | [下载](lstm__fixed-lstm.pdf) |
| Sync BatchNorm Backward Elemt | 1024 元素 / 本轮参数 | [查看](syncbn__fixed-syncbn.png) | [下载](syncbn__fixed-syncbn.pdf) |
| DeepNorm Grad | 1024 元素 / 本轮参数 | [查看](deepnorm__fixed-deepnorm.png) | [下载](deepnorm__fixed-deepnorm.pdf) |
| Softmax | 8 行 × 1024，FP32 | [查看](softmax_three_loop__history-softmax_three_loop.png) | [下载](softmax_three_loop__history-softmax_three_loop.pdf) |
| GeLU Poly | 6144 元素，FP32，96 次向量迭代 | [查看](gelu_poly__history-gelu_poly.png) | [下载](gelu_poly__history-gelu_poly.pdf) |
| SwiGLU Grad | 每个 tensor 6144 元素，FP32，双输出 | [查看](swiglu_grad__history-swiglu_grad.png) | [下载](swiglu_grad__history-swiglu_grad.pdf) |
| AdamApplyOne | 3072 元素，FP32，48 次迭代，三输出 | [查看](adam_apply_one__history-adam_apply_one.png) | [下载](adam_apply_one__history-adam_apply_one.pdf) |
| GeLU Grad | 4096 元素，FP32，64 次向量迭代 | [查看](gelu_grad__history-gelu_grad.png) | [下载](gelu_grad__history-gelu_grad.pdf) |
| GeLU Grad 三段循环 | 4096 元素，FP32；显式 UB 布局 96KiB | [查看](gelu_grad_three_stage__history-gelu_grad_three_stage.png) | [下载](gelu_grad_three_stage__history-gelu_grad_three_stage.pdf) |
| FlashAttention Grad 前段 | 提取 VF：FP32，8 行 × 768，输出 8 个行点积；不是完整 FlashAttention Grad 算子。 | [查看](flash_attention_grad_front__history-flash_attention_grad_front.png) | [下载](flash_attention_grad_front__history-flash_attention_grad_front.pdf) |
| MHC Sinkhorn | 8 矩阵 / 历史冻结参数 | [查看](mhc_sinkhorn__mhc-8-frozen.png) | [下载](mhc_sinkhorn__mhc-8-frozen.pdf) |
| MHC Sinkhorn | 8 矩阵 / 新校准 GSU | [查看](mhc_sinkhorn__mhc-8-calibrated.png) | [下载](mhc_sinkhorn__mhc-8-calibrated.pdf) |
| MHC Sinkhorn | 36 矩阵 / 历史冻结参数 | [查看](mhc_sinkhorn__mhc-36-frozen.png) | [下载](mhc_sinkhorn__mhc-36-frozen.pdf) |
| MHC Sinkhorn | 36 矩阵 / 新校准 GSU | [查看](mhc_sinkhorn__mhc-36-calibrated.png) | [下载](mhc_sinkhorn__mhc-36-calibrated.pdf) |
| MHC Sinkhorn | 64 矩阵 / 历史冻结参数 | [查看](mhc_sinkhorn__mhc-64-frozen.png) | [下载](mhc_sinkhorn__mhc-64-frozen.pdf) |
| MHC Sinkhorn | 64 矩阵 / 新校准 GSU | [查看](mhc_sinkhorn__mhc-64-calibrated.png) | [下载](mhc_sinkhorn__mhc-64-calibrated.pdf) |
| MHC Sinkhorn | 128 矩阵 / 新校准 GSU | [查看](mhc_sinkhorn__mhc-128-calibrated.png) | [下载](mhc_sinkhorn__mhc-128-calibrated.pdf) |
| Mish | 3072 元素 / 当前 A5 参数 | [查看](mish__round2-mish.png) | [下载](mish__round2-mish.pdf) |
| LogSigmoidGrad | 3072 元素 / 当前 A5 参数 | [查看](logsigmoid_grad__round2-logsigmoid_grad.png) | [下载](logsigmoid_grad__round2-logsigmoid_grad.pdf) |
| BinaryCrossEntropy（概率输入） | 3072 元素 / 当前 A5 参数 | [查看](binary_cross_entropy__round2-binary_cross_entropy.png) | [下载](binary_cross_entropy__round2-binary_cross_entropy.pdf) |
| GroupNormSwishGrad dx（双块） | 6144 / FP32 / 第三批 A5 | [查看](groupnorm_swish_dx__round3-groupnorm_swish_dx.png) | [下载](groupnorm_swish_dx__round3-groupnorm_swish_dx.pdf) |
| LogSigmoid Forward | 3072 / FP32 / 第三批 A5 | [查看](logsigmoid__round3-logsigmoid.png) | [下载](logsigmoid__round3-logsigmoid.pdf) |
| LayerNormGrad V3 dx | 3072 / FP32 / 第三批 A5 | [查看](layernorm_dx__round3-layernorm_dx.png) | [下载](layernorm_dx__round3-layernorm_dx.pdf) |
| Softplus DAG / 融合 CCE | 3072 / FP32 / 第三批 A5 | [查看](softplus__round3-softplus.png) | [下载](softplus__round3-softplus.pdf) |
| MoE 单专家 Softplus | 3072 / FP32 / 第三批 A5 | [查看](moe_softplus__round3-moe_softplus.png) | [下载](moe_softplus__round3-moe_softplus.pdf) |
| FusedCausalConv1D Residual | 49×192 / FP16 / 第三批 A5 | [查看](causal_conv__round3-causal_conv.png) | [下载](causal_conv__round3-causal_conv.pdf) |
| RoPE Grad Half | 1×1×48×128 / FP32 / 第三批 A5 | [查看](rope_grad__round3-rope_grad.png) | [下载](rope_grad__round3-rope_grad.pdf) |
| RMSNormGrad dx | 3072 / FP32 / 第三批 A5 | [查看](rmsnorm_dx__round3-rmsnorm_dx.png) | [下载](rmsnorm_dx__round3-rmsnorm_dx.pdf) |
| SoftmaxCrossEntropy 输出子 VF | 3072 / FP32 / 第三批 A5 | [查看](softmax_ce__round3-softmax_ce.png) | [下载](softmax_ce__round3-softmax_ce.pdf) |
| MHC Pre Partial Merge | 4×48×64 / FP32 / 第三批 A5 | [查看](mhc_partial__round3-mhc_partial.png) | [下载](mhc_partial__round3-mhc_partial.pdf) |
| Compressor DND Softmax | 4×8×128 / FP32 / 第三批 A5 | [查看](compressor_softmax__round3-compressor_softmax.png) | [下载](compressor_softmax__round3-compressor_softmax.pdf) |

## 重新生成

在结果仓库根目录执行（需要 matplotlib 和中文字体）：

```bash
python3 scripts/plot_version_cycles.py --font /path/to/chinese-font.ttf --output-dir figure_new
```

默认不覆盖已有文件；确认需要刷新时可显式添加 `--overwrite`。
