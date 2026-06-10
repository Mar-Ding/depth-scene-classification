---
title: "CLIP 跨模态传感器适配"
subtitle: "Cross-modal Sensor Adaptation with CLIP"
author: "Mar-Ding"
date: "2026-06-10"
documentclass: article
CJKmainfont: "Noto Sans CJK SC"
---

# 项目简介

## 目标

利用预训练视觉模型 CLIP 的图文对齐能力，通过轻量适配器实现深度传感器数据的零样本场景分类。即：输入深度图，输出场景类别（如 kitchen、bedroom、office 等 27 类室内场景）。

## 动机

- 深度/雷达传感器缺少大规模预训练模型
- 传统方案需为每个传感器重新训练完整模型，成本高
- CLIP 等预训练视觉模型内化了通用视觉知识，通过轻量适配即可泛化

---

# 方法

## 管道

```
深度图 → 重复3通道(伪RGB) → CLIP ViT编码器(冻结) → MLP Adapter → 文本类别匹配
```

## 关键设计

| 组件 | 说明 |
|------|------|
| **CLIP 模型** | openai/clip-vit-base-patch32, 冻结全部参数 |
| **MLP Adapter** | 2层全连接 (512→512→512), LayerNorm, ReLU, Dropout=0.2 |
| **可训练参数** | 528,384 (仅 Adapter) |
| **损失函数** | NT-Xent 对比损失 (温度 τ=0.07) |
| **优化器** | AdamW, LR=5e-4, Weight Decay=1e-3 |

## 与 Vision Banana 的联系

Vision Banana (Google, 2026, arXiv:2604.20329) 提出：**图像生成预训练本身就是通用视觉学习器**——扩散模型通过指令微调可在分割、深度估计等任务上达到 SOTA。我们的工作与之互补：CLIP（对比式）做分类，而非生成。两者共同验证了"预训练视觉模型可作为通用特征提取器"这一趋势。

---

# 数据集

## NYU Depth V2

| 项目 | 说明 |
|------|------|
| 样本数 | 1449 组 RGB-Depth 对 |
| 尺寸 | 深度图 640×480, RGB 图 480×640 |
| 场景 | 27 种室内场景 (bedroom, kitchen, bathroom, office 等) |
| 格式 | MATLAB v7.3 (.mat, 2.97GB, h5py 读取) |
| 划分 | 900 训练 / 200 验证 / 249 测试 |

---

# 实验结果

## 最优配置

| 指标 | 数值 |
|------|------|
| **Zero-shot Top-1** | **25.30%** |
| **Zero-shot Top-5** | **48.19%** |
| RGB Baseline (CLIP) | 27.71% |
| 相对 RGB 性能 | **91.3%** |
| 随机猜测 Top-1 | 3.7% |
| 参数总量 | 528K |

对比 Vision Banana：我们的参数量仅为其极小一部分（528K vs 完整扩散模型），专注分类任务，在深度 zero-shot 场景分类上达到 RGB 上限的 91%。

## 消融实验汇总

| 变量 | Top-1 | 结论 |
|------|-------|------|
| 基线 (MLP/600样本) | 19.50% | 过拟合 |
| Cross-Attention (3.8M) | 12.05% | 参数过多，欠拟合 |
| **最优 (LR=5e-4)** | **25.30%** | 更稳定的收敛 |
| 3层 MLP | 16.06% | 更深反而不行 |
| Dropout=0.3 | 18.47% | 过度正则化 |
| Temperature=0.1 | 18.88% | 损失不够锐利 |
| Prompt Ensemble | 24.10% | 无明显收益 |

## 各类别表现

| 类别 | 准确率 | 说明 |
|------|--------|------|
| bathroom | 58.8% | 结构独特，易于区分 |
| bookstore | 50.0% | 少样本但特征明显 |
| kitchen | 46.7% | 大量样本，稳定性好 |
| classroom | 33.3% | 少样本 |
| bedroom | 18.2% | 一般 |
| living_room | 11.1% | 与多类混淆 |

---

# 核心结论

1. **轻量跨模态适配可行** — 仅 528K 参数即可让深度图实现 zero-shot 场景分类
2. **与 Vision Banana 趋势一致** — 预训练视觉模型可作为通用特征提取器
3. **深度可达 RGB 性能的 91%** — 但进一步受限于纯几何信息的天花板
4. **MLP Adapter 优于复杂结构** — 在小数据量下，简单模型反而更好

## 下一步方向

- 数据增强（旋转/缩放/裁剪深度图）
- 更大 CLIP Backbone (ViT-L/14)
- 真实雷达数据验证
- 多模态融合（深度 + RGB）

---

# 环境与代码

| 项目 | 说明 |
|------|------|
| GPU | NVIDIA RTX 4060 Laptop (CUDA 12.1) |
| 框架 | PyTorch 2.5.1 + Transformers 4.47.1 |
| 代码 | github.com/Mar-Ding/cross-modal-clip |
| 训练时间 | ~5 分钟 (25 epochs) |
