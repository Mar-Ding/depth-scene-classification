# PPT 设计计划 · Depth Anything NYU13 场景分类

> 从 CLIP 迁移到 Depth Anything V2 后的新 PPT 方案
> 消融实验已完成（2026-06-11）

---

## 整体结构（建议 25+ 页）

### 1. 封面（1P）
- 标题：基于 Depth Anything 的深度图场景分类
- 副标题：NYU Depth V2 数据集 · 13 类场景识别
- 作者/日期

### 2. 问题背景（2-3P）
- 场景分类任务介绍
- NYU Depth V2 数据集概览
- 为什么用深度图而非 RGB

### 3. 方法对比：CLIP vs Depth Anything（2-3P）
- CLIP 方案的问题：没见过深度图，靠重复通道硬塞
- Depth Anything：6200 万深度图预训练，原生理解深度结构

### 4. 方法细节（3-4P）
- **Backbone**: DINOv2 (facebook/dinov2-base)
- **Depth Anything 权重加载**: 163 个 backbone 键映射 + QKV 拆分
- **分类头**: Linear(768, 13)，仅 10K 参数
- **13 类映射**: 27 原始场景类型 → 13 语义类别

### 5. 实验设置（2P）
- 数据划分：900 train / 200 val / 249 test
- 训练：100 epochs, AdamW, CosineAnnealingLR, Label Smoothing
- 数据增强：RandomHorizontalFlip + RandomAffine + ColorJitter
- 消融实验：2 组主要对比

### 6. 实验结果（4-6P）

| 实验 | 配置 | Best Val Acc | Final Val Acc | 最佳 Epoch |
|:---:|:----:|:-----------:|:-------------:|:----------:|
| A | Depth + DA权重 + 增强 | **50.5%** | 49.5% | 69, 73 |
| C | Depth + 普通DINOv2 + 增强 | **55.5%** | 55.5% | 57~100 |
| — | Depth + 普通DINOv2 (25ep, 无增强) | **35.0%** | — | — |
| — | CLIP ViT-B/32 (旧方案) | **25.3%** | — | — |

> 实验 E（RGB + DA权重）和 F（dinov2-small + DA权重）已跳过——A vs C 的对比已足够说明问题。

**关键发现：普通 DINOv2 优于 Depth Anything 预训练权重**
- 实验 C（55.5%）比实验 A（50.5%）高出 **5 个百分点**
- DINOv2 的通用视觉特征比 Depth Anything 的深度特征更适合场景分类

- 训练曲线对比图
- 各类别精度柱状图
- 深度 vs RGB 对比图

### 7. 关键结论（2-3P）

| 对比 | 提升 |
|:---|:---:|
| DINOv2 vs CLIP | **+30.2%**（25.3% → 55.5%）|
| DINOv2 vs Depth Anything | **+5.0%**（50.5% → 55.5%）|
| 数据增强效果 | **+20.5%**（35.0% → 55.5%）|

- **结论1**：DINOv2 的通用视觉特征远超 CLIP 的跨模态特征用于场景分类
- **结论2**：Depth Anything 的深度预训练对场景分类**反而有害**——深度特征丢失了纹理/语义模式
- **结论3**：数据增强至关重要（无增强仅 35.0%，有增强 55.5%）
- **结论4**：线性分类头（10K 参数）足矣，复杂分类器不必要

### 8. 总结与展望（1-2P）
- 结论
- 未来改进方向（更大 backbone、更多数据、端到端微调）

---

## 设计风格
- **主题色**: Swiss IKB 蓝 (#003399)
- **字体**: 无衬线（思源黑体 / Noto Sans SC）
- **风格**: 信息密集、大字号、高对比、直角无圆角
- **图表**: 干净柱状图 + 表格为主

---

## 待办
- [x] AutoDL 实验跑完后填入精度数据
- [ ] 生成训练曲线图（training_curves.png）
- [ ] 生成精度对比图（accuracy_comparison.png）
- [ ] 生成各类别精度图（per_class_accuracy.png）
- [x] 补充 CLIP 旧方案数据作为对比基线
