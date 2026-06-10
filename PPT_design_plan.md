# CLIP 跨模态传感器适配 · PPT 设计方案

## 总体风格

| 项目 | 说明 |
|------|------|
| 风格 | 瑞士国际主义 (Swiss Style) — 归藏 PPT Skill |
| 页数 | 25 页 |
| 主色 | 克莱因蓝 IKB `#002FA7` |
| 底色 | 高级米白 `#fafaf8` |
| 文字色 | 近黑 `#0a0a0a` |
| 辅色 | 浅灰 `#f0f0ee` / 中灰 `#d4d4d2` / 暗灰 `#737373` |
| 字体 | Inter / Helvetica (英文) + 微软雅黑 / PingFang SC (中文) |
| 字重规则 | 大标题 200 ExtraLight，正文 300-400，小标 600 SemiBold |
| 装饰 | 1px hairline 分割线，直角纯色块，无阴影/渐变/圆角 |
| 导航 | 键盘 ← → 翻页，底部圆点导航 |

## 页面详细设计

### P1 封面 (IKB 满屏蓝底)

- 背景：克莱因蓝 `#002FA7` 满屏
- 左上角：`CROSS-MODAL SENSOR ADAPTATION` (mono, 小字, 半透明白)
- 右上角：`CLIP · 2026 · 01 / 25` (mono, 小字)
- 中部大字：**CLIP 跨模态** 换行 **传感器适配** (7.2vw, weight 200, 白色)
- 下方一行副标题：用预训练视觉模型理解深度场景 · 跨模态零样本分类 (半透明白)
- 底部 hairline + 作者/日期信息
- 右下角：`→ swipe / arrow keys`
- 脚注：`IKB · 002FA7`

### P2 目录

- 左列小标题：`CONTENTS`
- 左侧大字：**目录** (4.2vw, weight 200)
- 右列 5 个条目，每个有序号(01-05)、名称、右侧 mono 标签
  1. 研究背景 — Vision Banana
  2. 方法与设计 — CLIP + Adapter
  3. 实验结果 — NYU Depth V2
  4. 效果分析 — vs Vision Banana
  5. 总结展望 — 下一步
- 每项之间用 1px grey hairline 分隔

### P3 章节幕封 (IKB 蓝底)

- 居中：`CHAPTER 01` (mono, 半透明白)
- 大字：**研究背景** (6.4vw)
- 下方小字：通用视觉模型的趋势

### P4 视觉模型的涌现 (白色底)

- 左栏：标题区
  - t-cat: `EMERGENCE`
  - 大字：**视觉模型的涌现**
- 右栏：3 张灰底卡片 (card-fill)，纵向排列
  1. **LLM 路径** — 文本生成预训练→涌现推理、翻译、代码能力→指令微调对齐
  2. **视觉模型路径** — 图像生成/对比预训练→涌现视觉理解→轻量适配
  3. **共同假设** — 预训练视觉模型都内化了通用视觉知识，只需轻量适配

### P5 Vision Banana (白色底, 4 卡)

- t-cat: `GOOGLE 2026 · ARXIV:2604.20329`
- 大字：**Vision Banana**
- 副标题：Image Generators are Generalist Vision Learners
- 3 张灰底卡片排列：
  1. **核心** — 扩散模型预训练→指令微调→所有视觉任务的输出表示为 RGB 图像
  2. **SOTA** — 超越 SAM 3 和 Depth Anything，证明生成式预训练=通用视觉学习器
  3. **启示** — 我们的方法与之互补：CLIP 做分类，共同验证"预训练模型=通用特征提取器"

### P6 跨模态传感器场景理解 (浅灰底)

- t-cat: `PROBLEM STATEMENT`
- 大字：**跨模态传感器场景理解**
- 6 格网格 (2×3)，每格包含图标+标题+描述：
  1. **问题** — 深度/雷达传感器没有大规模预训练模型
  2. **思路** — 冻住 CLIP，训练轻量 Adapter "翻译"深度特征
  3. **场景** — NYU Depth V2 27 类室内场景
  4. **方法** — NT-Xent 对比学习
  5. **评估** — Zero-shot 推理
  6. **愿景** — 任意传感器只需一个 Adapter

### P7 章节幕封 (IKB 蓝底)

- `CHAPTER 02`
- **方法与设计**

### P8 整体管道 (白色底)

- t-cat: `SYSTEM DIAGRAM`
- 大字：**整体管道**
- 流程图：深度图 → 重复3通道 → CLIP ViT(冻结) → MLP Adapter(🔥528K) → 文本匹配
- 下方 3 张指标卡：
  - 528,384 可训练参数
  - 512 特征维度
  - 0.07 温度系数 τ

### P9 CLIP + Adapter (白色底)

- 左栏：`ARCHITECTURE`
- 大字：**CLIP + Adapter**
- 右栏 3 张描边卡片：
  1. **CLIP** — openai/clip-vit-base-patch32，ViT-B/32，全部冻结
  2. **MLP** — 2 层 MLP: Input 512→FC 512 ReLU→Dropout 0.1→FC 512 ReLU→Output 512
  3. **Loss** — NT-Xent 对比损失，正样本对拉近，负样本推远

### P10 训练配置 (浅灰底)

- t-cat: `TRAINING CONFIGURATION`
- 大字：**训练配置**
- 双列表：
  - 左表：超参数（优化器 AdamW, LR 1e-3, Weight Decay 1e-4, Batch 32, Epochs 50）
  - 右表：数据集（NYU Depth V2, 1449 样本, 27 类, h5py, RTX 4060, ~7 分钟训练）

### P11 章节幕封 (IKB 蓝底)

- `CHAPTER 03`
- **实验结果**

### P12 核心指标 (白色底) — KPI Tower 版式

- t-cat: `CORE METRICS`
- 大字：**核心指标**
- 4 柱 KPI Tower (不同高度柱子)：
  - 19.5% Depth Top-1 (柱高 20vh)
  - 40.0% Depth Top-5 (柱高 33vh)
  - 31.0% RGB Baseline (柱高 28vh)
  - 63% 相对性能 (柱高 22vh)
- 底部注释：vs 随机 3.7% · 5.3× 提升 / Top-5 40% = 2.2× 随机

### P13 各类别准确率 (白色底) — H-Bar 版式

- t-cat: `PER-CLASS ACCURACY`
- 大字：**各类别 Top-1 准确率**
- 横向条形图（蓝条）：
  - bathroom 58.8%
  - bookstore 50.0%
  - kitchen 46.7%
  - classroom 33.3%
  - bedroom 18.2%
  - living_room 11.1%
  - dining_room 0%
  - office/study 0%
- 底部标签：结构独特→高 / 结构相似→混淆 / 小样本→不稳定

### P14 训练曲线 (浅灰底)

- t-cat: `TRAINING DYNAMICS`
- 大字：**训练曲线**
- 双栏：
  - 左表：Epoch 1→50 的 Train Loss / Val Loss / Val Acc
  - 右分析：4 条要点（Train Loss 持续下降、过拟合明显、最佳 15%@Epoch 26、建议减少 epochs）

### P15 混淆模式 (白色底) — Duo Compare

- 左右双栏对比：
  - 左栏：混淆模式 — 真实类 vs 常误判为 vs 原因 (表格)
  - 右栏：Top-5 Insight — 正确类别常在前 5 个候选中，Top-5 40% = 2.2× 随机

### P16 章节幕封 (IKB 蓝底)

- `CHAPTER 04`
- **效果分析**

### P17 vs Vision Banana (白色底) — Duo Compare

- 左右双栏对比：
  - 左栏 (Vision Banana)：扩散模型/输出=RGB/不侧重跨模态/模型大/任务不限
  - 右栏 (Our Method)：CLIP 对比式/输出=文本匹配/专攻跨模态/极轻量528K/专注分类

### P18 方法优势 (白色底) — 3 卡

- t-cat: `STRENGTHS`
- 大字：**方法优势**
- 3 张灰底卡片：
  1. **极轻量** — 528K 参数，~7 分钟训练，可部署边缘设备
  2. **零样本** — 不需要文本标签，推理时直接用类别名
  3. **易迁移** — 换传感器只需重新训练 Adapter，深度/雷达/声纳都适用

### P19 局限性 (浅灰底)

- 左栏：`LIMITATIONS`
- 大字：**局限性**
- 右栏 3 张卡片：
  1. **精度** — 信息天花板，63% 相对 RGB 性能
  2. **过拟合** — 样本不足，Val Acc 在 7-15% 间震荡
  3. **分布** — 长尾问题，Top-4 类占 65%

### P20 数据分布不均 (白色底)

- t-cat: `DATASET BIAS`
- 大字：**数据分布不均**
- 横向条形图（灰条+蓝条混合）：
  - kitchen 383, bedroom 225, dining_room 221, bathroom 121, living_room 131, home_office 86, bookstore 49, study 29
- 底部建议：聚合语义相近类别，减少到 12-15

### P21 章节幕封 (IKB 蓝底)

- `CHAPTER 05`
- **总结与展望**

### P22 核心结论 (白色底) — 3 卡

- t-cat: `KEY TAKEAWAYS`
- 大字：**核心结论**
- 3 张描边卡片：
  1. ✓ **跨模态适配可行** — 19.5% Top-1
  2. ✓ **通用视觉学习者** — 与 Vision Banana 一致
  3. ⚠ **精度受限** — 63% 相对 RGB

### P23 下一步方向 (浅灰底)

- t-cat: `NEXT STEPS`
- 大字：**下一步方向**
- 双栏：
  - 左：模型改进（Cross-Attention、早停、全量训练、数据增强）
  - 右：应用扩展（雷达数据验证、多模态融合、时序建模、对比 Vision Banana）

### P24 参考文献 (白色底)

- t-cat: `REFERENCES`
- 大字：**参考文献**
- 4 条引用：
  [1] Image Generators are Generalist Vision Learners, Google, 2026
  [2] Radford et al. CLIP, ICML 2021
  [3] Silberman et al. NYU Depth V2, ECCV 2012
  [4] T-CLIP
- 底部 hairline + 代码地址

### P25 收尾 (半 IKB 蓝底 + 半白底)

- 左半蓝底：**谢谢** / THANK YOU / Cross-modal Sensor Adaptation
- 右半白底：3 条信息
  - CODE: github.com/Mar-Ding/cross-modal-clip
  - OUTPUT: output/best_adapter.pt
  - DECK: 25 slides · Swiss Style · IKB

## 设计规则 (严格遵守)

1. **全程无衬线** — 任何衬线字体出现都是错的
2. **只有一个蓝色 accent** — 不允许出现第二个高亮色
3. **无渐变/阴影/圆角** — 直角纯色块
4. **大标题 weight 200** — 越大越细，禁止加粗大标题
5. **卡片类型互斥** — 同组卡用同一种填充类型
6. **图标用 lucide** — `<i data-lucide="name"></i>`
7. **1px hairline 是唯一分割装饰**
8. **每页 chrome-min 顶部栏** — `t-meta` 在左，页码在右
9. **IKB 蓝底页文字反白**
10. **正文最小字号 16px (投屏使用)**

## 输出格式

单文件 HTML，横向翻页 (← → 键)，可独立在浏览器打开，无外部图片依赖。
