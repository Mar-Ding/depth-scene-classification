# Changelog

## [2026-06-10] feat: 从 CLIP 迁移到 Depth Anything V2 骨干网络

### 架构变更
- **Backbone**: CLIP ViT-B/32 → Depth Anything V2 Base（DINOv2 backbone）
- **训练目标**: 对比学习（NT-Xent） → 交叉熵分类
- **可训练参数**: 528K（MLP Adapter） → 10K（线性分类头）
- **评估方式**: Zero-shot（文本相似度） → 标准分类准确率

### 文件变动

| 文件 | 变更 | 说明 |
|------|------|------|
| `src/models/depth_anything_wrapper.py` | 新增 | DINOv2 特征提取器，支持加载 Depth Anything 权重 |
| `train_best.py` | 新增 | 最优精度训练（13 类映射 + 增强 + 标签平滑） |
| `autodl_run.sh` | 重写 | AutoDL 一键消融实验脚本（含自动关机） |
| `main.py` | 重写 | 改为 Depth Anything + 线性分类头 pipeline |
| `src/training/trainer.py` | 重写 | ClassifierTrainer（交叉熵） |
| `src/evaluation/zero_shot.py` | 重写 | ClassifierEvaluator（分类评估） |
| `src/config.py` | 更新 | 移除 CLIP/Adapter 配置，添加 backbone 配置 |
| `migration_plan.md` | 新增 | 迁移方案文档 |

### 实验结果（13 类 NYU）

| 配置 | Val Acc | Test Acc |
|------|:-------:|:--------:|
| Depth + DA权重 + 增强（最优） | ~60% (预期) | ~55% (预期) |
| Depth + 普通DINOv2 + 增强 | ~45% (预期) | ~40% (预期) |
| RGB + DA权重 + 增强（基线） | ~35% (预期) | ~30% (预期) |
| **旧方案 CLIP + Adapter** | **~35%** | **~25.3%** |

### 依赖变更
- 新增: `transformers==4.47.1`, `h5py`
- 移除: `datasets`, `scikit-learn`
- PyTorch: 2.4.1+cu121

### 注意事项
- `huggingface.co` 直连不通，需使用 `HF_ENDPOINT=https://hf-mirror.com`
- `nyu_depth_v2_labeled.mat` 约 2.8GB，需在 AutoDL 手动上传或自动下载
- PPT_improve.md 待更新：原 CLIP 消融记录已不适用
