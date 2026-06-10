# CLIP 跨模态传感器适配

## 项目结构

```
cross-modal-clip/
├── main.py                      # 主入口
├── requirements.txt             # Python 依赖
├── autodl_run.sh                # AutoDL 一键运行脚本
├── src/
│   ├── config.py                # 配置（数据量、超参数等）
│   ├── models/
│   │   ├── clip_wrapper.py      # CLIP 模型封装（冻结）
│   │   ├── depth_processor.py   # 深度图预处理（repeat/normal）
│   │   └── sensor_adapter.py    # 传感器适配器（MLP / Cross-Attention）
│   ├── data/
│   │   └── nyu_dataset.py       # NYU Depth V2 数据加载器
│   ├── training/
│   │   ├── loss.py              # NT-Xent 对比学习损失
│   │   └── trainer.py           # 训练循环
│   ├── evaluation/
│   │   └── zero_shot.py         # Zero-shot 分类评估
│   └── visualization/
│       └── visualize.py         # 可视化（训练曲线、精度对比）
├── tests/
│   ├── test_models.py           # 单元测试（adapter, loss, processor）
│   └── run_mock_training.py     # Mock 训练测试（无需CLIP下载）
└── output/                      # 输出目录（训练后生成）
    ├── training_curves.png
    ├── accuracy_comparison.png
    ├── per_class_accuracy.png
    ├── best_adapter.pt
    ├── history.json
    └── results.json
```

## 快速开始（AutoDL）

```bash
git clone https://github.com/Mar-Ding/cross-modal-clip.git
cd cross-modal-clip
bash autodl_run.sh
```

## 自定义运行

```bash
# 完整流程（训练+评估+可视化）
python main.py --mode all --num-train 200 --epochs 30 --adapter mlp

# 仅训练
python main.py --mode train --num-train 500 --epochs 50

# 仅评估（需已有 best_adapter.pt）
python main.py --mode evaluate

# 使用 Cross-Attention Adapter
python main.py --adapter cross_attn --epochs 50
```

## 方法

参考 T-CLIP 思路：将深度图（单通道）转为 3 通道，通过冻结的 CLIP ViT 提取特征，再用轻量级 MLP Adapter 映射到 CLIP 对齐空间，通过对比学习使深度特征与 RGB 特征对齐。Zero-shot 分类时，通过深度→Adapter→与文本类别 Embedding 计算相似度。
