# 消融实验计划

所有实验共用基础参数: MLP, 900 train / 200 val / 249 test, 25 epochs, batch 32

## 超参数消融 (需完整训练)

| Exp | LR | WD | Dropout | Layers | Temp | dir | 验证 |
|-----|-----|----|---------|--------|------|-----|------|
| C | 1e-3 | 1e-3 | 0.2 | 2 | 0.07 | exp_c | **基线** ✅ |
| D | **5e-4** | 1e-3 | 0.2 | 2 | 0.07 | exp_d | 更低 LR |
| E | 1e-3 | 1e-3 | 0.2 | **3** | 0.07 | exp_e | 更深 Adapter |
| F | 1e-3 | 1e-3 | 0.2 | 2 | **0.1** | exp_f | 更高温度 |
| G | 1e-3 | 1e-3 | **0.3** | 2 | 0.07 | exp_g | 更高 Dropout |

## Prompt 策略消融 (复用 C 的 checkpoint, 仅重评估)

| Exp | Prompt | 说明 |
|-----|--------|------|
| H | **ensemble** | 5 个模板平均文本嵌入 |
| I | **contrast** | 正例减负例 logit 校准 |
