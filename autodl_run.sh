#!/bin/bash
# AutoDL setup & run script for CLIP cross-modal sensor adaptation
# Usage: bash autodl_run.sh

set -e

echo "=== CLIP Cross-modal Sensor Adaptation ==="
echo ""

# 1. Install dependencies
echo "[1/5] Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets pillow matplotlib tqdm scikit-learn

# 2. Verify GPU
echo ""
echo "[2/5] Checking GPU..."
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 3. Run tests
echo ""
echo "[3/5] Running unit tests..."
python tests/test_models.py

# 4. Run main pipeline (small sample)
echo ""
echo "[4/5] Running main pipeline (200 train / 50 val / 50 test samples)..."
python main.py \
    --mode all \
    --num-train 200 \
    --num-val 50 \
    --num-test 50 \
    --batch-size 16 \
    --epochs 30 \
    --lr 1e-3 \
    --adapter mlp \
    --output-dir ./output

# 5. Show results
echo ""
echo "[5/5] Results summary..."
python -c "
import json
with open('./output/history.json') as f:
    h = json.load(f)
print(f'  Best Val Acc: {max(h[\"val_acc\"]):.2%}')
"
echo ""
echo "=== Done! Check output/ directory ==="
echo "  Training curves: output/training_curves.png"
echo "  Accuracy comparison: output/accuracy_comparison.png"
echo "  Per-class accuracy: output/per_class_accuracy.png"
echo ""
