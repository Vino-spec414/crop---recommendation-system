#!/usr/bin/env bash
# =============================================================
# setup.sh — Automated environment setup for CropSense
# =============================================================
# Usage:  bash setup.sh
# =============================================================

set -e  # Exit on any error

echo ""
echo "=============================================="
echo "  🌾  CropSense — Environment Setup"
echo "=============================================="

# 1. Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python: $PYTHON_VERSION"

# 2. Create virtual environment
if [ ! -d "venv" ]; then
  echo ""
  echo "  Creating virtual environment..."
  python3 -m venv venv
  echo "  ✓ Virtual environment created"
else
  echo "  ℹ  Virtual environment already exists"
fi

# 3. Activate and install packages
echo ""
echo "  Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "  ✓ All packages installed"

# 4. Create placeholder data directory message
echo ""
echo "  Checking data directory..."
mkdir -p data models

if [ ! -f "data/crop_dataset.csv" ]; then
  echo "  ⚠️  data/crop_dataset.csv NOT FOUND"
  echo "     → Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset"
  echo "     → Place as: data/crop_dataset.csv"
else
  echo "  ✓ crop_dataset.csv found"
fi

if [ ! -f "data/soil_dataset.csv" ]; then
  echo "  ⚠️  data/soil_dataset.csv NOT FOUND (optional)"
  echo "     → Download from: https://data.mendeley.com/datasets/36xkf32pph/1"
  echo "     → Place as: data/soil_dataset.csv"
else
  echo "  ✓ soil_dataset.csv found"
fi

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Activate:  source venv/bin/activate"
echo "  2. Train:     python src/train_model.py"
echo "  3. Run app:   python app/app.py"
echo "  4. Open:      http://localhost:5000"
echo "=============================================="
echo ""
