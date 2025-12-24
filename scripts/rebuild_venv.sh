#!/bin/bash
# Script to rebuild the virtual environment cleanly

echo "🗑️  Removing old virtual environment..."
rm -rf .venv

echo "🔨 Creating fresh virtual environment..."
python3 -m venv .venv

echo "⚡ Activating and upgrading pip..."
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

echo "📦 Installing dependencies (with binary wheels)..."
pip install --no-cache-dir numpy pygame gymnasium stable-baselines3 torch streamlit psutil pandas optuna

echo "✅ Done! Test with:"
echo ".venv/bin/python -c 'import time; t=time.time(); import torch; print(f\"Loaded in {time.time()-t:.2f}s\")'"
