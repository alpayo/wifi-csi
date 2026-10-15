#!/bin/bash

export PYTHONPATH=.

echo "🔹 Preprocessing..."
python preprocessing/preprocess.py

echo "🔹 Training (NLOS)..."
python training/train.py --mode nlos

echo "🔹 Evaluation..."
python evaluation/evaluate.py --mode nlos

echo "✅ Pipeline completo ejecutado"
