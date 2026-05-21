#!/usr/bin/env bash
TRAIN_PATH="data/processed/v1.csv"

set -e
mkdir -p experiments/{logs,models,oof,predictions,reports}

echo "=== LightGBM L1 ==="
python -m scripts.run_experiment --config configs/lgbm.yaml --train $TRAIN_PATH

echo "=== LightGBM Tweedie ==="
python -m scripts.run_experiment --config configs/lgbm_tweedie.yaml --train $TRAIN_PATH

echo "=== XGBoost ==="
python -m scripts.run_experiment --config configs/xgb.yaml --train $TRAIN_PATH

echo "=== CatBoost ==="
python -m scripts.run_experiment --config configs/catboost.yaml --train $TRAIN_PATH

echo "=== Ridge ==="
python -m scripts.run_experiment --config configs/linear.yaml --train $TRAIN_PATH

echo "=== MLP ==="
python -m scripts.run_experiment --config configs/mlp.yaml --train $TRAIN_PATH

echo "=== REPORT ==="
python -m scripts.compare_experiments
