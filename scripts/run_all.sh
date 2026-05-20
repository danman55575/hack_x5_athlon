#!/usr/bin/env bash
set -e
mkdir -p experiments/{logs,models,oof,predictions,reports}

echo "=== LightGBM L1 ==="
python -m scripts.run_experiment --config configs/lgbm.yaml

echo "=== LightGBM Tweedie ==="
python -m scripts.run_experiment --config configs/lgbm_tweedie.yaml

echo "=== XGBoost ==="
python -m scripts.run_experiment --config configs/xgb.yaml

echo "=== CatBoost ==="
python -m scripts.run_experiment --config configs/catboost.yaml

echo "=== Ridge ==="
python -m scripts.run_experiment --config configs/linear.yaml

echo "=== MLP ==="
python -m scripts.run_experiment --config configs/mlp.yaml

echo "=== REPORT ==="
python -m scripts.compare_experiments
