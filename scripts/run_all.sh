#!/usr/bin/env bash
set -e
TRAIN_PATH="data/processed/v2.parquet"

mkdir -p experiments/{logs,models,oof,predictions,reports}
mkdir -p data/processed data/submissions

echo "=== preprocess ==="
python -m src.data.preprocess --in_path data/raw/train_2.csv --out_path $TRAIN_PATH

echo "=== ETS features ==="
python -m scripts.run_pipeline --stage ets

echo "=== LightGBM L1 ==="
python -m scripts.run_experiment --config configs/lgbm.yaml --train $TRAIN_PATH
echo "=== LightGBM MAPE-weight ==="
python -m scripts.run_experiment --config configs/lgbm_mape.yaml --train $TRAIN_PATH
echo "=== LightGBM Tweedie ==="
python -m scripts.run_experiment --config configs/lgbm_tweedie.yaml --train $TRAIN_PATH
echo "=== XGBoost ==="
python -m scripts.run_experiment --config configs/xgb.yaml --train $TRAIN_PATH
echo "=== CatBoost ==="
python -m scripts.run_experiment --config configs/catboost.yaml --train $TRAIN_PATH

echo "=== stacking ==="
python -m scripts.run_pipeline --stage stack
echo "=== blending ==="
python -m scripts.run_pipeline --stage blend
echo "=== super blend (stack + blend) ==="
python -m scripts.run_pipeline --stage super

echo "=== REPORT ==="
python -m scripts.compare_experiments
