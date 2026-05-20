# Hackathon RTO — X5 Retail

## Setup
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
mkdir -p data/raw data/processed data/submissions
mkdir -p experiments/{logs,models,oof,predictions,reports}
# Положить train_2.csv в data/raw/
```

## Запуск одного эксперимента
```bash
python -m scripts.run_experiment --config configs/lgbm.yaml
```

## Все бейзлайны
```bash
bash scripts/run_all.sh
```

## Hyperopt
```bash
python -m scripts.tune_optuna --model lightgbm --trials 60 --timeout 14400
```

## Стекинг
```bash
python -m scripts.stack_oof \
  --oof experiments/oof/lgbm_*.parquet experiments/oof/xgb_*.parquet experiments/oof/cat_*.parquet \
  --predictions experiments/predictions/lgbm_baseline_*.csv experiments/predictions/xgb_baseline_*.csv experiments/predictions/cat_baseline_*.csv \
  --out data/submissions/stack_v1.csv
```

## Бленд готовых посылок
```bash
python -m scripts.make_submission \
  --predictions experiments/predictions/lgbm_*.csv experiments/predictions/xgb_*.csv \
  --weights 0.6 0.4 --mode gmean --out data/submissions/blend_v1.csv
```

## Отчёт
```bash
python -m scripts.compare_experiments
cat experiments/reports/REPORT.txt
```

