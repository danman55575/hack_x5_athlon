
# Hackathon RTO — X5 Retail (Pyaterochka March-2025 forecast)

Прогноз РТО магазинов сети «Пятёрочка» на март 2025. Метрика: **MAPE**, итоговый балл:
`100 * ((100 - min(MAPE, 100)) / 100)^2`. Файл посылки: 18657 строк, `<1MB`, колонки `new_id, rto`.

## Setup
```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
mkdir -p data/raw data/processed data/submissions
mkdir -p experiments/{logs,models,oof,predictions,reports}
```
Положить `train_2.csv` в `data/raw/`.

## Полный воспроизводимый прогон (с нуля до финальной посылки)

Для **строгой** воспроизводимости запускать с явным `PYTHONHASHSEED` —
это критично, потому что некоторые операции хеширования в Python не подхватят
seed, если он установлен изнутри уже запущенного интерпретатора:

```bash
export PYTHONHASHSEED=2026

# 1. Препроцессинг сырых данных (rename + inflation adjustment + outliers + population fix)
python -m src.data.preprocess \
    --in_path data/raw/train_2.csv \
    --out_path data/processed/v2.parquet

# 2. ETS-фичи (Holt-Winters per-store, joblib parallel). ~20-40 мин на M1 Max.
python -m scripts.run_pipeline --stage ets

# 3. Обучение всех базовых моделей (LightGBM x3, XGBoost, CatBoost)
python -m scripts.run_pipeline --stage train

# 4. OOF-стекинг через Ridge в log-space
python -m scripts.run_pipeline --stage stack

# 5. Geometric-mean бленд по обратным CV-MAPE весам
python -m scripts.run_pipeline --stage blend

# 6. Super blend (stack ⊕ blend, gmean 50/50)
python -m scripts.run_pipeline --stage super

# 7. Сборка отчёта
python -m scripts.compare_experiments
cat experiments/reports/REPORT.txt
```

Или одной командой (`--stage all` поочерёдно выполняет preprocess→ets→train→stack→blend→super):
```bash
PYTHONHASHSEED=2026 python -m scripts.run_pipeline --stage all
```

## Запуск одного эксперимента
```bash
python -m scripts.run_experiment --config configs/lgbm.yaml \
    --train data/processed/v2.parquet
```
Доступные конфиги: `lgbm.yaml`, `lgbm_mape.yaml`, `lgbm_tweedie.yaml`,
`xgb.yaml`, `catboost.yaml`, `linear.yaml`, `mlp.yaml`.

## Hyperopt (Optuna)
```bash
python -m scripts.tune_optuna --model lightgbm --use_cat \
    --target_transform log1p --trials 60 --timeout 14400
python -m scripts.tune_optuna --model xgboost \
    --target_transform log1p --trials 60
python -m scripts.tune_optuna --model catboost --use_cat \
    --target_transform log1p --trials 40
```
Полностью синхронизирован с боевым pipeline (те же фолды/веса/TE/ETS/winsorize).

## Бленд готовых посылок вручную
```bash
python -m scripts.make_submission \
    --predictions experiments/predictions/lgbm_*.csv \
                  experiments/predictions/xgb_*.csv \
    --weights 0.6 0.4 --mode gmean \
    --out data/submissions/blend_v1.csv
```
`--mode gmean` (геометрическое среднее) предпочтительнее для MAPE.

## Reproducibility чеклист (требование организаторов)

* Random seed `2026` зафиксирован глобально (`src/utils/seed.py: set_seed`):
  `random`, `numpy`, `torch`, `torch.cuda`, `torch.mps`, `cudnn.deterministic=True`.
* `PYTHONHASHSEED` устанавливается из `set_seed`, **но** для критичной воспроизводимости
  родительского процесса экспортируйте его перед запуском Python.
* Per-experiment результаты: `experiments/reports/{name}_{ts}.json`.
* OOF-предсказания: `experiments/oof/{name}_{ts}_oof.parquet`.
* Финальные посылки: `experiments/predictions/{name}_{ts}.csv` и `data/submissions/`.
* Submission валидируется (18657 строк, no NaN, > 0, < 1MB) **до** сохранения файла —
  лучше упасть на ассерте, чем потратить попытку из 100 на битый файл.

## Финальная посылка для Контеста

1. Проверить топ моделей: `experiments/reports/best_models.csv` (top-10 по CV MAPE).
2. Для одной модели: `experiments/predictions/{best_name}_{ts}.csv`.
3. Для ансамбля: `data/submissions/super_blend.csv` (stack + blend, gmean 50/50).

## Структура репозитория
```
src/
  data/                # loader, preprocess, target encoding, build_features
  models/              # gbm (LGB/XGB/Cat), linear, mlp, ts_features (ETS), stacking
  validation/          # TimeFold CV, MAPE/score
  utils/               # seed, logging, io
  pipeline.py          # главный пайплайн одного эксперимента
  report.py            # сборка REPORT.txt и best_models.csv
scripts/
  run_pipeline.py      # главный entrypoint (stages: preprocess/ets/train/stack/blend/super/all)
  run_experiment.py    # запуск одного yaml-конфига
  tune_optuna.py       # hyperopt, синхронизирован с pipeline
  stack_oof.py         # Ridge-стекинг OOF
  make_submission.py   # бленд готовых csv-предсказаний
configs/               # yaml для каждой модели
experiments/
  logs/                # *.log по экспериментам
  reports/             # *.json reports + REPORT.txt + best_models.csv
  oof/                 # OOF-предсказания для стекинга
  predictions/         # финальные csv по экспериментам
data/
  raw/train_2.csv      # исходные данные
  processed/v2.parquet # после preprocess
  processed/ets_features.parquet  # после stage ets
  submissions/         # финальные блендованные посылки
```
