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
```

---

## 5. Что должен сделать ты как «человек на бейзлайне» сегодня (порядок шагов)

1. **Создать репозиторий**, скопировать всё выше как есть.
2. `uv venv --python 3.12 && uv sync`.
3. Положить `train_2.csv` в `data/raw/`.
4. **Прогнать smoke-test**: `python -m scripts.run_experiment --config configs/lgbm.yaml`.
   - Ожидание: CV MAPE первого бейзлайна должен быть в районе **6.5–8.0**.
   - Если намного хуже — что-то с фичами/лагами (проверить, что train_2 действительно содержит данные нескольких лет, и что `t` корректен).
5. **Прогнать остальные бейзлайны** через `scripts/run_all.sh` (можно в фоне `nohup`).
6. **Посмотреть REPORT.txt**, выбрать 2-3 лучших, попробовать `make_submission.py` для гео-бленда. Сабмит на LB.
7. Только теперь, после первой посылки и базовой картины, передавать эстафету ребятам:
   - тому, кто занимается EDA — список вопросов: распределение `РТО` по месяцам, корреляции, доля магазинов с историей < 12 мес, наличие выбросов, насколько таргет связан с year-over-year ratio.
   - тому, кто занимается фичами — пусть докручивает: target encoding (out-of-fold) по region/locality, kmeans-кластеры магазинов по статичным признакам, store-level эластичности и пр.

## 6. Несколько критически важных деталей, которые легко проглядеть

1. **Уникальность `new_id` в submission**. У теста — 18657 строк. Когда добавляем «синтетическую» строку март-2025, делаем это **один раз на магазин** (см. `add_target_row_for_march_2025`). Внутри `pipeline.run_experiment` стоит ассерт.

2. **YoY-фичи** будут с пропусками для магазинов с короткой историей. Это **нормально** — LightGBM/XGBoost/CatBoost умеют работать с NaN. Не заполняйте их нулями, ухудшит модель.

3. **MAPE и log-target**. Минимизация MAE на `log1p(y)` ≈ минимизация MAPE на `y`. Это самый стабильный объектив. Альтернатива — `tweedie` с `variance_power ∈ [1.3, 1.7]` напрямую на `y`.

4. **Не используйте `mape` как objective в LGBM/XGB напрямую** — он даёт нестабильные градиенты на больших значениях. Только как метрика (eval).

5. **Категориальные фичи**: в `lightgbm` мы передаём `categorical_feature=[...]` — обязательно их закодированы int. У нас это так (через `encode_categoricals_ordinal`).

6. **`region`, `locality`** имеют много категорий — это место для **target encoding по фолдам**. Это следующий шаг улучшения (добавьте в `features.py` функцию `add_target_encoding_oof`).

7. **CV-фолд `март 2024`** имитирует ровно ту же задачу, что и финальный прогноз. Дайте ему наибольший вес при отборе моделей. В пайплайне это можно сделать, поменяв формулу `mean_mape` на взвешенную (TODO для следующей итерации).

8. **Geometric mean blend** в `make_submission.py` (`--mode gmean`) — почти всегда лучше арифметического для MAPE-метрики. Используйте её по умолчанию.

9. **Сабмиты на LB**: первые 3-5 — для калибровки (понять, насколько ваш CV коррелирует с public LB). Дальше — только лучшие модели по CV.

Если запустишь `lgbm_baseline` и получишь CV MAPE в районе 6.5-8 — пайплайн жив, можно идти дальше. Пиши, что увидел в логах — продолжим.
