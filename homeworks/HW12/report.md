# HW12 – Time Series Forecasting with GRU

## 1. Кратко: что сделано

- Использован `S12-hw-dataset.csv` (период 2025-01-01…2025-06-29, почасовой), построен временной раздел 70/15/15.
- Проведена предобработка: пропуски ffill/bfill, сортировка по времени, генерация даты и календарных признаков.
- Сгенерированы признаки:
  - лаги `lag_1`, `lag_7`, `lag_14`
  - скользящие статистики `rolling_mean_7`, `rolling_std_7`
  - `dayofweek`, `is_weekend`
- Обучены модели:
  - B1 — naive last
  - B2 — moving average (7)
  - B3 — Ridge на признаках
  - R1 — GRU (последовательность 14 шагов)
- Сохранены артефакты: `runs.csv`, `best_gru.pt`, `best_gru_config.json`, `figures/`.

## 2. Среда и воспроизводимость

- Python 3.12
- torch 2.10.0, torchvision (с поддержкой cuda, если доступна)
- device: `cpu` (на gpu переходит, если `cuda` доступна)
- seed = 42
- Запуск: `homeworks/HW12/HW12.ipynb`, Run All (воспроизводится при каждом запуске при тех же данных).

## 3. Данные

- Датасет: `homeworks/HW12/S12-hw-dataset.csv`.
- Размер: 4320 строк
- Целевая: `target` (показатель целевой величины по часам).
- Разбивка:
  - train: 70% (3024 строки),
  - val: 15% (648 строки),
  - test: 15% (648 строки).
- Примечание: только последовательная временная разбивка.

## 4. Фичи

- `lag_1`, `lag_7`, `lag_14` (прямое прошлое)
- `rolling_mean_7`, `rolling_std_7`
- `dayofweek` (0–6)
- `is_weekend` (0/1)
- Для GRU: последовательность последних 14 точек `target`.

## 5. Валидация и метрики

Метрики: MAE, RMSE, MAPE.

### 5.1 B1 – naive last
- val: MAE=6.4448, RMSE=8.2010, MAPE=4.3979%
- test: MAE=6.3424, RMSE=8.0591, MAPE=4.1485%

### 5.2 B2 – MA(7)
- val: MAE=12.7020, RMSE=15.2176, MAPE=8.8169%
- test: MAE=12.7403, RMSE=15.2387, MAPE=8.5490%

### 5.3 B3 – Ridge (lag+rolling+calendar)
- val: MAE=7.8488, RMSE=9.3371, MAPE=5.2281%
- test: MAE=8.2862, RMSE=9.9908, MAPE=5.3094%

### 5.4 R1 – GRU(14→1)
- val: MAE=6.0789, RMSE=7.7706, MAPE=4.0833%
- test: MAE=7.2073, RMSE=8.9721, MAPE=4.6465%

## 6. Анализ

- На валидации лучшей оказалась R1 (GRU) по всем метрикам.
- На тесте GRU слегка хуже на MAE/RMSE/MAPE, чем naive B1; это ожидаемо из-за переобучения/тугой выборки.
- Ridge (B3) показывает хорошее качество (на уровне 8–10 RMSE) и стабильность, но уступает GRU по val.
- B2 (MA7) дает большое смещение (MAPE ≈ 8.5–8.8%), т.е. некачественная базовая стратегия.
- R1 обеспечивает лучшие адаптивные прогнозы при возрастающем тренде и сезонности, но требует обучения и веса.

## 7. Выводы и рекомендации

1. Основная выборка: использовать `R1 (GRU)` как production-кандидата с регулярным переобучением и контрольным циклом (roll forward validation) для уменьшения drift.
2. На фронте простоты и интерпретируемости `B1` (naive) - стронг-бейзлайн для мониторинга.
3. `B3` (Ridge) можно оставить в качестве резервной метрики GMT для детектирования дрифта в данных.
4. Следующее улучшение:
   - использовать расширенные внешние фичи (погода, праздники, события);
   - попробовать `Transformer` / `TFT` / `N-BEATS` и hyperparam tuning;
   - обучить модели на мультигоризонте (horizon=24).

## 8. Артефакты

- `homeworks/HW12/artifacts/runs.csv`
- `homeworks/HW12/artifacts/best_gru.pt`
- `homeworks/HW12/artifacts/best_gru_config.json`
- `homeworks/HW12/artifacts/figures/series_split.png`
- `homeworks/HW12/artifacts/figures/gru_learning_curves.png`
- `homeworks/HW12/artifacts/figures/best_forecast_test.png`

