# План проекта: Fraud Detection (Детекция мошеннических транзакций)

**Курс:** Инженерия Искусственного Интеллекта  
**Датасет:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
— 284 807 транзакций, 492 мошеннические (0.17%), сильный дисбаланс классов  
**Технологии:** Python, sklearn, PyTorch, LightGBM, FastAPI, MLflow, Docker

---

## Карта требований преподавателя → сессии

| # | Минимальное требование | Сессия |
|---|------------------------|--------|
| 1 | Задача, цели, метрики | 1 |
| 2 | Воспроизводимый пайплайн данных | 1–2 |
| 3 | Модели + эксперименты + сравнения | 2–3 |
| 4 | API-сервис на FastAPI | 4 |
| 5 | Docker-образ | 4 |
| 6 | Observability: логи, метрики, health-check | 4 |
| 7 | Секреты и данные по правилам курса | 1 + 4 |

---

## Сессия 1 — Постановка задачи + Репозиторий + EDA

**Закрываем:** требования №1, №7 (частично)

### Структура папки `project/`

```
project/
├── PLAN.md                ← этот файл
├── README.md              ← паспорт проекта (заполняем сразу)
├── report.md              ← отчёт (пишем по ходу сессий)
├── self-checklist.md      ← чеклист (отмечаем по мере готовности)
├── requirements.txt
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   └── data/
│       └── pipeline.py    ← загрузка + препроцессинг
├── data/
│   ├── .gitkeep
│   └── sample.csv         ← 1000 строк для тестов (можно коммитить)
├── configs/
│   ├── config.yaml
│   └── .env.example
├── models/                ← создаём позже
└── .gitignore             ← data/raw/, *.env, models/*.pkl
```

### Задача (для README.md)

- **Пользователь:** антифрод-система банка
- **Задача:** бинарная классификация транзакции → fraud (1) / legit (0)
- **Почему не accuracy:** 0.17% мошеннических → тривиальный классификатор даёт 99.83% accuracy
- **Целевые метрики:** PR-AUC (основная), ROC-AUC, Precision @ Recall ≥ 0.80
- **Ограничения:** анонимизированные PCA-фичи V1–V28, нет категориальных признаков

### Ноутбук `01_eda.ipynb`

- Размер датасета, типы, пропуски (зафиксировать что их нет)
- Распределение классов → визуализация дисбаланса
- Amount и Time: медиана/среднее по классам fraud vs legit
- Топ-10 признаков по корреляции с таргетом
- Вывод: насколько тяжёлый дисбаланс, какие фичи информативны

### Правила безопасности (сразу в .gitignore)

```
data/raw/
*.env
models/*.pkl
models/*.joblib
```

**Результат сессии:** репозиторий настроен, задача сформулирована, данные поняты

---

## Сессия 2 — Пайплайн данных + Baseline модели + MLflow

**Закрываем:** требования №2, №3 (начало)

### `src/data/pipeline.py` — воспроизводимый пайплайн

```python
def load_data(path: str) -> pd.DataFrame: ...

def split_data(df, test_size=0.2, val_size=0.2, random_state=42):
    # Стратифицированный split: 60% train / 20% val / 20% test
    ...

def build_preprocessor() -> Pipeline:
    # StandardScaler для Amount и Time
    # V1-V28 оставляем как есть (уже нормализованы PCA)
    ...
```

**Важно:** фиксированный `random_state=42` везде → воспроизводимость

### Решение дисбаланса (только на train!)

- `class_weight='balanced'` во всех sklearn-моделях
- SMOTE как альтернатива → сравниваем результаты

### Ноутбук `02_baseline_models.ipynb`

| Модель | Зачем включаем |
|--------|----------------|
| Logistic Regression | Интерпретируемый baseline |
| Random Forest | Ансамбль без буста |
| LightGBM | Основной кандидат |
| XGBoost | Альтернатива LightGBM |

### Настройка MLflow

```python
with mlflow.start_run(run_name="lgbm_baseline"):
    mlflow.log_params({"n_estimators": 300, "class_weight": "balanced"})
    mlflow.log_metrics({"pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1})
    mlflow.sklearn.log_model(model, "model")
```

### Анализ ошибок

- Confusion matrix на val-выборке
- Precision-Recall кривые для всех моделей на одном графике
- Выбор оптимального порога (не 0.5!)

**Результат сессии:** 4 обученные модели, все эксперименты залогированы, таблица сравнения

---

## Сессия 3 — Нейросеть + Улучшение + Финальная модель

**Закрываем:** требование №3 (полностью)

### Ноутбук `03_neural_net.ipynb` — MLP на PyTorch

```python
class FraudMLP(nn.Module):
    # Архитектура: 30 → 128 → 64 → 32 → 1
    # BatchNorm + Dropout(0.3) + ReLU
    # Focal Loss (alpha=0.25, gamma=2) для дисбаланса

# Adam, lr=1e-3, 50 epochs, early stopping по val PR-AUC
```

Сравнить с LightGBM: где лучше, где хуже, почему.

### Ноутбук `04_final_model.ipynb` — выбор и упаковка

**Итоговая таблица экспериментов** (войдёт в report.md):

| Модель | PR-AUC | ROC-AUC | F1 | Latency |
|--------|--------|---------|----|---------|
| LogReg | ? | ? | ? | ~1ms |
| RandomForest | ? | ? | ? | ~5ms |
| LightGBM | ? | ? | ? | ~2ms |
| XGBoost | ? | ? | ? | ~2ms |
| MLP | ? | ? | ? | ~3ms |

**Обоснование выбора** финальной модели: лучший PR-AUC + малый latency + не требует GPU

### Артефакты модели

```
project/models/
├── final_model.pkl       ← joblib.dump()
├── preprocessor.pkl      ← scaler
├── threshold.json        ← {"threshold": 0.42}
└── model_card.md         ← описание: метрики, ограничения, версия
```

**Результат сессии:** финальная модель выбрана, упакована, report.md наполовину готов

---

## Сессия 4 — API + Observability + Docker

**Закрываем:** требования №4, №5, №6, №7 (полностью)

### `src/service/main.py` — FastAPI эндпоинты

```
POST /predict
  Вход:  {"time": 12345, "amount": 49.99, "v1": -1.35, ..., "v28": 0.02}
  Выход: {"fraud_score": 0.87, "is_fraud": true, "threshold": 0.42}

GET /health
  Выход: {"status": "ok", "model_version": "lgbm_v1", "uptime_sec": 3600}

GET /metrics
  Выход: Prometheus-формат (счётчики запросов, latency, ошибки)
```

### `src/service/observability.py` — Observability

```python
# Логи — каждый запрос:
logger.info(f"predict | amount={amount} | score={score:.3f} | latency={ms}ms")

# Метрики — prometheus_client:
REQUEST_COUNT  = Counter("requests_total", "Total requests")
FRAUD_COUNT    = Counter("fraud_predicted_total", "Fraud predictions")
LATENCY        = Histogram("request_latency_seconds", "Latency")
```

### Конфигурация и секреты

```bash
# configs/.env.example  (коммитим в репозиторий)
MODEL_PATH=models/final_model.pkl
THRESHOLD=0.42
LOG_LEVEL=INFO
API_KEY=your_api_key_here

# .env  (в .gitignore — НЕ коммитим никогда!)
```

### `Dockerfile`

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY src/     ./src/
COPY models/  ./models/
COPY configs/ ./configs/
CMD ["uvicorn", "src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `docker-compose.yml`

```yaml
services:
  fraud-api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./models:/app/models
```

**Результат сессии:** `docker compose up` → сервис работает, `/predict` отвечает, логи пишутся

---

## Сессия 5 — Тесты + Документация + Финальная сборка

**Закрываем:** все оставшиеся пункты self-checklist.md

### `tests/` — минимальный набор

```python
# test_model.py — sanity tests
def test_model_loads():      assert model is not None
def test_predict_shape():    assert output in [0, 1]
def test_fraud_score_range(): assert 0 <= score <= 1

# test_api.py — smoke tests
def test_health_endpoint():     assert response.status_code == 200
def test_predict_legit():       # нормальная транзакция → is_fraud: false
def test_predict_invalid_input(): assert response.status_code == 422

# test_pipeline.py — unit tests
def test_split_stratified():  # доля мошенников сохранена в каждой части
def test_scaler_fitted():     # scaler обучен только на train, не на val/test
```

### Финальный `report.md` — структура

1. Постановка задачи и обоснование метрик
2. Описание датасета + ключевые выводы EDA
3. Таблица всех экспериментов с метриками
4. Обоснование выбора финальной модели
5. Описание API-сервиса (эндпоинты, формат запроса/ответа)
6. Как запустить (3 команды: clone → build → up)
7. Ограничения и что можно улучшить

### `self-checklist.md` — финальная проверка

- [ ] README.md с задачей и метриками
- [ ] EDA ноутбук с выводами
- [ ] Воспроизводимый пайплайн данных
- [ ] Несколько обученных моделей (минимум 3)
- [ ] Эксперименты залогированы в MLflow
- [ ] Финальная модель выбрана и обоснована
- [ ] FastAPI сервис с /predict, /health, /metrics
- [ ] Docker + docker-compose работают
- [ ] Логи и метрики подключены
- [ ] .env.example есть, .env в .gitignore
- [ ] Тесты проходят после чистого clone
- [ ] report.md заполнен

**Результат сессии:** проект полностью готов к защите

---

## Что нужно перед стартом Сессии 1

1. Скачать датасет с Kaggle → положить в `project/data/raw/creditcard.csv`
2. Убедиться что установлен Python 3.10+
3. Установить зависимости: `pip install -r requirements.txt`

---

## Стек технологий

| Категория | Инструмент |
|-----------|-----------|
| Язык | Python 3.11 |
| ML | scikit-learn, LightGBM, XGBoost |
| DL | PyTorch |
| Трекинг | MLflow |
| Сервис | FastAPI + Uvicorn |
| Метрики | prometheus_client |
| Контейнер | Docker + docker-compose |
| Тесты | pytest |