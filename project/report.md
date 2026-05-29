# Отчёт по итоговому проекту по курсу «Инженерия Искусственного Интеллекта»

> Рекомендуемый объём отчёта: 3-5 страниц в эквиваленте Markdown/печатного текста.  
> Отчёт должен позволить преподавателю понять задачу, данные, выбранные модели и результаты экспериментов.

---

## 1. Паспорт проекта

- **Название проекта:** Детекция мошеннических банковских транзакций (Fraud Detection)
- **Автор:** Петров Артём Андреевич
- **Группа:** ИКБО-42-24
- **Контакт:** @a_a_p01
- **Ссылка на репозиторий:** aie_dpo-1 (GitHub)

Проект реализует end-to-end сервис для автоматической детекции мошеннических транзакций по кредитным картам.
Обучено несколько ML-моделей (Logistic Regression, Random Forest, XGBoost, LightGBM, MLP), проведён сравнительный анализ,
лучшая модель (LightGBM, PR-AUC=0.84) упакована в REST-сервис на FastAPI и контейнеризирована через Docker.

---

## 2. Постановка задачи и контекст

1. **Предметная область и задача:**
   Банковские транзакции необходимо классифицировать как легитимные (Class=0) или мошеннические (Class=1).
   Потенциальный пользователь — антифрод-система банка, которая в режиме реального времени проверяет
   каждую транзакцию через REST API и при высокой вероятности мошенничества блокирует её.

2. **Формулировка задачи в терминах ML:**
   - Входные данные: 30 числовых признаков — Time (время с первой транзакции), Amount (сумма),
     V1–V28 (PCA-компоненты исходных признаков, анонимизированы).
   - Целевая переменная: бинарный Class (0 — легитимная, 1 — мошенничество).
   - Требования: latency < 10ms, высокий Recall (поймать как можно больше мошенничеств),
     умеренный FPR (ложные тревоги дорого стоят).

3. **Целевые метрики:**
   - **PR-AUC** (Area Under Precision-Recall Curve) — основная метрика при сильном дисбалансе классов
     (0.17% мошенничеств). ROC-AUC не информативен при таком дисбалансе.
   - **F1** — баланс Precision и Recall при выбранном пороге классификации.
   - **Precision@Recall≥0.80** — практическая метрика: насколько точен детектор, если требуем поймать 80%+ мошенничеств.

---

## 3. Данные

1. **Источник данных:**
   Открытый датасет [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   от ULB Machine Learning Group (Kaggle). Содержит транзакции европейских держателей карт за 2 дня сентября 2013 года.
   Датасет обезличен: признаки анонимизированы через PCA.

2. **Структура данных:**
   - Один файл `creditcard.csv`, 284 807 строк × 31 столбец.
   - Признаки: Time (float), Amount (float), V1–V28 (float, PCA), Class (int, целевая).
   - Дисбаланс: 492 мошенничества из 284 807 транзакций (0.17%).

3. **Предобработка и EDA** — `notebooks/01_eda.ipynb`, `src/data/pipeline.py`:
   - Стратифицированный сплит 60/20/20 (train/val/test), `random_state=42`.
   - StandardScaler только для Time и Amount (V1–V28 уже нормализованы PCA).
   - Препроцессор обучается ТОЛЬКО на train — утечки данных нет.
   - Ключевые находки EDA: классы сильно несбалансированы; мошеннические транзакции
     имеют меньший Amount и специфичные паттерны в V4, V11, V14, V17.

---

## 4. Модели и подходы

1. **Базовые модели** — `notebooks/02_baseline_models.ipynb`:
   - Logistic Regression (C=0.1, class_weight='balanced')
   - Random Forest (200 деревьев, max_depth=12, class_weight='balanced')
   - XGBoost (500 деревьев, scale_pos_weight=578)
   - LightGBM (500 деревьев, learning_rate=0.05, num_leaves=31, class_weight='balanced')
   - Все эксперименты залогированы в MLflow (experiment: `fraud_detection_baseline`).

2. **Ключевые настройки:**
   - `class_weight='balanced'` / `scale_pos_weight` — критически важны для несбалансированных данных.
   - Подбор порога классификации по val: оптимальный порог ≈ 0.11 (max F1), а не стандартные 0.5.

3. **Нейросетевая модель** — `notebooks/03_neural_net.ipynb`:
   - Архитектура: 30 → 128 (BN+ReLU+Dropout) → 64 (BN+ReLU+Dropout) → 32 → 1.
   - Focal Loss (α=0.25, γ=2) вместо стандартного BCE — снижает вес лёгких примеров.
   - Adam (lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau + Early Stopping (patience=8).
   - Обучение до 60 эпох с батчем 512 на CPU.

---

## 5. Экспериментальный протокол и результаты

1. **Экспериментальный протокол:**
   - Стратифицированный сплит: 60% train / 20% val / 20% test, `random_state=42`.
   - Все гиперпараметры подбирались на val, финальная оценка — только на test.
   - Порог классификации оптимизировался на val (max F1), применялся к test без повторного подбора.

2. **Сравнение моделей (val-выборка):**

   | Модель          | PR-AUC | ROC-AUC | F1     | P@R≥0.80 | Latency |
   |-----------------|--------|---------|--------|-----------|---------|
   | LogReg          | 0.6734 | 0.9748  | 0.1112 | 0.5128    | ~1ms    |
   | RandomForest    | 0.7798 | 0.9691  | 0.8022 | 0.6723    | ~5ms    |
   | XGBoost         | 0.8257 | 0.9722  | 0.8324 | 0.6957    | ~2ms    |
   | **LightGBM**    | **0.8359** | **0.9732** | **0.8432** | **0.8791** | **~2ms** |
   | MLP (FocalLoss) | см. nb03 | — | — | — | ~3ms |

3. **Выбор финальной модели:** LightGBM (`notebooks/04_final_model.ipynb`).
   - Лучший PR-AUC и F1 среди всех моделей на val.
   - P@R≥0.80 = 0.879 — при поимке 80%+ мошенничеств только ~12% ложных тревог.
   - Latency ~2ms — подходит для реального времени.
   - Не требует GPU, интерпретируем через feature importance.
   - Финальная модель дообучена на train+val; **PR-AUC на test = 0.886, F1 = 0.862**.

---

## 6. Архитектура решения и сервис

1. **Архитектура пайплайна:**
   ```
   creditcard.csv → pipeline.py (split + StandardScaler) → LightGBM (train)
                                                          → final_model.pkl + preprocessor.pkl + threshold.json
   Запрос → FastAPI /predict → preprocessor.transform() → model.predict_proba() → решение
   ```

2. **API endpoints** — `src/service/main.py`:
   - `GET /health` — возвращает `{"status": "ok", "model": "LightGBM", "threshold": 0.6316}`
   - `POST /predict` — принимает JSON с полями Time, Amount, V1–V28;
     возвращает `{"fraud_probability": float, "is_fraud": bool, "threshold": float}`
   - `GET /metrics` — Prometheus-метрики (счётчики запросов, latency, fraud count)
   - Опциональная авторизация через заголовок `x-api-key`

3. **Технологический стек:**
   - Python 3.12, FastAPI, uvicorn, Pydantic v2, prometheus-client, python-dotenv
   - scikit-learn, LightGBM, joblib
   - Docker (multi-stage build), docker-compose
   - Запуск: `docker compose up --build` или локально `uvicorn src.service.main:app --reload`

---

## 7. Наблюдаемость, конфигурация и безопасность

1. **Логи и наблюдаемость** — `src/service/observability.py`:
   - Структурированные логи в stdout (Python `logging`, формат timestamp+level+name+message).
   - При каждом вызове `/predict` логируется: сумма транзакции, вероятность мошенничества, решение.
   - Prometheus-метрики: `fraud_api_requests_total` (по статусу), `fraud_api_request_duration_seconds` (latency),
     `fraud_predictions_total` (по классу), `fraud_model_threshold`.
   - `/health` endpoint — немедленная проверка живости сервиса.

2. **Конфигурации** — `configs/.env.example`:
   - Пути к артефактам: `MODEL_PATH`, `PREPROCESSOR_PATH`, `THRESHOLD_PATH`.
   - Параметры сервера: `API_HOST`, `API_PORT`, `LOG_LEVEL`.
   - `API_KEY` — опциональная защита эндпоинтов.

3. **Безопасность:**
   - `.env` добавлен в `.gitignore`; в репозитории только `.env.example` с плейсхолдерами.
   - Датасет `creditcard.csv` (144MB, содержит финансовые данные) добавлен в `.gitignore`.
   - Модельные артефакты (`*.pkl`) тоже в `.gitignore` — пересоздаются из кода.
   - В Docker: non-root пользователь `appuser`.

---

## 8. Ограничения и дальнейшая работа

**Ограничения:**
- Данные 2013 года (европейские держатели карт) — возможен data drift на современных транзакциях.
- Признаки анонимизированы PCA — невозможно провести feature engineering или интерпретировать бизнес-смысл.
- Нет мониторинга data drift в production; модель не обновляется автоматически.
- Тестирование проводилось только на hold-out test set; нет A/B тестирования.

**Что добавить при наличии времени:**
- SMOTE / ADASYN для аугментации обучающей выборки.
- MLflow Model Registry для версионирования моделей.
- Grafana дашборд поверх Prometheus для визуализации метрик в реальном времени.
- CI/CD пайплайн с автоматическим переобучением при data drift.
- Мониторинг data drift через Evidently.

---

## 9. Сценарий демонстрации на защите

1. **Запускаю сервис** двумя способами (на выбор):
   ```bash
   # Вариант 1: Docker
   docker compose up --build
   # Вариант 2: локально
   cd project && uvicorn src.service.main:app --reload
   ```

2. **Два ключевых сценария:**
   - Запрос легитимной транзакции через `curl` / Swagger UI (`/docs`):
     ```bash
     curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"Time": 0, "Amount": 100, "V1": 0, ...}'
     ```
   - Открываю `GET /metrics` — показываю Prometheus-метрики: счётчики запросов, latency.

3. **На что обратить внимание:**
   - Весь ML-пайплайн воспроизводим: запуск ноутбуков 01→04 воссоздаёт все артефакты.
   - 15 тестов (`pytest tests/ -v`) проходят из чистого окружения.
   - Порог 0.6316 выбран на val, а не test — нет data leakage.
   - Все секреты в `.gitignore`, датасет не коммитился.

Команды запуска — в `project/README.md`, разделы 4.1–4.4.

---
