# S04 – eda_cli: HTTP-сервис качества датасетов (FastAPI)

Расширенная версия проекта `eda-cli` из Семинара 03.

К существующему CLI-приложению для EDA добавлен **HTTP-сервис на FastAPI** с эндпоинтами `/health`, `/quality` и `/quality-from-csv`.  
Используется в рамках Семинара 04 курса «Инженерия ИИ».

---

## Связь с S03 и HW03

Проект в S04/HW04 основан на том же пакете `eda_cli`, что и в S03/HW03:

- сохраняется структура `src/eda_cli/` и CLI-команда `eda-cli`;
- добавлен модуль `api.py` с FastAPI-приложением;
- в зависимости добавлены `fastapi` и `uvicorn[standard]`.

### Использование доработок из HW03

В рамках HW03 были реализованы **новые эвристики качества данных** в функции `compute_quality_flags()`:

- `has_constant_columns` – обнаружение колонок со всеми одинаковыми значениями;
- `has_high_cardinality_categoricals` – обнаружение категориальных колонок с очень большим числом уникальных значений (>50);
- `has_suspicious_id_duplicates` – обнаружение дубликатов в колонках, содержащих "id" в названии;
- `has_many_zero_values` – обнаружение числовых колонок с высокой долей нулевых значений (>30%).

Эти эвристики **полностью интегрированы** в HTTP-сервис HW04:

- эндпоинт `/quality-from-csv` использует их при расчёте `quality_score`;
- эндпоинт `/quality-flags-from-csv` возвращает все флаги, включая новые (это основной способ посмотреть результаты всех эвристик).

Цель S04/HW04 – показать, как поверх уже написанного EDA-ядра поднять простой HTTP-сервис и сделать его более функциональным благодаря доработкам из HW03.

---

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему
- Браузер (для Swagger UI `/docs`) или любой HTTP-клиент:
  - `curl` / HTTP-клиент в IDE / Postman / Hoppscotch и т.п.

---

## Инициализация проекта

В корне проекта (каталог S04/eda-cli):

```bash
uv sync
```

Команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml` (включая FastAPI и Uvicorn);
- установит сам проект `eda-cli` в окружение.

---

## Запуск CLI (как в S03)

CLI остаётся доступным и в S04.

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` - разделитель (по умолчанию `,`);
- `--encoding` - кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` - основной отчёт в Markdown;
- `summary.csv` - таблица по колонкам;
- `missing.csv` - пропуски по колонкам;
- `correlation.csv` - корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` - top-k категорий по строковым признакам;
- `hist_*.png` - гистограммы числовых колонок;
- `missing_matrix.png` - визуализация пропусков;
- `correlation_heatmap.png` - тепловая карта корреляций.

---

## Запуск HTTP-сервиса

HTTP-сервис реализован в модуле `eda_cli.api` на FastAPI.

### Запуск Uvicorn

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Пояснения:

- `eda_cli.api:app` - путь до объекта FastAPI `app` в модуле `eda_cli.api`;
- `--reload` - автоматический перезапуск сервера при изменении кода (удобно для разработки);
- `--port 8000` - порт сервиса (можно поменять при необходимости).

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000
```

---

## Эндпоинты сервиса

### 1. `GET /health`

Простейший health-check.

**Запрос:**

```http
GET /health
```

**Ожидаемый ответ `200 OK` (JSON):**

```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

Пример проверки через `curl`:

```bash
curl http://127.0.0.1:8000/health
```

---

### 2. Swagger UI: `GET /docs`

Интерфейс документации и тестирования API:

```text
http://127.0.0.1:8000/docs
```

Через `/docs` можно:

- вызывать `GET /health`;
- вызывать `POST /quality` (форма для JSON);
- вызывать `POST /quality-from-csv` (форма для загрузки файла);
- вызывать `POST /quality-flags-from-csv` (полный набор флагов);
- вызывать `POST /summary-from-csv` (JSON-сводка датасета).

---

### 3. `POST /quality` – заглушка по агрегированным признакам

Эндпоинт принимает **агрегированные признаки датасета** (размеры, доля пропусков и т.п.) и возвращает эвристическую оценку качества.

**Пример запроса:**

```http
POST /quality
Content-Type: application/json
```

Тело:

```json
{
  "n_rows": 10000,
  "n_cols": 12,
  "max_missing_share": 0.15,
  "numeric_cols": 8,
  "categorical_cols": 4
}
```

**Пример ответа `200 OK`:**

```json
{
  "ok_for_model": true,
  "quality_score": 0.8,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 3.2,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 10000,
    "n_cols": 12
  }
}
```

**Пример вызова через `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/quality" \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 10000, "n_cols": 12, "max_missing_share": 0.15, "numeric_cols": 8, "categorical_cols": 4}'
```

---

### 4. `POST /quality-from-csv` – оценка качества по CSV-файлу

Эндпоинт принимает CSV-файл, внутри:

- читает его в `pandas.DataFrame`;
- вызывает функции из `eda_cli.core`:

  - `summarize_dataset`,
  - `missing_table`,
  - `compute_quality_flags`;
- возвращает оценку качества датасета в том же формате, что `/quality`.

**Запрос:**

```http
POST /quality-from-csv
Content-Type: multipart/form-data
file: <CSV-файл>
```

Через Swagger:

- в `/docs` открыть `POST /quality-from-csv`,
- нажать `Try it out`,
- выбрать файл (например, `data/example.csv`),
- нажать `Execute`.

**Пример вызова через `curl` (Linux/macOS/WSL):**

```bash
curl -X POST "http://127.0.0.1:8000/quality-from-csv" \
  -F "file=@data/example.csv"
```

Ответ будет содержать:

- `ok_for_model` - результат по эвристикам;
- `quality_score` - интегральный скор качества;
- `flags` - булевы флаги из `compute_quality_flags`;
- `dataset_shape` - реальные размеры датасета (`n_rows`, `n_cols`);
- `latency_ms` - время обработки запроса.

---

### 5. `POST /quality-flags-from-csv` – полный набор флагов качества по CSV-файлу

Эндпоинт принимает CSV-файл и возвращает **полный набор флагов качества**, включая все эвристики, добавленные в HW03.

**Запрос:**

```http
POST /quality-flags-from-csv
Content-Type: multipart/form-data
file: <CSV-файл>
```

**Пример ответа `200 OK`:**

```json
{
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "max_missing_share": 0.08,
    "too_many_missing": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_suspicious_id_duplicates": false,
    "has_many_zero_values": false,
    "quality_score": 0.92
  },
  "latency_ms": 5.3,
  "dataset_shape": {
    "n_rows": 36,
    "n_cols": 14
  }
}
```

**Пример вызова через `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/quality-flags-from-csv" \
  -F "file=@data/example.csv"
```

Особенности:

- Возвращает **все флаги качества**, не только булевы (включая численные значения);
- Показывает результаты новых эвристик из HW03: константные колонки, высокая кардинальность, дубликаты ID, много нулей;
- Полезен для детального анализа проблем в датасете.

---

### 6. `POST /summary-from-csv` – JSON-сводка датасета

Эндпоинт принимает CSV-файл и возвращает **подробную JSON-сводку** датасета (аналог CLI-опции `--json-summary`).

**Запрос:**

```http
POST /summary-from-csv
Content-Type: multipart/form-data
file: <CSV-файл>
```

**Пример ответа `200 OK`:**

```json
{
  "summary": {
    "n_rows": 36,
    "n_cols": 14,
    "columns": [
      {
        "name": "user_id",
        "dtype": "int64",
        "non_null": 36,
        "missing": 0,
        "missing_share": 0.0,
        "unique": 35,
        "example_values": ["1001", "1002", "1003"],
        "is_numeric": true,
        "min": 1001.0,
        "max": 1035.0,
        "mean": 1018.194,
        "std": 10.167
      },
      ...
    ]
  },
  "latency_ms": 4.1
}
```

**Пример вызова через `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/summary-from-csv" \
  -F "file=@data/example.csv"
```

Особенности:

- Возвращает **полную сводку по каждой колонке** (типы, пропуски, уникальные значения, статистика);
- Включает примеры значений для каждой колонки;
- Полезна для построения более сложных аналитических инструментов и клиентских приложений.

---

## Структура проекта (упрощённо)

```text
S04/
  eda-cli/
    pyproject.toml
    README.md                # этот файл
    src/
      eda_cli/
        __init__.py
        core.py              # EDA-логика, эвристики качества
        viz.py               # визуализации
        cli.py               # CLI (overview/report)
        api.py               # HTTP-сервис (FastAPI)
    tests/
      test_core.py           # тесты ядра
    data/
      example.csv            # учебный CSV для экспериментов
```

---

## Тесты

Запуск тестов (как и в S03):

```bash
uv run pytest -q
```

Рекомендуется перед любыми изменениями в логике качества данных и API:

1. Запустить тесты `pytest`;
2. Проверить работу CLI (`uv run eda-cli ...`);
3. Проверить работу HTTP-сервиса (`uv run uvicorn ...`, затем `/health` и `/quality`/`/quality-from-csv` через `/docs` или HTTP-клиент).
