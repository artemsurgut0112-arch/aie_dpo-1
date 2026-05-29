import logging
import sys

from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Количество запросов к API",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "Время обработки запроса",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

FRAUD_COUNT = Counter(
    "fraud_predictions_total",
    "Количество предсказаний по классу",
    ["decision"],
)

MODEL_INFO = Gauge(
    "fraud_model_threshold",
    "Текущий порог классификации",
)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger("fraud_api")
