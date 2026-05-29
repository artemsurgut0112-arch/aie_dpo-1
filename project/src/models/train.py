"""
Полный пайплайн обучения финальной модели.

Запуск из папки project/:
    python -m src.models.train
    python -m src.models.train --data data/creditcard.csv --out models/
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from src.data.pipeline import (
    apply_preprocessor,
    fit_preprocessor,
    get_X_y,
    load_data,
    split_data,
)


def train(data_path: str = "data/creditcard.csv", out_dir: str = "models/") -> dict:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    print("Загрузка данных...")
    df = load_data(data_path)
    train_df, val_df, test_df = split_data(df)

    print("Препроцессинг...")
    preprocessor = fit_preprocessor(train_df, save_path=str(out / "preprocessor.pkl"))
    X_train = apply_preprocessor(preprocessor, train_df)
    X_val   = apply_preprocessor(preprocessor, val_df)
    X_test  = apply_preprocessor(preprocessor, test_df)
    _, y_train = get_X_y(train_df)
    _, y_val   = get_X_y(val_df)
    _, y_test  = get_X_y(test_df)

    # Подбираем порог на val, финальную модель учим на train+val
    print("Подбор порога на val...")
    val_model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", n_jobs=-1, verbose=-1, random_state=42,
    )
    val_model.fit(X_train, y_train)
    scores_val = val_model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    f1s = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = f1s.argmax()
    threshold = float(thresholds[best_idx])

    print(f"  порог = {threshold:.4f}, val F1 = {f1s[best_idx]:.4f}")

    print("Обучение финальной модели на train+val...")
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    model = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        class_weight="balanced", n_jobs=-1, verbose=-1, random_state=42,
    )
    model.fit(X_trainval, y_trainval)

    print("Оценка на test...")
    scores_test = model.predict_proba(X_test)[:, 1]
    preds_test  = (scores_test >= threshold).astype(int)

    metrics = {
        "pr_auc":  round(float(average_precision_score(y_test, scores_test)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, scores_test)), 4),
        "f1":      round(float(f1_score(y_test, preds_test)), 4),
        "threshold": round(threshold, 6),
    }

    print("\nМетрики на test:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nСохранение артефактов...")
    joblib.dump(model, out / "final_model.pkl")

    threshold_data = {
        "threshold": threshold,
        "model": "LightGBM",
        "val_f1": round(float(f1s[best_idx]), 4),
        "val_pr_auc": round(float(average_precision_score(y_val, scores_val)), 4),
    }
    with open(out / "threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)

    model_card = f"""# Model Card — LightGBM Fraud Detector

## Метрики (test)
- PR-AUC:  {metrics['pr_auc']}
- ROC-AUC: {metrics['roc_auc']}
- F1:      {metrics['f1']}
- Порог:   {threshold:.4f}

## Данные
- Time, Amount — StandardScaler
- V1–V28 — PCA-компоненты, без изменений

## Ограничения
- Датасет 2013 года (европейские держатели карт)
- Признаки анонимизированы — feature engineering невозможен
- Без мониторинга data drift не рекомендуется для продакшна
"""
    with open(out / "model_card.md", "w") as f:
        f.write(model_card)

    print(f"Готово. Артефакты в {out}/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели детекции мошенничеств")
    parser.add_argument("--data", default="data/creditcard.csv", help="Путь к датасету")
    parser.add_argument("--out",  default="models/",             help="Папка для артефактов")
    args = parser.parse_args()
    train(args.data, args.out)
