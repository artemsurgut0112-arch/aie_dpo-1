# Model Card — LightGBM Fraud Detector

## Метрики (test)
- PR-AUC:  0.886
- ROC-AUC: 0.979
- F1:      0.8615
- Порог:   0.1095

## Данные
- Time, Amount — StandardScaler
- V1–V28 — PCA-компоненты, без изменений

## Ограничения
- Датасет 2013 года (европейские держатели карт)
- Признаки анонимизированы — feature engineering невозможен
- Без мониторинга data drift не рекомендуется для продакшна
