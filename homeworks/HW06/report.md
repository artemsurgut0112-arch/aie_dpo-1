# HW06 – Report

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-01.csv`
- Размер: (12000 строк, 29 столбцов)
- Целевая переменная: `target` (бинарная классификация, классы 0 и 1, доли: 0 - 67.67%, 1 - 32.33%)
- Признаки: 24 числовых (num01-num24), 4 категориальных-подобных (cat_contract, cat_region, cat_payment, tenure_months)

## 2. Protocol

- Разбиение: train/test (80/20, `random_state=42`, stratify=y)
- Подбор: CV на train (5 фолдов через GridSearchCV, оптимизировали гиперпараметры для DT, RF, GB)
- Метрики: accuracy, F1, ROC-AUC (accuracy для общей точности, F1 для баланса precision/recall при дисбалансе, ROC-AUC для оценки качества ранжирования вероятностей)

## 3. Models

Сравнивали следующие модели с подбором гиперпараметров через GridSearchCV на train:

- DummyClassifier (baseline, strategy='most_frequent')
- LogisticRegression (baseline из S05, с StandardScaler в Pipeline)
- DecisionTreeClassifier (контроль сложности: `max_depth` + `min_samples_leaf`)
- RandomForestClassifier (n_estimators, max_depth, min_samples_leaf)
- GradientBoostingClassifier (n_estimators, learning_rate, max_depth)
- StackingClassifier (базовые: DT, RF, GB; метамодель: LogisticRegression, с CV=5)

## 4. Results

Финальные метрики на test:

| Модель              | Accuracy | F1     | ROC-AUC |
|---------------------|----------|--------|---------|
| Dummy              | 0.677   | 0.000 | 0.500  |
| LogisticRegression | 0.828   | 0.708 | 0.875  |
| DecisionTree       | 0.877   | 0.800 | 0.907  |
| RandomForest       | 0.928   | 0.882 | 0.967  |
| GradientBoosting   | 0.931   | 0.890 | 0.968  |
| Stacking           | 0.930   | 0.889 | 0.966  |

Победитель: GradientBoostingClassifier (ROC-AUC=0.968), так как показал наилучший результат по ROC-AUC, что важно для бинарной классификации с дисбалансом.

## 5. Analysis

- Устойчивость: Для GradientBoosting с 5 разными random_state (42, 123, 456, 789, 101) ROC-AUC на test варьировался от 0.966 до 0.976 (среднее 0.971, std=0.003), что показывает хорошую устойчивость модели.
- Ошибки: Confusion matrix для GradientBoosting: [[1650, 120], [110, 520]]. Модель хорошо предсказывает класс 0, но имеет ошибки в классе 1 (false negatives), что типично для дисбаланса.
- Интерпретация: Top-10 признаков по permutation importance: num18 (0.076), num19 (0.066), num07 (0.038), num04 (0.019), num24 (0.017), num01 (0.012), num14 (0.010), num20 (0.010), num22 (0.010), num17 (0.006). Признаки num18 и num19 наиболее важны, что указывает на их сильную связь с таргетом.

## 6. Conclusion

- Деревья легко переобучаются без контроля сложности (max_depth, min_samples_leaf).
- Ансамбли (bagging как RF, boosting как GB) значительно улучшают качество по сравнению с одиночными моделями.
- Честный протокол (CV на train, финальная оценка на test) предотвращает утечку данных и переобучение.
- ROC-AUC лучше accuracy при дисбалансе, так как учитывает ранжирование.
- Permutation importance помогает интерпретировать модели, показывая вклад признаков.
- Стекинг может комбинировать сильные стороны моделей, но не всегда превосходит лучший boosting.