import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42
TARGET_COL = "Class"
SCALE_COLS = ["Time", "Amount"]
V_FEATURES = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES = SCALE_COLS + V_FEATURES


def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            "Скачайте датасет с https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "и положите в project/data/creditcard.csv"
        )
    df = pd.read_csv(path)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    required = set(ALL_FEATURES + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {missing}")
    if df.isnull().any().any():
        raise ValueError("Датасет содержит пропуски")


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Стратифицированный сплит 60/20/20. Доля мошенничеств одинакова в каждой части."""
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state,
    )

    train = pd.concat([X_train, y_train], axis=1)
    val   = pd.concat([X_val,   y_val],   axis=1)
    test  = pd.concat([X_test,  y_test],  axis=1)

    print(f"Train: {len(train):,}  |  fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
    print(f"Val:   {len(val):,}  |  fraud: {y_val.sum()} ({y_val.mean()*100:.3f}%)")
    print(f"Test:  {len(test):,}  |  fraud: {y_test.sum()} ({y_test.mean()*100:.3f}%)")

    return train, val, test


def build_preprocessor() -> ColumnTransformer:
    # V1-V28 уже нормализованы PCA — масштабируем только Time и Amount
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), SCALE_COLS)],
        remainder="passthrough",
    )


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[ALL_FEATURES], df[TARGET_COL]


def fit_preprocessor(
    train: pd.DataFrame,
    save_path: str | None = None,
) -> ColumnTransformer:
    """Обучаем препроцессор только на train, чтобы не было утечки данных."""
    X_train, _ = get_X_y(train)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, save_path)
        print(f"Препроцессор сохранён: {save_path}")

    return preprocessor


def apply_preprocessor(preprocessor: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    X, _ = get_X_y(df)
    return preprocessor.transform(X)
