from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    get_problematic_columns,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_constant_columns():
    """Проверка detection of constant columns (все значения одинаковые)"""
    df = pd.DataFrame(
        {
            "constant_col": [1, 1, 1, 1],
            "normal_col": [10, 20, 30, 40],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_constant_columns"] is True


def test_high_cardinality_categoricals():
    """Проверка detection of high-cardinality categorical columns"""
    # Создаём DataFrame с категориальной колонкой, где 100+ уникальных значений
    df = pd.DataFrame(
        {
            "id": range(100),
            "category": [f"cat_{i}" for i in range(100)],  # 100 уникальных значений
            "normal": [1] * 100,
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_high_cardinality_categoricals"] is True


def test_suspicious_id_duplicates():
    """Проверка detection of duplicate IDs"""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 2, 3],  # user_id 2 дублируется
            "value": [10, 20, 30, 40],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_suspicious_id_duplicates"] is True


def test_many_zero_values():
    """Проверка detection of many zero values in numeric columns"""
    df = pd.DataFrame(
        {
            "sparse_col": [0, 0, 0, 0, 0, 1, 2, 3],  # 62.5% нулей (> 30%)
            "normal_col": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)

    assert flags["has_many_zero_values"] is True


def test_problematic_columns_by_missing_share():
    """Проверка get_problematic_columns с различными порогами"""
    df = pd.DataFrame(
        {
            "col_30pct": [1, 2, 3, None, None, None, 4, 5, 6, 7],  # 30% пропусков
            "col_50pct": [1, None, None, None, None, 6, 7, 8, 9, 10],  # 50% пропусков
            "col_complete": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0% пропусков
        }
    )
    missing_df = missing_table(df)

    # С порогом 0.25 (25%) должны быть найдены col_30pct и col_50pct
    problematic_025 = get_problematic_columns(missing_df, threshold=0.25)
    assert len(problematic_025) == 2
    assert "col_30pct" in problematic_025.index
    assert "col_50pct" in problematic_025.index

    # С порогом 0.4 (40%) должна быть найдена только col_50pct
    problematic_040 = get_problematic_columns(missing_df, threshold=0.4)
    assert len(problematic_040) == 1
    assert "col_50pct" in problematic_040.index

    # С порогом 0.6 (60%) не должны быть найдены колонки
    problematic_060 = get_problematic_columns(missing_df, threshold=0.6)
    assert len(problematic_060) == 0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2
