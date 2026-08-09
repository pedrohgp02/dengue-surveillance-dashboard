"""Tests for dengue feature engineering."""

import numpy as np
import pandas as pd

from src.features import create_features


def make_weekly_data(
    periods: int = 60,
) -> pd.DataFrame:
    """Create a small synthetic weekly dengue series."""
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2024-01-01",
                periods=periods,
                freq="W-MON",
            ),
            "notifications": np.arange(
                periods,
                dtype=int,
            ),
        }
    )


def test_create_features_adds_expected_columns():
    """Feature engineering should create all lag and seasonal fields."""
    data = make_weekly_data()

    result = create_features(data)

    expected_columns = {
        "lag1",
        "lag4",
        "lag52",
        "week_sin",
        "week_cos",
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_lag1_uses_previous_week():
    """lag1 should equal the previous week's notification count."""
    data = make_weekly_data()

    result = create_features(data)

    first_row = result.iloc[0]

    original_index = 4

    assert first_row["lag1"] == (
        data.iloc[
            original_index - 1
        ]["notifications"]
    )


def test_lag4_uses_four_weeks_before():
    """lag4 should equal the notification count four weeks earlier."""
    data = make_weekly_data()

    result = create_features(data)

    first_row = result.iloc[0]

    assert first_row["lag4"] == (
        data.iloc[0][
            "notifications"
        ]
    )


def test_seasonal_features_are_bounded():
    """Cyclical seasonal features must remain between -1 and 1."""
    data = make_weekly_data()

    result = create_features(data)

    assert (
        result["week_sin"]
        .between(-1, 1)
        .all()
    )

    assert (
        result["week_cos"]
        .between(-1, 1)
        .all()
    )


def test_original_dataframe_is_not_modified():
    """Feature engineering should operate on a copy."""
    data = make_weekly_data()

    original_columns = list(
        data.columns
    )

    create_features(data)

    assert list(
        data.columns
    ) == original_columns
