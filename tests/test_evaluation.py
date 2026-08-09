"""Tests for model evaluation and backtesting."""

import numpy as np
import pandas as pd

from src.evaluation import (
    build_error_table,
    determine_holdout_start,
    run_backtest,
    score_predictions,
)
from src.features import create_features


def test_perfect_predictions_have_zero_error():
    """Perfect predictions should produce zero MAE and RMSE."""
    actual = np.array(
        [10, 20, 30, 25],
        dtype=float,
    )

    predictions = actual.copy()

    previous_week = np.array(
        [8, 10, 20, 30],
        dtype=float,
    )

    scores = score_predictions(
        actual,
        predictions,
        previous_week,
    )

    assert scores["MAE"] == 0.0
    assert scores["RMSE"] == 0.0
    assert scores["Direction Acc."] == 1.0


def test_rising_week_metrics_are_correct():
    """Correct rise/no-rise predictions should score perfectly."""
    actual = np.array(
        [20, 10],
        dtype=float,
    )

    predictions = np.array(
        [18, 12],
        dtype=float,
    )

    previous_week = np.array(
        [10, 15],
        dtype=float,
    )

    scores = score_predictions(
        actual,
        predictions,
        previous_week,
    )

    assert scores["Rising Recall"] == 1.0
    assert scores["Rising Precision"] == 1.0
    assert scores["Rising F1"] == 1.0
    assert scores["False Alarm Rate"] == 0.0


def test_false_alarm_is_detected():
    """Predicting a rise when activity falls should count as a false alarm."""
    actual = np.array(
        [8],
        dtype=float,
    )

    predictions = np.array(
        [12],
        dtype=float,
    )

    previous_week = np.array(
        [10],
        dtype=float,
    )

    scores = score_predictions(
        actual,
        predictions,
        previous_week,
    )

    assert scores["False Alarm Rate"] == 1.0
    assert scores["Direction Acc."] == 0.0


def test_error_table_ranks_perfect_model_first():
    """The model with the lowest forecast error should rank first."""
    backtest = pd.DataFrame(
        {
            "actual": [
                10,
                20,
                30,
            ],
            "perfect_pred": [
                10,
                20,
                30,
            ],
            "poor_pred": [
                30,
                30,
                30,
            ],
        }
    )

    model_cols = {
        "Perfect Model": "perfect_pred",
        "Poor Model": "poor_pred",
    }

    table = build_error_table(
        backtest,
        model_cols,
    )

    assert table.iloc[0]["Model"] == "Perfect Model"
    assert table.iloc[0]["MAE"] == 0.0
    assert table.iloc[0]["RMSE"] == 0.0

    assert (
        table.iloc[1]["MAE"]
        > table.iloc[0]["MAE"]
    )


def test_holdout_start_uses_requested_final_window():
    """Holdout start should select the requested number of final weeks."""
    dates = pd.date_range(
        "2024-01-01",
        periods=60,
        freq="W-MON",
    )

    backtest = pd.DataFrame(
        {
            "date": dates,
        }
    )

    holdout_start = determine_holdout_start(
        backtest,
        holdout_weeks=10,
        minimum_holdout_weeks=1,
    )

    expected_start = dates[-10]

    assert holdout_start == expected_start


def test_default_holdout_enforces_minimum_window():
    """The default logic should preserve at least 26 holdout weeks."""
    dates = pd.date_range(
        "2024-01-01",
        periods=60,
        freq="W-MON",
    )

    backtest = pd.DataFrame(
        {
            "date": dates,
        }
    )

    holdout_start = determine_holdout_start(
        backtest,
        holdout_weeks=10,
    )

    expected_start = dates[-26]

    assert holdout_start == expected_start


def test_expanding_window_backtest_has_expected_length():
    """Walk-forward backtesting should produce one row per forecast week."""
    periods = 90

    data = pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-02",
                periods=periods,
                freq="W-MON",
            ),
            "notifications": (
                100
                + 20
                * np.sin(
                    np.arange(periods)
                    * 2
                    * np.pi
                    / 52
                )
                + np.arange(periods)
                * 0.2
            ).astype(int),
        }
    )

    features = create_features(
        data
    )

    min_train = 26

    results, model_cols = run_backtest(
        features,
        include_nb=False,
        min_backtest_train=min_train,
    )

    assert len(results) == (
        len(features)
        - min_train
    )

    assert {
        "Naive",
        "Seasonal Naive",
        "Linear Regression",
        "Random Forest",
    }.issubset(
        model_cols.keys()
    )


def test_backtest_predictions_follow_chronological_order():
    """Backtest output should remain ordered from earlier to later weeks."""
    periods = 90

    data = pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-02",
                periods=periods,
                freq="W-MON",
            ),
            "notifications": np.arange(
                50,
                50 + periods,
            ),
        }
    )

    features = create_features(
        data
    )

    results, _ = run_backtest(
        features,
        include_nb=False,
        min_backtest_train=26,
    )

    assert results[
        "date"
    ].is_monotonic_increasing

    assert (
        results.iloc[0]["date"]
        == features.iloc[26]["date"]
    )
