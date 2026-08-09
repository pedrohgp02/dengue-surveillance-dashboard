"""Tests for historical multi-horizon forecast evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.multihorizon import (
    run_multi_horizon_backtest,
    summarize_multi_horizon_backtest,
)


def make_history(
    periods: int = 76,
) -> pd.DataFrame:
    """Create a synthetic weekly dengue time series."""
    weeks = np.arange(periods)

    notifications = (
        100
        + 30
        * np.sin(
            2
            * np.pi
            * weeks
            / 52
        )
        + 0.25
        * weeks
    )

    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2023-01-02",
                periods=periods,
                freq="W-MON",
            ),
            "notifications": (
                notifications
                .round()
                .astype(int)
            ),
        }
    )


@pytest.fixture(scope="module")
def multihorizon_results():
    """Run one small real multi-horizon evaluation for shared tests."""
    history = make_history()

    results = run_multi_horizon_backtest(
        history,
        horizons=(1, 2, 4),
        include_nb=False,
        min_history_weeks=60,

        # Large step keeps CI fast while still exercising
        # the real recursive forecasting pipeline.
        origin_step=20,
    )

    return history, results


def test_requested_horizons_are_present(
    multihorizon_results,
):
    """The evaluator should return every requested horizon."""
    _, results = multihorizon_results

    assert set(
        results["horizon"]
    ) == {
        1,
        2,
        4,
    }


def test_expected_models_are_evaluated(
    multihorizon_results,
):
    """All standard non-NB models should appear in evaluation output."""
    _, results = multihorizon_results

    expected_models = {
        "Naive",
        "Seasonal Naive",
        "Linear Regression",
        "Random Forest",
    }

    assert expected_models.issubset(
        set(
            results["model"]
        )
    )


def test_target_dates_are_after_forecast_origins(
    multihorizon_results,
):
    """Every evaluated target must occur after its forecast origin."""
    _, results = multihorizon_results

    assert (
        results["target_date"]
        > results["origin_date"]
    ).all()


def test_target_date_matches_forecast_horizon(
    multihorizon_results,
):
    """A horizon-h forecast should target exactly h weeks ahead."""
    _, results = multihorizon_results

    expected_days = (
        results["horizon"]
        * 7
    )

    actual_days = (
        results["target_date"]
        - results["origin_date"]
    ).dt.days

    assert np.array_equal(
        actual_days.to_numpy(),
        expected_days.to_numpy(),
    )


def test_predictions_are_finite_and_nonnegative(
    multihorizon_results,
):
    """Historical forecasts should remain finite and nonnegative."""
    _, results = multihorizon_results

    predictions = results[
        "prediction"
    ].to_numpy(
        dtype=float
    )

    assert np.isfinite(
        predictions
    ).all()

    assert (
        predictions
        >= 0
    ).all()


def test_future_observations_do_not_change_past_forecasts():
    """Changing future data must not alter an earlier forecast.

    The first historical forecast origin contains only the first
    60 observations. Everything occurring after that point is changed
    dramatically in a second copy of the dataset.

    If the evaluation is leakage-free, predictions issued from that
    first origin must remain exactly the same.
    """
    original = make_history()

    modified = original.copy()

    # First forecast origin when min_history_weeks=60 is row 59.
    # Change everything AFTER that origin.
    modified.loc[
        modified.index >= 60,
        "notifications",
    ] += 10_000

    original_results = (
        run_multi_horizon_backtest(
            original,
            horizons=(1, 2, 4),
            include_nb=False,
            min_history_weeks=60,

            # With 76 observations and max horizon 4,
            # this yields only the first origin.
            origin_step=20,
        )
    )

    modified_results = (
        run_multi_horizon_backtest(
            modified,
            horizons=(1, 2, 4),
            include_nb=False,
            min_history_weeks=60,
            origin_step=20,
        )
    )

    original_predictions = (
        original_results
        .sort_values(
            [
                "horizon",
                "model",
            ]
        )[
            [
                "horizon",
                "model",
                "prediction",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    modified_predictions = (
        modified_results
        .sort_values(
            [
                "horizon",
                "model",
            ]
        )[
            [
                "horizon",
                "model",
                "prediction",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    assert (
        original_predictions[
            [
                "horizon",
                "model",
            ]
        ].equals(
            modified_predictions[
                [
                    "horizon",
                    "model",
                ]
            ]
        )
    )

    assert np.allclose(
        original_predictions[
            "prediction"
        ],
        modified_predictions[
            "prediction"
        ],
    )


def test_summary_calculates_metrics_correctly():
    """Summary metrics should match known hand-calculated values."""
    results = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(
                [
                    "2025-01-06",
                    "2025-01-13",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2025-01-13",
                    "2025-01-20",
                ]
            ),
            "horizon": [
                1,
                1,
            ],
            "model": [
                "Example Model",
                "Example Model",
            ],
            "actual": [
                100.0,
                120.0,
            ],
            "prediction": [
                90.0,
                130.0,
            ],
            "error": [
                -10.0,
                10.0,
            ],
            "absolute_error": [
                10.0,
                10.0,
            ],
        }
    )

    summary = (
        summarize_multi_horizon_backtest(
            results
        )
    )

    row = summary.iloc[0]

    assert row["Horizon"] == 1
    assert row["Model"] == "Example Model"
    assert row["N"] == 2

    assert row["MAE"] == pytest.approx(
        10.0
    )

    assert row["RMSE"] == pytest.approx(
        10.0
    )

    assert row["Bias"] == pytest.approx(
        0.0
    )


def test_empty_horizon_list_is_rejected():
    """At least one forecast horizon must be requested."""
    history = make_history()

    with pytest.raises(
        ValueError,
        match="At least one forecast horizon",
    ):
        run_multi_horizon_backtest(
            history,
            horizons=(),
        )


def test_zero_horizon_is_rejected():
    """Forecast horizons must be positive."""
    history = make_history()

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        run_multi_horizon_backtest(
            history,
            horizons=(0, 1),
        )


def test_invalid_origin_step_is_rejected():
    """Historical forecast origins must advance by at least one week."""
    history = make_history()

    with pytest.raises(
        ValueError,
        match="origin_step must be at least 1",
    ):
        run_multi_horizon_backtest(
            history,
            horizons=(1, 2),
            origin_step=0,
        )


def test_insufficient_history_is_rejected():
    """Evaluation should fail clearly when history is too short."""
    history = make_history(
        periods=40
    )

    with pytest.raises(
        ValueError,
        match="Not enough historical observations",
    ):
        run_multi_horizon_backtest(
            history,
            horizons=(1, 4, 8),
            min_history_weeks=35,
        )
