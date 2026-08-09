"""Tests for multi-step dengue forecasting."""

import numpy as np
import pandas as pd
import pytest

from src.features import create_features
from src.forecasting import (
    _bounded_seasonal_anchor,
    _stabilize_forecast,
    build_multi_step_forecast,
    build_next_week_features,
)
from src.models import fit_model_bundle


def make_history(
    periods: int = 80,
) -> pd.DataFrame:
    """Create a realistic synthetic weekly dengue series."""
    weeks = np.arange(periods)

    notifications = (
        100
        + 25
        * np.sin(
            weeks
            * 2
            * np.pi
            / 52
        )
        + 0.3 * weeks
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


def make_model_bundle(
    history: pd.DataFrame,
):
    """Fit the normal project models on synthetic history."""
    features = create_features(
        history
    )

    return fit_model_bundle(
        features,
        include_nb=False,
    )


def test_next_week_features_uses_correct_date():
    """Next-week features should advance exactly seven days."""
    history = make_history()

    result = build_next_week_features(
        history
    )

    expected_date = (
        history["date"].iloc[-1]
        + pd.Timedelta(weeks=1)
    )

    assert (
        result.iloc[0]["date"]
        == expected_date
    )


def test_next_week_features_uses_latest_notification_as_lag1():
    """lag1 should equal the latest observed notification count."""
    history = make_history()

    result = build_next_week_features(
        history
    )

    assert result.iloc[0]["lag1"] == (
        history[
            "notifications"
        ].iloc[-1]
    )


def test_next_week_features_uses_four_week_lag():
    """lag4 should equal the count four observations before the forecast."""
    history = make_history()

    result = build_next_week_features(
        history
    )

    assert result.iloc[0]["lag4"] == (
        history[
            "notifications"
        ].iloc[-4]
    )


def test_next_week_features_requires_seasonal_history():
    """Forecast feature creation should fail without a 52-week reference."""
    history = make_history(
        periods=30
    )

    with pytest.raises(
        ValueError,
        match="52 weeks",
    ):
        build_next_week_features(
            history
        )


def test_stabilizer_prevents_negative_forecast():
    """Forecast stabilization should never return a negative value."""
    result = _stabilize_forecast(
        raw_value=-100.0,
        last_value=10.0,
        anchor_value=-50.0,
        max_step_delta=20.0,
        blend_weight=0.5,
        ceiling=100.0,
    )

    assert result >= 0.0


def test_stabilizer_respects_upper_step_limit():
    """A learned forecast should not jump above the permitted step size."""
    result = _stabilize_forecast(
        raw_value=1000.0,
        last_value=100.0,
        anchor_value=1000.0,
        max_step_delta=25.0,
        blend_weight=0.5,
        ceiling=1000.0,
    )

    assert result <= 125.0


def test_stabilizer_respects_lower_step_limit():
    """A learned forecast should not fall faster than the permitted step."""
    result = _stabilize_forecast(
        raw_value=0.0,
        last_value=100.0,
        anchor_value=0.0,
        max_step_delta=25.0,
        blend_weight=0.5,
        ceiling=1000.0,
    )

    assert result >= 75.0


def test_stabilizer_respects_global_ceiling():
    """The forecast should never exceed the supplied ceiling."""
    result = _stabilize_forecast(
        raw_value=1000.0,
        last_value=100.0,
        anchor_value=1000.0,
        max_step_delta=500.0,
        blend_weight=0.5,
        ceiling=150.0,
    )

    assert result <= 150.0


def test_seasonal_anchor_caps_extreme_previous_year_value():
    """An extreme lag52 value should be bounded by recent activity."""
    result = _bounded_seasonal_anchor(
        lag1=100.0,
        lag4=90.0,
        lag52=5000.0,
        typical_step_delta=20.0,
    )

    recent_anchor = (
        0.65 * 100.0
        + 0.35 * 90.0
    )

    expected_cap = max(
        recent_anchor * 2.2,
        recent_anchor
        + 3.0 * 20.0,
    )

    assert result <= expected_cap


def test_zero_horizon_raises_error():
    """Forecast horizon must contain at least one future week."""
    history = make_history()

    bundle = make_model_bundle(
        history
    )

    with pytest.raises(
        ValueError,
        match="horizon must be at least 1",
    ):
        build_multi_step_forecast(
            history,
            bundle,
            horizon=0,
        )


def test_multi_step_forecast_returns_requested_horizon():
    """The forecast table should contain exactly one row per requested week."""
    history = make_history()

    bundle = make_model_bundle(
        history
    )

    horizon = 6

    forecast = build_multi_step_forecast(
        history,
        bundle,
        horizon=horizon,
    )

    assert len(forecast) == horizon


def test_multi_step_forecast_steps_are_sequential():
    """Forecast step numbers should run from 1 through the horizon."""
    history = make_history()

    bundle = make_model_bundle(
        history
    )

    forecast = build_multi_step_forecast(
        history,
        bundle,
        horizon=6,
    )

    assert forecast[
        "step"
    ].tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_multi_step_forecast_dates_advance_weekly():
    """Each recursive forecast should advance by exactly one week."""
    history = make_history()

    bundle = make_model_bundle(
        history
    )

    forecast = build_multi_step_forecast(
        history,
        bundle,
        horizon=4,
    )

    expected_dates = pd.date_range(
        history["date"].max()
        + pd.Timedelta(weeks=1),
        periods=4,
        freq="W-MON",
    )

    assert forecast[
        "date"
    ].tolist() == list(
        expected_dates
    )


def test_multi_step_forecasts_are_nonnegative():
    """Every model forecast should remain nonnegative."""
    history = make_history()

    bundle = make_model_bundle(
        history
    )

    forecast = build_multi_step_forecast(
        history,
        bundle,
        horizon=8,
    )

    model_columns = [
        "Naive",
        "Seasonal Naive",
        "Linear Regression",
        "Random Forest",
    ]

    for column in model_columns:
        assert (
            forecast[column]
            >= 0
        ).all()


def test_multi_step_forecasts_are_finite():
    """Forecasts should not contain NaN or infinite values."""
    history = make_history()

    bundle = make_model_bundle(
        history
    )

    forecast = build_multi_step_forecast(
        history,
        bundle,
        horizon=8,
    )

    model_columns = [
        "Naive",
        "Seasonal Naive",
        "Linear Regression",
        "Random Forest",
    ]

    for column in model_columns:
        assert np.isfinite(
            forecast[column]
        ).all()


def test_naive_forecast_stays_constant():
    """Recursive naive forecasting should repeat the latest observation."""
    history = make_history()

    bundle = make_model_bundle(
        history
    )

    forecast = build_multi_step_forecast(
        history,
        bundle,
        horizon=5,
    )

    latest_value = float(
        history[
            "notifications"
        ].iloc[-1]
    )

    assert np.allclose(
        forecast["Naive"],
        latest_value,
    )
