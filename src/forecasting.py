"""Forecast construction and stabilization utilities."""

import math

import numpy as np
import pandas as pd

from src.models import ModelBundle, predict_with_bundle


def build_next_week_features(history: pd.DataFrame) -> pd.DataFrame:
    """Create features for the next unobserved week.

    Parameters
    ----------
    history:
        Weekly dengue series with ``date`` and ``notifications`` columns.

    Returns
    -------
    pd.DataFrame
        One-row feature frame for the next week.

    Raises
    ------
    ValueError
        If a same-week-last-year observation is unavailable.
    """
    next_date = history["date"].max() + pd.Timedelta(weeks=1)

    lag52_series = history.loc[
        history["date"] == next_date - pd.Timedelta(weeks=52),
        "notifications",
    ]

    if lag52_series.empty:
        raise ValueError(
            "Need at least 52 weeks of history for seasonal features."
        )

    next_week_number = int(next_date.isocalendar().week)

    return pd.DataFrame(
        {
            "date": [next_date],
            "lag1": [history["notifications"].iloc[-1]],
            "lag4": [history["notifications"].iloc[-4]],
            "lag52": [lag52_series.iloc[0]],
            "week_sin": [
                np.sin(2 * np.pi * next_week_number / 52)
            ],
            "week_cos": [
                np.cos(2 * np.pi * next_week_number / 52)
            ],
        }
    )


def _stabilize_forecast(
    raw_value: float,
    last_value: float,
    anchor_value: float,
    max_step_delta: float,
    blend_weight: float,
    ceiling: float,
) -> float:
    """Blend a raw prediction with an anchor and constrain its movement."""
    blended = (
        (1.0 - blend_weight) * raw_value
        + blend_weight * anchor_value
    )

    lower = max(0.0, last_value - max_step_delta)
    upper = min(ceiling, last_value + max_step_delta)

    return float(
        np.clip(
            blended,
            lower,
            upper,
        )
    )


def _bounded_seasonal_anchor(
    lag1: float,
    lag4: float,
    lag52: float,
    typical_step_delta: float,
) -> float:
    """Cap a seasonal reference relative to recent dengue activity."""
    recent_anchor = 0.65 * lag1 + 0.35 * lag4

    seasonal_cap = max(
        recent_anchor * 2.2,
        recent_anchor + 3.0 * typical_step_delta,
    )

    return float(
        min(
            lag52,
            seasonal_cap,
        )
    )


def _estimate_recent_trend(series: pd.Series) -> float:
    """Estimate weekly trend from the most recent eight observations."""
    recent_short = series.tail(4)
    recent_previous = series.tail(8).head(4)

    if (
        len(recent_short) == 4
        and len(recent_previous) == 4
    ):
        return float(
            (
                recent_short.mean()
                - recent_previous.mean()
            )
            / 4.0
        )

    return 0.0


def choose_multi_step_display_model(
    ranking_frame: pd.DataFrame,
    forecast_df: pd.DataFrame,
    latest_actual: float,
    preferred_model: str,
) -> str:
    """Choose a stable learned model for the multi-step chart.

    Baseline models are excluded. Learned models receive a trajectory
    penalty when their forecasts end far above the latest observation,
    contain unusually large week-to-week jumps, or reach very high
    peaks. The lowest penalized score is selected.
    """
    baseline_models = {
        "Naive",
        "Seasonal Naive",
    }

    candidate_models = [
        model_name
        for model_name in ranking_frame.get(
            "Model",
            pd.Series(dtype=object),
        ).tolist()
        if (
            model_name in forecast_df.columns
            and model_name not in baseline_models
        )
    ]

    if not candidate_models:
        return preferred_model

    latest_scale = max(
        float(latest_actual),
        1.0,
    )

    scored_candidates: list[
        tuple[float, float, str]
    ] = []

    for model_name in candidate_models:
        series = np.asarray(
            forecast_df[model_name].to_numpy(dtype=float),
            dtype=float,
        )

        if (
            len(series) == 0
            or not np.isfinite(series).all()
        ):
            continue

        max_ratio = (
            float(series.max())
            / latest_scale
        )

        final_ratio = (
            float(series[-1])
            / latest_scale
        )

        step_changes = np.diff(
            np.concatenate(
                [[latest_actual], series]
            )
        )

        jump_ratio = (
            float(
                np.abs(step_changes).max()
            )
            / latest_scale
        )

        penalty = (
            1.0
            + max(
                0.0,
                final_ratio - 2.5,
            )
            + 0.5
            * max(
                0.0,
                jump_ratio - 1.0,
            )
            + 0.25
            * max(
                0.0,
                max_ratio - 3.0,
            )
        )

        if "Selection Score" in ranking_frame.columns:
            base_series = ranking_frame.loc[
                ranking_frame["Model"] == model_name,
                "Selection Score",
            ]
        elif "MAE" in ranking_frame.columns:
            base_series = ranking_frame.loc[
                ranking_frame["Model"] == model_name,
                "MAE",
            ]
        else:
            base_series = pd.Series(dtype=float)

        base_value = (
            float(base_series.iloc[0])
            if len(base_series) > 0
            else float("inf")
        )

        scored_candidates.append(
            (
                base_value * penalty,
                base_value,
                model_name,
            )
        )

    if not scored_candidates:
        return preferred_model

    scored_candidates.sort()

    return str(
        scored_candidates[0][2]
    )


def build_multi_step_forecast(
    history: pd.DataFrame,
    bundle: ModelBundle,
    horizon: int,
) -> pd.DataFrame:
    """Generate recursive multi-week forecasts for every model.

    Each model maintains its own rolling history. Learned-model
    forecasts are blended with recent, seasonal, and trend anchors,
    then constrained to reduce unrealistic jumps.
    """
    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1"
        )

    model_names = list(bundle.keys())

    base = (
        history[
            [
                "date",
                "notifications",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    model_series = {
        name: pd.DataFrame(base.copy())
        for name in model_names
    }

    rows: list[
        dict[
            str,
            float | int | pd.Timestamp,
        ]
    ] = []

    recent_diffs = (
        history["notifications"]
        .diff()
        .dropna()
        .tail(
            min(
                26,
                max(
                    1,
                    len(history) - 1,
                ),
            )
        )
    )

    if len(recent_diffs) > 0:
        typical_step_delta = float(
            np.quantile(
                np.abs(recent_diffs),
                0.9,
            )
        )
    else:
        typical_step_delta = 25.0

    typical_step_delta = max(
        typical_step_delta,
        20.0,
    )

    seasonal_profile = (
        history.groupby(
            history["date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )["notifications"]
        .median()
        .to_dict()
    )

    historical_max = float(
        history["notifications"].max()
    )

    recent_peak = (
        float(
            history["notifications"]
            .tail(26)
            .max()
        )
        if len(history) > 0
        else historical_max
    )

    latest_actual = float(
        history["notifications"].iloc[-1]
    )

    ceiling = max(
        historical_max * 0.9,
        recent_peak * 1.4,
        latest_actual * 3.0,
        75.0,
    )

    for step in range(
        1,
        horizon + 1,
    ):
        next_date = (
            history["date"].max()
            + pd.Timedelta(weeks=step)
        )

        week_number = int(
            next_date.isocalendar().week
        )

        row: dict[
            str,
            float | int | pd.Timestamp,
        ] = {
            "date": next_date,
            "step": step,
        }

        for model_name in model_names:
            series = model_series[
                model_name
            ]

            lag1 = float(
                series[
                    "notifications"
                ].iloc[-1]
            )

            lag4 = (
                float(
                    series[
                        "notifications"
                    ].iloc[-4]
                )
                if len(series) >= 4
                else lag1
            )

            recent_level = float(
                series[
                    "notifications"
                ]
                .tail(8)
                .median()
            )

            trend_per_week = (
                _estimate_recent_trend(
                    series[
                        "notifications"
                    ]
                )
            )

            lag52_date = (
                next_date
                - pd.Timedelta(weeks=52)
            )

            lag52_series = series.loc[
                series["date"]
                == lag52_date,
                "notifications",
            ]

            lag52 = (
                float(
                    lag52_series.iloc[0]
                )
                if len(lag52_series) > 0
                else lag1
            )

            seasonal_median = float(
                seasonal_profile.get(
                    week_number,
                    lag52,
                )
            )

            feature_row = pd.DataFrame(
                {
                    "date": [next_date],
                    "lag1": [lag1],
                    "lag4": [lag4],
                    "lag52": [lag52],
                    "week_sin": [
                        np.sin(
                            2
                            * np.pi
                            * week_number
                            / 52
                        )
                    ],
                    "week_cos": [
                        np.cos(
                            2
                            * np.pi
                            * week_number
                            / 52
                        )
                    ],
                }
            )

            raw_predictions = (
                predict_with_bundle(
                    bundle,
                    feature_row,
                )
            )

            raw_value = max(
                0.0,
                float(
                    raw_predictions[
                        model_name
                    ][0]
                ),
            )

            recent_anchor = (
                0.55 * lag1
                + 0.20 * lag4
                + 0.25 * recent_level
            )

            seasonal_reference = (
                0.5 * lag52
                + 0.5 * seasonal_median
            )

            seasonal_cap = max(
                recent_anchor * 1.45,
                recent_anchor
                + 1.75
                * typical_step_delta,
            )

            seasonal_anchor = float(
                min(
                    seasonal_reference,
                    seasonal_cap,
                )
            )

            trend_target = max(
                0.0,
                recent_anchor
                + trend_per_week
                * min(
                    step,
                    4,
                ),
            )

            anchor_value = (
                0.65 * recent_anchor
                + 0.20 * seasonal_anchor
                + 0.15 * trend_target
            )

            delta_limit = max(
                typical_step_delta
                * (
                    0.65
                    + 0.20
                    * math.sqrt(step)
                ),
                25.0,
            )

            if model_name == "Naive":
                forecast_value = lag1

            elif model_name == "Seasonal Naive":
                forecast_value = (
                    seasonal_anchor
                )

            else:
                forecast_value = (
                    _stabilize_forecast(
                        raw_value=raw_value,
                        last_value=lag1,
                        anchor_value=anchor_value,
                        max_step_delta=delta_limit,
                        blend_weight=min(
                            0.92,
                            0.55
                            + 0.08
                            * max(
                                0,
                                step - 1,
                            ),
                        ),
                        ceiling=ceiling,
                    )
                )

            row[model_name] = (
                forecast_value
            )

            model_series[
                model_name
            ] = pd.concat(
                [
                    series,
                    pd.DataFrame(
                        {
                            "date": [
                                next_date
                            ],
                            "notifications": [
                                forecast_value
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )

        rows.append(row)

    return pd.DataFrame(rows)
