"""Historical multi-horizon evaluation for dengue forecasts.

This module evaluates the same recursive forecasting procedure used by
the deployed application at multiple forecast horizons.

It is intentionally separate from the live Streamlit pipeline because
multi-origin model refitting is computationally expensive.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import MIN_BACKTEST_TRAIN
from src.features import create_features
from src.forecasting import build_multi_step_forecast
from src.models import fit_model_bundle


DEFAULT_HORIZONS = (1, 2, 4, 8, 12)


def run_multi_horizon_backtest(
    history_df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    include_nb: bool = False,
    min_history_weeks: int = 60,
    origin_step: int = 1,
) -> pd.DataFrame:
    """Evaluate recursive forecasts from multiple historical origins.

    At each forecast origin:

    1. only observations available up to that date are retained;
    2. features are rebuilt using that truncated history;
    3. models are refitted;
    4. recursive forecasts are generated;
    5. predictions are compared with observations that occurred later.

    Parameters
    ----------
    history_df:
        Complete weekly series containing ``date`` and
        ``notifications``.
    horizons:
        Forecast horizons, in weeks, to evaluate.
    include_nb:
        Whether to include Negative Binomial regression.
    min_history_weeks:
        Minimum number of observed weeks required before the first
        forecast origin.
    origin_step:
        Distance between successive forecast origins. Use 1 for every
        week, or a larger value for faster exploratory evaluation.

    Returns
    -------
    pd.DataFrame
        Long-form table containing one row per origin, horizon, and
        model.
    """
    horizons = tuple(
        sorted(
            set(
                int(h)
                for h in horizons
            )
        )
    )

    if not horizons:
        raise ValueError(
            "At least one forecast horizon is required."
        )

    if min(horizons) < 1:
        raise ValueError(
            "Forecast horizons must be at least 1."
        )

    if origin_step < 1:
        raise ValueError(
            "origin_step must be at least 1."
        )

    history = (
        history_df[
            [
                "date",
                "notifications",
            ]
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    history["date"] = pd.to_datetime(
        history["date"]
    )

    if history["date"].duplicated().any():
        raise ValueError(
            "history_df contains duplicate weekly dates."
        )

    max_horizon = max(
        horizons
    )

    if len(history) < (
        min_history_weeks
        + max_horizon
    ):
        raise ValueError(
            "Not enough historical observations for the requested "
            "minimum history and forecast horizons."
        )

    rows: list[dict[str, object]] = []

    # origin_idx refers to the final observation available when a
    # historical forecast would have been issued.
    first_origin = (
        min_history_weeks - 1
    )

    last_origin_exclusive = (
        len(history)
        - max_horizon
    )

    for origin_idx in range(
        first_origin,
        last_origin_exclusive,
        origin_step,
    ):
        train_history = (
            history.iloc[
                : origin_idx + 1
            ]
            .copy()
            .reset_index(drop=True)
        )

        train_features = (
            create_features(
                train_history
            )
        )

        if len(train_features) < MIN_BACKTEST_TRAIN:
            continue

        bundle = fit_model_bundle(
            train_features,
            include_nb=include_nb,
            verbose=False,
        )

        forecast = (
            build_multi_step_forecast(
                history=train_history,
                bundle=bundle,
                horizon=max_horizon,
            )
        )

        origin_date = pd.Timestamp(
            train_history[
                "date"
            ].iloc[-1]
        )

        for horizon in horizons:
            target_idx = (
                origin_idx
                + horizon
            )

            target_date = pd.Timestamp(
                history.iloc[
                    target_idx
                ]["date"]
            )

            actual = float(
                history.iloc[
                    target_idx
                ]["notifications"]
            )

            forecast_row = (
                forecast.loc[
                    forecast["step"]
                    == horizon
                ]
                .iloc[0]
            )

            for model_name in bundle:
                if model_name not in (
                    forecast.columns
                ):
                    continue

                prediction = float(
                    forecast_row[
                        model_name
                    ]
                )

                rows.append(
                    {
                        "origin_date": (
                            origin_date
                        ),
                        "target_date": (
                            target_date
                        ),
                        "horizon": (
                            horizon
                        ),
                        "model": (
                            model_name
                        ),
                        "actual": (
                            actual
                        ),
                        "prediction": (
                            prediction
                        ),
                        "error": (
                            prediction
                            - actual
                        ),
                        "absolute_error": abs(
                            prediction
                            - actual
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "origin_date",
                "target_date",
                "horizon",
                "model",
                "actual",
                "prediction",
                "error",
                "absolute_error",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "origin_date",
                "horizon",
                "model",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_multi_horizon_backtest(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize forecast accuracy separately for each horizon."""
    if results.empty:
        return pd.DataFrame(
            columns=[
                "Horizon",
                "Model",
                "N",
                "MAE",
                "RMSE",
                "Bias",
            ]
        )

    rows: list[
        dict[str, object]
    ] = []

    for (
        horizon,
        model_name,
    ), sample in results.groupby(
        [
            "horizon",
            "model",
        ],
        sort=True,
    ):
        actual = sample[
            "actual"
        ].to_numpy(
            dtype=float
        )

        prediction = sample[
            "prediction"
        ].to_numpy(
            dtype=float
        )

        error = (
            prediction
            - actual
        )

        rows.append(
            {
                "Horizon": int(
                    horizon
                ),
                "Model": str(
                    model_name
                ),
                "N": int(
                    len(sample)
                ),
                "MAE": float(
                    mean_absolute_error(
                        actual,
                        prediction,
                    )
                ),
                "RMSE": float(
                    np.sqrt(
                        mean_squared_error(
                            actual,
                            prediction,
                        )
                    )
                ),
                "Bias": float(
                    error.mean()
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return (
        summary
        .sort_values(
            [
                "Horizon",
                "MAE",
                "RMSE",
            ]
        )
        .reset_index(drop=True)
    )
