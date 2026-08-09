"""Model evaluation and selection utilities for dengue forecasting."""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    DEFAULT_HOLDOUT_WEEKS,
    MIN_BACKTEST_TRAIN,
)
from src.models import fit_model_bundle, predict_with_bundle


MetricValue = Union[str, float]


def score_predictions(
    actual: np.ndarray,
    pred: np.ndarray,
    previous_week: np.ndarray,
) -> dict[str, float]:
    """Evaluate forecast accuracy and week-over-week change detection.

    In addition to standard regression errors, this function measures
    whether predictions correctly identify rising dengue activity.

    Parameters
    ----------
    actual:
        Observed notification counts.
    pred:
        Predicted notification counts.
    previous_week:
        Notification count from the week immediately before each
        prediction.

    Returns
    -------
    dict[str, float]
        Regression and direction-detection metrics.
    """
    actual = np.asarray(actual)
    pred = np.asarray(pred)
    previous_week = np.asarray(previous_week)

    actual_direction = np.sign(
        actual - previous_week
    )

    predicted_direction = np.sign(
        pred - previous_week
    )

    # Binary surveillance signal:
    # did notifications rise relative to the previous week?
    actual_rising = actual > previous_week
    predicted_rising = pred > previous_week

    true_positives = (
        actual_rising & predicted_rising
    ).sum()

    rising_recall = (
        true_positives
        / max(1, actual_rising.sum())
    )

    rising_precision = (
        true_positives
        / max(1, predicted_rising.sum())
    )

    rising_f1 = (
        2
        * rising_precision
        * rising_recall
        / max(
            1e-9,
            rising_precision + rising_recall,
        )
    )

    # False alarm:
    # the model predicts a rise when the observed count
    # does not actually rise.
    actual_not_rising = ~actual_rising

    false_alarms = (
        actual_not_rising & predicted_rising
    ).sum()

    false_alarm_rate = (
        false_alarms
        / max(1, actual_not_rising.sum())
    )

    return {
        "MAE": float(
            mean_absolute_error(actual, pred)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(actual, pred)
            )
        ),
        "Direction Acc.": float(
            (
                actual_direction
                == predicted_direction
            ).mean()
        ),
        "Rising Recall": float(rising_recall),
        "Rising Precision": float(
            rising_precision
        ),
        "Rising F1": float(rising_f1),
        "False Alarm Rate": float(
            false_alarm_rate
        ),
    }


def run_backtest(
    features_df: pd.DataFrame,
    include_nb: bool = True,
    min_backtest_train: int = MIN_BACKTEST_TRAIN,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run an expanding-window walk-forward backtest.

    Each iteration trains only on observations that would have been
    available at that point in time and predicts the next week.

    Parameters
    ----------
    features_df:
        Feature-engineered weekly dengue data.
    include_nb:
        Whether Negative Binomial regression should be attempted.
    min_backtest_train:
        Minimum number of historical weeks required before forecasting
        begins.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        Backtest predictions and a mapping between model names and
        prediction columns.
    """
    model_cols = {
        "Naive": "naive_pred",
        "Seasonal Naive": "seasonal_naive_pred",
        "Linear Regression": "linear_pred",
        "Random Forest": "rf_pred",
    }

    backtest_frames: list[pd.DataFrame] = []

    # For every prediction, train only on rows occurring before it.
    for current_idx in range(
        min_backtest_train,
        len(features_df),
    ):
        fold_train = pd.DataFrame(
            features_df.iloc[:current_idx].copy()
        )

        fold_test = pd.DataFrame(
            features_df.iloc[[current_idx]].copy()
        )

        fold_bundle = fit_model_bundle(
            fold_train,
            include_nb=include_nb,
            verbose=False,
        )

        fold_predictions = predict_with_bundle(
            fold_bundle,
            fold_test,
        )

        fold_frame = pd.DataFrame(
            {
                "date": fold_test["date"].values,
                "actual": (
                    fold_test["notifications"]
                    .values
                    .astype(int)
                ),
                "previous_week": (
                    fold_test["lag1"].values
                ),
                "naive_pred": (
                    fold_predictions["Naive"]
                ),
                "seasonal_naive_pred": (
                    fold_predictions[
                        "Seasonal Naive"
                    ]
                ),
                "linear_pred": (
                    fold_predictions[
                        "Linear Regression"
                    ]
                ),
                "rf_pred": (
                    fold_predictions[
                        "Random Forest"
                    ]
                ),
            }
        )

        # Negative Binomial is optional because it may fail
        # to converge on individual training folds.
        if "Negative Binomial" in fold_predictions:
            fold_frame["nb_pred"] = (
                fold_predictions[
                    "Negative Binomial"
                ]
            )

        backtest_frames.append(fold_frame)

    backtest_results = pd.concat(
        backtest_frames,
        ignore_index=True,
    )

    if "nb_pred" in backtest_results.columns:
        model_cols["Negative Binomial"] = (
            "nb_pred"
        )

    return backtest_results, model_cols


def build_error_table(
    bt_slice: pd.DataFrame,
    model_cols: dict[str, str],
) -> pd.DataFrame:
    """Compute MAE and RMSE for each model.

    Models are ranked primarily by MAE, with RMSE used as the
    tiebreaker.
    """
    rows: list[dict[str, MetricValue]] = []

    for model_name, column in model_cols.items():
        sample = bt_slice.dropna(
            subset=[column]
        )

        if len(sample) == 0:
            continue

        rows.append(
            {
                "Model": model_name,
                "MAE": float(
                    mean_absolute_error(
                        sample["actual"],
                        sample[column],
                    )
                ),
                "RMSE": float(
                    np.sqrt(
                        mean_squared_error(
                            sample["actual"],
                            sample[column],
                        )
                    )
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Model",
                "MAE",
                "RMSE",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by=["MAE", "RMSE"]
        )
        .reset_index(drop=True)
    )


def determine_holdout_start(
    backtest_results: pd.DataFrame,
    holdout_weeks: int = DEFAULT_HOLDOUT_WEEKS,
    minimum_holdout_weeks: int = 26,
) -> pd.Timestamp:
    """Determine the starting date of the final evaluation holdout.

    The default holdout contains the most recent 52 weeks, while a
    minimum of 26 weeks is enforced when possible.
    """
    unique_dates = sorted(
        pd.unique(backtest_results["date"])
    )

    if len(unique_dates) == 0:
        raise ValueError(
            "Backtest results are empty; "
            "cannot determine holdout start."
        )

    effective_holdout_weeks = min(
        len(unique_dates),
        max(
            minimum_holdout_weeks,
            holdout_weeks,
        ),
    )

    return pd.Timestamp(
        unique_dates[-effective_holdout_weeks]
    )


def build_predictor_diagnostics(
    backtest_results: pd.DataFrame,
    model_cols: dict[str, str],
    prod_name: str,
    holdout_start: pd.Timestamp,
) -> pd.DataFrame:
    """Build diagnostics for the selected production model."""
    prod_col = model_cols[prod_name]

    prod_backtest = (
        backtest_results
        .dropna(subset=[prod_col])
        .copy()
        .reset_index(drop=True)
    )

    latest_eval = prod_backtest.iloc[-1]

    latest_abs_error = abs(
        float(latest_eval["actual"])
        - float(latest_eval[prod_col])
    )

    # Use only final holdout observations for the recent
    # out-of-sample MAE.
    recent_holdout = prod_backtest[
        prod_backtest["date"] >= holdout_start
    ].tail(8)

    if len(recent_holdout) > 0:
        recent_holdout_mae = float(
            mean_absolute_error(
                recent_holdout["actual"],
                recent_holdout[prod_col],
            )
        )
    else:
        recent_holdout_mae = float("nan")

    return pd.DataFrame(
        {
            "Metric": [
                "Production model",
                "Testing window start",
                "Latest tested week",
                "Latest tested actual",
                "Latest tested prediction",
                "Latest tested absolute error",
                "Recent holdout MAE",
            ],
            "Value": [
                prod_name,
                str(
                    pd.Timestamp(
                        holdout_start
                    ).date()
                ),
                str(
                    pd.Timestamp(
                        latest_eval["date"]
                    ).date()
                ),
                f"{int(latest_eval['actual']):,}",
                f"{float(latest_eval[prod_col]):.1f}",
                f"{latest_abs_error:.1f}",
                (
                    f"{recent_holdout_mae:.1f}"
                    if np.isfinite(
                        recent_holdout_mae
                    )
                    else "N/A"
                ),
            ],
        }
    )


def build_selection_score_table(
    evaluation_bt: pd.DataFrame,
    model_cols: dict[str, str],
) -> pd.DataFrame:
    """Rank models using the project's composite selection score.

    Lower scores indicate better candidates for production.
    """
    rows: list[dict[str, MetricValue]] = []

    for model_name, column in model_cols.items():
        sample = (
            evaluation_bt
            .dropna(subset=[column])
            .copy()
        )

        if len(sample) == 0:
            continue

        score = score_predictions(
            sample["actual"].values,
            sample[column].values,
            sample["previous_week"].values,
        )

        recent_mae = float(
            mean_absolute_error(
                sample["actual"],
                sample[column],
            )
        )

        latest_abs_error = abs(
            float(sample.iloc[-1]["actual"])
            - float(sample.iloc[-1][column])
        )

        rows.append(
            {
                "Model": model_name,
                "Recent MAE": recent_mae,
                "Latest Abs Error": (
                    latest_abs_error
                ),
                "Direction Acc.": float(
                    score["Direction Acc."]
                ),
                "Rising F1": float(
                    score["Rising F1"]
                ),
                "False Alarm Rate": float(
                    score["False Alarm Rate"]
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Model",
                "Recent MAE",
                "Latest Abs Error",
                "Direction Acc.",
                "Rising F1",
                "False Alarm Rate",
                "Selection Score",
            ]
        )

    score_table = pd.DataFrame(rows)

    # Normalize count-based errors before combining them
    # with metrics that already lie between 0 and 1.
    mae_scale = max(
        float(
            score_table[
                "Recent MAE"
            ].median()
        ),
        1.0,
    )

    latest_scale = max(
        float(
            score_table[
                "Latest Abs Error"
            ].median()
        ),
        1.0,
    )

    score_table["Selection Score"] = (
        0.45
        * (
            score_table["Recent MAE"]
            / mae_scale
        )
        + 0.20
        * (
            score_table["Latest Abs Error"]
            / latest_scale
        )
        + 0.20
        * (
            1.0
            - score_table[
                "Direction Acc."
            ]
        )
        + 0.20
        * (
            1.0
            - score_table[
                "Rising F1"
            ]
        )
        + 0.15
        * score_table[
            "False Alarm Rate"
        ]
    )

    score_order = list(
        np.argsort(
            score_table[
                "Selection Score"
            ].to_numpy(dtype=float)
        )
    )

    return (
        pd.DataFrame(
            score_table.iloc[
                score_order
            ].copy()
        )
        .reset_index(drop=True)
    )
