"""Historical multi-horizon evaluation for dengue forecasts.

This module evaluates the same recursive forecasting procedure used by
the deployed application at multiple forecast horizons.

It is intentionally separate from the live Streamlit pipeline because
multi-origin model refitting is computationally expensive.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from src.config import MIN_BACKTEST_TRAIN
from src.features import create_features
from src.forecasting import build_multi_step_forecast
from src.models import fit_model_bundle


DEFAULT_HORIZONS = (
    1,
    2,
    4,
    8,
    12,
)


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
        week or a larger value for faster exploratory evaluation.

    Returns
    -------
    pd.DataFrame
        One row per forecast origin, horizon, and model.
    """
    horizons = tuple(
        sorted(
            {
                int(horizon)
                for horizon in horizons
            }
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

    max_horizon = max(horizons)

    if len(history) < (
        min_history_weeks
        + max_horizon
    ):
        raise ValueError(
            "Not enough historical observations for the requested "
            "minimum history and forecast horizons."
        )

    rows: list[dict[str, object]] = []

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

        train_features = create_features(
            train_history
        )

        if len(train_features) < MIN_BACKTEST_TRAIN:
            continue

        bundle = fit_model_bundle(
            train_features,
            include_nb=include_nb,
            verbose=False,
        )

        forecast = build_multi_step_forecast(
            history=train_history,
            bundle=bundle,
            horizon=max_horizon,
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
                if model_name not in forecast.columns:
                    continue

                prediction = float(
                    forecast_row[
                        model_name
                    ]
                )

                error = (
                    prediction
                    - actual
                )

                rows.append(
                    {
                        "origin_date": origin_date,
                        "target_date": target_date,
                        "horizon": horizon,
                        "model": model_name,
                        "actual": actual,
                        "prediction": prediction,
                        "error": error,
                        "absolute_error": abs(
                            error
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
        actual = (
            sample[
                "actual"
            ]
            .to_numpy(
                dtype=float
            )
        )

        prediction = (
            sample[
                "prediction"
            ]
            .to_numpy(
                dtype=float
            )
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

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "Horizon",
                "MAE",
                "RMSE",
            ]
        )
        .reset_index(drop=True)
    )


def build_horizon_policy(
    results: pd.DataFrame,
    holdout_weeks: int = 52,
    selection_weeks: int = 52,
    min_skill_vs_naive: float = 0.10,
) -> pd.DataFrame:
    """Select a robust production model separately for each horizon.

    The model-selection policy is locked before the holdout begins.

    Selection uses only forecast targets whose outcomes were known
    before the first holdout forecast was issued.

    Naive persistence is the default champion. A learned model replaces
    it only when it:

    1. improves selection MAE by at least ``min_skill_vs_naive``;
    2. beats Naive during the first half of the selection period; and
    3. beats Naive during the second half of the selection period.

    Holdout forecasts are issued strictly after selection has ended.
    """
    if results.empty:
        raise ValueError(
            "Multi-horizon results are empty."
        )

    if holdout_weeks < 1:
        raise ValueError(
            "holdout_weeks must be at least 1."
        )

    if selection_weeks < 2:
        raise ValueError(
            "selection_weeks must be at least 2."
        )

    if min_skill_vs_naive < 0:
        raise ValueError(
            "min_skill_vs_naive cannot be negative."
        )

    data = results.copy()

    data["origin_date"] = pd.to_datetime(
        data["origin_date"]
    )

    data["target_date"] = pd.to_datetime(
        data["target_date"]
    )

    horizons = sorted(
        data[
            "horizon"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    # --------------------------------------------------------
    # Find forecast origins shared across every horizon.
    # --------------------------------------------------------

    origin_sets = []

    for horizon in horizons:
        horizon_origins = set(
            data.loc[
                data["horizon"]
                == horizon,
                "origin_date",
            ]
            .drop_duplicates()
            .tolist()
        )

        origin_sets.append(
            horizon_origins
        )

    common_origins = sorted(
        set.intersection(
            *origin_sets
        )
    )

    if len(common_origins) <= holdout_weeks:
        raise ValueError(
            "Not enough common forecast origins to create "
            "the requested holdout."
        )

    holdout_origins = (
        common_origins[
            -holdout_weeks:
        ]
    )

    holdout_origin_start = pd.Timestamp(
        holdout_origins[0]
    )

    holdout_origin_end = pd.Timestamp(
        holdout_origins[-1]
    )

    # --------------------------------------------------------
    # Selection cutoff.
    #
    # When the first holdout forecast is issued, only targets
    # strictly before that date are considered known.
    # --------------------------------------------------------

    selection_cutoff = (
        holdout_origin_start
    )

    eligible_selection = (
        data.loc[
            data["target_date"]
            < selection_cutoff
        ]
        .copy()
    )

    # Use target weeks shared by every forecast horizon.
    target_sets = []

    for horizon in horizons:
        horizon_targets = set(
            eligible_selection.loc[
                eligible_selection[
                    "horizon"
                ]
                == horizon,
                "target_date",
            ]
            .drop_duplicates()
            .tolist()
        )

        target_sets.append(
            horizon_targets
        )

    common_selection_targets = sorted(
        set.intersection(
            *target_sets
        )
    )

    if len(common_selection_targets) < 2:
        raise ValueError(
            "Not enough pre-holdout target dates "
            "for model selection."
        )

    effective_selection_weeks = min(
        selection_weeks,
        len(
            common_selection_targets
        ),
    )

    selection_dates = (
        common_selection_targets[
            -effective_selection_weeks:
        ]
    )

    selection_start = pd.Timestamp(
        selection_dates[0]
    )

    selection_end = pd.Timestamp(
        selection_dates[-1]
    )

    # --------------------------------------------------------
    # Split selection period into two temporal subperiods.
    # --------------------------------------------------------

    midpoint = (
        len(selection_dates)
        // 2
    )

    early_dates = (
        selection_dates[
            :midpoint
        ]
    )

    late_dates = (
        selection_dates[
            midpoint:
        ]
    )

    learned_models = {
        "Linear Regression",
        "Random Forest",
        "Negative Binomial",
    }

    rows: list[
        dict[str, object]
    ] = []

    # --------------------------------------------------------
    # Select one model independently for each horizon.
    # --------------------------------------------------------

    for horizon in horizons:
        horizon_data = (
            data.loc[
                data["horizon"]
                == horizon
            ]
            .copy()
        )

        selection = (
            horizon_data.loc[
                horizon_data[
                    "target_date"
                ].isin(
                    selection_dates
                )
            ]
            .copy()
        )

        holdout = (
            horizon_data.loc[
                horizon_data[
                    "origin_date"
                ].isin(
                    holdout_origins
                )
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Champion baseline: Naive.
        # ----------------------------------------------------

        naive_selection = (
            selection.loc[
                selection[
                    "model"
                ]
                == "Naive"
            ]
            .copy()
        )

        if naive_selection.empty:
            raise RuntimeError(
                f"No Naive selection results for horizon {horizon}."
            )

        naive_selection_mae = float(
            mean_absolute_error(
                naive_selection[
                    "actual"
                ],
                naive_selection[
                    "prediction"
                ],
            )
        )

        # ----------------------------------------------------
        # Evaluate learned challengers.
        # ----------------------------------------------------

        candidate_rows: list[
            dict[str, object]
        ] = []

        for model_name in sorted(
            learned_models
        ):
            candidate = (
                selection.loc[
                    selection[
                        "model"
                    ]
                    == model_name
                ]
                .copy()
            )

            if candidate.empty:
                continue

            candidate_mae = float(
                mean_absolute_error(
                    candidate[
                        "actual"
                    ],
                    candidate[
                        "prediction"
                    ],
                )
            )

            overall_skill = (
                1.0
                - candidate_mae
                / max(
                    naive_selection_mae,
                    1e-9,
                )
            )

            # Early selection subperiod.
            early_candidate = (
                candidate.loc[
                    candidate[
                        "target_date"
                    ].isin(
                        early_dates
                    )
                ]
            )

            early_naive = (
                naive_selection.loc[
                    naive_selection[
                        "target_date"
                    ].isin(
                        early_dates
                    )
                ]
            )

            if (
                len(early_candidate) > 0
                and len(early_naive) > 0
            ):
                early_candidate_mae = float(
                    mean_absolute_error(
                        early_candidate[
                            "actual"
                        ],
                        early_candidate[
                            "prediction"
                        ],
                    )
                )

                early_naive_mae = float(
                    mean_absolute_error(
                        early_naive[
                            "actual"
                        ],
                        early_naive[
                            "prediction"
                        ],
                    )
                )

                early_skill = (
                    1.0
                    - early_candidate_mae
                    / max(
                        early_naive_mae,
                        1e-9,
                    )
                )

            else:
                early_skill = np.nan

            # Late selection subperiod.
            late_candidate = (
                candidate.loc[
                    candidate[
                        "target_date"
                    ].isin(
                        late_dates
                    )
                ]
            )

            late_naive = (
                naive_selection.loc[
                    naive_selection[
                        "target_date"
                    ].isin(
                        late_dates
                    )
                ]
            )

            if (
                len(late_candidate) > 0
                and len(late_naive) > 0
            ):
                late_candidate_mae = float(
                    mean_absolute_error(
                        late_candidate[
                            "actual"
                        ],
                        late_candidate[
                            "prediction"
                        ],
                    )
                )

                late_naive_mae = float(
                    mean_absolute_error(
                        late_naive[
                            "actual"
                        ],
                        late_naive[
                            "prediction"
                        ],
                    )
                )

                late_skill = (
                    1.0
                    - late_candidate_mae
                    / max(
                        late_naive_mae,
                        1e-9,
                    )
                )

            else:
                late_skill = np.nan

            qualifies = (
                np.isfinite(overall_skill)
                and np.isfinite(early_skill)
                and np.isfinite(late_skill)
                and overall_skill >= min_skill_vs_naive
                and early_skill >= min_skill_vs_naive
                and late_skill >= min_skill_vs_naive
            )

            candidate_rows.append(
                {
                    "Model": model_name,
                    "MAE": candidate_mae,
                    "Overall Skill": (
                        overall_skill
                    ),
                    "Early Skill": (
                        early_skill
                    ),
                    "Late Skill": (
                        late_skill
                    ),
                    "Qualifies": (
                        qualifies
                    ),
                }
            )

        candidate_table = pd.DataFrame(
            candidate_rows
        )

        if not candidate_table.empty:
            candidate_table = (
                candidate_table
                .sort_values(
                    [
                        "MAE",
                        "Model",
                    ]
                )
                .reset_index(drop=True)
            )

            best_candidate = (
                candidate_table.iloc[0]
            )

            best_candidate_name = str(
                best_candidate[
                    "Model"
                ]
            )

            best_candidate_skill = float(
                best_candidate[
                    "Overall Skill"
                ]
            )

            best_candidate_early_skill = float(
                best_candidate[
                    "Early Skill"
                ]
            )

            best_candidate_late_skill = float(
                best_candidate[
                    "Late Skill"
                ]
            )

            qualifying = (
                candidate_table.loc[
                    candidate_table[
                        "Qualifies"
                    ]
                ]
                .copy()
            )

        else:
            best_candidate_name = "None"
            best_candidate_skill = np.nan
            best_candidate_early_skill = np.nan
            best_candidate_late_skill = np.nan

            qualifying = pd.DataFrame()

        # ----------------------------------------------------
        # Champion / challenger decision.
        # ----------------------------------------------------

        if not qualifying.empty:
            winner = (
                qualifying
                .sort_values(
                    [
                        "MAE",
                        "Model",
                    ]
                )
                .iloc[0]
            )

            selected_model = str(
                winner[
                    "Model"
                ]
            )

            decision = (
                "Learned model promoted"
            )

        else:
            selected_model = (
                "Naive"
            )

            decision = (
                "Naive retained"
            )

        selected_selection = (
            selection.loc[
                selection[
                    "model"
                ]
                == selected_model
            ]
            .copy()
        )

        selection_mae = float(
            mean_absolute_error(
                selected_selection[
                    "actual"
                ],
                selected_selection[
                    "prediction"
                ],
            )
        )

        selection_rmse = float(
            np.sqrt(
                mean_squared_error(
                    selected_selection[
                        "actual"
                    ],
                    selected_selection[
                        "prediction"
                    ],
                )
            )
        )

        selection_skill = (
            1.0
            - selection_mae
            / max(
                naive_selection_mae,
                1e-9,
            )
        )

        # ----------------------------------------------------
        # Untouched holdout.
        # ----------------------------------------------------

        selected_holdout = (
            holdout.loc[
                holdout[
                    "model"
                ]
                == selected_model
            ]
            .copy()
        )

        naive_holdout = (
            holdout.loc[
                holdout[
                    "model"
                ]
                == "Naive"
            ]
            .copy()
        )

        if (
            selected_holdout.empty
            or naive_holdout.empty
        ):
            continue

        actual = (
            selected_holdout[
                "actual"
            ]
            .to_numpy(
                dtype=float
            )
        )

        prediction = (
            selected_holdout[
                "prediction"
            ]
            .to_numpy(
                dtype=float
            )
        )

        errors = (
            prediction
            - actual
        )

        holdout_mae = float(
            mean_absolute_error(
                actual,
                prediction,
            )
        )

        holdout_rmse = float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    prediction,
                )
            )
        )

        holdout_bias = float(
            errors.mean()
        )

        naive_holdout_mae = float(
            mean_absolute_error(
                naive_holdout[
                    "actual"
                ],
                naive_holdout[
                    "prediction"
                ],
            )
        )

        holdout_skill = (
            1.0
            - holdout_mae
            / max(
                naive_holdout_mae,
                1e-9,
            )
        )

        rows.append(
            {
                "Horizon": (
                    int(horizon)
                ),
                "Selected Model": (
                    selected_model
                ),
                "Decision": (
                    decision
                ),
                "Best Learned Candidate": (
                    best_candidate_name
                ),
                "Candidate Skill vs Naive": (
                    best_candidate_skill
                ),
                "Candidate Early Skill": (
                    best_candidate_early_skill
                ),
                "Candidate Late Skill": (
                    best_candidate_late_skill
                ),
                "Selection N": int(
                    len(
                        selected_selection
                    )
                ),
                "Selection MAE": (
                    selection_mae
                ),
                "Selection RMSE": (
                    selection_rmse
                ),
                "Naive Selection MAE": (
                    naive_selection_mae
                ),
                "Selection Skill vs Naive": (
                    selection_skill
                ),
                "Holdout N": int(
                    len(
                        selected_holdout
                    )
                ),
                "Holdout MAE": (
                    holdout_mae
                ),
                "Holdout RMSE": (
                    holdout_rmse
                ),
                "Holdout Bias": (
                    holdout_bias
                ),
                "Naive Holdout MAE": (
                    naive_holdout_mae
                ),
                "Holdout Skill vs Naive": (
                    holdout_skill
                ),
                "Selection Start": (
                    selection_start
                ),
                "Selection End": (
                    selection_end
                ),
                "Policy Cutoff": (
                    selection_cutoff
                ),
                "Holdout Origin Start": (
                    holdout_origin_start
                ),
                "Holdout Origin End": (
                    holdout_origin_end
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Horizon"
        )
        .reset_index(drop=True)
    )


def build_horizon_holdout_comparison(
    results: pd.DataFrame,
    policy: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate every model on the untouched holdout origins.

    This is diagnostic only.

    Holdout performance must not be used to retroactively change the
    model policy that was selected before the holdout began.
    """
    if results.empty:
        raise ValueError(
            "Multi-horizon results are empty."
        )

    if policy.empty:
        raise ValueError(
            "Horizon policy is empty."
        )

    data = results.copy()

    data["origin_date"] = pd.to_datetime(
        data["origin_date"]
    )

    rows: list[
        dict[str, object]
    ] = []

    for _, policy_row in policy.iterrows():
        horizon = int(
            policy_row[
                "Horizon"
            ]
        )

        selected_model = str(
            policy_row[
                "Selected Model"
            ]
        )

        holdout_start = pd.Timestamp(
            policy_row[
                "Holdout Origin Start"
            ]
        )

        holdout_end = pd.Timestamp(
            policy_row[
                "Holdout Origin End"
            ]
        )

        horizon_holdout = (
            data.loc[
                (
                    data[
                        "horizon"
                    ]
                    == horizon
                )
                & (
                    data[
                        "origin_date"
                    ]
                    >= holdout_start
                )
                & (
                    data[
                        "origin_date"
                    ]
                    <= holdout_end
                )
            ]
            .copy()
        )

        naive = (
            horizon_holdout.loc[
                horizon_holdout[
                    "model"
                ]
                == "Naive"
            ]
            .copy()
        )

        if naive.empty:
            continue

        naive_mae = float(
            mean_absolute_error(
                naive[
                    "actual"
                ],
                naive[
                    "prediction"
                ],
            )
        )

        for (
            model_name,
            sample,
        ) in horizon_holdout.groupby(
            "model"
        ):
            actual = (
                sample[
                    "actual"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            prediction = (
                sample[
                    "prediction"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            errors = (
                prediction
                - actual
            )

            mae = float(
                mean_absolute_error(
                    actual,
                    prediction,
                )
            )

            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        prediction,
                    )
                )
            )

            skill = (
                1.0
                - mae
                / max(
                    naive_mae,
                    1e-9,
                )
            )

            rows.append(
                {
                    "Horizon": (
                        horizon
                    ),
                    "Model": str(
                        model_name
                    ),
                    "Selected": (
                        str(
                            model_name
                        )
                        == selected_model
                    ),
                    "N": int(
                        len(sample)
                    ),
                    "MAE": (
                        mae
                    ),
                    "RMSE": (
                        rmse
                    ),
                    "Bias": float(
                        errors.mean()
                    ),
                    "Skill vs Naive": (
                        skill
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "Horizon",
                "MAE",
            ]
        )
        .reset_index(drop=True)
    )
