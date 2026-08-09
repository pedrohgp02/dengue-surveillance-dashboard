"""End-to-end orchestration for the dengue surveillance pipeline."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import (
    BAND_QUANTILES,
    DATA_SOURCE_LABEL,
    DEFAULT_HOLDOUT_WEEKS,
    DEFAULT_SELECTION_WEEKS,
    TARGET_UF_CODE,
    TREND_THRESHOLD,
    UF_OPTIONS,
)
from src.data import (
    build_weekly_series,
    get_available_data_urls,
    get_cache_info,
)
from src.evaluation import (
    build_error_table,
    build_predictor_diagnostics,
    build_selection_score_table,
    determine_holdout_start,
    run_backtest,
)
from src.features import create_features
from src.forecasting import (
    build_multi_step_forecast,
    build_next_week_features,
    choose_multi_step_display_model,
)
from src.models import (
    fit_model_bundle,
    predict_with_bundle,
)


def build_monitoring_state(
    horizon: int = 1,
    include_nb: bool = True,
    refresh_data: bool = False,
    holdout_weeks: int = DEFAULT_HOLDOUT_WEEKS,
    selection_weeks: int = DEFAULT_SELECTION_WEEKS,
    uf_code: int = TARGET_UF_CODE,
) -> dict[str, Any]:
    """Build all data required by the surveillance dashboard.

    The pipeline performs the complete workflow:

    1. discover and load SINAN dengue data,
    2. construct the weekly surveillance series,
    3. engineer forecasting features,
    4. run expanding-window backtesting,
    5. separate model-selection and holdout periods,
    6. select a production model,
    7. fit models on all available training data,
    8. generate one- or multi-step forecasts,
    9. construct residual-based uncertainty bands,
    10. derive trend and risk indicators.

    Parameters
    ----------
    horizon:
        Number of future weeks to forecast.
    include_nb:
        Whether Negative Binomial regression should be attempted.
    refresh_data:
        If True, bypass the existing local cache and rebuild the
        surveillance series from the source datasets.
    holdout_weeks:
        Number of recent backtest weeks reserved for final evaluation.
    selection_weeks:
        Number of weeks immediately before the holdout used for
        production-model selection.
    uf_code:
        IBGE state code to analyze.

    Returns
    -------
    dict[str, Any]
        Complete dashboard state.
    """
    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1"
        )

    if selection_weeks < 1:
        raise ValueError(
            "selection_weeks must be at least 1"
        )

    if holdout_weeks < 1:
        raise ValueError(
            "holdout_weeks must be at least 1"
        )

    # -----------------------------------------------------------------
    # 1. Load surveillance data
    # -----------------------------------------------------------------

    available_data_urls = (
        get_available_data_urls(
            refresh=refresh_data
        )
    )

    available_years = sorted(
        available_data_urls
    )

    raw_all, full_df, df = (
        build_weekly_series(
            refresh=refresh_data,
            uf_code=uf_code,
        )
    )

    # -----------------------------------------------------------------
    # 2. Feature engineering
    # -----------------------------------------------------------------

    features_df = create_features(df)

    if len(features_df) == 0:
        raise RuntimeError(
            "Not enough weekly observations are available "
            "to construct forecasting features."
        )

    # -----------------------------------------------------------------
    # 3. Expanding-window backtest
    # -----------------------------------------------------------------

    backtest_results, model_cols = (
        run_backtest(
            features_df,
            include_nb=include_nb,
        )
    )

    if len(backtest_results) == 0:
        raise RuntimeError(
            "Backtesting produced no predictions."
        )

    # Track whether Negative Binomial was requested and whether
    # it remained available throughout the evaluation pipeline.
    nb_requested = include_nb
    nb_available = (
        "Negative Binomial"
        in model_cols
    )

    if (
        nb_requested
        and not nb_available
    ):
        nb_status = "omitted"

    elif (
        nb_requested
        and nb_available
    ):
        nb_status = "active"

    else:
        nb_status = "off"

    # -----------------------------------------------------------------
    # 4. Separate model-selection and holdout periods
    # -----------------------------------------------------------------

    full_err = build_error_table(
        backtest_results,
        model_cols,
    )

    holdout_start = determine_holdout_start(
        backtest_results,
        holdout_weeks=holdout_weeks,
    )

    selection_bt = (
        backtest_results.loc[
            backtest_results["date"]
            < holdout_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    holdout_bt = (
        backtest_results.loc[
            backtest_results["date"]
            >= holdout_start
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(holdout_bt) > 0:
        holdout_err = build_error_table(
            holdout_bt,
            model_cols,
        )

    else:
        holdout_err = full_err.copy()

    # The selection window is taken only from observations occurring
    # BEFORE the holdout period, preventing holdout leakage.
    if len(selection_bt) > 0:
        recent_selection_bt = (
            selection_bt
            .tail(selection_weeks)
            .copy()
            .reset_index(drop=True)
        )

    else:
        recent_selection_bt = (
            selection_bt.copy()
        )

    if len(recent_selection_bt) > 0:
        recent_err = build_error_table(
            recent_selection_bt,
            model_cols,
        )

        selection_score_table = (
            build_selection_score_table(
                recent_selection_bt,
                model_cols,
            )
        )

    else:
        recent_err = holdout_err.copy()

        selection_score_table = (
            pd.DataFrame()
        )

    # -----------------------------------------------------------------
    # 5. Select production model
    # -----------------------------------------------------------------

    if len(selection_score_table) > 0:
        ranking_frame = (
            selection_score_table[
                [
                    "Model",
                    "Selection Score",
                ]
            ]
            .copy()
        )

        ranking_basis = (
            "Pre-holdout selection score "
            f"({len(recent_selection_bt)} "
            "weeks before holdout)"
        )

    elif len(recent_err) > 0:
        ranking_frame = recent_err.copy()

        ranking_basis = (
            "Pre-holdout selection focus "
            f"({len(recent_selection_bt)} weeks)"
        )

    elif len(holdout_err) > 0:
        ranking_frame = (
            holdout_err.copy()
        )

        ranking_basis = (
            "Holdout evaluation "
            f"({len(holdout_bt)} weeks)"
        )

    else:
        ranking_frame = full_err.copy()

        ranking_basis = (
            "Full expanding-window backtest"
        )

    ranking_frame = (
        pd.DataFrame(
            ranking_frame
        )
        .reset_index(drop=True)
    )

    if len(ranking_frame) > 0:
        prod_name = str(
            ranking_frame.iloc[0][
                "Model"
            ]
        )

    else:
        prod_name = "Naive"

    prod_col = model_cols[
        prod_name
    ]

    # -----------------------------------------------------------------
    # 6. Identify best learned-model benchmark
    # -----------------------------------------------------------------

    baseline_models = {
        "Naive",
        "Seasonal Naive",
    }

    learned_models = [
        model_name
        for model_name in model_cols
        if model_name
        not in baseline_models
    ]

    if len(recent_err) > 0:
        learned_ranking_source = (
            recent_err
        )

    elif len(holdout_err) > 0:
        learned_ranking_source = (
            holdout_err
        )

    else:
        learned_ranking_source = (
            full_err
        )

    prod_learned_ranking = (
        learned_ranking_source.loc[
            learned_ranking_source[
                "Model"
            ].isin(
                learned_models
            )
        ]
        .copy()
    )

    if len(prod_learned_ranking) > 0:
        learned_order = np.argsort(
            prod_learned_ranking[
                "MAE"
            ].to_numpy(
                dtype=float
            )
        )

        prod_learned_ranking = (
            prod_learned_ranking
            .iloc[learned_order]
            .reset_index(drop=True)
        )

    # If the production model itself is learned, use the next-best
    # learned model as the comparison benchmark when available.
    if (
        prod_name in learned_models
        and len(
            prod_learned_ranking
        ) > 1
    ):
        prod_learned_name = str(
            prod_learned_ranking
            .iloc[1]["Model"]
        )

    elif (
        prod_name not in learned_models
        and len(
            prod_learned_ranking
        ) > 0
    ):
        prod_learned_name = str(
            prod_learned_ranking
            .iloc[0]["Model"]
        )

    else:
        prod_learned_name = (
            prod_name
        )

    # -----------------------------------------------------------------
    # 7. Fit models on all available training observations
    # -----------------------------------------------------------------

    prod_bundle = fit_model_bundle(
        features_df,
        include_nb=include_nb,
        verbose=False,
    )

    # -----------------------------------------------------------------
    # 8. Generate forecasts
    # -----------------------------------------------------------------

    if horizon <= 1:
        next_features = (
            build_next_week_features(
                df
            )
        )

        next_predictions = (
            predict_with_bundle(
                prod_bundle,
                next_features,
            )
        )

        next_date = pd.Timestamp(
            next_features[
                "date"
            ].iloc[0]
        )

        forecast_df = pd.DataFrame(
            {
                "date": [
                    next_date
                ],
                "step": [1],
                **{
                    name: [
                        float(
                            values[0]
                        )
                    ]
                    for name, values
                    in next_predictions.items()
                },
            }
        )

    else:
        forecast_df = (
            build_multi_step_forecast(
                df,
                prod_bundle,
                horizon,
            )
        )

        next_date = pd.Timestamp(
            forecast_df[
                "date"
            ].iloc[0]
        )

    # -----------------------------------------------------------------
    # 9. Choose model shown on multi-step chart
    # -----------------------------------------------------------------

    if horizon > 1:
        preferred_display_model = (
            prod_learned_name
            if prod_name
            in baseline_models
            else prod_name
        )

        display_model = (
            choose_multi_step_display_model(
                ranking_frame=ranking_frame,
                forecast_df=forecast_df,
                latest_actual=float(
                    df[
                        "notifications"
                    ].iloc[-1]
                ),
                preferred_model=(
                    preferred_display_model
                ),
            )
        )

    else:
        display_model = prod_name

    # -----------------------------------------------------------------
    # 10. Residual-based uncertainty bands
    # -----------------------------------------------------------------

    # Calibrate uncertainty only on predictions made BEFORE the final
    # holdout period. This keeps holdout interval coverage genuinely
    # out-of-sample.
    calibration_bt = (
        selection_bt
        .dropna(
            subset=[prod_col]
        )
        .copy()
    )
    
    if len(calibration_bt) < 8:
        raise RuntimeError(
            "Not enough pre-holdout predictions are available "
            "to calibrate forecast uncertainty."
        )
    
    residuals = (
        calibration_bt[
            "actual"
        ].to_numpy(
            dtype=float
        )
        - calibration_bt[
            prod_col
        ].to_numpy(
            dtype=float
        )
    )
    
    valid_residuals = residuals[
        np.isfinite(
            residuals
        )
    ]
    
    if len(valid_residuals) < 8:
        raise RuntimeError(
            "Not enough valid pre-holdout residuals are available "
            "for uncertainty estimation."
        )
    
    r_lo, r_hi = np.quantile(
        valid_residuals,
        BAND_QUANTILES,
    )

    steps = forecast_df[
        "step"
    ].to_numpy(
        dtype=float
    )

    forecast_df["lower"] = np.clip(
        forecast_df[
            display_model
        ].to_numpy(
            dtype=float
        )
        + r_lo
        * np.sqrt(steps),
        0,
        None,
    )

    forecast_df["upper"] = np.clip(
        forecast_df[
            display_model
        ].to_numpy(
            dtype=float
        )
        + r_hi
        * np.sqrt(steps),
        0,
        None,
    )

    # -----------------------------------------------------------------
    # 11. Dashboard headline values
    # -----------------------------------------------------------------

    latest_actual = int(
        df[
            "notifications"
        ].iloc[-1]
    )

    latest_date = pd.Timestamp(
        df[
            "date"
        ].iloc[-1]
    )

    if prod_name in forecast_df.columns:
        prod_forecast = float(
            forecast_df[
                prod_name
            ].iloc[0]
        )

    else:
        prod_forecast = float(
            forecast_df[
                display_model
            ].iloc[0]
        )

    display_forecast = float(
        forecast_df[
            display_model
        ].iloc[0]
    )

    learned_forecast = float(
        forecast_df[
            prod_learned_name
        ].iloc[0]
    )

    next_lower = float(
        forecast_df[
            "lower"
        ].iloc[0]
    )

    next_upper = float(
        forecast_df[
            "upper"
        ].iloc[0]
    )

    final_forecast = float(
        forecast_df[
            display_model
        ].iloc[-1]
    )

    # -----------------------------------------------------------------
    # 12. Trend classification
    # -----------------------------------------------------------------

    pct_change = (
        prod_forecast
        - latest_actual
    ) / max(
        latest_actual,
        1,
    )

    if pct_change > TREND_THRESHOLD:
        trend_label = "Rising"
        trend_icon = "📈"

    elif pct_change < -TREND_THRESHOLD:
        trend_label = "Falling"
        trend_icon = "📉"

    else:
        trend_label = "Stable"
        trend_icon = "➡️"

    # -----------------------------------------------------------------
    # 13. Historical risk classification
    # -----------------------------------------------------------------

    q50, q75, q90 = (
        df[
            "notifications"
        ]
        .quantile(
            [
                0.50,
                0.75,
                0.90,
            ]
        )
        .to_numpy()
    )

    if prod_forecast >= q90:
        risk_label = "Very High"
        risk_icon = "🔴"

    elif prod_forecast >= q75:
        risk_label = "High"
        risk_icon = "🟠"

    elif prod_forecast >= q50:
        risk_label = "Moderate"
        risk_icon = "🟡"

    else:
        risk_label = "Low"
        risk_icon = "🟢"

    # -----------------------------------------------------------------
    # 14. Holdout diagnostics
    # -----------------------------------------------------------------

    fr_card = holdout_bt.copy()

    fr_card["pl"] = np.clip(
        fr_card[
            prod_col
        ].to_numpy(
            dtype=float
        )
        + r_lo,
        0,
        None,
    )

    fr_card["pu"] = np.clip(
        fr_card[
            prod_col
        ].to_numpy(
            dtype=float
        )
        + r_hi,
        0,
        None,
    )

    if len(fr_card) > 0:
        cov = float(
            (
                (
                    fr_card[
                        "actual"
                    ]
                    >= fr_card["pl"]
                )
                & (
                    fr_card[
                        "actual"
                    ]
                    <= fr_card["pu"]
                )
            ).mean()
        )

    else:
        cov = np.nan

    recent_8 = (
        holdout_bt
        .tail(8)
        .copy()
    )

    if len(recent_8) > 0:
        recent_mae = float(
            mean_absolute_error(
                recent_8[
                    "actual"
                ],
                recent_8[
                    prod_col
                ],
            )
        )

    else:
        recent_mae = np.nan

    predictor_diagnostics = (
        build_predictor_diagnostics(
            backtest_results,
            model_cols,
            prod_name,
            holdout_start,
        )
    )

    # -----------------------------------------------------------------
    # 15. Metadata
    # -----------------------------------------------------------------

    cache_info = get_cache_info(
        uf_code=uf_code
    )

    uf_label = UF_OPTIONS.get(
        uf_code,
        f"UF {uf_code}",
    )

    # -----------------------------------------------------------------
    # 16. Return dashboard state
    # -----------------------------------------------------------------

    return {
        "available_data_urls": available_data_urls,
        "available_years": available_years,
        "raw_all": raw_all,
        "full_df": full_df,
        "df": df,
        "features_df": features_df,
        "backtest_results": backtest_results,
        "model_cols": model_cols,
        "full_err": full_err,
        "holdout_start": holdout_start,
        "selection_bt": selection_bt,
        "holdout_bt": holdout_bt,
        "holdout_err": holdout_err,
        "recent_selection_bt": recent_selection_bt,
        "recent_err": recent_err,
        "selection_score_table": selection_score_table,
        "ranking_frame": ranking_frame,
        "ranking_basis": ranking_basis,
        "prod_name": prod_name,
        "prod_col": prod_col,
        "prod_learned_name": prod_learned_name,
        "prod_learned_ranking": prod_learned_ranking,
        "prod_bundle": prod_bundle,
        "forecast_df": forecast_df,
        "next_date": next_date,
        "display_model": display_model,
        "r_lo": float(r_lo),
        "r_hi": float(r_hi),
        "latest_actual": latest_actual,
        "latest_date": latest_date,
        "prod_forecast": prod_forecast,
        "display_forecast": display_forecast,
        "learned_forecast": learned_forecast,
        "next_lower": next_lower,
        "next_upper": next_upper,
        "final_forecast": final_forecast,
        "pct_change": float(pct_change),
        "trend_label": trend_label,
        "trend_icon": trend_icon,
        "risk_label": risk_label,
        "risk_icon": risk_icon,
        "cov": (
            float(cov)
            if np.isfinite(cov)
            else np.nan
        ),
        "recent_mae": (
            float(recent_mae)
            if np.isfinite(recent_mae)
            else np.nan
        ),
        "predictor_diagnostics": predictor_diagnostics,
        "nb_status": nb_status,
        "uf_code": uf_code,
        "uf_label": uf_label,
        "data_source": DATA_SOURCE_LABEL,
        "cache_info": cache_info,
    }
