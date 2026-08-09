"""Streamlit dashboard for weekly dengue surveillance.

Presentation lives here.
Data ingestion, modeling, evaluation, forecasting, and orchestration
live in the src package.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import TARGET_UF_CODE, TREND_THRESHOLD, UF_OPTIONS
from src.data import build_weekly_series, cache_exists, get_cache_info
from src.evaluation import build_error_table, score_predictions
from src.pipeline import build_monitoring_state


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dengue Surveillance Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL CONSTANTS
# ============================================================

ACCENT = "#2A9D8F"
GREEN = "#2FBF71"
ORANGE = "#F4A261"
RED = "#E76F51"
BLUE = "#4EA8DE"
PURPLE = "#8B7CF6"

TEXT = "#E7EEF2"
MUTED = "#8FA6B2"
BACKGROUND = "#081318"
GRID = "rgba(170, 190, 200, 0.12)"

MODEL_COLORS = {
    "Naive": "#C9D1D9",
    "Seasonal Naive": "#7C86A3",
    "Linear Regression": BLUE,
    "Random Forest": GREEN,
    "Negative Binomial": PURPLE,
}


# ============================================================
# CSS
# ============================================================

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(
                    circle at top left,
                    #10232d 0%,
                    {BACKGROUND} 45%
                );
            color: {TEXT};
        }}

        [data-testid="stSidebar"] {{
            background: rgba(10, 24, 31, 0.97);
            border-right: 1px solid rgba(143, 166, 178, 0.12);
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        .subtitle {{
            color: {MUTED};
            font-size: 0.98rem;
            margin-top: -0.35rem;
        }}

        .meta {{
            color: {MUTED};
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }}

        .outlook {{
            background: linear-gradient(
                90deg,
                rgba(42, 157, 143, 0.16),
                rgba(42, 157, 143, 0.04)
            );
            border: 1px solid rgba(42, 157, 143, 0.24);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin: 1rem 0;
        }}

        .outlook-label,
        .metric-label {{
            color: {MUTED};
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .outlook-value {{
            font-size: 1.2rem;
            font-weight: 650;
            margin-top: 0.25rem;
        }}

        .metric-card {{
            background: rgba(15, 28, 35, 0.92);
            border: 1px solid rgba(143, 166, 178, 0.12);
            border-radius: 16px;
            padding: 1rem;
            min-height: 100px;
        }}

        .metric-value {{
            font-size: 1.45rem;
            font-weight: 650;
            margin-top: 0.4rem;
        }}

        .metric-note {{
            color: {MUTED};
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }}

        .info-box {{
            background: rgba(15, 28, 35, 0.75);
            border: 1px solid rgba(143, 166, 178, 0.10);
            border-radius: 14px;
            padding: 1rem;
        }}

        .footer {{
            color: {MUTED};
            font-size: 0.8rem;
            margin-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SMALL HELPERS
# ============================================================

def format_number(value: float) -> str:
    return f"{int(round(float(value))):,}"


def status_color(label: str) -> str:
    if label in {"Low", "Stable"}:
        return GREEN

    if label == "Moderate":
        return ORANGE

    if label in {"High", "Very High"}:
        return RED

    return BLUE


def metric_card(
    label: str,
    value: str,
    note: str = "",
    color: str = TEXT,
) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>

            <div
                class="metric-value"
                style="color:{color};"
            >
                {value}
            </div>

            <div class="metric-note">
                {note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CACHE WRAPPERS
# ============================================================

@st.cache_data(
    show_spinner=False,
    ttl=3600,
)
def get_weekly_data(
    uf_code: int,
):
    return build_weekly_series(
        refresh=False,
        uf_code=uf_code,
    )


@st.cache_data(
    show_spinner=False,
    ttl=3600,
)
def get_monitoring_state(
    uf_code: int,
    horizon: int,
    include_nb: bool,
):
    return build_monitoring_state(
        horizon=horizon,
        include_nb=include_nb,
        refresh_data=False,
        uf_code=uf_code,
    )


# ============================================================
# EVALUATION HELPERS
# ============================================================

def build_change_table(
    backtest: pd.DataFrame,
    model_cols: Dict[str, str],
) -> pd.DataFrame:
    rows: List[dict] = []

    for model_name, prediction_col in model_cols.items():
        sample = (
            backtest
            .dropna(
                subset=[prediction_col]
            )
            .copy()
        )

        if sample.empty:
            continue

        scores = score_predictions(
            sample["actual"].to_numpy(),
            sample[prediction_col].to_numpy(),
            sample["previous_week"].to_numpy(),
        )

        rows.append(
            {
                "Model": model_name,
                "Direction Accuracy": scores[
                    "Direction Acc."
                ],
                "Rising Recall": scores[
                    "Rising Recall"
                ],
                "Rising Precision": scores[
                    "Rising Precision"
                ],
                "Rising F1": scores[
                    "Rising F1"
                ],
                "False Alarm Rate": scores[
                    "False Alarm Rate"
                ],
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Rising F1",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# CHARTS
# ============================================================

def history_chart(
    df: pd.DataFrame,
    rolling_window: int,
) -> go.Figure:
    chart_df = df.copy()

    chart_df["rolling"] = (
        chart_df["notifications"]
        .rolling(
            rolling_window
        )
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["notifications"],
            name="Weekly notifications",
            line=dict(
                color=BLUE,
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["rolling"],
            name=(
                f"{rolling_window}-week "
                "rolling average"
            ),
            line=dict(
                color=ACCENT,
                width=2.7,
            ),
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(
            t=20,
            r=20,
            b=40,
            l=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
        xaxis_title="Week",
        yaxis_title="Notifications",
    )

    fig.update_xaxes(
        showgrid=False,
    )

    fig.update_yaxes(
        gridcolor=GRID,
    )

    return fig


def seasonal_chart(
    df: pd.DataFrame,
) -> go.Figure:
    profile = (
        df.groupby(
            "epi_week",
            as_index=False,
        )["notifications"]
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=profile["epi_week"],
            y=profile["notifications"],
            marker_color=ACCENT,
            name="Average notifications",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(
            t=20,
            r=20,
            b=40,
            l=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis_title="Epidemiological week",
        yaxis_title="Average notifications",
    )

    fig.update_xaxes(
        showgrid=False,
    )

    fig.update_yaxes(
        gridcolor=GRID,
    )

    return fig


def backtest_chart(
    backtest: pd.DataFrame,
    model_cols: Dict[str, str],
    selected_models: List[str],
    prod_col: str,
    r_lo: float,
    r_hi: float,
) -> go.Figure:
    chart_df = backtest.copy()

    chart_df["lower"] = np.clip(
        chart_df[prod_col] + r_lo,
        0,
        None,
    )

    chart_df["upper"] = np.clip(
        chart_df[prod_col] + r_hi,
        0,
        None,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["date"],
            y=chart_df["actual"],
            name="Actual",
            line=dict(
                color=TEXT,
                width=2.8,
            ),
        )
    )

    for model_name in selected_models:
        if model_name not in model_cols:
            continue

        column = model_cols[
            model_name
        ]

        line_style = {
            "color": MODEL_COLORS.get(
                model_name,
                BLUE,
            ),
            "width": 2,
        }

        if model_name == "Naive":
            line_style[
                "dash"
            ] = "dash"

        elif model_name == "Seasonal Naive":
            line_style[
                "dash"
            ] = "dot"

        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df[column],
                name=model_name,
                line=line_style,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=list(
                chart_df["date"]
            )
            + list(
                chart_df[
                    "date"
                ][::-1]
            ),
            y=list(
                chart_df["upper"]
            )
            + list(
                chart_df[
                    "lower"
                ][::-1]
            ),
            fill="toself",
            fillcolor=(
                "rgba(42,157,143,0.12)"
            ),
            line=dict(
                color=(
                    "rgba(42,157,143,0)"
                )
            ),
            name="Safety band",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=480,
        margin=dict(
            t=20,
            r=20,
            b=40,
            l=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
        xaxis_title="Week",
        yaxis_title="Notifications",
    )

    fig.update_xaxes(
        showgrid=False,
    )

    fig.update_yaxes(
        gridcolor=GRID,
    )

    return fig


def forecast_chart(
    history: pd.DataFrame,
    forecast_df: pd.DataFrame,
    display_model: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["notifications"],
            name="Observed",
            line=dict(
                color=BLUE,
                width=2.5,
            ),
            mode="lines+markers",
        )
    )

    latest_date = history[
        "date"
    ].iloc[-1]

    latest_value = float(
        history[
            "notifications"
        ].iloc[-1]
    )

    forecast_dates = [
        latest_date
    ] + forecast_df[
        "date"
    ].tolist()

    forecast_values = [
        latest_value
    ] + forecast_df[
        display_model
    ].astype(
        float
    ).tolist()

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            name="Forecast",
            line=dict(
                color=ACCENT,
                width=2.5,
                dash="dash",
            ),
            mode="lines+markers",
        )
    )

    upper_values = [
        latest_value
    ] + forecast_df[
        "upper"
    ].astype(
        float
    ).tolist()

    lower_values = [
        latest_value
    ] + forecast_df[
        "lower"
    ].astype(
        float
    ).tolist()

    fig.add_trace(
        go.Scatter(
            x=(
                forecast_dates
                + forecast_dates[::-1]
            ),
            y=(
                upper_values
                + lower_values[::-1]
            ),
            fill="toself",
            fillcolor=(
                "rgba(42,157,143,0.13)"
            ),
            line=dict(
                color=(
                    "rgba(42,157,143,0)"
                )
            ),
            name="Safety band",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        height=400,
        margin=dict(
            t=20,
            r=20,
            b=40,
            l=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            y=1.08,
        ),
        xaxis_title="Week",
        yaxis_title="Notifications",
    )

    fig.update_xaxes(
        showgrid=False,
    )

    fig.update_yaxes(
        gridcolor=GRID,
    )

    return fig


# ============================================================
# INTERPRETATION
# ============================================================

def classify_forecast(
    forecast: float,
    latest_actual: float,
    historical: pd.Series,
):
    pct_change = (
        forecast
        - latest_actual
    ) / max(
        latest_actual,
        1,
    )

    if pct_change > TREND_THRESHOLD:
        trend = "Rising"

    elif pct_change < -TREND_THRESHOLD:
        trend = "Falling"

    else:
        trend = "Stable"

    q50, q75, q90 = (
        historical.quantile(
            [
                0.50,
                0.75,
                0.90,
            ]
        )
    )

    if forecast >= q90:
        risk = "Very High"

    elif forecast >= q75:
        risk = "High"

    elif forecast >= q50:
        risk = "Moderate"

    else:
        risk = "Low"

    return trend, risk


# ============================================================
# MAIN APP
# ============================================================

def main() -> None:
    inject_css()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.title(
        "🦟 Dengue Surveillance"
    )

    uf_names = list(
        UF_OPTIONS.values()
    )

    uf_codes = list(
        UF_OPTIONS.keys()
    )

    default_index = (
        uf_codes.index(
            TARGET_UF_CODE
        )
        if TARGET_UF_CODE
        in uf_codes
        else 0
    )

    selected_state = (
        st.sidebar.selectbox(
            "State",
            uf_names,
            index=default_index,
        )
    )

    uf_code = uf_codes[
        uf_names.index(
            selected_state
        )
    ]

    horizon = (
        st.sidebar.slider(
            "Forecast horizon (weeks)",
            min_value=1,
            max_value=12,
            value=3,
        )
    )

    rolling_window = (
        st.sidebar.slider(
            "Rolling average",
            min_value=2,
            max_value=12,
            value=4,
        )
    )

    monitoring_weeks = (
        st.sidebar.slider(
            "Recent weeks shown",
            min_value=8,
            max_value=24,
            value=12,
        )
    )

    include_nb = (
        st.sidebar.checkbox(
            "Include Negative Binomial",
            value=False,
        )
    )

    refresh = (
        st.sidebar.button(
            "🔄 Refresh source data",
            use_container_width=True,
        )
    )

    cache_info = get_cache_info(
        uf_code=uf_code
    )

    if cache_info:
        st.sidebar.caption(
            "Latest cached week: "
            f"{cache_info.get('latest_date', 'unknown')}"
        )

    # --------------------------------------------------------
    # REFRESH / INITIAL LOAD
    # --------------------------------------------------------

    if refresh:
        with st.spinner(
            "Downloading latest SINAN data..."
        ):
            build_weekly_series(
                refresh=True,
                uf_code=uf_code,
            )

        get_weekly_data.clear()
        get_monitoring_state.clear()

        st.rerun()

    if not cache_exists(
        uf_code=uf_code
    ):
        with st.spinner(
            "Downloading dengue data..."
        ):
            build_weekly_series(
                refresh=True,
                uf_code=uf_code,
            )

    # --------------------------------------------------------
    # DETERMINE DATA GAP
    # --------------------------------------------------------

    with st.spinner(
        "Preparing surveillance data..."
    ):
        _, _, preliminary_df = (
            get_weekly_data(
                uf_code
            )
        )

    latest_source_date = pd.Timestamp(
        preliminary_df[
            "date"
        ].max()
    )

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    data_gap_weeks = max(
        0,
        int(
            (
                today
                - latest_source_date
            ).days
            // 7
        ),
    )

    effective_horizon = (
        horizon
        + data_gap_weeks
    )

    # --------------------------------------------------------
    # BUILD PIPELINE
    # --------------------------------------------------------

    with st.spinner(
        "Running forecasting pipeline..."
    ):
        state = (
            get_monitoring_state(
                uf_code,
                effective_horizon,
                include_nb,
            )
        )

    # --------------------------------------------------------
    # UNPACK
    # --------------------------------------------------------

    df = pd.DataFrame(
        state[
            "df"
        ]
    ).copy()

    forecast_df = pd.DataFrame(
        state[
            "forecast_df"
        ]
    ).copy()

    backtest_results = (
        pd.DataFrame(
            state[
                "backtest_results"
            ]
        )
        .copy()
    )

    holdout_bt = (
        pd.DataFrame(
            state[
                "holdout_bt"
            ]
        )
        .copy()
    )

    selection_bt = (
        pd.DataFrame(
            state[
                "recent_selection_bt"
            ]
        )
        .copy()
    )

    model_cols = dict(
        state[
            "model_cols"
        ]
    )

    prod_name = str(
        state[
            "prod_name"
        ]
    )

    prod_col = str(
        state[
            "prod_col"
        ]
    )

    display_model = str(
        state[
            "display_model"
        ]
    )

    latest_actual = int(
        state[
            "latest_actual"
        ]
    )

    latest_date = pd.Timestamp(
        state[
            "latest_date"
        ]
    )

    recent_mae = float(
        state[
            "recent_mae"
        ]
    )

    coverage = float(
        state[
            "cov"
        ]
    )

    r_lo = float(
        state[
            "r_lo"
        ]
    )

    r_hi = float(
        state[
            "r_hi"
        ]
    )

    # --------------------------------------------------------
    # FIND ACTUAL FUTURE FORECASTS
    # --------------------------------------------------------

    future_forecasts = (
        forecast_df.loc[
            forecast_df[
                "date"
            ]
            > today
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if future_forecasts.empty:
        future_forecasts = (
            forecast_df
            .tail(
                horizon
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

    future_forecasts = (
        future_forecasts
        .head(
            horizon
        )
        .copy()
    )

    future_forecasts[
        "week_ahead"
    ] = range(
        1,
        len(
            future_forecasts
        )
        + 1,
    )

    step_labels = {
        (
            f"Week +{row['week_ahead']} · "
            f"{pd.Timestamp(row['date']).strftime('%Y-%m-%d')}"
        ): int(
            row[
                "step"
            ]
        )
        for _, row
        in future_forecasts.iterrows()
    }

    if not step_labels:
        st.error(
            "No future forecast weeks are available."
        )

        st.stop()

    selected_label = (
        st.sidebar.selectbox(
            "Forecast week",
            list(
                step_labels.keys()
            ),
            index=0,
        )
    )

    selected_step = (
        step_labels[
            selected_label
        ]
    )

    selected_row = (
        forecast_df.loc[
            forecast_df[
                "step"
            ]
            == selected_step
        ]
        .iloc[0]
    )

    selected_date = pd.Timestamp(
        selected_row[
            "date"
        ]
    )

    selected_forecast = float(
        selected_row[
            prod_name
        ]
        if prod_name
        in selected_row.index
        else selected_row[
            display_model
        ]
    )

    selected_lower = float(
        selected_row[
            "lower"
        ]
    )

    selected_upper = float(
        selected_row[
            "upper"
        ]
    )

    trend_label, risk_label = (
        classify_forecast(
            selected_forecast,
            latest_actual,
            df[
                "notifications"
            ],
        )
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "Weekly Dengue Surveillance Dashboard"
    )

    st.markdown(
        f"""
        <div class="subtitle">
            {selected_state}, Brazil
        </div>

        <div class="meta">
            Source: SINAN/Dengue · Brazilian Ministry of Health
            · Latest observed week: {latest_date.date()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="outlook">
            <div class="outlook-label">
                Selected forecast outlook
            </div>

            <div
                class="outlook-value"
                style="color:{status_color(risk_label)};"
            >
                {trend_label} · {risk_label} risk
            </div>

            <div class="metric-note">
                {selected_date.strftime("%Y-%m-%d")}
                · {format_number(selected_forecast)} projected notifications
                · safety band
                {format_number(selected_lower)}
                –
                {format_number(selected_upper)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    peak_idx = (
        df[
            "notifications"
        ].idxmax()
    )

    peak_value = int(
        df.loc[
            peak_idx,
            "notifications",
        ]
    )

    peak_date = pd.Timestamp(
        df.loc[
            peak_idx,
            "date",
        ]
    )

    metrics = st.columns(
        4
    )

    with metrics[0]:
        metric_card(
            "Latest observed",
            format_number(
                latest_actual
            ),
            latest_date.strftime(
                "%Y-%m-%d"
            ),
        )

    with metrics[1]:
        metric_card(
            "Selected forecast",
            format_number(
                selected_forecast
            ),
            selected_date.strftime(
                "%Y-%m-%d"
            ),
            ACCENT,
        )

    with metrics[2]:
        metric_card(
            "Historical peak",
            format_number(
                peak_value
            ),
            peak_date.strftime(
                "%Y-%m-%d"
            ),
        )

    with metrics[3]:
        metric_card(
            "Production model",
            prod_name,
            f"Recent MAE: {recent_mae:.1f}"
            if np.isfinite(
                recent_mae
            )
            else "Recent MAE unavailable",
        )

    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    (
        overview_tab,
        models_tab,
        backtest_tab,
        monitoring_tab,
    ) = st.tabs(
        [
            "Overview",
            "Models & Evaluation",
            "Backtest",
            "Monitoring",
        ]
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with overview_tab:
        st.subheader(
            "Weekly notifications"
        )

        st.plotly_chart(
            history_chart(
                df,
                rolling_window,
            ),
            use_container_width=True,
        )

        col1, col2 = (
            st.columns(
                2
            )
        )

        with col1:
            st.subheader(
                "Yearly summary"
            )

            yearly_summary = (
                df.groupby(
                    "year"
                )
                .agg(
                    Total=(
                        "notifications",
                        "sum",
                    ),
                    Average=(
                        "notifications",
                        "mean",
                    ),
                    Peak=(
                        "notifications",
                        "max",
                    ),
                )
                .reset_index()
                .rename(
                    columns={
                        "year": "Year"
                    }
                )
            )

            yearly_summary[
                "Average"
            ] = yearly_summary[
                "Average"
            ].round(
                1
            )

            st.dataframe(
                yearly_summary,
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            st.subheader(
                "Seasonality"
            )

            st.plotly_chart(
                seasonal_chart(
                    df
                ),
                use_container_width=True,
            )

        with st.expander(
            "Data quality notes"
        ):
            zero_weeks = int(
                df[
                    "is_imputed_zero_week"
                ].sum()
            )

            st.markdown(
                f"""
                - The signal represents **weekly notifications**, not confirmed cases.
                - Missing calendar weeks are kept explicitly and filled with zero.
                - Zero-filled weeks in the current analysis window: **{zero_weeks}**.
                - Recent observations may later change because of reporting delays.
                """
            )

    # ========================================================
    # MODELS
    # ========================================================

    with models_tab:
        st.subheader(
            "Model selection"
        )

        selection_scores = (
            pd.DataFrame(
                state[
                    "selection_score_table"
                ]
            )
            .copy()
        )

        if not selection_scores.empty:
            st.dataframe(
                selection_scores,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No composite selection-score table was available."
            )

        st.subheader(
            "Final holdout performance"
        )

        holdout_errors = (
            build_error_table(
                holdout_bt,
                model_cols,
            )
        )

        st.dataframe(
            holdout_errors,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Production model: {prod_name}"
            f" · Chart model: {display_model}"
            f" · Selection basis: "
            f"{state['ranking_basis']}"
        )

        with st.expander(
            "Direction and outbreak-change metrics"
        ):
            change_metrics = (
                build_change_table(
                    holdout_bt,
                    model_cols,
                )
            )

            if not change_metrics.empty:
                percentage_cols = [
                    "Direction Accuracy",
                    "Rising Recall",
                    "Rising Precision",
                    "False Alarm Rate",
                ]

                formatted = (
                    change_metrics.copy()
                )

                for column in (
                    percentage_cols
                ):
                    formatted[
                        column
                    ] = (
                        formatted[
                            column
                        ]
                        * 100
                    ).round(
                        1
                    )

                formatted[
                    "Rising F1"
                ] = formatted[
                    "Rising F1"
                ].round(
                    3
                )

                st.dataframe(
                    formatted,
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander(
            "Pre-holdout selection-period errors"
        ):
            selection_errors = (
                build_error_table(
                    selection_bt,
                    model_cols,
                )
            )

            st.dataframe(
                selection_errors,
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # BACKTEST
    # ========================================================

    with backtest_tab:
        st.subheader(
            "Expanding-window backtest"
        )

        available_models = [
            name
            for name
            in MODEL_COLORS
            if name
            in model_cols
        ]

        default_models = [
            name
            for name
            in [
                "Naive",
                "Linear Regression",
                "Random Forest",
            ]
            if name
            in available_models
        ]

        selected_models = (
            st.multiselect(
                "Models shown",
                available_models,
                default=default_models,
            )
        )

        if not holdout_bt.empty:
            st.plotly_chart(
                backtest_chart(
                    holdout_bt,
                    model_cols,
                    selected_models,
                    prod_col,
                    r_lo,
                    r_hi,
                ),
                use_container_width=True,
            )

        else:
            st.info(
                "No holdout observations available."
            )

        cards = st.columns(
            3
        )

        with cards[0]:
            metric_card(
                "Production model",
                prod_name,
                "Chosen before final holdout evaluation",
            )

        with cards[1]:
            metric_card(
                "Recent MAE",
                (
                    f"{recent_mae:.1f}"
                    if np.isfinite(
                        recent_mae
                    )
                    else "N/A"
                ),
                "Last 8 holdout weeks",
            )

        with cards[2]:
            metric_card(
                "Safety-band coverage",
                (
                    f"{coverage:.0%}"
                    if np.isfinite(
                        coverage
                    )
                    else "N/A"
                ),
                "Final holdout",
            )

    # ========================================================
    # MONITORING
    # ========================================================

    with monitoring_tab:
        st.subheader(
            "Current monitoring outlook"
        )

        recent_history = (
            df.tail(
                monitoring_weeks
            )
            .copy()
        )

        selected_forecast_path = (
            forecast_df.loc[
                forecast_df[
                    "step"
                ]
                <= selected_step
            ]
            .copy()
        )

        st.plotly_chart(
            forecast_chart(
                recent_history,
                selected_forecast_path,
                display_model,
            ),
            use_container_width=True,
        )

        monitoring_cards = (
            st.columns(
                4
            )
        )

        with monitoring_cards[0]:
            metric_card(
                "Trend",
                trend_label,
                (
                    "Relative to latest "
                    "observed week"
                ),
                status_color(
                    trend_label
                ),
            )

        with monitoring_cards[1]:
            metric_card(
                "Risk",
                risk_label,
                "Historical percentile",
                status_color(
                    risk_label
                ),
            )

        with monitoring_cards[2]:
            metric_card(
                "Forecast",
                format_number(
                    selected_forecast
                ),
                selected_date.strftime(
                    "%Y-%m-%d"
                ),
                ACCENT,
            )

        with monitoring_cards[3]:
            metric_card(
                "Safety band",
                (
                    f"{format_number(selected_lower)}"
                    "–"
                    f"{format_number(selected_upper)}"
                ),
                "Residual-based interval",
            )

        st.markdown(
            "### Forecast trajectory"
        )

        display_columns = [
            column
            for column
            in [
                "date",
                "step",
                "Naive",
                "Seasonal Naive",
                "Linear Regression",
                "Random Forest",
                "Negative Binomial",
                "lower",
                "upper",
            ]
            if column
            in future_forecasts.columns
        ]

        trajectory_table = (
            future_forecasts[
                display_columns
            ]
            .copy()
        )

        trajectory_table[
            "date"
        ] = trajectory_table[
            "date"
        ].dt.strftime(
            "%Y-%m-%d"
        )

        st.dataframe(
            trajectory_table,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "### Interpretation"
        )

        if trend_label == "Rising":
            interpretation = (
                "The model expects dengue notifications "
                "to increase relative to the latest "
                "observed week."
            )

        elif trend_label == "Falling":
            interpretation = (
                "The model expects dengue notifications "
                "to decrease relative to the latest "
                "observed week."
            )

        else:
            interpretation = (
                "The model expects dengue notifications "
                "to remain relatively close to the latest "
                "observed level."
            )

        st.markdown(
            f"""
            <div class="info-box">
                <strong>{interpretation}</strong>
                <br><br>
                Risk is classified relative to the historical
                distribution of weekly notifications in the
                current analysis window.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================
    # SIDEBAR STATUS
    # ========================================================

    if (
        state.get(
            "nb_status"
        )
        == "omitted"
    ):
        st.sidebar.warning(
            "Negative Binomial was omitted because "
            "the fit was unstable or failed to converge."
        )

    st.sidebar.divider()

    st.sidebar.caption(
        f"Latest observed: {latest_date.date()}"
    )

    st.sidebar.caption(
        f"Source-data lag: ~{data_gap_weeks} week(s)"
    )

    st.sidebar.caption(
        f"Production model: {prod_name}"
    )

    if display_model != prod_name:
        st.sidebar.caption(
            f"Multi-step display: {display_model}"
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">
            Source: SINAN/Dengue open-data files,
            Brazilian Ministry of Health.<br>
            Forecasts are analytical estimates for surveillance
            purposes and are not official public-health alerts
            or clinical diagnoses.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
