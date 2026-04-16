import json
import pathlib
import types
import sys
from typing import Dict, List, Tuple, Union, cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _import_notebook(notebook_path: str, module_name: str = "cp_shared"):
    """Import all code cells from a Jupyter notebook as a Python module."""
    nb_path = pathlib.Path(notebook_path)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    code_parts = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "--- EXPORT CELL" in source:
            continue
        code_parts.append(source)
    full_source = "\n\n".join(code_parts)
    module = types.ModuleType(module_name)
    module.__file__ = str(nb_path)
    exec(compile(full_source, str(nb_path), "exec"), module.__dict__)
    sys.modules[module_name] = module
    return module


_nb_path = pathlib.Path(__file__).parent / "cp_shared_explained.ipynb"
cp = _import_notebook(str(_nb_path))

st.set_page_config(
    page_title="Weekly Dengue Surveillance Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#2A9D8F"
BG = "#081318"
TEXT = "#E7EEF2"
MUTED = "#8FA6B2"
GRID = "rgba(170, 190, 200, 0.12)"
GREEN = "#2FBF71"
ORANGE = "#F4A261"
RED = "#E76F51"
BLUE = "#4EA8DE"
LAVENDER = "#8B7CF6"

MODEL_COLOURS = {
    "Naive": "#C9D1D9",
    "Seasonal Naive": "#7C86A3",
    "Linear Regression": BLUE,
    "Random Forest": GREEN,
    "Negative Binomial": LAVENDER,
}

SettingsDict = Dict[str, Union[int, bool, str]]


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: radial-gradient(circle at top left, #10232d 0%, {BG} 44%);
                color: {TEXT};
            }}
            [data-testid="stSidebar"] {{
                background: rgba(10, 24, 31, 0.95);
                border-right: 1px solid rgba(143, 166, 178, 0.12);
            }}
            .block-container {{
                padding-top: 2.1rem;
                padding-bottom: 1.5rem;
            }}
            .title-block {{
                padding: 0.2rem 0 1rem 0;
            }}
            .subtitle, .meta-line, .section-note, .footer-note {{
                color: {MUTED};
            }}
            .subtitle {{
                font-size: 0.98rem;
                margin-top: -0.2rem;
            }}
            .meta-line {{
                font-size: 0.88rem;
                margin-top: 0.35rem;
            }}
            .outlook-banner {{
                background: linear-gradient(90deg, rgba(42,157,143,0.16) 0%, rgba(42,157,143,0.06) 100%);
                border: 1px solid rgba(42,157,143,0.22);
                border-radius: 18px;
                padding: 1rem 1.2rem;
                margin: 1rem 0 1.1rem 0;
            }}
            .outlook-kicker, .metric-label, .takeaway-title {{
                color: {MUTED};
                font-size: 0.77rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}
            .outlook-main {{
                font-size: 1.15rem;
                font-weight: 600;
                margin: 0.25rem 0 0.15rem 0;
            }}
            .outlook-sub, .metric-note {{
                color: {MUTED};
                font-size: 0.88rem;
            }}
            .metric-card, .mini-card, .sidebar-card, .surface, .takeaway-box {{
                background: rgba(15, 28, 35, 0.92);
                border: 1px solid rgba(143, 166, 178, 0.12);
                border-radius: 18px;
                box-shadow: 0 16px 38px rgba(0, 0, 0, 0.16);
            }}
            .recommendation-card {{
                background: rgba(255, 255, 255, 0.015);
                border-radius: 10px;
                padding: 0.85rem 0.95rem 0.75rem 0.95rem;
                border-left: 3px solid var(--rec-accent);
                margin-top: 0.2rem;
                margin-bottom: 0.2rem;
            }}
            .recommendation-title {{
                color: {TEXT};
                font-size: 0.98rem;
                font-weight: 600;
                margin-bottom: 0.55rem;
            }}
            .recommendation-list {{
                margin: 0;
                padding-left: 1.1rem;
                color: {TEXT};
            }}
            .recommendation-list li {{
                margin-bottom: 0.35rem;
                line-height: 1.4;
            }}
            .metric-card, .mini-card {{
                padding: 0.95rem 1rem;
                min-height: 92px;
            }}
            .sidebar-card, .surface, .takeaway-box {{
                padding: 0.95rem 1rem;
            }}
            .metric-value {{
                font-size: 1.5rem;
                font-weight: 650;
                line-height: 1.1;
                margin-top: 0.45rem;
            }}
            .takeaway-strip {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.85rem;
                margin-bottom: 1rem;
            }}
            .takeaway-value {{
                font-size: 1rem;
                font-weight: 600;
                margin-top: 0.35rem;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.45rem;
                margin-bottom: 0.9rem;
            }}
            .stTabs [data-baseweb="tab"] {{
                background: rgba(15, 28, 35, 0.55);
                border-radius: 999px;
                padding: 0.45rem 0.95rem;
                color: {MUTED};
            }}
            .stTabs [aria-selected="true"] {{
                background: rgba(42,157,143,0.14);
                color: {TEXT};
            }}
            .stExpander {{
                background: rgba(15, 28, 35, 0.55);
                border: 1px solid rgba(143, 166, 178, 0.09);
                border-radius: 14px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_int(value: float) -> str:
    return f"{int(round(float(value))):,}"


def status_colour(label: str) -> str:
    if label in {"Low", "Stable"}:
        return GREEN
    if label == "Moderate":
        return ORANGE
    if label in {"High", "Very High"}:
        return RED
    return BLUE


def recommendations_accent(risk_label: str) -> str:
    if risk_label == "Low":
        return GREEN
    if risk_label == "Moderate":
        return ORANGE
    return RED


def render_card(label: str, value: str, note: str = "", colour: str = TEXT, compact: bool = False) -> None:
    class_name = "mini-card" if compact else "metric-card"
    st.markdown(
        f"""
        <div class="{class_name}">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{colour};">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_weekly_series(uf_code: int = 32, _refresh_key: str = ""):
    """Thin wrapper so Streamlit can cache the weekly data between reruns."""
    return cp.build_weekly_series(refresh=False, uf_code=uf_code)


@st.cache_data(show_spinner=False, ttl=3600)
def load_state(include_nb: bool, horizon: int, uf_code: int = 32):
    return cp.build_monitoring_state(horizon=horizon, include_nb=include_nb, refresh_data=False, uf_code=uf_code)


def build_error_table(backtest_slice: pd.DataFrame, model_cols: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name, column in model_cols.items():
        sample = backtest_slice.dropna(subset=[column]).copy()
        if sample.empty:
            continue
        rows.append(
            {
                "Model": model_name,
                "MAE": float(np.mean(np.abs(sample["actual"] - sample[column]))),
                "RMSE": float(np.sqrt(np.mean((sample["actual"] - sample[column]) ** 2))),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Model", "MAE", "RMSE"])
    return pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)


def build_change_metrics(backtest_slice: pd.DataFrame, model_cols: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for model_name, column in model_cols.items():
        sample = backtest_slice.dropna(subset=[column]).copy()
        if sample.empty:
            continue
        score = cp.score_predictions(
            sample["actual"].to_numpy(),
            sample[column].to_numpy(),
            sample["previous_week"].to_numpy(),
        )
        rows.append(
            {
                "Model": model_name,
                "Direction accuracy": float(score["Direction Acc."]),
                "Rising-week recall": float(score["Rising Recall"]),
                "Rising-week precision": float(score["Rising Precision"]),
                "F1": float(score["Rising F1"]),
                "False-alarm rate": float(score["False Alarm Rate"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Model", "Direction accuracy", "Rising-week recall", "Rising-week precision", "F1", "False-alarm rate"])
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


def prepare_overview_data(df: pd.DataFrame, rolling_window: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overview_df = df.copy()
    overview_df["rolling"] = overview_df["notifications"].rolling(rolling_window).mean()
    yearly_summary = (
        overview_df.groupby("year")
        .agg(Total=("notifications", "sum"), Average=("notifications", "mean"), Peak=("notifications", "max"))
        .round(0)
        .astype(int)
        .reset_index()
        .rename(columns={"year": "Year"})
    )
    seasonal_profile = (
        overview_df.groupby("epi_week", as_index=False)["notifications"]
        .mean()
        .rename(columns={"epi_week": "Epi week", "notifications": "Average notifications"})
    )
    return overview_df, yearly_summary, seasonal_profile


def build_main_chart(overview_df: pd.DataFrame, peak_date: pd.Timestamp, peak_value: int) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=overview_df["date"], y=overview_df["notifications"], name="Weekly notifications", line=dict(color=BLUE, width=2)))
    figure.add_trace(go.Scatter(x=overview_df["date"], y=overview_df["rolling"], name="Rolling average", line=dict(color=ACCENT, width=2.6)))
    figure.add_trace(
        go.Scatter(
            x=[peak_date],
            y=[peak_value],
            mode="markers+text",
            marker=dict(size=9, color="#F7FAFC", line=dict(width=1, color=ACCENT)),
            text=[f"Peak {peak_value:,}"],
            textposition="top center",
            showlegend=False,
        )
    )
    figure.update_layout(height=430, margin=dict(t=16, r=16, b=48, l=16), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.03, x=0), xaxis_title="Week start", yaxis_title="Notifications")
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor=GRID)
    return figure


def build_seasonal_chart(seasonal_profile: pd.DataFrame) -> go.Figure:
    figure = px.bar(seasonal_profile, x="Epi week", y="Average notifications", color="Average notifications", color_continuous_scale=["#12313D", ACCENT, ORANGE])
    figure.update_layout(height=280, margin=dict(t=10, r=10, b=32, l=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, xaxis_title="Epi week", yaxis_title="Average notifications")
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor=GRID)
    return figure


def build_backtest_chart(backtest_2025: pd.DataFrame, model_cols: Dict[str, str], visible_models: List[str], prod_col: str, r_lo: float, r_hi: float) -> go.Figure:
    chart_df = backtest_2025.copy().reset_index(drop=True)
    chart_df["band_lower"] = np.clip(chart_df[prod_col] + r_lo, 0, None)
    chart_df["band_upper"] = np.clip(chart_df[prod_col] + r_hi, 0, None)

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=chart_df["date"], y=chart_df["actual"], name="Actual", line=dict(color="#F2F6FA", width=2.7)))
    for model_name in visible_models:
        if model_name not in model_cols:
            continue
        column = model_cols[model_name]
        style: Dict[str, object] = dict(color=MODEL_COLOURS.get(model_name, BLUE), width=2)
        if model_name == "Naive":
            style["dash"] = "dash"
            style["width"] = 2.2
        elif model_name == "Seasonal Naive":
            style["dash"] = "dot"
        figure.add_trace(go.Scatter(x=chart_df["date"], y=chart_df[column], name=model_name, line=style))

    figure.add_trace(
        go.Scatter(
            x=pd.concat([chart_df["date"], chart_df["date"][::-1]]),
            y=pd.concat([chart_df["band_upper"], chart_df["band_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(42,157,143,0.12)",
            line=dict(color="rgba(42,157,143,0.18)", width=1),
            name="Safety band",
            hoverinfo="skip",
        )
    )

    y_candidates = [chart_df["actual"], chart_df["band_upper"]]
    for model_name in visible_models:
        if model_name in model_cols:
            y_candidates.append(chart_df[model_cols[model_name]])
    y_cap = float(pd.concat(y_candidates, axis=0).quantile(0.985)) * 1.08
    y_cap = max(y_cap, float(chart_df["actual"].max()) * 1.05)

    figure.update_layout(height=480, margin=dict(t=14, r=16, b=48, l=16), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.03, x=0), xaxis_title="Week start", yaxis_title="Notifications", yaxis_range=[0, y_cap])
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor=GRID)
    return figure


def build_monitoring_chart(recent_df: pd.DataFrame, forecast_slice: pd.DataFrame, display_model: str) -> go.Figure:
    labels = recent_df["date"].dt.strftime("%Y-%m-%d").tolist()
    future_labels = forecast_slice["date"].dt.strftime("%Y-%m-%d").tolist()
    future_values = forecast_slice[display_model].astype(float).tolist()
    lower_values = forecast_slice["lower"].astype(float).tolist()
    upper_values = forecast_slice["upper"].astype(float).tolist()

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=labels, y=recent_df["notifications"], name="Observed", line=dict(color=BLUE, width=2.4), mode="lines+markers", marker=dict(size=5)))
    figure.add_trace(go.Scatter(x=[labels[-1]] + future_labels, y=[float(recent_df["notifications"].iloc[-1])] + future_values, name="Forecast", line=dict(color=ACCENT, width=2.2, dash="dash"), mode="lines+markers", marker=dict(size=8, symbol="diamond")))
    figure.add_trace(go.Scatter(x=[future_labels[-1]], y=[future_values[-1]], name="Selected week", mode="markers", marker=dict(size=13, symbol="diamond", color=ACCENT, line=dict(color="#F7FAFC", width=1.2))))
    figure.add_trace(go.Scatter(x=[labels[-1]] + future_labels + future_labels[::-1] + [labels[-1]], y=[float(recent_df["notifications"].iloc[-1])] + upper_values + lower_values[::-1] + [float(recent_df["notifications"].iloc[-1])], fill="toself", fillcolor="rgba(42,157,143,0.12)", line=dict(color="rgba(42,157,143,0.14)", width=1), name="Safety band", hoverinfo="skip"))
    y_cap = max(float(recent_df["notifications"].max()), max(upper_values)) * 1.12
    figure.update_layout(height=360, margin=dict(t=12, r=16, b=36, l=16), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.04, x=0), xaxis_title="Week start", yaxis_title="Notifications", yaxis_range=[0, y_cap])
    figure.update_xaxes(type="category", showgrid=False)
    figure.update_yaxes(gridcolor=GRID)
    return figure


def derive_selected_outlook(selected_forecast: float, latest_actual: int, df: pd.DataFrame) -> Tuple[str, str]:
    pct_change = (selected_forecast - latest_actual) / max(latest_actual, 1)
    if pct_change > cp.TREND_THRESHOLD:
        trend_label = "Rising"
    elif pct_change < -cp.TREND_THRESHOLD:
        trend_label = "Falling"
    else:
        trend_label = "Stable"

    q50, q75, q90 = df["notifications"].quantile([0.50, 0.75, 0.90]).values
    if selected_forecast >= q90:
        risk_label = "Very High"
    elif selected_forecast >= q75:
        risk_label = "High"
    elif selected_forecast >= q50:
        risk_label = "Moderate"
    else:
        risk_label = "Low"
    return trend_label, risk_label


def build_recommendations(risk_label: str, trend_label: str) -> Dict[str, object]:
    trend_text = trend_label.lower()
    sources: List[Tuple[str, str]] = [
        ("WHO Dengue Fact Sheet", "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue"),
        ("Brazilian Ministry of Health — Dengue", "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/d/dengue"),
    ]
    base_actions = [
        "Remove standing water from buckets, drains, gutters, plant pots, and other containers.",
        "Keep water tanks and storage containers covered and clean.",
        "Use repellent, screens, and clothing that reduces mosquito bites during the day.",
    ]

    if risk_label == "Low":
        return {
            "title": "Maintain prevention",
            "actions": base_actions
            + [
                "Keep a weekly home inspection routine even while activity is low.",
                "If fever begins with headache, body pain, nausea, or rash, seek care early.",
            ],
            "sources": sources,
        }
    if risk_label == "Moderate":
        return {
            "title": "Increase household vigilance",
            "actions": base_actions
            + [
                f"Inspect the home and nearby outdoor areas more than once per week while activity is moderate and trend is {trend_text}.",
                "Prioritize protection for older adults, children, pregnant people, and anyone with chronic conditions.",
                "If symptoms start, rest, drink fluids, and get medical evaluation promptly.",
            ],
            "sources": sources,
        }
    if risk_label == "High":
        return {
            "title": "Act now to reduce exposure",
            "actions": base_actions
            + [
                "Use daytime bite protection every day at home, school, and work.",
                "Do not self-medicate with ibuprofen or aspirin if dengue is suspected; seek medical guidance.",
                f"Seek care quickly for fever or worsening symptoms, especially while the trend is {trend_text}.",
            ],
            "sources": sources,
        }
    return {
        "title": "High-alert recommendations",
        "actions": base_actions
        + [
            "Treat suspected dengue symptoms as urgent and look for medical care quickly.",
            "Watch for warning signs such as severe abdominal pain, persistent vomiting, bleeding, breathing difficulty, fainting, or extreme weakness.",
            "Reduce exposure immediately for the whole household and follow local public-health guidance closely.",
        ],
        "sources": sources,
    }


def build_context_note(overview_df: pd.DataFrame, peak_year: int) -> str:
    latest_year = int(overview_df["year"].iloc[-1])
    if latest_year == peak_year:
        return f"{latest_year} remains the peak year in the current analysis window."
    latest_total = int(overview_df.loc[overview_df["year"] == latest_year, "notifications"].sum())
    peak_total = int(overview_df.loc[overview_df["year"] == peak_year, "notifications"].sum())
    return f"{latest_year} notifications so far are about {latest_total / max(peak_total, 1) * 100:.0f}% of the {peak_year} total."


def render_sidebar(settings: SettingsDict) -> SettingsDict:
    st.sidebar.markdown(
        """
        <div class="title-block">
            <div style="font-size:1.05rem;font-weight:650;line-height:1.25;">Weekly Dengue Surveillance Dashboard</div>
            <div class="meta-line">Source: SINAN/Dengue — Brazilian Ministry of Health</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ── Area selector ──
    uf_options = cp.UF_OPTIONS
    uf_names = list(uf_options.values())
    uf_codes = list(uf_options.keys())
    default_uf = int(settings.get("uf_code", cp.TARGET_UF_CODE) or cp.TARGET_UF_CODE)
    default_idx = uf_codes.index(default_uf) if default_uf in uf_codes else 0
    selected_name = st.sidebar.selectbox("Area", uf_names, index=default_idx)
    selected_uf_code = uf_codes[uf_names.index(selected_name)]
    settings["uf_code"] = selected_uf_code
    st.sidebar.markdown(f"<div class='subtitle'>{selected_name}, Brazil</div>", unsafe_allow_html=True)
    # ── Refresh button ──
    cache_info = cp.get_cache_info(uf_code=selected_uf_code)
    if cache_info:
        last_updated = cache_info.get("created_local", "unknown")[:16].replace("T", " ")
        latest_date = cache_info.get("latest_date", "?")
        st.sidebar.caption(f"📦 Data cached · latest week: {latest_date} · updated: {last_updated}")
    else:
        st.sidebar.caption("📦 No local cache — data will be downloaded")
    settings["refresh_data"] = st.sidebar.button("🔄 Refresh data from source", use_container_width=True)

    with st.sidebar.expander("Settings", expanded=False):
        settings["forecast_horizon"] = st.slider("Forecast weeks ahead of today", 1, 12, int(settings["forecast_horizon"]))
        settings["rolling_window"] = st.slider("Rolling-average window (weeks)", 2, 12, int(settings["rolling_window"]))
        settings["monitoring_weeks"] = st.slider("Weeks shown in monitoring card chart", 8, 12, int(settings["monitoring_weeks"]))
        settings["include_nb"] = st.checkbox("Include Negative Binomial model", value=bool(settings["include_nb"]))
    return settings


def _show_nb_status(state: Dict[str, object]) -> None:
    """Show a sidebar note if NB was requested but omitted."""
    nb_status = state.get("nb_status", "off")
    if nb_status == "omitted":
        st.sidebar.warning("⚠️ Negative Binomial model was enabled but omitted — the fit did not converge on this data window.", icon="⚠️")


def render_sidebar_summary(selected_date: pd.Timestamp, selected_forecast: float, trend_label: str, risk_label: str) -> None:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="metric-label">Forecast summary</div>
            <div class="metric-note" style="margin-top:0.6rem;">Forecast week</div>
            <div style="font-size:1.02rem;font-weight:600;">{selected_date.strftime('%Y-%m-%d')}</div>
            <div class="metric-note" style="margin-top:0.65rem;">Forecast</div>
            <div style="font-size:1.02rem;font-weight:600;">{format_int(selected_forecast)}</div>
            <div class="metric-note" style="margin-top:0.65rem;">Trend</div>
            <div style="font-size:1.02rem;font-weight:600;color:{status_colour(trend_label)};">{trend_label}</div>
            <div class="metric-note" style="margin-top:0.65rem;">Risk</div>
            <div style="font-size:1.02rem;font-weight:600;color:{status_colour(risk_label)};">{risk_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_models_tab(state: Dict[str, object], holdout_table: pd.DataFrame, selection_table: pd.DataFrame, full_table: pd.DataFrame, change_holdout: pd.DataFrame, change_selection: pd.DataFrame) -> None:
    selection_scores = pd.DataFrame(state["selection_score_table"]).copy()
    selection_winner = selection_scores.iloc[0]["Model"] if not selection_scores.empty else selection_table.iloc[0]["Model"] if not selection_table.empty else state["prod_name"]
    display_model_name = str(state.get("display_model", state["prod_name"]))
    st.markdown(
        f"""
        <div class="takeaway-strip">
            <div class="takeaway-box">
                <div class="takeaway-title">Pre-holdout selection winner</div>
                <div class="takeaway-value">{selection_winner}</div>
            </div>
            <div class="takeaway-box">
                <div class="takeaway-title">Production model (trend &amp; risk)</div>
                <div class="takeaway-value">{state['prod_name']}</div>
            </div>
            <div class="takeaway-box">
                <div class="takeaway-title">Chart display model</div>
                <div class="takeaway-value">{display_model_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 2025 Holdout — Key Real-World Test")
    st.dataframe(holdout_table.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}"}), width="stretch", hide_index=True)
    if not holdout_table.empty:
        best_holdout = str(holdout_table.iloc[0]["Model"])
        learned_only = holdout_table[~holdout_table["Model"].isin(["Naive", "Seasonal Naive"])]
        if not learned_only.empty:
            learned_name = str(learned_only.iloc[0]["Model"])
            gap = float(learned_only.iloc[0]["MAE"]) / max(float(holdout_table.iloc[0]["MAE"]), 0.01)
            st.markdown(f"<div class='section-note'>Best holdout model: {best_holdout} · Closest learned model: {learned_name} · Gap vs naive: {gap:.1f}x MAE</div>", unsafe_allow_html=True)

    with st.expander("Earlier backtests", expanded=False):
        st.markdown("**Pre-holdout selection period metrics**")
        st.dataframe(selection_table.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}"}), width="stretch", hide_index=True)
        st.markdown("**Full backtest metrics**")
        st.dataframe(full_table.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}"}), width="stretch", hide_index=True)

    with st.expander("Change-signal metrics", expanded=False):
        fmt = {"Direction accuracy": "{:.0%}", "Rising-week recall": "{:.0%}", "Rising-week precision": "{:.0%}", "F1": "{:.2f}", "False-alarm rate": "{:.0%}"}
        st.markdown("**2025 holdout**")
        st.dataframe(change_holdout.style.format(fmt), width="stretch", hide_index=True)
        st.markdown("**Pre-holdout selection period**")
        st.dataframe(change_selection.style.format(fmt), width="stretch", hide_index=True)


def main() -> None:
    inject_css()

    settings: SettingsDict = {"forecast_horizon": 3, "rolling_window": 4, "monitoring_weeks": 10, "include_nb": False, "uf_code": cp.TARGET_UF_CODE}
    settings = render_sidebar(settings)
    forecast_horizon = cast(int, settings["forecast_horizon"])
    rolling_window = cast(int, settings["rolling_window"])
    monitoring_weeks = cast(int, settings["monitoring_weeks"])
    include_nb = cast(bool, settings["include_nb"])
    uf_code = cast(int, settings["uf_code"])
    uf_label = cp.UF_OPTIONS.get(uf_code, f"UF {uf_code}")
    refresh_data = bool(settings.get("refresh_data", False))

    # ── Handle refresh button: download new data, then clear Streamlit caches ──
    if refresh_data:
        with st.spinner("Downloading latest data from Ministry of Health..."):
            cp.build_weekly_series(refresh=True, uf_code=uf_code)
        _cached_weekly_series.clear()
        load_state.clear()
        st.rerun()

    # ── Ensure parquet cache exists (first-ever visit) ──
    if not cp.cache_is_fresh(uf_code=uf_code):
        with st.spinner("Downloading data for the first time — this may take a minute..."):
            cp.build_weekly_series(refresh=True, uf_code=uf_code)

    # ── Fast path: read from Streamlit in-memory cache ──
    with st.spinner("Loading surveillance state..."):
        _pre_raw, _pre_full, _pre_df = _cached_weekly_series(uf_code=uf_code)
        _latest_data_date = _pre_df["date"].max()
        _today = pd.Timestamp.today().normalize()
        _data_gap_weeks = max(0, int((_today - _latest_data_date).days // 7))
        effective_horizon = _data_gap_weeks + forecast_horizon
        state = load_state(include_nb, effective_horizon, uf_code=uf_code)

    df = pd.DataFrame(state["df"]).copy()
    backtest_results = pd.DataFrame(state["backtest_results"]).copy()
    model_cols = dict(state["model_cols"])
    forecast_df = pd.DataFrame(state["forecast_df"]).copy()
    display_model = str(state["display_model"])

    peak_idx = int(df["notifications"].idxmax())
    peak_value = int(df.loc[peak_idx, "notifications"])
    peak_date = pd.Timestamp(df.loc[peak_idx, "date"])
    latest_actual = int(state["latest_actual"])
    latest_date = pd.Timestamp(state["latest_date"])
    prod_forecast = float(state["prod_forecast"])
    display_forecast = float(state.get("display_forecast", prod_forecast))
    trend_label = str(state["trend_label"])
    risk_label = str(state["risk_label"])
    prod_name = str(state["prod_name"])
    prod_col = str(state["prod_col"])
    prod_learned_name = str(state["prod_learned_name"])
    next_lower = float(state["next_lower"])
    next_upper = float(state["next_upper"])
    next_date = pd.Timestamp(state["next_date"])
    recent_mae = float(state["recent_mae"])
    holdout_coverage = float(state["cov"])
    total_notifications = int(df["notifications"].sum())
    average_weekly = float(df["notifications"].mean())

    overview_df, yearly_summary, seasonal_profile = prepare_overview_data(df, rolling_window)
    context_note = build_context_note(overview_df, peak_year=int(peak_date.year))
    imputed_zero_weeks = int(df["is_imputed_zero_week"].sum())

    holdout_2025 = backtest_results[backtest_results["date"].dt.year == 2025].copy()
    if holdout_2025.empty:
        holdout_2025 = pd.DataFrame(state["holdout_bt"]).copy()
    holdout_table = build_error_table(holdout_2025, model_cols)
    selection_table = build_error_table(pd.DataFrame(state["recent_selection_bt"]).copy(), model_cols)
    full_table = build_error_table(backtest_results, model_cols)
    change_holdout = build_change_metrics(holdout_2025, model_cols)
    change_selection = build_change_metrics(pd.DataFrame(state["recent_selection_bt"]).copy(), model_cols)

    # Filter forecast to only show weeks from today onward, and re-label
    _today = pd.Timestamp.today().normalize()
    future_forecast_df = forecast_df[forecast_df["date"] > _today].copy().reset_index(drop=True)
    if future_forecast_df.empty:
        future_forecast_df = forecast_df.tail(forecast_horizon).copy().reset_index(drop=True)
    future_forecast_df["future_step"] = range(1, len(future_forecast_df) + 1)

    forecast_step_options = {
        f"Week +{int(row['future_step'])} · {pd.Timestamp(row['date']).strftime('%Y-%m-%d')}": int(row["step"])
        for _, row in future_forecast_df.iterrows()
    }
    selected_step_label = st.sidebar.selectbox(
        "Forecast week",
        options=list(forecast_step_options.keys()),
        index=min(forecast_horizon - 1, len(forecast_step_options) - 1),
        key=f"forecast-step-{forecast_horizon}",
    )
    selected_step = forecast_step_options[selected_step_label]
    selected_row = forecast_df.loc[forecast_df["step"] == selected_step].iloc[0]
    forecast_slice = forecast_df.iloc[: int(selected_row["step"])].copy().reset_index(drop=True)
    selected_step = int(selected_row["step"])
    selected_date = pd.Timestamp(selected_row["date"])
    selected_display_forecast = float(selected_row[display_model])
    # Use production model for trend/risk (not display model)
    selected_prod_forecast = float(selected_row[prod_name]) if prod_name in selected_row.index else selected_display_forecast
    selected_lower = float(selected_row["lower"])
    selected_upper = float(selected_row["upper"])
    selected_trend, selected_risk = derive_selected_outlook(selected_prod_forecast, latest_actual, df)
    recommendations = build_recommendations(selected_risk, selected_trend)

    _show_nb_status(state)

    st.sidebar.caption(
        f"Latest available source week: {latest_date.strftime('%Y-%m-%d')} · Data lag: ~{_data_gap_weeks} week(s) · Selected forecast date: {selected_date.strftime('%Y-%m-%d')}"
    )
    if prod_name != display_model:
        st.sidebar.caption(f"Trend & risk: {prod_name} · Chart: {display_model}")

    render_sidebar_summary(selected_date, selected_prod_forecast, selected_trend, selected_risk)

    peak_share = latest_actual / max(peak_value, 1) * 100
    st.markdown(
        f"""
        <div class="title-block">
            <h1 style="margin:0;">Weekly Dengue Surveillance Dashboard</h1>
            <div class="subtitle">Multi-week surveillance forecast for {uf_label}, Brazil</div>
            <div class="meta-line">Analysis window: {df['date'].min().date()} to {df['date'].max().date()}</div>
            <div class="meta-line">Latest source data: {latest_date.date()} · Forecasting {forecast_horizon} week(s) ahead of today</div>
        </div>
        <div class="outlook-banner">
            <div class="outlook-kicker">Current outlook</div>
            <div class="outlook-main">{selected_trend} and {selected_risk.lower()}</div>
            <div class="outlook-sub">Selected week: {selected_date.strftime('%Y-%m-%d')} · Relative to peak: {selected_prod_forecast / max(peak_value, 1) * 100:.0f}% of {peak_value:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_card("Total notifications", format_int(total_notifications), "Current analysis window")
    with kpi_cols[1]:
        render_card("Weeks analyzed", format_int(len(df)), "Weekly surveillance series")
    with kpi_cols[2]:
        render_card("Peak weekly notifications", format_int(peak_value), "Highest observed week")
    with kpi_cols[3]:
        render_card("Average weekly notifications", format_int(average_weekly), "Mean across weeks shown")

    tabs = st.tabs(["Overview", "Models & Evaluation", "Backtest", "Monitoring Card"])


    with tabs[0]:
        st.markdown("### Weekly dengue notifications over time")
        st.plotly_chart(build_main_chart(overview_df, peak_date, peak_value), use_container_width=True)
        st.markdown(f"<div class='section-note'>{context_note}</div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("### Yearly summary")
            st.dataframe(yearly_summary, width="stretch", hide_index=True)
        with col_right:
            st.markdown("### Seasonal profile by epi week")
            st.plotly_chart(build_seasonal_chart(seasonal_profile), use_container_width=True)

        with st.expander("Data quality notes", expanded=False):
            st.markdown(
                "- Signal is weekly notifications, not confirmed cases.\n"
                f"- Zero-filled weeks in the analysis window: {imputed_zero_weeks}.\n"
                "- Missing weeks are treated as zero notifications in the weekly series.\n"
                "- Recent weeks may still move as reporting catches up."
            )

    with tabs[1]:
        render_models_tab(state, holdout_table, selection_table, full_table, change_holdout, change_selection)

    with tabs[2]:
        top_cols = st.columns(3)
        with top_cols[0]:
            render_card("Production model", prod_name, "Current operational default", compact=True)
        with top_cols[1]:
            render_card("Safety-band coverage", f"{holdout_coverage:.0%}", "Share of holdout weeks inside band", colour=ACCENT, compact=True)
        with top_cols[2]:
            avg_band_width = float((np.clip(holdout_2025[prod_col] + float(state['r_hi']), 0, None) - np.clip(holdout_2025[prod_col] + float(state['r_lo']), 0, None)).mean()) if not holdout_2025.empty else 0.0
            render_card("Avg band width", f"{avg_band_width:.1f}", "Notifications", compact=True)

        available_models = ["Naive", "Linear Regression", "Random Forest"]

        if "Seasonal Naive" in model_cols:
            available_models.append("Seasonal Naive")
        if include_nb and "Negative Binomial" in model_cols:
            available_models.append("Negative Binomial")

        visible_models = st.multiselect("Models to display", available_models, default=[m for m in ["Naive", "Linear Regression", "Random Forest"] if m in available_models])
        st.plotly_chart(build_backtest_chart(holdout_2025, model_cols, visible_models, prod_col, float(state["r_lo"]), float(state["r_hi"])), use_container_width=True)

    with tabs[3]:
        cards = st.columns(5)
        with cards[0]:
            render_card("Latest observed", format_int(latest_actual), latest_date.strftime("%Y-%m-%d"), compact=True)
        with cards[1]:
            forecast_label = "Forecast next week" if selected_step == 1 else f"Forecast week +{selected_step}"
            render_card(forecast_label, format_int(selected_prod_forecast), f"{selected_date.strftime('%Y-%m-%d')} · {prod_name}", colour=ACCENT, compact=True)
        with cards[2]:
            render_card("Safety band", f"{int(selected_lower):,}–{int(selected_upper):,}", selected_date.strftime("%Y-%m-%d"), compact=True)
        with cards[3]:
            render_card("Trend", selected_trend, f"Based on {prod_name}", colour=status_colour(selected_trend), compact=True)
        with cards[4]:
            render_card("Risk", selected_risk, f"Based on {prod_name}", colour=status_colour(selected_risk), compact=True)

        recent_window = df.tail(monitoring_weeks)
        st.plotly_chart(build_monitoring_chart(recent_window, forecast_slice, display_model), use_container_width=True)

        note_cols = st.columns([1.1, 1])
        with note_cols[0]:
            st.markdown("### Model Notes")
            notes_rows = [
                ["Production model", prod_name, "Drives trend, risk, and safety bands"],
                ["Chart display model", display_model, "Used for multi-step chart visualisation"],
                ["Learned benchmark", prod_learned_name, "Best learned (non-baseline) model"],
                ["Recent 8-week MAE", f"{recent_mae:.1f}", "Production model accuracy"],
                ["Holdout band coverage", f"{holdout_coverage:.0%}", "Share of holdout weeks inside band"],
            ]
            notes_df = pd.DataFrame(notes_rows, columns=["Label", "Value", "Explanation"])
            st.dataframe(notes_df, width="stretch", hide_index=True)
        with note_cols[1]:
            st.markdown("### Interpretation")
            if selected_trend == "Stable":
                interpretation = "The selected forecast week is expected to remain near the latest observed level."
            elif selected_trend == "Rising":
                interpretation = "The selected forecast week is expected to move above the latest observed level."
            else:
                interpretation = "The selected forecast week is expected to move below the latest observed level."
            st.markdown(f"<div class='surface'>{interpretation}</div>", unsafe_allow_html=True)

        recommendation_items = "".join(
            f"<li>{item}</li>" for item in cast(List[str], recommendations["actions"])
        )
        st.markdown(
            f"""
            <div style="--rec-accent: {recommendations_accent(selected_risk)};">
                <div class="recommendation-card">
                    <div class="recommendation-title">Recommendations — {recommendations['title']}</div>
                    <ul class="recommendation-list">{recommendation_items}</ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        source_links = " · ".join(
            f"[{label}]({url})" for label, url in cast(List[Tuple[str, str]], recommendations["sources"])
        )
        st.caption(f"Sources: {source_links}")

        with st.expander("All forecast weeks", expanded=False):
            forecast_columns = [column for column in future_forecast_df.columns if column not in {"date", "step", "future_step", "lower", "upper"}]
            all_forecasts = future_forecast_df[["date", "future_step"] + forecast_columns + ["lower", "upper"]].copy()
            all_forecasts = all_forecasts.rename(columns={"future_step": "week ahead"})
            all_forecasts["date"] = all_forecasts["date"].apply(lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"))
            st.dataframe(all_forecasts.style.format({column: "{:.1f}" for column in forecast_columns + ["lower", "upper"]}), width="stretch", hide_index=True)

    st.markdown(
        """
        <div class="footer-note">
            Source: SINAN/Dengue official open-data files, Brazilian Ministry of Health<br>
            Signal: weekly notifications, not confirmed cases<br>
            Prototype thresholds are heuristic
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
