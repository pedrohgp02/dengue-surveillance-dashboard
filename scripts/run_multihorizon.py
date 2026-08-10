"""Run multi-horizon evaluation on real dengue surveillance data."""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Make the repository root importable when this file is executed as:
#
#     python scripts/run_multihorizon.py
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import (  # noqa: E402
    TARGET_UF_CODE,
    UF_OPTIONS,
)

from src.data import (  # noqa: E402
    build_weekly_series,
)

from src.multihorizon import (  # noqa: E402
    build_horizon_policy,
    run_multi_horizon_backtest,
    summarize_multi_horizon_backtest,
)


# ---------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------

HORIZONS = (
    1,
    2,
    4,
    8,
    12,
)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate dengue forecasts at multiple historical "
            "forecast horizons."
        )
    )

    parser.add_argument(
        "--uf-code",
        type=int,
        default=TARGET_UF_CODE,
        help=(
            "IBGE state code. "
            f"Default: {TARGET_UF_CODE}."
        ),
    )

    parser.add_argument(
        "--origin-step",
        type=int,
        default=4,
        help=(
            "Spacing between historical forecast origins. "
            "Use 4 for a faster evaluation or 1 for every week. "
            "Default: 4."
        ),
    )

    parser.add_argument(
        "--selection-weeks",
        type=int,
        default=52,
        help=(
            "Number of pre-holdout target weeks used for "
            "horizon-specific model selection. Default: 52."
        ),
    )

    parser.add_argument(
        "--holdout-weeks",
        type=int,
        default=52,
        help=(
            "Number of final target weeks reserved for untouched "
            "evaluation. Default: 52."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Redownload source data before running the evaluation."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def print_summary_table(
    summary,
) -> None:
    """Print the overall horizon-by-model performance table."""
    print(
        "\nOverall multi-horizon results:\n"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )


def print_policy_table(
    policy,
) -> None:
    """Print horizon-specific model selection and holdout results."""
    print(
        "\nHorizon-specific production policy:\n"
    )

    columns = [
        "Horizon",
        "Selected Model",
        "Selection N",
        "Selection MAE",
        "Holdout N",
        "Holdout MAE",
        "Holdout RMSE",
        "Holdout Bias",
        "Naive Holdout MAE",
        "Skill vs Naive",
    ]

    policy_display = (
        policy[
            columns
        ]
        .copy()
    )

    # Convert skill from proportion to percentage for readability.
    policy_display[
        "Skill vs Naive"
    ] = (
        policy_display[
            "Skill vs Naive"
        ]
        * 100
    )

    print(
        policy_display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    if not policy.empty:
        selection_start = (
            policy[
                "Selection Start"
            ].iloc[0]
        )

        selection_end = (
            policy[
                "Selection End"
            ].iloc[0]
        )

        holdout_start = (
            policy[
                "Holdout Start"
            ].iloc[0]
        )

        holdout_end = (
            policy[
                "Holdout End"
            ].iloc[0]
        )

        print(
            "\nEvaluation windows:"
        )

        print(
            "  Selection: "
            f"{selection_start} → {selection_end}"
        )

        print(
            "  Holdout:   "
            f"{holdout_start} → {holdout_end}"
        )


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_results(
    results,
    summary,
    policy,
    uf_code: int,
) -> None:
    """Save evaluation outputs as CSV files."""
    output_dir = (
        PROJECT_ROOT
        / "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / f"multihorizon_{uf_code}_raw.csv"
    )

    summary_path = (
        output_dir
        / f"multihorizon_{uf_code}_summary.csv"
    )

    policy_path = (
        output_dir
        / f"multihorizon_{uf_code}_policy.csv"
    )

    results.to_csv(
        raw_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    policy.to_csv(
        policy_path,
        index=False,
    )

    print(
        "\nSaved outputs:"
    )

    print(
        f"  Raw forecasts: {raw_path}"
    )

    print(
        f"  Summary:       {summary_path}"
    )

    print(
        f"  Policy:        {policy_path}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete real-data multi-horizon evaluation."""
    args = parse_args()

    if args.uf_code not in UF_OPTIONS:
        available = ", ".join(
            f"{code} ({label})"
            for code, label
            in UF_OPTIONS.items()
        )

        raise ValueError(
            f"UF code {args.uf_code} is not configured. "
            f"Available states: {available}"
        )

    uf_label = UF_OPTIONS[
        args.uf_code
    ]

    print(
        "\n"
        "============================================\n"
        " Dengue Multi-Horizon Forecast Evaluation\n"
        "============================================"
    )

    print(
        f"\nState: {uf_label} "
        f"(IBGE code {args.uf_code})"
    )

    print(
        "Forecast horizons: "
        + ", ".join(
            f"{h}w"
            for h in HORIZONS
        )
    )

    print(
        f"Historical origin step: "
        f"{args.origin_step} week(s)"
    )

    print(
        f"Selection window: "
        f"{args.selection_weeks} target weeks"
    )

    print(
        f"Final holdout: "
        f"{args.holdout_weeks} target weeks"
    )

    # -------------------------------------------------------------
    # Load real SINAN data
    # -------------------------------------------------------------

    print(
        f"\nLoading dengue data for {uf_label}..."
    )

    _, _, history = (
        build_weekly_series(
            refresh=args.refresh,
            uf_code=args.uf_code,
        )
    )

    print(
        f"Loaded {len(history)} weekly observations."
    )

    print(
        "Analysis period: "
        f"{history['date'].min().date()} "
        "→ "
        f"{history['date'].max().date()}"
    )

    # -------------------------------------------------------------
    # Historical multi-horizon forecasting
    # -------------------------------------------------------------

    print(
        "\nRunning historical multi-horizon evaluation..."
    )

    results = (
        run_multi_horizon_backtest(
            history_df=history,
            horizons=HORIZONS,
            include_nb=False,
            min_history_weeks=60,
            origin_step=args.origin_step,
        )
    )

    if results.empty:
        raise RuntimeError(
            "The multi-horizon evaluation produced no results."
        )

    # -------------------------------------------------------------
    # Overall diagnostics
    # -------------------------------------------------------------

    summary = (
        summarize_multi_horizon_backtest(
            results
        )
    )

    print_summary_table(
        summary
    )

    # -------------------------------------------------------------
    # Horizon-specific model policy
    #
    # Models are selected on a pre-holdout period and then evaluated
    # on a later untouched period.
    # -------------------------------------------------------------

    policy = (
        build_horizon_policy(
            results,
            selection_weeks=(
                args.selection_weeks
            ),
            holdout_weeks=(
                args.holdout_weeks
            ),
        )
    )

    if policy.empty:
        raise RuntimeError(
            "The horizon-specific model policy produced no results."
        )

    print_policy_table(
        policy
    )

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------

    save_results(
        results=results,
        summary=summary,
        policy=policy,
        uf_code=args.uf_code,
    )

    print(
        "\nEvaluation complete.\n"
    )


if __name__ == "__main__":
    main()
