"""Run multi-horizon evaluation on real dengue surveillance data."""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Make repository root importable when running:
#
#     python scripts/run_multihorizon.py
# ---------------------------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

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
    build_horizon_holdout_comparison,
    build_horizon_policy,
    run_multi_horizon_backtest,
    summarize_multi_horizon_backtest,
)


HORIZONS = (
    1,
    2,
    4,
    8,
    12,
)


# ============================================================
# ARGUMENTS
# ============================================================

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
            "Use 1 for every week. Default: 4."
        ),
    )

    parser.add_argument(
        "--selection-weeks",
        type=int,
        default=52,
        help=(
            "Number of pre-holdout target weeks used for "
            "model selection. Default: 52."
        ),
    )

    parser.add_argument(
        "--holdout-weeks",
        type=int,
        default=52,
        help=(
            "Number of final forecast origins reserved "
            "for untouched evaluation. Default: 52."
        ),
    )

    parser.add_argument(
        "--min-skill",
        type=float,
        default=0.10,
        help=(
            "Minimum MAE improvement over Naive required "
            "for a learned model to be promoted. "
            "0.10 means 10%%. Default: 0.10."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Redownload source data before evaluation."
        ),
    )

    return parser.parse_args()


# ============================================================
# PRINT HELPERS
# ============================================================

def print_overall_summary(
    summary,
) -> None:
    """Print performance across the complete historical evaluation."""
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


def print_policy(
    policy,
) -> None:
    """Print the horizon-specific production policy."""
    print(
        "\nHorizon-specific production policy:\n"
    )

    columns = [
        "Horizon",
        "Selected Model",
        "Decision",
        "Best Learned Candidate",
        "Candidate Skill vs Naive",
        "Candidate Early Skill",
        "Candidate Late Skill",
        "Selection MAE",
        "Holdout MAE",
        "Holdout RMSE",
        "Holdout Bias",
        "Holdout Skill vs Naive",
    ]

    display = (
        policy[
            columns
        ]
        .copy()
    )

    percentage_columns = [
        "Candidate Skill vs Naive",
        "Candidate Early Skill",
        "Candidate Late Skill",
        "Holdout Skill vs Naive",
    ]

    for column in percentage_columns:
        display[column] = (
            display[column]
            * 100
        )

    print(
        display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    if not policy.empty:
        first = (
            policy.iloc[0]
        )

        print(
            "\nEvaluation timeline:"
        )

        print(
            "  Selection: "
            f"{first['Selection Start']} "
            "→ "
            f"{first['Selection End']}"
        )

        print(
            "  Policy locked: "
            f"{first['Policy Cutoff']}"
        )

        print(
            "  Holdout origins: "
            f"{first['Holdout Origin Start']} "
            "→ "
            f"{first['Holdout Origin End']}"
        )


def print_holdout_comparison(
    comparison,
) -> None:
    """Print every model's performance on the untouched holdout."""
    print(
        "\nAll-model untouched holdout comparison:\n"
    )

    display = (
        comparison.copy()
    )

    display[
        "Skill vs Naive"
    ] = (
        display[
            "Skill vs Naive"
        ]
        * 100
    )

    print(
        display.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.2f}"
            ),
        )
    )

    print(
        "\n"
        "IMPORTANT: this table is diagnostic only. "
        "Holdout performance must not be used to "
        "retroactively change the selected policy."
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_results(
    results,
    summary,
    policy,
    comparison,
    uf_code: int,
) -> None:
    """Save all evaluation outputs as CSV files."""
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
        / (
            f"multihorizon_"
            f"{uf_code}_raw.csv"
        )
    )

    summary_path = (
        output_dir
        / (
            f"multihorizon_"
            f"{uf_code}_summary.csv"
        )
    )

    policy_path = (
        output_dir
        / (
            f"multihorizon_"
            f"{uf_code}_policy.csv"
        )
    )

    comparison_path = (
        output_dir
        / (
            f"multihorizon_"
            f"{uf_code}_holdout_comparison.csv"
        )
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

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(
        "\nSaved outputs:"
    )

    print(
        f"  Raw forecasts:      {raw_path}"
    )

    print(
        f"  Overall summary:    {summary_path}"
    )

    print(
        f"  Production policy:  {policy_path}"
    )

    print(
        f"  Holdout comparison: {comparison_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete real-data evaluation."""
    args = parse_args()

    if args.uf_code not in UF_OPTIONS:
        available_states = ", ".join(
            (
                f"{code} "
                f"({label})"
            )
            for code, label
            in UF_OPTIONS.items()
        )

        raise ValueError(
            f"UF code {args.uf_code} is not configured. "
            f"Available states: {available_states}"
        )

    if args.min_skill < 0:
        raise ValueError(
            "--min-skill cannot be negative."
        )

    uf_label = (
        UF_OPTIONS[
            args.uf_code
        ]
    )

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
            f"{horizon}w"
            for horizon
            in HORIZONS
        )
    )

    print(
        "Historical origin step: "
        f"{args.origin_step} week(s)"
    )

    print(
        "Selection window: "
        f"{args.selection_weeks} target weeks"
    )

    print(
        "Final holdout: "
        f"{args.holdout_weeks} forecast origins"
    )

    print(
        "Promotion threshold: "
        f"{args.min_skill:.0%} MAE improvement "
        "over Naive"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # HISTORICAL FORECASTS
    # --------------------------------------------------------

    print(
        "\nRunning historical multi-horizon evaluation..."
    )

    results = (
        run_multi_horizon_backtest(
            history_df=history,
            horizons=HORIZONS,
            include_nb=False,
            min_history_weeks=60,
            origin_step=(
                args.origin_step
            ),
        )
    )

    if results.empty:
        raise RuntimeError(
            "Multi-horizon evaluation produced no results."
        )

    # --------------------------------------------------------
    # FULL-HISTORY DIAGNOSTICS
    # --------------------------------------------------------

    summary = (
        summarize_multi_horizon_backtest(
            results
        )
    )

    print_overall_summary(
        summary
    )

    # --------------------------------------------------------
    # LOCK PRODUCTION POLICY BEFORE HOLDOUT
    # --------------------------------------------------------

    policy = (
        build_horizon_policy(
            results=results,
            selection_weeks=(
                args.selection_weeks
            ),
            holdout_weeks=(
                args.holdout_weeks
            ),
            min_skill_vs_naive=(
                args.min_skill
            ),
        )
    )

    if policy.empty:
        raise RuntimeError(
            "Horizon policy produced no results."
        )

    print_policy(
        policy
    )

    # --------------------------------------------------------
    # DIAGNOSTIC HOLDOUT COMPARISON
    # --------------------------------------------------------

    comparison = (
        build_horizon_holdout_comparison(
            results=results,
            policy=policy,
        )
    )

    if comparison.empty:
        raise RuntimeError(
            "Holdout comparison produced no results."
        )

    print_holdout_comparison(
        comparison
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        results=results,
        summary=summary,
        policy=policy,
        comparison=comparison,
        uf_code=args.uf_code,
    )

    print(
        "\nEvaluation complete.\n"
    )


if __name__ == "__main__":
    main()
