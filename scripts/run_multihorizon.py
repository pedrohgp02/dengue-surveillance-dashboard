"""Run multi-horizon evaluation on real dengue surveillance data."""

import argparse
from pathlib import Path

from src.config import TARGET_UF_CODE, UF_OPTIONS
from src.data import build_weekly_series
from src.multihorizon import (
    run_multi_horizon_backtest,
    summarize_multi_horizon_backtest,
)


HORIZONS = (1, 2, 4, 8, 12)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dengue forecasts at multiple horizons."
    )

    parser.add_argument(
        "--uf-code",
        type=int,
        default=TARGET_UF_CODE,
        help="IBGE state code.",
    )

    parser.add_argument(
        "--origin-step",
        type=int,
        default=4,
        help=(
            "Spacing between historical forecast origins. "
            "Use 4 for a faster evaluation or 1 for every week."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload source data before evaluation.",
    )

    args = parser.parse_args()

    uf_label = UF_OPTIONS.get(
        args.uf_code,
        f"UF {args.uf_code}",
    )

    print(
        f"\nLoading dengue data for {uf_label}..."
    )

    _, _, history = build_weekly_series(
        refresh=args.refresh,
        uf_code=args.uf_code,
    )

    print(
        f"Loaded {len(history)} weekly observations "
        f"from {history['date'].min().date()} "
        f"to {history['date'].max().date()}."
    )

    print(
        "\nRunning historical multi-horizon evaluation..."
    )

    results = run_multi_horizon_backtest(
        history_df=history,
        horizons=HORIZONS,
        include_nb=False,
        min_history_weeks=60,
        origin_step=args.origin_step,
    )

    summary = summarize_multi_horizon_backtest(
        results
    )

    print("\nResults:\n")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    output_dir = Path("results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / f"multihorizon_{args.uf_code}_raw.csv"
    )

    summary_path = (
        output_dir
        / f"multihorizon_{args.uf_code}_summary.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nSaved raw results to {results_path}"
    )

    print(
        f"Saved summary to {summary_path}"
    )


if __name__ == "__main__":
    main()
