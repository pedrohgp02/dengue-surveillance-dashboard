"""Public API for the dengue surveillance package."""

from src.config import (
    TARGET_LABEL,
    TARGET_UF_CODE,
    TREND_THRESHOLD,
    UF_OPTIONS,
)

from src.data import (
    build_weekly_series,
    cache_exists,
    cache_is_fresh,
    get_available_data_urls,
    get_cache_info,
)

from src.evaluation import score_predictions

from src.pipeline import build_monitoring_state


__all__ = [
    "TARGET_LABEL",
    "TARGET_UF_CODE",
    "TREND_THRESHOLD",
    "UF_OPTIONS",
    "build_weekly_series",
    "cache_exists",
    "cache_is_fresh",
    "get_available_data_urls",
    "get_cache_info",
    "score_predictions",
    "build_monitoring_state",
]
