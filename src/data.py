"""Data loading, caching, and preprocessing for dengue surveillance."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import (
    ANALYSIS_START,
    BASE_DATA_URLS,
    CACHE_DIR,
    CACHE_META_PATH,
    DATA_URL_TEMPLATE,
    RAW_CACHE_PATH,
    TARGET_UF_CODE,
    UF_OPTIONS,
    WEEKLY_CACHE_PATH,
)


# Cache discovered yearly URLs in memory so the application does not
# repeatedly probe the Ministry of Health server during one session.
_AVAILABLE_DATA_URLS_CACHE: dict[int, str] | None = None


def _uf_cache_paths(
    uf_code: int = TARGET_UF_CODE,
) -> dict[str, Path]:
    """Return cache file paths for a specific Brazilian state."""
    return {
        "weekly": CACHE_DIR / f"weekly_{uf_code}.parquet",
        "raw": CACHE_DIR / f"raw_{uf_code}.parquet",
        "meta": CACHE_DIR / f"meta_{uf_code}.json",
    }


def _build_year_url(year: int) -> str:
    """Construct the SINAN dengue CSV URL for a given year."""
    return DATA_URL_TEMPLATE.format(
        year_suffix=str(year)[-2:]
    )


def _url_exists(
    url: str,
    timeout: int = 8,
) -> bool:
    """Check whether a remote dataset URL is reachable.

    A lightweight HEAD request is attempted first. Some servers reject
    HEAD requests, so HTTP 403 or 405 triggers a streaming GET request
    as a fallback.
    """
    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
        )

        if response.ok:
            return True

        if response.status_code in {403, 405}:
            response = requests.get(
                url,
                stream=True,
                timeout=timeout,
            )

            is_available = response.ok
            response.close()

            return is_available

        return False

    except requests.RequestException:
        return False


def get_available_data_urls(
    refresh: bool = False,
) -> dict[int, str]:
    """Return available yearly SINAN dengue dataset URLs.

    Known URLs from the project configuration are used first. The
    function then probes for any newer annual datasets through the next
    calendar year.

    Results are cached in memory for the duration of the Python process.
    """
    global _AVAILABLE_DATA_URLS_CACHE

    if (
        _AVAILABLE_DATA_URLS_CACHE is not None
        and not refresh
    ):
        return dict(_AVAILABLE_DATA_URLS_CACHE)

    detected_urls = dict(BASE_DATA_URLS)

    latest_known_year = max(detected_urls)

    max_probe_year = (
        pd.Timestamp.today().year + 1
    )

    for year in range(
        latest_known_year + 1,
        max_probe_year + 1,
    ):
        candidate_url = _build_year_url(year)

        if _url_exists(candidate_url):
            detected_urls[year] = candidate_url

    _AVAILABLE_DATA_URLS_CACHE = dict(
        sorted(detected_urls.items())
    )

    return dict(_AVAILABLE_DATA_URLS_CACHE)


def _ensure_cache_dir() -> None:
    """Create the local data-cache directory if necessary."""
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def _save_cache(
    raw_all: pd.DataFrame,
    full_df: pd.DataFrame,
    df: pd.DataFrame,
    uf_code: int = TARGET_UF_CODE,
    raw_count: int | None = None,
) -> None:
    """Persist processed dengue data and cache metadata.

    Weekly data are stored as Parquet for fast subsequent startup.
    Metadata include cache timestamps, row counts, geographic coverage,
    and available years.
    """
    _ensure_cache_dir()

    paths = _uf_cache_paths(uf_code)

    full_df.to_parquet(
        paths["weekly"],
        index=False,
    )

    # Legacy support: raw rows are currently not retained by the
    # memory-optimized ingestion pipeline, but keep this behavior in
    # case raw data are supplied in the future.
    raw_cols = [
        "DT_SIN_PRI",
        "SEM_PRI",
        "SG_UF",
        "ID_MN_RESI",
        "CLASSI_FIN",
        "week_start",
        "notifications",
    ]

    save_cols = [
        column
        for column in raw_cols
        if column in raw_all.columns
    ]

    if (
        save_cols
        and len(raw_all) > 0
    ):
        raw_all[
            save_cols
        ].to_parquet(
            paths["raw"],
            index=False,
        )

    n_raw = (
        int(raw_count)
        if raw_count is not None
        else int(len(raw_all))
    )

    metadata: dict[str, Any] = {
        "created_utc": (
            pd.Timestamp.utcnow().isoformat()
        ),
        "created_local": (
            pd.Timestamp.now().isoformat()
        ),
        "latest_date": str(
            df["date"].max().date()
        ),
        "n_weeks": int(len(df)),
        "n_raw": n_raw,
        "years": sorted(
            full_df["year"]
            .unique()
            .tolist()
        ),
        "uf_code": uf_code,
        "uf_label": UF_OPTIONS.get(
            uf_code,
            f"UF {uf_code}",
        ),
    }

    with paths["meta"].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def _load_cache(
    uf_code: int = TARGET_UF_CODE,
) -> (
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]
    | None
):
    """Load locally cached dengue data.

    Returns
    -------
    tuple or None
        ``(raw_all, full_df, analysis_df)`` when a usable cache exists,
        otherwise ``None``.
    """
    paths = _uf_cache_paths(uf_code)

    weekly_path = paths["weekly"]
    raw_path = paths["raw"]

    # Backward compatibility with the original single-state cache.
    if (
        not weekly_path.exists()
        and uf_code == TARGET_UF_CODE
        and WEEKLY_CACHE_PATH.exists()
    ):
        weekly_path = WEEKLY_CACHE_PATH
        raw_path = RAW_CACHE_PATH

    if not weekly_path.exists():
        return None

    try:
        full_df = pd.read_parquet(
            weekly_path
        )

        full_df["date"] = pd.to_datetime(
            full_df["date"]
        )

        if raw_path.exists():
            raw_all = pd.read_parquet(
                raw_path
            )
        else:
            raw_all = pd.DataFrame()

        if "DT_SIN_PRI" in raw_all.columns:
            raw_all[
                "DT_SIN_PRI"
            ] = pd.to_datetime(
                raw_all["DT_SIN_PRI"],
                errors="coerce",
            )

        analysis_df = (
            full_df.loc[
                full_df["date"]
                >= ANALYSIS_START
            ]
            .copy()
            .reset_index(drop=True)
        )

        return (
            raw_all,
            full_df,
            analysis_df,
        )

    except Exception:
        # A corrupt or incompatible cache should not prevent the
        # application from rebuilding the data from source.
        return None


def cache_exists(
    uf_code: int = TARGET_UF_CODE,
) -> bool:
    """Return whether a weekly cache exists for a state."""
    paths = _uf_cache_paths(uf_code)

    if paths["weekly"].exists():
        return True

    if (
        uf_code == TARGET_UF_CODE
        and WEEKLY_CACHE_PATH.exists()
    ):
        return True

    return False


def cache_is_fresh(
    max_age_hours: float = 12.0,
    uf_code: int = TARGET_UF_CODE,
) -> bool:
    """Return whether a state's cache is younger than a given age."""
    paths = _uf_cache_paths(uf_code)

    meta_path = paths["meta"]

    # Backward compatibility with the original cache metadata.
    if (
        not meta_path.exists()
        and uf_code == TARGET_UF_CODE
        and CACHE_META_PATH.exists()
    ):
        meta_path = CACHE_META_PATH

    if not meta_path.exists():
        return False

    try:
        with meta_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        created = pd.Timestamp(
            metadata["created_utc"]
        )

        age_hours = (
            (
                pd.Timestamp.utcnow()
                - created
            ).total_seconds()
            / 3600
        )

        return age_hours < max_age_hours

    except Exception:
        return False


def get_cache_info(
    uf_code: int = TARGET_UF_CODE,
) -> dict[str, Any]:
    """Return metadata for a state's current cache."""
    paths = _uf_cache_paths(uf_code)

    meta_path = paths["meta"]

    if (
        not meta_path.exists()
        and uf_code == TARGET_UF_CODE
        and CACHE_META_PATH.exists()
    ):
        meta_path = CACHE_META_PATH

    if not meta_path.exists():
        return {}

    try:
        with meta_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception:
        return {}


def load_dengue_year(
    year: int,
    uf_code: int = TARGET_UF_CODE,
) -> tuple[int, pd.DataFrame]:
    """Download and aggregate one year of SINAN dengue notifications.

    The source CSV is processed in chunks to reduce memory use. Only
    columns required by the surveillance application are loaded.

    Parameters
    ----------
    year:
        Calendar year of the SINAN dataset.
    uf_code:
        IBGE state code used to filter notifications.

    Returns
    -------
    tuple[int, pd.DataFrame]
        Number of raw notifications retained for the selected state and
        a DataFrame containing aggregated weekly counts.
    """
    data_urls = get_available_data_urls()

    if year not in data_urls:
        raise ValueError(
            f"No dengue dataset is available for {year}."
        )

    usecols = [
        "DT_SIN_PRI",
        "SEM_PRI",
        "SG_UF",
        "ID_MN_RESI",
        "CLASSI_FIN",
    ]

    raw_count = 0

    weekly_parts: list[
        pd.DataFrame
    ] = []

    for chunk in pd.read_csv(
        data_urls[year],
        compression="zip",
        encoding="latin1",
        usecols=usecols,
        low_memory=False,
        chunksize=250_000,
    ):
        chunk["DT_SIN_PRI"] = pd.to_datetime(
            chunk["DT_SIN_PRI"],
            errors="coerce",
        )

        chunk = chunk.dropna(
            subset=["DT_SIN_PRI"]
        )

        chunk = chunk.loc[
            chunk["SG_UF"] == uf_code
        ].copy()

        if chunk.empty:
            continue

        raw_count += int(len(chunk))

        # Align each notification to Monday so every observation is
        # grouped into a consistent weekly bucket.
        chunk["week_start"] = (
            chunk["DT_SIN_PRI"]
            - pd.to_timedelta(
                chunk[
                    "DT_SIN_PRI"
                ].dt.weekday,
                unit="D",
            )
        )

        chunk["notifications"] = 1

        weekly_part = (
            chunk.groupby(
                "week_start",
                as_index=False,
            )["notifications"]
            .sum()
            .sort_values(
                "week_start"
            )
        )

        weekly_parts.append(
            weekly_part
        )

    if weekly_parts:
        weekly = (
            pd.concat(
                weekly_parts,
                ignore_index=True,
            )
            .groupby(
                "week_start",
                as_index=False,
            )["notifications"]
            .sum()
            .sort_values(
                "week_start"
            )
            .reset_index(
                drop=True
            )
        )

    else:
        weekly = pd.DataFrame(
            columns=[
                "week_start",
                "notifications",
            ]
        )

    weekly["source_year"] = year

    return raw_count, weekly


def build_weekly_series(
    refresh: bool = False,
    uf_code: int = TARGET_UF_CODE,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build the complete weekly dengue-notification time series.

    The function returns:

    1. a legacy raw-data frame,
    2. the complete weekly series,
    3. the analysis series beginning at ``ANALYSIS_START``.

    Cached Parquet data are preferred on normal startup. Passing
    ``refresh=True`` forces the source datasets to be downloaded again.
    """
    if (
        not refresh
        and cache_exists(
            uf_code=uf_code
        )
    ):
        cached = _load_cache(
            uf_code=uf_code
        )

        if cached is not None:
            return cached

    data_urls = get_available_data_urls(
        refresh=refresh
    )

    if not data_urls:
        raise RuntimeError(
            "No SINAN dengue datasets could be located."
        )

    min_required_year = max(
        min(data_urls),
        int(ANALYSIS_START.year) - 1,
    )

    total_raw_count = 0

    weekly_parts: list[
        pd.DataFrame
    ] = []

    for year in sorted(data_urls):
        if year < min_required_year:
            continue

        year_raw_count, weekly_df = (
            load_dengue_year(
                year,
                uf_code,
            )
        )

        total_raw_count += int(
            year_raw_count
        )

        if not weekly_df.empty:
            weekly_parts.append(
                weekly_df
            )

    if not weekly_parts:
        raise RuntimeError(
            "No dengue notifications were found "
            f"for UF code {uf_code}."
        )

    # Raw rows are intentionally not retained. This keeps memory use
    # much lower while preserving the interface expected by the app.
    raw_all = pd.DataFrame(
        columns=[
            "DT_SIN_PRI",
            "SEM_PRI",
            "SG_UF",
            "ID_MN_RESI",
            "CLASSI_FIN",
            "week_start",
            "notifications",
        ]
    )

    weekly_all = (
        pd.concat(
            weekly_parts,
            ignore_index=True,
        )
        .groupby(
            "week_start",
            as_index=False,
        )["notifications"]
        .sum()
        .rename(
            columns={
                "week_start": "date"
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Construct a continuous Monday-based index so missing calendar
    # weeks remain explicit rather than disappearing from the series.
    full_weeks = pd.DataFrame(
        {
            "date": pd.date_range(
                weekly_all["date"].min(),
                weekly_all["date"].max(),
                freq="W-MON",
            )
        }
    )

    analysis = full_weeks.merge(
        weekly_all,
        on="date",
        how="left",
    )

    analysis[
        "is_imputed_zero_week"
    ] = analysis[
        "notifications"
    ].isna()

    analysis[
        "notifications"
    ] = (
        analysis[
            "notifications"
        ]
        .fillna(0)
        .astype(int)
    )

    analysis["year"] = (
        analysis["date"].dt.year
    )

    analysis["epi_week"] = (
        analysis["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    full_df = analysis.copy()

    analysis_df = (
        analysis.loc[
            analysis["date"]
            >= ANALYSIS_START
        ]
        .copy()
        .reset_index(drop=True)
    )

    _save_cache(
        raw_all=raw_all,
        full_df=full_df,
        df=analysis_df,
        uf_code=uf_code,
        raw_count=total_raw_count,
    )

    return (
        raw_all,
        full_df,
        analysis_df,
    )
