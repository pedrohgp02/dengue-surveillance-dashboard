"""Configuration for the dengue surveillance pipeline."""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = PROJECT_ROOT / ".data_cache"

# Legacy cache files kept temporarily for backward compatibility.
WEEKLY_CACHE_PATH = CACHE_DIR / "weekly_notifications.parquet"
RAW_CACHE_PATH = CACHE_DIR / "raw_notifications.parquet"
CACHE_META_PATH = CACHE_DIR / "cache_meta.json"


# ---------------------------------------------------------------------
# Ministry of Health data sources
# ---------------------------------------------------------------------

BASE_DATA_URLS = {
    2021: (
        "https://s3.sa-east-1.amazonaws.com/"
        "ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR21.csv.zip"
    ),
    2022: (
        "https://s3.sa-east-1.amazonaws.com/"
        "ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR22.csv.zip"
    ),
    2023: (
        "https://s3.sa-east-1.amazonaws.com/"
        "ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR23.csv.zip"
    ),
    2024: (
        "https://s3.sa-east-1.amazonaws.com/"
        "ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR24.csv.zip"
    ),
    2025: (
        "https://s3.sa-east-1.amazonaws.com/"
        "ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR25.csv.zip"
    ),
    2026: (
        "https://s3.sa-east-1.amazonaws.com/"
        "ckan.saude.gov.br/SINAN/Dengue/csv/DENGBR26.csv.zip"
    ),
}

DATA_URL_TEMPLATE = (
    "https://s3.sa-east-1.amazonaws.com/"
    "ckan.saude.gov.br/SINAN/Dengue/csv/"
    "DENGBR{year_suffix}.csv.zip"
)

DATA_SOURCE_LABEL = "SINAN/Dengue — Brazilian Ministry of Health"

DATA_SOURCE_API_URL = (
    "https://apidadosabertos.saude.gov.br/arboviroses/dengue"
)

DATA_SOURCE_PORTAL_URL = (
    "https://dadosabertos.saude.gov.br/dataset/"
    "arboviroses-dengue/resource/"
    "a9b73910-f233-417b-85c9-95230c269e1c"
)


# ---------------------------------------------------------------------
# Geographic configuration
# ---------------------------------------------------------------------

UF_OPTIONS = {
    32: "Espírito Santo",
    33: "Rio de Janeiro",
    31: "Minas Gerais",
    35: "São Paulo",
    29: "Bahia",
}

TARGET_UF_CODE = 32
TARGET_LABEL = "Espírito Santo"


# ---------------------------------------------------------------------
# Modeling configuration
# ---------------------------------------------------------------------

ANALYSIS_START = pd.Timestamp("2023-01-02")

FEATURE_COLS = [
    "lag1",
    "lag4",
    "week_sin",
    "week_cos",
]

MIN_BACKTEST_TRAIN = 26

TREND_THRESHOLD = 0.10

BAND_QUANTILES = (0.25, 0.75)

DEFAULT_HOLDOUT_WEEKS = 52

DEFAULT_SELECTION_WEEKS = 8
