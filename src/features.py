"""Feature engineering utilities for dengue forecasting."""

import numpy as np
import pandas as pd


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create lag and seasonal features used by the forecasting models.

    Parameters
    ----------
    data:
        Weekly dengue notification data containing at least
        ``date`` and ``notifications`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of the input data with lag and cyclical seasonal
        features added.
    """
    features = data.copy()

    # Recent disease activity
    features["lag1"] = features["notifications"].shift(1)
    features["lag4"] = features["notifications"].shift(4)

    # Same period in the previous year.
    # This is used by the forecasting pipeline as a seasonal reference.
    features["lag52"] = features["notifications"].shift(52)

    # Represent week-of-year cyclically so week 52 and week 1
    # remain close together instead of looking numerically far apart.
    week_number = features["date"].dt.isocalendar().week.astype(int)

    features["week_sin"] = np.sin(
        2 * np.pi * week_number / 52
    )

    features["week_cos"] = np.cos(
        2 * np.pi * week_number / 52
    )

    # lag1 and lag4 are required by every production model.
    return features.dropna(
        subset=["lag1", "lag4"]
    ).reset_index(drop=True)
