"""Model training and prediction utilities for dengue forecasting."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import FEATURE_COLS


ModelBundle = dict[str, dict[str, Any]]


def fit_negative_binomial_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Any:
    """Fit a Negative Binomial regression model for count data.

    Negative Binomial regression is useful for dengue notification
    counts because count data may exhibit overdispersion.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target notification counts.

    Returns
    -------
    Any
        Fitted statsmodels Negative Binomial result.
    """
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial

    X_with_constant = sm.add_constant(
        X,
        has_constant="add",
    )

    model = NegativeBinomial(
        y,
        X_with_constant,
    )

    return model.fit(
        disp=0,
        maxiter=200,
    )


def fit_model_bundle(
    feature_frame: pd.DataFrame,
    include_nb: bool = True,
    verbose: bool = False,
) -> ModelBundle:
    """Train all forecasting models.

    The bundle includes two baseline methods plus Linear Regression,
    Random Forest, and optionally Negative Binomial regression.

    Parameters
    ----------
    feature_frame:
        DataFrame containing model features and notification counts.
    include_nb:
        Whether to attempt fitting the Negative Binomial model.
    verbose:
        If True, report why Negative Binomial was omitted when fitting
        fails or produces an unstable result.

    Returns
    -------
    ModelBundle
        Mapping of model names to fitted model metadata.
    """
    X = np.asarray(
        feature_frame[FEATURE_COLS].to_numpy(dtype=float),
        dtype=float,
    )

    y = np.asarray(
        feature_frame["notifications"].to_numpy(dtype=float),
        dtype=int,
    )

    # Baselines require no fitted estimator. Their predictions are
    # generated directly from lag values in predict_with_bundle().
    bundle: ModelBundle = {
        "Naive": {
            "kind": "baseline",
        },
        "Seasonal Naive": {
            "kind": "baseline",
        },
    }

    # Linear regression
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    bundle["Linear Regression"] = {
        "kind": "sklearn",
        "model": linear_model,
    }

    # Conservative Random Forest parameters help limit overfitting
    # during smaller expanding-window training folds.
    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )

    random_forest.fit(X, y)

    bundle["Random Forest"] = {
        "kind": "sklearn",
        "model": random_forest,
    }

    # Negative Binomial is optional because optimization can be
    # unstable on some small or noisy training windows.
    if include_nb:
        try:
            nb_result = fit_negative_binomial_model(X, y)

            nb_alpha = float(nb_result.params[-1])

            converged = bool(
                nb_result.mle_retvals.get(
                    "converged",
                    False,
                )
            )

            finite_standard_errors = np.isfinite(
                np.asarray(nb_result.bse)
            ).all()

            if (
                not np.isfinite(nb_alpha)
                or abs(nb_alpha) > 100
            ):
                raise ValueError(
                    f"unstable alpha estimate ({nb_alpha})"
                )

            if not converged:
                raise ValueError(
                    "optimizer did not converge"
                )

            if not finite_standard_errors:
                raise ValueError(
                    "standard errors are not finite"
                )

            bundle["Negative Binomial"] = {
                "kind": "statsmodels",
                "model": nb_result,
                "alpha": nb_alpha,
            }

        except Exception as error:
            if verbose:
                print(
                    "Negative Binomial model omitted: "
                    f"{error}"
                )

    return bundle


def predict_with_bundle(
    bundle: ModelBundle,
    feature_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Generate predictions from every available model.

    Parameters
    ----------
    bundle:
        Model bundle returned by fit_model_bundle().
    feature_frame:
        DataFrame containing the features required for prediction.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from model name to prediction array.
    """
    X = np.asarray(
        feature_frame[FEATURE_COLS].to_numpy(dtype=float),
        dtype=float,
    )

    predictions: dict[str, np.ndarray] = {
        # Repeat the previous week's count.
        "Naive": np.asarray(
            feature_frame["lag1"].to_numpy(dtype=float),
            dtype=float,
        ),

        # Use the corresponding week from the previous year.
        "Seasonal Naive": np.asarray(
            feature_frame["lag52"].to_numpy(dtype=float),
            dtype=float,
        ),

        "Linear Regression": np.clip(
            bundle["Linear Regression"]["model"].predict(X),
            0,
            None,
        ),

        "Random Forest": np.clip(
            bundle["Random Forest"]["model"].predict(X),
            0,
            None,
        ),
    }

    if "Negative Binomial" in bundle:
        import statsmodels.api as sm

        X_with_constant = sm.add_constant(
            X,
            has_constant="add",
        )

        predictions["Negative Binomial"] = np.clip(
            bundle["Negative Binomial"]["model"].predict(
                X_with_constant
            ),
            0,
            None,
        )

    return predictions
