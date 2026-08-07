"""Pure NumPy helpers used by the temporal forecasting model.

Keeping data preparation outside the PyTorch training script makes the most
sensitive temporal logic easy to unit-test without downloading a deep-learning
runtime. The neural network remains implemented in ``train_transformer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "unemployment_rate",
    "gdp_qoq",
    "employment_qoq",
    "inflation_yoy",
    "business_climate",
)


@dataclass(frozen=True)
class Scale:
    """Scaling parameters learned exclusively from the historical sample."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_diff_std: float


def validate_model_frame(
    frame: pd.DataFrame,
    *,
    window: int,
    horizon: int,
) -> pd.DataFrame:
    """Validate and return the complete rows consumed by the model.

    The function deliberately drops incomplete optional rows but refuses a
    dataset that is too short for one complete input/output training example.
    """
    required_columns = ["period", *FEATURE_COLUMNS]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing model columns: {', '.join(missing)}. Run fetch_open_data.py first."
        )
    if window < 2:
        raise ValueError("The input window must contain at least two quarters")
    if horizon < 1:
        raise ValueError("The forecast horizon must be strictly positive")

    clean = frame.dropna(subset=required_columns).reset_index(drop=True)
    if len(clean) < window + horizon:
        raise ValueError(
            f"Only {len(clean)} complete rows; at least {window + horizon} are required"
        )

    values = clean.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The model dataset contains non-finite values")
    return clean


def compute_scale(values: np.ndarray) -> Scale:
    """Estimate stable standardisation parameters for multivariate windows."""
    _validate_values(values)
    target_changes = np.diff(values[:, 0])
    return Scale(
        feature_mean=values.mean(axis=0),
        feature_std=values.std(axis=0) + 1e-8,
        target_diff_std=float(target_changes.std() + 1e-8),
    )


def build_window(
    values: np.ndarray,
    start: int,
    end: int,
    scale: Scale,
) -> np.ndarray:
    """Standardise one historical window and append a relative time position."""
    if start < 0 or end > len(values) or start >= end:
        raise ValueError("Invalid window boundaries")

    standardised = (values[start:end] - scale.feature_mean) / scale.feature_std
    # The explicit position signal preserves order without relying on a date ID.
    position = np.linspace(-1.0, 1.0, end - start)
    return np.column_stack([standardised, position]).astype("float32")


def make_supervised_arrays(
    values: np.ndarray,
    window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, Scale]:
    """Create leakage-safe sliding windows and direct multi-horizon targets."""
    _validate_values(values)
    if window < 2 or horizon < 1:
        raise ValueError("window must be >= 2 and horizon must be >= 1")
    if len(values) < window + horizon:
        raise ValueError("Not enough observations for the requested window and horizon")

    scale = compute_scale(values)
    target = values[:, 0]
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for end in range(window, len(values) - horizon + 1):
        features.append(build_window(values, end - window, end, scale))
        # Each target is a movement from the last observation in its input window.
        movement = target[end : end + horizon] - target[end - 1]
        targets.append(movement / scale.target_diff_std)

    return (
        np.asarray(features, dtype="float32"),
        np.asarray(targets, dtype="float32"),
        scale,
    )


def next_periods(last_period: str, count: int) -> list[str]:
    """Return the next quarterly labels in the canonical ``YYYY-QN`` format."""
    if count < 1:
        raise ValueError("count must be strictly positive")

    normalized = last_period.replace("-T", "-Q")
    try:
        year_text, quarter_text = normalized.split("-Q")
        year, quarter = int(year_text), int(quarter_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid quarterly period: {last_period}") from exc
    if quarter not in {1, 2, 3, 4}:
        raise ValueError(f"Invalid quarterly period: {last_period}")

    periods = []
    for _ in range(count):
        quarter += 1
        if quarter == 5:
            quarter = 1
            year += 1
        periods.append(f"{year}-Q{quarter}")
    return periods


def _validate_values(values: np.ndarray) -> None:
    """Reject malformed numerical matrices before training starts."""
    if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected a 2D matrix with {len(FEATURE_COLUMNS)} feature columns")
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("The feature matrix must contain finite observations")
