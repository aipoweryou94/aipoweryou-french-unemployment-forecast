"""Unit tests for temporal preparation, independent from PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aipoweryou_forecast.modeling import (
    FEATURE_COLUMNS,
    build_window,
    make_supervised_arrays,
    next_periods,
    validate_model_frame,
)


def _sample_values(rows: int = 20) -> np.ndarray:
    """Build a deterministic five-feature sample for shape and leakage tests."""
    time = np.arange(rows, dtype=float)
    return np.column_stack(
        [
            7.0 + time * 0.1,
            np.sin(time),
            np.cos(time),
            2.0 + time * 0.02,
            100.0 - time * 0.1,
        ]
    )


def test_supervised_arrays_have_expected_shapes_and_targets() -> None:
    values = _sample_values()

    x, y, scale = make_supervised_arrays(values, window=8, horizon=4)

    assert x.shape == (9, 8, 6)
    assert y.shape == (9, 4)
    expected = (values[8:12, 0] - values[7, 0]) / scale.target_diff_std
    assert y[0] == pytest.approx(expected)


def test_build_window_appends_monotonic_position_signal() -> None:
    values = _sample_values()
    _, _, scale = make_supervised_arrays(values, window=8, horizon=4)

    window = build_window(values, 2, 10, scale)

    assert window[:, -1] == pytest.approx(np.linspace(-1.0, 1.0, 8))


def test_validate_model_frame_rejects_missing_feature() -> None:
    frame = pd.DataFrame({"period": ["2025-Q1"] * 20})

    with pytest.raises(ValueError, match="Missing model columns"):
        validate_model_frame(frame, window=8, horizon=4)


def test_validate_model_frame_accepts_complete_history() -> None:
    values = _sample_values()
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    frame.insert(0, "period", [f"{2020 + index // 4}-Q{index % 4 + 1}" for index in range(20)])

    clean = validate_model_frame(frame, window=8, horizon=4)

    assert len(clean) == 20


@pytest.mark.parametrize(
    ("window", "horizon", "message"),
    [
        (1, 4, "at least two quarters"),
        (8, 0, "strictly positive"),
        (18, 4, "at least 22"),
    ],
)
def test_validate_model_frame_rejects_invalid_dimensions(
    window: int,
    horizon: int,
    message: str,
) -> None:
    values = _sample_values()
    frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    frame.insert(0, "period", [f"{2020 + index // 4}-Q{index % 4 + 1}" for index in range(20)])

    with pytest.raises(ValueError, match=message):
        validate_model_frame(frame, window=window, horizon=horizon)


def test_temporal_helpers_reject_malformed_inputs() -> None:
    values = _sample_values()
    _, _, scale = make_supervised_arrays(values, window=8, horizon=4)

    with pytest.raises(ValueError, match="Invalid window boundaries"):
        build_window(values, -1, 8, scale)
    with pytest.raises(ValueError, match="window must be"):
        make_supervised_arrays(values, window=1, horizon=4)
    with pytest.raises(ValueError, match="Not enough observations"):
        make_supervised_arrays(values, window=18, horizon=4)
    with pytest.raises(ValueError, match="Expected a 2D matrix"):
        make_supervised_arrays(np.ones((20, 4)), window=8, horizon=4)
    with pytest.raises(ValueError, match="finite observations"):
        make_supervised_arrays(np.full((20, 5), np.nan), window=8, horizon=4)


@pytest.mark.parametrize(
    ("last_period", "expected"),
    [
        ("2026-Q2", ["2026-Q3", "2026-Q4", "2027-Q1", "2027-Q2"]),
        ("2026-T4", ["2027-Q1", "2027-Q2"]),
    ],
)
def test_next_periods_crosses_year_boundary(
    last_period: str,
    expected: list[str],
) -> None:
    assert next_periods(last_period, len(expected)) == expected


def test_next_periods_rejects_invalid_quarter() -> None:
    with pytest.raises(ValueError, match="Invalid quarterly period"):
        next_periods("2026-Q5", 4)

    with pytest.raises(ValueError, match="Invalid quarterly period"):
        next_periods("not-a-quarter", 4)

    with pytest.raises(ValueError, match="strictly positive"):
        next_periods("2026-Q2", 0)
