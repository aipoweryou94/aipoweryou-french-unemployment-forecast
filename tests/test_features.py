from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aipoweryou_forecast.features import (
    build_feature_frame,
    monthly_to_quarterly,
    quarterly_inflation_from_cpi,
    stitch_rebased_indices,
)
from aipoweryou_forecast.open_data import OpenDataError


def test_stitch_rebased_indices_uses_overlap_ratio() -> None:
    old = pd.DataFrame(
        {"period": ["2025-01", "2025-02", "2025-03"], "cpi_index": [120.0, 121.2, 122.4]}
    )
    new = pd.DataFrame(
        {
            "period": ["2025-01", "2025-02", "2025-03", "2025-04"],
            "cpi_index": [100.0, 101.0, 102.0, 103.0],
        }
    )

    linked = stitch_rebased_indices(old, new)

    assert linked.iloc[-1]["period"] == "2025-04"
    assert linked.iloc[-1]["cpi_index"] == pytest.approx(123.6)


def test_monthly_to_quarterly_drops_incomplete_quarter() -> None:
    monthly = pd.DataFrame(
        {
            "period": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05"],
            "business_climate": [98.0, 99.0, 100.0, 101.0, 102.0],
        }
    )

    quarterly = monthly_to_quarterly(monthly, value_column="business_climate")

    assert quarterly.to_dict("records") == [{"period": "2025-Q1", "business_climate": 99.0}]


def test_monthly_to_quarterly_supports_last_and_rejects_unknown_aggregation() -> None:
    monthly = pd.DataFrame(
        {
            "period": ["2025-01", "2025-02", "2025-03"],
            "value": [10.0, 11.0, 12.0],
        }
    )

    quarterly = monthly_to_quarterly(
        monthly,
        value_column="value",
        output_column="quarter_end",
        aggregation="last",
    )

    assert quarterly.iloc[0].to_dict() == {"period": "2025-Q1", "quarter_end": 12.0}
    with pytest.raises(ValueError, match="Unsupported aggregation"):
        monthly_to_quarterly(monthly, value_column="value", aggregation="median")


def test_quarterly_inflation_uses_only_information_available_in_quarter() -> None:
    periods = pd.period_range("2024-01", "2025-06", freq="M").astype(str)
    values = np.r_[np.repeat(100.0, 12), np.repeat(102.0, 6)]
    cpi = pd.DataFrame({"period": periods, "cpi_index": values})

    inflation = quarterly_inflation_from_cpi(cpi)

    assert inflation["period"].tolist() == ["2025-Q1", "2025-Q2"]
    assert inflation["inflation_yoy"].tolist() == pytest.approx([2.0, 2.0])


def test_feature_frame_inner_joins_core_and_keeps_optional_missing() -> None:
    quarters = [f"{year}-Q{quarter}" for year in range(2010, 2025) for quarter in range(1, 5)]
    core = {
        "unemployment_rate": 8.0,
        "gdp_qoq": 0.2,
        "employment_qoq": 0.1,
        "inflation_yoy": 1.8,
        "business_climate": 100.0,
    }
    sources = {
        name: pd.DataFrame({"period": quarters, name: [value] * len(quarters)})
        for name, value in core.items()
    }
    sources["job_vacancy_rate"] = pd.DataFrame(
        {"period": quarters[-8:], "job_vacancy_rate": [2.1] * 8}
    )

    frame = build_feature_frame(sources)

    assert len(frame) == len(quarters)
    assert frame["job_vacancy_rate"].isna().sum() == len(quarters) - 8
    assert frame["job_vacancy_rate_available"].sum() == 8


def test_feature_frame_rejects_too_short_history() -> None:
    quarters = ["2025-Q1", "2025-Q2"]
    sources = {
        name: pd.DataFrame({"period": quarters, name: [1.0, 1.1]})
        for name in [
            "unemployment_rate",
            "gdp_qoq",
            "employment_qoq",
            "inflation_yoy",
            "business_climate",
        ]
    }

    with pytest.raises(OpenDataError, match="minimum"):
        build_feature_frame(sources)


def test_feature_validation_rejects_missing_or_malformed_series() -> None:
    with pytest.raises(OpenDataError, match="Missing core series"):
        build_feature_frame({})

    missing_value_column = pd.DataFrame({"period": ["2025-01"], "other": [1.0]})
    with pytest.raises(OpenDataError, match="Expected columns"):
        monthly_to_quarterly(missing_value_column, value_column="value")

    duplicate_month = pd.DataFrame({"period": ["2025-01", "2025-01"], "value": [1.0, 2.0]})
    with pytest.raises(OpenDataError, match="Duplicate periods"):
        monthly_to_quarterly(duplicate_month, value_column="value")
