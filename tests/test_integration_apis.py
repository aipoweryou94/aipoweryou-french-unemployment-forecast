from __future__ import annotations

import os

import pytest

from aipoweryou_forecast.open_data import EurostatClient, InseeBDMClient

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to call public APIs"
)
def test_live_insee_unemployment_series() -> None:
    frame = InseeBDMClient().fetch_series(
        "001688527", start_period="2025-Q1", name="unemployment_rate"
    )
    assert frame.iloc[-1]["period"] >= "2026-Q1"
    assert 5.0 < frame.iloc[-1]["unemployment_rate"] < 15.0


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to call public APIs"
)
def test_live_eurostat_gdp_series() -> None:
    frame = EurostatClient().fetch_series(
        "namq_10_gdp",
        {"freq": "Q", "unit": "CLV_PCH_PRE", "na_item": "B1GQ", "s_adj": "SCA", "geo": "FR"},
        start_period="2025-Q1",
        name="gdp_qoq",
    )
    assert len(frame) >= 4
    assert frame["gdp_qoq"].between(-20.0, 20.0).all()
