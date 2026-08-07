"""Transform official series into a leakage-safe quarterly feature table."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .open_data import OpenDataError, normalize_period

CORE_FEATURES = [
    "unemployment_rate",
    "gdp_qoq",
    "employment_qoq",
    "inflation_yoy",
    "business_climate",
]


def stitch_rebased_indices(
    old: pd.DataFrame,
    new: pd.DataFrame,
    *,
    value_column: str = "cpi_index",
) -> pd.DataFrame:
    """Link two price indices using their median ratio over the overlap."""
    old_series = _as_series(old, value_column)
    new_series = _as_series(new, value_column)
    overlap = old_series.index.intersection(new_series.index)
    if len(overlap) < 3:
        raise OpenDataError("At least three overlapping months are required to link CPI bases")
    ratio = (
        (old_series.loc[overlap] / new_series.loc[overlap])
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if ratio.empty:
        raise OpenDataError("CPI overlap does not contain valid ratios")
    linked_new = new_series * float(ratio.median())
    linked = old_series.combine_first(linked_new).sort_index()
    # Once the old series stops, continue with the rescaled new base.
    linked.loc[linked_new.index[linked_new.index > old_series.index.max()]] = linked_new.loc[
        linked_new.index[linked_new.index > old_series.index.max()]
    ]
    return linked.rename_axis("period").rename(value_column).reset_index()


def monthly_to_quarterly(
    frame: pd.DataFrame,
    *,
    value_column: str,
    output_column: str | None = None,
    aggregation: str = "mean",
    require_complete_quarter: bool = True,
) -> pd.DataFrame:
    """Aggregate monthly observations without using an incomplete quarter."""
    series = _as_series(frame, value_column)
    monthly_index = pd.PeriodIndex(series.index, freq="M")
    quarterly_index = monthly_index.asfreq("Q")
    values = pd.Series(series.to_numpy(float), index=quarterly_index)
    grouped = values.groupby(level=0)
    if aggregation == "mean":
        result = grouped.mean()
    elif aggregation == "last":
        result = grouped.last()
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    if require_complete_quarter:
        result = result[grouped.count() == 3]
    result.index = [_format_quarter(period) for period in result.index]
    return result.rename_axis("period").rename(output_column or value_column).reset_index()


def quarterly_inflation_from_cpi(cpi: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly year-on-year CPI inflation, then average complete quarters."""
    series = _as_series(cpi, "cpi_index")
    inflation = series.pct_change(12, fill_method=None) * 100.0
    monthly = inflation.dropna().rename_axis("period").rename("inflation_yoy").reset_index()
    return monthly_to_quarterly(monthly, value_column="inflation_yoy")


def build_feature_frame(
    series: Mapping[str, pd.DataFrame],
    *,
    optional_columns: tuple[str, ...] = ("job_vacancy_rate",),
    minimum_rows: int = 40,
) -> pd.DataFrame:
    """Join core features on common quarters and retain optional sparse signals."""
    missing = [column for column in CORE_FEATURES if column not in series]
    if missing:
        raise OpenDataError(f"Missing core series: {', '.join(missing)}")

    frame: pd.DataFrame | None = None
    for column in CORE_FEATURES:
        current = _clean_quarterly(series[column], column)
        frame = (
            current
            if frame is None
            else frame.merge(current, on="period", how="inner", validate="one_to_one")
        )
    assert frame is not None

    for column in optional_columns:
        if column in series:
            current = _clean_quarterly(series[column], column)
            frame = frame.merge(current, on="period", how="left", validate="one_to_one")
            frame[f"{column}_available"] = frame[column].notna().astype("int8")

    frame = frame.sort_values(
        "period", key=lambda values: values.map(_quarter_sort_key)
    ).reset_index(drop=True)
    if frame["period"].duplicated().any():
        raise OpenDataError("Duplicate quarters in model dataset")
    if frame[CORE_FEATURES].isna().any().any():
        raise OpenDataError("Core feature table contains missing values")
    if len(frame) < minimum_rows:
        raise OpenDataError(f"Only {len(frame)} complete quarters; minimum is {minimum_rows}")
    return frame


def _as_series(frame: pd.DataFrame, value_column: str) -> pd.Series:
    if not {"period", value_column}.issubset(frame.columns):
        raise OpenDataError(f"Expected columns period and {value_column}")
    clean = frame[["period", value_column]].dropna().copy()
    clean["period"] = clean["period"].map(normalize_period)
    if clean["period"].duplicated().any():
        raise OpenDataError(f"Duplicate periods in {value_column}")
    return clean.set_index("period")[value_column].astype(float).sort_index()


def _clean_quarterly(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    clean = frame[["period", column]].dropna().copy()
    clean["period"] = clean["period"].map(normalize_period)
    if not clean["period"].str.fullmatch(r"\d{4}-Q[1-4]").all():
        raise OpenDataError(f"Non-quarterly period found in {column}")
    return clean.drop_duplicates("period", keep="last")


def _format_quarter(period: pd.Period) -> str:
    return f"{period.year}-Q{period.quarter}"


def _quarter_sort_key(period: str) -> tuple[int, int]:
    year, quarter = normalize_period(period).split("-Q")
    return int(year), int(quarter)
