"""Fetch official open data and build the quarterly model feature table."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from aipoweryou_forecast.features import (
    build_feature_frame,
    monthly_to_quarterly,
    quarterly_inflation_from_cpi,
    stitch_rebased_indices,
)
from aipoweryou_forecast.open_data import EurostatClient, InseeBDMClient

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "model_features.csv"

INSEE_SERIES = {
    "unemployment_rate": "001688527",
    "business_climate": "001565530",
    "cpi_base_2015": "001759970",
    "cpi_base_2025": "011814630",
}

EUROSTAT_QUERIES = {
    "gdp_qoq": (
        "namq_10_gdp",
        {"freq": "Q", "unit": "CLV_PCH_PRE", "na_item": "B1GQ", "s_adj": "SCA", "geo": "FR"},
    ),
    "employment_qoq": (
        "namq_10_a10_e",
        {
            "freq": "Q",
            "unit": "PCH_PRE_PER",
            "nace_r2": "TOTAL",
            "s_adj": "SA",
            "na_item": "EMP_DC",
            "geo": "FR",
        },
    ),
    "job_vacancy_rate": (
        "jvs_q_nace2",
        {
            "freq": "Q",
            "s_adj": "SA",
            "nace_r2": "B-S",
            "sizeclas": "TOTAL",
            "indic_em": "JVR",
            "geo": "FR",
        },
    ),
}


def fetch_model_data(
    *,
    start_quarter: str = "2000-Q1",
    output: Path = DEFAULT_OUTPUT,
    include_vacancies: bool = True,
) -> Path:
    insee = InseeBDMClient()
    eurostat = EurostatClient()
    raw_dir = output.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    start_month = start_quarter[:4] + "-01"

    unemployment = insee.fetch_series(
        INSEE_SERIES["unemployment_rate"], start_period=start_quarter, name="unemployment_rate"
    )
    climate_monthly = insee.fetch_series(
        INSEE_SERIES["business_climate"], start_period=start_month, name="business_climate"
    )
    cpi_old = insee.fetch_series(
        INSEE_SERIES["cpi_base_2015"], start_period=start_month, name="cpi_index"
    )
    cpi_new = insee.fetch_series(
        INSEE_SERIES["cpi_base_2025"], start_period="2025-01", name="cpi_index"
    )

    raw_sources = {
        "insee_unemployment": unemployment,
        "insee_business_climate_monthly": climate_monthly,
        "insee_cpi_base_2015": cpi_old,
        "insee_cpi_base_2025": cpi_new,
    }
    model_series = {
        "unemployment_rate": unemployment,
        "business_climate": monthly_to_quarterly(
            climate_monthly, value_column="business_climate", require_complete_quarter=True
        ),
    }

    linked_cpi = stitch_rebased_indices(cpi_old, cpi_new)
    model_series["inflation_yoy"] = quarterly_inflation_from_cpi(linked_cpi)
    raw_sources["insee_cpi_linked"] = linked_cpi

    for feature, (dataset, filters) in EUROSTAT_QUERIES.items():
        if feature == "job_vacancy_rate" and not include_vacancies:
            continue
        frame = eurostat.fetch_series(dataset, filters, start_period=start_quarter, name=feature)
        raw_sources[f"eurostat_{feature}"] = frame
        model_series[feature] = frame

    for filename, frame in raw_sources.items():
        frame.to_csv(raw_dir / f"{filename}.csv", index=False)

    features = build_feature_frame(model_series)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)

    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "first_period": str(features.iloc[0]["period"]),
        "last_period": str(features.iloc[-1]["period"]),
        "observations": len(features),
        "core_features": [
            "unemployment_rate",
            "gdp_qoq",
            "employment_qoq",
            "inflation_yoy",
            "business_climate",
        ],
        "insee_idbanks": INSEE_SERIES,
        "eurostat_queries": {
            feature: {"dataset": dataset, "filters": filters}
            for feature, (dataset, filters) in EUROSTAT_QUERIES.items()
            if include_vacancies or feature != "job_vacancy_rate"
        },
        "source_endpoints": {
            "insee_bdm": insee.base_url,
            "eurostat_statistics": eurostat.base_url,
        },
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Dataset written to {output}")
    coverage = f"{metadata['first_period']} to {metadata['last_period']}"
    print(f"Coverage: {coverage} ({len(features)} quarters)")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2000-Q1", help="First requested quarter, e.g. 2000-Q1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-vacancies",
        action="store_true",
        help="Do not fetch the optional vacancy-rate signal",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fetch_model_data(
        start_quarter=args.start, output=args.output, include_vacancies=not args.skip_vacancies
    )
