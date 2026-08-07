"""Small, dependency-light clients for official Insee and Eurostat APIs."""

from __future__ import annotations

import json
import math
import os
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import certifi
import pandas as pd

Transport = Callable[[str, Mapping[str, str]], bytes]


class OpenDataError(RuntimeError):
    """Raised when an open-data response is unavailable or inconsistent."""


def resolve_ca_bundle(explicit_path: str | None = None) -> str:
    """Return the trusted CA bundle while allowing an enterprise override.

    ``certifi`` fixes the common Microsoft Store Python certificate-chain issue.
    Organisations that inspect HTTPS can set ``SSL_CERT_FILE`` to a PEM bundle
    containing their internal root certificate instead of weakening TLS checks.
    """
    return explicit_path or os.getenv("SSL_CERT_FILE") or certifi.where()


def normalize_period(value: str) -> str:
    """Normalize SDMX period labels while keeping monthly values unchanged."""
    value = str(value).strip()
    if "-T" in value:
        return value.replace("-T", "-Q")
    return value


def _ordered_categories(category_index: dict[str, int] | list[str]) -> list[str]:
    if isinstance(category_index, list):
        return category_index
    ordered: list[str | None] = [None] * len(category_index)
    for label, position in category_index.items():
        ordered[position] = label
    if any(label is None for label in ordered):
        raise OpenDataError("Invalid JSON-stat category index")
    return [str(label) for label in ordered]


@dataclass
class RetryingTransport:
    """HTTP transport with conservative retries for public statistical APIs."""

    timeout: float = 45.0
    retries: int = 3
    backoff_seconds: float = 0.5
    user_agent: str = "AiPowerYou-open-data-forecast/0.3"
    ca_bundle: str | None = None

    def __call__(self, url: str, headers: Mapping[str, str]) -> bytes:
        request_headers = {"User-Agent": self.user_agent, **dict(headers)}
        ssl_context = ssl.create_default_context(cafile=resolve_ca_bundle(self.ca_bundle))
        for attempt in range(self.retries + 1):
            try:
                request = Request(url, headers=request_headers)
                with urlopen(
                    request,
                    timeout=self.timeout,
                    context=ssl_context,
                ) as response:
                    return bytes(response.read())
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.retries:
                    raise OpenDataError(f"HTTP {exc.code} for {url}") from exc
                retry_after = exc.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else self.backoff_seconds * 2**attempt
                )
                time.sleep(delay)
            except URLError as exc:
                if attempt == self.retries:
                    if isinstance(exc.reason, ssl.SSLCertVerificationError):
                        raise OpenDataError(
                            "TLS certificate verification failed. Update certifi or set "
                            "SSL_CERT_FILE to your organisation's trusted PEM bundle."
                        ) from exc
                    raise OpenDataError(f"Network error for {url}: {exc.reason}") from exc
                time.sleep(self.backoff_seconds * 2**attempt)
        raise AssertionError("unreachable")


class InseeBDMClient:
    """Read chronological series from Insee's SDMX 2.1 BDM API."""

    def __init__(
        self,
        base_url: str = "https://api.insee.fr/series/BDM/V1/data/SERIES_BDM",
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or RetryingTransport()

    def fetch_series(
        self,
        idbank: str,
        *,
        start_period: str | None = None,
        end_period: str | None = None,
        name: str = "value",
    ) -> pd.DataFrame:
        params = {}
        if start_period:
            params["startPeriod"] = start_period
        if end_period:
            params["endPeriod"] = end_period
        url = f"{self.base_url}/{quote(idbank, safe='')}"
        if params:
            url += "?" + urlencode(params)
        payload = self.transport(url, {"Accept": "application/xml"})
        frame = self.parse_series(payload, name=name)
        if frame.attrs.get("idbank") != idbank:
            raise OpenDataError(f"Requested idbank {idbank}, received {frame.attrs.get('idbank')}")
        frame.attrs["request_url"] = url
        return frame

    @staticmethod
    def parse_series(payload: bytes | str, *, name: str = "value") -> pd.DataFrame:
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as exc:
            raise OpenDataError("Invalid Insee SDMX XML") from exc

        series = next((node for node in root.iter() if node.tag.endswith("Series")), None)
        if series is None:
            error_text = " ".join((node.text or "") for node in root.iter() if node.text)
            raise OpenDataError(f"Insee response contains no series: {error_text[:200]}")

        rows = []
        for node in series.iter():
            if not node.tag.endswith("Obs"):
                continue
            period = node.attrib.get("TIME_PERIOD")
            value = node.attrib.get("OBS_VALUE")
            if period is None or value is None or value == "":
                continue
            rows.append(
                {
                    "period": normalize_period(period),
                    name: float(value),
                    "status": node.attrib.get("OBS_STATUS"),
                }
            )
        if not rows:
            raise OpenDataError("Insee series contains no numeric observations")

        frame = pd.DataFrame(rows).drop_duplicates("period", keep="first")
        frame = frame.sort_values(
            "period", key=lambda values: values.map(_period_sort_key)
        ).reset_index(drop=True)
        frame.attrs.update(
            {
                "source": "Insee BDM",
                "idbank": series.attrib.get("IDBANK"),
                "frequency": series.attrib.get("FREQ"),
                "title": series.attrib.get("TITLE_FR"),
                "last_update": series.attrib.get("LAST_UPDATE"),
            }
        )
        return frame


class EurostatClient:
    """Read one-dimensional time series from Eurostat's Statistics API."""

    def __init__(
        self,
        base_url: str = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data",
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.transport = transport or RetryingTransport()

    def fetch_series(
        self,
        dataset: str,
        filters: Mapping[str, str],
        *,
        start_period: str | None = None,
        name: str = "value",
    ) -> pd.DataFrame:
        params = {"lang": "en", **dict(filters)}
        if start_period:
            params["sinceTimePeriod"] = start_period
        url = f"{self.base_url}/{quote(dataset, safe='')}?{urlencode(params)}"
        payload = self.transport(url, {"Accept": "application/json"})
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OpenDataError("Invalid Eurostat JSON") from exc
        frame = self.parse_jsonstat_series(document, name=name)
        frame.attrs.update({"source": "Eurostat", "dataset": dataset, "request_url": url})
        return frame

    @staticmethod
    def parse_jsonstat_series(document: Mapping[str, Any], *, name: str = "value") -> pd.DataFrame:
        ids = list(document.get("id", []))
        sizes = list(document.get("size", []))
        dimensions = document.get("dimension", {})
        if not ids or len(ids) != len(sizes) or "time" not in ids:
            raise OpenDataError("Incomplete Eurostat JSON-stat structure")

        time_axis = ids.index("time")
        for axis, size in enumerate(sizes):
            if axis != time_axis and size != 1:
                dimension_id = ids[axis]
                raise OpenDataError(
                    f"Eurostat query returned an unfiltered cube on dimension "
                    f"{dimension_id} (size={size})"
                )

        categories: list[list[str]] = []
        for dimension_id in ids:
            try:
                index = dimensions[dimension_id]["category"]["index"]
            except KeyError as exc:
                raise OpenDataError(
                    f"Missing categories for Eurostat dimension {dimension_id}"
                ) from exc
            categories.append(_ordered_categories(index))

        values = document.get("value", {})
        if isinstance(values, list):
            values = {str(index): value for index, value in enumerate(values) if value is not None}

        rows = []
        for flat_index_text, value in values.items():
            flat_index = int(flat_index_text)
            remainder = flat_index
            coordinates = []
            for axis, _size in enumerate(sizes):
                stride = math.prod(sizes[axis + 1 :])
                position = remainder // stride if stride else 0
                remainder %= stride if stride else 1
                coordinates.append(position)
            period = categories[time_axis][coordinates[time_axis]]
            rows.append({"period": normalize_period(period), name: float(value)})

        if not rows:
            raise OpenDataError("Eurostat query returned no observations")
        frame = pd.DataFrame(rows).drop_duplicates("period", keep="last")
        return frame.sort_values(
            "period", key=lambda values: values.map(_period_sort_key)
        ).reset_index(drop=True)


def _period_sort_key(period: str) -> tuple[int, int, int]:
    period = normalize_period(period)
    if "-Q" in period:
        year, quarter = period.split("-Q")
        return int(year), int(quarter) * 3, 0
    if len(period) == 7 and period[4] == "-":
        year, month = period.split("-")
        return int(year), int(month), 0
    return int(period[:4]), 0, 0
