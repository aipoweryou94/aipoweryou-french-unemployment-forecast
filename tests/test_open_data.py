from __future__ import annotations

import ssl
from urllib.error import HTTPError, URLError

import pytest

from aipoweryou_forecast.open_data import (
    EurostatClient,
    InseeBDMClient,
    OpenDataError,
    RetryingTransport,
    normalize_period,
    resolve_ca_bundle,
)

INSEE_XML = b"""<?xml version='1.0' encoding='UTF-8'?>
<message:StructureSpecificData
  xmlns:message='http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message'
  xmlns:ss='http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/structurespecific'>
  <message:DataSet>
    <ss:Series IDBANK='001688527' FREQ='T' TITLE_FR='Taux de chomage' LAST_UPDATE='2026-08-07'>
      <ss:Obs TIME_PERIOD='2026-Q2' OBS_VALUE='8.3' OBS_STATUS='A'/>
      <ss:Obs TIME_PERIOD='2026-Q1' OBS_VALUE='8.1' OBS_STATUS='A'/>
    </ss:Series>
  </message:DataSet>
</message:StructureSpecificData>
"""


def test_insee_parser_returns_sorted_numeric_series() -> None:
    frame = InseeBDMClient.parse_series(INSEE_XML, name="unemployment_rate")

    assert frame["period"].tolist() == ["2026-Q1", "2026-Q2"]
    assert frame["unemployment_rate"].tolist() == [8.1, 8.3]
    assert frame.attrs["idbank"] == "001688527"
    assert frame.attrs["last_update"] == "2026-08-07"


def test_insee_client_builds_expected_url_without_live_network() -> None:
    calls = []

    def fake_transport(url: str, headers: dict[str, str]) -> bytes:
        calls.append((url, headers))
        return INSEE_XML

    client = InseeBDMClient(transport=fake_transport)
    client.fetch_series("001688527", start_period="2025-Q1", end_period="2026-Q2")

    assert calls[0][0].endswith("/001688527?startPeriod=2025-Q1&endPeriod=2026-Q2")
    assert calls[0][1]["Accept"] == "application/xml"


def test_insee_parser_rejects_invalid_responses() -> None:
    with pytest.raises(OpenDataError, match="Invalid Insee SDMX XML"):
        InseeBDMClient.parse_series(b"<invalid")

    with pytest.raises(OpenDataError, match="contains no series"):
        InseeBDMClient.parse_series(b"<root><message>No series</message></root>")

    empty_series = b"<root><Series IDBANK='empty'></Series></root>"
    with pytest.raises(OpenDataError, match="no numeric observations"):
        InseeBDMClient.parse_series(empty_series)


def test_insee_client_rejects_an_unexpected_idbank() -> None:
    client = InseeBDMClient(transport=lambda _url, _headers: INSEE_XML)

    with pytest.raises(OpenDataError, match="Requested idbank"):
        client.fetch_series("unexpected")


def test_eurostat_parser_handles_sparse_jsonstat_values() -> None:
    document = {
        "id": ["freq", "geo", "time"],
        "size": [1, 1, 3],
        "dimension": {
            "freq": {"category": {"index": {"Q": 0}}},
            "geo": {"category": {"index": {"FR": 0}}},
            "time": {"category": {"index": {"2025-Q1": 0, "2025-Q2": 1, "2025-Q3": 2}}},
        },
        "value": {"0": 0.1, "2": -0.2},
    }

    frame = EurostatClient.parse_jsonstat_series(document, name="gdp_qoq")

    assert frame.to_dict("records") == [
        {"period": "2025-Q1", "gdp_qoq": 0.1},
        {"period": "2025-Q3", "gdp_qoq": -0.2},
    ]


def test_eurostat_parser_rejects_unfiltered_cube() -> None:
    document = {
        "id": ["geo", "time"],
        "size": [2, 1],
        "dimension": {
            "geo": {"category": {"index": {"FR": 0, "DE": 1}}},
            "time": {"category": {"index": {"2025-Q1": 0}}},
        },
        "value": {"0": 0.1, "1": 0.2},
    }

    with pytest.raises(OpenDataError, match="unfiltered cube"):
        EurostatClient.parse_jsonstat_series(document)


def test_eurostat_client_encodes_filters() -> None:
    calls = []
    response = {
        "id": ["freq", "geo", "time"],
        "size": [1, 1, 1],
        "dimension": {
            "freq": {"category": {"index": {"Q": 0}}},
            "geo": {"category": {"index": {"FR": 0}}},
            "time": {"category": {"index": {"2025-Q1": 0}}},
        },
        "value": {"0": 0.2},
    }

    def fake_transport(url: str, headers: dict[str, str]) -> bytes:
        import json

        calls.append((url, headers))
        return json.dumps(response).encode()

    client = EurostatClient(transport=fake_transport)
    client.fetch_series("namq_10_gdp", {"geo": "FR", "freq": "Q"}, start_period="2020-Q1")

    assert "namq_10_gdp?" in calls[0][0]
    assert "geo=FR" in calls[0][0]
    assert "sinceTimePeriod=2020-Q1" in calls[0][0]


def test_eurostat_client_rejects_invalid_json() -> None:
    client = EurostatClient(transport=lambda _url, _headers: b"not-json")

    with pytest.raises(OpenDataError, match="Invalid Eurostat JSON"):
        client.fetch_series("dataset", {"geo": "FR"})


def test_eurostat_parser_accepts_dense_values_and_rejects_empty_payload() -> None:
    document = {
        "id": ["time"],
        "size": [2],
        "dimension": {
            "time": {"category": {"index": ["2025-Q1", "2025-Q2"]}},
        },
        "value": [0.1, 0.2],
    }

    frame = EurostatClient.parse_jsonstat_series(document, name="value")

    assert frame["value"].tolist() == [0.1, 0.2]
    with pytest.raises(OpenDataError, match="Incomplete"):
        EurostatClient.parse_jsonstat_series({})
    with pytest.raises(OpenDataError, match="no observations"):
        EurostatClient.parse_jsonstat_series({**document, "value": []})


def test_retrying_transport_retries_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        @staticmethod
        def read() -> bytes:
            return b"ok"

    def flaky_urlopen(
        _request: object,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> FakeResponse:
        nonlocal calls
        calls += 1
        assert timeout == 1.0
        assert isinstance(context, ssl.SSLContext)
        if calls == 1:
            raise HTTPError("https://example.test", 503, "busy", {}, None)
        return FakeResponse()

    monkeypatch.setattr("aipoweryou_forecast.open_data.urlopen", flaky_urlopen)
    monkeypatch.setattr("aipoweryou_forecast.open_data.time.sleep", lambda _delay: None)
    transport = RetryingTransport(timeout=1.0, retries=1, backoff_seconds=0.0)

    assert transport("https://example.test", {}) == b"ok"
    assert calls == 2


def test_retrying_transport_wraps_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(
        _request: object,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        raise URLError("offline")

    monkeypatch.setattr("aipoweryou_forecast.open_data.urlopen", unavailable)
    transport = RetryingTransport(timeout=1.0, retries=0)

    with pytest.raises(OpenDataError, match="Network error"):
        transport("https://example.test", {})


def test_retrying_transport_explains_certificate_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_certificate(
        _request: object,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        reason = ssl.SSLCertVerificationError(1, "unknown issuer")
        raise URLError(reason)

    monkeypatch.setattr("aipoweryou_forecast.open_data.urlopen", invalid_certificate)
    transport = RetryingTransport(timeout=1.0, retries=0)

    with pytest.raises(OpenDataError, match="SSL_CERT_FILE"):
        transport("https://example.test", {})


def test_ca_bundle_allows_an_enterprise_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "C:/certificates/company.pem")

    assert resolve_ca_bundle() == "C:/certificates/company.pem"
    assert resolve_ca_bundle("C:/explicit.pem") == "C:/explicit.pem"


def test_normalize_period_accepts_french_quarter_notation() -> None:
    assert normalize_period(" 2026-T2 ") == "2026-Q2"
    assert normalize_period("2026-07") == "2026-07"
