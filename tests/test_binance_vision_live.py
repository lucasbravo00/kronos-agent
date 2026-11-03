"""Live checks against data.binance.vision (marker ``network``, opt-in, public host only).

Run once with ``KRONOS_AGENT_NETWORK_TESTS=1``. One module-scoped client paces every request at
>= 0.25 s and the module sends four unauthenticated GETs (retries add more): a published
``.CHECKSUM``, an unpublished one (the 2019 monthly archives return 404, observed 2026-11-01), and
the 1h archive of 2020-01 (tens of kilobytes) with its checksum, downloaded into ``tmp_path``,
verified and parsed. Nothing is written outside ``tmp_path``; no key or credential exists here.
"""

from __future__ import annotations

import re
import zipfile

import pytest

from kronos_agent.data.binance_vision import (
    KLINE_COLUMNS,
    Archive,
    BinanceVisionClient,
    DownloadBudget,
    read_archive,
    verify_local,
)

pytestmark = pytest.mark.network

HOURLY_2020_01 = Archive("monthly", "1h", "2020-01")
UNPUBLISHED = Archive("monthly", "1m", "2019-09")


@pytest.fixture(scope="module")
def client():
    with BinanceVisionClient(timeout=30.0) as live_client:
        yield live_client


def test_a_published_checksum_is_one_line_naming_its_archive(client, record_property):
    digest = client.fetch_checksum(HOURLY_2020_01)  # parse_checksum ran on the remote text
    assert digest is not None, f"{HOURLY_2020_01.checksum_name} is not published (404)"
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    record_property("checksum_1h_2020_01", digest)


def test_the_2019_monthly_archives_are_not_published(client):
    assert client.fetch_checksum(UNPUBLISHED) is None


def test_the_2020_hourly_archive_downloads_verifies_and_parses(client, tmp_path, record_property):
    digest = client.fetch_checksum(HOURLY_2020_01)
    assert digest is not None
    budget = DownloadBudget(limit_bytes=8 * 2**20)
    path = client.download(HOURLY_2020_01, expected_sha256=digest, root=tmp_path, budget=budget)
    assert path.is_relative_to(tmp_path) and verify_local(HOURLY_2020_01, tmp_path)
    record_property("archive_bytes", path.stat().st_size)
    with zipfile.ZipFile(path) as bundle:
        assert bundle.namelist() == [HOURLY_2020_01.member_name]
        first_line = bundle.read(HOURLY_2020_01.member_name).split(b"\n", 1)[0].decode("ascii")
    has_header = first_line.split(",")[0] == KLINE_COLUMNS[0]
    record_property("has_header", has_header)
    assert not has_header, "fact changed: 2020 files were observed without a header line"
    frame = read_archive(HOURLY_2020_01, tmp_path)
    record_property("rows", len(frame))
    assert 0 < len(frame) <= HOURLY_2020_01.expected_rows
    assert int(frame["open_ms"].iloc[0]) >= HOURLY_2020_01.start_ms
    assert int(frame["open_ms"].iloc[-1]) < HOURLY_2020_01.end_ms
    formats = {"whole" if "." not in value else "decimal" for value in frame["close"]}
    record_property("close_formats", sorted(formats))
