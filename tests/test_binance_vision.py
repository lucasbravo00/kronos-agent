"""Binance archive client, plan, sync, strict parse, stores and the resample check (no network).

Every request goes through a real :class:`BinanceVisionClient` on an ``httpx.MockTransport``
(``step2_fakes.FakeBinanceVision``); sleeps advance a fake clock, so retries and pacing are
checked without waiting. Archives are small synthetic ones with the formats of the real files
(header or none, millisecond or microsecond times, ``7225`` and ``7171.10`` price forms).
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import zipfile
from datetime import UTC, date, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
import step2_fakes as fakes

from kronos_agent.data import binance_vision as bv
from kronos_agent.data.binance_vision import (
    Archive,
    ArchiveFormatError,
    BinanceVisionClient,
    BinanceVisionError,
    BinanceVisionUnavailable,
    ChecksumMismatchError,
    DownloadBudget,
    DownloadBudgetExceeded,
    HostNotAllowedError,
)
from kronos_agent.data.fsutil import LockBusyError, exclusive_lock
from kronos_agent.data.partition import SEALED_START_MS, TRAINING_END, VALIDATION_START
from kronos_agent.data.resample import resample_exact
from kronos_agent.data.store import (
    BINANCE_COLUMNS,
    StoreId,
    StoreIntegrityError,
    read_meta,
    read_store,
)
from kronos_agent.data.timeutil import to_ms
from kronos_agent.report.writer import REQUIRED_KEYS, write_report

MINUTE_MS = 60_000
HOUR_MS = 3_600_000
TODAY = date(2019, 10, 2)  # plan: 2019-09 (monthly or its dailies) and 2019-10-01
SEPT_DAYS = ("2019-09-29", "2019-09-30")
DAY0 = fakes.utc_ms(2020, 1, 1)


def make_client(
    fake: fakes.FakeBinanceVision, **kwargs
) -> tuple[BinanceVisionClient, fakes.FakeClock]:
    """A real client on ``httpx.MockTransport(fake)``; pacing off unless asked for."""
    clock = fakes.FakeClock()
    options = {"min_interval_s": 0.0, **kwargs}
    client = BinanceVisionClient(
        transport=httpx.MockTransport(fake), sleep=clock.sleep, clock=clock, **options
    )
    return client, clock


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put_zip(root: Path, archive: Archive, data: bytes) -> bytes:
    """Place an archive and its local .CHECKSUM as a verified download would."""
    path = archive.local_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    archive.checksum_path(root).write_text(bv.checksum_line(sha256(data), archive))
    return data


def put_csv(root: Path, archive: Archive, payload: bytes, *, member: str | None = None) -> bytes:
    return put_zip(root, archive, fakes.make_zip(member or archive.member_name, payload))


def lines_of(frame: pd.DataFrame, interval: str = "1m", **kwargs) -> list[str]:
    return fakes.kline_csv(frame, interval=interval, **kwargs).decode("ascii").splitlines()


def payload_of(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("ascii")


def small_day(rows: int = 30, *, seed: int = 0) -> pd.DataFrame:
    return fakes.binance_frame(DAY0, rows, interval="1m", seed=seed)


DAY_ARCHIVE = Archive("daily", "1m", "2020-01-01")


def serve_days(
    fake: fakes.FakeBinanceVision, interval: str, days, *, seed: int = 0, **kwargs
) -> dict[str, pd.DataFrame]:
    frames = {}
    for offset, day in enumerate(days):
        frame = fakes.period_frame("daily", day, interval=interval, seed=seed + offset)
        fake.add_archive("daily", interval, day, frame, **kwargs)
        frames[day] = frame
    return frames


def concat(frames) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True)


def no_temp_files(directory: Path) -> bool:
    return not any(path.name.startswith(".") for path in directory.rglob("*") if path.is_file())


# -- archives --------------------------------------------------------------------------------------


def test_archive_names_paths_and_half_open_bounds(tmp_path):
    monthly = Archive("monthly", "1m", "2020-01")
    assert monthly.name == "BTCUSDT-1m-2020-01.zip"
    assert monthly.checksum_name == "BTCUSDT-1m-2020-01.zip.CHECKSUM"
    assert monthly.member_name == "BTCUSDT-1m-2020-01.csv"
    assert monthly.url_path == "/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01.zip"
    assert monthly.checksum_url_path == monthly.url_path + ".CHECKSUM"
    assert monthly.relpath == "raw/binance/monthly/1m/BTCUSDT-1m-2020-01.zip"
    assert monthly.local_path(tmp_path) == tmp_path / monthly.relpath
    assert monthly.checksum_path(tmp_path) == tmp_path / (monthly.relpath + ".CHECKSUM")
    assert (monthly.start_ms, monthly.end_ms) == (
        fakes.utc_ms(2020, 1, 1),
        fakes.utc_ms(2020, 2, 1),
    )
    assert monthly.expected_rows == 31 * 1440
    assert Archive("monthly", "1m", "2020-02").expected_rows == 29 * 1440  # leap year
    december = Archive("monthly", "1h", "2019-12")
    assert december.end_ms == fakes.utc_ms(2020, 1, 1) and december.expected_rows == 744
    daily = Archive("daily", "15m", "2024-03-31")
    assert (daily.start_ms, daily.end_ms) == (fakes.utc_ms(2024, 3, 31), fakes.utc_ms(2024, 4, 1))
    assert daily.expected_rows == 96
    assert daily.relpath == "raw/binance/daily/15m/BTCUSDT-15m-2024-03-31.zip"


@pytest.mark.parametrize(
    ("kind", "interval", "period"),
    [
        ("weekly", "1m", "2020-01"),
        ("monthly", "4h", "2020-01"),
        ("monthly", "1m", "2020-1"),
        ("monthly", "1m", "2020-13"),
        ("monthly", "1m", "2020-01-01"),
        ("daily", "1m", "2020-01"),
        ("daily", "1m", "2020-02-30"),
        ("daily", "1m", "20200101"),
        ("daily", "1m", None),
    ],
)
def test_archive_refuses_bad_fields(kind, interval, period):
    with pytest.raises(ValueError):
        Archive(kind, interval, period)


# -- the pinned client -----------------------------------------------------------------------------


def test_constructor_has_no_host_url_or_headers_parameter():
    names = list(inspect.signature(BinanceVisionClient).parameters)
    assert names == ["transport", "timeout", "min_interval_s", "max_retries", "sleep", "clock"]
    assert bv.HOST == fakes.BINANCE_HOST


@pytest.mark.parametrize(
    "origin",
    [
        "http://" + bv.HOST,
        "https://" + bv.HOST + ":8443",
        "https://user:pw" + "@" + bv.HOST,  # userinfo
        "https://" + bv.HOST + ".example.com",
        "https://example.com",
        "https://binance.vision",
    ],
)
def test_transport_refuses_every_other_origin(origin):
    fake = fakes.FakeBinanceVision()
    client, _ = make_client(fake)
    with pytest.raises(HostNotAllowedError):
        client._transport.handle_request(httpx.Request("GET", origin + DAY_ARCHIVE.url_path))
    assert fake.requests == []


@pytest.mark.parametrize(
    "path",
    [
        "/data/futures/um/daily/klines/ETHUSDT/1m/ETHUSDT-1m-2020-01-01.zip",
        "/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01-01.zip",
        "/data/futures/cm/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01-01.zip",
        "/data/futures/um/daily/klines/BTCUSDT/4h/BTCUSDT-4h-2020-01-01.zip",
        "/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1h-2020-01-01.zip",
        "/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01.zip",
        "/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01-01.zip",
        "/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2020-02-30.zip",
        "/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01-01.zip.sha256",
        "/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01-01.zip?x=1",
        "/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01-01.zip/",
        "/data/futures/um/daily/klines/BTCUSDT/1m/..%2F..%2Fsecret",
        "/data/futures/um/daily/klines/BTCUSDT/1m/",
        "/",
    ],
)
def test_paths_outside_the_allow_list_are_refused_before_the_wire(path):
    fake = fakes.FakeBinanceVision()
    client, _ = make_client(fake)
    with pytest.raises(HostNotAllowedError, match="allow-list"):
        client._transport.handle_request(httpx.Request("GET", "https://" + bv.HOST + path))
    with pytest.raises(HostNotAllowedError, match="allow-list"):
        client._get(path, lambda response: response.read())
    assert fake.requests == []


def test_allowed_paths_reach_the_server_and_only_get_is_sent():
    fake = fakes.FakeBinanceVision()
    client, _ = make_client(fake)
    for archive in (
        Archive("monthly", "1m", "2020-01"),
        Archive("daily", "5m", "2020-01-01"),
        Archive("daily", "15m", "2020-01-01"),
        Archive("monthly", "1h", "2019-09"),
    ):
        assert client.fetch_checksum(archive) is None  # 404: not published
    assert len(fake.requests) == 4 and all(r.method == "GET" for r in fake.requests)
    allowed = "https://" + bv.HOST + DAY_ARCHIVE.url_path
    with pytest.raises(HostNotAllowedError, match="GET"):
        client._transport.handle_request(httpx.Request("POST", allowed))
    assert len(fake.requests) == 4


def test_no_credentials_cookies_or_environment():
    fake = fakes.FakeBinanceVision()
    path = DAY_ARCHIVE.checksum_url_path
    fake.queue(
        path,
        httpx.Response(404, headers={"set-cookie": "session=abc; Path=/"}),
        httpx.Response(404),
    )
    client, _ = make_client(fake)
    assert client.fetch_checksum(DAY_ARCHIVE) is None
    assert client.fetch_checksum(DAY_ARCHIVE) is None
    for request in fake.requests:
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert request.headers["accept-encoding"] == "identity"
    assert client._http.trust_env is False
    assert "binance.vision" in repr(client)


def test_404_unexpected_status_and_redirects():
    fake = fakes.FakeBinanceVision()
    client, clock = make_client(fake)
    assert client.fetch_checksum(DAY_ARCHIVE) is None
    fake.queue(DAY_ARCHIVE.checksum_url_path, httpx.Response(403, text="denied"))
    with pytest.raises(BinanceVisionError, match="HTTP 403"):
        client.fetch_checksum(DAY_ARCHIVE)
    redirect = httpx.Response(302, headers={"location": "https://example.com/elsewhere"})
    fake.queue(DAY_ARCHIVE.checksum_url_path, redirect)
    with pytest.raises(BinanceVisionError, match="redirect refused"):
        client.fetch_checksum(DAY_ARCHIVE)
    fake.queue(DAY_ARCHIVE.checksum_url_path, httpx.Response(301, headers={"location": "http://"}))
    with pytest.raises(BinanceVisionError, match="redirect"):
        client.fetch_checksum(DAY_ARCHIVE)
    assert len(fake.requests) == 4  # nothing retried, nothing followed
    assert clock.sleeps == []


def test_timeouts_and_5xx_are_retried_after_1_5_15_seconds():
    fake = fakes.FakeBinanceVision()
    data = fakes.make_zip(DAY_ARCHIVE.member_name, b"")
    fake.add_file(DAY_ARCHIVE.checksum_url_path, fakes.checksum_text(data, DAY_ARCHIVE.name))

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return timeout(request)
        if calls["n"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, content=fake.files[DAY_ARCHIVE.checksum_url_path])

    fake.overrides[DAY_ARCHIVE.checksum_url_path] = flaky
    client, clock = make_client(fake)
    assert client.fetch_checksum(DAY_ARCHIVE) == sha256(data)
    assert clock.sleeps == [1.0, 5.0]
    fake.overrides[DAY_ARCHIVE.checksum_url_path] = lambda request: httpx.Response(502)
    clock.sleeps.clear()
    with pytest.raises(BinanceVisionUnavailable, match="HTTP 502 after 4 attempts"):
        client.fetch_checksum(DAY_ARCHIVE)
    assert clock.sleeps == [1.0, 5.0, 15.0]
    fake.overrides[DAY_ARCHIVE.checksum_url_path] = timeout
    clock.sleeps.clear()
    with pytest.raises(BinanceVisionUnavailable, match="ReadTimeout"):
        client.fetch_checksum(DAY_ARCHIVE)
    assert clock.sleeps == [1.0, 5.0, 15.0]
    assert len(fake.requests) == 3 + 4 + 4


def test_requests_are_paced_at_least_a_quarter_second_apart():
    fake = fakes.FakeBinanceVision()
    client, clock = make_client(fake, min_interval_s=bv.MIN_INTERVAL_S)
    for _ in range(3):
        client.fetch_checksum(DAY_ARCHIVE)
    assert clock.sleeps == [0.25, 0.25]
    clock.now += 10.0
    client.fetch_checksum(DAY_ARCHIVE)
    assert clock.sleeps == [0.25, 0.25]


# -- checksums -------------------------------------------------------------------------------------


def test_parse_checksum_is_strict():
    digest = "ab" * 32
    name = DAY_ARCHIVE.name
    assert bv.parse_checksum(f"{digest}  {name}\n", DAY_ARCHIVE) == digest
    assert bv.parse_checksum(f"{digest}  {name}", DAY_ARCHIVE) == digest
    for bad in (
        f"{digest.upper()}  {name}\n",
        f"{digest} {name}\n",
        f"{digest}  {name}\r\n",
        f"{digest}  {name}\n\n",
        f"{digest}  BTCUSDT-1m-2020-01-02.zip\n",
        f"{digest[:-1]}  {name}\n",
        f"{digest} *{name}\n",
        "",
    ):
        with pytest.raises(ArchiveFormatError):
            bv.parse_checksum(bad, DAY_ARCHIVE)


def test_remote_checksum_body_is_capped():
    fake = fakes.FakeBinanceVision()
    fake.add_file(DAY_ARCHIVE.checksum_url_path, b"a" * (bv.CHECKSUM_MAX_BYTES + 1))
    client, _ = make_client(fake)
    with pytest.raises(BinanceVisionError, match="cap"):
        client.fetch_checksum(DAY_ARCHIVE)


# -- downloads and the gate-4 budget ---------------------------------------------------------------


def test_download_verifies_then_publishes_the_archive_and_its_checksum(tmp_path):
    fake = fakes.FakeBinanceVision()
    data = fake.add_archive("daily", "1m", "2020-01-01", small_day())
    client, _ = make_client(fake)
    budget = DownloadBudget(limit_bytes=10**6)
    expected = client.fetch_checksum(DAY_ARCHIVE)
    path = client.download(DAY_ARCHIVE, expected_sha256=expected, root=tmp_path, budget=budget)
    assert path == DAY_ARCHIVE.local_path(tmp_path)
    assert path.read_bytes() == data
    assert (
        DAY_ARCHIVE.checksum_path(tmp_path).read_text() == f"{sha256(data)}  {DAY_ARCHIVE.name}\n"
    )
    assert budget.used_bytes == len(data)
    assert bv.verify_local(DAY_ARCHIVE, tmp_path)
    assert no_temp_files(tmp_path)


def test_checksum_mismatch_deletes_the_download(tmp_path):
    fake = fakes.FakeBinanceVision()
    fake.add_archive("daily", "1m", "2020-01-01", small_day())
    client, _ = make_client(fake)
    with pytest.raises(ChecksumMismatchError, match="deleted"):
        client.download(
            DAY_ARCHIVE, expected_sha256="0" * 64, root=tmp_path, budget=DownloadBudget()
        )
    assert not DAY_ARCHIVE.local_path(tmp_path).exists()
    assert not DAY_ARCHIVE.checksum_path(tmp_path).exists()
    assert no_temp_files(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        client.download(
            DAY_ARCHIVE, expected_sha256="AB" * 32, root=tmp_path, budget=DownloadBudget()
        )


def test_download_of_an_unpublished_archive_fails(tmp_path):
    client, _ = make_client(fakes.FakeBinanceVision())
    with pytest.raises(BinanceVisionError, match="404"):
        client.download(
            DAY_ARCHIVE, expected_sha256="0" * 64, root=tmp_path, budget=DownloadBudget()
        )
    assert no_temp_files(tmp_path)


def test_archive_size_cap_announced_and_streamed(tmp_path, monkeypatch):
    fake = fakes.FakeBinanceVision()
    data = fake.add_archive("daily", "1m", "2020-01-01", small_day())
    monkeypatch.setattr(bv, "ARCHIVE_MAX_BYTES", len(data) - 1)
    client, _ = make_client(fake)
    budget = DownloadBudget()
    with pytest.raises(BinanceVisionError, match="announced"):
        client.download(DAY_ARCHIVE, expected_sha256=sha256(data), root=tmp_path, budget=budget)
    fake.overrides[DAY_ARCHIVE.url_path] = lambda request: httpx.Response(
        200, content=iter([data[:100], data[100:]])
    )
    with pytest.raises(BinanceVisionError, match="exceeds"):
        client.download(DAY_ARCHIVE, expected_sha256=sha256(data), root=tmp_path, budget=budget)
    assert not DAY_ARCHIVE.local_path(tmp_path).exists()
    assert no_temp_files(tmp_path)
    monkeypatch.setattr(bv, "ARCHIVE_MAX_BYTES", len(data))
    client.download(DAY_ARCHIVE, expected_sha256=sha256(data), root=tmp_path, budget=budget)
    assert DAY_ARCHIVE.local_path(tmp_path).read_bytes() == data


def test_budget_is_cumulative_across_downloads(tmp_path):
    fake = fakes.FakeBinanceVision()
    first, second = Archive("daily", "1m", "2020-01-01"), Archive("daily", "1m", "2020-01-02")
    data_1 = fake.add_archive("daily", "1m", first.period, small_day())
    data_2 = fake.add_archive(
        "daily", "1m", second.period, fakes.binance_frame(second.start_ms, 30, interval="1m")
    )
    client, _ = make_client(fake)
    budget = DownloadBudget(limit_bytes=len(data_1) + len(data_2) - 1)
    client.download(first, expected_sha256=sha256(data_1), root=tmp_path, budget=budget)
    with pytest.raises(DownloadBudgetExceeded, match="gate 4"):
        client.download(second, expected_sha256=sha256(data_2), root=tmp_path, budget=budget)
    assert budget.used_bytes == len(data_1)
    assert not second.local_path(tmp_path).exists()
    assert no_temp_files(tmp_path)
    budget.charge(len(data_2) - 1)  # exactly at the limit is allowed
    with pytest.raises(DownloadBudgetExceeded):
        budget.charge(1)
    with pytest.raises(ValueError):
        budget.charge(-1)


def test_budget_from_disk_counts_every_byte_under_raw_binance(tmp_path, monkeypatch):
    assert DownloadBudget.from_disk(tmp_path).used_bytes == 0
    (tmp_path / "raw" / "binance" / "monthly" / "1m").mkdir(parents=True)
    (tmp_path / "raw" / "binance" / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "raw" / "binance" / "monthly" / "1m" / "b.zip").write_bytes(b"x" * 5)
    (tmp_path / "raw" / "bingx").mkdir(parents=True)
    (tmp_path / "raw" / "bingx" / "c.csv").write_bytes(b"x" * 100)  # another source
    (tmp_path / "other.bin").write_bytes(b"x" * 1000)
    budget = DownloadBudget.from_disk(tmp_path)
    assert (budget.used_bytes, budget.limit_bytes) == (15, bv.BUDGET_BYTES)
    monkeypatch.setattr(bv, "BUDGET_BYTES", 20)
    assert DownloadBudget.from_disk(tmp_path).remaining_bytes == 5


# -- strict parse ----------------------------------------------------------------------------------


def test_header_and_no_header_give_the_same_frame(tmp_path):
    frame = small_day()
    assert any("." not in value for value in frame["open"])  # whole-dollar strings present
    assert any("." in value for value in frame["open"])
    for header in (True, False):
        put_csv(tmp_path, DAY_ARCHIVE, fakes.kline_csv(frame, interval="1m", header=header))
        parsed = bv.read_archive(DAY_ARCHIVE, tmp_path)
        assert parsed.equals(frame)
        assert list(parsed.columns) == list(BINANCE_COLUMNS)
        assert parsed["open_ms"].dtype == np.int64


def test_microsecond_times_and_crlf_endings_are_accepted(tmp_path):
    frame = small_day()
    put_csv(tmp_path, DAY_ARCHIVE, fakes.kline_csv(frame, interval="1m", microseconds=True))
    assert bv.read_archive(DAY_ARCHIVE, tmp_path).equals(frame)
    put_csv(tmp_path, DAY_ARCHIVE, fakes.kline_csv(frame, interval="1m", line_ending="\r\n"))
    assert bv.read_archive(DAY_ARCHIVE, tmp_path).equals(frame)


def edit_field(lines: list[str], line: int, field: int, value: str) -> list[str]:
    """Replace one field of a 1-based line."""
    out = list(lines)
    parts = out[line - 1].split(",")
    parts[field] = value
    out[line - 1] = ",".join(parts)
    return out


def assert_refused(tmp_path, lines, message, *, archive=DAY_ARCHIVE):
    put_csv(tmp_path, archive, payload_of(lines))
    with pytest.raises(ArchiveFormatError, match=message) as info:
        bv.read_archive(archive, tmp_path)
    assert archive.name in str(info.value)


def test_foreign_header_is_refused(tmp_path):
    lines = lines_of(small_day())
    lines[0] = lines[0].replace("quote_volume", "quote_asset_volume", 1)
    assert_refused(tmp_path, lines, "line 1: header")
    lines = lines_of(small_day())
    lines[0] = lines[0].upper()
    assert_refused(tmp_path, lines, "line 1: header")


def test_short_long_and_blank_lines_are_refused_with_their_line(tmp_path):
    lines = lines_of(small_day())  # header on line 1
    short = list(lines)
    short[4] = short[4].rsplit(",", 1)[0]
    assert_refused(tmp_path, short, "line 5: 11 fields, expected 12")
    long = list(lines)
    long[6] = long[6] + ",0"
    assert_refused(tmp_path, long, "line 7: 13 fields")
    blank = [*lines[:3], "", *lines[3:]]
    assert_refused(tmp_path, blank, "line 4: blank line")
    put_csv(tmp_path, DAY_ARCHIVE, payload_of(lines) + b"\n")
    with pytest.raises(ArchiveFormatError, match=f"line {len(lines) + 1}: blank line"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)


def test_close_time_alignment_order_and_period_are_checked(tmp_path):
    lines = lines_of(small_day(), header=False)
    assert_refused(
        tmp_path,
        edit_field(lines, 3, 6, str(int(lines[2].split(",")[6]) + 1)),
        "line 3: close_time",
    )
    shifted = edit_field(lines, 4, 0, str(int(lines[3].split(",")[0]) + 1))
    shifted = edit_field(shifted, 4, 6, str(int(lines[3].split(",")[6]) + 1))
    assert_refused(tmp_path, shifted, "line 4: open_time is not aligned")
    duplicate = [*lines[:5], lines[4], *lines[5:]]
    assert_refused(tmp_path, duplicate, "line 6: open_time is not strictly increasing")
    swapped = list(lines)
    swapped[7], swapped[8] = swapped[8], swapped[7]
    assert_refused(tmp_path, swapped, "line 9: open_time is not strictly increasing")


def test_rows_must_lie_inside_the_archive_period_half_open(tmp_path):
    archive = Archive("daily", "1h", "2020-01-01")
    frame = fakes.binance_frame(archive.start_ms, 24, interval="1h")
    put_csv(tmp_path, archive, fakes.kline_csv(frame, interval="1h"))
    assert len(bv.read_archive(archive, tmp_path)) == 24  # first and last hour of the day
    late = fakes.binance_frame(archive.start_ms, 25, interval="1h")  # one row at end_ms
    assert_refused(tmp_path, lines_of(late, "1h"), "line 26: open_time is outside", archive=archive)
    early = fakes.binance_frame(archive.start_ms - HOUR_MS, 2, interval="1h")
    assert_refused(tmp_path, lines_of(early, "1h"), "line 2: open_time is outside", archive=archive)


def test_mixed_time_units_are_refused(tmp_path):
    frame = small_day()
    micro = lines_of(frame, microseconds=True)
    milli = lines_of(frame)
    mixed = [*micro[:6], milli[6], *micro[7:]]
    assert_refused(tmp_path, mixed, "line 7: open_time is not a whole millisecond")
    mixed = [*milli[:3], micro[3], *milli[4:]]
    assert_refused(tmp_path, mixed, "line 4: open_time is in microseconds")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (1, "1e5", "open is not a plain decimal"),
        (2, "", "high is not a plain decimal"),
        (3, "nan", "low is not a plain decimal"),
        (4, "+7200", "close is not a plain decimal"),
        (5, " 1.5", "volume is not a plain decimal"),
        (7, "7225.", "quote_volume is not a plain decimal"),
        (8, "1.0", "count is not an unsigned integer"),
        (9, "-", "taker_buy_volume is not a plain decimal"),
        (10, '"5"', "taker_buy_quote_volume is not a plain decimal"),
        (0, "abc", "open_time is not an integer time"),
        (6, "1577836859999.0", "close_time is not an integer time"),
    ],
)
def test_values_must_be_plain_decimals(tmp_path, field, value, message):
    lines = lines_of(small_day())
    assert_refused(tmp_path, edit_field(lines, 9, field, value), f"line 9: {message}")


def test_the_earliest_bad_line_is_named(tmp_path):
    lines = lines_of(small_day())
    lines = edit_field(lines, 12, 1, "x")
    lines = edit_field(lines, 5, 9, "y")
    assert_refused(tmp_path, lines, "line 5: taker_buy_volume")


def test_zip_structure_and_bytes_are_checked(tmp_path, monkeypatch):
    payload = fakes.kline_csv(small_day(), interval="1m")
    put_csv(tmp_path, DAY_ARCHIVE, payload, member="other.csv")
    with pytest.raises(ArchiveFormatError, match="member"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(DAY_ARCHIVE.member_name, payload)
        bundle.writestr("extra.csv", payload)
    put_zip(tmp_path, DAY_ARCHIVE, buffer.getvalue())
    with pytest.raises(ArchiveFormatError, match="2 members"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    put_zip(tmp_path, DAY_ARCHIVE, b"not a zip at all")
    with pytest.raises(ArchiveFormatError, match="not a readable zip"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    good = fakes.make_zip(DAY_ARCHIVE.member_name, payload)
    corrupt = bytearray(good)
    corrupt[60] ^= 0xFF  # inside the deflated member
    put_zip(tmp_path, DAY_ARCHIVE, bytes(corrupt))
    with pytest.raises(ArchiveFormatError):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    put_csv(tmp_path, DAY_ARCHIVE, payload.replace(b"\n", b"\n\xc3\xa9", 1))
    with pytest.raises(ArchiveFormatError, match="line 2: non-ASCII"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    put_csv(tmp_path, DAY_ARCHIVE, b"")
    with pytest.raises(ArchiveFormatError, match="no rows"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    put_csv(tmp_path, DAY_ARCHIVE, payload.split(b"\n", 1)[0] + b"\n")
    with pytest.raises(ArchiveFormatError, match="no rows"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    put_csv(tmp_path, DAY_ARCHIVE, payload)
    monkeypatch.setattr(bv, "CSV_MAX_BYTES", len(payload) - 1)
    with pytest.raises(ArchiveFormatError, match="cap"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)


def test_read_archive_checks_the_bytes_against_the_local_checksum(tmp_path):
    with pytest.raises(BinanceVisionError, match="CHECKSUM"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    data = put_csv(tmp_path, DAY_ARCHIVE, fakes.kline_csv(small_day(), interval="1m"))
    assert bv.verify_local(DAY_ARCHIVE, tmp_path)
    DAY_ARCHIVE.local_path(tmp_path).write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert not bv.verify_local(DAY_ARCHIVE, tmp_path)
    with pytest.raises(ChecksumMismatchError):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    DAY_ARCHIVE.local_path(tmp_path).unlink()
    with pytest.raises(BinanceVisionError, match="missing"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)


def test_local_archives_above_the_size_cap_are_never_read(tmp_path, monkeypatch):
    data = put_csv(tmp_path, DAY_ARCHIVE, fakes.kline_csv(small_day(), interval="1m"))
    monkeypatch.setattr(bv, "ARCHIVE_MAX_BYTES", len(data) - 1)
    assert not bv.verify_local(DAY_ARCHIVE, tmp_path)
    with pytest.raises(ArchiveFormatError, match="cap"):
        bv.read_archive(DAY_ARCHIVE, tmp_path)
    monkeypatch.setattr(bv, "ARCHIVE_MAX_BYTES", len(data))
    assert bv.verify_local(DAY_ARCHIVE, tmp_path)
    assert len(bv.read_archive(DAY_ARCHIVE, tmp_path)) == 30


# -- plans -----------------------------------------------------------------------------------------


def test_plan_prefers_monthly_archives_and_probes_dailies_only_where_needed():
    fake = fakes.FakeBinanceVision()
    checksums = {}
    for archive in (
        Archive("daily", "1h", "2019-09-29"),
        Archive("daily", "1h", "2019-09-30"),
        Archive("monthly", "1h", "2019-10"),
        Archive("daily", "1h", "2019-10-05"),  # never probed: October has its monthly
        Archive("daily", "1h", "2019-11-01"),
        Archive("daily", "1h", "2019-11-02"),
        Archive("daily", "1h", "2019-11-03"),  # today: not yet complete, never probed
    ):
        digest = sha256(archive.name.encode())
        checksums[archive.name] = digest
        fake.add_file(archive.checksum_url_path, f"{digest}  {archive.name}\n".encode())
    client, _ = make_client(fake)
    plan = bv.plan_archives(client, "1h", today=date(2019, 11, 3))
    names = [entry["name"] for entry in plan["archives"]]
    assert names == [
        "BTCUSDT-1h-2019-09-29.zip",
        "BTCUSDT-1h-2019-09-30.zip",
        "BTCUSDT-1h-2019-10.zip",
        "BTCUSDT-1h-2019-11-01.zip",
        "BTCUSDT-1h-2019-11-02.zip",
    ]
    assert [entry["sha256"] for entry in plan["archives"]] == [checksums[n] for n in names]
    assert plan["monthly_missing"] == ["2019-09"]
    assert plan["daily_missing"] == [f"2019-09-{day:02d}" for day in range(1, 29)]
    requested = fake.requested_paths()
    assert len(requested) == plan["checksum_requests"] == 2 + 30 + 2
    assert all(path.endswith(".CHECKSUM") for path in requested)
    assert not any("2019-10-" in path for path in requested)
    assert not any("2019-11-03" in path for path in requested)
    assert plan["estimated_bytes"] == 2 * 24 * 80 + 744 * 80 + 2 * 24 * 80
    assert plan["interval"] == "1h" and plan["today"] == "2019-11-03"


def test_plan_on_the_first_of_a_month_and_refusals():
    fake = fakes.FakeBinanceVision()
    client, _ = make_client(fake)
    plan = bv.plan_archives(client, "1m", today=date(2019, 10, 1))
    assert plan["archives"] == [] and plan["monthly_missing"] == ["2019-09"]
    assert len(plan["daily_missing"]) == 30
    assert not any("2019-10" in path for path in fake.requested_paths())
    with pytest.raises(ValueError, match="before the first month"):
        bv.plan_archives(client, "1m", today=date(2019, 8, 31))
    with pytest.raises(TypeError):
        bv.plan_archives(client, "1m", today=datetime(2019, 10, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        bv.plan_archives(client, "4h", today=TODAY)


def test_plan_save_and_load(tmp_path):
    plan = {
        "interval": "1m",
        "archives": [{"kind": "daily", "period": "2020-01-01", "sha256": "a" * 64}],
        "monthly_missing": [],
        "daily_missing": [],
    }
    path = bv.save_plan(plan, tmp_path)
    assert path == tmp_path / "raw" / "binance" / "plan-1m.json"
    assert bv.load_plan("1m", tmp_path) == plan
    with pytest.raises(BinanceVisionError, match="no saved plan"):
        bv.load_plan("1h", tmp_path)
    path.write_text("{")
    with pytest.raises(BinanceVisionError, match="JSON"):
        bv.load_plan("1m", tmp_path)
    overlapping = {
        **plan,
        "archives": [
            {"kind": "monthly", "period": "2020-01", "sha256": "a" * 64},
            {"kind": "daily", "period": "2020-01-31", "sha256": "b" * 64},
        ],
    }
    with pytest.raises(BinanceVisionError, match="overlap"):
        bv.save_plan(overlapping, tmp_path)
    with pytest.raises(BinanceVisionError, match="malformed"):
        bv.save_plan({**plan, "archives": [{"kind": "daily", "period": "2020-01-01"}]}, tmp_path)


# -- sync and stores -------------------------------------------------------------------------------


def september_fake(**kwargs) -> tuple[fakes.FakeBinanceVision, dict[str, pd.DataFrame]]:
    """1m dailies 2019-09-29 (header), 2019-09-30 (no header) and 2019-10-01 (microseconds)."""
    fake = fakes.FakeBinanceVision()
    frames = {}
    for offset, (day, options) in enumerate(
        (
            ("2019-09-29", {"header": True}),
            ("2019-09-30", {"header": False}),
            ("2019-10-01", {"microseconds": True}),
        )
    ):
        frame = fakes.period_frame("daily", day, seed=offset, **kwargs)
        fake.add_archive("daily", "1m", day, frame, **options)
        frames[day] = frame
    return fake, frames


def test_sync_downloads_once_and_the_build_writes_exact_stores(tmp_path):
    fake, frames = september_fake()
    client, _ = make_client(fake)
    messages: list[str] = []
    result = bv.sync_archives(client, "1m", today=TODAY, root=tmp_path, progress=messages.append)
    assert result["archives"] == 3 and result["present"] == 0
    assert [item["name"] for item in result["downloaded"]] == [
        f"BTCUSDT-1m-{day}.zip" for day in frames
    ]
    assert result["plan_relpath"] == "raw/binance/plan-1m.json"
    assert result["monthly_missing"] == ["2019-09"] and len(result["daily_missing"]) == 28
    assert result["estimated_download_bytes"] == 3 * 1440 * 80
    assert messages[0].startswith("plan 1m: 3 archives (0 present, 3 to download")
    zip_requests = [p for p in fake.requested_paths() if p.endswith(".zip")]
    assert len(zip_requests) == 3
    again = bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    assert again["present"] == 3 and again["downloaded"] == []
    assert len([p for p in fake.requested_paths() if p.endswith(".zip")]) == 3  # no re-download

    metas = bv.build_binance_stores(root=tmp_path)
    minutes = concat(frames.values())
    stored = read_store(StoreId("binance", "1m"), root=tmp_path)
    assert stored.equals(minutes)
    for target in ("5m", "15m", "1h"):
        assert read_store(StoreId("binance", target), root=tmp_path).equals(
            resample_exact(minutes, target)
        )
        provenance = metas[target]["provenance"]
        assert provenance["from_csv_sha256"] == metas["1m"]["csv_sha256"]
        assert provenance["incomplete_buckets"] == []
    assert metas["1h"]["rows"] == 72 and metas["5m"]["rows"] == 3 * 288
    provenance = metas["1m"]["provenance"]
    assert [item["name"] for item in provenance["archives"]] == [
        f"BTCUSDT-1m-{day}.zip" for day in frames
    ]
    for item in provenance["archives"]:
        local = Archive("daily", "1m", item["period"]).local_path(tmp_path)
        assert item["sha256"] == sha256(local.read_bytes())
        assert item["rows"] == 1440
    assert provenance["host"] == bv.HOST and provenance["monthly_missing"] == ["2019-09"]
    assert read_meta(StoreId("binance", "1m"), tmp_path)["csv_sha256"] == metas["1m"]["csv_sha256"]
    assert no_temp_files(tmp_path / "candles")


def test_incomplete_buckets_are_gaps_listed_with_their_minute_count(tmp_path):
    fake = fakes.FakeBinanceVision()
    day = "2019-10-01"
    frame = fakes.period_frame("daily", day)
    start = fakes.utc_ms(2019, 10, 1, 5)
    keep = ~frame["open_ms"].isin(
        [start + 7 * MINUTE_MS, start + 8 * MINUTE_MS, start + 59 * MINUTE_MS]
    )
    frame = frame[keep].reset_index(drop=True)
    fake.add_archive("daily", "1m", day, frame)
    client, _ = make_client(fake)
    bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    metas = bv.build_binance_stores(root=tmp_path)
    assert metas["1m"]["missing_candles"] == 3
    assert metas["1h"]["provenance"]["incomplete_buckets"] == [[start, 57]]
    assert metas["1h"]["gaps"] == [[start, start + HOUR_MS]]
    assert metas["15m"]["provenance"]["incomplete_buckets"] == [
        [start, 13],
        [start + 45 * MINUTE_MS, 14],
    ]
    assert metas["5m"]["provenance"]["incomplete_buckets"] == [
        [start + 5 * MINUTE_MS, 3],
        [start + 55 * MINUTE_MS, 4],
    ]
    hours = read_store(StoreId("binance", "1h"), root=tmp_path)["open_ms"].tolist()
    assert start not in hours and start - HOUR_MS in hours and start + HOUR_MS in hours


def test_build_needs_every_planned_archive_verified(tmp_path):
    fake, _ = september_fake()
    client, _ = make_client(fake)
    bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    missing = Archive("daily", "1m", "2019-09-30")
    missing.local_path(tmp_path).unlink()
    with pytest.raises(BinanceVisionError, match=r"1 planned archive\(s\).*2019-09-30"):
        bv.build_binance_stores(root=tmp_path)
    assert not StoreId("binance", "1m").csv_path(tmp_path).exists()
    with pytest.raises(BinanceVisionError, match="no saved plan"):
        bv.build_official_1h(root=tmp_path)


def test_a_malformed_archive_fails_the_build_naming_archive_and_line(tmp_path):
    fake, frames = september_fake()
    bad = Archive("daily", "1m", "2019-09-30")
    lines = lines_of(frames[bad.period], header=False)
    lines = edit_field(lines, 700, 6, "1")
    data = fakes.make_zip(bad.member_name, payload_of(lines))
    fake.add_file(bad.url_path, data)
    fake.add_file(bad.checksum_url_path, fakes.checksum_text(data, bad.name))
    client, _ = make_client(fake)
    bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)  # the checksum is fine
    with pytest.raises(ArchiveFormatError, match="BTCUSDT-1m-2019-09-30.zip line 700: close_time"):
        bv.build_binance_stores(root=tmp_path)
    for interval in ("1m", "5m", "15m", "1h"):
        assert not StoreId("binance", interval).csv_path(tmp_path).exists()
    assert no_temp_files(tmp_path / "candles")


def monthly_from(
    frames: dict[str, pd.DataFrame], changes: dict[tuple[int, str], str]
) -> pd.DataFrame:
    """The monthly archive's rows for the given days, with ``(row, column) -> value`` changes."""
    monthly = concat(frames.values())
    for (row, name), value in changes.items():
        values = monthly[name].to_numpy(dtype=object).copy()
        values[row] = value
        monthly[name] = pd.Series(values, dtype=object)
    return monthly


def test_monthly_archive_replaces_dailies_and_the_rows_are_compared_as_decimals(tmp_path):
    fake = fakes.FakeBinanceVision()
    frames = serve_days(fake, "1m", SEPT_DAYS)
    client, _ = make_client(fake)
    first = bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    assert first["monthly_missing"] == ["2019-09"]
    row = 1440 + 100  # 2019-09-30T01:40Z
    close = frames["2019-09-30"]["close"].iloc[100]
    reformatted = format(Decimal(close).quantize(Decimal("0.0001")), "f")  # same decimal
    changed_open = format(Decimal(frames["2019-09-29"]["open"].iloc[5]) + Decimal("0.01"), "f")
    monthly = monthly_from(frames, {(row, "close"): reformatted, (5, "open"): changed_open})
    fake.add_archive("monthly", "1m", "2019-09", monthly)
    second = bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    assert second["monthly_missing"] == [] and second["archives"] == 1
    assert [item["name"] for item in second["downloaded"]] == ["BTCUSDT-1m-2019-09.zip"]
    metas = bv.build_binance_stores(root=tmp_path)
    stored = read_store(StoreId("binance", "1m"), root=tmp_path)
    assert stored.equals(monthly)  # the monthly archive is what the store holds
    replaced = metas["1m"]["provenance"]["replaced_dailies"]
    assert [item["period"] for item in replaced] == list(SEPT_DAYS)
    day_29, day_30 = replaced
    assert day_30["equal"] is True and day_30["differing_values"] == 0  # 4 decimals == 2
    assert day_29["equal"] is False and day_29["differing_values"] == 1
    assert day_29["rows_compared"] == 1440 and day_29["only_in_daily"] == 0
    assert day_29["difference_samples"] == [
        {
            "open_ms": fakes.utc_ms(2019, 9, 29, 0, 5),
            "column": "open",
            "daily": frames["2019-09-29"]["open"].iloc[5],
            "monthly": changed_open,
        }
    ]


def test_replaced_dailies_in_the_sealed_segment_record_no_values(tmp_path, monkeypatch):
    monkeypatch.setattr(bv, "FIRST_MONTH", "2026-10")
    fake = fakes.FakeBinanceVision()
    days = ("2026-10-30", "2026-10-31")
    frames = serve_days(fake, "1m", days)
    client, _ = make_client(fake)
    today = date(2026, 11, 1)
    bv.sync_archives(client, "1m", today=today, root=tmp_path)
    secret = "123456.789"
    monthly = monthly_from(frames, {(10, "close"): secret})
    fake.add_archive("monthly", "1m", "2026-10", monthly)
    bv.sync_archives(client, "1m", today=today, root=tmp_path)
    metas = bv.build_binance_stores(root=tmp_path)
    replaced = metas["1m"]["provenance"]["replaced_dailies"]
    assert replaced[0]["differing_values"] == 1
    assert replaced[0]["difference_samples"] == [
        {"open_ms": fakes.utc_ms(2026, 10, 30, 0, 10), "column": "close"}
    ]
    text = json.dumps(metas["1m"]["provenance"])
    assert secret not in text and frames["2026-10-30"]["close"].iloc[10] not in text


def test_republished_archive_is_reported_and_kept_unless_refreshed(tmp_path):
    fake = fakes.FakeBinanceVision()
    frames = serve_days(fake, "1m", SEPT_DAYS)
    client, _ = make_client(fake)
    bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    archive = Archive("daily", "1m", "2019-09-30")
    original = archive.local_path(tmp_path).read_bytes()
    new_frame = monthly_from({"d": frames["2019-09-30"]}, {(3, "volume"): "999.999"})
    republished = fake.add_archive("daily", "1m", "2019-09-30", new_frame)
    messages: list[str] = []
    kept = bv.sync_archives(client, "1m", today=TODAY, root=tmp_path, progress=messages.append)
    assert kept["downloaded"] == []
    assert kept["republished"] == [
        {
            "kind": "daily",
            "period": "2019-09-30",
            "local_sha256": sha256(original),
            "remote_sha256": sha256(republished),
            "action": "kept_local",
        }
    ]
    assert any(message.startswith("republished: daily 2019-09-30") for message in messages)
    assert archive.local_path(tmp_path).read_bytes() == original
    assert bv.load_plan("1m", tmp_path)["republished"] == kept["republished"]
    metas = bv.build_binance_stores(root=tmp_path)  # builds from the kept local copy
    record = metas["1m"]["provenance"]["archives"][1]
    assert record["sha256"] == sha256(original) and record["remote_sha256"] == sha256(republished)
    assert read_store(StoreId("binance", "1m"), root=tmp_path).equals(concat(frames.values()))

    refreshed = bv.sync_archives(
        client, "1m", today=TODAY, root=tmp_path, refresh=["2019-09-30", "2019-09-01"]
    )
    assert [item["reason"] for item in refreshed["downloaded"]] == ["republished"]
    assert refreshed["republished"][0]["action"] == "refreshed"
    assert refreshed["refresh_unused"] == ["2019-09-01"]
    assert archive.local_path(tmp_path).read_bytes() == republished
    bv.build_binance_stores(root=tmp_path)
    stored = read_store(StoreId("binance", "1m"), root=tmp_path)
    assert stored["volume"].iloc[1440 + 3] == "999.999"
    with pytest.raises(ValueError, match="refresh"):
        bv.sync_archives(client, "1m", today=TODAY, root=tmp_path, refresh=["30-09-2019"])


def test_the_build_refuses_a_local_archive_that_is_not_the_planned_one(tmp_path):
    fake = fakes.FakeBinanceVision()
    frames = serve_days(fake, "1m", SEPT_DAYS)
    client, _ = make_client(fake)
    bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    archive = Archive("daily", "1m", "2019-09-29")
    other = monthly_from({"d": frames[archive.period]}, {(0, "count"): "1"})
    put_csv(tmp_path, archive, fakes.kline_csv(other, interval="1m"))  # valid, but not planned
    assert bv.verify_local(archive, tmp_path)
    with pytest.raises(BinanceVisionError, match="not the one planned"):
        bv.build_binance_stores(root=tmp_path)
    assert not StoreId("binance", "1m").csv_path(tmp_path).exists()


def test_a_local_archive_that_fails_its_checksum_is_downloaded_again(tmp_path):
    fake = fakes.FakeBinanceVision()
    serve_days(fake, "1m", SEPT_DAYS)
    client, _ = make_client(fake)
    bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    archive = Archive("daily", "1m", "2019-09-29")
    good = archive.local_path(tmp_path).read_bytes()
    archive.local_path(tmp_path).write_bytes(b"truncated")
    result = bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    assert result["downloaded"] == [{"name": archive.name, "bytes": len(good), "reason": "invalid"}]
    assert archive.local_path(tmp_path).read_bytes() == good


def test_gate_4_refuses_the_run_before_any_download(tmp_path, monkeypatch):
    fake = fakes.FakeBinanceVision()
    serve_days(fake, "1m", SEPT_DAYS)
    client, _ = make_client(fake)
    probe_root = tmp_path / "probe"
    probe = bv.sync_archives(client, "1m", today=TODAY, root=probe_root)
    needed = probe["disk_bytes_before"] + probe["estimated_download_bytes"]
    assert probe["estimated_download_bytes"] == 2 * 1440 * 80

    refused_root = tmp_path / "refused"
    monkeypatch.setattr(bv, "BUDGET_BYTES", needed - 1)
    fake.requests.clear()
    with pytest.raises(DownloadBudgetExceeded, match="nothing downloaded"):
        bv.sync_archives(client, "1m", today=TODAY, root=refused_root)
    assert not any(path.endswith(".zip") for path in fake.requested_paths())
    assert not (refused_root / "raw" / "binance" / "daily").exists()
    assert (refused_root / bv.plan_relpath("1m")).is_file()  # the plan is saved first

    allowed_root = tmp_path / "allowed"
    monkeypatch.setattr(bv, "BUDGET_BYTES", needed)
    result = bv.sync_archives(client, "1m", today=TODAY, root=allowed_root)
    assert len(result["downloaded"]) == 2 and result["budget_limit_bytes"] == needed

    # bytes already on disk count: the same run no longer fits once they are there
    crowded_root = tmp_path / "crowded"
    (crowded_root / "raw" / "binance").mkdir(parents=True)
    (crowded_root / "raw" / "binance" / "old.bin").write_bytes(b"x" * 1000)
    with pytest.raises(DownloadBudgetExceeded):
        bv.sync_archives(client, "1m", today=TODAY, root=crowded_root)


def test_sync_intervals_plans_every_interval_and_checks_gate_4_once(tmp_path, monkeypatch):
    """binance-1: one estimate for the whole run, checked before any download of any interval;
    the downloads of every interval then charge one budget."""
    fake = fakes.FakeBinanceVision()
    serve_days(fake, "1m", SEPT_DAYS)
    serve_days(fake, "1h", SEPT_DAYS)
    client, _ = make_client(fake)
    probe = bv.sync_intervals(client, ["1m", "1h"], today=TODAY, root=tmp_path / "probe")
    disk = probe[0]["disk_bytes_before"]
    estimate_1m, estimate_1h = (item["estimated_download_bytes"] for item in probe)
    assert (estimate_1m, estimate_1h) == (2 * 1440 * 80, 2 * 24 * 80)
    assert probe[1]["run_estimated_download_bytes"] == estimate_1m + estimate_1h
    downloaded = sum(item["downloaded_bytes"] for item in probe)
    assert probe[1]["budget_used_bytes"] == disk + downloaded  # one budget for both intervals

    fake.requests.clear()
    messages: list[str] = []
    refused = tmp_path / "refused"
    monkeypatch.setattr(bv, "BUDGET_BYTES", disk + estimate_1m)  # each interval alone fits
    with pytest.raises(DownloadBudgetExceeded, match=r"run \(1m, 1h\).*nothing downloaded"):
        bv.sync_intervals(client, ["1m", "1h"], today=TODAY, root=refused, progress=messages.append)
    assert not any(path.endswith(".zip") for path in fake.requested_paths())
    assert not list(refused.rglob("*.zip"))
    assert [m.split(":")[0] for m in messages if m.startswith("plan ")] == ["plan 1m", "plan 1h"]
    assert messages[-1].startswith("gate 4: estimated download 0.2 MB for 1m, 1h, ")
    for interval in ("1m", "1h"):  # both planned before the check
        assert (refused / bv.plan_relpath(interval)).is_file()
    with pytest.raises(ValueError, match="distinct"):
        bv.sync_intervals(client, ["1m", "1m"], today=TODAY, root=refused)
    with pytest.raises(TypeError):
        bv.sync_intervals(client, "1m", today=TODAY, root=refused)


def test_the_sync_removes_the_temporary_files_of_killed_runs(tmp_path):
    fake = fakes.FakeBinanceVision()
    serve_days(fake, "1m", SEPT_DAYS)
    client, _ = make_client(fake)
    name = ".BTCUSDT-1m-2019-09-29.zip.k1ll3d00.part"
    leftover = tmp_path / "raw" / "binance" / "daily" / "1m" / name
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b"partial")
    result = bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    assert result["removed_temporaries"] == [
        {"relpath": f"raw/binance/daily/1m/{name}", "bytes": 7}
    ]
    assert not leftover.exists() and len(result["downloaded"]) == 2


def test_sync_and_builds_refuse_to_run_concurrently(tmp_path):
    fake = fakes.FakeBinanceVision()
    serve_days(fake, "1m", SEPT_DAYS)
    client, _ = make_client(fake)
    with exclusive_lock(tmp_path / bv.LOCK_RELPATH):
        with pytest.raises(LockBusyError):
            bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
        with pytest.raises(LockBusyError):
            bv.build_binance_stores(root=tmp_path)
        with pytest.raises(LockBusyError):
            bv.build_official_1h(root=tmp_path)
    assert fake.requests == []
    bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)  # released


# -- the resample check ----------------------------------------------------------------------------


def rounded(value: str, places: str = "0.01") -> str:
    return format(Decimal(value).quantize(Decimal(places), rounding=ROUND_HALF_EVEN), "f")


def test_check_resample_end_to_end_without_sealed_values(tmp_path, monkeypatch):
    # 2026-07-31 (before the sealed start) and 2026-08-01 (sealed), 1m dailies and 1h archives.
    monkeypatch.setattr(bv, "FIRST_MONTH", "2026-07")
    today = date(2026, 8, 2)
    fake = fakes.FakeBinanceVision()
    days = ("2026-07-31", "2026-08-01")
    frames = serve_days(fake, "1m", days, seed=3)
    gap_hour = fakes.utc_ms(2026, 7, 31, 6)
    july = frames["2026-07-31"]
    keep = ~july["open_ms"].isin([gap_hour + 11 * MINUTE_MS])
    frames["2026-07-31"] = july[keep].reset_index(drop=True)
    fake.add_archive("daily", "1m", "2026-07-31", frames["2026-07-31"])
    official = {}
    for day in days:
        hourly = resample_exact(fakes.period_frame("daily", day, seed=3 + days.index(day)), "1h")
        quotes = [rounded(value) for value in hourly["quote_volume"]]
        hourly["quote_volume"] = pd.Series(quotes, dtype=object)  # within the tolerance
        official[day] = hourly
    pre_hour = fakes.utc_ms(2026, 7, 31, 2)
    sealed_hour = fakes.utc_ms(2026, 8, 1, 3)
    secret_close = "87654.3219"
    july_official = monthly_from({"d": official["2026-07-31"]}, {(2, "close"): "11111.1"})
    august_official = monthly_from({"d": official["2026-08-01"]}, {(3, "close"): secret_close})
    august_official = august_official[august_official["open_ms"] != fakes.utc_ms(2026, 8, 1, 9)]
    fake.add_archive("monthly", "1h", "2026-07", july_official.reset_index(drop=True))
    fake.add_archive("daily", "1h", "2026-08-01", august_official.reset_index(drop=True))
    client, _ = make_client(fake)
    bv.sync_archives(client, "1m", today=today, root=tmp_path)
    bv.sync_archives(client, "1h", today=today, root=tmp_path)
    bv.build_binance_stores(root=tmp_path)
    official_meta = bv.build_official_1h(root=tmp_path)
    assert official_meta["rows"] == 47

    reports = tmp_path / "reports"
    report = bv.check_resample(root=tmp_path, directory=reports)
    assert set(REQUIRED_KEYS) <= set(report)
    assert report["study"] == bv.RESAMPLE_CHECK_STEM and report["tests_consumed"] == 0
    assert (reports / f"{bv.RESAMPLE_CHECK_STEM}.config.json").is_file()
    comparison = report["comparison"]
    assert comparison["hours_compared"] == 46  # 48 - the incomplete hour - the missing official
    assert comparison["hours_compared_sealed"] == 23
    assert comparison["columns"]["close"]["mismatches_before_sealed"] == 1
    assert comparison["columns"]["close"]["mismatches_sealed"] == 1
    assert comparison["columns"]["quote_volume"]["mismatches"] == 0
    assert comparison["exact_mismatches"] == 2 and report["owner_attention"] is True
    assert [item["open_ms"] for item in comparison["mismatch_samples"]] == [pre_hour]
    assert comparison["sealed_mismatches"] == [
        {"open_ms": sealed_hour, "hour": "2026-08-01T03:00Z", "column": "close"}
    ]
    assert report["only_official"] == [
        {"open_ms": gap_hour, "hour": "2026-07-31T06:00Z", "minutes": 59, "sealed": False}
    ]
    assert report["only_resampled"] == [
        {
            "open_ms": fakes.utc_ms(2026, 8, 1, 9),
            "hour": "2026-08-01T09:00Z",
            "sealed": True,
        }
    ]
    assert report["incomplete_hours"]["count"] == 1
    assert report["incomplete_hours"]["by_segment"]["embargo_2"] == 1
    assert report["samples"]["hours_compared"] == 46
    markdown = bv.render_resample_markdown(report)
    assert markdown.isascii() and "reported to the owner" in markdown
    assert "11111.1" in markdown  # the pre-sealed mismatch is shown with its values
    # decision hours lost for the 512 h Kronos context only (no 100 h profile window, D-042);
    # two days of minutes hold no complete context
    assert list(report["lost_decision_hours"]) == ["512h"]
    assert "\n| segment | incomplete hours | eligible (512h) | lost (512h) |\n" in markdown
    assert "\n| embargo_2 | 1 | 0 | 0 |\n" in markdown
    assert "100h" not in markdown and "profile" not in markdown
    text = json.dumps(report)
    sealed_day = frames["2026-08-01"]  # the resampled close of the sealed hour, never shown
    resampled_close = resample_exact(sealed_day, "1h")["close"].iloc[3]
    difference = format(Decimal(resampled_close) - Decimal(secret_close), "f")
    for forbidden in (secret_close, difference, difference.lstrip("-")):
        assert forbidden not in text and forbidden not in markdown
    json_path, md_path = write_report(bv.RESAMPLE_CHECK_STEM, report, markdown, directory=reports)
    assert md_path.read_text() == markdown
    written = json.loads(json_path.read_text())
    assert bv.render_resample_markdown(written) == markdown  # re-rendered from the JSON alone
    again = bv.check_resample(root=tmp_path, directory=reports)  # same recorded config
    assert again["config_sha256"] == report["config_sha256"]


def test_check_resample_refuses_stale_resampled_stores(tmp_path):
    fake, _ = september_fake()
    one_hour = Archive("daily", "1h", "2019-09-29")
    fake.add_archive(
        "daily", "1h", one_hour.period, fakes.period_frame("daily", one_hour.period, interval="1h")
    )
    client, _ = make_client(fake)
    bv.sync_archives(client, "1m", today=TODAY, root=tmp_path)
    bv.sync_archives(client, "1h", today=TODAY, root=tmp_path)
    bv.build_binance_stores(root=tmp_path)
    bv.build_official_1h(root=tmp_path)
    meta_path = StoreId("binance", "1h").meta_path(tmp_path)
    meta = json.loads(meta_path.read_text())
    meta["provenance"]["from_csv_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(StoreIntegrityError, match="not resampled from the current"):
        bv.check_resample(root=tmp_path, directory=tmp_path / "reports")


@pytest.mark.parametrize(
    ("first_hour", "hours", "incomplete", "expected"),
    [
        # all decision hours inside validation: a 512 h window loses j in [512, 612]
        (to_ms(VALIDATION_START), 700, [100], {"512h": {"validation": (189, 101)}}),
        # hour 588 is lost for j in [589, 1100]; the last one, j = 1100, is D = TRAINING_END,
        # which belongs to embargo_1 (half-open segments), and its window starts at hour 588
        (
            to_ms(TRAINING_END) - 1100 * HOUR_MS,
            1200,
            [588],
            {"512h": {"training": (588, 511), "embargo_1": (101, 1)}},
        ),
        # fewer hours than one context: no decision hour is eligible
        (to_ms(TRAINING_END) - 300 * HOUR_MS, 400, [250], {"512h": {}}),
    ],
)
def test_lost_decision_hours_per_segment(first_hour, hours, incomplete, expected):
    opens = np.array([first_hour + k * HOUR_MS for k in incomplete], dtype=np.int64)
    result = bv._lost_decision_hours(opens, first_hour, first_hour + (hours - 1) * HOUR_MS)
    assert list(result) == ["512h"]  # the Kronos context only: no profile window (D-042)
    for window, segments in expected.items():
        counts = {
            name: (item["eligible"], item["lost"])
            for name, item in result[window].items()
            if item["eligible"]
        }
        assert counts == segments


def test_resample_check_config_is_publishable():
    config = bv.resample_check_config()
    assert config["study"] == bv.RESAMPLE_CHECK_STEM
    assert config["sealed_start"] == "2026-08-01T00:00Z"
    assert SEALED_START_MS == to_ms(datetime(2026, 8, 1, tzinfo=UTC))
    assert bv.LOST_HOURS_WINDOWS == (512,)
    assert config["lost_decision_hours"]["windows_hours"] == [512]  # the Kronos context only
    assert json.dumps(config).isascii()
    assert bv.resample_check_config() == config  # the recorded definition is stable
