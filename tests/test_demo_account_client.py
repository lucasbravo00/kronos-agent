"""Host-refusal matrix, signing and redaction of the demo account client.

Every request here goes to a mock transport: gate 1 (SPEC section 17) stays closed and the
``conftest`` socket guard would refuse any real connection anyway. The live host appears only as
a value that must be refused.
"""

from __future__ import annotations

import inspect
import logging
import ssl
from typing import Any

import httpx
import pytest

from kronos_agent.exchange import demo_account as da
from kronos_agent.exchange.demo_account import (
    Credentials,
    DemoAccountClient,
    DemoApiError,
    DemoSchemaError,
    DemoTransportError,
    HostNotAllowedError,
    MissingCredentials,
    PositionsQueryFailed,
    UnexpectedRedirect,
    _Secret,
)

DEMO_HOST = "open-api-vst.bingx.com"
LIVE_HOST = "open-api.bingx.com"
API_KEY = "unit-test-api-key-0001"
API_SECRET = "unit-test-secret-value-0001"
VECTOR_SECRET = "SECRET_KEY"
VECTOR_QUERY = "recvWindow=0&symbol=BTC-USDT&timestamp=1696751141337"
VECTOR_SIGNATURE = "fe041f159118c90ac13eab4d32f9e2d75b80ca6fe17ca8acd290aba864753ce2"
BALANCE_ITEM = {
    "userId": "0",
    "asset": "USDT",
    "balance": "0",
    "equity": "0",
    "unrealizedProfit": "0",
    "realisedProfit": "0",
    "availableMargin": "0",
    "usedMargin": "0",
    "freezedMargin": "0",
}

REFUSED_URLS = [
    f"https://{LIVE_HOST}/openApi/swap/v3/user/balance",
    "https://open-api-vst.bingx.pro/openApi/swap/v3/user/balance",
    f"http://{DEMO_HOST}/openApi/swap/v3/user/balance",
    f"https://{DEMO_HOST}:8443/openApi/swap/v3/user/balance",
    f"https://{DEMO_HOST}.evil.invalid/openApi/swap/v3/user/balance",
    f"https://{DEMO_HOST}@evil.invalid/openApi/swap/v3/user/balance",
    f"https://user:pw@{DEMO_HOST}/openApi/swap/v3/user/balance",
    f"https://evil.invalid/?u={DEMO_HOST}",
    f"{DEMO_HOST}/openApi/swap/v3/user/balance",
    f"//{DEMO_HOST}/openApi/swap/v3/user/balance",
    "",
    "https://xn--80ak6aa92e.com/openApi/swap/v3/user/balance",  # valid punycode homograph
    f"https://{DEMO_HOST}./openApi/swap/v3/user/balance",  # trailing dot
]
# Well-formed ASCII but invalid A-labels: ``httpx.URL.host`` IDNA-decodes them and raises an
# ``idna`` error, so the check must never read it.
INVALID_A_LABEL_HOSTS = ["xn--zzzzzzzzz.bingx.com", "xn--a.bingx.com"]
POSITIONS = "/openApi/swap/v2/user/positions"


def ok(data: Any) -> httpx.Response:
    return httpx.Response(200, json={"code": 0, "msg": "", "data": data})


class Recorder:
    """Mock handler recording every request that reaches the (inner) transport."""

    def __init__(self, response: httpx.Response | None = None):
        self.requests: list[httpx.Request] = []
        self.response = response if response is not None else ok([])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response


@pytest.fixture
def creds() -> Credentials:
    return Credentials(API_KEY, _Secret(API_SECRET))


def make_client(creds: Credentials, recorder: Recorder, **kwargs: Any) -> DemoAccountClient:
    return DemoAccountClient(creds, transport=httpx.MockTransport(recorder), **kwargs)


# -- no host parameter, pinned constant ---------------------------------------------------------------


def test_constructor_signature_has_no_host_base_url_or_url_parameter():
    signature = inspect.signature(DemoAccountClient)
    names = list(signature.parameters)
    assert names == ["credentials", "transport", "timeout", "recv_window_ms", "clock"]
    for name in names:
        assert not any(word in name.lower() for word in ("host", "url", "base", "header"))
    for name in names[1:]:
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["timeout"].default == 10.0
    assert signature.parameters["recv_window_ms"].default == 5000
    assert da.DEMO_HOST == DEMO_HOST


def test_monkeypatched_constant_before_construction_refuses_to_start(creds, monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(da, "DEMO_HOST", LIVE_HOST)
    with pytest.raises(HostNotAllowedError):
        make_client(creds, recorder)
    assert recorder.requests == []


@pytest.mark.parametrize("host", [LIVE_HOST, *INVALID_A_LABEL_HOSTS])
def test_monkeypatched_constant_after_construction_is_refused_before_sending(
    creds, monkeypatch, host
):
    recorder = Recorder()
    client = make_client(creds, recorder)
    monkeypatch.setattr(da, "DEMO_HOST", host)
    with pytest.raises(HostNotAllowedError) as info:
        client.balance()
    assert recorder.requests == []
    # the query is signed before the URL is asserted; the refusal carries none of it
    text = str(info.value)
    for forbidden in ("signature=", "timestamp=", "recvWindow=", API_KEY, API_SECRET):
        assert forbidden not in text, forbidden


class _Relay(httpx.BaseTransport):
    """A network-capable transport that would forward the signed request to another host."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.url = request.url.copy_with(host=LIVE_HOST)
        raise AssertionError("never reached: the constructor refuses this transport")


class _SubMock(httpx.MockTransport):
    """Only the exact ``httpx.MockTransport`` class is accepted, not subclasses."""


@pytest.mark.parametrize(
    "factory",
    [
        lambda: httpx.HTTPTransport(verify=False, trust_env=False),
        lambda: httpx.HTTPTransport(proxy="http://proxy.invalid:8080", trust_env=False),
        lambda: httpx.HTTPTransport(trust_env=False),
        _Relay,
        lambda: _SubMock(Recorder()),
        lambda: httpx.WSGITransport(app=lambda env, start: []),
    ],
    ids=["tls-unverified", "proxy", "default-like", "relay", "mock-subclass", "wsgi"],
)
def test_only_mock_transports_can_be_injected(creds, factory):
    transport = factory()
    try:
        with pytest.raises(TypeError, match="MockTransport"):
            DemoAccountClient(creds, transport=transport)
    finally:
        transport.close()


def test_default_transport_is_built_here_tls_verified_and_direct(creds, monkeypatch, tmp_path):
    # trust_env=False on the inner transport: with trust_env=True httpx would load these missing
    # CA paths while building it and fail.
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "missing"))
    # ssl.create_default_context would log every TLS session secret to this file
    monkeypatch.setenv("SSLKEYLOGFILE", str(tmp_path / "keys.log"))
    with DemoAccountClient(creds) as client:
        assert type(client._transport) is da._DemoPinnedTransport
        inner = client._transport._inner
        assert type(inner) is httpx.HTTPTransport
        pool = inner._pool  # private attributes of the pinned httpx 0.28 / httpcore 1.0
        assert type(pool).__name__ == "ConnectionPool" and pool._proxy is None  # not HTTPProxy
        assert pool._ssl_context.verify_mode == ssl.CERT_REQUIRED
        assert pool._ssl_context.check_hostname is True
        assert pool._ssl_context.keylog_filename is None
        assert pool._retries == 0
    assert not (tmp_path / "keys.log").exists()


# -- the refusal matrix at both layers ------------------------------------------------------------------


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_pinned_transport_refuses(url):
    inner = Recorder()
    transport = da._DemoPinnedTransport(httpx.MockTransport(inner))
    with pytest.raises(HostNotAllowedError):
        transport.handle_request(httpx.Request("GET", url))
    assert inner.requests == []


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_request_hook_refuses_absolute_urls_through_the_inner_client(creds, url):
    recorder = Recorder()
    client = make_client(creds, recorder)
    with pytest.raises(HostNotAllowedError):
        client._http.get(url)
    assert recorder.requests == []


@pytest.mark.parametrize("host", INVALID_A_LABEL_HOSTS)
def test_invalid_punycode_host_is_refused_as_host_not_allowed(creds, host):
    url = f"https://{host}/openApi/swap/v3/user/balance"
    with pytest.raises(HostNotAllowedError):
        da._check_demo_url(httpx.URL(url))
    # httpx itself reads url.host while building a Request unless a Host header is given, so
    # this is the only way such a request can reach the hook and the transport
    inner = Recorder()
    transport = da._DemoPinnedTransport(httpx.MockTransport(inner))
    with pytest.raises(HostNotAllowedError):
        transport.handle_request(httpx.Request("GET", url, headers={"Host": "x"}))
    assert inner.requests == []
    recorder = Recorder()
    client = make_client(creds, recorder)
    with pytest.raises(HostNotAllowedError):
        client._http.get(url, headers={"Host": "x"})
    assert recorder.requests == []


def test_literal_host_check_runs_at_three_points_of_every_request(creds, monkeypatch):
    seen: list[str] = []
    original = da._check_demo_url

    def spy(url: httpx.URL) -> None:
        seen.append(url.raw_host.decode("ascii"))
        original(url)

    monkeypatch.setattr(da, "_check_demo_url", spy)
    recorder = Recorder(ok([BALANCE_ITEM]))
    make_client(creds, recorder).balance()
    assert seen == [DEMO_HOST] * 3  # pre-send assertion, request hook, pinned transport
    assert len(recorder.requests) == 1


def test_hook_is_installed_and_transport_is_the_pinned_wrapper(creds):
    recorder = Recorder()
    client = make_client(creds, recorder)
    assert da._request_hook in client._http.event_hooks["request"]
    assert isinstance(client._transport, da._DemoPinnedTransport)
    assert client._http.follow_redirects is False
    assert client._http.trust_env is False
    assert str(client._http.base_url) == ""


def test_demo_host_passes_both_layers(creds):
    recorder = Recorder(ok([BALANCE_ITEM]))
    client = make_client(creds, recorder)
    assert client.balance() == [BALANCE_ITEM]
    request = recorder.requests[0]
    assert request.url.scheme == "https"
    assert request.url.host == DEMO_HOST
    assert request.url.port is None
    assert request.url.userinfo == b""
    assert request.url.path == "/openApi/swap/v3/user/balance"
    assert request.method == "GET"


def test_redirect_to_live_host_is_not_followed(creds):
    redirect = httpx.Response(
        302, headers={"location": f"https://{LIVE_HOST}/openApi/swap/v3/user/balance"}
    )
    recorder = Recorder(redirect)
    client = make_client(creds, recorder)
    with pytest.raises(UnexpectedRedirect):
        client.balance()
    assert len(recorder.requests) == 1
    assert recorder.requests[0].url.host == DEMO_HOST


@pytest.mark.parametrize(
    "location",
    ["https://xn--zzzzzzzzz.bingx.com/openApi/swap/v3/user/balance", "https:x"],
    ids=["invalid-a-label", "no-host"],
)
def test_redirect_with_an_unparseable_location_is_refused_without_chaining(creds, location):
    # httpx builds the never-followed redirect request while sending and fails outside HTTPError
    recorder = Recorder(httpx.Response(302, headers={"location": location}))
    client = make_client(creds, recorder)
    with pytest.raises(UnexpectedRedirect, match="unparseable Location") as excinfo:
        client.balance()
    assert len(recorder.requests) == 1
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None


# -- environment ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["BINGX_HOST", "BINGX_BASE_URL"])
@pytest.mark.parametrize(
    "value",
    [
        LIVE_HOST,
        f"https://{LIVE_HOST}",
        "open-api-vst.bingx.pro",
        f"http://{DEMO_HOST}",
        f"https://{DEMO_HOST}:8443",
        "",
    ],
)
def test_env_host_override_pointing_elsewhere_refuses_to_start(creds, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    recorder = Recorder()
    with pytest.raises(HostNotAllowedError):
        make_client(creds, recorder)
    assert recorder.requests == []


@pytest.mark.parametrize("name", ["BINGX_HOST", "BINGX_BASE_URL"])
@pytest.mark.parametrize("value", [DEMO_HOST, f"https://{DEMO_HOST}", f"https://{DEMO_HOST}/"])
def test_env_host_override_equal_to_demo_host_changes_nothing(creds, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    recorder = Recorder()
    client = make_client(creds, recorder)
    client.balance()
    assert recorder.requests[0].url.host == DEMO_HOST


def test_proxy_environment_is_ignored(creds, monkeypatch):
    # httpx reads HTTP(S)_PROXY / ALL_PROXY only when trust_env is true AND no transport is given
    # (httpx 0.28 Client.__init__: allow_env_proxies = trust_env and transport is None). This
    # client always passes its pinned transport, so the variables are unreachable by construction.
    # trust_env=False on the Client is the second layer (it would keep them out if the explicit
    # transport were ever dropped), and _mounts is the routing table httpx consults: {} states
    # directly that nothing can divert a request past the pinned transport.
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.invalid:1080")
    recorder = Recorder()
    client = make_client(creds, recorder)
    client.balance()
    assert recorder.requests[0].url.host == DEMO_HOST
    assert client._http._mounts == {}
    with DemoAccountClient(creds) as default_client:
        assert default_client._http._mounts == {}
        assert default_client._http.trust_env is False


# -- signing --------------------------------------------------------------------------------------------------


def test_signature_vector():
    assert da._hmac_sha256_hex(VECTOR_SECRET, VECTOR_QUERY) == VECTOR_SIGNATURE


def test_signature_vector_end_to_end():
    recorder = Recorder(ok([]))
    client = make_client(
        Credentials(API_KEY, _Secret(VECTOR_SECRET)),
        recorder,
        recv_window_ms=0,
        clock=lambda: 1696751141.337,
    )
    client.positions("BTC-USDT")
    request = recorder.requests[0]
    assert request.url.query.decode() == f"{VECTOR_QUERY}&signature={VECTOR_SIGNATURE}"
    assert request.headers["X-BX-APIKEY"] == API_KEY
    assert request.url.path == "/openApi/swap/v2/user/positions"


def test_parameters_sorted_in_ascii_order_with_signature_last(creds):
    recorder = Recorder(ok([]))
    client = make_client(creds, recorder, clock=lambda: 1700000000.0)
    client._signed_request(
        "GET", "/openApi/swap/v2/user/positions", {"symbol": "BTC-USDT", "Zeta": "1", "alpha": 2}
    )
    query = recorder.requests[0].url.query.decode()
    keys = [pair.split("=", 1)[0] for pair in query.split("&")]
    assert keys == ["Zeta", "alpha", "recvWindow", "symbol", "timestamp", "signature"]
    unsigned = query.rsplit("&signature=", 1)[0]
    assert unsigned == "Zeta=1&alpha=2&recvWindow=5000&symbol=BTC-USDT&timestamp=1700000000000"
    assert query.endswith("&signature=" + da._hmac_sha256_hex(API_SECRET, unsigned))


def test_bracket_values_are_url_encoded_on_the_wire_but_signed_raw(creds):
    recorder = Recorder(ok([]))
    client = make_client(creds, recorder, clock=lambda: 1700000000.0)
    raw = '[{"symbol":"BTC-USDT"}]'
    client._signed_request("GET", "/openApi/swap/v2/user/positions", {"batchOrders": raw})
    query = recorder.requests[0].url.query.decode()
    assert query.startswith(
        "batchOrders=%5B%7B%22symbol%22%3A%22BTC-USDT%22%7D%5D&recvWindow=5000&timestamp=1700000000000"
    )
    signed_string = f"batchOrders={raw}&recvWindow=5000&timestamp=1700000000000"
    assert query.endswith("&signature=" + da._hmac_sha256_hex(API_SECRET, signed_string))
    assert "[" not in query and "{" not in query


def test_printable_json_values_still_pass_signed_raw_and_fully_encoded(creds):
    recorder = Recorder(ok([]))
    client = make_client(creds, recorder, clock=lambda: 1700000000.0)
    raw = '{"type": "STOP_MARKET", "stopPrice": 31000, "workingType": "MARK_PRICE"}'
    client._signed_request("GET", POSITIONS, {"takeProfit": raw})
    query = recorder.requests[0].url.query.decode("ascii")
    assert not any(char in query for char in '[{" :,')
    signed_string = f"recvWindow=5000&takeProfit={raw}&timestamp=1700000000000"
    assert query.endswith("&signature=" + da._hmac_sha256_hex(API_SECRET, signed_string))


@pytest.mark.parametrize(
    "value",
    ["BTC#USDT", "BTC+USDT", "BTC%41USDT", "100%", "BTC USDT", "BTC\u00e9", "a\nb", 'a"b', ""],
)
def test_plain_values_outside_the_safe_alphabet_raise_before_any_request(creds, value):
    recorder = Recorder()
    client = make_client(creds, recorder)
    with pytest.raises(ValueError, match="parameter 'symbol'") as info:
        client._signed_request("GET", POSITIONS, {"symbol": value})
    assert not isinstance(info.value, UnicodeError)  # the validation, not an encoding accident
    assert recorder.requests == []


def test_positions_symbol_is_validated_before_any_request(creds):
    recorder = Recorder()
    client = make_client(creds, recorder)
    with pytest.raises(ValueError, match="parameter 'symbol'"):
        client.positions("BTC#USDT")  # an unwrapped httpx.InvalidURL before the fix
    assert recorder.requests == []


@pytest.mark.parametrize("key", ["sym#bol", "a=b", "a&b", "", "sym bol", "sym\u00e9"])
def test_parameter_names_are_restricted_too(creds, key):
    recorder = Recorder()
    client = make_client(creds, recorder)
    with pytest.raises(ValueError, match="parameter name"):
        client._signed_request("GET", POSITIONS, {key: "1"})
    assert recorder.requests == []


@pytest.mark.parametrize("raw", ['{"a":"caf\u00e9"}', '{"a":"x\ty"}', '[{"a":"\x00"}]'])
def test_json_values_must_be_printable_ascii(creds, raw):
    recorder = Recorder()
    client = make_client(creds, recorder)
    with pytest.raises(ValueError, match="parameter 'batchOrders'"):
        client._signed_request("GET", POSITIONS, {"batchOrders": raw})
    assert recorder.requests == []


def test_timestamp_and_recv_window_come_from_injection(creds):
    recorder = Recorder(ok([]))
    client = make_client(creds, recorder, recv_window_ms=7000, clock=lambda: 1700000000.0)
    client.balance()
    params = recorder.requests[0].url.params
    assert params["timestamp"] == "1700000000000"
    assert params["recvWindow"] == "7000"
    assert len(params["signature"]) == 64 and params["signature"] == params["signature"].lower()


@pytest.mark.parametrize(
    "recv_window_ms, error",
    [(-1, ValueError), (60_001, ValueError), ("5000", TypeError), (True, TypeError)],
)
def test_recv_window_validation(creds, recv_window_ms, error):
    with pytest.raises(error):
        make_client(creds, Recorder(), recv_window_ms=recv_window_ms)


# -- credentials and redaction ----------------------------------------------------------------------------


def test_credentials_come_only_from_the_two_env_names(monkeypatch):
    monkeypatch.setenv("BINGX_API_KEY", API_KEY)
    monkeypatch.setenv("BINGX_API_SECRET", API_SECRET)
    creds = Credentials.from_env()
    assert creds.api_key == API_KEY
    assert creds.api_secret.reveal() == API_SECRET

    monkeypatch.delenv("BINGX_API_SECRET")
    with pytest.raises(MissingCredentials) as info:
        Credentials.from_env()
    assert "BINGX_API_SECRET" in str(info.value)
    assert API_KEY not in str(info.value) and API_SECRET not in str(info.value)

    monkeypatch.setenv("BINGX_API_SECRET", "")
    with pytest.raises(MissingCredentials):
        Credentials.from_env()

    monkeypatch.delenv("BINGX_API_KEY")
    monkeypatch.setenv("BINGX_API_SECRET", API_SECRET)
    with pytest.raises(MissingCredentials) as info:
        Credentials.from_env()
    assert "BINGX_API_KEY" in str(info.value) and API_SECRET not in str(info.value)


def test_credentials_reject_raw_secret_strings_and_empty_keys():
    with pytest.raises(TypeError):
        Credentials(API_KEY, API_SECRET)
    with pytest.raises(MissingCredentials):
        Credentials("", _Secret(API_SECRET))
    with pytest.raises(MissingCredentials):
        _Secret("")


def test_secret_wrapper_redacts():
    secret = _Secret(API_SECRET)
    assert repr(secret) == "<redacted>" and str(secret) == "<redacted>"
    assert secret.reveal() == API_SECRET
    assert API_SECRET not in f"{secret!r}{secret}"
    creds = Credentials(API_KEY, secret)
    assert repr(creds) == "Credentials(api_key=<redacted>, api_secret=<redacted>)"


def test_client_repr_redacts(creds):
    client = make_client(creds, Recorder())
    text = repr(client)
    assert "<redacted>" in text
    assert API_KEY not in text and API_SECRET not in text
    assert DEMO_HOST in text


def test_secret_absent_from_exception_url_headers_and_repr_on_4xx(creds):
    recorder = Recorder(httpx.Response(401, json={"code": 100413, "msg": "Incorrect apiKey"}))
    client = make_client(creds, recorder)
    with pytest.raises(DemoApiError) as info:
        client.balance()
    exc = info.value
    assert exc.status == 401 and exc.code == 100413 and exc.path == "/openApi/swap/v3/user/balance"
    text = f"{exc!s} {exc!r}"
    for forbidden in (
        API_SECRET,
        API_KEY,
        "signature=",
        "timestamp=",
        "https://",
        "?",
        "X-BX-APIKEY",
    ):
        assert forbidden not in text
    request = recorder.requests[0]
    assert API_SECRET not in str(request.url)
    assert API_KEY not in str(request.url)
    assert all(API_SECRET not in value for value in request.headers.values())
    assert request.headers["X-BX-APIKEY"] == API_KEY
    assert API_SECRET not in repr(client) and API_SECRET not in repr(creds)


def test_plain_text_server_error_gives_code_none(creds):
    client = make_client(creds, Recorder(httpx.Response(500, text="Internal Server Error")))
    with pytest.raises(DemoApiError) as info:
        client.balance()
    assert info.value.status == 500 and info.value.code is None
    assert "Internal Server Error" in str(info.value)


def test_json_code_non_zero_raises_with_truncated_message(creds):
    client = make_client(
        creds, Recorder(httpx.Response(200, json={"code": 100001, "msg": "x" * 500}))
    )
    with pytest.raises(DemoApiError) as info:
        client.balance()
    assert info.value.code == 100001 and info.value.status == 200
    assert len(info.value.msg) == 200


@pytest.mark.parametrize(
    "error", [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError]
)
def test_transport_errors_hide_the_url(creds, error):
    def boom(request: httpx.Request) -> httpx.Response:
        raise error("connection failed", request=request)

    client = DemoAccountClient(creds, transport=httpx.MockTransport(boom))
    with pytest.raises(DemoTransportError) as info:
        client.balance()
    text = str(info.value)
    assert error.__name__ in text and "/openApi/swap/v3/user/balance" in text
    assert "https://" not in text and "signature" not in text and API_SECRET not in text
    # no reference to the httpx exception survives: its .request holds the signed URL and the key
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_undecodable_body_is_a_module_error_without_the_url(creds):
    # stream=, not content=: the body must be decoded by Client.send, as on the wire
    bad_gzip = httpx.Response(
        200, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(b"this is not gzip")
    )
    client = DemoAccountClient(creds, transport=httpx.MockTransport(lambda request: bad_gzip))
    with pytest.raises(DemoTransportError) as info:
        client.balance()
    text = str(info.value)
    assert "DecodingError" in text and "/openApi/swap/v3/user/balance" in text
    assert "https://" not in text and "signature" not in text
    assert API_KEY not in text and API_SECRET not in text
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_httpx_request_log_line_with_the_signed_url_reaches_no_handler(creds, caplog):
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="httpx")
    recorder = Recorder(ok([BALANCE_ITEM]))
    client = make_client(creds, recorder, clock=lambda: 1700000000.0)
    client.balance()
    client.positions("BTC-USDT")
    assert len(recorder.requests) == 2
    # control: an httpx record about any other host still reaches the handlers
    logging.getLogger("httpx").info('HTTP Request: GET https://example.invalid/x "HTTP/1.1 200 OK"')
    text = caplog.text + "\n".join(record.getMessage() for record in caplog.records)
    assert "example.invalid" in text
    for forbidden in ("signature=", "timestamp=", "recvWindow=", DEMO_HOST, API_KEY, API_SECRET):
        assert forbidden not in text, forbidden
    make_client(creds, Recorder())
    assert logging.getLogger("httpx").filters.count(da._LOG_FILTER) == 1  # installed once


def test_log_filter_fails_closed_on_a_record_it_cannot_render():
    record = logging.LogRecord("httpx", logging.INFO, "x.py", 1, "%d", ("not a number",), None)
    assert da._LOG_FILTER.filter(record) is False


# -- read-only surface ------------------------------------------------------------------------------------


def test_balance_returns_the_data_list(creds):
    recorder = Recorder(ok([BALANCE_ITEM]))
    client = make_client(creds, recorder)
    assert client.balance() == [BALANCE_ITEM]
    client = make_client(creds, Recorder(ok({"balance": BALANCE_ITEM})))
    with pytest.raises(DemoSchemaError):
        client.balance()


def test_positions_code_109500_raises_instead_of_an_empty_list(creds):
    recorder = Recorder(
        httpx.Response(200, json={"code": 109500, "msg": "query failed", "data": []})
    )
    client = make_client(creds, recorder)
    with pytest.raises(PositionsQueryFailed) as info:
        client.positions("BTC-USDT")
    assert isinstance(info.value, DemoApiError)
    assert info.value.code == 109500


def test_positions_returns_list_and_passes_symbol_only_when_given(creds):
    position = {"symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "0.0010"}
    recorder = Recorder(ok([position]))
    client = make_client(creds, recorder)
    assert client.positions("BTC-USDT") == [position]
    assert recorder.requests[0].url.params["symbol"] == "BTC-USDT"
    recorder.response = ok([])
    assert client.positions() == []
    assert "symbol" not in recorder.requests[1].url.params
    recorder.response = httpx.Response(200, json={"code": 80001, "msg": "other"})
    with pytest.raises(DemoApiError) as info:
        client.positions()
    assert not isinstance(info.value, PositionsQueryFailed)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("true", True),
        ("false", False),
        (True, True),
        (False, False),
        ("TRUE", True),
        ("False", False),
    ],
)
def test_position_mode_parses_strings_and_booleans(creds, value, expected):
    client = make_client(creds, Recorder(ok({"dualSidePosition": value})))
    assert client.position_mode_is_hedge() is expected
    assert client._http.event_hooks["request"]  # the hook was in place for that request too


@pytest.mark.parametrize(
    "data", [{"dualSidePosition": "maybe"}, {"dualSidePosition": 1}, {}, [], "true"]
)
def test_position_mode_rejects_other_shapes(creds, data):
    client = make_client(creds, Recorder(ok(data)))
    with pytest.raises(DemoSchemaError):
        client.position_mode_is_hedge()


def test_no_order_cancel_or_leverage_methods_exist_yet():
    names = [name.lower() for name in dir(DemoAccountClient)]
    for word in ("order", "cancel", "leverage", "trade", "margin"):
        assert not any(word in name for name in names), word


@pytest.mark.parametrize(
    "path",
    [
        "openApi/swap/v3/user/balance",
        "/openapi/swap/v3/user/balance",
        "/other/openApi/x",
        "/openApi//swap/x",
        "/openApi/swap/x?y=1",
        "/openApi/swap/x#frag",
        f"https://{DEMO_HOST}/openApi/swap/x",
        "/openApi/swap/x:443",
        "/openApi/swap\\x",
        "",
        "/openApi/swap/x\ny",
        "/openApi/swap/\u00e9",
    ],
)
def test_path_validation_raises_before_any_request(creds, path):
    recorder = Recorder()
    client = make_client(creds, recorder)
    with pytest.raises(ValueError):
        client._signed_request("GET", path)
    assert recorder.requests == []


def test_parameter_values_cannot_break_the_signed_string(creds):
    recorder = Recorder()
    client = make_client(creds, recorder)
    with pytest.raises(ValueError):
        client._signed_request(
            "GET", "/openApi/swap/v2/user/positions", {"symbol": "BTC-USDT&recvWindow=0"}
        )
    with pytest.raises(TypeError):
        client._signed_request("GET", "/openApi/swap/v2/user/positions", {"symbol": 1.5})
    assert recorder.requests == []


def test_credentials_type_is_enforced():
    with pytest.raises(TypeError):
        DemoAccountClient(
            {"api_key": API_KEY, "api_secret": API_SECRET},
            transport=httpx.MockTransport(Recorder()),
        )
