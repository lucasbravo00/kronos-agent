"""Authenticated BingX client hard-pinned to the DEMO host ``open-api-vst.bingx.com``.

SPEC rule 2: the authenticated client is fixed to the demo host, refuses to start with any other
host and validates every request. Every host comparison is against the LITERAL host string, never
against the module constant, so a monkeypatched constant fails closed.

The constructor refuses to start if ``DEMO_HOST`` no longer equals the literal or if
``BINGX_HOST`` / ``BINGX_BASE_URL`` point anywhere else, and it accepts no injected transport but
an ``httpx.MockTransport`` (unit tests): the only network transport is the one it builds itself
(TLS verification on, no proxy, environment ignored, ``SSLKEYLOGFILE`` included). Then one
literal-host check, ``_check_demo_url`` (HTTPS, the canonical ASCII host equal to the literal,
default port, no userinfo), runs at three points of every request:

1. a pre-send assertion in ``_signed_request`` on the URL built from the constant;
2. an ``httpx`` request event hook (redirects included);
3. a private transport wrapper, the last gate before the wire.

This is the only module in the package allowed to import ``hmac``. Signing happens inline in
``_signed_request``: the path and every parameter name and value are validated, the query is
signed, then the URL is asserted, and nothing leaves the process before all three checks have
passed. Nothing here is shared with the public market client (SPEC rule 3). The module never
opens files; credentials come from the environment variables ``BINGX_API_KEY`` and
``BINGX_API_SECRET`` only, and ``repr()`` of every object prints ``<redacted>`` for them.
Exceptions carry the path, status and API code, never the URL, query, headers or credentials, and
never chain an ``httpx`` exception (its ``.request`` holds the signed URL and the API-key header).
``httpx`` logs every request URL at INFO: the constructor installs a filter on the ``httpx``
logger that drops every record naming the demo host, so the signed query reaches no log handler.

Step-1 surface is read-only (balance, positions, position mode) and is not called anywhere
outside unit tests with mock transports until the owner opens gate 1 (SPEC section 17).
"""

from __future__ import annotations

import hmac
import logging
import os
import re
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self
from urllib.parse import quote

import certifi
import httpx

DEMO_HOST = "open-api-vst.bingx.com"
API_KEY_ENV = "BINGX_API_KEY"
API_SECRET_ENV = "BINGX_API_SECRET"
HOST_OVERRIDE_ENV_VARS: tuple[str, ...] = ("BINGX_HOST", "BINGX_BASE_URL")
API_KEY_HEADER = "X-BX-APIKEY"
API_PATH_PREFIX = "/openApi/"
FORBIDDEN_PATH_FRAGMENTS: tuple[str, ...] = ("://", "//", "?", "#", ":", "\\")
PLAIN_PARAM_RE = re.compile(r"[A-Za-z0-9._-]+")
MAX_RECV_WINDOW_MS = 60_000
HTTPX_LOGGER_NAME = "httpx"

BALANCE_PATH = "/openApi/swap/v3/user/balance"
POSITIONS_PATH = "/openApi/swap/v2/user/positions"
POSITION_MODE_PATH = "/openApi/swap/v1/positionSide/dual"
POSITIONS_QUERY_FAILED_CODE = 109500
REDACTED = "<redacted>"
USER_AGENT = "kronos-agent/0.1 (demo account)"


class DemoAccountError(Exception):
    """Base class of every error raised by this module."""


class HostNotAllowedError(DemoAccountError):
    """Something tried to make this client talk to anything but the demo host."""


class MissingCredentials(DemoAccountError):
    """``BINGX_API_KEY`` / ``BINGX_API_SECRET`` are absent or empty (values are never echoed)."""


class UnexpectedRedirect(DemoAccountError):
    """The server answered 3xx; redirects are never followed."""


class DemoTransportError(DemoAccountError):
    """The request could not complete or its body could not be decoded; no URL is reported.

    Timeouts, connection failures and a body that does not match its ``Content-Encoding`` all
    land here, named only by the path and the ``httpx`` exception class.
    """


class DemoApiError(DemoAccountError):
    """HTTP >= 400 (``code`` may be None) or a JSON ``code != 0`` for ``path``."""

    def __init__(self, path: str, status: int, code: int | None, msg: str):
        self.path = path
        self.status = status
        self.code = code
        self.msg = msg[:200]
        super().__init__(f"{path}: HTTP {status}, code {code}: {self.msg}")


class PositionsQueryFailed(DemoApiError):
    """``code 109500``: the positions query FAILED; it does not mean "no open position"."""


class DemoSchemaError(DemoAccountError):
    """The response does not have the documented shape."""


class _Secret:
    """Wraps the API secret so it never prints; ``reveal()`` is the only way to read it."""

    __slots__ = ("_value",)

    def __init__(self, value: str):
        if not isinstance(value, str) or not value:
            raise MissingCredentials("the API secret must be a non-empty string")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return REDACTED

    __str__ = __repr__


@dataclass(frozen=True, repr=False)
class Credentials:
    """API key plus wrapped secret; built with :meth:`from_env` from the two environment variables."""

    api_key: str
    api_secret: _Secret

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str) or not self.api_key:
            raise MissingCredentials("the API key must be a non-empty string")
        if not isinstance(self.api_secret, _Secret):
            raise TypeError("api_secret must be wrapped in _Secret")

    @classmethod
    def from_env(cls) -> Credentials:
        """Read ``BINGX_API_KEY`` and ``BINGX_API_SECRET`` from the process environment."""
        missing = [name for name in (API_KEY_ENV, API_SECRET_ENV) if not os.environ.get(name)]
        if missing:
            raise MissingCredentials(
                f"{API_KEY_ENV} and {API_SECRET_ENV} must be set in the environment "
                f"(missing or empty: {', '.join(missing)})"
            )
        return cls(api_key=os.environ[API_KEY_ENV], api_secret=_Secret(os.environ[API_SECRET_ENV]))

    def __repr__(self) -> str:
        return f"Credentials(api_key={REDACTED}, api_secret={REDACTED})"


def _check_demo_url(url: httpx.URL) -> None:
    """Literal host check shared by the pre-send assertion, the hook and the transport.

    The host is compared in its canonical ASCII form (``raw_host``: lowercase, IDNA-encoded, the
    bytes that go into SNI and the ``Host`` header), so no punycode or Unicode spelling can equal
    the literal, and ``url.host`` is never read: it IDNA-decodes and raises ``idna`` errors
    instead of :class:`HostNotAllowedError` for an invalid A-label.
    """
    if url.scheme != "https":
        raise HostNotAllowedError(f"refusing non-https scheme {url.scheme!r}")
    host = url.raw_host.decode("ascii")  # ASCII by construction in httpx
    if host != "open-api-vst.bingx.com":
        raise HostNotAllowedError(
            f"refusing host {host!r}: this client only talks to the demo host"
        )
    if url.port is not None:
        raise HostNotAllowedError(f"refusing non-default port {url.port}")
    if url.userinfo != b"":
        raise HostNotAllowedError("refusing URL with userinfo")


def _request_hook(request: httpx.Request) -> None:
    """httpx request event hook (point 2): runs before every request, redirects included."""
    _check_demo_url(request.url)


class _DemoPinnedTransport(httpx.BaseTransport):
    """Point 3, the last gate before the wire; wraps the network transport or a test's mock."""

    def __init__(self, inner: httpx.BaseTransport):
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        _check_demo_url(request.url)
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


class _DemoHostLogFilter(logging.Filter):
    """Drops every record whose message names the demo host.

    ``httpx`` logs each request at INFO as ``HTTP Request: <method> <url> ...``; for this client
    the URL carries the signed query (parameters, ``timestamp``, ``recvWindow``, ``signature``).
    A filter on a logger runs before any handler, its own or an ancestor's, sees the record.
    Fails closed: a record whose message cannot be rendered is dropped too.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "open-api-vst.bingx.com" not in record.getMessage()
        except Exception:  # noqa: BLE001 - fail closed: drop a record that cannot be rendered
            return False


_LOG_FILTER = _DemoHostLogFilter()


def _install_log_filter() -> None:
    """Attach the filter to the ``httpx`` logger once; every constructor calls it."""
    logger = logging.getLogger(HTTPX_LOGGER_NAME)
    if _LOG_FILTER not in logger.filters:
        logger.addFilter(_LOG_FILTER)


def _check_environment_overrides() -> None:
    """``BINGX_HOST`` / ``BINGX_BASE_URL`` pointing anywhere but the demo host refuse to start."""
    for name in HOST_OVERRIDE_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower().removeprefix("https://").rstrip("/")
        if value != "open-api-vst.bingx.com":
            raise HostNotAllowedError(
                f"{name} is set to something other than the demo host; this client refuses to start"
            )


def _validate_path(path: str) -> str:
    if not isinstance(path, str) or not path.startswith(API_PATH_PREFIX):
        raise ValueError(f"path must start with {API_PATH_PREFIX!r}")
    if not path.isascii() or not path.isprintable():
        raise ValueError("path must be printable ASCII")
    for fragment in FORBIDDEN_PATH_FRAGMENTS:
        if fragment in path:
            raise ValueError(f"path contains forbidden fragment {fragment!r}")
    return path


def _param_text(key: str, value: Any) -> str:
    """Validate one parameter before anything is signed and return its text.

    Names and plain values are restricted to ``[A-Za-z0-9._-]``, so they go on the wire exactly
    as signed and can never make ``httpx`` or the ASCII encoding of the query raise. A value
    containing ``[`` or ``{`` (JSON) may hold any printable ASCII but ``&`` and ``=``; it is the
    only kind :func:`_signed_query` URL-encodes.
    """
    if not isinstance(key, str) or PLAIN_PARAM_RE.fullmatch(key) is None:
        raise ValueError(f"parameter name {key!r} must match [A-Za-z0-9._-]+")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"parameter {key!r} must be a string or an integer")
    text = str(value)
    if "&" in text or "=" in text:
        raise ValueError(f"parameter {key!r} must not contain '&' or '='")
    if "[" in text or "{" in text:
        if not (text.isascii() and text.isprintable()):
            raise ValueError(f"parameter {key!r} (JSON) must be printable ASCII")
    elif PLAIN_PARAM_RE.fullmatch(text) is None:
        raise ValueError(f"parameter {key!r} must match [A-Za-z0-9._-]+ (JSON values aside)")
    return text


def _hmac_sha256_hex(secret: str, message: str) -> str:
    """Lowercase hex HMAC-SHA256 of ``message`` (the unencoded sorted query string)."""
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), "sha256").hexdigest()


def _signed_query(params: Mapping[str, Any], secret: str) -> str:
    """ASCII-sorted ``k=v&...`` signed UNENCODED (parameters checked by :func:`_param_text`).

    Plain values go on the wire as signed; values containing ``[`` or ``{`` (JSON) are the only
    ones URL-encoded on the wire, still signed raw.
    """
    items = sorted((key, _param_text(key, value)) for key, value in params.items())
    to_sign = "&".join(f"{key}={value}" for key, value in items)
    on_wire = "&".join(
        f"{key}={quote(value, safe='') if ('[' in value or '{' in value) else value}"
        for key, value in items
    )
    return f"{on_wire}&signature={_hmac_sha256_hex(secret, to_sign)}"


def _parse_code_msg(response: httpx.Response) -> tuple[int | None, str]:
    try:
        payload = response.json()
    except ValueError:
        return None, response.text[:200]
    if not isinstance(payload, dict):
        return None, "non-object JSON body"
    code = payload.get("code")
    if isinstance(code, bool) or not isinstance(code, int):
        code = None
    return code, str(payload.get("msg", ""))[:200]


def _tls_context() -> ssl.SSLContext:
    """Certificate and hostname verification against the certifi bundle (the one ``httpx`` uses
    for ``verify=True``), built directly: ``ssl.create_default_context`` would write every session
    secret to the file named by ``SSLKEYLOGFILE`` whenever that variable is set."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # CERT_REQUIRED and check_hostname
    context.load_verify_locations(cafile=certifi.where())
    return context


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise DemoSchemaError(f"{POSITION_MODE_PATH}: field {field!r} is not a boolean")


class DemoAccountClient:
    """Signed requests to the demo account. Read-only surface in step 1; no order methods exist yet.

    ``transport`` accepts only an ``httpx.MockTransport`` (unit tests); the only network transport
    is the one built here (TLS verification on, no proxy, no key log, environment ignored). A mock
    handler is arbitrary code, so this guards against mistakes, not against code that already
    holds the credentials. ``clock`` (epoch seconds) is injectable for tests. There is
    deliberately no host, base URL or URL parameter anywhere.
    """

    def __init__(
        self,
        credentials: Credentials,
        *,
        transport: httpx.MockTransport | None = None,
        timeout: float = 10.0,
        recv_window_ms: int = 5000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if DEMO_HOST != "open-api-vst.bingx.com":
            raise HostNotAllowedError("DEMO_HOST was altered; this client refuses to start")
        _check_environment_overrides()
        if not isinstance(credentials, Credentials):
            raise TypeError("credentials must be a Credentials instance")
        if transport is not None and type(transport) is not httpx.MockTransport:
            raise TypeError(
                "transport must be an httpx.MockTransport (unit tests only); "
                "the network transport is built by this client"
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive seconds")
        if isinstance(recv_window_ms, bool) or not isinstance(recv_window_ms, int):
            raise TypeError("recv_window_ms must be an integer")
        if not 0 <= recv_window_ms <= MAX_RECV_WINDOW_MS:
            raise ValueError(f"recv_window_ms must be in [0, {MAX_RECV_WINDOW_MS}]")
        self._credentials = credentials
        self._recv_window_ms = recv_window_ms
        self._clock = clock
        inner = (
            transport
            if transport is not None
            else httpx.HTTPTransport(trust_env=False, verify=_tls_context(), retries=0)
        )
        self._transport = _DemoPinnedTransport(inner)
        _install_log_filter()
        self._http = httpx.Client(
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(timeout),
            event_hooks={"request": [_request_hook]},
            headers={"User-Agent": USER_AGENT},
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __repr__(self) -> str:
        return (
            f"DemoAccountClient(host={DEMO_HOST!r}, api_key={REDACTED}, api_secret={REDACTED}, "
            f"recv_window_ms={self._recv_window_ms})"
        )

    # -- signed transport ------------------------------------------------------------------------

    def _signed_request(
        self, method: str, path: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Validate path and parameters, sign the query, assert the literal host, send, check."""
        _validate_path(path)
        full_params = dict(params or {})
        full_params["timestamp"] = int(self._clock() * 1000)
        full_params["recvWindow"] = self._recv_window_ms
        query = _signed_query(full_params, self._credentials.api_secret.reveal())
        url = httpx.URL(scheme="https", host=DEMO_HOST, path=path, query=query.encode("ascii"))
        _check_demo_url(url)
        request = self._http.build_request(
            method, url, headers={API_KEY_HEADER: self._credentials.api_key}
        )
        response = self._send(request, path)
        status = response.status_code
        if 300 <= status < 400:
            raise UnexpectedRedirect(f"{path}: HTTP {status} redirect refused (never followed)")
        if status >= 400:
            code, msg = _parse_code_msg(response)
            raise DemoApiError(path, status, code, msg)
        try:
            payload = response.json()
        except ValueError:
            raise DemoSchemaError(f"{path}: response body is not JSON") from None
        if not isinstance(payload, dict):
            raise DemoSchemaError(f"{path}: expected a JSON object")
        code = payload.get("code")
        if isinstance(code, bool) or not isinstance(code, int):
            raise DemoSchemaError(f"{path}: missing or non-integer 'code'")
        if code != 0:
            raise DemoApiError(path, status, code, str(payload.get("msg", "")))
        return payload

    def _send(self, request: httpx.Request, path: str) -> httpx.Response:
        """Send; every ``httpx`` error (transport, decoding...) becomes ``DemoTransportError``.

        The module error is raised after the ``except`` block, so neither ``__cause__`` nor
        ``__context__`` keeps the ``httpx`` exception, whose ``.request`` carries the signed URL
        and the API-key header. Only the exception class name survives. ``httpx`` also builds the
        (never followed) redirect request while sending: a 3xx ``Location`` it cannot parse
        raises ``UnicodeError`` (an invalid A-label) or ``httpx.InvalidURL`` (no host), outside
        the ``HTTPError`` tree, and is refused as :class:`UnexpectedRedirect`.
        """
        try:
            return self._http.send(request)
        except httpx.HTTPError as exc:
            failure = type(exc).__name__
        except (UnicodeError, httpx.InvalidURL):
            failure = None
        if failure is None:
            raise UnexpectedRedirect(f"{path}: redirect with an unparseable Location refused")
        raise DemoTransportError(f"{path}: {failure}")

    @staticmethod
    def _data_list(payload: Mapping[str, Any], path: str) -> list[dict[str, Any]]:
        data = payload.get("data")
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise DemoSchemaError(f"{path}: expected 'data' to be a list of objects")
        return data

    # -- read-only surface (gate 1 applies before any real call) --------------------------------

    def balance(self) -> list[dict[str, Any]]:
        """``GET /openApi/swap/v3/user/balance`` -> ``data`` list (asset, balance, equity... as strings)."""
        payload = self._signed_request("GET", BALANCE_PATH)
        return self._data_list(payload, BALANCE_PATH)

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """``GET /openApi/swap/v2/user/positions``; ``code 109500`` raises, it never means "flat"."""
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = symbol
        try:
            payload = self._signed_request("GET", POSITIONS_PATH, params)
        except DemoApiError as exc:
            if exc.code == POSITIONS_QUERY_FAILED_CODE:
                raise PositionsQueryFailed(exc.path, exc.status, exc.code, exc.msg) from exc
            raise
        return self._data_list(payload, POSITIONS_PATH)

    def position_mode_is_hedge(self) -> bool:
        """``GET /openApi/swap/v1/positionSide/dual`` -> ``data.dualSidePosition`` as a bool."""
        payload = self._signed_request("GET", POSITION_MODE_PATH)
        data = payload.get("data")
        if not isinstance(data, dict) or "dualSidePosition" not in data:
            raise DemoSchemaError(f"{POSITION_MODE_PATH}: expected 'data.dualSidePosition'")
        return _parse_bool(data["dualSidePosition"], "dualSidePosition")


__all__ = [
    "API_KEY_HEADER",
    "DEMO_HOST",
    "Credentials",
    "DemoAccountClient",
    "DemoAccountError",
    "DemoApiError",
    "DemoSchemaError",
    "DemoTransportError",
    "HostNotAllowedError",
    "MissingCredentials",
    "PositionsQueryFailed",
    "UnexpectedRedirect",
]
