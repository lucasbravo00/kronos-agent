# Decisions log

Dated record of design decisions and of the owner's answers to the single question batch of
step 1 (SPEC sections 2 and 18). Until the owner answers an item, the SPEC-compliant default
applies and is marked "pending".

## Step-1 question batch (delivered 2026-10-29; defaults applied 2026-10-30)

| # | Topic | Default applied | Owner's answer |
| --- | --- | --- | --- |
| 1 | License | MIT proposed; no `LICENSE` file until the owner's OK (SPEC section 19) | pending |
| 2 | Gate 4: Kronos-base download for the timing measurement | not downloaded; Kronos-small + Kronos-Tokenizer-base only | pending |
| 3 | Commits | one initial skeleton commit on `main`, no remote; no further commits until told | pending |
| 4 | Vendoring scope | code-only subset (`model/`, `finetune/`, `finetune_csv/` code and configs, `tests/`, `LICENSE`, `README.md`, `requirements.txt`, `.gitignore`); the manifest lists all 92 upstream blobs | pending |
| 5 | Volume-profile source | 1 h candles, each candle's volume spread uniformly over the rows its range covers (the owner's VPVR method); a 1 m-based profile only as a discovery-time variant | pending |
| 6 | Event definition | (a) strictly decreasing volume over n candles; (b) divergence = the candle setting the new extreme has less volume than the candle that set the previous extreme inside the swing window; (c) Wilder ATR; (d) direction = opposite sign of `close[t]/close[t-m] - 1`, zero return = no event; (e) 75 equal-height rows between the window's low and high | pending |
| 7 | Target levels | next level among the ACTIVE types only (`target_level_types: active`) | pending |
| 8 | Partition (UTC, all sources) | training 2020-01-01 -> 2024-06-09T08:00Z; embargo 520 h; validation 2024-07-01T00:00Z -> 2026-07-10T08:00Z; embargo 520 h; sealed 2026-08-01T00:00Z -> open. N1/N2 sub-partition: train 2024-07-01 -> 2025-12-31, embargo 520 h, validation 2026-01-22T16:00Z -> 2026-07-10T08:00Z. The BingX-real overlap 2024-04-27 -> 2024-06-09 serves only the Binance-vs-BingX mismatch study | pending |
| 9 | `amount` channel for Kronos | (A) proxy `volume * mean(open, high, low, close)`, identical in training CSVs; proxy-vs-`quote_volume` error reported in step 2 | pending |
| 10 | Fine-tuning (step 6) | scripts verbatim first, plus a separate test with a wrapper doing context-only normalisation; best-validation checkpoint instead of early stopping; MPS attempt through a wrapper with CPU fallback; pre-cut CSVs following the partition; the YAML is cited as the source of `predictor_learning_rate = 1e-6` | pending |
| 11 | Gate-1 logistics | first read-only authenticated call from the owner's Mac, key IP-restricted to it (server IP added at deployment); the owner loads `.env` in their own shell; key permissions read + futures trading; the owner confirms the real futures account is at zero | pending |
| 12 | Data and weights folders | in-repo `data/` and `weights/` (gitignored), relocatable through `KRONOS_AGENT_DATA_DIR` / `KRONOS_AGENT_WEIGHTS_DIR` | pending |
| 13 | Time stop vs horizon | `t = 8` and `N = 4` as in the SPEC table | pending |

Notices given with the batch: `docs/SPEC.md` stays in Spanish verbatim; the "no VST balance ->
no trading" kill condition is coded after gate 1 reveals the demo asset label; credentials are
`BINGX_API_KEY` / `BINGX_API_SECRET`; Python 3.12, 2 threads for comparable inference, branch
`main`; `reports/` and `reviews/` are tracked.

## Design decisions

- D-001 (2026-10-30) Python 3.12 (newest interpreter compatible with the upstream pins; pandas
  2.2.2 has no cp313 wheels) with the upstream `requirements.txt` pins plus `pyyaml`, `httpx`,
  `pytest`; `torch==2.14.1` (patch release with a month of field use; 2.15.0 was released on
  2026-10-28); PyPI wheel on macOS arm64, PyTorch CPU index on Linux only; `uv.lock` restricted
  to darwin-arm64 and linux-x86_64.
- D-002 (2026-10-30) Kronos vendored by copy (not a submodule): reproducible without network,
  immutability enforced by a strict GNU sha256 manifest, provenance tied to the commit through
  the git blob sha1 of every file.
- D-003 (2026-10-30) Weights registry (`manifests/weights.json`) pre-registered from the Hugging
  Face API before any download; downloads verified by size and sha256; models loaded with
  `strict=True`, `.eval()`, on CPU; `state_dict` element counts checked (they include persistent
  buffers; `.parameters()` would not).
- D-004 (2026-10-30) Both exchange clients are host-pinned (the public one to
  `open-api.bingx.com`, the demo one to `open-api-vst.bingx.com`) with `httpx`, redirects refused,
  environment proxies ignored; signing code exists only in `demo_account.py`.
- D-005 (2026-10-30) No `.env.example`: the `Read(.env.*)` deny rule would block it; variable
  names are documented instead. The `Bash(...)` deny entries are tripwires, not a security
  boundary; `export`/`set` are not denied because scripts use them legitimately.
- D-006 (2026-10-30) The fan of trajectories is built with `predict_batch([df]*K, sample_count=1)`
  because `predict(sample_count=K)` returns the mean path at the pinned commit.
- D-007 (2026-10-30) The upstream greedy regression fixtures are compared report-only (their
  generating platform is unknown); weight integrity = sha256 + strict load + element counts +
  same-seed determinism.
- D-008 (2026-10-30) Owner's instruction: `research/strategy/` is a separate session's working
  folder where the owner defines the trading criterion. This tree never reads, modifies or deletes
  it; it is gitignored and excluded from the hygiene scan and from ruff. While that work is open,
  SPEC section 4 is under review: no event definition is registered and
  `params.yaml[criterion]` is not frozen; steps 1-3 and the section-9 retraining go ahead.
- D-009 (2026-10-30) `[tool.ruff] extend-exclude = ["third_party", "research"]`: a repo-root
  `ruff check --fix` or `ruff format` must never rewrite the vendored tree (hard rule 4; measured
  before the fix: 94 lint findings and 17 format diffs, all under `third_party/kronos`).
- D-010 (2026-10-30) Deny tripwires extended with `Bash(env *)`, `Bash(uv run --env-file*)` and
  `Bash(uv run * --env-file*)` so that the forms CLAUDE.md names are all tripped; still tripwires,
  not a security boundary (D-005).

Decisions D-011 to D-021 come from the review of the step-1 tree (finding ids in brackets) and
were applied before the initial commit.

- D-011 (2026-10-31) Demo client logging and errors [C1, A5, C2]. httpx logs every request URL at
  INFO, and a signed URL carries `signature` and `timestamp`: the demo client attaches, once, a
  filter to the `httpx` logger that drops every record whose message names
  `open-api-vst.bingx.com` (a record that cannot be rendered is dropped too). A logger filter
  rather than `setLevel(WARNING)`: it survives a `logging.config.dictConfig` that names `httpx`
  and leaves the public client's request lines alone. Every `httpx.HTTPError` raised while
  sending (transport failure, timeout, undecodable body) becomes `DemoTransportError` naming only
  the path and the httpx class, raised outside the `except` block so that no `__cause__` or
  `__context__` keeps the httpx exception, whose `.request` holds the signed URL and the key
  header. An undecodable body is therefore a transport error; a step-8 retry policy must decide
  deliberately whether to retry it. The frame locals of `_signed_request` stay reachable from a
  traceback: never enable local-variable capture (pytest `--showlocals`, crash reporters that
  record locals) around the demo client.
- D-012 (2026-10-31) Demo client inputs [A2, C4]. `DemoAccountClient(transport=...)` accepts only an
  exact `httpx.MockTransport` (unit tests); the network transport is always the client's own
  `httpx.HTTPTransport(trust_env=False, verify=<context>, retries=0)`, whose SSL context is built
  directly (certificate and hostname verification against certifi's bundle, the one httpx uses)
  rather than by `ssl.create_default_context`, which would write every TLS session secret to the
  file named by `SSLKEYLOGFILE`. This is mistake-proofing, not a sandbox: a mock handler is
  arbitrary in-process code. Parameter names and plain values must
  match `[A-Za-z0-9._-]+`; a value containing `[` or `{` (JSON) may hold any printable ASCII except
  `&` and `=`, is signed raw and percent-encoded on the wire. Whether BingX decodes percent-encoded
  JSON before verifying the signature stays unverified until step 8 sends such a request (after
  gate 2).
- D-013 (2026-10-31) Host comparisons [A6]. The demo client, the public client and the weights
  redirect check compare `url.raw_host` (lowercase, IDNA-encoded ASCII: the bytes that go into SNI
  and the `Host` header) with their literal host; `url.host` is never read because it IDNA-decodes
  and raises `idna` errors for an invalid A-label. httpx itself builds the never-followed redirect
  request while sending: a 3xx `Location` it cannot parse raises `UnicodeError` (an invalid
  A-label) or `httpx.InvalidURL` (no host), outside the `httpx.HTTPError` tree. Both exchange
  clients refuse it as `UnexpectedRedirect` (the demo one without chaining the exception), and the
  weights download maps it to `WeightsError`.
- D-014 (2026-10-31) Vendored import [E6, A4]. `import_kronos()` restores `sys.path` after
  importing upstream's `model`, also when the import fails; `model`, `model.kronos` and
  `model.module` stay in `sys.modules`. A spawn- or forkserver-started worker that unpickles
  upstream objects must call `import_kronos()` first (for example as a `multiprocessing.Pool`
  initializer); fork-started workers and the step-6 subprocess runs of `finetune_csv` are
  unaffected. The import compiles the verified sources and reads or writes no bytecode: a `.pyc`
  runs in place of the source it was compiled from, so the strict manifest reports any bytecode
  cache under `third_party/kronos` as an unlisted file, `sys.dont_write_bytecode` is set and
  `sys.pycache_prefix` (which would point the loader at caches outside the tree) is switched off
  for the import. The owner approved the
  one-time removal of the interpreter cache `third_party/kronos/model/__pycache__/` (3 files, not
  upstream content) and removed it on 2026-10-31. Anything that imports `third_party/kronos`
  outside `import_kronos()` (the step-6 `finetune_csv` subprocesses, upstream's own tests) must run
  with `python -B`, or `kronos-agent verify` fails.
- D-015 (2026-10-31) `vendor-kronos` extraction [A3, E1] refuses, before writing each tarball
  member, absolute names (`//`-prefixed ones included), `..` traversal, empty names, links,
  devices, FIFOs and unknown member types, and checks that the resolved target stays inside the
  temporary extraction directory. GitHub's pax global header is consumed by `tarfile` and is not a
  member.
- D-016 (2026-10-31) Personal-data checks [D-1]. `envinfo.personal_markers` (used by `env-report`,
  `bench` and `tests/test_repo_hygiene.py`) skips a login or host name that is exactly the
  project's name or one of its words (`kronos`, `agent`, `kronos-agent`, `kronos_agent`, any
  case), so a `kronos` service account or a `kronos-agent` server does not make them fail. A
  shorter piece such as `ron` can be a person's name and is still checked; the home directory is
  never skipped, and every other login or host name is matched as a substring. Known and left as is: run as `root`
  on macOS, `env-report` and `bench` fail (the kernel version string contains `root:xnu`); nothing
  in the project runs as root.
- D-017 (2026-10-31) Environment report [C5, E9, E7]. `env-report --network` measures the clock
  through `PublicMarketClient` (host pin, path allow-list, status before body, schema guard) with
  pacing off and no retries; a transport error, a 5xx or a rate-limit answer is recorded as
  `clock.error` after exactly one GET. A weights repository is "absent" only when none of its
  registered files is present; a partially present one (an interrupted download) is a problem:
  `verify` exits 1, `env-report` records `problems`, and `fetch-weights --model <name>` completes
  it.
- D-018 (2026-10-31) Command line [D-8, E8, E4]. The registry in `kronos_agent/cli/__init__.py`
  imports every command module strictly: a command is registered in the same change that adds its
  module, and a broken module fails `--help` instead of silently dropping the command. The report
  label helpers (`default_label`, `validate_label`) are defined once, in `paths.py`.
  `fetch-weights` exits 0 when every file is downloaded and verified, 1 on a download failure
  (`ERROR: ...`; an interrupted transfer leaves no `.part` file) or a verification problem
  (`PROBLEM: ...`), and 2 on a gate-4 refusal (`REFUSED: ...`).
- D-019 (2026-10-31) Parameter validation [E5, E2]. `schema_version` must be the integer 1 (YAML
  `true` or `1.0` are rejected); `kronos.seed_salt` must follow the seed grammar of
  `kronos_agent.seeds` (ASCII, no `/`, no newline); `kronos.trajectories <= 100` and
  `kronos.horizon_n <= 48` are sanity bounds at twice the SPEC's largest values, not a memory
  guarantee for the server (the step-8 linux-x86_64 peak-RSS measurement and the 4 GB swap remain
  the server guard). `params.yaml` itself is unchanged (version `4dcb783980a9`).
- D-020 (2026-10-31) Test suite [D-6, D-2, D-10, D-5]. The socket guard wraps each test's whole
  protocol (setup, call, teardown), so session- and module-scoped fixtures such as `loaded_small`
  are guarded too; the `weights` and `network` decisions read real markers
  (`get_closest_marker`), never `item.keywords`, which also holds parametrize ids. The guard's
  own probes target the loopback discard port and run the demo-host lookup with the real resolver
  stubbed, so even a broken guard sends nothing off the machine.
  `tests/test_client_separation.py` resolves every `os.environ` key read by `demo_account.py` to
  string literals (failing closed) and checks that the credential names and the BingX host
  literals appear only in the exchange package. No test asserts a wall-clock budget: timing is the
  job of `kronos-agent bench`, and the 20 x 8 test fan only records its wall time as a junit
  property. The full suite peaks at 1254 MiB of RAM summed over the test process and its
  benchmark worker (largest single process 1025 MiB; the weights tests alone 1103 MiB; sampled
  every 0.1 s on darwin-arm64 on 2026-10-31); on the 1.9 GB server run
  `uv run pytest -m "not weights"` or create the 4 GB swap first.
- D-021 (2026-10-31) Live public checks [C3, C6]. Deviation from the approved step-1 plan: the
  Binance Vision `.CHECKSUM` HEAD probe is not in `tests/test_public_market_live.py`. No step-1
  module talks to data.binance.vision; the check moves to step 2, where every downloaded zip is
  verified against its `.CHECKSUM` and an unreachable data.binance.vision is reported to the owner
  (SPEC section 5). The live module gains a two-page (1100 closed hours) backfill test of the page
  seam (`endTime = oldest_open - 1`) that the step-2 BingX backfill relies on, and records the
  first raw page so that an ignored `startTime` cannot pass for an inclusive one. Observed on
  2026-10-31 (two runs of 8 public GETs, all 8 network tests passed each time): `startTime` is
  honoured and inclusive at an exact candle boundary (the first page starts at `startTime`: 49
  candles, 48 closed plus the open one) and the seam returns the preceding candle as the newest
  of the next page (1100 closed hours in 2 pages, no duplicate, no gap); server clock offset
  -174 and -172 ms.
