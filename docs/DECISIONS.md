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
| 5 | Volume-profile source | 1 h candles, each candle's volume spread uniformly over the rows its range covers (the owner's VPVR method); a 1 m-based profile only as a discovery-time variant (the allocation wording is made explicit in D-030) | moot: the volume profile left with D-042 |
| 6 | Event definition | (a) strictly decreasing volume over n candles; (b) divergence = the candle setting the new extreme has less volume than the candle that set the previous extreme inside the swing window; (c) Wilder ATR; (d) direction = opposite sign of `close[t]/close[t-m] - 1`, zero return = no event; (e) 75 equal-height rows between the window's low and high | without effect (D-042) |
| 7 | Target levels | next level among the ACTIVE types only (`target_level_types: active`) | without effect (D-042) |
| 8 | Partition (UTC, all sources) | training 2020-01-01 -> 2024-06-09T08:00Z; embargo 520 h; validation 2024-07-01T00:00Z -> 2026-07-10T08:00Z; embargo 520 h; sealed 2026-08-01T00:00Z -> open. N1/N2 sub-partition: train 2024-07-01 -> 2025-12-31, embargo 520 h, validation 2026-01-22T16:00Z -> 2026-07-10T08:00Z. The BingX-real overlap 2024-04-27 -> 2024-06-09 serves only the Binance-vs-BingX mismatch study | confirmed 2026-11-02 (D-051) |
| 9 | `amount` channel for Kronos | (A) proxy `volume * mean(open, high, low, close)`, identical in training CSVs; proxy-vs-`quote_volume` error reported in step 2 | confirmed 2026-11-02 (D-051) |
| 10 | Fine-tuning (step 6) | scripts verbatim first, plus a separate test with a wrapper doing context-only normalisation; best-validation checkpoint instead of early stopping; MPS attempt through a wrapper with CPU fallback; pre-cut CSVs following the partition; the YAML is cited as the source of `predictor_learning_rate = 1e-6` | decided 2026-11-02 through D-042 (3) and D-048 |
| 11 | Gate-1 logistics | first read-only authenticated call from the owner's Mac, key IP-restricted to it (server IP added at deployment); the owner loads `.env` in their own shell; key permissions read + futures trading; the owner confirms the real futures account is at zero | pending |
| 12 | Data and weights folders | in-repo `data/` and `weights/` (gitignored), relocatable through `KRONOS_AGENT_DATA_DIR` / `KRONOS_AGENT_WEIGHTS_DIR` | confirmed 2026-11-02 (D-051) |
| 13 | Time stop vs horizon | `t = 8` and `N = 4` as in the SPEC table | without effect (D-042) |

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
  never skipped, and every other login or host name is matched as a substring. Known and left as
  is: run as `root` on macOS, `env-report` and `bench` fail (the kernel version string contains
  `root:xnu`); nothing in the project runs as root.
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

Decisions D-022 to D-039 are the design of step 2 (data layer, volume profile, stored forecasts,
studies and commands); D-040 to D-050 record its gate-3 approval, its live checks, the owner's
redirect (D-042, which withdraws the volume profile: D-030 and D-031), the fixes of its review
(D-043 to D-047, with the owner's decision on revised BingX candles in D-045, checked in D-050),
the owner's answers on the design of the retraining (D-048) and the archive of the BingX 1m and
5m candles (D-049). `docs/DATA.md` is their binding description for the data layer; D-048's
design becomes `docs/RETRAINING.md`. None of them registers an event definition or freezes
`params.yaml[criterion]` (D-008), and none consumes test budget.

- D-022 (2026-11-01) Sources and observed facts (SPEC section 5). Training and discovery: the
  Binance USD(S)-M perpetual BTCUSDT kline archives of data.binance.vision (public, no keys);
  evaluation and live: the BingX real market through the step-1 public client. Observed on
  2026-11-01: monthly 1m archives exist from 2020-01 and the 2019 monthly archives return 404 (so
  every plan probes the 2019 daily archives); the month just ended has its daily archives before
  its monthly one; every zip holds one 12-column CSV, with a header line in recent files and none
  in 2020 files; price strings change format across periods (`7225` and `7171.10` in one 2020
  file); `.CHECKSUM` is `<sha256 hex>  <zip name>`. From step 1, the BingX real market serves 1h
  from 2024-04-27T10:00Z and keeps 1m (from about 2025-07-31) and 5m (from about 2026-09-14) on
  rolling retention (about 450 and 48 days); 15m depth is unknown. Hence the BingX chunks are an
  archive that is never deleted and `data bingx --interval 5m` runs at least monthly. The Binance
  Vision client mirrors the public client (GET only, the host pinned on the default port and
  compared on `url.raw_host`, a path allow-list, no redirects, environment ignored, no
  credentials, paced at >= 0.25 s, retries after 1, 5 and 15 s); an unreachable host is a failure
  reported to the owner. The download is pre-approved (gate 4) under a 4.5 GB budget that counts
  every byte under `raw/binance` (estimate: 80 bytes per candle, deliberately high; expected about
  0.15 GB); a run plans every interval it syncs and checks that budget once, for all of them,
  before any download (amended by D-046). The facts are re-checked by the opt-in network modules
  `tests/test_binance_vision_live.py` and `tests/test_bingx_history_live.py`.
- D-023 (2026-11-01) Store format and float conversion. One CSV per source and interval holding
  the exchange's decimal strings verbatim (header line, ascending unique `open_ms`, `\n`, no
  quoting, values matching `^-?[0-9]+(\.[0-9]+)?$`), and a meta written after it as the commit
  marker (gaps, missing candles, OHLC-relation violations counted with `Decimal` and never fixed,
  CSV size and sha256, data provenance, code). Strings, not floats, so that a window's hash can be
  recomputed from the store and the same rows always give the same bytes; CSV and JSON rather
  than Parquet because nothing may be installed (gate 4). Model inputs are `float()` of the
  strings; bulk numeric reads use `read_csv(float_precision="round_trip")`, the same correctly
  rounded conversion; pandas' default parser and `to_numeric` are forbidden for candle values
  (`"465655238.20240235"` comes out one ulp off). Two read paths: `read_study_frame` (studies and
  window sources, behind the sealed guard) and `read_store` (integrity work only). A ranged read
  binary-searches the byte offsets of the sorted rows and parses only the rows of the range, so
  sealed rows are never parsed by a study.
- D-024 (2026-11-01) Durable writes and locks. Every file goes through a temporary file in its
  own directory (`mkstemp`), `flush`, `fsync`, mode 0o644, `os.replace` and an `fsync` of the
  directory, so a crash leaves the old file or the new one; stores publish the CSV and then the
  meta, BingX chunks the CSV and then the sidecar. Files that must never change (stored fans,
  saved windows, recorded study configs) are published with `os.link`, which fails when the name
  exists. The Binance sync and builds and the BingX backfill and merge hold a non-blocking
  `fcntl.flock` on `raw/<source>/.lock`; a second concurrent run is refused (exit 2). The lock
  file is never deleted: deleting it would let two runs lock two different files. Amended by D-043
  [foundation-4, foundation-5]: reads take the same lock shared (`store_read_lock`): a read during
  a run is refused (`StoreBusyError`) and a run during a read is refused; a read opens the CSV
  once, so the bytes hashed are the bytes parsed. `exclusive_lock` yields a `HeldLock`; `job_lock`
  (`kronos_agent.data.store`) takes a job lock and first removes the `mkstemp` temporaries killed
  runs left in the directories only that lock's holders write (`candles/binance`,
  `candles/binance_official`, `raw/binance`; `candles/bingx`, `raw/bingx`), which is safe because
  no other writer can be live; `data status` lists leftovers.
- D-025 (2026-11-01) Source-tree provenance. The repository stays at one commit (CLAUDE.md, "Git"),
  so a commit hash does not identify the code that produced a result. Every store meta, report and
  stored fan records `{commit, dirty, source_tree_sha256}`; the tree hash is one sha256 fed, for
  every file of `src/kronos_agent/**/*.py`, `params.yaml`, `pyproject.toml`, `uv.lock` and
  `manifests/*` in sorted order of its relative POSIX path, with `<path> NUL <sha256 hex> LF`
  (`__pycache__` and `.DS_Store` excluded), reproducible without the package. Git is only asked
  read-only questions with `GIT_OPTIONAL_LOCKS=0`; `dirty` counts untracked files. A narrower hash
  of the forecasting code (`kronos/forecast.py`, `kronos/features.py`, `kronos/decision.py`,
  `kronos/weights.py`, `kronos/vendored.py`, `data/windows.py`, `seeds.py`, `runtime.py` and
  `manifests/kronos_source.sha256`) enters the forecast `config_id` (D-032). It hashes whole files
  on purpose: an edit anywhere in them costs recomputation, never a wrong reuse. `data/store.py`
  stays out of it, covered by each fan's `context_sha256` (D-044). The models' construction
  (`kronos/weights.py`, `kronos/vendored.py`) joined the hash with D-047 [foundation-1]: covered
  only by the configuration and the refusal of D-044, an edit of `load_models` that no config key
  sees (a quantisation, another checkpoint under the registry's weight hashes) reused stored fans.
- D-026 (2026-11-01) Exact resampling and the official check. `binance/1m` is resampled to 5m, 15m
  and 1h from the strings, archive by archive (archive periods are whole UTC days or months, so no
  bucket crosses one): first open, last close, the string of the first largest high and first
  smallest low compared as decimals, exact `Decimal` sums of the four volumes (`prec = 60`,
  `Inexact` and `Rounded` trapped: a sum that would round raises), integer sum of `count`; a
  bucket is kept only when all its minutes exist, otherwise it is a gap listed with its minute
  count. SPEC section 5 asks to check "a sample" against Binance's own 1h archives; the check
  compares every hour present in both 1h stores, all months, because it is cheap and stronger:
  open, high, low, close, volume, count and taker-buy volume must be equal as decimals; the two
  quote volumes may differ by `sum(0.5 x 10^-d_i) + 0.5 x 10^-d_official`, the rounding of the 60
  summands and of the official value. Hours at or after the sealed start contribute counts and
  open times only; any OHLC, volume or count mismatch is reported to the owner.
- D-027 (2026-11-01) BingX backfill rules. Chunks of 1000 candles on an epoch-aligned grid, one
  `klines` page each: `startTime` the chunk start (inclusive, observed), `endTime = min(chunk end -
  1 ms, server time)` moved one millisecond back when it lands on a candle boundary (only the
  `boundary - 1` seam form was verified in step 1) and never in the future. A candle is kept only
  when `open + step + 90 s <= server time` (the server time read once per run; SPEC section 5:
  closed candles, at least 90 s after the close); a chunk is complete when its last slot is
  settled. Each chunk is fetched twice and the settled candles must be byte-identical, up to 3
  pairs, then the run fails naming the chunk. A chunk is archived as CSV plus a JSON sidecar
  written after it (the commit marker); empty chunks are never archived; the whole archive is
  verified before any request; complete chunks are never fetched again; incomplete ones are
  refetched every run; archived candles are immutable (amended by D-045 and D-047): a candle the
  refetch returns with other strings is a revision, which keeps its archived, first-seen row and
  is recorded once in `raw/bingx/revision-<interval>.json` while the run goes on (the owner's
  decision of 2026-11-02, D-045); candles the refetch no longer returns are kept and listed in the
  sidecar's `archive_only` (sidecar schema 2), new settled candles are added, an empty refetch
  keeps the archive; a refreshed chunk completes only when the 1m or 5m retention explains every
  candle kept that way (older than the observed retention and than the first candle the refetch
  returned; never for 1h or 15m), otherwise it stays incomplete and the next run fetches it again;
  a server time earlier than an archived chunk's stops the run. The
  walk goes back from the newest settled candle and stops after 3 consecutive empty chunks older
  than the oldest archived one; three empty chunks inside the history would end it early (3000
  candles), an accepted limit. The oldest archived candle is the history start only if nothing
  older was served when it was archived: the run that stops fetched it, or it opens after its
  chunk's start, or `raw/bingx/history-<interval>.json` already names it; otherwise the empty
  chunks below may have rolled off since an earlier, possibly bounded, run archived it, and the
  start stays open (amended by D-043 [bingx-2]). An established
  start is recorded once in `raw/bingx/history-<interval>.json` (stop reason, server time of the
  run that established it); a run that its `--max-chunks` bound stops (stop reason `max_chunks`)
  never writes or changes it, while a bounded run whose walk ends before the bound establishes the
  start like any other (D-047). `data status` warns when the newest archived 1m or 5m candle is
  older than a third of the observed retention, and, for every interval, lists the recorded
  revisions (D-045) and warns about a chunk kept incomplete for candles the exchange no longer
  returns and about another incomplete chunk older than the newest archived chunk (a backfill
  stopped before refreshing it) (D-047).
- D-028 (2026-11-01; amended 2026-11-02 by D-043) Partition additions (question 8 defaults
  unchanged). Horizon purge: a sample at `D` whose targets reach `D + (h - 1) h` is used only if
  that time is before the next sample-holding segment's start minus `max_context` hours; longer
  samples are purged, never shortened. It is the only cut-off for samples that read targets: the
  last validation decisions keep targets in the second embargo, as SPEC section 6's embargo =
  context + N intends (every decision up to 2026-07-10T07:00Z for `h <= 9`). `check_embargo` refuses
  `max_context + horizon > 520` (Kronos-mini, `N >= 9`): the embargo bounds the context plus the
  trading horizon N (SPEC section 6). Evaluation horizons longer than N (SPEC section 8, up to 24
  candles) will be handled by the horizon purge with an explicit evaluation length designed with the
  retraining, not by recomputing the partition. Kronos cutoff: forecasts with the pretrained weights
  refuse `D < 2024-07-01T00:00Z` (tests opt in). The sealed segment (gate 6) is guarded at the
  data-access layer by candle open time: `read_study_frame` refuses a range that can hold a candle
  opening at or after 2026-08-01T00:00Z unless a `SealedAccess` is given, and every command that
  takes a decision hour refuses a sealed one. A `SealedAccess` is issued only by
  `sealed_access(order_id)` after the owner's order is verified in the experiment ledger of step 5;
  until that ledger exists the verification always raises (a direct construction raises too), so
  nothing in step 2 can open the sealed segment. The live window refuses unless `D + 90 s <= server
  time < D + 15 min` (SPEC section 11), so the live path can never fetch history.
- D-029 (2026-11-01) Decision windows. Kronos timestamps are labelled by candle open time: the
  context ends at `D - 1h`, the targets start at `D` (the predictor derives its time embeddings
  from these labels; a close-time labelling would be a registered variant). One builder makes
  every window from any candle sequence (exactly the candles opening in `[D - 512h, D)`, a missing
  one refused, everything else ignored), so the store path and the live path give the same
  `sha256`, `inputs_sha256` and floats bit for bit (tested). `inputs_sha256` also covers the
  `amount` policy (and the quote-volume strings under `binance_quote_volume`), so a policy change
  changes it while the candle hash stays. The live path fetches `[D - 512h, D - 1 ms]` twice,
  up to 3 pairs, each at a freshly read server time and only while it is before `D + 15 min`, then
  refuses (SPEC section 11: no trading when the two downloads differ). A saved window holds the
  window's inputs only (the candles, their sha256 and, for a Binance window, the quote volumes)
  under `windows/<source>/context-<max_context>/<D>.json`; the `amount` policy, `inputs_sha256`
  and the fetch pairs are recorded by each fan (amended by D-046). It is published once; an
  identical file is kept, a different one refused.
- D-030 (2026-11-01) Withdrawn by D-042: Volume allocation, a clarification of question 5. The
  question-5 default ("each candle's volume spread uniformly over the rows its range covers") has
  two readings, and the indicator's code is not available. The reference (`overlap`) spreads the
  volume uniformly over the candle's price range, so a row receives the fraction of the range it
  overlaps; the variant (`equal_per_row`) gives every row the range touches an equal share. Both are
  computed, study 3 reports how often their POC rows differ, and the platform comparison decides;
  the owner's answer to question 5 is still pending. The owner's live chart may include the forming
  bar: the reference excludes it (no lookahead) and `profile show` prints, next to the reference, 99
  closed candles plus the candle opening at `D` after `M` minutes (default 3) from the source's 1m
  store, the "last 100 bars" of a chart. That tool reads through the study path, so an hour at or
  after the sealed start is refused: a screenshot to compare must show an earlier hour (for example
  in the chart's bar replay); comparing a current hour would read the sealed segment and is left to
  the owner. Other fixed choices: row bounds `linspace(low, high, rows + 1)`, the top bound in the
  top row, POC ties to the lowest row, windows with high equal to low or no volume refused.
- D-031 (2026-11-01) Withdrawn by D-042: Levels algorithms. Value area grown from the POC one row at
  a time towards the larger neighbour (ties upwards; a side past the edge never wins) until it holds
  at least `value_area_pct`. HVN/LVN on the profile smoothed with a normalised Gaussian (radius
  `ceil(4 sigma)`, weights normalised with `math.fsum`, symmetric padding, a fixed summation order
  without BLAS, so the Mac and the server agree): local maxima of the profile and of its negation
  with the semantics of `scipy.signal.find_peaks` and `peak_prominences(wlen=None)` reimplemented in
  numpy (scipy is not a dependency; tests pin literal vectors), kept when `100 x prominence >= pct x
  max(smoothed)`. The parameters come from `params.yaml[criterion]`, which stays unfrozen (D-008);
  `levels_from_params` refuses a profile with other rows or window than the criterion's, and the
  `fixed_bp` row mode is not implemented.
- D-032 (2026-11-01; amended 2026-11-02 by D-044) Forecasts. Seeds use the project symbol `BTC-USDT`
  for both sources, so the Binance and BingX fans of one hour and stream share their seeds and
  differ only by their data; the `mismatch-floor` stream gives other seeds on the same window. The
  signal is the sign of `mean_k close[k, N] / close[D - 1h] - 1`; `p_up` the share of trajectories
  strictly above `close[D - 1h]`. `config_id` (16 hex) hashes every input that changes a fan: model,
  weight hashes, device, `max_context`, the predictor clip (`PREDICTOR_CLIP` = 5), `horizon_n`, T,
  `top_p`, `top_k`, trajectories, threads, `amount` policy, seed salt, scheme and symbol, torch,
  numpy and pandas versions, platform and the forecasting-code hash (D-025; the models'
  construction included since D-047); fans of other configurations never share a directory.
  Models that would not forecast as `load_models` leaves them (another clip, context length or
  device; training mode; gradients enabled) are refused; models changed in memory in other ways
  after `load_models` returned them (other tensors, a quantised copy) are not detected, so only
  `load_models` builds the models of stored fans (D-047). Fans
  are `.npz` files read with `allow_pickle=False` and published once; a stored fan is reused only
  after its config id and its metadata (window hashes, `context_sha256`, `D`, source, stream,
  policy, context, horizon) match, never recomputed or overwritten. Batches build every window in
  the parent and drop incomplete hours with their missing ranges; one worker runs in-process, more
  are spawn-started processes of 2 threads (D-014: each calls `import_kronos()` and loads its own
  models), at most `cpu_count // 2` and only with at least 4 GiB of RAM; results come back in task
  order, and the worker count and task order never change a fan (tested bit for bit). The two-worker
  test raises the full suite's RAM peak to 1789 MiB summed over the test process tree (5 processes,
  the largest 703 MiB at that moment; the largest single process over the run 1259 MiB; sampled
  every 0.1 s on darwin-arm64 on 2026-11-01, the method of D-020, which measured 1254 MiB for step
  1).
- D-033 (2026-11-01; amended 2026-11-02 by D-046) Gate-3 rule and flag (SPEC section 17, gate
  3: more than about one hour of CPU). Every batch entry point measures the seconds per fan after
  an untimed warm-up on a synthetic window at the governed thread count, estimates
  `cpu_seconds = pending fans x seconds per fan x threads` (stored fans are not pending) and
  `wall_seconds = cpu_seconds / (threads x workers)`, prints both, and refuses (exit 2) above 3600
  CPU-seconds unless `--approved-by-owner CPU_HOURS` gives the CPU-hours the owner approved in chat
  for the estimate shown and the measured CPU-hours are at most 1.25 times that figure
  (`GATE3_APPROVAL_TOLERANCE`); the refusal prints both figures and their ratio. The figure is
  passed only after the owner's explicit go-ahead in chat for that estimate, never on the agent's
  initiative; `--estimate` stops before the batch's fans; a batch refuses to compute more fans
  than its estimate counted (a stale estimate). The time the flag was given is the command's
  start. `mismatch --kronos` appends its gate-3 record (the approved figure, the measured
  estimate, the config id, the design, the dropped tasks, the command's start) to
  `reports/data/mismatch.gate3.jsonl` once gate 3 has passed and before its first fan, and the
  report lists every record of its fans' config id. The 873 fans of the mismatch's Kronos part
  (about 1.5 CPU-hours at 3.15 s per fan) therefore waited for the owner, who approved them on
  2026-11-01 (D-040); the pre-registered fallback (549 fans, about 0.96 CPU-hours, `--fallback`)
  fits under the limit.
- D-034 (2026-11-01) Study pre-registration and report header (SPEC rules 6 and 7). Every study's
  definition (samples, statistics, blocks, resamples, seed keys, comparators, reading rules) is
  canonical JSON recorded once in `reports/data/<stem>.config.json` before its first run; every
  study command verifies it before reading a candle or running a fan, and a different definition is
  refused (exit 2): a changed study is a new study with a new stem. `--record-config` records a
  definition alone. Every report carries the study, the config and `docs/DATA.md` hashes, the code,
  the generation time, the test budget consumed (0: these studies measure the data, not a strategy,
  and look at no trade outcome; the indicator studies left with D-042), the number of statistics
  with intervals and the 0.05 per interval expected to miss by chance, the seeds, the samples with
  every exclusion and a comparator per metric. The writer refuses a missing header key, a NaN (an
  undefined statistic is `null` with a reason), non-ASCII Markdown, personal data, an absolute path
  and a BingX host name.
- D-035 (2026-11-01) Bootstrap rules. Every interval is a 95 % percentile block bootstrap: 1000
  resamples of whole blocks drawn with replacement, numpy seed = the first 4 bytes of
  `sha256("kronos-agent/bootstrap/v1/" + key)` with `key = <study>/<group>/<statistic>`, quantiles
  `numpy.quantile(method="linear")` at exactly 0.025 and 0.975. Blocks are calendar weeks from
  Monday 00:00 UTC for hourly series and groups of 4 weeks for the 61-hour Kronos sample (its
  contexts overlap for about 3 weeks) and the `amount` `dz`. Statistics aggregate per-row values
  built before resampling, so block seams never create values; paired comparisons resample both
  columns in one call. Point estimate only, with the reason: fewer than 20 blocks, more than 1 %
  non-finite resamples, or a degenerate distribution, except a share of exactly 0 or 1, which gets
  a Wilson interval at the number of blocks; shares also report `deff` and `n_eff = n / deff`;
  p99 and max are descriptive.
- D-036 (2026-11-01; amended 2026-11-02 by D-042 and D-046) Comparators and reading rules. Every
  metric names its comparator; the mismatch (study 3): the lag +-1 h correlations, one 0.1 USDT
  tick, the median `|1h return|`, the SPEC's 0.05 bp demo-vs-real median (external), the `dz` of
  the volume channel and the two-seed floor pair. Pre-registered readings: the mismatch report
  fails unless the lag-0 return correlation is the largest; `A_x - A_f` entirely above 0 reads
  "the data effect is smaller than seed noise" and entirely below 0 "the data effect is larger
  than seed noise"; `A_xd - A_f` entirely below 0 reads "the sources shift the signal beyond seed
  noise" and entirely above 0 "the cross-source pair agrees more often than the two-seed floor";
  an interval containing 0, or no interval, reads "not distinguishable"; always with the minimum
  detectable difference `(z_0.975 + z_0.80) x SE` read from the interval. The owner's 88 %
  two-seed floor is an external reference, not a threshold. The comparators of the first design's
  studies 3 to 5 (the median `|1h close-to-close return|`, 5m against 1m, the reference profile's
  own one-hour movement), the BingX one-hour POC noise and the POC reading rule left with the
  volume profile (D-042).
- D-037 (2026-11-01) Figures (SPEC section 14). matplotlib's object-oriented API on an Agg canvas
  (no pyplot, no GUI backend), a fixed size of 8 x 4.5 in at 100 DPI, matplotlib's defaults plus
  fixed rcParams (DejaVu Sans, which ships with matplotlib; a user's matplotlibrc never applies),
  PNG without the `Software` metadata, written atomically; two renders give identical bytes
  (tested). Figures are regenerated by the command that writes their report, under
  `reports/data/figures/`, with their sha256 in the report. Amended by D-043 and D-046:
  `RC_PARAMS` also pins `timezone` UTC and `date.epoch`, which `rcdefaults()` leaves alone, so
  date axes are placed and labelled in UTC whatever a `matplotlibrc` says [foundation-2]; a report
  without a part it had before removes that part's figure and series of the earlier report, so the
  files beside a report are the ones it lists [mismatch-core-3].
- D-038 (2026-11-01) ATR moves to steps 4-5 (deviation from the step-1 `docs/ARCHITECTURE.md`,
  which listed `kronos_agent/indicators/atr.py` under step 2). ATR is used only by the criterion
  (arrival within `k x ATR`, stop at `s x ATR`), SPEC section 4 is under review (D-008) and no
  step-2 study needs it; it is built with the criterion. `indicators` stays a reserved name.
- D-039 (2026-11-01; amended 2026-11-02 by D-042 and D-046) Command line of step 2. Commands `data`
  (`binance`, `bingx`, `status`, `coverage`), `forecast` and `mismatch` (the `profile` command left
  with D-042). Exit codes: 0 done; 1 failure (`ERROR:`); 2 refusal by a governance rule (`REFUSED:`:
  gate 3, the gate-4 budget, the sealed partition, the Kronos cutoff or the embargo, a changed study
  config, a busy lock); argparse usage errors (an invalid hour, options of the other mode) also exit
  2; an unexpected exception keeps its traceback. Command modules import only the standard library,
  the registry's helpers and `kronos_agent.{paths, params, models, data.timeutil, data.partition}`
  at load (checked on their syntax tree), and `--help` imports no numpy, pandas, matplotlib or torch
  (checked in a fresh interpreter); a command that runs Kronos configures the runtime with the
  governed thread count first. The network clients and the clock of `data` are module functions that
  tests replace, so no unit test touches the network; data go under `KRONOS_AGENT_DATA_DIR` (default
  `data/`), reports under `reports/data/`. A store read while a data run holds its job lock is a
  busy lock for `forecast` and `mismatch` too (exit 2); `data status` exits 1 when a store's size or
  sha256 does not match its meta; SIGTERM and SIGHUP end a command through `SystemExit(128 + n)`, so
  its clean-up runs (a signal the parent process ignores stays ignored).
- D-040 (2026-11-01; amended 2026-11-02 by D-046) Gate-3 approval of the mismatch's Kronos
  part. Shown the estimate in chat (873 fans: 291 decision hours x 3 fans of 20 trajectories x 4
  steps; 3.15 s per fan measured with 2 threads, about 1.5 CPU-hours; about 46 min of wall time
  with one worker, 12-15 min with four), the owner approved the run on 2026-11-01 (their words,
  translated: "yes, run the 873"). The approval covers exactly that batch, `mismatch --kronos` on
  the primary 61-hour sample, at that estimate: the run passes `--approved-by-owner 1.53` (the
  CPU-hours shown: 873 x 3.15 s x 2 threads, about 5,500 CPU-seconds), appends its gate-3 record
  to `reports/data/mismatch.gate3.jsonl` before its first fan, and its report lists that record
  and the measured estimate; a measured estimate well above the one shown (more than about 25 %:
  the command refuses more than 1.25 times the figure) goes back to the owner before any fan of
  the sample runs. It does not extend to any other batch (forecast-command batches, the fallback
  sample, step-3 fans).
- D-041 (2026-11-01) Live checks of step 2 (opt-in network tests, public hosts only: 4 GETs to
  data.binance.vision and 7 to the BingX real market; all 7 tests passed). Binance Vision: the
  2020-01 1h archive (37,271 bytes) verifies against its `.CHECKSUM`, has no header line and holds
  744 rows, every hour of the month, with whole and decimal close strings; the 2019 monthly
  `.CHECKSUM` answers 404. BingX: the backfill's request for a long-closed 1h chunk (`startTime`
  at the chunk start, `endTime` at the next chunk start - 1 ms) returned all 1000 candles in one
  page and `fetch_chunk` matched on its first pair; the same request with `endTime` exactly at the
  last candle's open returned only the first 999 (an exact-boundary `endTime` excludes the candle
  opening there, so the backfill keeps the `boundary - 1` form); a chunk entirely before the
  history answered `code 0` with an empty list; the 1h history still starts at 2024-04-27T10:00Z.
  Amended by D-043 [tests-docs-2]: the test fake (`tests/step2_fakes.FakeBingX`) now models these
  bounds, `startTime <= open < endTime` and `open <= server time`; tests pin that the backfill
  gives identical archives under the exclusive and an inclusive `endTime` (production never sends
  an exact boundary). Amended by D-047 [tests-docs-2]: the step-1 public client's `closed_klines`
  and `fetch_twice` select the half-open `[start_ms, end_ms)` (a candle opening exactly at
  `end_ms` is dropped, as the exchange's `endTime` drops it, whatever the exchange does), and the
  step-1 test fake (`tests/test_public_market_client.py`) models the observed bounds too.
- D-042 (2026-11-02) Owner directive: the volume criterion is discarded and the work narrows to
  retraining Kronos. The owner stated that the volume-based criterion of SPEC section 4
  (rebounds at volume-profile levels) was discarded long ago; their new strategy is being defined
  in `research/strategy/` (still off limits) and comes after the retraining. Decided with the
  owner on 2026-11-02:
  (1) SPEC section 4, the volume-profile and POC parts of section 5 (the profile, the platform
  VPVR comparison, where the POC falls in each source) and every step of section 18 that depends
  on the criterion are without effect until the owner supplies the new strategy; `docs/SPEC.md`
  stays verbatim. The step-2 "done when" becomes the mismatch table and the no-future-candles
  test; the hour-by-hour POC is dropped.
  (2) The step-2 volume-profile code (studies 3 to 5, the `profile` command, the POC part of the
  mismatch study) leaves the repository; a copy is archived outside it. D-030 and D-031 are
  withdrawn; D-036's comparators for studies 3 to 5 and its POC reading rule, and D-039's
  `profile` command, go with them. The mismatch study is now study 3 of `docs/DATA.md`, and
  question 5 (the volume-profile source) is moot.
  (3) The current work is SPEC section 9 in full: tokenizer diagnostic on BTC, a predictor-only
  fine-tune with the pinned `finetune_csv` and early stopping on validation, walk-forward
  validation of the recipe against Kronos as shipped by CRPS, then the model trained to date with
  its model card.
  (4) The Binance vs BingX mismatch study stays without its POC part, with the Kronos signal
  agreement and its 873 fans (D-040): the model is trained on Binance and used on BingX candles.
  (5) `params.yaml[criterion]` still describes the discarded criterion; it is neither used nor
  frozen (D-008).
- D-043 (2026-11-02) Review of the step-2 data layer and its fixes. An adversarial review ran 11
  reviewers over the uncommitted step-2 tree and verified each finding independently: 50
  findings, 44 confirmed (6 of them in the volume code removed by D-042) and 6 refuted. Its
  completeness round did not run because the owner's redirect (D-042) stopped it. Every confirmed
  finding outside the removed code was fixed, the code ones with tests that fail without the fix
  (finding ids in brackets):
  (1) Forecast configuration [foundation-1, kronos-2, rules-4]: the config id also covers the
  predictor clip and the numpy and pandas versions, models that would not forecast as loaded are
  refused, and each fan records the `context_sha256` of the floats it read (D-044; D-025, D-032).
  (2) Stores, locks and the BingX archive [foundation-2, foundation-4, foundation-5, bingx-1,
  bingx-2, tests-docs-2, tests-docs-3, tests-docs-5, deviations-3]: store reads hold the source's
  job lock shared and parse the bytes they verified; `job_lock` removes the temporary files of
  killed runs; archived BingX candles are immutable and a missed run no longer blocks the backfill
  (D-045); the history start is recorded only when a walk establishes it; `data status` checks
  every store's sha256; the BingX test fake follows D-041; figures pin their time zone (D-024,
  D-027, D-037, D-041).
  (3) Gate 3, saved windows and the commands [foundation-3, binance-1, kronos-1, mismatch-core-1,
  mismatch-core-3, mismatch-core-4, mismatch-kronos-1, mismatch-kronos-2, mismatch-kronos-3,
  mismatch-kronos-4, mismatch-kronos-5, mismatch-kronos-8, rules-1, rules-2, rules-3,
  tests-docs-1, tests-docs-4, tests-docs-6, deviations-1, deviations-2, deviations-5,
  deviations-7]: `--approved-by-owner CPU_HOURS` with a 1.25 tolerance and a persisted approval
  record, every section without Kronos before any fan, a plain `mismatch` that rebuilds the Kronos
  part from stored fans, readings of both directions, saved windows that hold their inputs only,
  gate 4 checked once per run (D-046; D-022, D-029, D-033, D-036, D-039, D-040).
  (4) Documentation [lookahead-1, lookahead-2, deviations-8, tests-docs-7]: `docs/DATA.md` states
  one horizon-purge cut-off, once; `check_embargo` bounds the trading horizon N, and longer
  evaluation horizons go through the purge (D-028); the README says that the 873 fans are approved
  (D-040) and not yet run.
  (5) The 6 findings in the volume code [profile-1, profile-2, profile-3, profile-4, profile-6,
  mismatch-core-2] left with that code (D-042).
  Deferred: what to do with a revised BingX candle is the owner's decision (D-045); until then a
  revision stopped every backfill of its interval (decided on 2026-11-02: the archive keeps the
  first-seen row, the revision is recorded and the backfill goes on, D-045). The explicit
  evaluation length for horizons longer than N is designed with the retraining (D-028). Two
  optional suggestions were not done: listing the distinct source-tree hashes of the fans behind
  the Kronos part [rules-4], and a `ForecastStoreError` hint that other fans of the same
  configuration and source may be stale too [mismatch-kronos-3]. `StoreWindowSource.meta` is
  still read after its frame, outside one lock hold [foundation-4]; nothing records it yet, and a
  study that does must read both under `store_read_lock`.
- D-044 (2026-11-02) Forecast configuration after the review [foundation-1, kronos-2, rules-4].
  (1) `kronos.weights.PREDICTOR_CLIP` = 5 is the clip `load_models` builds every predictor with,
  and the config key `predictor_clip`. (2) `inference_problems` refuses, in `forecast_window`
  (`DecisionError`) and `forecast_config` (`ForecastStoreError`), a predictor whose clip, context
  length or device differs from what `load_models` builds, and a tokenizer or model with any
  submodule in training mode or any parameter requiring gradients. Models without torch modules
  (the registry description, test stand-ins) are not asked for a clip. (3) `numpy_version` and
  `pandas_version` sit next to `torch_version` in the config. (4) Each stored fan records
  `context_sha256` (its float context and its x and y timestamps) and `check_reusable` compares
  it; fan schema 2 (`load_fan` refuses schema 1). (5) The forecasting-code hash keeps hashing
  whole files (the safe side); `tests/test_provenance.py` classifies every module of the fan
  path's import closure as hashed or covered otherwise. Consequence: every `config_id` changes,
  so fans stored under the old ids are not reused; none exist (this working copy has no `data/`).
  The D-040 batch (873 fans, same estimate) runs under the new id. Amended by D-047
  [foundation-1]: (1) and (2) cover only the clip, context length, device, eval mode and
  gradients, so the models' construction (`kronos/weights.py`, `kronos/vendored.py`) is hashed
  whole as well.
- D-045 (2026-11-02; amended 2026-11-02 by the owner's decision) BingX archive refresh after
  roll-off and revisions [bingx-1, tests-docs-3, deviations-3; bingx-2]. Archived candles are never
  overwritten or deleted. A missed run whose incomplete chunk rolled off the 1m or 5m retention no
  longer blocks the interval: the archived candles are kept (`archive_only` in the sidecar,
  sidecar schema 2) and the chunk completes (amended by D-047: only when the retention explains
  every kept candle; otherwise the chunk stays incomplete for the next run). A revised candle (a
  different string for an archived open) stopped every backfill of that interval at that chunk,
  naming both values, with a `data status` warning (from the first failed run on, through
  `raw/bingx/revision-<interval>.json`, D-047); `data bingx` still rebuilt that interval's store
  from the chunks archived so far and ran the other intervals. The history start is recorded in
  `raw/bingx/history-<interval>.json` only when a walk establishes it, and the coverage study
  reports the observed history depth only then (D-027). What to do with a revision was left to
  the owner. Amended on 2026-11-02 by the owner's decision: a revised candle keeps its first-seen
  row in the archive, which is what a live decision would have used; the revision is recorded
  durably and idempotently in `raw/bingx/revision-<interval>.json` (record schema 2: interval,
  chunk, open time, the archived and the refetched rows, the server times of the first and the
  last observation; the same revision seen again by a later run is not recorded twice, only its
  last observation moves; a schema-1 record is read as its one revision); and the backfill
  continues. The rest of the chunk is refreshed as before: new settled candles are added,
  archive-only candles are kept, completion follows the existing rules. `data bingx` reports the
  revisions a run observed as a WARNING and exits 0 when nothing else failed; `data status` lists
  every recorded revision as a WARNING (open times and observations; the candle values stay in
  the record, so no value of the sealed segment is printed). Revisions no longer stop any
  backfill. The store is rebuilt from the archive, so it holds the first-seen rows. Nothing ever
  overwrites or deletes an archived candle. Amended by D-050: `data bingx` prints each chunk's
  revisions as soon as they are recorded, also when the interval then fails, and a chunk fetched
  again after its files were removed takes the first-seen rows of its recorded revisions.
- D-046 (2026-11-02) Gate 3 carries the approved figure; saved windows and the commands after the
  review (finding ids in D-043). Design decisions of the review fixes: no approval of the owner
  changes, and (a) enforces the condition that D-040 attaches to the approval.
  (a) `--approved-by-owner CPU_HOURS` (forecast `--from`/`--to` batches, `mismatch --kronos
  [--fallback]`) gives the CPU-hours the owner approved for the estimate shown; above 3600
  CPU-seconds a measured estimate more than 1.25 times it (`GATE3_APPROVAL_TOLERANCE`) is refused
  before any fan, with both figures and their ratio; a bare flag, or a figure that is not a finite
  number > 0, is a usage error (exit 2). Amends D-033 and D-040: D-040's run passes
  `--approved-by-owner 1.53`.
  (b) `mismatch --kronos` appends its gate-3 record to `reports/data/mismatch.gate3.jsonl` before
  its first fan (append-only, stamped with the command's start; a rerun with nothing pending adds
  none), and the report lists every record of its fans' config id, so no rerun or resumed run
  erases the approval its fans were computed under. Amends D-033.
  (c) A plain `mismatch` rebuilds the Kronos part from the stored fans when a whole
  pre-registered sample is stored under the current config (no fan computed, gate 3 not
  involved); otherwise it says the part has not run and removes the agreement figure and the
  Kronos series of an earlier report, so the files beside a report match its listing (D-037).
  Amended by D-047 [mismatch-core-3]: a stored fan of a sample that no longer matches its window
  also leaves the part not run, its reason naming the fan to move aside, instead of failing the
  command.
  (d) Every section without Kronos, the alignment check included, runs before any Kronos work
  (also with `--estimate`): approved compute is never spent on a report that then fails.
  (e) The readings name both directions of each paired difference; only an interval containing 0,
  or none, is "not distinguishable". Amends D-036.
  (f) A saved window holds its inputs only (the candles, their sha256 and, for Binance, the quote
  volumes) under `windows/<source>/context-<max_context>/<D>.json`; the `amount` policy,
  `inputs_sha256` and the fetch pairs are left to each fan's metadata, so a policy change or a
  live retry keeps the saved file, and `forecast --save` checks the window before any model is
  loaded. Amends D-029; window schema 2.
  (g) Gate 4 is checked once per `data binance` run, for all its intervals, before any download.
  Amends D-022.
  (h) `data status` checks every store's sha256 and lists the temporary files of killed runs;
  `forecast` and `mismatch` refuse a store read while a data run holds its job lock (D-039);
  SIGTERM and SIGHUP end a command through `SystemExit(128 + n)`, so its clean-up runs. The
  mismatch study records each store's `csv_sha256` as read (both job locks held shared; the meta
  read before and after each frame must agree), and its Markdown lists the `amount` exclusions
  and the skipped sample hours.
- D-047 (2026-11-02) Second check of the review fixes. Independent checkers replayed the fixes of
  D-043 to D-046 and found six problems, all confirmed and fixed with tests that fail without the
  fix (finding ids in brackets):
  (1) [foundation-1] The forecast `config_id` covered `load_models` only through five properties
  (clip, context length, device, eval mode, gradients): an edit that changes none of them (a
  dynamic int8 quantisation, another checkpoint under the registry's weight hashes) reused the
  stored fans and mixed new ones into their directory. `kronos/weights.py` and
  `kronos/vendored.py` are now hashed whole into the forecasting-code hash, every `config_id`
  changes again (no fans are stored in this working copy; the D-040 batch keeps its estimate and
  approval). Models changed in memory after `load_models` returned them are checked only by
  `inference_problems` (D-044), so only `load_models` builds the models of stored fans; a step-6
  loader of fine-tuned weights must record those weights' own hashes (D-025, D-032, D-044).
  (2) [deviations-3] A revised candle in the newest chunk, the usual case, stopped every backfill
  of its interval while `data status` stayed silent until a newer chunk was archived (up to one
  chunk span, 1000 h for 1h). The backfill now records the revision in
  `raw/bingx/revision-<interval>.json` (the chunk, the number of revised candles, the first one
  with both values, the server time) before it stops, and `data status` warns from the first
  failed run on, while no refresh of that chunk has succeeded since; the error says that the chunk
  is left as archived and that chunks archived earlier in the run are kept (D-027, D-045).
  Superseded on 2026-11-02 by the owner's decision (D-045): a revision no longer stops the
  backfill, the record lists every revision once (schema 2), and `data status` lists them all.
  (3) [tests-docs-3] A refreshed chunk completed whatever its refetch returned, so one anomalous
  empty or short answer turned the slots never archived into a permanent gap. It now completes
  only when the 1m or 5m retention explains every archived candle the refetch no longer returned
  (older than the observed retention and than the first candle the refetch returned; never for
  1h, whose history is fixed, or 15m, whose retention is unknown); otherwise it stays incomplete,
  those candles count in `kept_inside_served_range` (a WARNING) and `data status` warns until a
  later refetch fills the chunk (D-027, D-045).
  (4) [tests-docs-2] The public client's `closed_klines` promised an inclusive `end_ms` and the
  step-1 test fake certified it, while the exchange excludes a candle opening exactly at
  `endTime` (D-041). The client now selects the half-open `[start_ms, end_ms)` whatever the
  exchange does at that edge, and the step-1 fake models the observed bounds; no production path
  changed (the live window and the backfill never send an exact boundary) (D-041).
  (5) [bingx-2] The documents said that any run bounded with `--max-chunks` never writes the
  history-start record. That holds only for a run that the bound stops (stop reason
  `max_chunks`); a bounded run whose walk ends before the bound establishes the start like any
  other, which is right (D-027). Documents, help text and messages only; the coverage study's
  definition says the same (it has not been recorded in this working copy).
  (6) [mismatch-core-3] A plain `mismatch` failed (exit 1, nothing written) on a stored fan that
  no longer matched its window, keeping an earlier report made from the old stores. It now writes
  the report of the current stores with the Kronos part not run, its reason naming the fan to
  move aside, and removes the earlier Kronos figure and series (D-046).
- D-048 (2026-11-02) The owner's answers to the six questions of the retraining design panel
  (SPEC section 9, the current work of D-042 (3)); the owner chose every recommended option:
  (1) Compute: training runs on the Mac's GPU (MPS) after a measurement step. It uses 24 monthly
  cold-start folds if the measured worst case is at most 72 h, otherwise 8 quarterly folds over
  the same test decisions. Every run above about an hour waits for a gate-3 approval of a
  measured estimate.
  (2) Acceptance of recipe R1 against Kronos as shipped: the 95 % interval of `Delta_N` (the CRPS
  difference at N = 4) lies entirely below 0, `Delta_N < 0` in at least 5 of the 8 quarters, and
  the relative improvement is at least 1 %.
  (3) The sealed segment opens once, only if R1 is accepted, on one recorded order of the owner
  that also covers training the model to date. Otherwise the model to date is trained up to
  2026-08-01 without opening it, and its model card states the negative result.
  (4) Seeds: one training seed per fold, plus two extra seeds in the folds of 2024-07, 2025-06 and
  2026-06.
  (5) Normalisation: the first test uses the vendored script's normalisation unchanged, with
  windows of 513 rows (lookback 512, predict 0), so one future row enters it. If a measurement
  before registration shows more than 5 % of context tokens changing, the owner is told first.
  Context-only normalisation is the second test.
  (6) Embargo: 512 h between the end of each fold's training data and its test month (the
  per-model reading of SPEC section 6).
  The full design becomes `docs/RETRAINING.md` together with its implementation.
- D-049 (2026-11-02) BingX 1m and 5m archive. The owner chose on 2026-11-02 to archive the BingX
  real market's 1m and 5m candles too (`data bingx --interval 1m --interval 5m`, the 5m at least
  monthly). They are insurance for the owner's future strategy, because the exchange rolls them
  off (about 450 and 48 days, D-022): a candle not archived in time is lost. The retraining does
  not use them. The facts of the first download are added here when it finishes (see D-050 (4)).
- D-050 (2026-11-02) Check of the owner's decision on revised candles (D-045). An independent
  checker replayed the change and found four problems, all confirmed. The first three are fixed
  with tests that fail without the fix (and under mutations of it); the fourth waits for D-049:
  (1) Nothing pinned the order D-045's durability relies on: a revision recorded after its
  chunk's commit marker is lost for good when a kill falls between the two and that refresh
  completed the chunk, because a complete chunk is never fetched again; nor that the merge never
  reads the revision record. Tests now kill a run at the chunk's CSV and at its sidecar after a
  revision (the record already lists it, the chunk is as archived, the next run moves only the
  last observation), and rebuild the store beside every record that cannot be read.
  (2) A revision recorded before a later chunk of the same walk failed (a rate limit, a transport
  error, fetch pairs that never matched) reached stdout nowhere: only a log line on stderr,
  without its open time, while `data bingx` was documented to print it. The backfill now hands
  each chunk's revisions, once recorded, to a `warn` callback (default: its logger), and
  `data bingx` prints them as a WARNING at once (the chunk, how many were new, their open
  times), so also when the interval then fails; it no longer prints them from the summary.
  (3) A kill between a refresh's CSV and its sidecar leaves a CSV its sidecar does not describe.
  The only repair the error names, removing both files, archived the exchange's current values
  over the first-seen rows of the chunk's recorded revisions, while the record and `data status`
  said the archive keeps them. A chunk fetched again now takes, for the candles the exchange
  serves, the archived row of each open's earliest recorded revision (another value served is a
  revision like any other), and the error and `docs/DATA.md` say what that repair keeps and what
  it loses (the candles past the retention).
  (4) The retention figures. `RETENTION_DAYS` and the documents give 5m about 48 days, but the
  step-1 start (about 2026-09-14 on 2026-10-29) and the first download's (2026-09-18T14:00Z at
  2026-11-02T14:02Z) both give 45 days. The first download also measured 15m at 100 days (from
  2026-07-25T14:00Z; the documents call it unknown) and found the 1m start at 2025-07-31T16:00Z,
  the date step 1 saw four days earlier, which a rolling 450-day retention would have moved. Not
  changed here: `RETENTION_DAYS` is part of the recorded definition of the coverage study
  (`reports/data/coverage.config.json`), so a change makes coverage a new study; the 1m
  observation questions the 1m retention itself; and a 15m entry (completion after a roll-off,
  an age warning) concerns an interval D-049 does not name. The figures are revised with the
  facts of D-049. Meanwhile no candle is at risk: 48 days never completes a chunk early, and a 5m
  chunk whose candles rolled off 45 to 48 days ago stays incomplete, with a warning, for at most
  three days; a 15m chunk whose candles rolled off stays incomplete and warns until then.
- D-051 (2026-11-02) Owner answers on the step-2 results and on three open step-1 questions,
  given in chat after the step-2 summary (every recommended option).
  (1) Binance 1h for training and the studies is the store resampled from 1m, as it is. The 16
  hours of 59,928 (0.03 %) where it differs from Binance's official 1h archives stay documented
  in `reports/data/binance_resample_check.md` and are neither replaced nor excluded; the largest
  is 2023-11-10 15:00-17:00Z, where the 1m archives hold about a tenth of the official volume.
  (2) Question 8: the partition of the table above is confirmed.
  (3) Question 9: the `amount` proxy (A) is confirmed; the mismatch study measured its error
  against the Binance quote volume at a median of 0.08 bp (`reports/data/mismatch.md`).
  (4) Question 12: `data/` and `weights/` stay in the repository, gitignored and relocatable.
  Questions 5, 6, 7 and 13 concern the discarded criterion and are without effect (D-042);
  question 10 is decided through D-042 (3) and D-048.

Decisions D-052 to D-055 concern step 6, the retraining of `docs/RETRAINING.md` (D-042 (3),
D-048): the build of its phase 1 (item 1 of the order of work in its section (c)), the first run
of its measurement step (item 2), the fix that run led to and the owner's gate-3 approval of the
second run. They record the deviations from the design and from the interface specification its
implementers worked from (a scratch document, not kept), with the finding ids of the phase-1
review in brackets. None registers an event definition, freezes `params.yaml[criterion]` (D-008)
or consumes test budget.

- D-052 (2026-11-02; (6) amended 2026-11-02 by D-054) Phase 1 of the retraining: the code the
  measurement step needs. Three implementers (data, training and model sides) built it, two
  reviewers (look-ahead; training mechanics) replayed it in scratch, a fix-up applied their
  findings and three gates ran the offline suite and ruff. New: `finetune/recipe.py`, `folds.py`,
  `export_csv.py`, `nll.py`, `runner.py`, `train.py`, `registry.py`, `timing.py` and
  `cli/finetune.py` (`finetune timing`); additive changes to `kronos/weights.py`,
  `kronos/decision.py`, `kronos/batch.py` and `data/partition.py` (`docs/ARCHITECTURE.md`,
  "Step-6 modules"). R1's `recipe_id` is
  `0ee0d123268b2c2fbca2b42e4b0f5b5e23023807105fcb666d159841bee5fdfa` (recipe8 `0ee0d123`),
  pinned as a test vector with the pilot's training seeds. The offline suite went from 1,439 to
  1,888 passing tests (15 network tests deselected; the 4 xpasses of D-007). During the build
  nothing was trained, exported or timed for real, and nothing was written under `data/`,
  `reports/` or `weights/` (D-053 is the first real run).
  (1) Runs and checkpoints, as `docs/RETRAINING.md` 3.1 and 3.6 designed them, replace the
  step-1 plan of `docs/ARCHITECTURE.md` (the `finetune_csv` scripts driven by subprocess,
  checkpoints in a second manifest). Each fine-tune is one child process,
  `python -B -P -m kronos_agent.finetune.runner`, that imports the pinned predictor phase from
  the verified tree and runs it unmodified behind switchable wrappers (with every wrapper off it
  reproduces `python -B train_sequential.py --skip-tokenizer` bit for bit: test T1); its
  environment is built from an allow-list (no credential variable reaches it) and sets
  `PYTHONHASHSEED=0`. Checkpoints live in `<weights>/finetuned/<checkpoint_id>/`, registered in
  the append-only, hash-chained `<weights>/finetuned/registry.jsonl`, outside `manifests/`,
  whose files enter `source_tree_sha256` (D-025).
  (2) Interfaces. A checkpoint is named by `CheckpointRef(checkpoint_id, root, line_sha256)`, so
  scratch and test registries work (`LoadedModels.checkpoint` stays the id). In the
  `evaluation` role of `partition.require_model_period`, every caller's default, a checkpoint
  forecasts only `D >= max(data_end, 2024-07-01)`, stricter than the design's `D >= data_end`
  because every checkpoint starts from the shipped weights (no planned decision is affected);
  the opt-in roles `pilot-check` and `timing` (new: the synthetic timing window is dated
  2026-10-29T17:00Z) allow every hour, `cutoff-check` only for the shipped weights, and their
  callers record them. `require_kronos_period` stays for `cli/forecast.py`. `run_fan_batch` takes
  the evaluation length, the checkpoint and the role from the `PreparedBatch`. The identity
  control is `kronos.batch.identity_control`, not part of `finetune/registry.py`, which
  `kronos.weights` imports and which therefore stays off torch and the batch layer (the fan-path
  import closure of `tests/test_provenance.py`). The exporter writes each fold's NLL4-TF inputs
  once (`nll_inputs.npz`, deterministic bytes), so the child never builds a window or reads a
  store. `export.json` holds `content`, whose sha256 identifies the export, and `provenance`
  (the store meta read, which changes with the future candles test T3 alters, so T3 compares
  `content`). Device, threads, determinism and activation checkpointing are compute choices
  outside `recipe_id`; the batch size is inside it (a batch change is a recipe change). The 9973
  rule is checked on the kept fit-window count and on the validation count. A registration also
  removes the run's `selected/` directory. Not in phase 1: spend control inside a fan batch and
  the walk-forward projection (design 3.5).
  (3) Forecast configuration. The `kronos/weights.py` and `kronos/decision.py` edits (both
  `FORECASTING_FILES` members, D-047) changed every forecast `config_id`, once and before any
  evaluation fan (design (c) names only `decision.py`; `weights.py` is hashed since D-047): the
  shipped model's configuration is now `62906d91f853a8e5`, while the 873 D-040 fans keep
  `b2723b9626339a08`. Those fans stay valid for their own study (`reports/data/mismatch.md`), but
  a plain `kronos-agent mismatch` would now find no whole sample stored under the current
  configuration, report its Kronos part as not run and remove its figure and series (D-046 (c)):
  it is not run again without the owner's decision. No later phase-1 change touched a
  `FORECASTING_FILES` member.
  (4) Training child. MPS children also get `PYTORCH_MPS_LOW_WATERMARK_RATIO` = min(1.4, high):
  torch 2.14.1 on this Mac refuses a low watermark above the high one, so the high ratio 1.0 of
  design 3.5 alone fails every MPS allocation ("invalid low watermark ratio 1.4"). The preflight
  statistics that design (a) left open: `multinomial` passes at a Pearson chi-square p >= 1e-3
  over the categories (cells under 5 expected draws pooled) with no draw of a zero-probability
  category, beside a CPU control; `parity` needs max |device - CPU| / max |CPU| < 1e-3 on the s1
  logits and a cosine above 0.9999 between all s1-loss gradients (train mode, dropout 0);
  `steps` runs 20 steps of the vendored loop body on the device and on the CPU, finite and with
  mean losses within 1 %; `deterministic` is one step under deterministic algorithms, refused
  when torch names determinism; `checkpoint_gradients` compares every parameter gradient of one
  CPU step, dropout on, bit for bit with and without the wrap. NLL4-TF takes `-log_softmax` in
  float64 of the float32 logits and reduces the context statistics on a Fortran-ordered float32
  copy, the layout the predictor reduces (a C-ordered float32 sum differs in the last bits; test
  T16's bit identity needs it). On torch 2.14.1 a single `.to("cpu", torch.float64)` of an MPS
  tensor returns NaN: tensors move to the CPU first. The memory guard reads `sysctl` (macOS), so
  the server runs with `Compute(memory_guard=False)`. With early stopping off, a train run still
  records NLL4-TF at every pseudo-epoch (it draws no random number) and keeps the vendored
  keep-best. The child recomputes the recipe id and the seeds, requires `run.yaml` to equal the
  recipe's document and the vendored split to give the export's row counts, and in a train run
  refuses a pseudo-epoch that would draw more windows than the fold holds; `start_run` refuses an
  export of another fold. The integration parts of tests T2, T3 and T8 are in
  `tests/test_finetune_train.py` and `tests/test_finetune_runner.py`. Not written: a `weights`
  test of `fit(register=True)` with the real Kronos-small (about a minute of training).
  (5) Fold plan and export. The fit rows and the inner validation must each hold more than one
  window (at least 514 rows for R1; the vendored dataset raises otherwise). A CSV ending inside
  the test embargo is a `PartitionError`, one ending earlier a `FoldError`. `check_fold` takes
  the recipe (default R1) and `nll_inputs_bytes` the grid (default 4 h). `Recipe.history_start`
  accepts only its canonical form (one hour, one recipe id). Synthetic folds are labelled
  `synthetic-<lowercase segments>`, at most 48 characters, so a checkpoint id stays within 64. A
  model to date before 2026-08-01T00:00Z is refused (D-048 (3)). `export.json` lists every
  missing hour of `[csv_start, csv_end)`. `export_fold` applies the sealed guard before anything
  else and re-reads the store meta after the frame (the rule of D-046 (h)).
  (6) The measurement step (`finetune/timing.py`, design (a)). Unit order: the synthetic export
  and the shipped models, M9, M8, M5, the models released, M3, M1 (the CPU at 2 and at 8
  threads, then MPS once M3 passed), M2 on MPS when needed, M4 with M7 during it, M2 on the CPU
  when needed, M6, where the interface specification ran M3, M1, M2, M4 with M7, M5, M6, M8, M9.
  So the cheap measurements every estimate needs, and M9, which must pass before any fold, cannot
  be lost to a budget stop; the parent's models (about 1 GB) are freed before the training
  children; M2 on the CPU follows the device choice; M6, the least critical, runs last. M4
  deadline: the 20-minute M4 child gets the remaining GPU time, its CPU time is read with
  `/bin/ps` every 5 s, and it is stopped once that time plus the spend passes a limit; the rule
  of the other MPS children (the smaller of the remaining GPU-minutes x 60 and the remaining
  CPU-seconds / 2) would cut it short. CPU charging: each measurement is charged the larger of
  the CPU time measured (`getrusage` of the process and its reaped children) and the figure it
  reports (a child's CPU time; fans: wall x threads, D-033), and an MPS child's wall time as GPU
  time. Exit codes: a run its budget stopped writes its report and exits 2 (`REFUSED:`, like an
  estimate above the budget); a failure (an M9 mismatch, a crashed child, a measurement or
  environment error) exits 1 after the report; `--estimate` exits 0 and runs nothing. Swap
  tolerance: "no swap growth" allows 256 MiB, judged on the larger of the child's guard reading
  and the parent's `vm.swapusage` before and after the child (the guard reads every 50 steps, so
  a 13-step child would show none). Also: an MPS peak is the child's peak RSS plus its largest
  MPS driver allocation (unified memory); M3 runs at batch 32 with deterministic mode off, since
  its check (d) tests that mode; M7 starts 180 s into M4; M2 is recorded per device; the timing
  recipes are R1 with the configuration's batch and pseudo-epochs of 4,096 steps, so that no
  vendored validation runs inside a measurement, on the fold `synthetic-timing` (4,096 synthetic
  hours); the timing scratch (about 0.1 GB) is kept; one timing run at a time (a lock under the
  fine-tune root). Amended by D-054: the CPU's M1 and M2 run before any MPS unit; MPS units start
  with activation checkpointing where the CPU's batch 32 needs it, and a memory abort without it
  is retried once with it; M2 is recorded once (`gradient_check`) with its runs per device; MPS is
  compared with the fastest CPU configuration that runs batch 32; M3 and the retries are priced
  by their content and M4 by its whole lifetime; the record is written under a new stem.
  (7) Review. Two reviewers reported 8 findings [F1-F8]; all were confirmed and fixed with tests
  that fail without the fix (each defect re-introduced in memory, and seven batch-layer
  regressions, made them fail):
  (a) [F1, F5] Only a recipe run is registered. `fit(register=True)` registered whatever the
  child selected, so a run with every wrapper off (selected by the vendored keep-best on the
  eval-mode validation loss, which reads its target) took the recipe's checkpoint id. It now
  refuses before the child starts (`TrainError`) unless every wrapper of `runner.WRAPPERS` is on
  and the run starts from the registered shipped weights, and after the run a record whose
  wrappers are not all on or whose `selected.source` is not `nll4_tf`. Diagnostic configurations
  (T1) run with `register=False` or `train.run`.
  (b) [F8] An id already registered, or whose directory exists, is refused (`CheckpointExists`)
  before the child starts, and so is a broken registry (`RegistryChainError`): no fold is
  trained only to be refused. `install_checkpoint` checks the id again under its lock.
  (c) [F7] The hash chain protects every line but the last: an edited or removed last line left
  a valid chain, and its `data_end_ms` drives the model-period guard outside the forecast
  configuration. `CheckpointRef` gained `line_sha256`; `find_entry` and `checkpoint_problems`,
  hence `load_models(checkpoint=...)`, `registered_weights_sha256` and `prepare_tasks`, refuse an
  entry that no longer hashes to it, and `prepare_tasks` pins the line of the entry it read, so
  an entry changed after the preparation is refused when the batch runs. The fan store does not
  record the line (its `META_KEYS` stay, design (c)); the ledger's fit entry and the model card,
  built later, must record it and name their checkpoints with it.
  (d) [F5] `activation_checkpointing` is a registry payload key (false for identity entries): 33
  keys, the schema still 1 because no registry existed yet. The wrapper set is not recorded:
  after (a) it is always all on.
  (e) [F2] A gapped export with the `gap_aware` wrapper off is refused by `RunRequest` (before a
  run directory exists) and by the child, and the child compares its datasets' window counts
  with the export's whichever dataset class runs (design 1.6).
  (f) [F3, F6] Tests the model side's implementer did not write (it stopped early; its T6 and
  T23 tests cover `forecast_window` and the partition, in `tests/test_decision.py` and
  `tests/test_partition.py`). Added: T21 (`tests/test_finetune_registry.py`, 37 tests), the
  batch-layer parts of T6 (`tests/test_model_period.py`) and of T23
  (`tests/test_evaluation_length.py`; a `weights` case stores a 24-step batch under its own
  `config_id`).
  (g) [F4] `docs/RETRAINING.md` 1.8: the validation phase's BingX targets end at
  2026-07-01T21:00Z for the 24-step fans of the last `G25` decision and at 2026-06-30T23:00Z for
  the 4-step primary fans; no read reaches 2026-07-02, as before.
  (8) T22, the identity control (design 3.6), is two `weights` tests instead of the planned
  `tests/test_finetune_identity.py`: in `tests/test_finetune_timing.py`,
  `test_the_identity_control_runs_through_the_timing_path_with_the_real_weights` (M9 itself: the
  shipped weights through `save_pretrained`, a scratch registry, `load_models(checkpoint=...)`
  and a spawn-started worker; the fan identical, the configurations distinct); in
  `tests/test_finetune_registry.py`,
  `test_the_childs_output_path_registers_and_loads_the_shipped_weights` (the shipped state saved
  by the child's own code is byte-identical to the shipped `model.safetensors`, passes the real
  `checkpoint_dir_problems`, registers, loads by id and forecasts bit for bit as the shipped
  model under its own `config_id`). The real M9 passed (D-053).
- D-053 (2026-11-02; (3) and (9) corrected 2026-11-02 by D-054) First run of the measurement
  step (`kronos-agent finetune timing`, design section (a), phase P0; `reports/finetune/timing.md`
  and `.json`). It ran once, from 2026-11-02T20:44:30Z to 21:01:36Z, on darwin-arm64 (10 CPUs, 16
  GiB, torch 2.14.1), on synthetic candles only: no market data, 0 tests. Status completed, not
  stopped: it spent 2,049 CPU-seconds and 1.8 GPU-minutes against its estimate of 2,620 and 25.0
  and its limits of 3,000 and 45, below gate 3. The record is tied to the training code sha256
  `fb82f2ec48fe06cdee817965473051b2fff4061f83c7640c20617fee96d14be5` and the forecasting code
  sha256 `37c3bf99e48a6aee17a804432e4ae40726310e9d82aef40adbdc4f8d66e593a1`. Its choice: the CPU
  at 2 threads, batch 32 with activation checkpointing, deterministic, 9.68 s per step; MPS not
  usable because its preflight did not pass.
  (1) Training step on the CPU (M1: a fresh child per configuration, 3 untimed and 10 timed
  steps; medians; the peak is the child's peak RSS). At 2 threads and batch 8, 16 and 32: 1.53,
  3.16 and 10.36 s per step (5.23, 5.07 and 3.09 windows per second), peaks 3.77, 6.80 and 9.23
  GiB, swap growth 0, 0 and 1.52 GiB. At 8 threads: 1.72, 3.43 and 10.61 s (4.64, 4.66 and 3.02
  windows per second), peaks 3.77, 6.80 and 9.16 GiB, swap growth 0, 0 and 1.34 GiB. The peak
  grows by 0.217 GiB per window at 2 threads (0.214 at 8), the slope design (a) reads as the
  attention probabilities being materialised. At batch 32, 8 threads run 0.98 times as fast as
  2 (1.3 needed), so training uses 2 threads.
  (2) Activation checkpointing (M2, the CPU at 2 threads, batch 32; run because the batch-32 peak
  passed 9 GiB and swap grew): 9.68 s per step (3.31 windows per second), peak 3.16 GiB, no swap
  growth, and the gradients of one step bit-identical with and without the wrap (127 tensors).
  So batch 32 does not fit as is and runs with activation checkpointing, which design 3.5 puts
  before any batch change.
  (3) MPS preflight (M3, before any MPS timing): at batch 32, without activation checkpointing
  and with deterministic mode off, the child ran out of MPS memory during its checks: "MPS
  backend out of memory (MPS allocated: 12.17 GiB, other allocations: 1.72 MiB, max allowed:
  11.84 GiB)", at the watermark ratio 1.0, with memory pressure at level 2 (warn) and 4.59 GiB of
  swap growth. At least its first check (`multinomial`) had ended: the child samples the MPS
  driver memory only after a check ends, and it recorded a peak of 8.20 GiB. Which check ran out
  of memory is unknown, because the child records the checks' results only once all of them
  have returned, so the record holds none (`checks` is empty). Under the device rule MPS was then
  not timed (M1 on MPS, M4 and M7 did not run): there is no MPS step time and no thermal factor,
  and for the CPU the record takes `theta` from the pilot's first pseudo-epoch. MPS was never
  tried in the configuration that fits: on the CPU, checkpointing cut the batch-32 peak from 9.23
  to 3.16 GiB. "MPS not usable" is therefore not a measured result.
  (4) Fan speed (M5, 2 threads, one warm-up and 3 timed fans): t(4) 3.23 s, t(8) 6.06 s, t(24)
  19.0 s, the fit t(L) = -0.107 + 0.795 L s, against the prior 0.44 + 0.678 L (3.15, 5.86 and
  16.7 s).
  (5) Worker scaling (M6, 12 four-step fans; the clock starts once every worker has loaded its
  model and run a warm-up fan, which took 5.3 s with one worker and 14.6 s with four): 0.232,
  0.290, 0.325 and 0.332 fans per second with 1, 2, 3 and 4 workers, so four workers give 1.43
  times the throughput of one, where design 8.2 gave wall times "if they scale"; one worker
  takes 4.31 s per fan against t(4) = 3.23 s in the parent. Contention with MPS training (M7)
  did not run.
  (6) NLL4-TF (M8, the CPU at 2 threads, 48 synthetic decisions): 0.158 s per decision, 53 s per
  fold evaluation of 336 decisions (prior: about 50 s).
  (7) The identity control (M9) passed: the shipped weights saved, registered in a scratch
  registry (both files byte-identical to the shipped ones, with the sha256 of
  `manifests/weights.json`), loaded with `load_models(checkpoint=...)` and run in a
  spawn-started worker gave a fan identical to the shipped model's, under the configurations
  `62906d91f853a8e5` (shipped) and `d49ec38a515fe546` (checkpoint). The fine-tune output path
  cannot by itself create a difference between the two models.
  (8) Spend against the priors, in CPU-seconds: M1 769 of 1,350, M2 205 of 200, M3 37 of 150 (and
  1.8 of 2 GPU-minutes), M5 225 of 210, M6 779 of 300 (charged at its declared wall x workers x
  2 threads, D-052 (6)), M8 15 of 20, M9 18 of 20; M4 and M7 none.
  (9) The measurement is repeated before any gate-3 estimate, with the fix of D-054: no estimate
  of the pilot, the walk-forward or the evaluation fans is computed from this record, its device
  choice included. In the second run M3, the MPS batch 32 of M1 and M4 start with activation
  checkpointing, because the CPU's batch 32 does not fit as is here and M2 passed (2); the one
  retry with it covers only an MPS unit that ends `memory_abort` without it (here the MPS batches
  8 and 16 of M1). Its plan differs from the one this run recorded, so it is a new study with a
  new stem (D-034): it writes `reports/finetune/timing-2.*`, and this run's three files,
  `reports/finetune/timing.*`, stay unchanged as this decision's source (D-054 (9)). It waits
  for the owner's decision on its budget (D-054 (8)).
- D-054 (2026-11-02; (8) settled 2026-11-02 by D-055) The measurement step after its first run:
  the fix of `finetune/timing.py` and `finetune/runner.py` that its second run executes, that
  run's budget, and how its record is stored. It amends D-052 (6) and design sections (a), 3.4
  and 3.5, and corrects D-053 (3) and (9). An independent check of the fix found four problems,
  all confirmed: how the second run is stored (9), the fix recorded nowhere and the texts it made
  stale (this decision, D-053 (9)), the wording of D-053 (3), and the budget (7); the code ones
  are fixed with tests that fail without the fix (each fix reverted in memory).
  (1) Order (amends D-052 (6)): the synthetic export and the shipped models, M9, M8, M5, the
  models released, M1 on the CPU (2 and 8 threads, batch 8, 16, 32), M2 on the chosen CPU
  configuration when its batch 32 does not fit as is (the timing with the wrap, then the CPU
  gradient bit-identity), M3, M1 on MPS, M2 on MPS only when its batch 32 completed as is and
  does not fit, M4 with M7 during it, M6. Whether batch 32 needs activation checkpointing is thus
  known from the CPU before any MPS unit.
  (2) Activation checkpointing on MPS (design 3.5 puts it before any batch change): when the
  chosen CPU configuration's batch 32 does not fit as is and M2 passed there, M3, the MPS batch 32
  of M1 and M4 run with it from their first attempt; on this Mac they will if the CPU measures as
  in D-053 (2). The checks of M3 that train (parity, steps, deterministic) build wrapped models on
  MPS and on the CPU (`runner._training_model`): on the CPU the wrap changes no value, and the CPU
  side no longer holds unwrapped batch-32 activations beside the MPS allocations. `multinomial`
  and `checkpoint_gradients` are unchanged.
  (3) One retry: an MPS unit (M3, an M1 batch, M4) that ends `memory_abort` without activation
  checkpointing is retried once with it, only once M2's CPU gradient bit-identity has passed (run
  first when it has not, charged to M2) and when the retry's prior fits the budget; the MPS units
  after it, from that batch up, run with it. On this Mac only the MPS batches 8 and 16 of M1,
  which start without it, can be retried. Both attempts are recorded: the unit's entry holds the
  retry and `first_attempt` the aborted attempt (a retried M4 also keeps the M7 that ran during
  it, and M7 runs again); a unit that is not retried says why under `retry`. A retry is charged
  to the spend item `retries`.
  (4) Device comparison (amends design 3.4 and (a)): MPS's sustained step time (M4, with
  activation checkpointing when it ran with it, which then needs the gradient check) against the
  fastest CPU configuration that runs batch 32, as is where it fits and with activation
  checkpointing where M2 passed (the times as is when none does): `choice.best_cpu`. On the first
  run's record the old and the new rule give the same choice: the CPU at 2 threads with
  activation checkpointing, 9.679 s per step.
  (5) A torch 2.14.1 fact: with `preserve_rng_state`, `torch.utils.checkpoint` restores only the
  CPU generator for the recomputation on MPS (`torch.mps` has no `_initialized`), so the
  recomputed dropout masks differ from the forward's (a tiny model with 4 blocks over 4 steps:
  gradients 0.037 apart). On MPS `enable_activation_checkpointing` also passes
  `context_fn=runner._mps_generator_contexts`: the recomputation replays the forward's MPS
  generator state and the generator is put back (the same model: equal losses, gradients within
  4.1e-8, an equal generator). M2 checks only the CPU, so on MPS this rests on
  `test_checkpointing_reproduces_the_unwrapped_training_steps[mps]` alone.
  (6) Record schema 2: M2 is `{gradient_check, runs}`; `mps_checkpointing` says from which batch
  MPS units ran with the wrap, and why; `choice.best_cpu` and
  `choice.mps_sustained_seconds_per_step`; the spend item `retries`. `load_timing` and
  `require_fresh_timing` refuse schema 1, the first run's.
  (7) Budget. M3's prior of 150 CPU-seconds and 2 GPU-minutes, which the retries' prior copied,
  left out its CPU side: check (c) runs its 20 training steps at batch 32 on the CPU too, about
  387 CPU-seconds at the first run's 9.68 s per step and 2 threads. M3 and the retries are now
  priced by their content: 21 steps on the CPU at 9.68 s (check (c)'s and check (b)'s forward and
  backward) and 23 on MPS at the design's 3.2 s (checks (a) to (d)), CPU-seconds at wall x 2
  threads (D-033), the child's whole wall as GPU time: about 554 CPU-seconds and 4.6 GPU-minutes
  each (`timing.PREFLIGHT_PRIOR`). An M4 child lives about 20.5 minutes (its start-up, warm-up
  and last step beside its 20 timed minutes), so a retry admitted on M4's 20-minute prior with
  20.0 to 20.5 minutes left could only end at the 45-minute stop. M4 is now priced at its whole
  lifetime, 21 GPU-minutes (`M4_OVERHEAD_SECONDS` = 60), and starts, or is retried, only when
  that fits. The estimate becomes about 3,578 CPU-seconds and 33.2 GPU-minutes with MPS (2,050
  and 0 without), up from 2,770 and 27.0: above the step's 3,000 CPU-seconds, so
  `finetune timing` refuses to start (exit 2, nothing runs) until the owner decides (8).
  (8) Open, for the owner, before the second run. A projection from the first run's measured
  costs (D-053 (8)): loads, M9, M8, M5, M1 and M2 on the CPU and M6 as measured (2,013
  CPU-seconds), M7 at 1.25 times M6's one- and two-worker runs (336, an assumed slowdown), M3's
  CPU side at its content (407), and the MPS children (M3's MPS side, M1 on MPS, M4) at a CPU
  rate never measured: about 2,900 CPU-seconds at 0.1 CPU-seconds per second, 3,190 at 0.3 and
  4,190 at 1, plus about 410 to 490 for one retry of M3's size, and about 27 GPU-minutes. Under
  the 3,000 limit the run would complete only if the MPS children stayed near 0.1 and nothing
  were retried; otherwise it would stop in M4 or M6. The options: a limit still below gate 3 and
  not below the estimate (3,578 to 3,599 CPU-seconds), with a run that may still stop; a change
  of the plan (for example not measuring again what the first run measured with code this fix
  did not change: M5, M6, M8 and the CPU at 8 threads, about 1,530 CPU-seconds in the first run);
  or a run above gate 3, with the owner's approval of its estimate.
  (9) Storage of the second run. Its plan differs from the one the first run recorded
  (`reports/finetune/timing.config.json`, sha256 `7408a6f6...`), so `record_config` refused it
  (`ConfigChangedError`, exit 2) before any unit started: the next run could not rewrite
  `reports/finetune/timing.*`, as D-053 (9), the README and CLAUDE.md said. As D-034 requires, a
  changed study is a new study with a new stem: `TIMING_STEM` is `timing-2`
  (`reports/finetune/timing-2.json`, `.md` and `.config.json`; `load_timing` reads it), and the
  first run's three files stay unchanged as D-053's source. Moving them aside instead would need
  the owner's explicit approval and is not done. A plan changed after `timing-2.config.json` is
  recorded takes a new stem the same way.
  (10) Code and tests. The training-code sha256 is now
  `f28c0f30d217c24c914fe6b8f8c75aed27aa44115198b4a1821122706a5216d8` (`runner.py`; the first run
  had `fb82f2ec...`); the forecasting code is unchanged (`37c3bf99...`); `timing.py` is in neither
  hash, so (7) and (9) change only the plan. The offline suite went from 1,907 to 1,915 passing
  tests (15 network tests deselected; the 4 xpasses of D-007).
  Not done, for the owner to decide: `runner._run_preflight` could keep the checks that ended and
  name the one in progress when a check runs out of memory, so that an aborted attempt (a retry's
  `first_attempt`) would show which check it was (D-053 (3)).
- D-055 (2026-11-02; the opening, (2) to (5) corrected 2026-11-02 after the check of (7))
  The owner's gate-3 approval of the second run of the measurement step, and `finetune timing
  --approved-by-owner CPU_HOURS`. The owner was shown that the first run used 0.57 CPU-hours
  (2,049 CPU-seconds, D-053) and that the full second run (its estimate about 3,578 CPU-seconds
  and 33.2 GPU-minutes, above the step's own 3,000 CPU-seconds: D-054 (7)) needs up to about 1.3
  CPU-hours (D-054 (8): 4,190 CPU-seconds with the MPS children at 1 CPU-second per second, plus
  490 for one retry of M3's size, 4,680) and about 35 GPU-minutes (the estimate's 33.2, rounded
  up; the projection of D-054 (8) gives about 27, about 32 with one retry of M3's 4.6). The
  owner approved up to 1.5 CPU-hours for it on 2026-11-02 (their words, translated: "approve up
  to 1.5 h of CPU"). This settles D-054 (8) by its third option, a run above gate 3 with the
  owner's approval; the plan is unchanged. The approval is gate 3 as a ceiling and covers this
  measurement only, the run that writes `reports/finetune/timing-2.*`: not a rerun of it, the
  pilot, the walk-forward or any fan batch. That run passes `--approved-by-owner 1.5`.
  (1) The flag is the batch commands' of D-046 (a): a finite number > 0; a bare flag or any other
  value is a usage error (exit 2) before anything runs. The time it was given is the command's
  start (D-033).
  (2) With it the run's CPU limit is CPU_HOURS x 3600 CPU-seconds (5,400 for 1.5) instead of the
  step's own 3,000 (`timing.TIMING_MAX_CPU_SECONDS`), a ceiling the run's spend is kept within.
  The run works against its stop limit, the ceiling minus a reserve of 600 CPU-seconds
  (`timing.APPROVAL_RESERVE_CPU_SECONDS`; 4,800 for 1.5, above the estimate of 3,578 and the
  projection of 4,680): the estimate printed first shows both and is refused above the stop
  limit (exit 2, nothing runs), a unit or retry whose prior would pass it is not started, each
  child's deadline and the watch of M4 follow it, and the run stops once its measured spend
  passes it. The reserve covers the most one admitted unit can add past the stop limit: M6's
  four-worker configuration, admitted on its prior of 75 CPU-seconds, was charged 289 in the
  first run (214 more); M7, at 1.25 times M6's two-worker 165 while the M4 child, which is not
  polled during M7, runs at up to 2 CPU-seconds per second, about 270 (M7 admits each
  configuration, with or without the flag, on the M4 child's CPU read again at its start, no
  longer on the one read when M7 began); a training child one 10.6 s step past its deadline at 8
  threads, 85. M5, M6, M7, M8 and M9 have no deadline of their own, so for them the reserve,
  sized on the first run's measured costs, is the bound. The step's own 3,000 CPU-seconds keep
  the same margin below gate 3, so without the flag nothing changes (3). Unlike the batch
  commands, whose figure is compared with a measured estimate up to 1.25 times it (D-040), the
  figure here is a ceiling without tolerance: the run is not to spend more than the owner
  approved. The GPU limit stays 45 GPU-minutes, stopped once passed (at most one 5-second poll of
  M4 past it, below gate 3's 60). `timing.TimingBudget` refuses a CPU limit above gate 3 without
  an approved figure, and any GPU limit above gate 3.
  (3) Without the flag nothing changes: the step's own 3,000 CPU-seconds and 45 GPU-minutes,
  below gate 3, with the same estimate line, stops, record values and Markdown; the estimate with
  MPS is still refused. The one exception is M7's fresh read of the M4 child's CPU (2), which
  applies to every run.
  (4) The budget left the plan. `timing_plan()`, the definition recorded as
  `timing-2.config.json`, held the limits (`limits`: 3,000 CPU-seconds, 45 GPU-minutes and the
  gate-3 figures), so a run with another budget would have been refused as a changed study
  (D-034) and needed a new stem. The plan is what is measured and the budget how far a run may
  go, so `limits` was removed (`rules.budget` says where a run's limits are recorded): the plan is
  the same for every budget, and a run without the flag and one with it verify the same
  `timing-2.config.json` (a test). No `timing-2.config.json` existed when `limits` was removed
  (every earlier attempt was refused before recording it, D-054 (7)), so the stem stays
  `timing-2`. The record holds the run's limits and gate 3 under `spend`, and the new `approval`
  (null without the flag): `approved_cpu_hours`, `given_at` (the command's start),
  `limit_cpu_seconds` (the ceiling) with `reserve_cpu_seconds` and `stop_cpu_seconds` (2),
  against `own_limit_cpu_seconds` and `gate3_cpu_seconds`, `above_gate3` (its CPU limit passes
  gate 3: a run above the gate-3 limit under the owner's approval) and the estimate shown. The
  Markdown's budget section states the approval and whether the spend stayed within the ceiling
  and passed gate 3; the command prints the approval before anything runs and again in its summary.
  No schema-2 record had been written, so the schema stays 2 with `approval` among its required
  keys. `timing.py` and `cli/finetune.py` enter neither code hash: the training code stays
  `f28c0f30...` and the forecasting code `37c3bf99...`.
  (5) Tests (mocked: stand-ins for every measurement, temporary directories): a bare or invalid
  flag; an estimate between 3,000 and the stop limit runs only with the flag; an estimate above
  the stop limit (also one within the ceiling), or above the 45 GPU-minutes, is refused before
  anything runs; the spend, a child's deadline and the M4 watch stop at the stop limit, and the
  spend stays within the ceiling when a child stopped by its deadline ends one step late, when M6
  costs what it cost in the first run from any spend up to the stop limit, and when M4's CPU
  grows while M7 runs; the record, the printed lines and the Markdown state the approval and compare
  the spend with the ceiling; without the flag nothing changes; a budget change is not a changed
  study. The offline suite went from 1,915 to 1,958 passing tests with this change, and to 1,962
  with the fixes of (7) (15 network tests deselected; the 4 xpasses of D-007).
  (6) An incident while this change was tested. A test run started before
  `tests/test_finetune_timing.py` was updated still held the old usage-error case `finetune
  timing --approved-by-owner 1`, which the new parser accepts, and that test called the command
  with the repository's roots: the real measurement step started at 2026-11-02T23:38Z, a run no
  one had asked for, under a CPU limit of 3,600 CPU-seconds (1 CPU-hour, not above gate 3). It
  was stopped after about 17 minutes, during the MPS batch 16 of M1. Its 11 training children
  (M1 on the CPU at 2 and 8 threads, batch 8, 16 and 32; M2 on the CPU with its gradient check;
  M3 with activation checkpointing; the MPS batch 8, and the batch 16 it was stopped in) spent
  about 1,210 CPU-seconds and 6.3 GPU-minutes, besides M9, M8 and M5 in the parent (about 260
  CPU-seconds in the first run). It wrote no `timing-2.json` or `.md`, nothing under `data/`, no
  registry and no bytecode under `third_party/`. It left `reports/finetune/timing-2.config.json`
  (the plan of (4), sha256 `2128b4bb...`: the bytes the approved run records) and, under
  `weights/finetuned/`, the scratch `.scratch/timing-20261102T233832Z` (197 MB) and 11 run
  directories `.runs/synthetic-timing-20261102T234035Z` to `...T235530Z`. They are not a record:
  no estimate or choice is taken from them. Removing them is the owner's decision and was not
  done. As they stand, the approved run verifies the recorded plan; a plan change before it
  would be refused as a changed study while that file exists. The usage-error tests now give
  only invalid figures and run on the stand-ins in temporary directories, so a figure that
  became valid could not start the real step again; `tests/test_cli.py` only parses.
  (7) An independent check of this change found three problems, all confirmed and fixed. (a)
  With the flag the spend control stopped the run only once its measured spend had passed the
  ceiling, and admitted M6 and M7 on priors below their cost, so the run could spend more than
  the owner approved: stand-in runs under `--approved-by-owner 1.5` with M6 charged its first-run
  costs ended at 5,479 and 5,614 CPU-seconds, one with M7 admitted on the M4 child's CPU read
  when M7 began at 5,581, and two tests asserted a spend above 5,400. Fixed by the stop limit and
  the fresh read of (2), with the tests of (5); each of the three code changes (the reserve, the
  fresh read, the estimate checked against the stop limit) makes tests fail when reverted in
  memory, and the same stand-in runs now stay within 5,400. (b) The opening attributed the 35
  GPU-minutes to D-054 (8), which gives about 27, and left out the first run's 0.57 CPU-hours:
  corrected above. (c) `docs/RETRAINING.md` (a) and (c) still said the step stays below gate 3
  and waits for the owner, and `docs/ARCHITECTURE.md` gave the command without the flag: both
  now name this approval. `timing_plan()` is unchanged, so the recorded `timing-2.config.json`
  (sha256 `2128b4bb...`) still verifies; `timing.py` and `cli/finetune.py` stay outside both code
  hashes (training `f28c0f30...`, forecasting `37c3bf99...`).
- D-056 (2026-11-03) Gate-3 approval of the pilot; the walk-forward layout; the clean-up of the
  measurement leftovers. The owner was shown the second measurement (`reports/finetune/
  timing-2.md`: MPS chosen at 5.449 s per step at batch 32 with activation checkpointing, theta
  1.004; the best CPU configuration 8.790 s) and the per-fold estimate: at most 4.9 h of MPS at
  the 3,072-step cap, 9.7 h if the pre-registered pilot rule doubles the cap once, then a pilot
  check of about 0.9 CPU-hours. On 2026-11-03 the owner approved "up to 10 h" for the pilot
  (fold 1, training-segment data only) and its check. The approval covers that run only; a
  measured estimate more than 1.25 times the approved figure goes back to the owner (D-040,
  D-046). By the rule of D-048 (1) the measured worst case of the 24 monthly folds (about 116 h)
  is above 72 h, so the walk-forward uses the 8 quarterly folds over the same test decisions
  (about 39 h at the cap, plus up to 29 h for the seed replicates of D-048 (4)); its gate-3 ask
  comes after the pilot. The owner also approved deleting the synthetic leftovers of the three
  measurement runs (`weights/finetuned/.scratch` and `.runs`, about 590 MB, the D-055 (6)
  incident's included): they were deleted; the reports under `reports/finetune/` are kept.
- D-057 (2026-11-03) Phase 3a of the retraining: the test suite's root guard, the experiment
  ledger and the test budget, the tokenizer diagnostic, the pilot check and the fold command that
  trains the pilot; their deviations, the review fixes and what stays open. It is the part of
  item 3 of the order of work of `docs/RETRAINING.md` (c) that the pilot needs. The root guard
  came first, as the follow-up of the D-055 (6) incident; three implementers (governance,
  diagnostics, fold command) then built the rest in parallel from an interface specification (a
  scratch document, not kept), a gate ran the offline suite, every new `--help` in a fresh
  interpreter and ruff, two reviewers (the governance code; the pilot path) replayed it in
  scratch, and a fix-up applied their findings (8). Nothing real ran: no training, export,
  timing, fan batch, tokenizer pass or network access. Nothing was written under `data/`,
  `weights/` or `reports/` (`weights/finetuned/` is empty, no `reports/ledger/` exists), so the
  experiment ledger has no entry, no test is spent, no event definition is registered and
  `params.yaml[criterion]` is not frozen (D-008). No training-code or forecasting-code file
  changed: the training code stays `f28c0f30...`, so `reports/finetune/timing-2.json` stays
  fresh, and the forecasting code `37c3bf99...`, so no forecast `config_id` changed. The offline
  suite went from 1,962 passing tests to 1,987 with the guard and to 2,208 with the rest (3
  skipped, (6); 15 network tests deselected; the 4 xpasses of D-007).
  (1) The root guard (`tests/root_guard.py`, registered by `tests/conftest.py`). Every test runs
  on its own roots: the autouse fixture `isolated_roots` (its own monkeypatch, so that a test's
  `monkeypatch.undo()` keeps it) points `KRONOS_AGENT_DATA_DIR`, `KRONOS_AGENT_WEIGHTS_DIR` and
  the new `KRONOS_AGENT_REPORTS_DIR` (`paths.reports_dir()` had no override; every root the code
  resolves goes through `paths`) at a new temporary directory, which the processes the test
  starts inherit; between tests they point at the session's own roots. These live in one
  `kronos-agent-test-roots-*` directory per session in the system's temporary directory, not
  under pytest's base, whose path names the login the environment report refuses; it is removed
  after a passing session and kept after a failing one. Opt-ins read a real root and never write
  it: the `weights` marker makes the weights root a view of the shipped weights (top-level
  symlinks; `finetuned/` stays the test's own), and the new marker `real_roots("data",
  "weights", "reports")` a view of each named root (`tests/test_manifests.py` needs it, or its
  weights check would check nothing). An audit hook refuses, before it happens, every write under
  a real root that the test process attempts through Python (opening for writing, creating,
  renaming, linking, removing, truncating, changing metadata, shutil copies and moves, tempfile;
  symlinks and `dir_fd` resolved), and since (8)(c) the native writers the code uses in-process
  (safetensors' `save_file` and `save_model`, `torch.save`) are wrapped to refuse the same. A
  refusal is a `RealRootWriteError` (a `PermissionError`) and fails its test at teardown, even
  when the code swallowed it. Path, kind, size and mtime of every entry under `data/`,
  `weights/` and `reports/` (no hashing; Finder's `.DS_Store` ignored) are listed at session
  start and end, and any difference fails the session (exit status 1): this also catches child
  processes and other native writes, after the fact. With `KRONOS_AGENT_TEST_ROOTS_LISTING` set
  to a file outside the real roots, the start listing is written there. Limits: `pytest
  --noconftest` bypasses the guard (registering it from `pyproject.toml` would close that; not
  done); `-p no:real-roots-guard` fails every test; a real command that writes under the real
  roots while a suite runs (the pilot, a monthly `data bingx`) fails that session's listing
  check, by design. Before the guard, a scratch diagnostic found that the suite wrote nothing
  under the real roots and that 26 tests read them: the `weights`-marked ones now read through
  the views, the others pass on the temporary roots.
  (2) The experiment ledger (`kronos_agent/ledger.py`, design 7.3). It lives at
  `<reports>/ledger/experiments.jsonl` (`paths.ledger_dir()`), not at the design's
  `ledger/experiments.jsonl`: under the reports root it is tracked and published with the
  reports (SPEC section 19), relocates with them, and the guard of (1) isolates and protects it.
  One canonical-JSON line per entry (LF-terminated, at most 65,536 bytes) with exactly `schema`
  (`experiment`), `schema_version` 1, `seq` (the line number), `ts` (UTC, whole seconds),
  `prev_sha256` (64 zeros for line 1), `kind`, `params_version`, `code`
  (`provenance.code_version()`, where `docs/ARCHITECTURE.md` named the commit alone) and
  `payload` (exactly its kind's keys). Six kinds: `exploration` (training-segment data or inputs
  only, 0 tests; validation or sealed outcomes and sealed inputs are refused), `fit` (one training
  child: the checkpoint and the sha256 of its registry line when registered, as D-052 (7)(c)
  asked, the status, attempt, blinding, device, spend, approval and order; never a loss, an NLL
  value or a selected pseudo-epoch), `gate3_approval` (the approved hours, the estimate shown,
  their ratio, at most 1.25, and the command's start), `preregistration` (a bundle file under the
  reports root and its sha256, once per recipe), `owner_order` (`order-NNNN` in sequence, scope
  `open_sealed`, the T-B candidate, the owner's words translated) and `test` (3). Against the
  design: `owner_answer` is deferred (no command writes it); an approval holds no owner's words
  (they stay in this file: D-056 for the pilot); there is no `verify_chain`, because
  `read_ledger` and `ledger_status` verify every line. An append holds the exclusive lock
  `<ledger dir>/.lock` (a held lock is a refusal), builds its payload from the entries read under
  it (a check and its append are one step) and replaces the file atomically with one more line;
  readers take no lock. Nothing is ever repaired or rewritten, and the last line is protected
  only by the `line_sha256` its users record, which the commands print after every append. Every
  line is checked again on read: its form and keys, the chain, references to earlier entries of
  the right kind, uniqueness (test ids, order ids, one bundle per recipe, one order per
  candidate), times that never decrease, the budget counts and the order of tests as the `test`
  lines record them, printable ASCII, finite numbers, no IPv4 literal, no absolute path (a URL
  included) and no BingX host name; only the bundle files and the writer's personal data are
  checked at append alone ((8)(e)). At load the ledger imports only light modules, because the
  sealed guard imports it; reading a non-empty ledger imports `exchange.public_market` for the
  BingX domain. The binding rule ((8)(a); `ledger.binding_problem`): only the repository's
  `reports/ledger` (compared by device and inode, so a reports root that links to it counts)
  authorizes reading the repository's `data/`, a data root inside it, or one that links into it
  within 3 levels (a view); any other ledger (a relocated reports root: a scratch run, a test)
  authorizes only a data root of its own. A copy of the data elsewhere is not recognised: this
  is mistake-proofing, not security.
  (3) The test budget (`kronos_agent/budget.py`; SPEC rule 8). `spend_test` appends a `test`
  entry before any statistic on reserved hours is computed (by the later `finetune report
  --test`), under the ledger lock and only if every rule passes: the ledger may read the data
  root (2); a known test and recipe; the recipe's bundle recorded, unchanged and registering the
  test; the test id never read; T-B after a T-A reading of the same recipe in an earlier ISO week
  when the bundle registers both; T-C after the recipe's T-B, on an `open_sealed` order for that
  T-B candidate that no other T-C reading names; at most 3 tests in the ISO week and 15 in the
  calendar quarter (UTC, every recipe). So "compute, peek, abandon" spends the test. No command
  spends one yet. A malformed argument is a `LedgerError` (exit 1), not a `ValueError`.
  (4) The sealed guard and `kronos-agent ledger`. `partition.sealed_access(order_id, *,
  ledger_root=None, data_root=None)` checks the binding of (2) first, then looks the order up
  with `ledger.verify_open_order` (scope `open_sealed`, in a ledger whose every line verifies);
  any problem is a `SealedPartitionError` naming the ledger and gate 6. Unlike design 7.3 ("the
  sealed commands mark it used"), an order is not refused after its T-C reading, because the
  same order covers the sealed folds, the sealed fans and the model to date (D-048 (3)): "once"
  is one order per candidate and one T-C reading per order, and the commands that use an order
  check that its candidate is their recipe's T-B test. `ledger show [--kind KIND]` lists the
  entries, the head and the tests of the current ISO week and quarter; `ledger verify` checks
  every line and every bundle (exit 1 on a broken or unreadable ledger, 2 on a changed or
  missing bundle); `ledger record-order --scope open_sealed --candidate TEST_ID --text
  "<words>"`, run only after the owner's explicit order in chat (gate 6), refuses (exit 2) a
  candidate without a recorded T-B reading, a second order for it, personal data in the text, a
  clock earlier than the last entry and a busy lock, and prints the order id, the seq and the
  line sha256.
  (5) The tokenizer diagnostic (`finetune/tokenizer_diag.py`; `finetune tokenizer-diag --part
  training|validation [--record-config]`; design section 4, 0 tests). Part A: the 1,534 `G25`
  windows of the Binance training segment (D from 2020-01-22T21:00Z to 2024-06-06T18:00Z), each
  with its next 49 candles, not 48: the template's W = 561 is 512 + 48 + 1 rows. Part B: 710
  windows per source (Binance and BingX validation inputs), only after a recorded bundle (the
  command refuses otherwise) and after Part A's report, written under its recorded definition.
  The 100 GBM reference windows have no decision hour (one block): point estimates only. The
  normalisations reduce on a Fortran-ordered float32 copy, bit-identical to the vendored dataset
  and to `nll.normalise`; `context_only` must change no token (a positive control; an error
  otherwise). The flag of D-048 (5), `r1_token_mismatch`, reads the share of context rows whose
  s1 or s2 token changes under R1's normalisation ((8)(f)): every row carries one token of each,
  so that share bounds the s1 share, the s2 share and the share of the 1,024 context tokens, and
  the owner is told whichever reading of "more than 5 % of context tokens changing" passes 5 %;
  the s1 and s2 shares are reported beside it. Flags read point estimates of the real sources,
  never the reference: the volume-and-amount saturation flag reads the share of context entries,
  the `rho` flag the full codes' `rho`, and Part B's perplexity flag each validation source's s1
  perplexity against Part A's, recomputed from its counts. Comparators, codebook figures (at full
  size and at matched row counts) and per-bit shares are descriptive; a return whose
  reconstructed close is not positive is dropped and counted. The estimate times one batch and
  the pooled bootstrap and is refused above gate 3 (there is no approval path). Reports:
  `reports/finetune/tokenizer-training.*` and `tokenizer-validation.*`, two figures each. A
  read-only probe of the shipped tokenizer on 32 synthetic windows found no `context_only`
  change, `z_pre` and the decode identical to the forward pass, and 1.29 s per batch of 32 Part A
  windows: Part A should take about 65 s of encoding plus about a minute of bootstrap.
  (6) The pilot check (`finetune/pilot_check.py`; `finetune pilot-check [--record-config]
  [--estimate] [--approved-by-owner CPU_HOURS]`; design 6.2 steps 3-4, 0 tests). On the 168 `G8`
  hours of the pilot's inner validation (2024-04-14T16:00Z to 2024-06-09T08:00Z): three batches of
  Binance fans under the opt-in role `pilot-check` (the pilot and shipped on `forecast`, shipped
  on `forecast-aa`), `c_ref` checked bit for bit against every fan. Point estimates only (2
  four-week blocks): `A_same`, `A_floor`, the mean |Delta p_up|, the first-step s1 KL both ways
  from teacher-forced logits (its maximum: the largest per-decision KL(pilot || shipped)), the
  relative weight change, the SD and lag-1..6 autocorrelation of d(D) (fair CRPS at h = 4; a fan
  with a close <= 0 drops its decision from the CRPS statistics only), an approximate minimum
  detectable difference for 4,380 decisions (this build's choice: n_eff = 4,380 / 2 / max(1, 1 +
  2 sum rho_k), MDD = 2.8016 x SD / sqrt(n_eff)) and the NLL4-TF curve from the fold record. The
  reading is NO-GO when `A_same >= 0.98`, otherwise GO; both exit 0. Beyond the interface it
  checks that the registry entry describes this pilot and that its fold record shows a completed
  run with a selected pseudo-epoch of at least 1. Estimate: one pilot fan timed after a warm-up,
  plus a KL prior of 0.16 s per model and decision: about 504 x 3.06 x 2 + 108 = 3,195
  CPU-seconds (0.89 CPU-hours) at timing-2's t(4), about 3,530 (0.98) at M6's one-worker 3.40 s
  per fan; below gate 3 it needs no figure. Spend control of design 3.5 ((8)(d)): the run is
  charged the larger of its CPU time and wall x threads (D-033); checked after every fan, before
  each batch and before each teacher-forced pass, it stops once that passes 1.25 times the
  approved figure, or 3,600 CPU-seconds without one (`PilotCheckStopped`, exit 2; the stored fans
  are kept and a rerun reuses them). Missing: the fair CRPS of design 5.1, `eval/crps.py`
  (`fair_crps`), was assigned to this side but not built. `pilot_check` imports it at its CRPS
  step, after the fans, so `finetune pilot-check` cannot complete until it exists (it would end
  in a `ModuleNotFoundError` with its 504 fans stored), and the 3 tests that compute the CRPS
  skip; they passed against a scratch stand-in.
  (7) The fold command (`finetune/walkforward.py`, the per-fold path of design (c); `finetune fold
  --fold LABEL [--replicate K] [--estimate] [--approved-by-owner HOURS] [--order ID]`). The
  layout is quarterly (D-056): `wf-` and `final-` labels are refused (exit 2) and quarterly
  replicates do not exist yet (exit 1; (9)(d)). The pilot is `q-2024-07`, whose CSV, `[2020-01-01,
  2024-06-09T16:00Z)`, is `wf-2024-07`'s: training-segment data only (asserted), shown, no bundle
  needed. Every other fold needs its recipe's bundle in the ledger (refused today: no command
  records one) and is blinded: the command prints `FitOutcome.printable()`, the checkpoint id and
  12 hex of its weights' sha256, the ledger holds no selection value, and after a successful run
  the fold record is never opened (its sha256 comes from the registry entry). The estimate
  (design 8.1 on `timing-2.json`, refused when stale) is theta x steps x t_step plus (E_max + 1)
  x (t_nll + t_val) plus t_setup, with t_nll from M8 (52.2 s for 336 decisions) and two priors
  the measurement step did not measure, t_val (a third of a training step per batch of the 832
  validation windows, 47.2 s) and t_setup (300 s): 17,801.8 s = 4.945 MPS-hours at E_max 6 and
  35,204.3 s = 9.779 at E_max 12, its CPU side (about 0.57 and 1.03 CPU-hours) recorded, not
  gated. D-056's 4.9 and 9.7 h (and its 116, 39 and 29 h) leave t_setup out. Gate 3 counts the
  chosen device's hours (MPS wall-hours; CPU-hours on the CPU): above one hour the first unit
  needs `--approved-by-owner HOURS`, and an estimate more than 1.25 times it is refused before
  anything is written. The approval is recorded before the first unit, and the invocation's spend cap is
  1.25 times the figure (one hour without one): before every child the spend so far plus the
  unit's estimate must fit it, and the child's deadline is the smaller of twice its estimate
  (design 3.5) and what the cap leaves. The pilot rule is recorded in the pilot's `planned` entry
  before its result exists; when it fires, the doubled recipe (E_max 12, still named R1 but a
  recipe of its own, `56e3d2f9`, with its own export, seeds and checkpoint id
  `ks-btc-56e3d2f9-q-2024-07`) refits fold 1 from the shipped weights in the same invocation if
  the cap admits it; otherwise the command refuses the refit after recording the pilot, and the
  next invocation starts there. The stages (train, bookkeeping, refit, done) are read from the
  registry and the ledger: a registered fold other than the pilot is never trained again, and a
  run killed between its registration and its ledger entries gets them without compute ((8)(g)).
  Failures (design 6.3): `train.py` is a training-code file and is used unchanged, so a failed
  child's status is read from the run directory the call created; `failed` and `no_result` are
  retried once with the same seeds when the cap admits it; a second failure, a memory abort, a
  time abort or a killed child is `FoldFailed` (exit 1) after its `fit` entries; an interrupt
  records the run before it propagates; a run whose record says completed but that could not be
  registered is recorded as `failed`. Also refused: a timing record of another batch or without
  a way to run batch 32, a running timing step, and a stage that moved since the estimate was
  printed (re-read under the fold lock `<fine-tune root>/.scratch/fold.lock`); a runtime thread
  count other than `params.yaml`'s is an error.
  (8) Review. The two reviewers reported 8 findings [F1-F8], all confirmed. Seven were fixed with
  tests that fail without the fix (each fix reverted in scratch); [F4] is a design conflict left
  to the owner, (9)(a).
  (a) [F1, high] An order recorded in another ledger opened the repository's data: with
  `KRONOS_AGENT_REPORTS_DIR` pointing at a scratch reports root and the default data root,
  `sealed_access` honoured the scratch ledger's order and `require_study_range` passed over the
  first sealed day (no candle was read). Fixed by the binding rule of (2), applied in
  `partition.sealed_access` (before the order lookup, so that the tests' one-argument
  `_verify_order` stand-ins cannot skip it), in `budget.spend_test` and in the fold command, at
  plan time (`fold --fold s-2026-09 --order ID --estimate` is refused in that setup) and at the
  run's sealed access.
  (b) [F2] The budget limits and the order of tests were checked at append only, and a reading
  dated before the last entry (a backdated `now`, a clock set back) could escape a full ISO week.
  A line's `ts` may no longer precede the previous line's, and both rules are also checked on
  read; `ledger record-order` takes the command's clock. Consequence: after a clock that ran
  ahead is set back, appends are refused until real time passes the last entry.
  (c) [F3] safetensors' `save_file` and `torch.save` write natively without an audit event, so the
  guard of (1) saw such a write under a real root only in the end-of-session listing; they are
  wrapped for the session now. Still seen only by the listing: other native writers, a writer
  bound before the session started, and child processes.
  (d) [F5] The pilot check had no spend stop (design 3.5): it has the one of (6).
  (e) [F6] The personal-data check ran on read with the reader's login, host name and home, so a
  ledger written on one machine was a broken chain on a machine whose login is a substring of a
  line ("dev" in the key "device"). It runs at append only now, for the writer's own data, and
  reading is machine-independent; a machine whose login is a substring of a fixed key still
  cannot append such a line (the `envinfo` rule, unchanged).
  (f) [F7] The D-048 (5) flag read the s1 share only, while the owner's words are "context
  tokens"; it reads the share of (5). No tokenizer definition had been recorded, so no recorded
  definition changed.
  (g) [F8] A bookkeeping `fit` entry named the current invocation's approval and attempt 1; it
  names the approval and attempt of the run it records now, read from the entries before it.
  (9) Open, for the owner:
  (a) [F4] Blinding (design 6.3). A blinded fold's wall time is printed (design 6.3 prints it) and
  recorded as spend in its `fit` entry, and its run id, the ledger's times, the run directories
  and the operator's clock show the same duration. The duration gives the number of
  pseudo-epochs the fold ran, hence its selected pseudo-epoch when it stopped early (under
  patience 2, the last pseudo-epoch run minus 2). Hiding the wall time would hide nothing and
  lose the spend record of design 3.5, so the code was not changed. Proposed amendment of design
  6.3, which lists the selected pseudo-epoch as blinded: the number of pseudo-epochs a fold runs,
  hence its selected pseudo-epoch when it stops early, follows from its duration and is not
  blinded; loss values, NLL4-TF values and the selected pseudo-epoch as a recorded value stay
  only in the fold record until T-B. No blinded fold can train before the bundle, so nothing has
  been exposed.
  (b) The refit's figure. The pilot's estimate, 4.945 MPS-hours, is 0.49 times the 10 h the owner
  approved (D-056), and the command lets that invocation spend up to 12.5. If the pilot rule
  fires, the refit (9.779) would take it to about 14.7, so the command refuses the refit after
  the pilot and says so; the next invocation, `finetune fold --fold q-2024-07
  --approved-by-owner H`, needs a figure H of at least 7.83 (9.779 / 1.25). D-056 does not say
  whether its "up to 10 h" also covers that invocation; the command never assumes it.
  (c) The pilot check's figure. Its estimate (0.89 to 0.98 CPU-hours) is below gate 3, so it
  runs without a figure and stops at 3,600 CPU-seconds charged; a figure, if one is passed, is
  the owner's CPU-hours for the check, which D-056's "up to 10 h" does not state.
  (d) Quarterly replicates (D-048 (4); D-056 estimates up to 29 h for them). Under the quarterly
  layout the months 2024-07, 2025-06 and 2026-06 fall in `q-2024-07`, `q-2025-04` and
  `q-2026-04`, but `folds.REPLICATE_FOLDS` lists the monthly labels only, so `--replicate` on a
  quarterly fold is refused until the owner decides the mapping (then a one-line change of
  `folds.py`, which is in neither code hash), before the walk-forward's gate-3 ask.
  (10) Not done, for later: `eval/crps.py` (6); `.gitignore` does not list
  `/reports/ledger/.lock`, which the first append creates; `docs/DATA.md` still calls the
  experiment ledger step 5's and does not name `KRONOS_AGENT_REPORTS_DIR`; the `folds.py`
  docstring still calls `wf-2024-07` the pilot; `cli/bench.py` prints
  `json_path.relative_to(REPO_ROOT)`, which raises with a relocated reports root (no test runs
  it; `cli.display_path` would not raise); `report/writer.py` refuses the repository and data
  roots in report text, not a relocated reports root; the training child gets no
  `KRONOS_AGENT_REPORTS_DIR` (it resolves no report directory, and an edit of `train.py` would
  make `timing-2.json` stale); the compute control of design 6.2 step 5, the walk-forward loop
  over folds 2 onwards, and the other commands of design (c).
  (11) Files. New: `ledger.py`, `budget.py`, `cli/ledger.py`, `finetune/walkforward.py`,
  `finetune/tokenizer_diag.py`, `finetune/pilot_check.py`; `tests/root_guard.py`,
  `ledger_fakes.py`, `fold_fakes.py` and the test files `test_root_guard.py` (31 cases),
  `test_ledger.py` (57), `test_budget.py` (11), `test_finetune_tokenizer_diag.py` (29),
  `test_finetune_pilot_check.py` (26, of which 3 skip), `test_finetune_walkforward.py` (54) and
  `test_finetune_fold_cli.py` (25). Changed: `paths.py` (`KRONOS_AGENT_REPORTS_DIR`,
  `ledger_dir()`), `data/partition.py` (4), `cli/__init__.py` (the `ledger` command),
  `cli/finetune.py` (three subcommands; `timing` unchanged), `tests/conftest.py`, and cases of
  `test_partition.py`, `test_cli.py`, `test_finetune_folds.py` (the quarterly pilot's facts),
  `test_manifests.py` and `test_provenance.py` (`kronos_agent.ledger` and `kronos_agent.envinfo`,
  which the sealed guard's lazy import adds to the fan path's import closure). Not edited, on
  purpose: the training-code files (`finetune.train.TRAINING_CODE_FILES`), whose edit would make
  every fold estimate refuse `timing-2.json` as stale until the measurement step is repeated
  (another gate-3 ask); `folds.py` (quarterly labels already work), `registry.py`, `timing.py`,
  `kronos/batch.py`, `report/writer.py` and `eval/bootstrap.py`.
