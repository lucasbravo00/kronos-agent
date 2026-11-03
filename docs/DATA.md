# Data layer (step 2)

How candles reach the repository, how a decision window is defined and hashed, how forecasts are
stored, and how the Binance-vs-BingX mismatch is measured. SPEC sections 5, 6 and 11 are the
requirements; the defaults of the open question batch (`docs/DECISIONS.md`, questions 8, 9 and
12) apply until the owner answers. Nothing here reads, defines or registers trading events: SPEC
section 4 and the volume-profile and POC parts of section 5 are without effect (D-042). The design
decisions behind this file are D-022 to D-049 of `docs/DECISIONS.md` (D-030 and D-031 withdrawn by
D-042).

Step-2 "done when" (SPEC section 18 as D-042 narrows it: the mismatch table and the
no-future-candles test):
- `reports/data/mismatch.md` holds every row of SPEC section 5's mismatch table except the POC
  rows (D-042), the Kronos signal agreement and its measured two-seed floor included, each with its
  interval and comparator. The Kronos part runs only after the owner's gate-3 go-ahead (given on
  2026-11-01 for the 873 fans of the primary sample, D-040); step 2 stays open until it exists.
- `uv run pytest tests/test_no_lookahead.py -rA` passes with verified weights and zero skips (it
  covers the decision windows and the fans).

## Conventions

- **Time.** Everything is UTC. A candle is keyed by `open_ms` (epoch milliseconds of its open);
  every `open_ms` is a multiple of its interval. Kronos timestamps are tz-naive `datetime64[ns]`
  meaning UTC, labelled by the candle's OPEN time (the context ends at `D - 1h`, the targets start
  at `D`); a close-time labelling would be a separate, registered variant.
- **Ranges are half-open.** Every `*_ms` range in every API is `[start_ms, end_ms)`; an inclusive
  bound is named `last_open_ms`. Batch commands take `--from` (inclusive) and `--to` (exclusive).
  The public client's `closed_klines` and `fetch_twice` too: an `end_ms` exactly at a candle's
  open excludes that candle, as the exchange's `endTime` does (D-041), and the client drops it
  whatever the exchange does; an `end_ms` off the grid (`D - 1 ms`) keeps the candle before it.
- **Decision hour `D`.** An exact UTC hour. At `D` the last closed 1h candle opened at `D - 1h`.
  The Kronos context for `D` is the 512 hourly candles opening in `[D - 512h, D)`; the targets open
  at `D, ..., D + (N - 1)h`. Nothing opening at or after `D` is read for `D`. Seeds come from `D`
  with the project symbol `BTC-USDT` for both sources, so the same `D` and stream give the same
  seeds on Binance and on BingX.
- **Strings are the record.** Stores keep the exchange's decimal strings verbatim. A window's
  `sha256` is `window_sha256` of `kronos_agent.exchange.public_market` over the
  `[open_ms, open, high, low, close, volume]` rows; its `inputs_sha256` is the sha256 of the
  canonical JSON of those rows, the `amount` policy and, under the `binance_quote_volume` policy,
  the quote-volume strings: every string the model reads and how it is turned into channels.
- **Floats.** Model inputs are parsed from the strings with Python's `float()`. Bulk numeric reads
  use `pandas.read_csv(..., float_precision="round_trip")`, which is the same correctly rounded
  conversion. `pandas.to_numeric` and `read_csv` without `round_trip` are forbidden for candle
  values: they are not correctly rounded (`"465655238.20240235"` is one ulp off).
- **Never fabricated.** No store fills, interpolates or deduplicates candles. Missing candles are
  gaps; a window that needs one is refused, and every study reports what it lost.
- **No new dependencies** (gate 4): CSV and JSON, `zipfile`, `csv`, `decimal`, numpy, pandas,
  httpx, matplotlib.
- **Durable writes** (`kronos_agent.data.fsutil`). Every file is written to a unique temporary
  file in its directory (`tempfile.mkstemp`), flushed and `fsync`ed, moved into place with
  `os.replace`, and the directory is `fsync`ed. Files that must never be overwritten (stored fans,
  saved windows, recorded study configs) are published with `os.link`, which fails if the name
  exists. The network jobs and the builds and merges from their archives take a non-blocking
  `fcntl.flock` on `raw/binance/.lock` or `raw/bingx/.lock` (the lock file is never deleted); a
  second concurrent run is refused.
- **Store reads hold the job lock shared** (`kronos_agent.data.store`). Every store read
  (`read_store`, hence `read_study_frame`) holds its source's job lock shared (`raw/binance/.lock`
  for `binance` and `binance_official`, `raw/bingx/.lock` for `bingx`) and opens the CSV once:
  size and sha256 are checked on that open file, which is then searched and parsed, so the rows
  returned are the bytes verified. A read while a run holds the lock is refused (`StoreBusyError`,
  both a `LockBusyError` and a `StoreError`), and a run is refused while a read holds it; a study
  that records a store's meta holds `store_read_lock` around the meta and its reads (coverage and
  the mismatch do). A lock file that does not exist yet leaves reads unlocked, and reads never
  create it.
- **Temporary files of killed runs.** A process stopped without its clean-up (SIGKILL, the OOM
  killer; SIGTERM or SIGHUP outside the command line, which turns those two into `SystemExit` so
  that the clean-up runs) leaves its `mkstemp` temporaries (`.<name>.<8 characters>.tmp`,
  downloads `.part`). Every writer of `candles/binance`, `candles/binance_official` and
  `raw/binance` holds `raw/binance/.lock`, and every writer of `candles/bingx` and `raw/bingx`
  holds `raw/bingx/.lock`. The next run that takes the lock through
  `kronos_agent.data.store.job_lock` removes them before it writes and reports them; nothing else
  is ever removed (lock files, `.DS_Store`, archives, symlinks). `data status` lists those present
  and marks the ones of a run in progress. Temporaries under `windows/`, `forecasts/` and
  `studies/` are not swept (no job lock covers them).
- **Provenance** (`kronos_agent.provenance`). Every store meta, report and stored fan records the
  code as `{commit, dirty, source_tree_sha256}`. The tree hash is one sha256 fed, for every file of
  `src/kronos_agent/**/*.py`, `params.yaml`, `pyproject.toml`, `uv.lock` and `manifests/*` in sorted
  order of its relative POSIX path (`__pycache__` and `.DS_Store` excluded), with the line
  `<path> NUL <sha256 hex of its bytes> LF`. `dirty` counts untracked files. The repository stays
  at one commit, so the tree hash is what identifies the code.
- **Reports** go to `reports/data/<stem>.json` + `<stem>.md` with figures under
  `reports/data/figures/`. They carry no absolute path (data paths are relative to the data root),
  pass `kronos_agent.envinfo.assert_no_personal_data` (JSON and Markdown), and are written with
  `allow_nan=False` (an undefined statistic is `null` with a reason); the Markdown is ASCII. They
  never name a BingX host outside `kronos_agent.exchange` (the writer refuses one). Per-hour series
  go to `data/studies/<stem>.csv`, with their sha256 recorded in the report. Values of candles
  opening at or after the sealed start are never printed. Append-only record files sit beside the
  reports (`reports/data/<name>.jsonl`, one canonical-JSON object per line). They are written
  through `kronos_agent.report.writer.append_record` under the same refusals: no NaN, no personal
  data, no absolute path, no BingX host. Lines already written are kept byte for byte, and a torn
  or malformed file is refused, never repaired.

## Layout (`data/`, gitignored, relocatable with `KRONOS_AGENT_DATA_DIR`)

```
data/
  raw/binance/monthly/<interval>/BTCUSDT-<interval>-YYYY-MM.zip (+ .zip.CHECKSUM)
  raw/binance/daily/<interval>/BTCUSDT-<interval>-YYYY-MM-DD.zip (+ .zip.CHECKSUM)
  raw/binance/plan-<interval>.json                          last archive plan (sync -> build)
  raw/binance/.lock, raw/bingx/.lock                        job locks (runs exclusive, reads shared)
  raw/bingx/<interval>/<chunk_start_ms>.csv (+ .json)        BingX chunks (archive, never deleted)
  raw/bingx/history-<interval>.json                         history start a backfill established
  raw/bingx/revision-<interval>.json                        revised candles, recorded once (D-045)
  candles/binance/1m.csv + 1m.meta.json                     from the 1m archives
  candles/binance/{5m,15m,1h}.csv + .meta.json              resampled from binance/1m (training)
  candles/binance_official/1h.csv + 1h.meta.json            Binance's own 1h archives (check only)
  candles/bingx/{1h,15m,5m,1m}.csv + .meta.json             BingX real market (public GET)
  windows/<source>/context-<max_context>/<YYYY-MM-DDTHH>Z.json  saved windows (their inputs)
  forecasts/<source>/<config_id>/<stream>/<YYYY-MM-DDTHH>Z.npz stored fans
  studies/<stem>.csv                                        per-hour study series
```

Expected footprint: about 0.15 GB of Binance zips, about 0.35 GB for `binance/1m.csv`, under
0.1 GB for everything else.

## Store format (`kronos_agent.data.store`)

`candles/<source>/<interval>.csv`: a header line, one row per candle, ascending and unique by
`open_ms`, `\n` endings, no quoting, columns:

- BingX: `open_ms,open,high,low,close,volume`
- Binance (both): the same plus `quote_volume,count,taker_buy_volume,taker_buy_quote_volume`

Values are plain decimals (`^-?[0-9]+(\.[0-9]+)?$`; `count` `^[0-9]+$`), the exchange's strings
verbatim; the same rows always give the same bytes. A streaming writer appends archive by archive
(the 1m store is never held whole as strings) and publishes the CSV, then its meta (the commit
marker). `<interval>.meta.json` records: schema version, source, symbol (`BTCUSDT` or
`BTC-USDT`), interval and step, columns, row count, first/last `open_ms` (and UTC text), gaps as
`[first_missing_open_ms, end_ms)` ranges between the first and last row, the number of missing
candles, OHLC-relation violations (`high < max(o, c, l)`, `low > min(o, c, h)`, `volume < 0`,
compared as decimals; counted and sampled by `open_ms`, never fixed), CSV relative path, sha256
and size, provenance (archives with their checksums, or BingX chunks with their sha256 and
completeness), creation time and the code. Reads hold the source's job lock shared and verify the
CSV size and sha256 against the meta on the very file they then parse (opened once); a ranged
read finds `[start_ms, end_ms)` by binary search over the byte offsets of the sorted rows and
parses only those rows.

## Sources

### Binance USD(S)-M perpetual BTCUSDT (training and discovery)

`https://data.binance.vision/data/futures/um/{monthly,daily}/klines/BTCUSDT/<interval>/`, public, no
keys. Observed on 2026-11-01: monthly 1m archives exist from `2020-01` (the 2019 monthly archives
return 404, so every plan probes the daily archives of the 2019 months); the monthly archive of
the month just ended is not yet published while its daily archives are; each `.zip` holds one CSV
with columns `open_time, open, high, low, close, volume, close_time, quote_volume, count,
taker_buy_volume, taker_buy_quote_volume, ignore`; recent files start with that header line, 2020
files have none; times are epoch milliseconds; price strings change format across periods (`7225`
and `7171.10` in one 2020 file); `.CHECKSUM` is `<sha256 hex>  <zip name>`.

- **Client** (`BinanceVisionClient`): GET only, pinned to `https://data.binance.vision` on the
  default port (compared on `url.raw_host`), path allow-list `/data/futures/um/{monthly,daily}/
  klines/BTCUSDT/{1m,5m,15m,1h}/BTCUSDT-<interval>-<period>.zip[.CHECKSUM]`, redirects refused,
  environment ignored, no credentials, cookies cleared, identity encoding, paced (>= 0.25 s between
  requests), retries for timeouts, network errors and 5xx (after 1, 5, 15 s; then
  `BinanceVisionUnavailable`, which the owner must hear about: SPEC section 5), 404 = "not
  published", any other status an error.
- **Plan.** For an interval, months from `2019-09` (the perpetual's launch) to the month before the
  current one: a month uses its monthly archive when published, otherwise each published daily
  archive of that month; the current month uses its daily archives up to yesterday (UTC).
  Publication is probed with the `.CHECKSUM` files (one small GET each), and each download is then
  verified against the checksum the plan read. The plan (archives, missing periods) is saved to
  `raw/binance/plan-<interval>.json`.
- **Download.** Stream to a temporary file, cap 64 MiB per archive, verify the sha256 against
  `.CHECKSUM`, publish. A checksum mismatch deletes the download and fails. Gate 4: the budget is
  4.5 GB and counts every byte already under `raw/binance` plus this run's. A run first plans
  every interval it syncs and saves each plan. It prints each interval's estimate (80 bytes per
  candle of every archive to download, deliberately high) and the run's total. It is refused
  (exit 2) before any download when the bytes on disk plus that total would pass the budget. The
  downloads of every interval then charge one budget, every byte received included. The sync
  takes `raw/binance/.lock` through `job_lock`, which first removes the temporary files of killed
  runs. Expected total: about 0.15 GB.
- **Republished archives.** A remote `.CHECKSUM` that differs from a locally verified archive is
  reported and the local copy is kept, unless `--refresh <period>` is given.
- **Parse** (archives are untrusted data): read the single expected member in memory (uncompressed
  cap 256 MiB, no extraction to disk); parse with the `csv` module, exactly 12 fields per line, no
  blank lines; the first line is a header only if its first field is not all digits, and then it
  must equal the column list exactly; detect microsecond times (`open_time >= 1e15`) and convert
  them; check `close_time == open_time + interval - 1 ms`; rows aligned, unique, increasing and
  inside the archive's period. A malformed archive fails the build naming the archive and the line.
- **Stores.** `binance/1m` from the planned 1m archives (all must be present and verified);
  `binance/{5m,15m,1h}` resampled from it on the way; `binance_official/1h` from Binance's own 1h
  archives. When a monthly archive replaces daily archives already on disk, their rows are
  compared as decimals and any difference is reported in the store's provenance.

### BingX real market (evaluation and live)

`PublicMarketClient` (host-pinned, paced at <= 0.5 req/s). History observed in step 1: 1h from
2024-04-27T10:00Z; 1m from about 2025-07-31 and 5m from about 2026-09-14, both on rolling
retention (about 450 and 48 days, the figures of `RETENTION_DAYS`); 15m depth is unknown. The
first download (2026-11-02) measured 45 days of 5m and 100 days of 15m and found the 1m start
unchanged: these figures are revised with the facts of D-049 (D-050). The raw chunks are
therefore an archive: never deleted, never refetched once complete. The owner archives the 1m
and 5m candles as insurance for their future strategy; the retraining does not use them
(D-049). The 5m retention (six to seven weeks) means `data bingx --interval 5m` must run at least
monthly; `data status` warns when the newest archived 1m or 5m candle opened more than a third
of its retention ago, or when there is none, and, for every interval, lists the recorded
revisions (below) and warns about an incomplete chunk holding archived candles its last refetch
no longer returned (below) and about any other incomplete chunk older than the newest archived
chunk (a backfill that stopped before refreshing it: a failed run or a `--max-chunks` bound).

- **Chunks.** An epoch-aligned grid of 1000 candles: chunk `k` holds the open times
  `[k * 1000 * step, (k + 1) * 1000 * step)`. A chunk is one `klines` page with
  `startTime = k * 1000 * step` (inclusive at an exact boundary: observed) and
  `endTime = min((k + 1) * 1000 * step - 1, server_time_ms)`, one millisecond earlier when that
  falls exactly on a candle boundary, so `endTime` is never an exact boundary (an exact boundary
  excludes the candle opening there, and the `boundary - 1` form returns the whole chunk:
  observed, D-041) and never in the future; `limit = 1000`, fetched with `client.klines` (not
  `closed_klines`, which refuses internal gaps).
- **Settled candles only.** A candle is kept only if `open_ms + step + 90 s <= server_time_ms`,
  with the server time read once at the start of the run (SPEC section 5: closed candles only, at
  least 90 s after the close). A chunk is complete when its last slot passes the same test.
- **Twice until they match.** Each chunk is fetched twice; after the settled filter, the two
  results must be byte-identical (`canonical_bytes`). A differing pair is repeated, up to 3 pairs,
  then the backfill fails naming the chunk. Every candle must lie inside the request, on the grid,
  unique, with plain decimal strings; `code != 0` is an error, never an empty chunk.
- **Archive.** Each non-empty chunk is written to `raw/bingx/<interval>/<start>.csv`, then its
  sidecar `<start>.json` (the commit marker: chunk bounds, request bounds, server time, settle
  delay, settled slots, `complete`, rows, first/last open, gaps, `archive_only`, CSV sha256 and
  size, requests, fetch pairs and mismatch retries, fetch time, code). Empty chunks are never
  archived. Before any request the whole archive is verified: a CSV without a valid sidecar, a
  sidecar without its CSV, or a CSV whose size or sha256 does not match its sidecar (modified, or
  a run killed between a refresh's CSV and its sidecar) stops the backfill with an error naming
  the file; a server time earlier than an archived chunk's stops it too. Nothing is repaired
  automatically: removing both files of a chunk lets the next backfill fetch it again as a new
  chunk, which loses the candles past the exchange's retention, while the candles it serves with
  a revision recorded in `raw/bingx/revision-<interval>.json` take their recorded first-seen rows
  (the archived row of the open's earliest revision; another value served is recorded as a
  revision like any other). Complete chunks are skipped without a request. Incomplete chunks
  (normally the newest one) are refetched every run, and archived candles are never overwritten
  or deleted. A candle the refetch returns with other strings than the archive is a revision.
  The owner decided on 2026-11-02 (D-045) that the archive keeps the first-seen row, which is
  what a live decision would have used, and that the backfill goes on. The revision is recorded
  in `raw/bingx/revision-<interval>.json` before the chunk's CSV and sidecar are written, so a
  run killed in between never loses it: the chunk stays as archived and the next run observes
  the revision again (tested). Record schema 2: `columns` and a `revisions` list, one entry per
  revision with the chunk (start and CSV path), the open time, the `archived` and the
  `refetched` rows (`open_ms,open,high,low,close,volume` as the chunk CSV holds them, without the
  line end) and the server times of the first and the last run that observed it. A run that
  observes the same pair of rows again changes only the last observation, so a revision is never
  recorded twice. Another value for the same open is another revision. A schema-1 record (the
  one revision of a backfill that stopped at it, before the owner's decision) is read as that
  revision and rewritten in schema 2 by the next run that records one. A record that cannot be
  read stops the backfill before any request, like an inconsistent archive, but never the merge,
  which reads the chunks alone (tested). The rest of the chunk is refreshed as below, its
  completion included: a revision never keeps a chunk incomplete, and a complete chunk is never
  fetched again, so its revisions are no longer observed. `data bingx` prints each chunk's
  revisions as a WARNING as soon as they are recorded (the chunk, how many were new, their open
  times, ten by name and the rest counted), so also when a later chunk of the walk fails, and
  still exits 0 when nothing else failed; `data status` lists every recorded revision as a
  WARNING (open time, first and last observation, ten by name and the rest counted). Neither
  prints candle values, which stay in the record, so no value of the sealed segment is printed
  (the coverage report carries the status warnings). An archived candle the refetch no longer
  returns (rolled off the retention after a missed run, or no longer served) is kept from the
  archive and listed in the sidecar's `archive_only` (`{rows, ranges}`; sidecar schema 2, schema
  1 sidecars are read as having none). New settled candles are added. A refreshed chunk
  completes (its last slot settled) only when the rolling retention explains every candle kept
  that way: 1m and 5m only, a candle opening more than the observed retention (`RETENTION_DAYS`:
  450 and 48 days) before the server time and before the first candle the refetch returned. So
  a refetch that returns nothing keeps the archived candles and completes the chunk only when
  they are older than the retention. The 1h history is fixed and 15m has no known retention, so
  none of their candles is explained. Otherwise the chunk stays incomplete and the next run
  fetches it again (a transient empty or short answer never becomes a permanent gap), and
  `data status` warns about it. The backfill summary counts the candles kept
  (`kept_from_archive`), those not explained by the retention, inside the range the exchange
  still serves (`kept_inside_served_range`; the command prints a WARNING: report them to the
  owner), the revised candles observed (`revised_candles`, their opens in `revised_opens`) and
  the revisions not recorded before (`new_revisions`).
- **Walk and history start.** From the chunk holding the newest settled candle backwards. The walk
  stops after 3 consecutive empty chunks (both fetches empty) older than the oldest archived chunk
  (so every run fetches those 3 chunks again); the history start is the oldest non-empty chunk's
  first candle only if nothing older was served when that candle was archived: the run that stops
  fetched it, or it opens after its chunk's start, or `raw/bingx/history-<interval>.json` already
  names it. Otherwise (it opens at its chunk's start and an earlier, possibly bounded, run
  archived it) the empty chunks below may have rolled off since, and the start stays open. An
  established start is recorded once in `raw/bingx/history-<interval>.json` (stop reason, server
  time of the run that established it), which is written only when the start changes. An empty
  chunk between non-empty ones is a recorded gap. Three consecutive empty chunks inside the
  history would end the walk early (3000 candles: 125 days of 1h, 50 hours of 1m), an accepted
  limit. `--max-chunks N` bounds a run: a run that the bound stops (stop reason `max_chunks`)
  never writes or changes the record, while a bounded run whose walk ends before the bound (3
  empty chunks below the oldest archived one, or the epoch) establishes the start like any other.
  The opt-in network tests observed (D-041) that a range entirely before the history answers
  `code 0` with an empty list, the stop signal.
- **Merge.** All chunks of an interval, complete or not, form `candles/bingx/<interval>`; the meta
  names the incomplete ones. The store is rebuilt from the archive alone, so a revised candle is
  in it with its first-seen row. Candles missing inside the range are gaps. The provenance also
  records, per chunk, the candles kept from the archive alone (`archive_only_rows`), and
  `history_start` (established, the start, its stop reason and server time, or the reason it is
  not established). The merge holds `raw/bingx/.lock` through `job_lock`.

## Resampling and its check

`binance/1m` -> `5m`, `15m`, `1h`, archive by archive (buckets never cross an archive boundary):
`open` = first minute's string, `close` = last minute's string, `high`/`low` = the string of the
largest/smallest value (first occurrence on ties), `volume`, `quote_volume`, `taker_buy_volume`,
`taker_buy_quote_volume` = exact decimal sums (`decimal.localcontext` with `prec = 60` and the
`Inexact` and `Rounded` traps on; result `format(total, "f")`), `count` = integer sum, bucket
labelled by its open. A bucket is kept only when every minute is present; an incomplete bucket is a
gap, listed with its minute count in the store's provenance.

Check (`reports/data/binance_resample_check`, written by `data binance` once both 1h stores
exist): every hour present in both `binance/1h` and `binance_official/1h` (all months, not a
sample) is compared as decimals. `open`, `high`, `low`, `close`, `volume`, `count` and
`taker_buy_volume` must be equal. Quote volumes may differ by at most
`sum(0.5 * 10^-d_i) + 0.5 * 10^-d_official` (the rounding of the 60 summands and of the official
value, `d` = digits after the decimal point of each string). The report gives counts and the
largest difference per column, lists the hours present on one side only (with the minute count of
each incomplete bucket), and the decision hours lost to incomplete buckets per segment for the
512 h context (`D` is lost when `[D - 512h, D)` holds an hour with 1 to 59 of its minutes in
`binance/1m`). For hours at or after the sealed start it prints counts and `open_ms` only, never
values. Any OHLC, volume or count mismatch is reported to the owner.

## Partition (question 8 defaults; `kronos_agent.data.partition`)

Half-open UTC intervals, by decision hour, for every source:

| Segment | From | To |
| --- | --- | --- |
| training / discovery | 2020-01-01T00:00Z | 2024-06-09T08:00Z |
| embargo (520 h = 512 + N=8) | 2024-06-09T08:00Z | 2024-07-01T00:00Z |
| validation | 2024-07-01T00:00Z | 2026-07-10T08:00Z |
| embargo (520 h) | 2026-07-10T08:00Z | 2026-08-01T00:00Z |
| sealed | 2026-08-01T00:00Z | open |

N1/N2 sub-partition inside validation: train 2024-07-01T00:00Z -> 2026-01-01T00:00Z, embargo 520 h,
validation 2026-01-22T16:00Z -> 2026-07-10T08:00Z.

- **Horizon purge.** The 520 h embargo protects a sample whose targets span at most 9 candles. A
  sample at `D` in a segment that reads targets up to `D + (h - 1)h` is usable only if that time is
  earlier than the start of the next sample-holding segment minus `max_context` hours; longer
  samples are purged, never shortened (`sample_fits`; the N1 validation segment is followed by the
  main partition's second embargo). This is the only cut-off for samples that read targets. At the
  end of validation and of N1 validation the last target must open before 2026-07-10T16:00Z (the
  sealed start minus 512 h, the first candle of the sealed segment's first context): every
  validation decision up to 2026-07-10T07:00Z is kept for `h <= 9`, its targets running into the
  second embargo as SPEC section 6 intends (embargo = context + N), and up to
  `2026-07-10T16:00Z - h` for longer horizons (2026-07-09T16:00Z at `h = 24`).
  `check_embargo(max_context, horizon)` refuses any configuration with
  `max_context + horizon > 520` (Kronos-mini, or `N >= 9` with a 512 context): the embargo bounds
  the context plus the trading horizon N (SPEC section 6: embargo = context + N). Evaluation
  horizons longer than N (the skill curve of SPEC section 8 reaches 24 candles) will be handled by
  the horizon purge with an explicit evaluation length designed with the retraining, not by
  recomputing the partition.
- **Kronos cutoff.** Kronos as shipped saw Binance up to June 2024: forecasts with the pretrained
  weights refuse `D < 2024-07-01T00:00Z` unless the caller opts in explicitly (tests).
- **Sealed segment (gate 6), guarded by candle open time at the data-access layer.** Studies and
  window sources read candles only through `read_study_frame`, which refuses any range reaching
  `open_ms >= 2026-08-01T00:00Z` without a `SealedAccess`. `read_store` stays for integrity work
  (checksums, gaps, merges, the resample check, status), whose values never feed a study; tests
  check that the study and window modules never call it. A `SealedAccess` can be created only from
  the owner's order recorded in the experiment ledger (step 5); until then creating one raises, so
  nothing in step 2 can open the sealed segment. A decision hour at or after the sealed start is
  refused (`require_accessible`) by every command that takes one. A study that reads targets keeps
  exactly the samples the horizon purge allows (above), so no target opens in the sealed segment;
  a descriptive candle study stops its candles at the sealed start.
- **Live window.** `live_decision_window` reads the server time itself and refuses unless
  `D + 90 s <= server time < D + 15 min` (SPEC section 11: no trading more than 15 minutes late),
  so it can never fetch history.

## Decision windows (`kronos_agent.data.windows`)

A `DecisionWindow` holds the source, `D`, the 512 candles as strings (and, for a Binance window,
their quote-volume strings), `sha256`, `inputs_sha256`, the float context (`open, high, low, close,
volume, amount`), `x_timestamps` (512 tz-naive hours ending at `D - 1h`), `y_timestamps` (`N`
hours from `D`), the `amount` policy and, on the live path, the number of fetch pairs. One builder
makes it from any candle sequence: it keeps exactly the candles opening in `[D - 512h, D)`, refuses
a missing one (`WindowGapError`, naming the missing ranges) and ignores everything else (later
candles, the unclosed one). The store path (`StoreWindowSource`, a 1h study frame of strings
loaded once per source object) and the live path (`live_decision_window`: `fetch_twice` over
`[D - 512h, D - 1 ms]`, at most 3 pairs in all, each at a freshly read server time and only while
it is before `D + 15 min`, then refused, SPEC section 11) both go through it, so both give the
same hashes and floats.

A window built from a Binance store carries the quote volumes of its candles under every `amount`
policy; only `binance_quote_volume` reads them, and `inputs_sha256` is unchanged under the other
policies. `save_window` publishes the inputs of a decision's window to
`windows/<source>/context-<max_context>/<D>.json`: the candles as strings, their `sha256` and, for
a Binance window, the quote volumes (a Binance window without them is refused). The `amount`
policy, `inputs_sha256`, the horizon and the live fetch pairs are not in the file; each fan's
metadata records them. One saved window therefore serves every policy and every retry of its hour.
An existing file must be identical; other candles or quote volumes for the same source, context
and hour are refused, never overwritten.

`amount` policies (question 9, default A): `volume_x_mean_ohlc` adds the proxy column with the
predictor's own expression (`kronos_agent.kronos.features.add_amount_proxy`, bit-identical to the
implicit one); `binance_quote_volume` uses Binance's quote volume (Binance stores only); `zeros`
passes zero `volume` and `amount` (what the predictor does when volume is missing).

## Forecasts

`kronos_agent.kronos.decision.forecast_window(models, window, params, stream=...)` checks the window
against the governed parameters (model, device, `max_context`, `horizon_n`, `amount_policy`),
refuses models that would not forecast as `load_models` leaves them (below), checks that the
window's candles, hashes, float context and timestamps describe the same inputs, applies
`check_embargo` and the Kronos cutoff, and calls `fan_from_params` with `D` and
`symbol=params.symbol` (never the source's own symbol). The **signal** of a fan is the sign of
`mean_k(close[k, N]) / close[D - 1h] - 1` (the directional control of SPEC section 8 at the live
horizon `N = 4`); `p_up` is the fraction of trajectories whose close at `N` is strictly above
`close[D - 1h]`.

Stored fans (`kronos_agent.data.forecast_store`): `forecasts/<source>/<config_id>/<stream>/<D>.npz`
with `paths` (`K x N x 6` float64), `y_open_ms` and a canonical-JSON metadata string (source, `D`,
stream, window `sha256` and `inputs_sha256`, `context_sha256` (the floats and timestamps the model
read), context and horizon, the live fetch pairs, seeds, model, weight hashes, threads, torch
version, platform, params version, amount policy, the config and its id, code, the sha256 of
`paths`, elapsed seconds, creation time), read with `allow_pickle=False`.

`config_id` is the first 16 hex characters of the sha256 of the canonical JSON of every input of a
configuration that changes a fan: model, weight hashes, device, `max_context`, the predictor clip
(`kronos.weights.PREDICTOR_CLIP` = 5, the constant `load_models` builds every `KronosPredictor`
with), `horizon_n`, temperature, `top_p`, `top_k`, trajectories, threads, amount policy, seed salt,
seed scheme, seed symbol, the torch, numpy and pandas versions, platform, and one sha256 over the
whole files of the forecasting code (`kronos/forecast.py`, `kronos/features.py`,
`kronos/decision.py`, `kronos/weights.py`, `kronos/vendored.py`, `data/windows.py`, `seeds.py`,
`runtime.py`) and the vendored-source manifest. `kronos/weights.py` loads the weights and builds
the predictor (`load_models`) and `kronos/vendored.py` imports the classes it builds them from, so
any edit of the models' construction (a quantisation, another checkpoint under the registry's
weight hashes, a dtype cast, another class) gives a new `config_id` too. Whole files are hashed on
purpose. An edit anywhere in them, even one that cannot change a fan (the weights download, saving
a window, a fan's signal), gives a new `config_id`: it costs recomputation (and a new gate-3
go-ahead above one CPU-hour), never a wrong reuse. One file that shapes a fan's numbers is not
hashed: `data/store.py::float_values` is covered by each fan's `context_sha256`. On top of that,
`forecast_window` and the configuration refuse models that would not forecast as `load_models`
leaves them (`weights.inference_problems`): a predictor whose clip (read back from it), context
length or device differs from what `load_models` builds, or a tokenizer or model (as `LoadedModels`
and as the predictor hold them) with any submodule in training mode or any parameter requiring
gradients. That refusal is the only check of models changed in memory after `load_models` returned
them (other tensors or a quantised copy pass it), so only `load_models` builds the models of
stored fans. `tests/test_provenance.py` classifies every module of the fan path's import closure
as hashed or covered otherwise (the configuration, a refusal, the window hashes or
`context_sha256`).

A stored fan is reused only when its config id and its metadata match the current window's hashes,
`D`, source, stream, amount policy, context, horizon and `context_sha256`; otherwise
`ForecastStoreError` names the file, which is never overwritten. `context_sha256` is one sha256 of
the canonical JSON of the context's columns, shape and timestamp counts, then the context as
C-ordered little-endian float64, then `x_timestamps` and `y_timestamps` as little-endian int64
nanoseconds. A changed string-to-float conversion is therefore refused on reuse wherever its code
lives, never reused silently: a stale fan can remain in its directory, but no batch uses it. The
same holds for a fan whose store changed after it was computed (`data binance --refresh` of a
republished archive, say). `forecast` (a batch, or one hour with `--save`) and `mismatch --kronos`
fail naming the first stale fan they meet; a plain `mismatch` reports its Kronos part as not run,
naming it. The way out is to move the named fan aside (other fans of the same source may be stale
too: each run names the first it meets), after which the batch computes it again. Fans of
different configurations never share a directory. Stored fans are schema 2; `load_fan` refuses
schema-1 files (no `context_sha256`).

**Batches** (`kronos_agent.kronos.batch`): the parent process checks the runtime threads, builds
every task's window first and drops (and lists, with the missing range) the hours whose window is
incomplete. With one worker the fans run in the parent; otherwise the parent verifies the weights
and the vendored tree, then starts spawn-started worker processes of 2 threads each, at most
`os.cpu_count() // 2` and only with at least 4 GiB of RAM (each worker calls `import_kronos()` and
loads its own models once, about 1 GB); results come back in task order; a fan depends only on its
inputs, so worker count and task order never change it (tested).

**Gate 3.** Every batch entry point first measures the seconds per fan (after a warm-up, at 2
threads) and estimates `cpu_seconds = pending fans x seconds per fan x 2 threads` (fans already
stored are not pending) and `wall_seconds = cpu_seconds / (2 x workers)`. It prints both;
`--estimate` never runs the batch's fans. Above 3600 CPU-seconds the batch runs only with
`--approved-by-owner CPU_HOURS`, the CPU-hours the owner approved in chat for the estimate shown,
and only when the measured CPU-hours are at most 1.25 times that figure (`GATE3_APPROVAL_TOLERANCE`,
D-040). Otherwise it is refused (`REFUSED:`, exit 2) before any fan, and the refusal prints both
figures and their ratio. Below the limit the figure is not needed. The time the flag was given is
the command's start. A batch also refuses to compute more fans than its estimate counted. The
mismatch study appends, once gate 3 has passed and before its first fan, a record to
`reports/data/mismatch.gate3.jsonl`: the approved CPU-hours, the measured estimate with its pending
fans and CPU-seconds, `config_id`, design (primary or fallback), the dropped tasks and the command's
start. Every `mismatch --kronos` batch that computes fans appends one, approved or not (below the
limit no approval is needed); a rerun with nothing pending adds none. The report lists every record
whose `config_id` is the one of its fans. `kronos-agent forecast` prints the estimate and the
approval line (figure, start time, measured CPU-hours, ratio).

## Studies and reports

**Recorded before they run.** Each study's definition (samples, statistics, blocks, resamples, seed
keys, comparators, reading rules) is canonical JSON in `reports/data/<stem>.config.json`, written
before its first run; every study command verifies it (or records it) before reading any candle
or running any fan, and a run whose definition differs is refused (exit 2; a changed study is a
new, separately reported study). `--record-config` records a definition without running the
study. Every report embeds that hash and the sha256 of this file.

**Every report carries** the study name, config and `DATA.md` hashes, the code, the test budget
consumed (0: these measure the data, not a strategy, and look at no trade outcome), the number of
statistics with intervals and the number expected to miss their true value by chance at 95 % (0.05
per interval), the bootstrap seed key of each statistic, the Kronos seed scheme and streams where
used, eligible and used hours with the reasons and ranges of every exclusion, and a named
comparator for every metric.

**Intervals** (`kronos_agent.eval.bootstrap`). Each statistic gets a 95 % percentile block-bootstrap
interval: 1000 resamples of whole blocks with replacement, numpy seed = first 4 bytes of
`sha256("kronos-agent/bootstrap/v1/<key>")` with `key = <study>/<group>/<statistic>`. Blocks are
calendar weeks (Monday 00:00 UTC) for hourly series and 4 calendar weeks for the 61-hour Kronos
sample (its contexts overlap for about 3 weeks; also the `dz` of the `amount` section). Statistics
aggregate per-row values built before resampling (returns, price differences, agreement flags), so
block seams never create values. Means, shares, correlations and paired differences get intervals;
medians and quantiles only when their bootstrap distribution is not a single value; p99 and max are
descriptive. With fewer than 20 blocks, or more than 1 % of failed resamples, only the point
estimate is given, with the reason. A share of exactly 0 or 1 gets a Wilson interval at the number
of blocks. Each share's interval also reports `n_eff = n / deff`, where
`deff = var_bootstrap / (p (1 - p) / n)`. Paired comparisons resample both columns in one call.

**Figures** (SPEC section 14): matplotlib `Agg`, object-oriented API, fixed size (8 x 4.5 in), DPI
(100) and rcParams (DejaVu Sans, which ships with matplotlib), PNG with the `Software` metadata
removed, regenerated by the command that writes the report; two renders give identical bytes
(tested). `timezone` (UTC) and `date.epoch` (1970-01-01T00:00:00) are pinned in the rcParams too,
because `matplotlib.rcdefaults()` keeps them: date axes are placed and labelled in UTC whatever a
`matplotlibrc` or the process says (tested with a `MATPLOTLIBRC` in a fresh process).

1. **Coverage** (`reports/data/coverage`, `kronos-agent data coverage`): first and last candle of
   every store with the date observed and gap ranges and, per source and interval (BingX 1h, 15m,
   5m, 1m; Binance 1m and 1h), the decision hours of each segment before the sealed start
   (eligible) and those whose 512 h context is complete at that interval (usable: every candle
   opening in `[D - 512h, D)` present), with the unusable ranges. It answers SPEC section 5's
   "verify how far back 1m and 5m go". The history depth observed (`history_days_observed`, the
   first candle's age at its chunk's fetch) is reported only when the store's provenance says a
   backfill established the history start at its first candle and the oldest chunk's last fetch
   kept no candle from the archive alone. Otherwise it is null with `history_days_observed_reason`;
   when the start is not established, the Markdown shows `not established` and marks that store's
   usable decision hours as lower bounds. Each store's meta and reads are taken under
   `store_read_lock`; a run holding the lock makes the study refuse (exit 2).
2. **Resample check** (`reports/data/binance_resample_check`), above.
3. **Binance vs BingX mismatch** (`reports/data/mismatch`, `kronos-agent mismatch`), on every hour
   present in both sources from 2024-04-27T10:00Z to the sealed start (validation outcomes are not
   involved: these are contemporaneous candles of two venues). Its sections are alignment,
   returns, prices, volume, amount and kronos; its series are `studies/mismatch_hours.csv`,
   `mismatch_amount.csv`, `mismatch_amount_windows.csv` and, for the Kronos part,
   `mismatch_kronos.csv`. Each store's `csv_sha256` is recorded as read: both job locks are held
   shared around the reads, and the meta read before and after each frame must agree.
   - **Alignment:** the correlation of 1h log returns at lags -2..+2 h; the report fails unless lag
     0 is the largest. It is checked, with every section without Kronos, before any Kronos work
     (also with `--estimate`): a misaligned pair fails before any model load, estimate, record or
     fan.
   - **Returns:** 1h log returns only where both hours `t` and `t - 1h` exist in both sources:
     Pearson and Spearman correlation, `1 - rho`, std and MAD of `r_bingx - r_binance` in bp and
     their ratio to `std(r_binance)`. Comparator: the lag +-1 h correlations.
   - **Prices:** `(bingx - binance) / binance` in bp for open, high, low and close: signed mean and
     median, mean and median of the absolute value (intervals), p90, p99 and max (descriptive),
     share exactly equal as decimals. Comparators: one price tick (0.1 USDT) in bp, the median
     `|1h return|` in bp, and SPEC section 5's demo-vs-real median of 0.05 bp as an external
     reference.
   - **Volume:** Spearman correlation of hourly volumes (comparator: lag +-1 h) and the median ratio
     BingX/Binance (descriptive: the units may differ).
   - **`amount` proxy (question 9):** on `binance/1h`, `e_t = proxy_t / quote_volume_t - 1` in bp
     (signed mean and median with intervals; `|e|` median, p90, p99, max descriptive); at the Kronos
     sample hours, `dz` = the median and the max over the 512 rows of `|z(proxy) - z(quote_volume)|`
     with the predictor's own normalisation (float32 channels, per-window mean and population std,
     `(x - mean) / (std + 1e-5)` clipped at +-5). Comparator: the same `dz` for the `volume` channel
     between Binance and BingX at those hours. The Markdown lists the section's exclusions (zero
     quote volume, with ranges) and the sample hours skipped, with their reasons.
   - **Kronos signal agreement** (`mismatch --kronos`, after the gate-3 go-ahead, given for the
     primary sample: D-040): `D_k = 2024-07-01T00:00Z + 61 k h` while `D_k < 2026-07-10T08:00Z`
     (291 hours, every hour of the day), both windows complete. Fans: `binance`/`forecast`,
     `bingx`/`forecast` (the same seeds) and `bingx`/`mismatch-floor` (other seeds): 873 fans, about
     1.5 CPU-hours (3.15 s per fan measured on 2026-11-01). Agreements `A_x = agree(binance, bingx)`
     (data effect only, shared seeds), `A_f = agree(bingx, bingx-floor)` (seed effect only) and
     `A_xd = agree(binance, bingx-floor)` (data and seed). The primary statistics are the paired
     differences `A_x - A_f` and `A_xd - A_f` (one bootstrap call each, 4-week blocks).
     Pre-registered reading: an interval of `A_x - A_f` entirely above 0 means "the data effect is
     smaller than seed noise", entirely below 0 "the data effect is larger than seed noise"; an
     interval of `A_xd - A_f` entirely below 0 means "the sources shift the signal beyond seed
     noise" (SPEC section 5: say so), entirely above 0 "the cross-source pair agrees more often than
     the two-seed floor"; an interval containing 0, or no interval, "not distinguishable"; always
     with the minimum detectable difference (about 9 points at 291 hours). The owner's 88 % is
     printed as an external reference. For `p_up`: the correlation and the mean `|delta p_up|` of
     the three pairs, the floor pair as comparator. The lag 1-8 autocorrelation of the agreement
     flags is reported. Fallback if the owner declines the compute (`mismatch --kronos
     --fallback`): a 97-hour step (183 hours, 549 fans, about 0.96 CPU-hours, minimum detectable
     difference about 11 points). A run without `--kronos` rebuilds the Kronos part from the stored
     fans when every fan of a pre-registered sample (primary first, then fallback) is stored under
     the current forecast config. Nothing is computed and gate 3 is not involved; the gate-3
     records that cover the fans are listed. Otherwise the report says the part has not run, with
     the stored counts, and removes `figures/mismatch_agreement.png` and
     `studies/mismatch_kronos.csv` of an earlier report (D-037). A stored fan of a sample that no
     longer matches its window (a store changed after the fan was computed, for example after
     `data binance --refresh`) is never overwritten: the plain run reports the part as not run, its
     reason naming that fan, still writes the report of the current stores and removes the earlier
     report's Kronos figure and series. Move the named fan aside (other fans of the sample may be
     stale too), then `mismatch --kronos` computes the missing fans behind gate 3; `--kronos` itself
     fails on a stale fan (exit 1, nothing written).
   - Figures: return scatter, price-difference histograms, agreement vs floor with intervals.

## Commands

Exit codes: 0 done; 1 failure (`ERROR: ...`); 2 refusal (`REFUSED: ...`) by a governance rule:
gate 3 (above the limit without `--approved-by-owner CPU_HOURS`, or a measured estimate more than
1.25 times that figure), the gate-4 download budget, the sealed partition, the other partition
rules (the Kronos cutoff, the embargo), a changed study config, a busy lock (a store read while a
data run holds its job lock is refused by `forecast` and `mismatch` too). argparse usage errors
(an invalid hour, options of the wrong mode, an approved figure that is not a finite number > 0)
also exit 2. A command stopped by SIGTERM or SIGHUP exits through `SystemExit` (128 + the signal
number) after its clean-up. Data go under the data root, reports under `reports/data/`.

- `kronos-agent data binance [--interval 1m] [--interval 1h] [--refresh <period>]
  [--record-config]`: plan every requested interval (both by default), check gate 4 once for the
  run, then download and verify the archives, then build `binance/1m` and its resamples (1m) and
  `binance_official/1h` (1h), and write the resample check once both 1h stores exist. The check's
  definition is verified before any download.
- `kronos-agent data bingx --interval 1h|15m|5m|1m [--interval ...] [--max-chunks N]`: backfill the
  chunk archive of each interval and rebuild its store; its summary lists the temporary files
  removed and the archived candles kept from the archive alone (a WARNING for those the
  retention does not explain). Each chunk's revisions are a WARNING printed as soon as they are
  recorded, so also when the interval then fails (the archive and the store keep the first-seen
  rows and `raw/bingx/revision-<interval>.json` records each revision once; not a failure). An
  interval whose backfill fails is reported (`ERROR:`) and its store is still rebuilt from the
  chunks archived so far; the remaining intervals run; exit 1 when an interval failed, 0
  otherwise. A busy lock is an immediate refusal.
- `kronos-agent data status`: stores (rows, range, gaps, missing candles, OHLC violations, size and
  sha256 against the meta; exit 1 when a store does not match its meta; a store whose source's lock
  a run holds is listed from its meta as busy, its CSV not checked), the saved archive plans, the
  BingX archive warnings (1m and 5m retention; every revision recorded in
  `raw/bingx/revision-<interval>.json`, or a record that cannot be read; a chunk kept incomplete
  for archived candles the exchange no longer returns; another incomplete chunk older than the
  newest archived one), and the temporary files of killed runs under `candles/` and `raw/` (a
  WARNING; a NOTE for those under a lock a run holds now; the next run of their source removes
  them). Warnings do not change the exit code.
- `kronos-agent data coverage [--record-config]`: study 1.
- `kronos-agent forecast --source binance|bingx --hour <D> [--save]`: one fan in this process
  (`--save` first saves the window's inputs, refusing a different saved window before any model is
  loaded, then stores the fan; a stored fan is reused), or a batch with `--from <D1>`, `--to <D2>`
  (exclusive), `--step-hours H`, `--workers N`, `--estimate` and `--approved-by-owner CPU_HOURS`
  (gate 3).
- `kronos-agent mismatch`: study 3 without computing any fan (the Kronos part is rebuilt from
  stored fans as in the study text; a stale stored fan leaves it not run, naming the fan to move
  aside); `mismatch --record-config` records its definition only;
  `mismatch --kronos [--fallback] --estimate`; `mismatch --kronos [--fallback] [--workers N]
  [--approved-by-owner CPU_HOURS]`.

## Live use (step 8)

The live path loads only the 512-candle live window and one in-process fan (about 1 GB measured);
it never opens the Binance stores or the BingX 1m store, and batches refuse `--workers > 1` when
the machine has less than 4 GiB of RAM. A retried decision may fetch its window again: a saved
window holds its inputs only, so a retry whose fetches matched with another number of pairs keeps
the saved file (the fan records the pairs).

## Not in step 2

No ATR, no event, no trade: they belong to the owner's strategy, which comes after the retraining
(D-042). CVD and delta: Binance klines carry taker-buy volume (kept in the store, unused); BingX's
public klines carry no taker-side volume, so for the evaluation and live source CVD and delta do
not exist (SPEC section 5).
