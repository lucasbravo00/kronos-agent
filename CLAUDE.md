# CLAUDE.md - kronos-agent

Read this file and `docs/SPEC.md` (the owner's full specification, in Spanish) at the start of
EVERY session, before anything else. The SPEC is the only source of requirements; this file is the
operational summary plus the facts that are easy to get wrong. The SPEC stays on the development
machine only: it is gitignored and not published (D-063 (5)), so a clone without it has this file
and `docs/DECISIONS.md` as the record of its rules.

## Status: closed (D-063)

The owner closed the project on 2026-11-04 (`docs/DECISIONS.md` D-063). Nothing runs and nothing
is scheduled. Any further work, run, download or look at data needs the owner's explicit order;
read D-063, D-061 (6) and D-062 (8) first.

## Hard rules (SPEC section 1) - they survive every session

1. Kronos NEVER opens a trade on its own, not even at 100 % probability. It only modulates
   (confirms, vetoes or sizes) what the owner's criterion enables. "Kronos only" exists solely as a
   research control, without orders.
2. Demo only. The authenticated client is hard-pinned to the host `open-api-vst.bingx.com`, refuses
   to start with any other host and validates every request. Nothing is ever signed towards
   `open-api.bingx.com`. The owner's real futures account must stay at zero.
3. Signals and volume indicators ALWAYS use candles of the REAL market (public GET, no keys).
   Orders go to the demo account. Two separate clients, with no shared signing code.
4. Never edit `third_party/`. Never read or print `.env`. `.env`, `weights/`, `data/` and `.venv`
   never go to git.
5. No parameter change is applied live without the owner's explicit approval. No push to any
   remote without the owner's approval.
6. Every result is reported with intervals, comparators, seeds and the number of tests performed.
   A well-measured negative result is worth as much as a positive one. No metric dressing.
7. Event definitions and the configuration of every test are recorded BEFORE looking at
   validation data. The sealed partition is opened once per candidate, on a recorded order from
   the owner.
8. One variable at a time. Test budget: 3 per week, 15 per quarter, enforced by the command
   itself. Changing `goals.yaml` counts as a test.

## Gates (SPEC section 17) - stop and ask the owner before

1. The first authenticated connection to the account (the read-only check included).
2. The first order (and before it, >= 2 weeks of shadow mode completed).
3. Runs of more than ~1 hour of CPU or GPU (show the estimate first; USD cost if GPU).
4. Downloading models other than Kronos-small and Kronos-Tokenizer-base, or installing anything
   outside `pyproject.toml`. The BTCUSDT download from data.binance.vision is pre-approved; warn
   before exceeding ~5 GB.
5. Parameter or model changes in live.
6. Opening the sealed partition.
7. Push or publication to any remote (GitHub, Hugging Face or other).
8. Attempting N4.
9. Freezing `goals.yaml`, and any later change to it.

## Language and workflow

- Talk to the owner in Spanish (conversation, summaries, questions, warnings).
- Everything inside the repository is English: code, identifiers, file and folder names, comments,
  docstrings, logs, errors, commands, commits, README and docs. The only exception is
  `docs/SPEC.md`, the owner's original document, which stays verbatim and is never published:
  nothing published is in Spanish (D-063 (5)).
- Design questions are collected in ONE batch at the end of step 1 (SPEC section 18); until
  answered, the initial values of `params.yaml` apply. Decisions and the owner's answers live in
  `docs/DECISIONS.md`.
- Every step ends with a short Spanish summary with numbers. A step does not start before the
  previous one meets its "done when" (SPEC section 18).

## Scope until the close (owner directive of 2026-11-02, D-042; closed by D-063)

- The owner discarded the volume-based criterion of SPEC section 4. `docs/SPEC.md` stays verbatim,
  but section 4, the volume-profile and POC parts of section 5 and every step of section 18 that
  depends on the criterion are without effect. Never
  build anything that only serves a trading criterion (levels, volume profile, events, order
  signals) without the owner's explicit go-ahead; when a SPEC item looks outdated, ask first.
- Until the close the work was SPEC section 9 in full: retrain (fine-tune) Kronos-small for BTC
  to date, with the data, partition, reproducible forecasts and evaluation it needs, plus the
  Binance vs BingX mismatch study without its POC part.

## Another session's folder

- `research/strategy/` is where the owner, in a separate session, defined their new strategy
  (strategy v1, D-062 (1)). Never read, modify or delete anything in it; `research/` is
  gitignored and excluded from tests and lint. This repository registers no event definition of
  it and does not freeze `params.yaml[criterion]` (it still describes the discarded criterion).

## Secrets

- Keys live only in `.env` (mode 600, service user), created and loaded by the owner. The agent
  never reads, prints, greps, hashes or sources it, and never runs a command that loads it (no
  `uv run --env-file`, no `dotenv`, no `env`/`printenv` dumps). `.claude/settings.json` denies
  `Read(.env)`, `Read(.env.*)`, `Edit(.env)` and `Edit(.env.*)`.
- Code reads credentials only from the environment variables `BINGX_API_KEY` and
  `BINGX_API_SECRET`, only inside `src/kronos_agent/exchange/demo_account.py`, never logs them,
  and `repr()` prints `<redacted>`. Locally the owner exports them in their own shell
  (`set -a; source .env; set +a; uv run ...`); on the server systemd's `EnvironmentFile=` does it.
- A signed query (signature, timestamp) never reaches a log: `demo_account` drops `httpx` records
  naming the demo host (D-011). Never log a signed request's URL, and never enable local-variable
  capture around the demo client (pytest `--showlocals`, crash reporters that record locals).
- No IPs, hostnames, usernames, emails, account ids, balances or personal paths anywhere in the
  repo, in its history or in generated reports. `tests/test_repo_hygiene.py` enforces part of it,
  on the working tree's text files only: IPv4 literals, e-mail addresses, home-directory path
  prefixes, the running login name, hostname and home directory, and credential assignments.
  Commit metadata, history, image metadata, account ids and balances are not scanned: check them
  by hand before any push (D-063 (5)).

## Environment and determinism

- `uv` only, with its own Python 3.12 (`.python-version`); never the conda base or any system
  Python; run everything as `uv run ...`. Nothing is installed outside `pyproject.toml` (gate 4);
  `uv python install` on a new machine is part of "uv with its own Python" (SPEC section 16), not
  an extra install.
- Kronos is vendored at commit `67b630e67f6a18c9e9be918d9b4337c960db1e9a` in `third_party/kronos/`
  (MIT, see `third_party/README.md`). `manifests/kronos_source.sha256` is verified before every
  import (`kronos_agent.kronos.vendored`). The upstream package is imported under its own
  top-level name `model`; no project package may use that name. The import writes no bytecode
  and any `__pycache__`/`.pyc` under `third_party/kronos` fails verification (D-014): run
  anything that imports the vendored code outside `import_kronos()` (the step-6 training child,
  upstream tests) with `python -B`.
- Weights live under `weights/` (relocatable with `KRONOS_AGENT_WEIGHTS_DIR`) as local directories
  (`config.json` + `model.safetensors`), are verified against the pre-registered
  `manifests/weights.json` (size + sha256) before loading (`manifests/weights.sha256` is a derived
  GNU manifest for `shasum -a 256 -c`) and are loaded with `HF_HUB_OFFLINE=1` set before any
  import: call `kronos_agent.runtime.configure_runtime()` before importing `torch` or
  `huggingface_hub`. Downloads only through `kronos-agent fetch-weights`, which verifies sha256
  against the pre-registered `manifests/weights.json`.
- Comparable inference: CPU only, `torch.set_num_threads(2)` (the server's vCPU count),
  `device="cpu"` passed explicitly, both models in `.eval()`, deterministic algorithms on. MPS is
  not a reference device (documented nondeterministic ops and an open `multinomial` correctness
  bug); it may only be used for training (step 6) and for informational benchmarks.
- Seeds derive from the sha256 of the decision hour (`kronos_agent.seeds`); every forecast
  records seed, threads, torch version, model name and weight hashes; stored fans also record the
  numpy and pandas versions (in the config) and the `context_sha256` of the floats the model read.
- Data lives in `data/` (relocatable with `KRONOS_AGENT_DATA_DIR`). Both folders are gitignored.
  Reports live in `reports/` (tracked; relocatable with `KRONOS_AGENT_REPORTS_DIR`, which the
  test suite sets for every test), the experiment ledger in `reports/ledger/experiments.jsonl`.
  A scratch run relocates all three roots: with only the data or weights root relocated it
  would append to the repository's ledger, which is never repaired (D-058 (8)).

## Kronos facts verified at the pinned commit (do not "fix" them from memory)

- Call `.eval()` on the tokenizer and the model yourself: the pinned `huggingface_hub==0.33.1`
  does not call it on the safetensors path, `KronosPredictor` does not either, Kronos-small has
  dropout 0.1/0.25/0.25/0.1 and cross-attention uses `is_causal = self.training`.
- `predict(sample_count=K)` returns the MEAN of K sampled paths. The fan of K trajectories is
  `predict_batch([df]*K, [x_ts]*K, [y_ts]*K, pred_len=N, sample_count=1)`.
- `KronosPredictor(device=None)` selects MPS on Apple silicon: always pass `device="cpu"`.
- `load_models` builds every predictor with `PREDICTOR_CLIP` = 5 (upstream's default).
  `forecast_window` and the forecast configuration refuse a predictor with another clip, context
  length or device, and a tokenizer or model with any submodule in training mode or any parameter
  requiring gradients (D-044).
- `kronos/weights.py` and `kronos/vendored.py` (the models' construction) are hashed whole into
  the forecast `config_id` with the rest of the forecasting code (D-047): any edit of them gives a
  new id, and only `load_models` builds the models of stored fans (`load_models(checkpoint=...)`
  records a fine-tuned checkpoint's own hashes, so its fans get their own `config_id`).
- Normalisation statistics use every row of the DataFrame passed: pass exactly `max_context`
  rows (512 for small/base).
- `top_k > 0` disables `top_p`: keep `top_k = 0`.
- If `amount` is missing and `volume` is present, the predictor sets
  `amount = volume * mean(open, high, low, close)` per row (zeros only when `volume` itself is
  missing). `finetune_csv` reads `amount` verbatim, so training CSVs must carry the same proxy.
- There is no seed argument: seed `torch`, `numpy` and `random` before every forecast.
- `max_context` is not in the HF `config.json`: 512 for small and base, 2048 for mini; the
  tokenizer pairing (small/base -> Tokenizer-base, mini -> Tokenizer-2k) is enforced by code, not
  by weight shapes.
- The HF parameter totals (small 24,741,696; Tokenizer-base 3,958,258) count every tensor in
  `model.safetensors`, i.e. `state_dict()` including persistent buffers, never `.parameters()`.

## Exchange facts (BingX, verified live on 2026-10-29)

- Public klines `GET /openApi/swap/v3/quote/klines` (no signature): objects with
  `open, close, high, low, volume, time` as strings (no quote volume), NEWEST FIRST, the current
  unclosed candle included first; send `startTime` AND `endTime` (`startTime` alone returns an
  empty list); `limit` is effectively capped at 1000 anchored at `endTime`; `code != 0` is an
  error (100410 = rate limit, observed at ~0.8 req/s), never "no data"; pace <= 0.5 req/s.
- Klines bounds (observed 2026-10-31 by `tests/test_public_market_live.py`): `startTime` is
  honoured and inclusive at an exact candle boundary (the first page starts at `startTime`: 49
  candles, 48 closed plus the open one); a page seam `endTime = oldest_open - 1` returns the
  preceding candle as the newest of the next page (1100 closed hours in 2 pages, no duplicate, no
  gap). Observed 2026-11-01 by `tests/test_bingx_history_live.py` (D-041): an `endTime` exactly at
  a candle's open excludes that candle; `endTime = next chunk start - 1` returns a whole
  1000-candle chunk in one page; a range entirely before the history answers `code 0` with `[]`.
  The public client's `closed_klines` and `fetch_twice` select the half-open `[start_ms, end_ms)`
  whatever the exchange does at that edge (D-047).
- Real 1h history starts 2024-04-27T10:00Z; 1m starts ~2025-07-31 and 5m ~2026-09-14 (rolling
  retention of about 450 and 48 days, `RETENTION_DAYS`; the first download on 2026-11-02 measured
  45 days of 5m and 100 days of 15m and an unchanged 1m start, to be recorded with D-049, D-050):
  the archive was refreshed until the close and no longer grows (D-063 (4)); `data bingx` runs
  only on the owner's order (`data status` warns once the newest archived 1m or 5m candle is a
  third of its retention old). The `raw/bingx` chunk archive
  is the only copy of what rolled off: never delete it. The owner archives 1m and 5m as insurance
  for their future strategy; the retraining does not use them (D-049).
- Backfill rules (D-027): epoch-aligned chunks of 1000 candles; `endTime` is never an exact
  candle boundary (that excludes the candle opening there; `boundary - 1` returns the whole
  chunk) and never in the future; a candle is kept only when `open + step + 90 s <= server time`
  (read once per run); each chunk is fetched twice and the settled candles must be byte-identical
  (at most 3 pairs); complete chunks are never fetched again; archived candles are never
  overwritten or deleted. An incomplete chunk's refetch adds its new settled candles, keeps the
  archived ones it no longer returns (listed in the sidecar's `archive_only`) and keeps the
  first-seen version of a revised one (other strings for an archived open: the owner's decision,
  D-045); each revision is recorded once in `raw/bingx/revision-<interval>.json`, before its
  chunk is written, with its first and last observation; the run continues, `data bingx` prints
  it as a WARNING as soon as it is recorded (so also when the interval then fails) and
  `data status` lists it. A chunk whose files were removed (the only repair of an inconsistent
  one) is fetched again as a new chunk, with the first-seen rows its recorded revisions hold
  (D-050). A refreshed chunk completes only when the 1m or 5m retention explains its
  archive-only candles (never for 1h or 15m). The history start is recorded in
  `raw/bingx/history-<interval>.json` only when a walk established it.
- Demo host `https://open-api-vst.bingx.com` only (no `.pro` fallback). Signing: HMAC-SHA256 hex
  of the ASCII-sorted, unencoded `key=value&...` string including `timestamp`; header
  `X-BX-APIKEY`.
- Hedge mode: every order carries `positionSide` (BingX defaults a missing one to LONG); never
  send `reduceOnly`; `clientOrderId` only for LIMIT/MARKET; positions `code 109500` means the
  query failed, not "flat".
- Fees taker 0.05 % / maker 0.02 %; funding every 8 h at 00/08/16 UTC for BTC-USDT (read
  `fundingIntervalHours` from `premiumIndex` at runtime).

## Binance Vision facts (data.binance.vision, observed 2026-11-01)

- USD(S)-M perpetual BTCUSDT klines: monthly 1m archives from 2020-01; the 2019 monthly archives
  return 404 (every plan probes the 2019 dailies); the month just ended has its daily archives
  days before its monthly one. One 12-column CSV per zip, with a header line in recent files and
  none in 2020 files; price strings change format (`7225` and `7171.10` in one 2020 file); keep
  them verbatim. `.CHECKSUM` is `<sha256 hex>  <zip name>`; every zip is verified against it.
- The download is pre-approved (gate 4): `data binance` plans every requested interval and
  refuses (exit 2), before any download, a run whose total estimate plus the bytes under
  `raw/binance` would pass 4.5 GB (about 0.15 GB expected). If the host is unreachable, tell the
  owner (SPEC section 5).

## Data layer (step 2, `docs/DATA.md`)

- Studies and window sources read candles only through `read_study_frame`; nothing in step 2 can
  open the sealed segment (candles opening from 2026-08-01T00:00Z): a `SealedAccess` exists only
  through `partition.sealed_access`, once the experiment ledger (`reports/ledger/experiments.jsonl`,
  D-057 (4)) holds the owner's order and may open that data root (`ledger.binding_problem`: a
  relocated reports root never opens the repository's `data/`); it opens only the data root it
  was issued for (D-058 (3)(c)). `read_store` is for integrity work only.
- Every study verifies its recorded definition (`reports/data/<stem>.config.json`) before reading
  a candle; a changed definition is refused, never overwritten (a changed study gets a new stem).
- A stored fan that no longer matches its window (a store changed after the fan was computed) is
  never overwritten: `forecast` and `mismatch --kronos` fail naming it, a plain `mismatch` reports
  the Kronos part as not run naming it, and the fan is moved aside to recompute it (D-047).
- Gate 3 is enforced by the batch commands: they print the measured estimate and refuse above
  3600 CPU-seconds unless `--approved-by-owner CPU_HOURS` gives the CPU-hours the owner approved
  for the estimate shown, and refuse a measured estimate more than 1.25 times that figure (D-040).
  Run `--estimate` first, show the estimate to the owner, and pass the figure they approve only
  after their explicit go-ahead. `mismatch --kronos` records the approval in
  `reports/data/mismatch.gate3.jsonl` before its first fan.
- Every store read holds its source's job lock (`raw/binance/.lock`, `raw/bingx/.lock`) shared and
  parses the bytes it verified; a read during a `data binance`/`data bingx` run is refused
  (`StoreBusyError`). Runs take the lock through `kronos_agent.data.store.job_lock`, which first
  removes the temporary files a killed run left; `data status` lists them and checks every store's
  sha256.
- Spawn-started batch workers each verify the vendored tree and load their own Kronos-small
  (about 1 GB each); more than one worker needs at least 4 GiB of RAM (refused otherwise): one
  worker on the 1.9 GB server.
- The phase-1 retraining edits of `kronos/weights.py` and `kronos/decision.py` changed every
  forecast `config_id` (D-052 (3)): the 873 D-040 fans keep `b2723b9626339a08`, so a plain
  `mismatch` would now report its Kronos part as not run and remove its figure and series. Do not
  rerun it without the owner's decision.

## Retraining (step 6, `docs/RETRAINING.md`; do not "fix" these from memory)

- Fine-tune runs go only through `kronos-agent finetune` (`kronos_agent.finetune.train`): one
  child `python -B -P -m kronos_agent.finetune.runner --run-dir <dir>` per run, with an
  allow-listed environment (no credential reaches it) and the vendored tree verified before and
  after. Never run `train_sequential.py` or `finetune_base_model.py` directly. The child runs the
  pinned predictor phase unmodified behind the wrappers of design 3.1 (D-052 (1)).
- Only a recipe run (every wrapper on, started from the shipped weights, the state NLL4-TF
  selected) is registered, in `<weights>/finetuned/registry.jsonl`: append-only and
  hash-chained, but its last line is protected only by the `line_sha256` its users record, so a
  checkpoint is named as `CheckpointRef(id, root, line_sha256)` wherever its use is recorded.
- The pinned `finetune_csv` behaves as follows, and the design works around it without editing
  it (design 3.1 and 10 cite the lines): each training window is normalised over ALL its rows,
  future rows included; in eval mode the s2 cross-attention is NOT causal
  (`is_causal = self.training`), so the vendored validation loss reads its target; that loss is
  also random (s2 is conditioned on an s1 drawn by `torch.multinomial`); there is NO early
  stopping (every epoch runs and the lowest vendored validation loss is kept), no MPS path and no
  date split. NLL4-TF (design 3.3) is the stopping metric.
- Measured on this Mac (D-053): batch 32 needs activation checkpointing (CPU peak 9.2 GiB with
  1.5 GiB of swap growth without it, 3.2 GiB with it, gradients bit-identical); 8 CPU threads
  are no faster than 2 at batch 32 (10.6 against 10.4 s per step). The first MPS preflight ran
  at batch 32 without checkpointing and ran out of MPS memory during its checks (which one is
  not recorded). The second run (D-055; `reports/finetune/timing-2.*`, D-056) started MPS at
  batch 32 with activation checkpointing and measured 5.449 s per step (theta 1.004) against
  8.790 s for the best CPU configuration, so training runs on MPS; every fold estimate reads
  that record, and refuses it once a training-code file changes.
- Never edit `finetune.train.TRAINING_CODE_FILES` (`finetune/recipe.py`, `export_csv.py`,
  `nll.py`, `runner.py`, `train.py`, `kronos/features.py`, `data/windows.py`,
  `manifests/kronos_source.sha256`) without the owner's decision: an edit changes the training
  code sha256 (`f28c0f30...`), every fold estimate then refuses `timing-2.json` as stale, and
  the measurement step must run again (a new gate-3 ask).
- The walk-forward runs the 8 quarterly folds (D-048 (1): the monthly worst case, about 116 h,
  passes 72 h; D-056). The pilot is fold `q-2024-07` (its CSV is `wf-2024-07`'s, training-segment
  data only); the owner approved up to 10 h for it and its check (D-056). Its estimate was 4.945
  MPS-hours (9.779 for the refit); at `--approved-by-owner 10` the pilot's invocation recorded
  the pilot and refused the refit, as foreseen (the refit could start in it only if the pilot
  spent at most 2.72 MPS-hours, 12.5 - 9.779), and the refit ran in its own invocation, the
  same command again with `--approved-by-owner 10`, under the owner's 10 MPS-hours (D-059 (1);
  its deadline 45,000 s, 27.8 % above its worst case). The pilot ran on 2026-11-03 (5.306
  wall-hours, a median of 6.03 s per step on a Mac in use) and the rule fired (D-060); the refit
  ran overnight on the idle Mac (9.262 wall-hours, 5.236 s per step, no thermal slowdown) and
  the pilot check read GO (D-061), so the pilot is done and `E_max` is 12 for every fold. Apps
  in use slowed a step by about 15 % on the median and by up to about 30 % over 50-step
  intervals on this fanless Mac (the pilot's 6.03 s median and 6.8 s slowest intervals against
  the idle refit's 5.24 s), so every fold needs an otherwise idle Mac. The pilot check took 1.17
  times the wall its estimate implies, and its spend control stopped it at 3,622 of 3,600
  CPU-seconds; the owner approved up to 1.1 CPU-hours for the whole check, those 3,622
  included (D-061 (4); 3,726 in all). Start every fold so that the end of a terminal session
  cannot kill it: a parent killed with SIGKILL leaves its child training unrecorded, and
  `finetune fold` then refuses to start (D-058 (3)(e)).
- The experiment ledger `reports/ledger/experiments.jsonl` (`kronos_agent.ledger`, D-057 (2) to
  (4)) is append-only and hash-chained, verified on every read and never repaired or rewritten.
  Appends take `reports/ledger/.lock` (gitignored; the ledger itself is tracked). A line's `ts`
  never precedes the last one, so after a clock that ran ahead is set back, appends are refused
  until real time passes the last entry; its last line is protected only by the line sha256 its
  users record. Only the repository's `reports/ledger` opens the sealed segment of, spends a test
  on, or admits the bundle-gated reads (Part B, the folds after the pilot) of the repository's
  `data/` (`ledger.binding_problem`). Its lines refuse personal data, IPv4 literals, absolute
  paths, BingX host names and whatever the hygiene scan of `reports/` refuses. `ledger
  record-order` runs only after the owner's explicit order in chat (gate 6), with their words
  translated. The test budget is `budget.spend_test`; no command spends a test yet.
- Two readings of the owner's strategy v1 with Kronos as a filter ran outside the repository on
  2026-11-03 (shipped) and 2026-11-04 (the pilot). They used the validation window, 166
  decisions, and neither met the registered criterion against the strategy's floor filter
  (D-062). They spent 2 tests the ledger does not count (2026-W45: 2 of 3; 2026-Q4: 2 of 15):
  `ledger show` and `budget.spend_test` do not see them; how to record them lapsed at the close
  (D-063 (4)), so a reopened project counts them by hand (at most 1 more test in 2026-W45 and 13
  in 2026-Q4). They saw validation
  outcomes before the bundle, so the T-A and T-B reports and the bundle must state them as a
  deviation (`docs/RETRAINING.md` 6.3, 7.1). Another reading of strategy v1 on that window, with
  or without Kronos, and any further statistic on the 332 fans those readings stored outside the
  repository, needs the owner's explicit order.
- Activation checkpointing on MPS (torch 2.14.1): `preserve_rng_state` restores only the CPU
  generator for the recomputation, so the recomputed dropout masks would differ from the
  forward's; `runner.enable_activation_checkpointing` replays the MPS generator through
  `context_fn` (`_mps_generator_contexts`). M2's bit-identity checks only the CPU: on MPS the
  wrap rests on `test_checkpointing_reproduces_the_unwrapped_training_steps[mps]` (D-054 (5)).
- MPS on torch 2.14.1: a `PYTORCH_MPS_HIGH_WATERMARK_RATIO` below 1.4 needs
  `PYTORCH_MPS_LOW_WATERMARK_RATIO` at or below it, or every allocation fails (`finetune.train`
  sets both); `.to("cpu", torch.float64)` of an MPS tensor returns NaN, so move it to the CPU
  first.
- The predictor reduces its normalisation statistics on a Fortran-ordered float32 array; a
  C-ordered reduction differs in the last bits, so code that re-normalises "exactly as
  inference" reduces the same way (`finetune.nll.normalise`).
- The training child's memory guard reads `sysctl` (macOS only): the server runs with
  `Compute(memory_guard=False)`.

## Git

- Remote: a public GitHub repository (`origin`), published on 2026-11-04 on the owner's order
  (gate 7) with a history rewritten to the owner's GitHub no-reply address and UTC dates, without
  `docs/SPEC.md` (D-063 (5)); the original private repository, pushed on 2026-11-03, was renamed
  and kept private. Commit and push only when the owner asks; every push still needs their
  approval, and before it check the commit identity (no personal e-mail). Show `git status` in
  step summaries.
- Code provenance does not rely on the commit: `provenance.py` records the source tree sha256,
  which identifies the code whatever the number of commits.

## Commands

`uv sync --frozen` - `uv run pytest` (markers `weights` and `network` skip with an explicit reason
when prerequisites are absent; network tests are opt-in via `KRONOS_AGENT_NETWORK_TESTS=1` and
touch public hosts only) - `uv run kronos-agent env-report | verify | fetch-weights | bench |
vendor-kronos | params`. The full suite peaks at about 1.75 GiB of RAM (1789 MiB summed over the
test process tree while the two-worker batch test runs; measured 2026-11-01, step 1 was 1254 MiB):
on the 1.9 GB server run `uv run pytest -m "not weights"` until the 4 GB swap exists.

The suite never writes the real roots (D-057 (1), `tests/root_guard.py`): the autouse fixture
`isolated_roots` points `KRONOS_AGENT_DATA_DIR`, `KRONOS_AGENT_WEIGHTS_DIR` and
`KRONOS_AGENT_REPORTS_DIR` at each test's own temporary roots; the `weights` marker and
`real_roots("data", "weights", "reports")` give read-only views of the real ones; a write under a
real root is refused and fails its test; and the session fails (exit 1) when `data/`,
`weights/` or `reports/` changed during it, also through a real command run meanwhile (the
pilot, `data bingx`): by design, so do not run the suite while one of those writes. A test that
calls `cli.main` uses stand-ins and temporary roots (D-055 (6)). `pytest --noconftest` bypasses
the guard; `KRONOS_AGENT_TEST_ROOTS_LISTING=<file>` writes the start listing.

Step 2: `kronos-agent data binance | bingx --interval I | status | coverage`, `forecast --source S
--hour D [--save]` or `--from D1 --to D2 [--step-hours H] [--workers N] [--estimate]
[--approved-by-owner CPU_HOURS]`, `mismatch [--kronos [--fallback] [--workers N] [--estimate]
[--approved-by-owner CPU_HOURS]]` (full synopsis in `docs/DATA.md`, "Commands"); `--record-config`
records a study's definition without running it. Exit codes: 0 done, 1 `ERROR:`, 2 `REFUSED:`
(gate 3, the gate-4 budget, the sealed segment, the Kronos cutoff or embargo, a changed study
config, a busy lock). Data under `KRONOS_AGENT_DATA_DIR` (default `data/`), reports under
`reports/data/` (the reports root relocates with `KRONOS_AGENT_REPORTS_DIR`).

Step 6 (`docs/RETRAINING.md`): `kronos-agent finetune timing [--estimate] [--approved-by-owner
CPU_HOURS]`, the measurement step (synthetic candles, 0 tests; `reports/finetune/timing-2.json`
and `.md`, D-054). The first run's `reports/finetune/timing.*` hold another plan and are never
rewritten: a run whose plan differs from the recorded one is refused, so a changed plan takes a
new stem (D-034); the budget is not part of the plan (D-055). It prints its estimate first,
refuses to start above its budget and stops once its measured spend passes it: 3,000
CPU-seconds and 45 GPU-minutes, below gate 3, or with `--approved-by-owner` a CPU limit of
CPU_HOURS x 3600 CPU-seconds, a ceiling without tolerance (the GPU limit stays 45 minutes); a
bare flag or a figure that is not a finite number > 0 exits 2; `--estimate` runs nothing. Pass
the figure only after the owner's explicit go-ahead for it, as for the batch commands. It ran
twice; `reports/finetune/timing-2.*` (D-056: MPS, batch 32 with activation checkpointing, 5.449
s per step, theta 1.004) is the record every fold estimate reads, and the owner's 1.5 CPU-hours
(D-055) covered that run only: never rerun it without a new gate-3 approval (a rerun overwrites
`timing-2.*`). Its leftovers under `weights/finetuned/` were deleted (D-056). Exit codes: 0
done; 1 `ERROR:` (an M9 mismatch, a crashed training child or a measurement error, after the
report is written; or a run that could not start); 2 `REFUSED:` (an estimate above the budget, a
run its budget stopped (after the report is written), a changed timing plan, a concurrent
timing run). Fold exports go under `<data>/finetune/`; runs, scratch and checkpoints under
`<weights>/finetuned/`.

Phase 3a (D-057, D-058); every `finetune` subcommand but `timing` first verifies every recorded
pre-registration bundle (a changed or missing one exits 2) and records what it ran in the
ledger. Exit codes as D-039: 0 done, 1 `ERROR:`, 2 `REFUSED:`.
- `finetune fold --fold LABEL [--replicate K] [--estimate] [--approved-by-owner HOURS] [--order
  ID]`: one quarterly fold (`wf-` and `final-` labels exit 2; quarterly replicates exit 1 until
  their mapping is decided); the pilot is `--fold q-2024-07`. It prints the estimate on
  `timing-2.json` (refused when stale), the figure lines (the smallest figure that admits each
  unit, the deadline each figure gives) and the run label's earlier fit entries; `--estimate`
  stops there. Gate 3 counts the chosen device's hours (MPS wall-hours, as `timing-2` chose
  MPS): above one hour it needs `--approved-by-owner HOURS`, the figure the owner approved in
  chat, and refuses an estimate more than 1.25 times it; the approval is recorded in the ledger
  before the first unit, and nothing starts that could take the invocation's spend past 1.25
  times the figure (one hour without one). Folds after the pilot need the bundle and are
  blinded; a sealed fold needs `--order` (gate 6). Refused (exit 2) also: a training child still
  running on the machine, a running timing step, a registered fold other than the pilot (a
  registered and recorded pilot has nothing to run), a stage that moved since the estimate. A
  fold whose training failed for good (a second failure, a memory or time abort, a killed child)
  exits 1 after its `fit` entries. Run it on an otherwise idle Mac, on power, lid open, under
  `caffeinate -i -s`, started so that the end of the terminal session cannot kill it.
- `finetune pilot-check [--record-config] [--estimate] [--approved-by-owner CPU_HOURS]`: 504 fans
  and the KL on the registered pilot (estimated at about 0.89 CPU-hours, below gate 3, so its
  first run needs no figure; the run of D-061 (4) passed 3,600 charged CPU-seconds and stopped,
  and finishing it took `--approved-by-owner 1.1`: 3,726 CPU-seconds in all); `--record-config`
  records its definition only. Gate 3 as for the batch commands, with the approval in the
  ledger before the first fan. It stops between fans, or before or between its two
  teacher-forced passes, once its charged spend passes 1.25 times the figure, or 3,600
  CPU-seconds without one (exit 2, the stored fans kept, the stopped run and its spend in the
  ledger); a rerun counts what the stopped runs spent.
  The reading is GO or NO-GO, both exit 0; without a registered and recorded pilot it exits 2.
- `finetune tokenizer-diag --part training|validation [--record-config]`: no `--estimate`; it
  prints its measured estimate and runs at once when that is within gate 3 (a few minutes of
  CPU), and is refused above it (exit 2). Part A (`training`) records its definition at its
  first run, and its report the sha256 of `docs/RETRAINING.md`; Part B (`validation`) runs only
  after the bundle, in a ledger that may read the data root (exit 2 otherwise). Each raised flag
  prints as a `WARNING:`; the `r1_token_mismatch` flag (more than 5 % of context rows with an s1
  or s2 token changed) means telling the owner before the bundle (D-048 (5)).
- `kronos-agent ledger show [--kind KIND] | verify | record-order --scope open_sealed --candidate
  TEST_ID --text "<owner's words>"`: `verify` exits 1 on a broken or unreadable ledger and 2 on
  a changed or missing bundle; `record-order` (only after the owner's explicit order in chat)
  exits 2 for a candidate without a T-B reading, a second order, personal data in the text, a
  clock earlier than the last entry or a busy lock, and prints the order id, the seq and the
  line sha256.
