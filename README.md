# kronos-agent

Research system built to combine **Kronos** (a decoder-only foundation model for candlestick
sequences, [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos), MIT) with the owner's own
trading strategy to trade **BTC-USDT perpetual on a BingX DEMO account**. It was closed on
2026-11-04 (D-063), before any order. The volume-profile criterion of the original specification
was discarded by the owner (D-042); the owner defined a new strategy separately (strategy v1,
D-062), and the work until the close was retraining Kronos for BTC to date (SPEC section 9). The
question it tried to answer, one parameter at a time and with explicit multiple-comparison
corrections, was whether that combination has an edge out of sample and whether the edge survives
a live market. A well-measured "no" is a valid result.

> **Research project. Demo account (virtual funds) only. Nothing here is financial advice.**

The owner's specification (`docs/SPEC.md`, in Spanish) is not published (D-063 (5)): the "SPEC
section" references in the code and the docs point to it, and `CLAUDE.md` summarises its rules.

## Status: closed (2026-11-04)

The owner closed the project on 2026-11-04 (D-063). What it established:

- **The fine-tuning pipeline ran end to end, on one fold.** Kronos-small was fine-tuned on Binance
  BTCUSDT perpetual 1 h candles from 2020-01 to 2024-06 on the development Mac's GPU, as the pilot
  fold of a quarterly walk-forward: 12 pseudo-epochs, the checkpoint chosen by a teacher-forced
  NLL defined in advance on the export's last 8 weeks, held out from training (the 12th
  pseudo-epoch was still the best), 9.3 wall-hours for the refit (D-060, D-061). Its check read
  GO: the fine-tune changed the model's signals (`A_same` 0.75 against the 0.98 threshold).
  Whether it forecasts better was not measured (the check's CRPS figures are descriptive, on the
  hours that chose the checkpoint); that is the walk-forward's test (T-B), which was not run.
- **As a filter for the owner's strategy, Kronos showed no edge.** Two pre-registered readings on
  the validation window (164 trades; net R, the profit in units of the trade's risk after costs;
  90 % calendar-week bootstrap intervals; D-062). With Kronos as shipped, the trades it kept did
  +0.117 R per trade better than the ones it vetoed, an interval of [-0.021, +0.260] that
  contains 0 (the AUC of its expected R against winning trades: 0.580 [0.498, 0.657], where 0.5
  is no information); with the fine-tuned pilot the difference was -0.011 R [-0.175, +0.138] (AUC
  0.499 [0.417, 0.578]). Neither met the registered criterion against the strategy's floor
  filter, the mean of the last 200 daily closes, whose own contrast, +0.070 [-0.089, +0.223],
  contains 0 too; the strategy alone made -0.040 R per trade (-3.25 R a year [-9.43, +2.62]). On
  that window no filter can be told apart from chance.
- **Not done:** the 7 remaining walk-forward folds (about 65 MPS-hours of training) and their
  evaluation (about 36 CPU-hours of forecasts), with the code both need, which was not built; the
  sealed test, shadow mode and any order. The project was never given an exchange key, so it made
  no authenticated connection and sent no order, and the sealed segment was never opened.

Reusable parts: the data layer with its integrity checks (`docs/DATA.md`), the chronological
partition with embargoes and a sealed segment, deterministic forecasts under hashed
configurations, the fine-tuning wrappers around the pinned upstream code (`docs/RETRAINING.md`)
and the hash-chained experiment ledger with its test budget (D-057). The project's own code is
MIT-licensed (`LICENSE`).

At the close (2026-11-04): steps 1 and 2 are done; steps 4 and 5 depended on the discarded
criterion (D-042) and were not built, and step 3's evaluation of Kronos as shipped (SPEC section
8) had become step 6's test T-A (`docs/RETRAINING.md` 6.4), which was not run; step 6, the
retraining, is designed (`docs/RETRAINING.md`), its measurement step ran
(`reports/finetune/timing-2.md`: training on the Mac's GPU, D-056) and the code the pilot needs
is built (D-057, D-058). The pilot, the first fold of the walk-forward, trained (D-060), fired the
pilot rule and was refit at 12 pseudo-epochs, and its check read GO (D-061,
`reports/finetune/pilot-check-56e3d2f9.md`); the remaining folds were not run (D-063). Two tests
of the budget were spent outside the repository's commands: two readings of the owner's strategy
v1 on the validation window with Kronos as a filter, shipped and fine-tuned (D-062). The owner
answered the step-1 design questions the work reached (8, 9 and 12 in D-051; 10 through D-042
and D-048; 1, the license, and 3, the commits, in D-063); 2 and 11 lapsed at the close (the table
at the top of `docs/DECISIONS.md`).

Step 1 (environment, skeleton, safe clients, tests, weight verification) was done on
2026-10-31. Measured on the development machine (darwin-arm64, Apple M4, CPU,
2 threads, torch 2.14.1; `reports/environment/`): a Kronos-small fan of 20 trajectories x 8
candles takes 5.86 s (median of 3 after a warm-up), the model loads in 0.23 s, the benchmark
process peaks at 979 MiB, and the same seed reproduces the fan bit for bit. See
`docs/DECISIONS.md`.

Step 2 (candle stores from Binance's public archives and the BingX real market, exact
resampling, the chronological partition with a sealed segment, decision windows, stored Kronos
fans, the step-2 studies and their commands) is implemented and tested (2026-11-01; design in
`docs/DATA.md`). On 2026-11-02 its volume-profile code left the repository (D-042), the fixes of
an adversarial review were applied (D-043), the data were downloaded and the step-2 studies ran
(`reports/data/`), the Kronos part of the Binance-vs-BingX mismatch study included: the 873 fans
the owner approved on 2026-11-01 (D-040, about 1.5 CPU-hours; the approval covered exactly that
batch), computed under forecast config `b2723b9626339a08`. The owner's answers on the results
are D-051.

## How it is built

- Offline walk-forward over real history decides WHAT runs; the demo decides whether the system
  BEHAVES as the backtest predicted (the design: the demo stage was never built, D-063). Signals
  always use real-market candles (public endpoint, no keys); orders go to the demo host only.
- Kronos never opens a trade on its own; it only modulates what the owner's strategy enables.
- Two exchange clients with no shared signing code (`hmac` exists only in the demo module):
  `exchange/public_market.py` (real market data,
  `open-api.bingx.com`, GET only) and `exchange/demo_account.py` (authenticated, hard-pinned to
  `open-api-vst.bingx.com`: two start-up checks, then one literal-host check at three points of
  every request: pre-send assertion, request hook and pinned transport).
- Kronos is vendored at a pinned commit under `third_party/kronos/` with a sha256 manifest;
  weights are verified against pre-registered sha256 values and loaded offline on CPU with fixed
  threads and seeds derived from the decision hour.
- History is split by date for every source: training up to 2024-06-09, validation from
  2024-07-01 to 2026-07-10, embargoes of 520 h between them, and a sealed segment from 2026-08-01
  that no step-2 code can read (it opens only on the owner's order recorded in the experiment
  ledger, `reports/ledger/experiments.jsonl`, built in step 6: D-057). Every study's definition
  is recorded before it runs, and every interval is a seeded block bootstrap with a named
  comparator.

## Layout

See `docs/ARCHITECTURE.md`.

## Reproduce step 1

```
uv sync --frozen
uv run pytest                                   # weights/network tests skip with a reason
uv run kronos-agent vendor-kronos               # already done; re-verifies the manifest
uv run kronos-agent fetch-weights --model small # Kronos-small + Kronos-Tokenizer-base
uv run pytest                                   # now includes the weights tests
KRONOS_AGENT_NETWORK_TESTS=1 uv run pytest -m network  # opt-in; public hosts only (BingX, Binance)
uv run kronos-agent env-report --network        # one public GET for the clock offset
uv run kronos-agent bench --model small --trajectories 20 --horizon 8 --threads 2 \
    --repeats 3 --threads-sweep 1,2,10 --include-mps
```

The full suite peaks at about 1.75 GiB of RAM (summed over the test process tree while the
two-worker batch test runs, measured on darwin-arm64 on 2026-11-01); on a smaller machine run
`uv run pytest -m "not weights"`. The suite runs on its own temporary roots and never writes
`data/`, `weights/` or `reports/` (D-057 (1)); a session also fails when a real command (the
pilot, `data bingx`) writes there while it runs.

## Reproduce step 2

Data go under `data/` (gitignored; relocatable with `KRONOS_AGENT_DATA_DIR`), reports under
`reports/data/` (the reports root `reports/` is relocatable with `KRONOS_AGENT_REPORTS_DIR`,
D-057 (1)). Only public endpoints are contacted (data.binance.vision and the BingX real
market's public klines); nothing is authenticated. Every study command verifies its recorded
definition (`reports/data/<stem>.config.json`) before reading any candle and refuses a changed
one; `--record-config` records a definition without running the study.

```
uv run kronos-agent data binance                # plan, download, verify, build, resample, check
uv run kronos-agent data bingx --interval 1h --interval 15m --interval 5m --interval 1m
uv run kronos-agent data status                 # stores (size and sha256 vs the meta), warnings
uv run kronos-agent data coverage               # reports/data/coverage
uv run kronos-agent mismatch                    # reports/data/mismatch, no fan computed
uv run kronos-agent mismatch --kronos --estimate    # gate-3 estimate of the 873 fans
uv run kronos-agent mismatch --kronos --workers 4 --approved-by-owner 1.53  # D-040
uv run pytest tests/test_no_lookahead.py -rA    # must pass with verified weights and zero skips
```

- `data binance` refuses (exit 2), before any download, a run whose estimate for all its
  intervals plus the bytes already under `raw/binance` would pass the 4.5 GB budget (about 0.15 GB
  expected). The BingX real market keeps 1m and 5m candles for about 450 and 48 days only, so
  `data bingx --interval 5m` ran at least monthly (`data status` warns); the archive no longer
  grows after the close (D-063 (4)).
- `data status` checks every store's size and sha256 against its meta (exit 1 on a mismatch) and
  lists the gaps, the archive plans, the BingX warnings (1m and 5m retention; every revised candle
  recorded in `raw/bingx/revision-<interval>.json`; a chunk kept incomplete for archived candles
  the exchange no longer returns; an incomplete chunk a backfill left behind) and the temporary
  files of killed runs (the next `data binance` or `data bingx` run removes them). A revised
  candle (an archived candle the exchange later serves with other values) keeps its first-seen
  row in the archive and the store; `data bingx` reports it as a WARNING and goes on (D-045).
- `--approved-by-owner 1.53` is the figure of CPU-hours the owner approved for the Kronos part
  (D-040: 873 fans x 3.15 s x 2 threads, about 5,500 CPU-seconds). A measured estimate more than
  1.25 times it is refused before any fan, and the run appends its gate-3 record to
  `reports/data/mismatch.gate3.jsonl` before its first fan. A plain `mismatch` computes no fan: it
  rebuilds the Kronos part from the stored fans once a whole sample is stored. A stored fan that
  no longer matches its window (a store changed after the fan was computed) is never
  overwritten: `forecast` and `mismatch --kronos` fail naming it, a plain `mismatch` reports the
  Kronos part as not run naming it, and moving the fan aside lets the next batch recompute it.
- Since the phase-1 retraining edits (D-052) every forecast configuration differs from the one
  the 873 fans were computed under: a plain `mismatch` would now report its Kronos part as not
  run and remove its figure and series, and `mismatch --kronos` would have to compute the fans
  again under the new configuration. Neither is rerun without the owner's decision.
- Exit codes: 0 done, 1 failure (`ERROR:`), 2 refusal by a governance rule (`REFUSED:`: gate 3,
  the download budget, the sealed segment, the Kronos cutoff or the embargo, a changed study
  config, a busy lock: a concurrent run, or a store read while a data run writes it).
- `kronos-agent forecast --source bingx --hour 2025-03-03T12:00Z [--save]` runs one fan;
  `forecast --source S --from D1 --to D2 [--step-hours H] [--workers N] [--estimate]` runs a batch
  behind the same gate 3: above one CPU-hour it needs `--approved-by-owner CPU_HOURS`, the figure
  the owner approved, and a measured estimate more than 1.25 times it is refused. Each
  spawn-started worker loads its own Kronos-small (about 1 GB); more than one worker needs at
  least 4 GiB of RAM.

Credentials (never needed for step 1) are read only from the environment variables
`BINGX_API_KEY` and `BINGX_API_SECRET`.

## Retraining (step 6)

The work of D-042 until the close (D-063): fine-tuning Kronos-small for BTC to date (SPEC section
9). Design: `docs/RETRAINING.md`, with the owner's answers in D-048. Phase 1 is built and tested
(D-052): the fold plan and the fold CSV exporter, the training child that runs the pinned
`finetune_csv` predictor phase behind the design's wrappers, NLL4-TF early stopping, the checkpoint
registry and `load_models(checkpoint=...)`, evaluation lengths up to 24 candles, and the measurement
step. The measurement step ran twice. The first run (2026-11-02, `reports/finetune/timing.md`,
D-053) measured the CPU only: batch 32 needs activation checkpointing (9.68 s per step at 2 threads;
8 threads are no faster), and its MPS preflight ran out of memory without checkpointing. The second
run, under the owner's gate-3 approval of up to 1.5 CPU-hours (D-055; it spent 3,038 CPU-seconds),
chose the Mac's GPU (`reports/finetune/timing-2.md`, D-056): MPS at 5.449 s per step at batch 32
with activation checkpointing, theta 1.004, against 8.790 s for the best CPU configuration; a 20 x 4
fan takes 3.06 s. With these numbers the 24 monthly folds would take about 116 h, above the 72 h of
D-048 (1), so the walk-forward runs the 8 quarterly folds; the owner approved up to 10 h for the
pilot, fold `q-2024-07`, and its check (D-056), up to 10 more for its refit once the pilot rule
fired (D-059) and up to 1.1 CPU-hours for the whole pilot check (D-061 (4)). Phase 3a is built and
checked (D-057, D-058): the experiment ledger and the test budget, the tokenizer diagnostic, the
pilot check, the fair CRPS and the fold command. Part A of the tokenizer diagnostic and the pilot
ran (D-060: 5.306 wall-hours, all 6 pseudo-epochs with the last one selected, so the pilot rule
fired), then the refit at 12 pseudo-epochs (9.262 wall-hours) and the pilot check, which read GO
(D-061: `A_same` 0.75 against 0.98). The project closed before the compute control, the bundle and
the walk-forward code were built and the remaining folds trained (D-063). No test of this plan was
spent; the two readings of D-062 spent 2 outside it.

```
uv run kronos-agent finetune tokenizer-diag --part training   # Part A, a few minutes of CPU
uv run kronos-agent finetune fold --fold q-2024-07 --estimate  # runs nothing
caffeinate -i -s uv run kronos-agent finetune fold --fold q-2024-07 --approved-by-owner 10  # D-056
# if the pilot rule fires, it records the pilot and refuses the refit; the refit is then:
caffeinate -i -s uv run kronos-agent finetune fold --fold q-2024-07 --approved-by-owner 10  # D-059
uv run kronos-agent finetune pilot-check --estimate            # after the refit; runs nothing
uv run kronos-agent finetune pilot-check                       # estimated 0.89 CPU-hours
# it stopped at 3,622 charged CPU-seconds; finishing it took the owner's figure (D-061 (4)):
uv run kronos-agent finetune pilot-check --approved-by-owner 1.1  # reused the fans; 3,726 in all
uv run kronos-agent ledger show                                # entries, head, tests spent
uv run kronos-agent ledger verify                              # every line and every bundle
```

- `finetune fold --fold LABEL [--replicate K] [--estimate] [--approved-by-owner HOURS] [--order
  ID]` trains one quarterly fold (`wf-` labels are refused). It prints its estimate first (on
  `timing-2.json`: 4.945 MPS-hours at `E_max` 6, as for the pilot, and 9.779 at `E_max` 12, as
  for its refit and, since the pilot rule fired, for every later fold) with the smallest figure
  that admits each unit and the deadline each figure gives. Above one hour it needs
  `--approved-by-owner HOURS`, the figure the owner approved in chat, refuses an estimate more
  than 1.25 times it, records the approval in the experiment ledger before the
  first unit and starts nothing that could take the run's spend past 1.25 times it. Folds after
  the pilot wait for the pre-registration bundle and are blinded. It refuses to start while a
  training child is still running on the machine. Run it on an otherwise idle Mac, on power,
  lid open, under `caffeinate -i -s`, started so that the end of the terminal session cannot
  kill it.
- `finetune pilot-check [--record-config] [--estimate] [--approved-by-owner CPU_HOURS]` runs the
  pilot check of design 6.2 (504 fans and the KL on the registered pilot; 0 tests). It stops
  between fans, or before or between its two teacher-forced passes, once its spend passes 1.25
  times the approved figure, or 3,600 CPU-seconds without one (exit 2, the stored fans kept and
  the stopped run recorded); its reading, GO or NO-GO, exits 0.
- `finetune tokenizer-diag --part training|validation [--record-config]` has no `--estimate`: it
  prints its measured estimate and runs at once when that is within gate 3, and is refused above
  it. Part B (`validation`) runs only after the pre-registration bundle. A raised flag is printed
  as a `WARNING:`; when the token-mismatch flag is raised (more than 5 % of context rows whose s1
  or s2 token changes, D-048 (5)), the owner is told before the bundle is recorded.
- `kronos-agent ledger show [--kind KIND] | verify | record-order`: the experiment ledger,
  `reports/ledger/experiments.jsonl` (tracked and published; append-only, hash-chained, verified
  on every read, never repaired). `record-order` records the owner's order to open the sealed
  segment and is run only after that order is given in chat (gate 6).
- `finetune timing [--estimate] [--approved-by-owner CPU_HOURS]`, the measurement step (synthetic
  candles only, 0 tests), is not run again: `timing-2.*` is the record every fold estimate reads
  and refuses once a training-code file changes, the approval of D-055 covered that run only, and
  a rerun would overwrite the record (it needs a new gate-3 approval). The first run's
  `reports/finetune/timing.*` hold another plan and stay as they are.
- Exit codes: 0 done; 1 failure (`ERROR:`: a fold whose training failed for good, a broken ledger
  or registry, a measurement error); 2 refusal (`REFUSED:`: gate 3, the spend cap or stop, a stale
  timing record, a missing or changed pre-registration bundle, the layout, a registered fold, the
  sealed segment, a changed study config, a busy lock). A bare `--approved-by-owner`, or a figure
  that is not a finite number > 0, is a usage error (exit 2).
- Fine-tunes run only through `kronos-agent finetune`, one child process
  `python -B -P -m kronos_agent.finetune.runner` each. Runs, scratch and checkpoints go under
  `weights/finetuned/` (the registry is `registry.jsonl`), fold exports under `data/finetune/`;
  both are gitignored and relocatable. The other subcommands of `docs/RETRAINING.md` (c)
  (`register`, `walk-forward`, `evaluate`, `report`, `sealed`, `final`, `card`) were not built.

## License

The project's own code is released under the MIT License (`LICENSE`), approved by the owner on
2026-11-04 (D-063). Kronos keeps its own MIT license and attribution under `third_party/kronos/`;
see `third_party/README.md` for the citation.
