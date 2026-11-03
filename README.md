# kronos-agent

Research system that combines **Kronos** (a decoder-only foundation model for candlestick
sequences, [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos), MIT) with the owner's own
trading strategy to trade **BTC-USDT perpetual on a BingX DEMO account**. The volume-profile
criterion of the original specification was discarded by the owner (D-042); the new strategy is
defined separately and comes after the current work, retraining Kronos for BTC to date (SPEC
section 9). The question it tries to answer, one parameter at a time and with explicit
multiple-comparison corrections, is whether that combination has an edge out of sample and whether
the edge survives a live market. A well-measured "no" is a valid result.

> **Research project. Demo account (virtual funds) only. Nothing here is financial advice.**

Status (2026-11-03): steps 1 and 2 are done; steps 3 to 5 depended on the discarded criterion
(D-042) and wait for the owner's new strategy; step 6, the retraining, is designed
(`docs/RETRAINING.md`) and its measurement step ran (`reports/finetune/timing-2.md`, D-055), but
no model has been trained yet and no test of the budget has been spent. The owner answered the
step-1 design questions (D-051; the table at the top of `docs/DECISIONS.md`).

Step 1 (environment, skeleton, safe clients, tests, weight verification) was done on
2026-10-31. Measured on the development machine (darwin-arm64, Apple M4, CPU,
2 threads, torch 2.14.1; `reports/environment/`): a Kronos-small fan of 20 trajectories x 8
candles takes 5.86 s (median of 3 after a warm-up), the model loads in 0.23 s, the benchmark
process peaks at 979 MiB, and the same seed reproduces the fan bit for bit. See `docs/SPEC.md`
(the owner's specification, Spanish) and `docs/DECISIONS.md`.

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
  BEHAVES as the backtest predicted. Signals always use real-market candles (public endpoint, no
  keys); orders go to the demo host only.
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
  ledger of step 5). Every study's definition is recorded before it runs, and every interval is a
  seeded block bootstrap with a named comparator.

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
`uv run pytest -m "not weights"`.

## Reproduce step 2

Data go under `data/` (gitignored; relocatable with `KRONOS_AGENT_DATA_DIR`), reports under
`reports/data/`. Only public endpoints are contacted (data.binance.vision and the BingX real
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
  expected). The BingX real market keeps 1m and 5m candles for about 450 and 48 days only: run
  `data bingx --interval 5m` at least monthly (`data status` warns).
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

The current work (D-042): fine-tuning Kronos-small for BTC to date (SPEC section 9). Design:
`docs/RETRAINING.md`, with the owner's answers in D-048. Phase 1 is built and tested (D-052): the
fold plan and the fold CSV exporter, the training child that runs the pinned `finetune_csv`
predictor phase behind the design's wrappers, NLL4-TF early stopping, the checkpoint registry
and `load_models(checkpoint=...)`, evaluation lengths up to 24 candles, and the measurement
step. Nothing has been trained and no test has been spent. The measurement step ran once, on
2026-11-02 (`reports/finetune/timing.md`, D-053): the CPU at 2 threads, batch 32 with activation
checkpointing, 9.68 s per step (8 threads are no faster; without checkpointing batch 32 peaked at
9.23 GiB and swapped); fans take t(L) = -0.107 + 0.795 L seconds; the identity control passed.
Its MPS preflight ran out of memory at batch 32 without activation checkpointing, so MPS was
never measured in a configuration that fits. The step is repeated before any gate-3 estimate
(D-054): the CPU is measured first, MPS at batch 32 starts with activation checkpointing where
the CPU's batch 32 needs it, and an MPS memory abort without it is retried once with it. With
MPS its estimate, about 3,578 CPU-seconds, is above the step's own 3,000-second limit; the owner
approved up to 1.5 CPU-hours for this second run (gate 3, D-055), given with
`--approved-by-owner 1.5`.

```
uv run kronos-agent finetune timing --estimate --approved-by-owner 1.5  # runs nothing
caffeinate -i -s uv run kronos-agent finetune timing --approved-by-owner 1.5  # D-055
```

- Synthetic candles only, 0 tests. Run it on an otherwise idle Mac, on power, lid open. It
  prints its estimate first, refuses to start above its budget and stops once its measured spend
  passes it. The budget is 3,000 CPU-seconds and 45 GPU-minutes, below gate 3;
  `--approved-by-owner CPU_HOURS`, the CPU-hours the owner approved in chat for the estimate
  shown, makes the CPU limit CPU_HOURS x 3600 CPU-seconds, a ceiling without tolerance: the
  estimate must fit, and the run stops at, 600 CPU-seconds less, a reserve for the unit running
  then, so the spend stays within the ceiling (the GPU limit stays). The budget is not part of
  the plan, and the record states the approval: the CPU-hours, the command's start as the time
  the flag was given, the ceiling and the stop limit, the estimate, and a limit above gate 3
  under that approval. It writes `reports/finetune/timing-2.json` and `.md` with its plan
  in `timing-2.config.json` (D-054). The first run's `reports/finetune/timing.*` hold another
  plan and stay as they are: a run whose plan differs from the recorded one is refused, so a
  changed plan takes a new stem. Later estimates refuse a record made with another device, torch
  version, platform or training code.
- Exit codes: 0 done; 1 failure (an M9 mismatch, a crashed training child or a measurement
  error, after the report is written, or a run that could not start); 2 refusal (an estimate
  above the budget, a run its budget stopped, a changed timing plan, a concurrent run). A bare
  `--approved-by-owner`, or a figure that is not a finite number > 0, is a usage error (exit 2).
- Fine-tunes run only through `kronos-agent finetune`, one child process
  `python -B -P -m kronos_agent.finetune.runner` each. Runs, scratch and checkpoints go under
  `weights/finetuned/` (the registry is `registry.jsonl`), fold exports under `data/finetune/`;
  both are gitignored and relocatable. The other subcommands of `docs/RETRAINING.md` (c)
  (`fold`, `walk-forward`, `evaluate`, `report`, ...) are not built yet.

## License

MIT proposed for the project's own code, pending the owner's approval. Kronos keeps its own MIT
license and attribution under `third_party/kronos/`; see `third_party/README.md` for the citation.
