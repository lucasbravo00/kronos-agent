# Retraining Kronos-small for BTC (step 6, SPEC section 9)

How Kronos-small is fine-tuned on BTC, how the recipe is validated by walk-forward against Kronos
as shipped, how the sealed segment and the model trained to date are handled, and what each part
costs. SPEC sections 1, 3, 6, 8, 9, 12, 14, 16 and 17 are the requirements. D-042 (owner,
2026-11-02) sets the scope: a tokenizer diagnostic on BTC, a predictor-only fine-tune with the
pinned `finetune_csv` and early stopping on validation, walk-forward validation of the recipe
against Kronos as shipped by CRPS, then the model trained to date with its model card. Data
access, decision windows, stored fans and studies follow `docs/DATA.md`. Nothing here reads,
defines or uses the trading criterion of SPEC section 4 (D-042).

The owner answered the six questions of this design on 2026-11-02 and took every recommended
option: the answers are D-048 in `docs/DECISIONS.md`, summarised in section (b), and the sections
below apply them. The design decisions of this file get new numbers in `docs/DECISIONS.md`, after
the last existing one. D-042 (3) supersedes the question-10 default "best-validation checkpoint
instead of early stopping": this design keeps the best checkpoint AND stops early.

Status (2026-11-02): design decided, implementation in progress. Nothing has been trained for it
and no test has been spent; the measurement step ran once (D-053) and is repeated with the fix of
D-054, under the owner's gate-3 approval of up to 1.5 CPU-hours (D-055). The step-2 data it
reads are in place: the Binance and BingX stores, the coverage study and the mismatch study with
its 873 fans (`docs/DATA.md`).

Step-6 "done when" (SPEC section 18; the shipped-model part is step 3's):
- `reports/finetune/ta-<recipe8>.md` (test T-A): the SPEC 8 evaluation of Kronos as shipped,
  with the decision tree written with intervals.
- `reports/finetune/wf-<recipe8>.md` (test T-B): the walk-forward of recipe R1 against Kronos
  as shipped by CRPS, its pre-registered reading and the SPEC 8 re-measurement for R1.
- `reports/finetune/model_card_<checkpoint>.md`: the model card of the model trained to date.

## Summary

- **Recipe R1.** The pinned predictor phase of `finetune_csv` (`SequentialTrainer.
  train_basemodel_phase` and `train_model`) runs unmodified in a `python -B` child process, on
  one pre-cut Binance 1h CSV per fold. It uses lr 1e-6 (the template's), AdamW and OneCycleLR
  as vendored, batch 32, windows of 513 rows (lookback 512, predict 0) and the vendored
  normalisation (D-048 (5)). The project deviates only where a rule forces it (table in 3.1).
- **Budget per fold.** Pseudo-epochs of 512 optimizer steps (16,384 windows), at most 6 of them
  (3,072 steps, the OneCycle span), identical in every fold. A pre-registered pilot rule doubles
  the cap once.
- **Early stopping.** NLL4-TF: a deterministic, causal, teacher-forced negative log-likelihood
  of the next 4 candles in the exact inference geometry, on the fold's inner validation (the
  CSV's last 1,344 hours, 336 decisions), with patience 2. The shipped weights are measured,
  never selected.
- **Walk-forward.** 24 monthly cold-start folds, test months 2024-07 to 2026-06. Each fold's CSV
  ends 512 h before its test month: the per-model reading of SPEC 6 (D-048 (6)), equal to the
  partition at both ends. Quarterly folds (8, over the same test decisions) are the
  pre-registered compute fallback, taken when the measured worst case passes 72 h (D-048 (1)).
  Fold 1 is the pilot and uses training-segment data only. One training seed per fold, plus two
  replicate seeds in the folds wf-2024-07, wf-2025-06 and wf-2026-06 (D-048 (4)).
- **Evaluation.** BingX real windows, CPU, 2 threads, deterministic, the same seeds for both
  models. Primary statistic: `Delta_N = mean CRPS(R1) - mean CRPS(shipped)` of the N-hour log
  return, N = 4, over the 4,380 decisions of the 4-hour grid, with a 95 % block bootstrap over
  4-week blocks (27 blocks). SPEC 8 skill curves h = 1..24 for both models on the 25-hour grid
  (701 decisions), with a simultaneous band over h <= 8. The mismatch study triggered the
  pre-registered Binance copy of the primary fans, a T-B secondary (5.2).
- **Acceptance.** The interval of `Delta_N` lies entirely below 0, `Delta_N < 0` in at least 5
  of the 8 calendar quarters, and the relative improvement is at least 1 % (D-048 (2)). Only
  then the sealed test (gate 6), on one recorded order of the owner that also covers training
  the model to date (D-048 (3)), read against a band derived from validation. The model to date
  is produced whatever the reading, with its model card.
- **Tests.** T-A (shipped, SPEC 8), T-B (R1 walk-forward) and T-C (sealed, only after
  acceptance): at most 3 of the 15 per quarter. None is spent yet. The ledger and the budget are
  enforced by the commands.
- **Compute (priors until the measurement step).** Training runs on the Mac's GPU (MPS) after
  the measurement step (D-048 (1)). MPS at 3.2 s per step at batch 32 gives about 46 h typical
  and 69 h worst case for 24 folds, or 15 to 23 h for 8 quarterly folds; the seed replicates add
  at most 17.3 h. The evaluation needs 14,542 fans, 36.0 CPU-hours, plus 8,760 fans and 15.3
  CPU-hours for the Binance copy. Every run above about one hour waits for a gate-3 approval of
  a measured estimate.

## Conventions

The conventions of `docs/DATA.md` apply (UTC, half-open ranges, decision hour `D`, strings as the
record, correctly rounded floats, never fabricated, durable writes, provenance, reports). In
addition:

- **Candle boundaries of the partition.** The partition is by decision hour, but a sample at `D`
  reads candles up to `D + (h - 1) h`. The last candle a training-segment sample may read opens
  before `VALIDATION_START - 512 h = 2024-06-09T16:00Z`; the last a validation sample may read
  opens before `SEALED_START - 512 h = 2026-07-10T16:00Z` (`sample_fits`).
- **Fold.** A fold is one fine-tune and its test month (or quarter) starting at `T`, 00:00Z on
  the 1st. Labels: `wf-YYYY-MM` (monthly), `q-YYYY-MM` (quarterly), `s-YYYY-MM` (sealed),
  `final-YYYY-MM-DD`; replicates add `-r1`, `-r2`.
- **Grids.** `G4`: decision hours with `D.hour % 4 == 0` (non-overlapping at N = 4). `G25`:
  decision hours whose hour count since 2020-01-01T00:00Z is divisible by 25 (non-overlapping up
  to h = 24, and it walks through every hour of the day). `G8` (contingency): `D.hour % 8 == 0`.
- **Streams.** `forecast`: the live seeds, used by both models (common random numbers).
  `forecast-aa`: a second shipped stream for the A/A check and the two-seed floor. Fan seeds come
  from `D`, the stream and the project symbol (`kronos_agent.seeds`).
- **Vendored citations.** The vendored tree is pinned, so its line numbers are stable:
  `FB` = `third_party/kronos/finetune_csv/finetune_base_model.py`, `TS` = `.../train_sequential.py`,
  `CL` = `.../config_loader.py`, `CFG` = `.../configs/config_ali09988_candle-5min.yaml`,
  `K` = `third_party/kronos/model/kronos.py`, `M` = `third_party/kronos/model/module.py`,
  `HM` = huggingface_hub's `hub_mixin.py` (0.33.1). Project code is cited by module and function,
  because its lines move.
- **SPEC quotations** are translated from the Spanish of `docs/SPEC.md`.

## 1. Sources and folds

### 1.1 Source per role

| Role | Source | Read through | Why |
| --- | --- | --- | --- |
| Fit (gradient steps) | Binance USD(S)-M BTCUSDT 1h, resampled from 1m (`candles/binance/1h`) | `read_study_frame` (strings), exported to the fold CSV | SPEC 5: Binance trains; D-042 (4) |
| Early stopping (NLL4-TF) and the vendored validation diagnostic | the same store, the fold's last 1,344 hours before the CSV end | NLL4-TF: `read_study_frame` and `build_window`; diagnostic: the vendored split of the CSV | selection is part of training and stays on the fit source |
| Pilot check (0 tests) | Binance 1h, fold 1's inner-validation hours | `StoreWindowSource` | training segment only |
| Test | BingX real 1h (`candles/bingx/1h`) | `StoreWindowSource` through the batch layer | SPEC 5: BingX real is the evaluation and live source |
| Test, secondary (T-B) | Binance 1h, the same `G4` hours and seeds (the D-040 contingency, triggered: 5.2) | `StoreWindowSource` through the batch layer | separates "the model improved" from "the source changed" (D-040, D-042 (4)) |
| Comparator | Kronos as shipped, same BingX windows, same seeds | same | D-042 (3) |
| Cutoff check (in T-A) | Binance 1h, decisions in 2024-01 to 2024-12 | NLL4-TF | SPEC 3: "verify what you can" about the June 2024 cutoff |
| Tokenizer diagnostic | Part A: Binance training segment; Part B: Binance and BingX validation | `StoreWindowSource` | SPEC 9.1 |

`history: all` means from 2020-01-01T00:00Z (`TRAINING_START`). The 2019 Binance months exist but
stay out, so the first fold reads only candles the partition assigns to training.

### 1.2 Folds of the walk-forward

For a fold whose test month starts at `T`:

- **CSV rows:** Binance candles opening in `[2020-01-01T00:00Z, T - 512 h)`.
- **Inner validation:** the CSV's last 1,344 rows, `[T - 1,856 h, T - 512 h)` (8 weeks): 832
  vendored validation windows of 513 rows, and 336 NLL4-TF decisions `D` on the 4-hour grid with
  `T - 1,856 h <= D <= T - 516 h`, so that the 4 targets `D .. D + 3 h` open before the CSV end.
- **Fit rows:** the CSV rows before `T - 1,856 h`; fit windows = fit rows - 512.
- **Test decisions:** the `G4` and `G25` hours in `[T, next T)`.

Counts assume no gap, which the coverage study (`reports/data/coverage.md`) confirms for
`binance/1h` and `bingx/1h` before the sealed start (the one Binance gap, 2026-10-28, is a sealed
date); a gap only reduces them (1.6).

| Fold | CSV end (excl.) | Inner validation from | Fit windows | G4 | G25 |
| --- | --- | --- | --- | --- | --- |
| wf-2024-07 (pilot) | 2024-06-09T16:00Z | 2024-04-14T16:00Z | 37,064 | 186 | 30 |
| wf-2024-08 | 2024-07-10T16:00Z | 2024-05-15T16:00Z | 37,808 | 186 | 29 |
| wf-2024-09 | 2024-08-10T16:00Z | 2024-06-15T16:00Z | 38,552 | 180 | 29 |
| wf-2024-10 | 2024-09-09T16:00Z | 2024-07-15T16:00Z | 39,272 | 186 | 30 |
| wf-2024-11 | 2024-10-10T16:00Z | 2024-08-15T16:00Z | 40,016 | 180 | 29 |
| wf-2024-12 | 2024-11-09T16:00Z | 2024-09-14T16:00Z | 40,736 | 186 | 29 |
| wf-2025-01 | 2024-12-10T16:00Z | 2024-10-15T16:00Z | 41,480 | 186 | 30 |
| wf-2025-02 | 2025-01-10T16:00Z | 2024-11-15T16:00Z | 42,224 | 168 | 27 |
| wf-2025-03 | 2025-02-07T16:00Z | 2024-12-13T16:00Z | 42,896 | 186 | 30 |
| wf-2025-04 | 2025-03-10T16:00Z | 2025-01-13T16:00Z | 43,640 | 180 | 29 |
| wf-2025-05 | 2025-04-09T16:00Z | 2025-02-12T16:00Z | 44,360 | 186 | 29 |
| wf-2025-06 | 2025-05-10T16:00Z | 2025-03-15T16:00Z | 45,104 | 180 | 29 |
| wf-2025-07 | 2025-06-09T16:00Z | 2025-04-14T16:00Z | 45,824 | 186 | 30 |
| wf-2025-08 | 2025-07-10T16:00Z | 2025-05-15T16:00Z | 46,568 | 186 | 30 |
| wf-2025-09 | 2025-08-10T16:00Z | 2025-06-15T16:00Z | 47,312 | 180 | 28 |
| wf-2025-10 | 2025-09-09T16:00Z | 2025-07-15T16:00Z | 48,032 | 186 | 30 |
| wf-2025-11 | 2025-10-10T16:00Z | 2025-08-15T16:00Z | 48,776 | 180 | 29 |
| wf-2025-12 | 2025-11-09T16:00Z | 2025-09-14T16:00Z | 49,496 | 186 | 30 |
| wf-2026-01 | 2025-12-10T16:00Z | 2025-10-15T16:00Z | 50,240 | 186 | 30 |
| wf-2026-02 | 2026-01-10T16:00Z | 2025-11-15T16:00Z | 50,984 | 168 | 26 |
| wf-2026-03 | 2026-02-07T16:00Z | 2025-12-13T16:00Z | 51,656 | 186 | 30 |
| wf-2026-04 | 2026-03-10T16:00Z | 2026-01-13T16:00Z | 52,400 | 180 | 29 |
| wf-2026-05 | 2026-04-09T16:00Z | 2026-02-12T16:00Z | 53,120 | 186 | 30 |
| wf-2026-06 | 2026-05-10T16:00Z | 2026-03-15T16:00Z | 53,864 | 180 | 29 |
| **Total** | | | | **4,380** | **701** |

The vendored training mapping `start = (idx * 9973 + (10000 * epoch + 1) * 104729) mod n`
(FB:99-102, FB:112-114, FB:273) is a bijection only when 9973 does not divide `n`; otherwise only
`n / 9973` windows are ever drawn. No planned count is a multiple of 9973 (24 folds, 832
validation windows, the sealed folds, and a final model on any day from 2026-08-01 to
2027-03-31). The exporter recomputes it on the real, gap-reduced counts and refuses a multiple.

### 1.3 Other runs of the same recipe

- **Pilot = fold `wf-2024-07`.** Its CSV ends at 2024-06-09T16:00Z, the last candle a
  training-segment sample may read. Every row it fits or early-stops on is training-segment data,
  where exploration is free and logged (SPEC 7). Its checkpoint serves as fold 1 if the recipe is
  unchanged after the pilot.
- **Excluded tail.** 2026-07-01 to 2026-07-10T08:00Z, the rest of the validation segment, would
  need a 25th fine-tune for 56 `G4` decisions (1.3 %). It is excluded, and the exclusion is
  pre-registered.
- **Quarterly fallback.** `T` in {2024-07-01, 2024-10-01, 2025-01-01, 2025-04-01, 2025-07-01,
  2025-10-01, 2026-01-01, 2026-04-01}, each tested on its 3 months, with the same CSV and
  inner-validation rules. It covers the same 4,380 `G4` and 701 `G25` decisions; its first fold
  is the pilot.
- **Sealed folds** (only after the owner's recorded order, 6.6). One per complete calendar month
  before the order: `T` = 2026-08-01 (CSV ends 2026-07-10T16:00Z, exactly `SEALED_START - 512 h`,
  readable without a `SealedAccess`), 2026-09-01, 2026-10-01 and 2026-11-01 (their CSVs read
  sealed candles). Test decisions: `G4` 186, 180, 186, 180; `G25` 30, 29, 30, 28.
- **Model to date.** CSV `[2020-01-01, T_final)`, where `T_final` is 00:00Z of the run day,
  capped at the last complete day the Binance store holds. Inner validation: its last 1,344
  rows. No test follows, so there is no test embargo. Without an accepted recipe, `T_final` is
  2026-08-01T00:00Z (D-048 (3)).

### 1.4 Boundaries

1. **Fit and inner validation: no gap.** The vendored split happens before windowing
   (FB:75-97, FB:48), so no window crosses it. NLL4-TF targets are inner-validation rows; their
   contexts may contain fit rows, which are past inputs at decision time.
2. **CSV and test: 512 h.** SPEC 6 says a date used in training with any source does not appear
   in validation. Walk-forward requires reading this per model: later folds train on earlier
   validation months, but never on their own test samples, contexts included. The CSV's last row
   opens at `T - 513 h` and the first test context starts at `T - 512 h`. The `+ N` of SPEC 6's
   embargo is 0 here because a training window's targets are CSV rows. This boundary equals the
   partition's candle boundary at both ends (fold 1 and the first sealed fold). Cost: the
   evaluated models are 21 days staler than a model retrained to the day it goes live, a
   conservative bias. The owner chose this reading: D-048 (6).
3. **Test targets.** N = 4 fans read targets `D .. D + 3 h`, 24-step fans `D .. D + 23 h`.
   Inside the validation segment no purge between folds is needed, because the next fold is
   another model. Every test decision (`D < 2026-07-01`) passes `sample_fits(D, L)` for
   `L <= 24`; the boundary is 2026-07-10T16:00Z.
4. **Validation and sealed.** The 520 h second embargo is unchanged. No validation-phase read
   reaches 2026-07-02.
5. **Model periods.** The shipped weights forecast only `D >= 2024-07-01`. A checkpoint forecasts
   only `D >= data_end` (its CSV end), and the walk-forward planner also asserts
   `D >= data_end + 512 h` for every test decision of the fold.

### 1.5 The `amount` proxy, exactly as at inference

- The CSV's `amount` is the float64 `volume * mean(open, high, low, close)` computed by
  `kronos.features.add_amount_proxy` on `store.float_values` of the store's strings: the functions
  the inference window uses under policy `volume_x_mean_ohlc`, bit-identical to the predictor's
  own expression (K:608-609). It is written with `repr()`. The other columns are the store's
  decimal strings verbatim.
- The vendored reader parses with pandas' default float parser (FB:53), which D-023 forbids for
  candle values because it is not correctly rounded, then casts to float32 (FB:122), as the
  predictor does (K:620). The exporter therefore checks every row: the float32 array the vendored
  reader produces from the CSV must equal, bit for bit, the float32 of the inference path's
  `context_frame`. One difference refuses the CSV. (A 1-ulp float64 error changes the float32 only
  at a rounding tie, so the expected count is 0; the check makes it a fact.)

### 1.6 Gaps

- Stores never fill gaps; a Binance hour with any missing minute is a gap. The CSV holds only
  real candles, so the vendored dataset would build windows across a gap with a time jump inside
  (FB:107-127).
- When a fold CSV has a gap, the child uses `GapAwareKlineDataset`, a subclass of the vendored
  `CustomKlineDataset`. Its `__init__` calls the vendored one, then keeps the window starts whose
  513 rows span exactly 512 hours. Its `__getitem__` repeats the vendored body (FB:118-132) with
  the vendored index mappings (FB:112-116) restricted to those starts. Without a gap the vendored
  class runs as is. A run on a CSV with a gap and the `gap_aware` wrapper off (3.1 row 8) is
  refused, by the parent before a run directory exists and by the child, and the child checks
  its datasets' window counts against the export's whichever class runs.
- The export record lists every gap and the windows dropped. A fold that drops more than 5 % of
  its fit windows is refused and the owner is told. NLL4-TF decisions whose context or targets
  miss a candle are dropped and counted; a fold with fewer than 168 of its 336 is refused.
- BingX test hours with an incomplete window are dropped and listed by the batch layer; both
  models are always scored on the same hours.

### 1.7 The fold CSV

`finetune.export_csv.export_fold(fold, *, sealed_access=None)`:

1. `read_study_frame(StoreId("binance", "1h"), start_ms=2020-01-01, end_ms=csv_end,
   numeric=False)`. The sealed guard applies: a CSV end after 2026-08-01T00:00Z needs a
   `SealedAccess`.
2. Columns `timestamps, open, high, low, close, volume, amount`, where `timestamps` is the open
   time as tz-naive `YYYY-MM-DD HH:MM:SS`. Labels by open time give the same time features as
   inference (FB:60-64, K:472-479; D-029).
3. Checks before publishing: strictly increasing open times; no row at or after the CSV end; the
   float32 equality of 1.5; the gap scan of 1.6; `n_fit` = rows opening before
   `csv_end - 1,344 h`; `n_val = N - n_fit >= 514` (FB:108-110 needs more rows than `W`); the 9973
   condition; and `int(N * (n_fit + 0.5) / N) == n_fit`, the vendored split arithmetic.
4. Publish `data/finetune/<recipe8>/<fold>/train.csv` (durable write, D-024) and `export.json`:
   rows, first and last open times, `n_fit`, `n_val`, `train_ratio`, gaps, dropped windows, the CSV
   sha256, the store meta sha256 and the code provenance (D-025).

### 1.8 The sealed segment

- The validation phase never reads a candle opening at or after 2026-07-02. Fold CSVs end by
  2026-05-10T16:00Z; BingX targets end at 2026-07-01T21:00Z (the 24-step skill-curve fan of the
  last `G25` decision, 2026-06-30T22:00Z), and at 2026-06-30T23:00Z for the 4-step primary fans.
- Sealed folds run only with a `SealedAccess` from `sealed_access(order_id)`, which today always
  raises. The experiment ledger of section 7 replaces `partition._verify_order` with a lookup of
  the owner's recorded order.
- The model to date is trained on sealed candles only after the sealed test has used them
  (SPEC 9.5: "sealed segment included, only after it has been used").

## 2. Windows and normalisation

### 2.1 Window length

- Inference reads exactly 512 context rows and normalises over them.
- A vendored window has `W = lookback + predict + 1` rows (FB:33), and the loss covers all
  `W - 1` positions (FB:288-292); `predict_window` only lengthens `W`. The template's
  `512 + 48 + 1 = 561` trains 560 positions, 48 of them attending to more than 512 tokens, which
  never happens at inference. The template's 48 was chosen for 5-minute stock bars.
- R1 uses `lookback_window: 512` and `predict_window: 0`, so `W = 513`: 512 trained positions,
  exactly the inference context. This is a configuration value (the "window" SPEC 9.2 records),
  not a code change.

### 2.2 Normalisation: the vendored one in R1

- The vendored dataset normalises each window over all its rows (FB:122-127): float32, mean and
  population std, `(x - mean) / (std + 1e-5)`, clip +-5. Inference normalises over the 512
  context rows only (K:620-631) and de-normalises the generated rows with those statistics
  (K:657).
- **What this leaks.** Over the 513 rows of a window, each channel's z-values sum to zero before
  the clip. So the target of position 511, row 512, is minus the sum of the 512 input z-values,
  and every earlier position carries the matching "the rest of the window sums to minus what came
  before" information. This is a look-ahead inside training windows and a train/inference
  mismatch. Its size is not bounded by how little one row moves the mean and std. It is limited
  by tokenisation, because the model sees 10+10-bit codes rather than z-values, and only a
  measurement can size it.
- **What it does not do.** It is not test leakage: every evaluation fan uses the inference
  normalisation, so the walk-forward CRPS measures its net effect honestly.
- **Why R1 keeps it.** SPEC 9.2 asks for "the repo's `finetune_csv`, unedited" in the first run,
  and the step-1 default for question 10 is "scripts verbatim first, plus a separate test with
  context-only normalisation". The owner decided so: D-048 (5).
- **Measured before registration.** Part A of the tokenizer diagnostic (section 4) measures the
  train/inference token mismatch on training-segment windows: the share of context rows whose
  tokens differ from the inference encoding under R1's all-row `W = 513`, the template's all-row
  `W = 561`, and context-only `W = 513`. If R1's s1 mismatch exceeds 5 % of rows, the owner is
  told before the pre-registration bundle is recorded (D-048 (5)).
- **Later variants, one variable each.** First, context-only statistics (rows 0..511 applied to
  all rows), through the same dataset subclass. Second, context-only statistics plus a loss mask
  on the positions that predict rows after the context. `DualHead.compute_loss` already has a
  `padding_mask` branch (M:494-502), and the loop calls it as `model.head.compute_loss`
  (FB:292, FB:327), so an instance attribute on `model.head` installs the mask without editing
  anything. That variant needs a longer window (for example 561 rows, positions 511..559 scored).

### 2.3 Other train/inference differences that remain

They cannot be removed without editing `third_party/` and go into the model card.

- **RoPE in the s2 cross-attention.** At inference the s2 query has length 1, so the rotary cache
  is built for one position and nothing is rotated (M:303-308 with K:440). In training, positions
  0..511 are rotated.
- **s2 conditioning.** In training, s2 is conditioned on an s1 drawn at T = 1 without top-p
  (K:267-272). At inference the draw uses the governed T and `top_p` (K:436-442).
- **Causality of the s2 cross-attention.** It is causal only in training mode (M:387). At
  inference the single query attends only to the past, which is correct there.

CRPS on fans runs the real inference code and therefore sees all of them.

## 3. Training (recipe R1)

### 3.1 What runs unmodified, and what the project wraps

The parent command prepares the fold and spawns one child per fine-tune:

```
<venv python> -B -m kronos_agent.finetune.runner --run-dir <weights>/finetuned/.runs/<run_id>
```

with `PYTHONDONTWRITEBYTECODE=1`, the offline and thread variables of `kronos_agent.runtime`
and, for MPS runs, `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (section 3.5). A child process keeps the
patched vendored module globals, the device state and the training RNG out of the parent. The
child:

1. Calls `configure_runtime(threads, deterministic=...)` before importing torch, then
   `import_kronos()`, which verifies the vendored tree.
2. Puts `third_party/kronos/finetune_csv` on `sys.path` with bytecode off, imports the vendored
   `train_sequential` (which imports `finetune_base_model`, `finetune_tokenizer` and
   `config_loader`), and removes the `'../'` entries they append (TS:10, FB:20).
3. Checks that every file the YAML points at hashes to the sha256 the parent took from
   `manifests/weights.json`.
4. Installs the wrappers of the table below.
5. Calls `train_basemodel_phase()` (TS:148-262) directly, on the `SequentialTrainer` subclass of
   row 1 built from the run YAML, not `run_training`, which would turn the early-stop exception
   into a failure.
6. Writes `runner_result.json`: per-pseudo-epoch training loss, the cleaned vendored validation
   loss, NLL4-TF, the selected pseudo-epoch, stop reason, seconds per step, peak memory, device,
   threads, torch version. It exits 0, or 1 with the traceback in the run log.

The parent then verifies the vendored tree again (a `__pycache__` fails it, D-014), verifies the
selected checkpoint and registers it (3.6). The vendored logs and stdout carry absolute paths
(for example TS:233-241); they stay in the gitignored run directory and never reach a report.

| # | Vendored behaviour | This design | Forced by |
| --- | --- | --- | --- |
| 1 | Device is CUDA or CPU (TS:29-38) | A subclass overrides only `_setup_device` | SPEC 16: small is fine-tuned on the Mac with MPS |
| 2 | `torch.multinomial` samples s1 on the training device from the global RNG in every forward (K:271) | Routed to the CPU with the fold's own `torch.Generator`, on every device | The open MPS `multinomial` bug (CLAUDE.md); s1 draws independent of dropout's RNG use |
| 3 | Validation runs in eval mode, where the s2 cross-attention is not causal (FB:313, M:387): the query at position i attends to position i+1, which encodes i's target | The vendored validation is kept as a diagnostic only, made causal: `model.dep_layer.cross_attn` in train mode during it (its dropouts are 0, M:447, K:221) | No look-ahead |
| 4 | The validation loss is random: s2 is conditioned on a sampled s1 (K:267-272) | The diagnostic reseeds its own generator and restores every RNG state; the states are saved before the validation iterator is created, so the loader's base-seed draw is undone | Determinism |
| 5 | No early stopping; every epoch runs and the lowest vendored validation loss is kept (FB:269, FB:354-362) | Early stopping on NLL4-TF with patience 2 (3.3); the child saves the selected state itself | SPEC 9.2 and D-042 (3); no look-ahead in selection |
| 6 | An epoch is every window, shuffled from the global RNG (FB:213-221); OneCycle spans `epochs x len(loader)` | A pseudo-epoch is 512 batches drawn without replacement by `RandomSampler(num_samples=512 x 32)` with the fold's own generator; OneCycle spans `E_max x 512` steps | Compute control (gate 3, the Mac) and an identical schedule in every fold |
| 7 | The split is a share of rows, without dates or embargo (FB:75-97) | One pre-cut CSV per fold; `train_ratio = (n_fit + 0.5) / N`, `val_ratio = 1.0`, so the split lands on an exact date | SPEC 6 and 9.4; question-10 default "pre-cut CSVs" |
| 8 | A window may span a missing candle (FB:107-127) | `GapAwareKlineDataset` (1.6) | DATA.md: never fabricated |
| 9 | `from_pretrained` loads with `strict=False` and checks nothing (HM:785) | sha256 before loading; after loading, every tensor of the model's `state_dict` equals the verified safetensors file bit for bit | D-003 |
| 10 | `num_workers: 6`, `pin_memory=True` | `num_workers: 0`; the rebuilt train loader has `pin_memory=False` | D-014: workers would import the vendored tree outside `import_kronos()` |

**Equivalence guarantee.** With every wrapper switched off and `num_workers` equal, the child
reproduces `python -B train_sequential.py --config <yaml> --skip-tokenizer` on CPU bit for bit
(the `model.safetensors` bytes), tested on a tiny synthetic CSV with a tiny random Kronos
(plan (c), test T1). The evidence that only the documented layers differ is that test.

### 3.2 Hyperparameters of R1

Every key the loader reads (CL:119-182) is written explicitly into the run YAML, including the
unused tokenizer keys, so the recipe never depends on a code default.

| Item | Value | Source |
| --- | --- | --- |
| Model and tokenizer | shipped Kronos-small; shipped Kronos-Tokenizer-base, frozen (`finetuned_tokenizer` points at it; `train_tokenizer: false`) | SPEC 9.2; TS:153-171 |
| Learning rate | `predictor_learning_rate: 1e-6` (OneCycle `max_lr`) | SPEC 9.2; CFG:25 (the code default is 4e-5, CL:147) |
| Schedule (hard-coded) | OneCycleLR, `pct_start` 0.03, `div_factor` 10, `final_div_factor` 1e4 (torch default), cosine, stepped every batch: 1e-7 to 1e-6 over 3 % of the span, then down to 1e-11. The vendored log prints the LR as `0.000000` (FB:306), so the child records it | FB:253-260 |
| Budget | pseudo-epochs of 512 steps; `basemodel_epochs: 6` (`E_max`), 3,072 steps, 98,304 window draws: 2.65 full passes in fold 1, 1.83 in fold 24 | 3.1 row 6 |
| Pilot rule (once, mechanical) | if the pilot's selected pseudo-epoch is the last one, `E_max` becomes 12 for every fold and fold 1 is refit | 6.2 |
| Batch | 32, `drop_last` | CFG:19; FB:219 |
| Optimizer | AdamW, betas (0.9, 0.95) configured, weight decay 0.1 on every parameter; OneCycle cycles beta1 0.95 -> 0.85 -> 0.95, so the configured 0.9 has no effect | CFG:27-29; FB:246-251 |
| Gradient clip | max norm 3.0 (hard-coded) | FB:296 |
| Loss | `(CE_s1 + CE_s2) / 2` over all 512 positions, s2 conditioned on a sampled s1 | FB:288-292; M:494-507 |
| Window | lookback 512, predict 0, `W = 513`, clip 5.0, vendored normalisation | section 2 |
| Split | `train_ratio = (n_fit + 0.5) / N`, `val_ratio = 1.0`, `test_ratio = 0.0` | 1.7 |
| Early stopping | NLL4-TF on 336 inner-validation decisions, patience 2, strict improvement, the shipped weights not a candidate | 3.3 |
| Seeds | per fold and replicate (3.4) | 3.4 |
| Workers | 0 | 3.1 row 10 |
| Device, threads, activation checkpointing | fixed by the measurement step before the pilot, recorded | section (a) |

The alternative budget B_T (GPU only; not taken, D-048 (1)) keeps everything above but restores
the template's epochs: an epoch is a full pass with the vendored loader, `basemodel_epochs: 20`,
and NLL4-TF runs after every epoch.

### 3.3 Early stopping: NLL4-TF

**Definition.** For an inner-validation decision `D` (Binance), with its 512 context rows
`[D - 512 h, D)` and its 4 target rows `D .. D + 3 h`:

1. The six channels of the 516 rows come from the inference functions (`context_frame`, policy
   `volume_x_mean_ohlc`). They are cast to float32, normalised with the mean and population std of
   the 512 context rows only, and clipped at +-5: rows 0..511 are exactly the predictor's input
   (K:620-631).
2. Context tokens: `tokenizer.encode` of the 512 context rows, exactly as inference encodes them.
   Target tokens: tokens 512..515 of the encoding of all 516 rows (the encoder is causal: its
   self-attention always uses `is_causal=True`, M:345-350).
3. For each step j = 0..3, mirroring inference step j (K:420-442): `decode_s1` on tokens and time
   stamps `[j, 512 + j)` (true target tokens beyond the context, i.e. teacher forcing), then
   `-log_softmax` of the last position at the true s1 of row `512 + j`; then `decode_s2(context,
   true s1)` with a query of length 1, as at inference (K:440), and `-log_softmax` at the true s2.
4. NLL4-TF = mean over decisions and steps of `nll_s1 + nll_s2`, in nats per candle, at T = 1
   without top-p truncation.

It runs in eval mode, without sampling and without dropout, on a CPU copy of the weights: it is
deterministic for a fixed thread count whatever the training device (on a CUDA instance it runs
on the GPU with deterministic algorithms). Its query has length 1 and every key is in the past,
so the eval-mode look-ahead of M:387 cannot occur, and its rotary geometry is the inference one.
NLL compares checkpoints with the same tokenizer, which holds here (SPEC 9.3); CRPS on fans
remains the metric that compares models.

**Why not the vendored validation loss.** In eval mode it attends to the target (M:387); it is
random (K:267-272); it scores every position under the all-row normalisation of 2.2; and its
rotary geometry differs from inference. The vendored code has no early stopping at all, so the
stopping rule is project code whichever metric it reads.

**Rule.**
- NLL4-TF is computed for the shipped weights before training (epoch 0, a diagnostic) and after
  every pseudo-epoch e = 1..E_max.
- `best` is the argmin over e >= 1, with strict improvement; ties go to the earlier pseudo-epoch.
  Training stops when 2 consecutive evaluations fail to beat `best`. The fold's model is `best`.
- The shipped weights are never selected. SPEC 9 makes a retrained model a deliverable, produced
  "always", and the vendored keep-best starts at infinity (FB:266), so epoch 1 is always a
  candidate. The manipulation check (5.4) says whether the fine-tune changed the model at all.

**Diagnostics, never used for decisions.** The vendored validation pass, made causal and reseeded
(3.1 rows 3-4), on the 832 inner windows; the training loss; the vendored `best_model` directory
it writes (its hash is recorded, the directory deleted after registration). The T-B report gives
how often the vendored-loss choice and the NLL4-TF choice agree, as evidence for later recipes.

**Mechanics.** `EarlyStopGate` wraps the train loader and forwards `__len__` (512), so OneCycle is
sized as before. Its `__iter__` runs when the next pseudo-epoch starts (FB:281), after the
vendored validation of the previous one has finished. It evaluates NLL4-TF on a CPU copy of the
current weights, keeps the best state as a CPU `state_dict` copy (99 MB), and raises `EarlyStop`
once patience is exhausted. After a normal return the last pseudo-epoch is evaluated once more.
The evaluation consumes no RNG, so CPU training gives the same weights with and without it.

**Cost.** 336 decisions x 4 steps x about 28 GFLOP = 38 TFLOP, about 50 s on CPU at 2 threads,
up to 7 times per fold.

### 3.4 Seeds, determinism and device

- **Training seeds.** `digest = sha256("kronos-agent/train-seed/v1/<recipe_id>/<fold>/r<k>")`.
  Bytes 0-3 are the YAML `seed` (32 bits, because the vendored `set_seed` calls
  `np.random.seed`; it seeds `random`, numpy and torch, MPS included, before the phase, TS:165).
  Bytes 4-11 seed the sampler's generator, bytes 12-19 the multinomial router's training
  generator (63 bits each). The diagnostic validation uses a fixed generator seeded from
  `sha256("kronos-agent/finetune/val-seed/v1")`.
- **Batch order** comes from the sampler's own generator, so it does not depend on dropout or on
  s1 draws. The rebuilt train loader gets the same generator, so its per-epoch base-seed draw
  does not consume the global RNG either. s1 draws come from the router's generator on the CPU on
  every device.
- **CPU training** runs `configure_runtime(threads, deterministic=True)` with the thread count
  the measurement step chooses. Same seed, threads, torch build and platform give the same
  checkpoint (tested on a tiny model).
- **MPS training** is not bit-reproducible. It keeps deterministic mode on unless an op refuses
  (the measurement step finds out; recorded either way). Reproducibility of results rests on the
  registered checkpoint: every fan from it is CPU-deterministic. MPS runs use 2 CPU threads.
- **Device rule** (compute only, fixed after the measurement step and before the pilot): MPS if
  its preflight passes and its sustained throughput at the recipe batch is at least 1.5 times the
  best CPU configuration's (the fastest one that runs batch 32: as is where it fits, with
  activation checkpointing where M2 passed; D-054 (4)); otherwise CPU at the faster of 2 and 8
  threads. All folds of a test, the sealed folds, the model to date and routine retrains use the
  same device and torch build; if that becomes impossible, the change is recorded and disclosed
  in the model card.
- **Evaluation** (every fan and every NLL4-TF on test hours) is CPU, 2 threads, deterministic,
  `device="cpu"`, both models in eval mode (CLAUDE.md).

### 3.5 Memory, interruptions and spend control

- **Memory.** MPS runs set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (1.0 unless the measurement step
  chooses another value; recorded), so an allocation beyond the recommended working set fails
  instead of swapping. Every 50 steps the child reads `sysctl -n
  kern.memorystatus_vm_pressure_level` and `sysctl -n vm.swapusage` (macOS tools, nothing
  installed). It aborts at the critical level or when swap use has grown by more than 1 GiB since
  it started. A memory abort is not a fold failure: the walk-forward stops, the owner is told,
  and it resumes from that fold. If batch 32 does not fit, each `TransformerBlock.forward` is
  wrapped at instance level with `torch.utils.checkpoint` (non-reentrant, RNG preserved); CPU
  gradients must stay bit-identical with and without the wrap. On MPS, torch 2.14.1 restores
  only the CPU generator for the recomputation, so the wrap also replays the forward's MPS
  generator state (D-054 (5)); the measurement step decides on the CPU first and starts MPS at
  batch 32 with the wrap when the CPU's batch 32 needs it (D-054 (2)).
- **Interruptions.** The vendored loop saves no optimizer or scheduler state, so a fold cannot
  resume. The fold is the unit of restart: a registered fold is never rerun and an incomplete
  one restarts from scratch, so a lost run costs at most one fold (about 3 h on MPS at the
  prior). Unattended runs use `caffeinate -i -s`, with the lid open, the Mac on power, Low Power
  Mode off and system updates deferred, and never alongside the other workflow's heavy jobs. A
  lost run's compute counts against the approved estimate.
- **Spend control.** Every command records wall and process CPU time. Before each fold the
  walk-forward projects elapsed time plus the remaining folds at the measured mean fold time; if
  the projection exceeds 1.25 times the approved hours it stops and asks the owner. A fold that
  runs past twice its own worst-case estimate is stopped as a failure. A fan batch stops when its
  CPU-seconds pass 1.25 times the approved figure; stored fans are kept. The D-040 rule (a
  measured estimate above 1.25 times the approval goes back to the owner before anything runs)
  applies to every command.

### 3.6 Checkpoints, registry and loading

- The child saves the selected state with `save_pretrained` into the run directory (a fresh CPU
  `Kronos` loaded with the best `state_dict`). The parent then:
  1. checks that `config.json` equals Kronos-small's registered `expected_config` (the vendored
     serialisation reproduces the shipped 228-byte file);
  2. loads `model.safetensors` with `strict=True` into a fresh `Kronos` and checks
     `state_dict` numel = 24,741,696;
  3. hashes both files and moves them (not the generated `README.md`) atomically to
     `<weights>/finetuned/<checkpoint_id>/`, mode 0o644;
  4. appends an entry to `<weights>/finetuned/registry.jsonl`, append-only and hash-chained
     (each line carries the sha256 of the previous one).
- **Only a recipe run is registered** under the recipe's checkpoint id. Before the child starts,
  the parent refuses a request with any wrapper of 3.1 off (with early stopping off the child
  keeps the vendored keep-best, selected by the vendored validation loss, which reads its target
  in eval mode, M:387), one that does not start from the registered shipped weights, and an id
  that is registered already or whose directory exists (a registered fold is never trained
  again, 3.5). After it, a record that does not show every wrapper on and the NLL4-TF selection
  is refused. Diagnostic configurations (T1) run without registration.
- The chain protects every line but the last: an edit that keeps the last line a valid entry, or
  its removal, leaves a valid chain. A checkpoint is therefore named, wherever its use is
  recorded, with the sha256 of its registry line (`CheckpointRef(id, root, line_sha256)`; `fit`
  returns the entry), and every reader of such a reference refuses an entry that no longer
  hashes to it. A prepared fan batch records the line of the entry it read, so its loaders
  refuse an entry changed after the preparation (an earlier `data_end_ms`, say).
- The registry stays outside `manifests/`: `manifests/*` enters `source_tree_sha256` (D-025), so
  a tracked registry would change the recorded code hash at every registration. The experiment
  ledger (section 7) and the model card record the sha256 of every checkpoint they use and of
  its registry line.
- **Entry:** `checkpoint_id` (`ks-btc-<recipe8>-<fold>[-r<k>]`), `recipe_id`, fold, replicate,
  base and tokenizer hashes, files (name, size, sha256), `state_dict_numel`, `expected_config`,
  `data_end_ms` (the CSV end), fit and inner-validation ranges, first test decision, train CSV
  and store meta sha256, device, threads, determinism, activation checkpointing, torch version,
  platform, seeds, code provenance (source tree, training code, vendored manifest), creation
  time. Selection values (the NLL4-TF curve, the selected pseudo-epoch) live in the blinded fold
  record, not in the registry.
- **Loading:** `load_models("small", checkpoint=None | id)`. With an id it verifies the registry
  chain (and the recorded line, when the reference carries one) and the entry's files (size,
  sha256, config), loads `strict=True`, checks the element count, sets eval mode without
  gradients on CPU, builds the predictor with clip 5, and sets `LoadedModels.checkpoint` and
  `LoadedModels.data_end_ms`. `spec` stays the `small` spec, so the model-name checks of
  `fan_from_params` and `forecast_config` pass unchanged. The checkpoint's file hashes enter
  `weights_sha256`, so its fans get their own `config_id` and never share a directory with the
  shipped model's.
- **Identity control.** Before any fold is trained, the shipped weights go through the whole
  output path: `save_pretrained`, verification, registration in a scratch registry,
  `load_models(checkpoint=...)`, a spawn-started batch worker. The resulting fan must be
  bit-identical to the shipped model's on CPU (only `config_id` differs). It runs in the
  measurement step (M9) and as a `weights` test, so the evaluation pipeline cannot by itself
  create a difference between the two models.

## 4. Tokenizer diagnostic (SPEC 9.1)

A recorded study (D-034): descriptive, 0 tests, CPU at 2 threads, deterministic (the quantiser is
a sign function and `decode` is deterministic, so there is no seed). It never changes R1: the
tokenizer stays frozen and the clip stays 5. A flag can only motivate the separate, later
tokenizer test SPEC 9.2 asks for.

**Samples.** Decision windows on `G25` built by the inference window builder (512 candles,
`amount` policy A, `volume_x_mean_ohlc`), incomplete windows dropped with their ranges.
- Part A, before registration: Binance, training segment, `D` from 2020-01-22T08:00Z (the first
  full context) to `TRAINING_END - 48 h` (about 1,540 windows), each with its next 48 rows.
- Part B, after registration, inputs only: Binance and BingX, validation segment (about 700
  windows each). It is descriptive and cannot shape the recipe.
- References: 100 synthetic GBM windows (`features.synthetic_window`) as a featureless reference.

**Per window**, rows `j = 0..511`, channels `c` in {open, high, low, close, volume, amount}: `x`
the float32 context; `mu`, `sigma` the per-channel mean and population std over the 512 rows;
`z = (x - mu) / (sigma + 1e-5)`; `zc = clip(z, -5, 5)`; `(s1, s2) = encode(zc, half=True)`;
`zr = decode(...)`; `xr = zr * (sigma + 1e-5) + mu`; and the coarse reconstruction from the s1
bits alone (`z_pre`).

**Definitions.**
- Clip saturation: per channel, the share of context entries with `|z| > 5`; the share of
  windows with at least one; the share of next rows (row 512 with the context statistics) beyond
  +-5, i.e. candles the model cannot represent as the next step; `max |z|` descriptive.
  Comparator: `P(|N(0,1)| > 5) = 5.7e-7`.
- Reconstruction: RMSE of `zr - zc` per channel (all rows and the last 24); close error
  `1e4 * (xr - x) / x` in bp (median `|e|` with intervals; p90, p99, max descriptive); the
  return resolution `rho = median|er| / median|r|`, where `er` is the error of the reconstructed
  1h log return and `r` the true one (the headline number: how coarse the codebook is relative to
  a typical hourly move); volume and amount in normalised space only; s1-only versus full.
  Comparators: the median `|1h return|` and one 0.1 USDT tick in bp.
- Codebook usage: distinct s1 and s2 codes (of 1,024) and pairs (of 2^20) at matched row counts;
  entropy, perplexity and normalised entropy; the fewest s1 codes covering 50, 90 and 99 % of
  rows; per-bit share of ones (a bit is dead below 1 % or above 99 %); Jensen-Shannon divergence
  of the s1 marginals, training against validation and Binance against BingX, with the first
  against the second half of Part A as comparator.
- Bit margin: the share of latent coordinates within `1e-3` and `1e-2` of zero after the L2
  normalisation, i.e. token bits a rounding difference can flip (relevant to MPS and to
  cross-machine agreement).
- **Token mismatch** (Part A; the 5 % check of D-048 (5)): for each window, the share of its 512
  context rows whose s1 (and s2) token differs from the inference encoding when the window is
  normalised as R1 trains (all 513 rows), as the template trains (all 561 rows), and
  context-only (513 rows), overall and for the last 24 rows.

**Flags** (pre-registered, descriptive, no automatic action): price-channel saturation above
0.1 % of context entries or above 1 % of next rows; volume or amount saturation above 1 %;
`rho` above 0.25; s1 normalised entropy below 0.5 or more than half the s1 codes unused; s1
perplexity on validation below half the training value (Part B); R1's s1 token mismatch above
5 % of rows.

**Intervals.** 4-week block bootstrap over `D` (D-035) for shares, means and medians; p99 and max
descriptive. **Cost:** about 3,100 windows (each Part A window encoded three more times for the
token mismatch), under 3 minutes on CPU.

## 5. Evaluation

### 5.1 Targets, scores and references

- `c_ref` = `float()` of the close of the candle opening at `D - 1h` (`DecisionWindow.last_close`).
  Trajectory k: `x_k(h) = 1e4 * log(close_k(h) / c_ref)`. Realised: `y(h) = 1e4 *
  log(close(D + (h - 1) h) / c_ref)` from the BingX 1h store. A missing target candle drops the
  decision at that horizon, with its reason; a decoded close `<= 0` makes the fan invalid
  (dropped and counted).
- **Fair ensemble CRPS** (K = 20), in bp:
  `CRPS = (1/K) sum_i |x_i - y| - (1 / (2 K (K - 1))) sum_{i != j} |x_i - x_j|`.
- **No-skill reference** (SPEC 8: zero return with historical volatility): `N(0, s_h^2)`,
  `s_h = s_1 * sqrt(h)`, `s_1` the sample std (ddof 1) of the 511 hourly log returns of the
  context closes; closed form `s [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]`, `z = y / s`.
- **Lazy reference** (SPEC 7: next = last): point forecast 0, `CRPS = |y|`.
- **Kronos as shipped:** the same fans with the shipped weights and the same seeds.
- **Spread-recalibrated shipped** (secondary comparator; SPEC 10 orders sampling before
  fine-tuning): for fold k, a scale `s_k` on the grid 0.50, 0.55, ..., 2.00 minimises the mean
  CRPS at h = N of the stored shipped `G4` fans transformed as `x_i' = xbar + s (x_i - xbar)`,
  over every earlier decision whose targets open before `T_k - 512 h` (an expanding window from
  2024-07-01; `s_1 = 1` for the first fold). No new fan is needed. It tells whether a gain is
  only a dispersion fix that needs no retraining.

### 5.2 Grids, streams and fans

All fans are 20 trajectories by `predict_batch([df] * 20, ..., sample_count=1)`, T 1.0, `top_p`
0.9, `top_k` 0, CPU, 2 threads, deterministic, BingX windows, seeds from `D` and the stream. A
4-step fan is not the prefix of a 24-step one, because the final decode reads a token window that
starts later (K:456-464), so both are computed.

| Set | Grid and span | Decisions | Length | Model / stream | Fans | CPU-h |
| --- | --- | --- | --- | --- | --- | --- |
| T-A primary | `G4`, 2024-07-01 to 2026-07-01 | 4,380 | 4 | shipped / `forecast` | 4,380 | 7.7 |
| T-A skill curve | `G25`, same span | 701 | 24 | shipped / `forecast` | 701 | 6.5 |
| T-A A/A | `G4` | 4,380 | 4 | shipped / `forecast-aa` | 4,380 | 7.7 |
| T-B primary | `G4` | 4,380 | 4 | R1 fold models / `forecast` | 4,380 | 7.7 |
| T-B skill curve | `G25` | 701 | 24 | R1 fold models / `forecast` | 701 | 6.5 |
| **Total** | | | | | **14,542** | **36.0** |
| Pilot check (0 tests) | Binance, every 8 h in fold 1's inner validation | 168 | 4 | pilot, shipped / `forecast`; shipped / `forecast-aa` | 504 | 0.9 |
| Seed replicates (D-048 (4)) | `G4` and `G25` of folds wf-2024-07, wf-2025-06, wf-2026-06 | 546 + 88 | 4, 24 | replicates r1, r2 | 1,268 | 3.5 |
| Contingency: N = 8 | `G8` | 2,190 | 8 | both | 4,380 | 14.3 |
| T-B secondary: Binance copy (the D-040 contingency, triggered) | Binance `G4` | 4,380 | 4 | both | 8,760 | 15.3 |
| Optional tier 3 | the other three phases of the 4-hour grid | 13,140 | 4 | both | 26,280 | 46.0 |
| Sealed, Aug to Oct 2026 | `G4` and `G25` | 552 + 89 | 4, 24 | both | 1,282 | 3.6 |

- **Fan time** at 2 threads: `t(L) = 0.44 + 0.678 L` seconds, fitted to the two measurements on
  this Mac (3.15 s for 20 x 4, D-040; 5.86 s for 20 x 8, the step-1 benchmark): `t(4)` = 3.15 s,
  `t(8)` = 5.86 s, `t(24)` = 16.71 s. CPU-seconds = 2 x wall (D-033). The measurement step
  replaces this line.
- **Reuse.** The shipped fans of T-A are the shipped half of T-B, under one forecasting-code hash.
  Step 3 shares them only because it adopts the `G4` and `G25` grids (its draft used one decision
  a day at 00:00Z). The 873 fans of the mismatch study (D-040) are not reused: about 73 of its 291
  hours fall on `G4`, and the `kronos/decision.py` edit of 5.3 changes every `config_id`.
- **Contingencies**, decided before the readout and never after it: the `G8` fans only if T-A sets
  N = 8 (6.4); the Binance copy only if the D-040 report reads "the sources shift the signal
  beyond seed noise" (then it is a T-B secondary that separates "the model improved" from "the
  source changed"); tier 3 (SPEC 8's second pass over all hours and SPEC 7's hour shifts
  +1/+2/+3 h) only if the owner adds it at the walk-forward's gate-3 ask, before T-B is read.
- **The D-040 contingency has been read: it is triggered.** The mismatch report,
  `reports/data/mismatch.md` (2026-11-02; 291 sample hours, 873 fans), reads "the sources shift
  the signal beyond seed noise" for `A_xd - A_f` = -0.1237 (95 % interval [-0.1707, -0.0742]) and
  "not distinguishable" for `A_x - A_f` = -0.0584 ([-0.119, 0]). `A_x` is the signal agreement of
  the Binance and BingX fans under shared seeds, `A_f` that of two BingX seed streams and `A_xd`
  that of the Binance fans and the other BingX stream (`docs/DATA.md`, study 3). Under the rule
  above, the Binance copy is in the plan as a T-B secondary: the `G4` fans of both models on
  Binance 1h windows, with the seeds of the primary, scored against the Binance 1h closes (8,760
  fans, 15.3 CPU-hours at the prior fan time). With it the evaluation needs 23,302 fans and 51.3
  CPU-hours.

### 5.3 Evaluation length and the horizon purge

`check_embargo` refuses `max_context + horizon > 520`, which refuses a 24-step fan, and today a
fan's length must equal `kronos.horizon_n`. The design:

- `params.yaml` keeps `kronos.horizon_n` = the trading N. `check_embargo(max_context, N)` stays
  as it is and governs the configuration (SPEC 6).
- `data/partition.py` gains `EVAL_LENGTH_MAX = 24`, `check_evaluation_length(L)` (1 <= L <= 24)
  and `require_model_period(D, *, data_end_ms=None, role, allow_pretraining_period=False)`, which
  replaces `require_kronos_period` at its call sites (shipped: `D >= 2024-07-01`; a checkpoint:
  `D >= data_end`; the opt-in roles of 6.2 and 6.4 are named and recorded).
- `kronos/decision.py`: `forecast_window(..., evaluation_length=None)`. With a length `L`, the
  fan is computed with `params_L = replace(params, kronos=replace(params.kronos,
  horizon_n=L))`, after `check_embargo` on the governed N, `check_evaluation_length(L)` and
  `sample_fits(D, L)`. Every later step (`_check_against_params`, `fan_from_params`,
  `forecast_config`, the store's fan check) then sees `horizon_n = L`, so a 24-step fan records
  `horizon_n = 24` and gets its own `config_id`. Longer samples are purged, never shortened.
- `kronos/batch.py`: `prepare_tasks`, `timing_window`, `measure_seconds_per_fan` (so gate 3 times
  the length it estimates), `run_fan_batch` and the worker initialiser take `evaluation_length`
  and `checkpoint`; `registry_config(kronos, checkpoint=None)` reads a checkpoint's hashes from
  the fine-tune registry. One batch runs one checkpoint.
- `kronos/decision.py` is in `FORECASTING_FILES`, so this edit changes every `config_id` once. It
  lands after the D-040 fans and before any evaluation fan; the pre-registration bundle freezes
  the forecasting-code hash, and the commands refuse fans of another hash.

### 5.4 Statistics

**T-A, Kronos as shipped (SPEC 8; step 3):**
- Skill curve h = 1..24 on `G25` against no-skill and lazy: `SS_h = 1 - sum CRPS / sum
  CRPS_ref` per h, pointwise intervals, a simultaneous 95 % band over h = 1..8 (max-|t| bootstrap
  on the same resamples), and the first horizon whose pointwise interval contains 0.
- Calibration: coverage of the central intervals between order statistics `[x_(i), x_(21-i)]`,
  whose nominal coverage for K = 20 exchangeable members is exactly `(21 - 2i) / 21`: 90.5 %
  (i = 1), 81.0 % (i = 2), 52.4 % (i = 5), at h in {1, N, 8, 24}; the PIT histogram (rank of `y`
  among the 20, 21 bins) at h = N and 24; reliability of `p_up` at h = N (10 bins, Wilson
  intervals at block level).
- Direction: hit rate of the sign of the mean path at h in {1, N, 8} against 50 %, with the
  SPEC's 43-57 % example band quoted.
- A/A: `Delta_AA = mean [CRPS(shipped, forecast-aa) - CRPS(shipped, forecast)]` at h = N with its
  interval, and the two-seed signal agreement. If the interval of `Delta_AA` excludes 0, the
  reports flag that the block bootstrap may be anti-conservative for paired differences.
- Cutoff check (SPEC 3): shipped NLL4-TF on Binance `G4` decisions per calendar month, 2024-01 to
  2024-12. Read descriptively: "consistent with memorisation of the first half of 2024" if the
  mean of January-June is lower than that of July-December by more than twice the
  month-to-month standard deviation. The January-June forecasts use a named opt-in of the
  pre-cutoff rule.
- The SPEC 8 decision tree and the N rule of 6.4.

**T-B, R1 against shipped (decides the recipe):**
- Primary: `d(D) = CRPS_R1(D, h=N) - CRPS_shipped(D, h=N)` on the `G4` test decisions, the R1 fan
  coming from the fold whose test month holds `D`; `Delta_N = mean d(D)` in bp and
  `RS_N = 1 - sum CRPS_R1 / sum CRPS_shipped`; the minimum detectable difference
  `(z_0.975 + z_0.80) x SE`, read from the interval (D-036).
- Per calendar quarter: `Delta_N` for 2024Q3 to 2026Q2 (8 point estimates; 3-4 blocks each, no
  interval).
- Fold bootstrap (sensitivity): 1,000 resamples of the 24 fold means. The fold is the unit in
  which training randomness enters.
- Post-cutoff sensitivity: `Delta_N` over test months 2024-10 to 2026-06, whose folds early-stop
  entirely after 2024-07-01.
- The SPEC 8 statistics of T-A for R1, the paired `Delta_h` for h = 1..24, and the SPEC 8 tree for
  R1.
- Agreement: the share of `G4` decisions where R1 and shipped give the same signal, against the
  A/A two-seed floor (SPEC 9's 88 % is quoted as the external reference).
- Manipulation check per fold: relative weight change `||theta_k - theta_0|| / ||theta_0||`,
  signal agreement with shipped under common seeds, mean `|Delta p_up|`.
- `Delta` against the spread-recalibrated shipped comparator (5.1), with its interval.
- NLL4-TF on the `G4` test decisions (BingX) for both models: a secondary metric SPEC 9.3 allows
  between models with the same tokenizer (about 22 CPU-minutes).
- Binance copy (5.2; the D-040 contingency, triggered): `Delta_N` with its interval on the Binance
  `G4` fans of both models, scored against the Binance 1h closes. It separates "the model
  improved" from "the source changed".
- Seed replicates (D-048 (4)): `Delta_N` per replicate on the replicate folds, and the count of
  seeds with `Delta_N < 0`.
- Unblinded fold records: NLL4-TF curves, selected pseudo-epochs, epoch-0 values, each fold's
  inner improvement next to its test improvement, and the agreement of the two selection
  metrics.
- Multiplicity across recipes (7.4).

Each report prints its number of intervals and the 0.05 per interval expected to miss: about 90
for T-A and 190 for T-B, so about 4.5 and 9.5 misses by chance.

**Figures** (D-037, SPEC 14): skill curves with pointwise and simultaneous bands for both models
and both references; `Delta_h`; reliability; coverage; PIT; `Delta_N` per fold and per quarter;
the NLL4-TF curves per fold (after unblinding); the pilot's curves; tokenizer saturation and
codebook histograms.

### 5.5 Intervals

D-035 applies: 95 % percentile block bootstrap, 1,000 resamples, numpy seed from the statistic's
key (`finetune-wf-<recipe8>/<group>/<statistic>`), quantiles at 0.025 and 0.975, both columns of a
paired comparison resampled in one call. Blocks are groups of 4 calendar weeks
(`week_blocks(open_ms, weeks=4)`), because Kronos contexts overlap for about 3 weeks; weekly
blocks would understate the variance. The test span touches 27 blocks (at least 20 are required).
Ratios of means are computed per resample. Fewer than 20 blocks gives a point estimate only, with
the reason (per quarter, per fold, the sealed months).

## 6. Walk-forward, readings, sealed test and model to date

### 6.1 Layout, cold start and the compute control

- **Cold start.** Every fold starts from the shipped Kronos-small and trains on its whole CSV.
  SPEC 9.4 ("train up to T") and 9.5 ("what is validated is the RECIPE") describe the model to
  date as one fine-tune of the shipped weights on everything to date, so the recipe validated must
  be that procedure applied at each `T`. Folds are independent: a bad fold does not propagate,
  and the fold bootstrap stays meaningful. A warm-start chain (each fold continuing the previous
  one on the new month) is a different, cheaper recipe and a later test.
- **Frequency.** Monthly (`params.yaml`, the SPEC's initial value) when the measured worst case is
  at most 72 h (D-048 (1)) and the measured cost is approved; otherwise the quarterly fallback
  over the same test decisions, which then becomes the deployed frequency until monthly is tested
  (one test). The choice uses compute only, after the pilot and before any fold beyond the first,
  under D-048 (1), and is recorded in the bundle.

### 6.2 The pilot and the go/no-go

1. **Train fold 1** with the recipe (gate-3 ask with the measured estimate).
2. **Pilot rule** (mechanical, once): if the selected pseudo-epoch is the last (`E_max`), set
   `E_max = 12` for every fold and refit fold 1.
3. **Pilot check** (`kronos-agent finetune pilot-check`, 504 fans, 0.88 CPU-hours, 0 tests). On
   the 168 Binance decisions every 8 h of fold 1's inner validation: fans of the pilot model and of
   shipped (stream `forecast`), and a second shipped stream (`forecast-aa`); the mean KL
   divergence between the two models' first-step s1 distributions (deterministic); the relative
   weight change; the NLL4-TF curve (shown: training segment). Shipped forecasts there are before
   the cutoff, and the pilot model's are inside its own inner validation, so both run under a
   named opt-in recorded in the report; no skill is read from them. Statistics: `A_same` = signal
   agreement of pilot and shipped under common seeds, `A_floor` = the two shipped streams'
   agreement, mean `|Delta p_up|`, and the paired SD and lag-1..6 autocorrelation of `d(D)`,
   projected to the minimum detectable difference for 4,380 decisions in 27 blocks (labelled
   approximate).
4. **Go/no-go** (pre-registered): if `A_same >= 0.98` (SPEC 12's live-against-replay success level
   for "the same system"), the fine-tune barely changed the model. The work stops and the owner
   chooses: run anyway as a measured null; the GPU with the template budget; or lr 4e-5 (the code
   default), a recipe change only the owner can make because SPEC 9.2 names 1e-6.
5. **Compute control** and the gate-3 ask for the remaining folds, with the measured fold time,
   the thermal factor, the pilot's pseudo-epochs, the expected and worst-case totals, and the
   projected minimum detectable difference.

Disclosure: fold 1 early-stops on April-June 2024, inside Kronos's pretraining period, which
biases its curve towards the shipped weights (conservative); folds 2 and 3 are partly affected.
The post-cutoff sensitivity of 5.4 isolates it.

### 6.3 Blinding, failures and no interim look

- **Blinding.** From fold 2 on, the inner validation lies inside earlier folds' test months by
  date, and the epoch-0 value is the shipped model's NLL on those dates: reading it would preview
  the test. For folds 2 and later, every loss value (training loss, the vendored diagnostic,
  NLL4-TF, epoch 0) and the selected pseudo-epoch go only into the fold record. The runner prints
  only the fold label, wall time, finiteness checks and the checkpoint sha256; the child's stdout
  and logs stay in the run directory and are not read. `finetune report --test T-B` opens them
  after its ledger entry. Fold 1 (the pilot) is shown.
- **Failures.** A fold whose training fails (NaN or infinite loss, crash) is retried once with the
  same seed. If it fails again, its test month uses the previous fold's model, mirroring SPEC 9's
  "if it fails, the previous one stays", and the substitution is reported. If fold 1 fails twice,
  the walk-forward stops and the owner is told. Memory aborts, power loss and sleep are not
  failures (3.5).
- **No interim look.** Fans may be computed fold by fold while later folds train (computing a fan
  is not reading it), but no statistic on test targets is computed until every checkpoint and
  every fan of the pre-registered sets exists; the report command refuses a partial sample.

### 6.4 T-A and the interaction with step 3

- T-A and T-B are registered together in the bundle (7.1), before either is read. T-A is read
  first, as SPEC 18 orders step 3 before step 6.
- **SPEC 8 tree, mechanical.** Case 1: the simultaneous band of `SS_h` against no-skill lies
  entirely above 0 for some h <= 8. Case 2: not case 1; every order-statistic coverage interval
  at h = N contains its nominal level; the Spearman correlation between the fan's interquartile
  range at h = N and `|y(N)|` has an interval entirely above 0; and the direction hit rate's
  interval contains 50 %. Case 3: otherwise. Retraining proceeds in every case (SPEC 9).
- **N rule.** N stays 4, unless T-A is case 1 with skill shown at h = 8 and not at h = 4; then
  N = 8 (the SPEC's only listed variant; 512 + 8 still passes `check_embargo`). A new N enters
  `params.yaml` before T-B is read, the `G8` fans are computed for both models (14.3 CPU-hours,
  gate 3), and T-B's primary uses them. Nothing else may change between T-A's readout and T-B's,
  and the T-B report says that T-A was read first.

### 6.5 Reading rules of T-B (pre-registered)

| Reading | Condition |
| --- | --- |
| **ACCEPT** (candidate for the sealed test) | the upper bound of `Delta_N`'s interval < 0, AND `Delta_N < 0` in at least 5 of the 8 quarters (SPEC 12: "improvement per quarter: majority"), AND `RS_N >= 1 %`, the smallest effect of interest (D-048 (2)); from the second recipe on, also the reality check of 7.4 |
| **IMPROVES, BELOW THE SMALLEST EFFECT OF INTEREST** | the first two hold and `RS_N < 1 %`: reported, not taken to the sealed test |
| **WORSE** | the lower bound > 0 |
| **NOT DISTINGUISHABLE** | anything else, with the minimum detectable difference |

Flags added to any reading:
- **FRAGILE:** the fold bootstrap reads differently from the primary.
- **UNINFORMATIVE:** the reading is not ACCEPT and the median fold agreement with shipped under
  common seeds is at least 98 %. The recipe barely changed the model, so the result says nothing
  about retraining as such, and the SPEC 12 abandonment input is recorded as "not established"
  rather than "no skill".
- **DISPERSION ONLY:** ACCEPT, but the interval of `Delta` against the spread-recalibrated shipped
  model does not lie below 0. The owner is told before any sealed order.
- **A/A WARNING:** carried over from T-A.

The SPEC 8 tree is re-read for R1 (SPEC 8 case 3, step 6), and the report states whether the
retrained Kronos shows skill at h <= 8 by the simultaneous band: the third condition of SPEC 12's
abandonment criterion. Secondary statistics never change the reading.

### 6.6 The sealed test, T-C (gate 6)

- **When.** Only after ACCEPT, on the owner's order recorded in the ledger; it opens the sealed
  segment once for this candidate. The order fixes the end: every complete month from 2026-08
  before the order (Aug to Oct 2026 today: 552 `G4` and 89 `G25` decisions).
- **What.** The walk-forward continued: the sealed folds of 1.3, same recipe, device and seed
  scheme, same fans and statistics on sealed BingX hours. Fold `s-2026-08` needs no sealed candle
  to train and may be trained before the order; its fans may not.
- **Reading.** Three months touch about 5 four-week blocks, below the 20 an interval needs, so the
  sealed test is read against a band from validation, as SPEC 13 reads live trading. Draw 1,000
  resamples, each the mean of `d(D)` over `m` whole validation 4-week blocks drawn with
  replacement, `m` being the number of blocks the sealed decisions touch; take q05 and q95.

| Reading | Condition |
| --- | --- |
| **CONFIRMED** | `Delta_s < 0` and `q05 <= Delta_s <= q95` |
| **BETTER THAN VALIDATION PREDICTED** (recorded, not evidence of more skill) | `Delta_s < q05` |
| **CONSISTENT, NO IMPROVEMENT** | `0 <= Delta_s <= q95` |
| **CONTRADICTS VALIDATION** | `Delta_s > q95` |

Only the first two lead to shadow mode with R1. The report states the low power of a few months.
T-C consumes 1 test, and the sealed segment is then used up for this candidate. An interval for
the sealed segment would need 20 four-week blocks, which it reaches only around 2028-02.

### 6.7 The model to date, its card, routine retraining and deployment

- **Always produced** (SPEC 9: "the retraining IS ALWAYS DONE"). After CONFIRMED or BETTER:
  recipe R1 at `T_final`, trained after the sealed test on the same order's access (D-048 (3)).
  Without an accepted and confirmed recipe: `T_final = 2026-08-01T00:00Z` (no sealed candle), with
  the negative result in its card. Gate-3 ask: one fold, at most about 2.9 h on MPS.
- **Binance up to the run day.** `kronos-agent data binance` first: the current month comes from
  its daily archives (yesterday is the newest). The CSV ends at the last complete day; if the
  newest Binance day is more than 3 days old, the run waits and tells the owner. A gap is handled
  as in 1.6. The model's `data_end` is recorded, and it forecasts only `D >= data_end`; live
  decisions read BingX, so the Binance store's end matters only for training.
- **Model card**, `reports/finetune/model_card_<checkpoint>.md` (English, SPEC 19): intended use
  (a forecast input that may only modulate the owner's criterion and never opens a trade, hard
  rule 1; research on a demo account; no financial advice); base model and tokenizer with their
  hashes; data (sources, ranges, gaps, CSV sha256, `amount` proxy); the full recipe including the
  hard-coded vendored values; device, seeds, threads, torch version; the selected pseudo-epoch and
  NLL4-TF curve; the checkpoint sha256. Evidence: the recipe's walk-forward and sealed results with
  intervals, comparators, seeds and the number of tests (the model to date has no out-of-sample
  data of its own, SPEC 9.5). Limitations: 2.2, 2.3, one training seed per fold, MPS
  nondeterminism, Binance-trained and BingX-used (the D-040 agreement), the 21-day staleness of
  the evaluated folds, the pretraining overlap of the first folds' inner validation, lr 1e-6 and
  the step budget. Licences: Kronos MIT and the weights' licence, checked before any publication
  (gate 7).
- **Routine retraining** (after deployment, SPEC 9 "Deployment"): the same recipe with the CSV
  ending at the retrain date, then SPEC 9's short check: CRPS over the last 4 weeks (BingX `G4`,
  stream `forecast`) not worse than the previous model's, and signal agreement of at least 88 %;
  otherwise the previous model stays and the owner is told. Known bias: those 4 weeks are the new
  model's own inner-validation dates (on Binance), so the check detects breakage, not
  improvement, and its record says so. The unbiased alternative, a prospective 7-day shadow
  comparison, is an owner decision for step 8.
- **Before shadow mode** (step 8, its own gates): SPEC 16's cross-machine check for the new
  weights (signal agreement between the Mac and the server with the same seeds over a 7-day
  replay, on CPU, with 88 % as the floor) and the peak RAM of a forecast on the 1.9 GB server; then
  2 weeks of shadow mode, the owner's approval and gate 5. The live model is named by a new
  governed key `kronos.weights` (`shipped` or a checkpoint id), added in step 8.

### 6.8 Later one-variable tests

Each costs one test, plus its own sealed opening if accepted: context-only normalisation (2.2);
the leak-free objective with a loss mask (2.2); lr 4e-5; the budget `E_max`; BingX early
stopping; a warm-start chain (cheaper routine retrains); quarterly against monthly; history `2y`;
the 15m timeframe; no test embargo (models 21 days fresher); a tokenizer fine-tune (SPEC 9.2).

## 7. Governance: pre-registration, ledger and test budget

### 7.1 The pre-registration bundle

One canonical-JSON file, `reports/finetune/prereg-<recipe8>.config.json`, published once with
`os.link` (D-024, D-034) and entered in the ledger (kind `preregistration`, 0 tests) after the
pilot and before any fold beyond the first trains and any validation outcome is computed. Every
`finetune` command verifies it and refuses a changed bundle (exit 2). It holds:

- **Recipe R1** with every value of 3.2 (hard-coded vendored values included), the outcome of the
  pilot rule, the device, threads, batch and activation-checkpointing choices, and the code
  hashes: the vendored manifest, the project's training code, and the forecasting-code hash every
  compared fan must carry.
- **Folds:** the fold table (monthly or quarterly as chosen; sealed and final rules), the
  boundaries of 1.4, the gap policy, the failure and blinding rules.
- **Evaluation:** grids, streams and seeds, scores and references, every statistic of 5.4,
  bootstrap keys and blocks, the reading rules of 6.4 and 6.5, the smallest effect of interest,
  the contingencies, the sealed band rule.
- **Tests:** the accounting of 7.2, including which optional sets belong to T-B.
- **The tokenizer diagnostic definitions** (Part B runs after the bundle).

The recipe is also written to `params.yaml` as `kronos.finetune_recipe` (SPEC 9 lists the
fine-tuning recipe among the `kronos` parameters), which changes `params_version` but not any
forecast `config_id`.

**Disclosure rule.** If any validation outcome of shipped Kronos is seen before the bundle is
recorded, the T-A and T-B reports state it as a deviation. None has been seen as of 2026-11-02:
the step-2 studies compare sources and stores, not forecasts with outcomes (the mismatch study
compares the fans of two sources and two seed streams with each other, and its report states that
no validation outcome is involved).

### 7.2 What a test is, and what this plan spends

A **test** is any statistic that compares a configuration's forecasts with realised outcomes on
hours reserved for evaluation: the walk-forward test months, the SPEC 8 validation set or the
sealed segment.

| Id | What is read | Tests |
| --- | --- | --- |
| Tokenizer diagnostic (A and B) | inputs only | 0 |
| Measurement step | synthetic windows | 0 |
| Pilot = fold 1, with its curves and the pilot check | training-segment data (SPEC 7: free exploration, logged) | 0 |
| Early stopping inside every fold | part of the recipe, automatic, blinded | 0 |
| T-A: Kronos as shipped, SPEC 8 on validation | validation outcomes | 1 |
| T-B: R1 against shipped, with the SPEC 8 re-measurement of R1 and the pre-registered optional sets | validation outcomes | 1 |
| T-C: the sealed test, only after ACCEPT | sealed outcomes | 1 |
| Model to date; routine retrain checks | no evaluation; pre-registered monitoring | 0 |

At most 3 of the 15 per quarter, with T-A and T-B read in different ISO weeks. An optional set
added after T-B is read is a new test. If R1 is not accepted, the next one-variable test of 6.8
costs 1.

### 7.3 Ledger, budget, orders and approvals

- `kronos_agent/ledger.py` (the interface of `docs/ARCHITECTURE.md`): append-only canonical JSON
  lines in `ledger/experiments.jsonl`, tracked (SPEC 19 publishes it), each with `prev_sha256` of
  the previous line (genesis 64 zeros), `ts`, `params_version`, code provenance and a payload;
  `verify_chain` before every append. Kinds: `preregistration`, `exploration`, `fit` (checkpoint
  hash and status, no values), `gate3_approval` (the estimate shown, the figure approved, the
  owner's words translated), `owner_order`, `owner_answer`, `test`. Standard library only, because
  `data/partition.py` imports nothing heavy. It holds no personal data (the hygiene test covers
  it). It is built now: steps 4-5, which were to build it, depend on the discarded criterion.
- `kronos_agent/budget.py`: `finetune report --test <id>` verifies the bundle, checks the ledger,
  appends the `test` entry, and only then computes any statistic on reserved hours. It refuses
  (exit 2, `REFUSED:`) a 4th test in an ISO week (UTC), a 16th in a calendar quarter (UTC), a test
  without a matching bundle, a changed bundle, and a second reading of the same id. A test counts
  from its first statistic, so "compute, peek, abandon" still spends it.
- **Orders:** `ledger record-order --scope open_sealed --candidate <test id>` with the owner's
  words, run only after the owner's explicit order in chat. `partition._verify_order` finds an
  unused order of that scope for the candidate; the sealed commands mark it used.

### 7.4 Multiple recipes on the same months

Every later recipe is read on the same 24 test months. From the second recipe on, each reading
adds a reality check over every recipe read on them so far: the per-decision differences
`d_j(D)` of all recipes are resampled in the same 4-week blocks; the statistic is the largest
standardised mean improvement over shipped; its 95 % critical value is the 95th percentile of the
bootstrap distribution of the largest centred standardised mean. ACCEPT then also requires the
recipe's standardised improvement to exceed that value (SPEC 7: "bootstrap of the maximum"). Every
report prints the number of recipes read on these months. The sealed segment protects the final
claim.

## 8. Compute

### 8.1 Formulas

- **FLOP.** About 56.6 MFLOP per token for the Kronos-small forward; a training window of 513 rows
  is `512 x 56.6 MFLOP x 3 + 2.9 GFLOP` (frozen tokenizer encode) = 90 GFLOP; an optimizer step at
  batch 32 is 2.87 TFLOP (0.72 at batch 8, 1.44 at batch 16).
- **One fold:** `C_fold = theta x E_run x 512 x t_step + (E_run + 1) x (t_nll + t_val) + t_setup`,
  with `E_run <= E_max` (6, or 12 after the pilot rule), `t_step` the measured seconds per step,
  `theta` the thermal factor from the sustained measurement (>= 1), `t_nll` about 50 s and `t_val`
  (the vendored diagnostic on 832 windows) about 25 s.
- **Walk-forward:** the sum of `C_fold` over the folds; the worst case puts `E_run = E_max`
  everywhere; the expected case uses the pilot's `E_run`.
- **Fans:** `cpu_seconds = sum over sets of fans x t(L) x 2`, `t(L) = 0.44 + 0.678 L` s until
  measured; `wall = cpu_seconds / (2 x W_eff)`, `W_eff` the measured effective worker count.
- **Gate-3 accounting** (D-033): CPU-hours = wall x threads for CPU work; GPU-hours = wall time on
  MPS or a cloud GPU.

### 8.2 Estimates (per-step priors)

Priors per optimizer step at batch 32: MPS 3.2 s, CPU at 8 threads 5.0 s, CPU at 2 threads 6.1 s,
A10G 0.22 s. The MPS prior assumes 0.75 times the only MPS measurement on this Mac (forward-only
inference, about 1.2 TFLOP/s), i.e. about 10 windows per second; the defensible range is 8-12.
`theta` is not included.

| Phase | Work | MPS | CPU 8 threads | AWS A10G |
| --- | --- | --- | --- | --- |
| Measurement step | section (a) | about 33 GPU-min + 3,578 CPU-s (D-054) | | |
| Tokenizer diagnostic | about 3,100 windows | under 3 min of CPU | | |
| Pilot (fold 1) | <= 3,072 steps | <= 2.9 h (5.7 h if the pilot rule fires) | <= 4.4 h | <= 20 min |
| Pilot check | 504 fans | 0.88 CPU-h | | |
| Folds 2-24 (monthly) | <= 23 x 3,072 steps | 44 h typical (4 pseudo-epochs), 66 h worst | 68 h, 101 h | <= 8 h |
| Folds 2-8 (quarterly) | <= 7 x 3,072 steps | 13 h, 20 h | 21 h, 31 h | <= 2.3 h |
| Seed replicates (D-048 (4)) | 6 fits | <= 17.3 h | <= 27 h | <= 2 h |
| Evaluation fans T-A + T-B | 14,542 fans | 36.0 CPU-h; 4.5 h of wall with 4 workers if they scale | | stays on the Mac |
| Binance copy (T-B secondary, 5.2) | 8,760 fans | 15.3 CPU-h; 1.9 h of wall with 4 workers if they scale | | stays on the Mac |
| Sealed, Aug-Oct 2026 | 3 fits + 1,282 fans | <= 8.6 h + 3.6 CPU-h | | |
| Model to date | 1 fit | <= 2.9 h | | |

Whole monthly program on MPS to the T-B readout (pilot included): about 46 h typical and 69 h
worst case of training before the thermal factor, plus about 36 CPU-hours of fans (51 with the
Binance copy) that may overlap MPS training if the measurement step allows it. The template budget
B_T (20 full epochs) would take 185 h typical and 617 h worst case on MPS: not viable on the Mac.

### 8.3 Gate-3 asks, in order

1. The pilot (fold 1), with the measured estimate.
2. The remaining folds, monthly or quarterly, with the measured fold time, `theta`, the pilot's
   pseudo-epochs, expected and worst totals, the projected minimum detectable difference, the
   seed replicates of D-048 (4), and tier 3, which the owner may add before T-B is read.
3. The evaluation fans (36.0 CPU-hours; 14.2 for R1 alone if the shipped fans already exist), and
   the Binance copy (15.3 CPU-hours; 7.7 for R1 alone).
4. If T-A sets N = 8: the `G8` fans (14.3 CPU-hours).
5. The sealed folds and fans, after the gate-6 order.
6. The model to date.

Every command measures and prints its estimate (`--estimate` stops there), refuses above 3,600
CPU-seconds or one GPU-hour without `--approved-by-owner <hours>` (the figure the owner approved
in chat), refuses when the measured estimate exceeds 1.25 times that figure (D-040), and records
the approval in the ledger.

### 8.4 GPU option (not taken: D-048 (1))

The owner chose the Mac (D-048 (1)). A rented GPU stays documented as an alternative the owner
may still choose, for example at the go/no-go of 6.2.

| Run on an AWS g5.xlarge (1 x A10G 24 GB, about USD 1.01/h on demand, to be checked on the day) | GPU hours, typical / worst | USD, typical / worst |
| --- | --- | --- |
| Walk-forward, 24 folds, budget of R1, plus about 1 h of setup | 6.5 / 9 | about 7 / 9 |
| Walk-forward, 24 folds, template budget B_T (20 full epochs), plus setup | 17 / 53 | about 17 / 54 |
| Seed replicates (6 fits, R1 budget) | about 2 | about 2 |
| Sealed folds and model to date (R1 budget) | about 1.5 | about 2 |
| Five full seeds (4 extra R1 walk-forwards) | 26 / 36 | about 26 / 36 |

A g6.xlarge (1 x L4) costs about USD 0.80/h and is about 1.3 times slower. Spot is about a third
of the price, but an interrupted fold restarts. Requirements and logistics:
- Gate 3 in USD, with the worst case shown, for every run.
- Gate 4: the CUDA build of torch 2.14.1 is not in `uv.lock`, whose Linux resolution uses the CPU
  index (D-001). It is installed on the instance only; the Mac's environment does not change.
- New AWS accounts have an on-demand quota of 0 vCPUs for G instances ("Running On-Demand G and VT
  instances"); a g5.xlarge or g6.xlarge needs at least 4. A quota increase can take hours to days,
  and capacity varies by region and availability zone. The owner requests the quota, launches the
  instance and grants access; no credential or BingX key goes to it, only public market data.
- Procedure: clone at the same source-tree hash; `uv sync` with the CUDA torch build; verified
  `fetch-weights`; the fold CSVs copied over (or rebuilt from data.binance.vision, pre-approved);
  a short GPU timing; the folds with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and deterministic
  algorithms (bit-reproducible on that GPU type); checkpoints and fold records copied back and
  re-verified by sha256 on the Mac before registration (2.4 GB of egress, about USD 0.2). Every
  evaluation fan stays on the Mac CPU.
- Routine monthly retrains of a GPU-validated recipe run on the same instance type (about USD 1
  each), or the device change is recorded and disclosed (3.4).

### 8.5 Disk and memory

- Disk: about 2.4 GB of selected checkpoints (0.6 GB more with replicates), 0.15 GB of CSVs,
  0.2 GB of run scratch at a time, 0.15 GB of fans: about 3.5 GB of the 20 GB free, relocatable
  with `KRONOS_AGENT_WEIGHTS_DIR` and `KRONOS_AGENT_DATA_DIR`.
- RAM: training at batch 32 is estimated at 5-11 GB. With dropout active, the attention may run
  the math kernel, which stores B x H x L x L tensors (0.25 GiB each at batch 32, about 6.8 GiB
  for three per attention module over the 9 modules). Each fan worker needs about 1 GB. Heavy
  phases run one at a time unless the measurement step shows training plus workers within
  12 GiB with no swap growth.

## 9. Risks

1. **A null result is likely.** At lr 1e-6, 3,072 steps move each weight by at most about 1.5e-3
   even if every step agreed. The pilot's go/no-go and the UNINFORMATIVE flag keep such a null
   from being read as "the retrained Kronos has no skill". A well-measured negative is a valid
   result (SPEC 0, rule 6).
2. **Compute on a fanless laptop:** days of runtime, throttling, a busy machine. Mitigations: the
   measurement step, the thermal factor, the quarterly fallback, the GPU option (8.4, if the owner
   chooses it), the fold as the unit of restart, spend control.
3. **MPS:** nondeterministic training; possible refusals under deterministic mode; the
   `multinomial` bug, which the CPU routing removes from training.
4. **Memory:** 5-11 GB at batch 32 on 16 GB shared with fan workers and the other workflow;
   watermark and pressure abort; activation checkpointing before any batch change.
5. **Normalisation look-ahead in R1** (2.2): measured before registration; the first follow-up
   test removes it.
6. **Pretraining overlap:** the first three folds early-stop partly before the cutoff; the
   cutoff check and the post-cutoff sensitivity show its effect.
7. **Binance-trained, BingX-used:** the mismatch study finds that the sources shift the signal
   beyond seed noise (5.2), so the Binance copy is in the plan as a T-B secondary.
8. **Staleness:** the evaluated models are 21 days staler than live ones; the bias is against the
   retrained model.
9. **Short sealed segment:** months, not years; a band reading, with low power.
10. **Repeated use of validation months:** the recipe count and the reality check of 7.4; the
    sealed segment protects the final claim.
11. **Moving code:** the forecasting-code hash changes once with `kronos/decision.py`; the bundle
    freezes it afterwards.
12. **Interpretation:** SPEC 6 is read per model (training on earlier validation months, each
    test out of sample for its own model); the owner chose this reading (D-048 (6)).

## 10. Facts that are easy to get wrong

- A vendored window has `lookback + predict + 1` rows and every position is scored; predict 0
  gives 513 rows and 512 scored positions.
- The vendored normalisation uses all rows of the window, future ones included (FB:125-127); its
  leak is the within-window zero sum, not the size of the shift of the statistics.
- In eval mode the s2 cross-attention is not causal (M:387): the vendored validation loss reads
  its target and is random (K:267-272). NLL4-TF avoids both.
- `DualHead.compute_loss` supports a `padding_mask` (M:494-502); a loss mask needs no new loss.
- The vendored scripts have no early stopping, no MPS path and no date split; OneCycle overwrites
  `adam_beta1`; the LR log prints `0.000000` at 1e-6.
- A decision `D` reads targets `D .. D + (h - 1) h`: for targets strictly before a cut-off `C`, the
  last decision with N = 4 is `C - 4 h`.
- The per-model candle embargo is 512 h, not 520 h: `VALIDATION_START - 512 h = 2024-06-09T16:00Z`
  is the partition's candle boundary.
- BingX 1h history starts 2024-04-27T10:00Z; its first full 512-hour context is at
  2024-05-18T18:00Z.
- Kronos fan samples use 4-week bootstrap blocks (D-035), not calendar weeks.
- Fan time at 2 threads: 3.15 s (20 x 4) and 5.86 s (20 x 8) measured; about 16.7 s for 20 x 24.
- With K = 20 members, central intervals use order statistics `[x_(i), x_(21-i)]`, nominal
  `(21 - 2i) / 21`; linear-interpolated quantiles at 0.05/0.95 cover only about 81 %.
- An evaluation length must reach `forecast_config` and the batch (5.3); a check in
  `forecast_window` alone would store 24-step fans under the 4-step `config_id` and be refused.
- Editing any `FORECASTING_FILES` member changes every `config_id`; stored fans then stay valid
  for their own study but are not reused.
- `manifests/*` enters `source_tree_sha256`: the fine-tune registry lives under `<weights>/`.
- `torch.multinomial` is looked up on the `torch` module at call time (K:271), so routing it is a
  module attribute in the training child; inference fans never see the router.

## (a) The measurement step to run first

**Command:** `kronos-agent finetune timing [--estimate] [--approved-by-owner CPU_HOURS]` (phase
P0). It needs no market data: it uses synthetic GBM candles (`features.synthetic_window`)
exported by the same CSV exporter, the verified shipped weights and the real training child with
every R1 wrapper on and early stopping off. It runs when the machine is otherwise idle, on power,
lid open, under `caffeinate -i -s`. It prints its own estimate first. Without the flag it
refuses to start above 3,000 CPU-seconds or 45 GPU-minutes, and stops if either is passed while
it runs, so it stays below gate 3; such a run is announced in the step summary, not asked. With
MPS the estimate below passes 3,000, so the second run needed the owner's gate-3 approval: up to
1.5 CPU-hours, for that run only, given as `--approved-by-owner 1.5` (D-055). The flag makes the
CPU limit a ceiling of CPU_HOURS x 3600 CPU-seconds that the spend is kept within: the estimate
must fit, units start and the run stops against that ceiling minus a reserve of 600 CPU-seconds
for the unit running then (5,400 and 4,800 for 1.5). The GPU limit stays 45 GPU-minutes.

| Id | Measurement | Configurations | Work | CPU-s | GPU-min |
| --- | --- | --- | --- | --- | --- |
| M1 | Training step time and peak memory | CPU at 2 threads, CPU at 8 threads, MPS (2 CPU threads) x batch 8, 16, 32; smallest batch first | a fresh child per configuration; 3 untimed and 10 timed optimizer steps, W = 513 | about 1,200 (CPU) + 150 (MPS side) | about 2 |
| M2 | Activation checkpointing | batch 32 on the chosen CPU configuration, before any MPS unit, when it does not fit as is (peak above 9 GiB, swap growth or a memory abort); on MPS only when its batch 32 completed as is and does not fit (D-054) | 3 + 10 steps with the wrap; once, before the first configuration that relies on the wrap, the CPU gradient bit-identity of one step with and without it | <= 200 | <= 1 |
| M3 | MPS preflight | MPS at batch 32, with activation checkpointing when the CPU's batch 32 needs it; the checks that train wrap their models on MPS and on the CPU (D-054) | (a) `torch.multinomial` chi-square at the training shape (16,384 rows x 1,024 categories from a real s1 softmax, 64 repetitions; CPU control); (b) CPU/MPS parity of s1 logits and s1-loss gradients on one fixed batch with dropout set to 0 on the instances (max relative difference < 1e-3, gradient cosine > 0.9999); (c) 20 steps without NaN, mean loss within 1 % of the CPU run's; (d) whether deterministic mode refuses an op | about 554 (D-054: check (b)'s and (c)'s 21 steps on the CPU at 9.68 s and 23 MPS steps at 3.2 s, wall x 2 threads) | about 4.6 (the child's whole wall) |
| M4 | Sustained throughput | MPS at the recipe batch, with activation checkpointing as the MPS batch 32 of M1 (D-054) | 20 minutes; thermal factor `theta` = median step time of the last 10 minutes / median of the first 2 | about 120 | 21 (20 timed minutes plus a minute of start-up, warm-up and last step: it starts, or is retried, only when that fits; D-054) |
| M5 | Fan speed | 2 threads, one worker | `t(4)`, `t(8)`, `t(24)`: one warm-up and 3 timed fans each on `timing_window` | about 210 | |
| M6 | Worker scaling | 1, 2, 3 and 4 workers | 12 N = 4 fans per configuration | about 300 | |
| M7 | Contention | 1 and 2 workers during M4 | 12 fans each | about 150 | (in M4) |
| M8 | NLL4-TF time | CPU, 2 threads | 48 synthetic decisions | about 20 | |
| M9 | Identity control and checkpoint round trip | CPU | shipped weights through `save_pretrained`, a scratch registry, `load_models(checkpoint=...)` and a worker: the fan must be bit-identical to the shipped fan | about 20 | |
| | Model loads and child start-ups | | about 15 | about 100 | |
| Retries | One retry with activation checkpointing (D-054) | an MPS unit (M3, an M1 batch, M4) that ended `memory_abort` without it | once, after M2's CPU gradient bit-identity passed; both attempts recorded | about 554 (one retry of M3's size) | about 4.6 |
| | **Total** | | | **about 3,578** | **about 33** |

D-054 amends this section after the first run (D-053): the CPU's M1 and M2 run before any MPS
unit; MPS units start with activation checkpointing where the CPU's batch 32 needs it, and a
memory abort without it is retried once with it; M3 and the retries are priced by their content
and M4 by its whole lifetime. With them the estimate passes the 3,000 CPU-seconds above, so
without the flag the step refuses to start (D-054 (8)). D-055 settles D-054 (8) with a run above
gate 3 under the owner's approval of up to 1.5 CPU-hours, `finetune timing --approved-by-owner
1.5`, and leaves the plan unchanged; the GPU limit stays 45 GPU-minutes.

Recorded for every configuration: median and spread of seconds per step, windows per second,
peak RSS of the child (CPU), the maximum of `torch.mps.driver_allocated_memory()` (MPS), the swap
growth and the highest memory-pressure level, and any refusal. If MPS is unusable, M4 does not
run on CPU (it would pass the hour); the thermal factor then comes from the pilot's first
pseudo-epoch, the median of its second half against its first 50 steps. Output:
`reports/finetune/timing-2.json` and `.md` (device, threads, batch, torch version, platform label;
no path; the first run's `timing.*` hold another plan and stay as D-053's record, D-054 (9)). A
timing record is tied to device, torch, platform and code, and later estimates refuse a stale one.

**What each number decides:**

| Number | Decides |
| --- | --- |
| Seconds per step at batch 32: MPS against the best CPU configuration, sustained | The device: MPS if the preflight passes and it is at least 1.5 times faster; otherwise CPU |
| CPU at 2 against 8 threads | The CPU thread count for training (8 only if at least 1.3 times faster than 2), and whether CPU training leaves cores for fan workers |
| Seconds per step x `theta` | The per-fold time, hence the gate-3 estimates of the pilot and of the walk-forward, and monthly against quarterly under D-048 (1) (monthly when the worst case is at most 72 h) |
| Peak memory at batch 8, 16 and 32, and its slope per window | Whether batch 32 fits (peak <= 9 GiB, no swap growth); otherwise activation checkpointing at 32 (M2), and batch 16 only if that fails (a recipe change before registration). A slope near 0.2 GiB per window shows that the attention probabilities are materialised (math kernel) |
| Step time at batch 8 and 16 | The cost of those fallbacks, and whether a smaller batch would only be slower per window |
| MPS preflight (a)-(c) | Whether MPS may be used at all. The `multinomial` result says whether the CPU routing is a necessity or a precaution; the routing stays on either way |
| MPS deterministic-mode refusal (d) | Whether MPS runs keep deterministic mode on (recorded either way) |
| `t(4)`, `t(8)`, `t(24)` | The fan estimates of T-A, T-B, the pilot check, the N = 8 contingency and the sealed test (replacing `0.44 + 0.678 L`) |
| Worker scaling and contention | The number of fan workers, and whether fan batches may run during MPS training |
| NLL4-TF time | `t_nll` in the fold formula |
| Identity control | That the fine-tune output path cannot itself create a difference; it must pass before any fold is trained |

## (b) Owner decisions

The owner answered the six questions of this design on 2026-11-02 and took every recommended
option. The answers are D-048 in `docs/DECISIONS.md`; the sections named below apply them.

| D-048 | Topic | Decided option | Applied in |
| --- | --- | --- | --- |
| (1) | Compute | The Mac: training runs on its GPU (MPS) after the measurement step of section (a). The walk-forward uses 24 monthly cold-start folds if the measured worst case is at most 72 h (69 h at the priors of 8.2), otherwise 8 quarterly folds over the same test decisions. Every run above about an hour waits for a gate-3 approval of a measured estimate. The rented GPU and the template budget B_T are not taken | 3.2, 3.4, 6.1, 8.2-8.4, (a) |
| (2) | Acceptance of R1 against Kronos as shipped | The 95 % interval of `Delta_N` (the CRPS difference at N = 4) lies entirely below 0, `Delta_N < 0` in at least 5 of the 8 quarters, and the relative improvement `RS_N` is at least 1 % | Summary, 6.5 |
| (3) | Sealed segment and model to date | The sealed segment opens once, only if R1 is accepted, on one recorded order of the owner that also covers training the model to date. Otherwise the model to date is trained up to 2026-08-01 without opening it, and its model card states the negative result | 1.3, 6.6, 6.7 |
| (4) | Training seeds | One training seed per fold, plus two extra seeds (replicates r1 and r2) in the folds of 2024-07, 2025-06 and 2026-06 | 3.4, 5.2, 5.4, 8.2 |
| (5) | Normalisation of the first test | The vendored script's normalisation unchanged, with windows of 513 rows (lookback 512, predict 0), so one future row enters it. If the measurement before registration (Part A of section 4) shows more than 5 % of context tokens changing, the owner is told first. Context-only normalisation is the second test | 2.2, 4, 6.8 |
| (6) | Embargo | 512 h between the end of each fold's training data and its test month: the per-model reading of SPEC section 6 | 1.2, 1.4 |

## (c) Implementation plan

**Sequencing.** The step-2 edits this work waited for (`kronos/batch.py`,
`data/forecast_store.py`, `kronos/decision.py`, `data/partition.py`, `data/store.py` and
`data/windows.py`) have landed, and the 873 D-040 fans were computed on 2026-11-02 (forecast
config `b2723b9626339a08`, `reports/data/mismatch.md`). The `kronos/decision.py` edit (a
`FORECASTING_FILES` member) can therefore land; it lands before any evaluation fan. Nothing in
`third_party/` or `research/` is touched.

**New modules** (`kronos_agent/finetune/`, the name `docs/ARCHITECTURE.md` reserves):

| Module | Content |
| --- | --- |
| `finetune/recipe.py` | Strict schema of `kronos.finetune_recipe`, canonical JSON, `recipe_id`, the vendored YAML writer with every key explicit, the two budgets |
| `finetune/folds.py` | Fold planner (pilot, monthly, quarterly, sealed, final, replicates): CSV end, inner validation, NLL4-TF decisions, test grids, the invariants of 1.4 |
| `finetune/export_csv.py` | Fold CSV through `read_study_frame`; float32 equality; gap scan; split arithmetic; 9973 check; `export.json` |
| `finetune/runner.py` | The child: runtime, vendored import, device subclass, `train_model` wrapper with the bitwise weight check, `create_dataloaders` wrapper with the pseudo-epoch sampler, `GapAwareKlineDataset`, `CleanValidation`, `EarlyStopGate`, the multinomial router, the memory guard, activation checkpointing, `runner_result.json` |
| `finetune/nll.py` | NLL4-TF in the inference geometry, batched, deterministic |
| `finetune/train.py` | Parent side of one fine-tune: export, YAML, environment, spawn, wait, tree verification before and after, registration, spend control |
| `finetune/registry.py` | `<weights>/finetuned/registry.jsonl` with its hash chain, atomic install, verification, the identity control |
| `finetune/timing.py` | Section (a) |
| `finetune/tokenizer_diag.py` | Section 4, token mismatch included |
| `finetune/pilot_check.py` | 6.2 steps 3-4 |
| `finetune/walkforward.py` | Fold loop, blinding, failure policy, resumption, projection-based spend control |
| `finetune/model_card.py` | 6.7 |
| `eval/crps.py` | Fair ensemble CRPS, Gaussian closed form, lazy, spread recalibration |
| `eval/skill.py` | Skill curves, simultaneous band, order-statistic coverage, PIT, reliability, direction, agreement |
| `eval/finetune_report.py` | T-A, T-B and T-C statistics, reading rules, fold bootstrap, quarters, A/A, manipulation check, sealed band, reality check, reports and figures (D-034, D-037) |
| `ledger.py`, `budget.py` | Section 7.3; `partition._verify_order` delegates to the ledger |
| `cli/finetune.py`, `cli/ledger.py` | The commands below, registered in the strict registry (D-018) |

**Changes to existing modules** (additive; governed defaults unchanged):
- `kronos/weights.py`: `load_models(name, *, checkpoint=None)`; `LoadedModels.checkpoint` and
  `data_end_ms`.
- `kronos/decision.py`: `forecast_window(..., evaluation_length=None)` as in 5.3;
  `require_model_period` in place of `require_kronos_period`.
- `kronos/batch.py`: `evaluation_length` and `checkpoint` through `prepare_tasks`,
  `timing_window`, `measure_seconds_per_fan`, `run_fan_batch` and the worker initialiser;
  `registry_config(kronos, checkpoint=None)`; spend control during a batch.
- `data/partition.py`: `EVAL_LENGTH_MAX`, `check_evaluation_length`, `require_model_period`;
  `_verify_order` through the ledger.
- `params.yaml`: `kronos.finetune_recipe` written when the bundle is recorded; `params.py`
  validates it through `finetune/recipe.py`.
- `provenance.py` and `data/forecast_store.py`: no change (`CONFIG_KEYS` and `META_KEYS` stay).

**Commands** (exit codes as D-039: 0 done, 1 `ERROR:`, 2 `REFUSED:`):
- `kronos-agent finetune timing`
- `kronos-agent finetune tokenizer-diag --part training|validation [--record-config]`
- `kronos-agent finetune fold --fold <label> [--replicate K] [--estimate] [--approved-by-owner H]
  [--order ID]` (the pilot is `--fold wf-2024-07`)
- `kronos-agent finetune pilot-check [--record-config]`
- `kronos-agent finetune register [--record-config]` (the bundle and its ledger entry)
- `kronos-agent finetune walk-forward --layout monthly|quarterly [--estimate]
  [--approved-by-owner H]` (folds 2 onwards, blinded output, resumable)
- `kronos-agent finetune evaluate --set ta|tb|g8|binance|tier3|replicates|sealed [--workers N]
  [--estimate] [--approved-by-owner H] [--order ID]` (fans only, no statistic)
- `kronos-agent finetune report --test T-A|T-B|T-C` (ledger entry and budget, then statistics,
  figures and unblinding)
- `kronos-agent finetune sealed --order ID [--estimate] [--approved-by-owner H]`
- `kronos-agent finetune final [--order ID] [--estimate] [--approved-by-owner H]`
- `kronos-agent finetune card --checkpoint ID`
- `kronos-agent ledger show | verify | record-order --scope open_sealed --candidate ID --text
  "<owner's words>"`

**Tests** (pytest; the `weights` and `network` markers as today; tiny random Kronos models keep
most tests fast and weight-free):

Training data and look-ahead:
- T1 Equivalence: with every wrapper off, the child reproduces `python -B train_sequential.py
  --skip-tokenizer` bit for bit on a tiny CSV (CPU, same seeds and threads); the vendored tree
  verifies afterwards.
- T2 CSV boundary: for every planned fold on a synthetic store, no row at or after the CSV end;
  the vendored train split ends before `csv_end - 1,344 h` and the validation split is exactly the
  inner validation.
- T3 Invented future candles (SPEC 5, applied to training): changing every price after the CSV
  end by a factor of 10 and every volume by 0.1 leaves the CSV, `export.json`, the NLL4-TF
  decisions and values, and a tiny CPU checkpoint bit-identical.
- T4 Window contents: each training window's tensor is the vendored normalisation of rows
  `[s, s + 513)` only (positive control: the vendored look-ahead is detected); for the
  context-only variant, the per-position invariance holds on a tiny model.
- T5 Split arithmetic and the 9973 refusal; T6 test embargo and model periods; T7 the sealed
  guard (refused without access, accepted with a monkeypatched recorded order); T8 gaps (exactly
  the windows spanning a gap are dropped, nothing fabricated); T9 `amount` float32 equality and a
  detected 1-ulp change; T10 no `finetune` module calls `read_store` (syntax tree).

Training mechanics:
- T11 Causal validation diagnostic (the eval-mode leak exists without the hook and not with it;
  dropout stays off); T12 reseeding (the same loss twice; RNG states unchanged after a pass; CPU
  training identical with and without the hooks).
- T13 Early stopping on scripted metrics (stop point, selected state, OneCycle sized by
  `E_max x 512` and never stepped past its total); T14 pseudo-epoch sampler (exactly 512 batches,
  no repeat inside a pseudo-epoch, order from its own generator); T15 multinomial router (indices
  on the input's device, CPU generator, inference untouched).
- T16 NLL4-TF: with `sample_from_logits` patched in the test to return the true tokens, the logits
  inside `auto_regressive_inference` equal NLL4-TF's bit for bit at every step; equal values in two
  processes with the same threads.
- T17 Activation checkpointing (CPU gradients bit-identical); T18 bitwise weight check refuses a
  perturbed tensor; T19 memory guard and spend control on simulated readings; T20 determinism (two
  CPU runs of a tiny fold give the same checkpoint sha256).

Registry, fans and governance:
- T21 Registry (a registered checkpoint loads; a changed byte, size or config is refused; a broken
  chain is refused; its `config_id` differs from the shipped one); T22 identity control
  (`weights` marker).
- T23 Evaluation length (24 allowed, 25 refused; the purge at 2026-07-09T16:00Z; a 24-step fan
  records `horizon_n = 24` and its own `config_id`; `check_embargo` still sees N; the governed
  configuration keeps its keys and values except the code hash).
- T24 CRPS (fair estimator against brute force; Gaussian closed form against numerical
  integration; lazy = `|y|`; order-statistic coverage levels); T25 reading rules on synthetic
  `d(D)` (ACCEPT, below the smallest effect, WORSE, NOT DISTINGUISHABLE, FRAGILE, UNINFORMATIVE;
  the sealed band); T26 reality check on synthetic recipes.
- T27 Ledger and budget (a 4th test in an ISO week and a 16th in a quarter refused; a missing or
  changed bundle refused; a second reading refused; the chain detects an edited line; an order is
  found by id and scope and used once).
- T28 Blinding (the walk-forward's printed output and logs it shows contain no loss value or
  selected pseudo-epoch for folds 2 onwards); T29 hygiene (reports carry no absolute path from the
  vendored logs, no personal data).

**Docs.** `docs/RETRAINING.md` (this file); in `docs/DECISIONS.md`, the owner's answers (D-048)
and, under new numbers after the last existing one, the design decisions of this file, the
pre-cutoff opt-ins of 6.2 and 6.4, and the deviation from the subprocess-only plan where it
applies; `CLAUDE.md` lines (the facts of section 10, the commands, the blinding rule, `python -B`
for the child); the step-6 row of `docs/ARCHITECTURE.md`; a README section; the model card
template.

**Order of work** (a short Spanish summary with numbers and `git status` after each phase):
1. Code and tests needed by the measurement step: exporter, child runner, NLL4-TF, registry,
   `load_models(checkpoint)`, the evaluation-length change, the timing command (synthetic data
   only).
2. The measurement step (P0): the first run below gate 3 (D-053), the second above it under the
   owner's approval of up to 1.5 CPU-hours (`--approved-by-owner 1.5`, D-055). Device, threads,
   batch and memory choices recorded.
3. The rest of the code and tests: ledger and budget, tokenizer diagnostic, pilot check,
   walk-forward, evaluation and reports, model card.
4. Data in place (step 2, done on 2026-11-02): the Binance and BingX stores, the coverage study
   (no gap in `binance/1h` or `bingx/1h` before the sealed start, so no walk-forward fold loses a
   window to a gap), the mismatch report (it triggered the D-040 contingency: 5.2).
5. Tokenizer diagnostic, Part A.
6. The owner's answers: received on 2026-11-02 (D-048).
7. Pilot (gate 3), pilot rule, pilot check, go/no-go, compute control (gate 3 for the remaining
   folds).
8. Pre-registration bundle for T-A and T-B; tokenizer diagnostic Part B.
9. Folds 2 onwards (blinded) and the evaluation fans (gate 3), overlapping only if P0 allows it.
10. `finetune report --test T-A`; the N rule; `G8` fans if needed (gate 3); `finetune report
    --test T-B`.
11. If ACCEPT: the owner's gate-6 order, sealed folds and fans (gate 3), `finetune report --test
    T-C`.
12. The model to date (gate 3) and its model card. Shadow mode and deployment belong to step 8.
