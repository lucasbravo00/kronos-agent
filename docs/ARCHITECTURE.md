# Architecture

Package `kronos_agent` (src layout) with a console script `kronos-agent <command>`; each command
is one module under `kronos_agent/cli/`. Configuration lives in `params.yaml` (sections
`criterion`, `kronos`, `costs`) and is loaded by `kronos_agent.params`.

## Step-1 modules

| Module | Responsibility |
| --- | --- |
| `runtime.py` | Sets `HF_HUB_OFFLINE=1` and the thread environment BEFORE `torch`/`huggingface_hub` are imported; fixes `torch` threads; deterministic algorithms; refuses reconfiguration with another thread count. |
| `paths.py` | Repository root, `weights/`, `data/` and `reports/` with `KRONOS_AGENT_WEIGHTS_DIR` / `KRONOS_AGENT_DATA_DIR` / `KRONOS_AGENT_REPORTS_DIR` overrides (the reports override since D-057 (1)); `ledger_dir()` (`<reports>/ledger`); home-directory redaction; report labels (`default_label`, `validate_label`). |
| `hashing.py` | sha256 of files and bytes; git blob sha1; canonical JSON. |
| `manifests.py` | GNU `sha256sum`-format manifests: generate, parse, verify (strict: extra files, bytecode caches included, are modifications). |
| `seeds.py` | Seeds derived from the sha256 of the decision hour. |
| `params.py` | Loads and validates `params.yaml`; `params_version()`. |
| `models.py` | `MODEL_SPECS`: repo ids, tokenizer pairing, `max_context`, `state_dict` element counts (verified against the HF API on 2026-10-29); the single source of truth for the pairing. |
| `exchange/public_market.py` | Real-market data client (public GET, no keys, host pinned to `open-api.bingx.com`); `closed_klines` and `fetch_twice` select the half-open `[start_ms, end_ms)` whatever the exchange does at that edge (D-047). |
| `exchange/demo_account.py` | Authenticated demo client (host pinned to `open-api-vst.bingx.com` and compared on the ASCII wire form; inline HMAC signing; only `httpx.MockTransport` injectable; parameters restricted to `[A-Za-z0-9._-]` or printable-ASCII JSON; `httpx` log records naming the demo host dropped). |
| `kronos/vendored.py` | Verifies the vendored tree, then imports the upstream `model` package without writing bytecode and restores `sys.path`; upstream code is importable only through `import_kronos()`, in every process. |
| `kronos/weights.py` | Pre-registered weights registry, verified download, offline strict load with `.eval()` and no gradients on CPU; the predictor clip `PREDICTOR_CLIP`; `inference_problems` (models that would not forecast as loaded). |
| `kronos/features.py` | `amount` proxy and window validation. |
| `kronos/forecast.py` | Fan of trajectories via `predict_batch`. |
| `envinfo.py`, `bench.py` | Redacted environment report; 20 x 8 forecast benchmark. |

## Step-2 modules

The data layer, stored forecasts and the step-2 studies; `docs/DATA.md` is their design and D-022
to D-049 of `docs/DECISIONS.md` record the decisions (D-030 and D-031 withdrawn by D-042).

| Module | Responsibility |
| --- | --- |
| `provenance.py` | Code provenance `{commit, dirty, source_tree_sha256}` for metas, reports and fans; the forecasting-code hash (whole files, the models' construction in `kronos/weights.py` and `kronos/vendored.py` included since D-047) that enters the forecast `config_id`. |
| `data/timeutil.py`, `data/partition.py` | UTC hours and epoch milliseconds (standard library only); the partition by decision hour, the horizon purge, the embargo check, the Kronos cutoff and the sealed guard (`SealedAccess` only through `sealed_access`: the owner's order in the experiment ledger, read from a ledger that may open that data root, and valid for that data root only; D-057 (4), D-058 (3)(c)). |
| `data/fsutil.py` | Durable atomic writes, never-overwrite publication (`os.link`), the non-blocking per-source lock (exclusive for runs, shared for reads), the temporary files of killed writes. |
| `data/store.py` | Candle stores: CSV of the exchange's decimal strings plus a meta (gaps, OHLC violations, sha256, provenance); `read_study_frame` (the study path, behind the sealed guard) and `read_store` (integrity work only), both under the source's job lock held shared (`store_read_lock`); `job_lock` (runs; removes the temporaries of killed runs), `leftover_temporaries` and `store_status` (size and sha256 against the meta). |
| `data/binance_vision.py`, `data/resample.py` | data.binance.vision client (host-pinned GET), archive plans, verified downloads under the gate-4 budget, strict parsing, the Binance stores, exact decimal resampling and the check against Binance's own 1h archives. |
| `data/bingx_history.py` | BingX real-market backfill: epoch-aligned 1000-candle chunks, settled candles only, two fetches that must match, a never-deleted chunk archive whose candles never change (`archive_only` keeps what rolled off; chunks stay incomplete until the 1m or 5m retention explains the candles a refetch no longer returns), the revision record (`raw/bingx/revision-<interval>.json`: each archived candle the exchange later served with other values, once, with its first and last observation, while the archive keeps the first-seen row, D-045), the history-start record, the merge into `candles/bingx/<interval>` and the archive warnings. |
| `data/windows.py` | Decision windows (one builder for the store and the live path), their hashes, the live timing and fetch-pair rules, saved windows (their inputs only). |
| `data/forecast_store.py` | Stored fans (`.npz` without pickles), the forecast `config_id`, the reuse check (window hashes and `context_sha256`). |
| `data/coverage.py`, `data/mismatch.py` | Study 1 (coverage) and study 3 (Binance vs BingX mismatch, its Kronos part included). |
| `kronos/decision.py`, `kronos/batch.py` | One fan of a decision window with the governed parameters and its signal; batches with windows built in the parent, spawn-started workers and gate 3 (the approved CPU-hours and their 1.25 tolerance). |
| `eval/bootstrap.py` | Week-block bootstrap intervals with seeds derived from the statistic's key, Wilson intervals for 0/1 shares, `n_eff`. |
| `report/writer.py`, `report/figures.py` | Recorded study configs, report headers, guarded JSON + Markdown writing, per-hour series, append-only record files (`<name>.jsonl`); deterministic PNG figures (UTC date axes). |
| `cli/data.py`, `cli/forecast.py`, `cli/mismatch.py` | Commands `data`, `forecast` (and the gate-3 batch flow), `mismatch`; exit codes 0 / 1 failure / 2 refusal. |

## Step-6 modules (retraining, phases 1 and 3a)

Retraining Kronos-small for BTC (SPEC section 9, D-042 (3)): `docs/RETRAINING.md` is the design,
D-048 the owner's answers, D-052 the phase-1 build, D-053 to D-056 the measurement step (its
second run chose MPS, D-056) and D-057 and D-058 phase 3a. Phase 1 holds what the measurement
step needs, phase 3a what the pilot needs; the rest of the design's plan (c) was not built: the
project closed first (D-063; the step-6 row below).

| Module | Responsibility |
| --- | --- |
| `finetune/recipe.py` | Recipe schema (`R1`) and `recipe_id` (sha256 of the canonical mapping, the values the vendored code hard-codes included; the compute choices outside it, the batch size inside), training seeds, the run YAML with every key the vendored loader reads. |
| `finetune/folds.py` | Fold plan: monthly, quarterly (the walk-forward's layout, D-056; `q-2024-07` is the pilot, its CSV `wf-2024-07`'s), sealed, model-to-date and synthetic folds; CSV end `T - 512 h`, the inner validation (the CSV's last 1,344 hours), the NLL4-TF decisions, the test grids `G4`, `G8`, `G25`; `check_fold` (a broken partition rule is a `PartitionError`); the seed replicates of the monthly folds (the quarterly mapping waits for the owner, D-057 (9)(d)). |
| `finetune/export_csv.py` | `export_fold`: the fold's `train.csv` (the store's strings and the inference `amount` proxy, float32-equal to the inference path; gaps, the vendored split arithmetic, the 9973 rule), `nll_inputs.npz` and `export.json` (written last) under `<data>/finetune/<recipe8>/<fold>/`, read through `read_study_frame` behind the sealed guard and never overwritten. |
| `finetune/nll.py` | NLL4-TF: the teacher-forced NLL of the next 4 candles in the inference geometry (context-only statistics, target tokens from the causal encoder, a length-1 s2 query), deterministic on the CPU. |
| `finetune/runner.py` | The training child, `python -B -P -m kronos_agent.finetune.runner --run-dir DIR`: verifies the vendored tree and its inputs and runs the pinned predictor phase (`train_basemodel_phase`) unmodified behind the switchable wrappers of design 3.1 (device subclass, CPU multinomial router, causal reseeded validation diagnostic, NLL4-TF early stopping, pseudo-epochs, gap-aware dataset, bitwise weight check), with activation checkpointing, the memory guard and the deadline; modes `train`, `timing`, `preflight`; always writes `runner_result.json`. |
| `finetune/train.py` | The parent of one child: verified export and weights, `run.yaml` and `run.json` in `<weights>/finetuned/.runs/<run>/`, an allow-listed environment, timeout and termination, the vendored tree verified again afterwards; `fit` registers only a recipe run. |
| `finetune/registry.py` | `<weights>/finetuned/registry.jsonl`: append-only, hash-chained canonical JSON (33 payload keys, never a selection value); checkpoints installed atomically, under a lock, into `<weights>/finetuned/<checkpoint_id>/`; `CheckpointRef(id, root, line_sha256)`, whose recorded line protects the chain's last entry. Standard library only: `kronos.weights` imports it on the fan path. |
| `finetune/timing.py` | The measurement step `kronos-agent finetune timing [--estimate] [--approved-by-owner CPU_HOURS]` (M1 to M9 within its budget: its own below gate 3, or a CPU ceiling the owner approved, D-055; the compute choice) and `reports/finetune/timing-2.json` and `.md` (D-054; it chose MPS, D-056; the first run's `timing.*` stay as D-053's record); `require_fresh_timing` refuses a stale record. |
| `cli/finetune.py` | `kronos-agent finetune` `timing`, `fold`, `pilot-check` and `tokenizer-diag`; every subcommand but `timing` verifies every recorded pre-registration bundle first and records what it ran in the experiment ledger. |
| `ledger.py`, `budget.py`, `cli/ledger.py` | The experiment ledger `<reports>/ledger/experiments.jsonl` (six kinds, every line verified on read, never repaired, bound to the data roots it may authorize: `binding_problem`), the test budget `spend_test` (3 per ISO week, 15 per quarter, the order of tests; no command calls it yet) and `kronos-agent ledger show [--kind KIND] / verify / record-order` (D-057 (2) to (4), D-058 (3)). |
| `finetune/tokenizer_diag.py` | Design section 4: `finetune tokenizer-diag --part training` or `--part validation` (the token mismatch of D-048 (5), the flags, the block-bootstrap intervals; Part B after the bundle; D-057 (5)). |
| `finetune/pilot_check.py` | Design 6.2 steps 3-4: `finetune pilot-check` (504 fans and the KL on the registered pilot, GO or NO-GO, its spend stop and the ledger's record of a stopped run; D-057 (6), D-058 (3)(j)). |
| `finetune/walkforward.py` | The per-fold path `finetune fold`: quarterly labels, the estimate on `timing-2.json`, gate 3 and the spend cap, the figure lines, the pilot rule, the stages and bookkeeping, the failure policy, the refusal of a training child left running, blinding (D-057 (7), D-058 (3)); the loop over folds 2 onwards comes later. |
| `eval/crps.py` | The scores of design 5.1: `fair_crps` (the fair ensemble CRPS in its sorted pair-distance form), `gaussian_crps` (the no-skill reference) and `lazy_crps` (D-058 (5)); in neither code hash. |
| `tests/root_guard.py` | The test suite's guard of the real `data/`, `weights/` and `reports/` (registered by `tests/conftest.py`): each test's own temporary roots, read-only views (`weights` marker, `real_roots`), refused writes, and the listing compared at session end (D-057 (1)). |

Changed for step 6 (additive): `kronos/weights.py` (`load_models(name, *, checkpoint=None)`
loads a registered checkpoint with its own hashes and `data_end_ms`; `checkpoint_dir_problems`),
`kronos/decision.py` (`forecast_window(..., evaluation_length, period_role)`), `kronos/batch.py`
(the evaluation length, checkpoint and role through `prepare_tasks` and `PreparedBatch`, the
pinned registry line, `time_fan_workers`, `identity_control`) and `data/partition.py`
(`EVAL_LENGTH_MAX` = 24, `check_evaluation_length`, `require_model_period`; since phase 3a
`sealed_access(order_id, *, ledger_root, data_root)` through the ledger, its `SealedAccess` tied
to its data root, which `data/store.py`'s `read_study_frame` checks), `paths.py`
(`KRONOS_AGENT_REPORTS_DIR`, `ledger_dir()`) and `cli/__init__.py` (the `ledger` command).

## Where steps 3-8 plug in (no restructuring)

| Step | New code | Uses from steps 1-2 |
| --- | --- | --- |
| 3 evaluation | `kronos_agent/eval/` (crps, built in step 6 as `eval/crps.py`, D-058 (5); skill_curve, calibration, comparators) next to `eval/bootstrap.py`; reports and figures through `kronos_agent/report/`; command `evaluate` | stored fans (`.npz` with seeds, config id and params version), `kronos.batch`, `data.partition` (the horizon purge, with an explicit evaluation length for horizons beyond N: D-028) |
| 4-5 N0 (not built: planned for the discarded volume criterion, D-042; closed with the project, D-063) | `kronos_agent/indicators/atr.py` (ATR moved here from step 2: it belongs to the criterion), `kronos_agent/criterion/` (events, backtest, controls, multiple_comparisons), `goals.yaml` (a change of it counts as a test, SPEC rule 8); commands `events`, `backtest`. The test budget, the experiment ledger and the `ledger` command they were to build are built in step 6 (D-057) | `params.criterion`, partition, the stores |
| 6 fine-tune (D-042; closed by D-063) | Phases 1 and 3a built ("Step-6 modules", D-052, D-057, D-058): one child `python -B -P -m kronos_agent.finetune.runner` per fine-tune, which runs the pinned `finetune_csv` predictor phase unmodified behind the design's wrappers (not the scripts driven by path); checkpoints in `<weights>/finetuned/<checkpoint_id>/`, registered in the hash-chained `<weights>/finetuned/registry.jsonl` (not a manifest: `manifests/*` enters the source-tree hash, D-025). Phase 3a adds the experiment ledger and the test budget, the tokenizer diagnostic, the pilot check, the per-fold path `finetune fold` and the fair CRPS. Not built at the close (D-063; `docs/RETRAINING.md` (c)): `finetune/` model_card and the loop over folds 2 onwards in `walkforward.py`; `eval/` the spread recalibration and the order-statistic coverage, skill, finetune_report; the `finetune` subcommands `register`, `walk-forward`, `evaluate`, `report`, `sealed`, `final` and `card` | `kronos.weights` (same local-dir layout, `load_models(checkpoint=...)`), `kronos.vendored`, `kronos.batch`, `data.store.read_study_frame`, `data.partition` |
| 7 ladder | `kronos_agent/ladder/` (n1_logistic, n2_head; N2 reads `Kronos.decode_s1` hidden state through a wrapper) | `kronos.vendored`, fans dataset |
| 8 live | `kronos_agent/live/` (decide, shadow, reconcile, risk, killswitch), `deploy/` (systemd timer, healthchecks), `kronos_agent/review/weekly.py`, `docs/LIVE_EVALUATION.md`; the demo client gains `place_order` (mandatory `positionSide`, never `reduceOnly`), `cancel_order`, `query_order`, `set_leverage` | `exchange.demo_account`, `ledger`, `seeds`, benchmark numbers |

Package names in use (`data`, `eval`, `finetune`, `report`; the modules `ledger` and `budget`)
or reserved (do not reuse): `model` (the vendored upstream package), `indicators`, `criterion`,
`ladder`, `live`, `review`.

Live-path import discipline: `kronos_agent.live` imports only numpy, pandas, torch, httpx and
pyyaml (no matplotlib) to keep memory low on the 2 vCPU / 1.9 GB server.

## Ledger interface (the experiment ledger, built in step 6, D-057 (2))

`<reports>/ledger/experiments.jsonl` (`kronos_agent.ledger`; `paths.ledger_dir()`): append-only
jsonl. Every line is canonical JSON (sorted keys, no whitespace, ASCII, LF-terminated, at most
65,536 bytes) with exactly `schema` (`experiment`), `schema_version` 1, `seq` (the line number),
`ts` (`YYYY-MM-DDTHH:MM:SSZ`, never earlier than the previous line's), `prev_sha256` (sha256 of
the previous line's bytes without its LF; genesis `"0" * 64`), `kind` (`exploration`, `fit`,
`gate3_approval`, `preregistration`, `owner_order`, `test`), `params_version`, `code`
(`provenance.code_version()`: commit, dirty, source tree sha256) and `payload` (exactly its
kind's keys). There is no `verify_chain`: `read_ledger` and `ledger_status` verify every line;
appends hold `<ledger dir>/.lock`, and nothing is ever repaired. `binding_problem` says which
data roots a ledger may authorize: only the repository's `reports/ledger` authorizes the
repository's `data/`. The `trade` schema and the raw trade ledger (balances) in `data/live/`,
with a sanitised export published, belong to step 8 and are not built.
