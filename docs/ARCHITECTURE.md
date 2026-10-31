# Architecture

Package `kronos_agent` (src layout) with a console script `kronos-agent <command>`; each command
is one module under `kronos_agent/cli/`. Configuration lives in `params.yaml` (sections
`criterion`, `kronos`, `costs`) and is loaded by `kronos_agent.params`.

## Step-1 modules

| Module | Responsibility |
| --- | --- |
| `runtime.py` | Sets `HF_HUB_OFFLINE=1` and the thread environment BEFORE `torch`/`huggingface_hub` are imported; fixes `torch` threads; deterministic algorithms; refuses reconfiguration with another thread count. |
| `paths.py` | Repository root, `weights/` and `data/` with `KRONOS_AGENT_WEIGHTS_DIR` / `KRONOS_AGENT_DATA_DIR` overrides; home-directory redaction; report labels (`default_label`, `validate_label`). |
| `hashing.py` | sha256 of files and bytes; git blob sha1; canonical JSON. |
| `manifests.py` | GNU `sha256sum`-format manifests: generate, parse, verify (strict: extra files, bytecode caches included, are modifications). |
| `seeds.py` | Seeds derived from the sha256 of the decision hour. |
| `params.py` | Loads and validates `params.yaml`; `params_version()`. |
| `models.py` | `MODEL_SPECS`: repo ids, tokenizer pairing, `max_context`, `state_dict` element counts (verified against the HF API on 2026-10-29); the single source of truth for the pairing. |
| `exchange/public_market.py` | Real-market data client (public GET, no keys, host pinned to `open-api.bingx.com`). |
| `exchange/demo_account.py` | Authenticated demo client (host pinned to `open-api-vst.bingx.com` and compared on the ASCII wire form; inline HMAC signing; only `httpx.MockTransport` injectable; parameters restricted to `[A-Za-z0-9._-]` or printable-ASCII JSON; `httpx` log records naming the demo host dropped). |
| `kronos/vendored.py` | Verifies the vendored tree, then imports the upstream `model` package without writing bytecode and restores `sys.path`; upstream code is importable only through `import_kronos()`, in every process. |
| `kronos/weights.py` | Pre-registered weights registry, verified download, offline strict load with `.eval()` on CPU. |
| `kronos/features.py` | `amount` proxy and window validation. |
| `kronos/forecast.py` | Fan of trajectories via `predict_batch`. |
| `envinfo.py`, `bench.py` | Redacted environment report; 20 x 8 forecast benchmark. |

## Where steps 2-8 plug in (no restructuring)

| Step | New code | Uses from step 1 |
| --- | --- | --- |
| 2 data | `kronos_agent/data/` (binance_vision, bingx_history, resample, store, windows, partition), `kronos_agent/profile/` (volume_profile, levels), `kronos_agent/indicators/atr.py`; commands `data`, `profile`, `forecast` | `exchange.public_market` (paged backfill), `hashing` (window sha256), `kronos.forecast`, `seeds`, `paths` |
| 3 evaluation | `kronos_agent/eval/` (crps, skill_curve, calibration, comparators, bootstrap), `kronos_agent/report/figures.py` -> `reports/figures/`; command `evaluate` | stored fans (`.npz` with seed and params version) |
| 4-5 N0 | `kronos_agent/criterion/` (events, backtest, controls, multiple_comparisons), `goals.yaml`, `kronos_agent/budget.py` (3/week, 15/quarter), `kronos_agent/ledger.py` (append-only hash-chained jsonl, see below); commands `events`, `backtest`, `ledger` | `params.criterion`, partition |
| 6 fine-tune | `kronos_agent/finetune/` (export_csv, run_finetune_csv, model_card) driving `third_party/kronos/finetune_csv/*.py` by subprocess with a generated YAML; fine-tuned weights under `weights/finetuned/<exp>/` registered in a second manifest | `kronos.weights` (same local-dir layout) |
| 7 ladder | `kronos_agent/ladder/` (n1_logistic, n2_head; N2 reads `Kronos.decode_s1` hidden state through a wrapper) | `kronos.vendored`, fans dataset |
| 8 live | `kronos_agent/live/` (decide, shadow, reconcile, risk, killswitch), `deploy/` (systemd timer, healthchecks), `kronos_agent/review/weekly.py`, `docs/LIVE_EVALUATION.md`; the demo client gains `place_order` (mandatory `positionSide`, never `reduceOnly`), `cancel_order`, `query_order`, `set_leverage` | `exchange.demo_account`, `ledger`, `seeds`, benchmark numbers |

Reserved names (do not reuse): `model` (the vendored upstream package), `data`, `profile`,
`indicators`, `criterion`, `eval`, `report`, `finetune`, `ladder`, `live`, `review`, `ledger`.

Live-path import discipline: `kronos_agent.live` imports only numpy, pandas, torch, httpx and
pyyaml (no matplotlib) to keep memory low on the 2 vCPU / 1.9 GB server.

## Ledger interface (implemented in step 5)

Append-only jsonl. Every line is canonical JSON (`sort_keys=True`, separators `(",", ":")`,
`ensure_ascii=True`) with `schema` (`experiment` | `trade`), `ts` (UTC ISO), `prev_sha256` (sha256
of the previous line's bytes; genesis `"0" * 64`), `params_version`, `commit` and the payload.
`verify_chain(path)` recomputes the chain. The raw trade ledger (balances) lives in `data/live/`;
a sanitised export is published.
