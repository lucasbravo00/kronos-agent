# CLAUDE.md - kronos-agent

Read this file and `docs/SPEC.md` (the owner's full specification, in Spanish) at the start of
EVERY session, before anything else. The SPEC is the only source of requirements; this file is the
operational summary plus the facts that are easy to get wrong.

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
  `docs/SPEC.md`, the owner's original document, which stays verbatim.
- Design questions are collected in ONE batch at the end of step 1 (SPEC section 18); until
  answered, the initial values of `params.yaml` apply. Decisions and the owner's answers live in
  `docs/DECISIONS.md`.
- Every step ends with a short Spanish summary with numbers. A step does not start before the
  previous one meets its "done when" (SPEC section 18).

## Another session's folder

- `research/strategy/` is where the owner, in a separate session, defines the trading criterion.
  Never read, modify or delete anything in it; it is gitignored and excluded from tests and lint.
  Until the owner says the criterion is defined, SPEC section 4 is under review: do not register
  event definitions and do not freeze `params.yaml[criterion]` (steps 1-3 and the section-9
  retraining go ahead).

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
  repo or in generated reports (`tests/test_repo_hygiene.py` enforces it).

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
  anything that imports the vendored code outside `import_kronos()` (step-6 `finetune_csv`
  subprocesses, upstream tests) with `python -B`.
- Weights live under `weights/` (relocatable with `KRONOS_AGENT_WEIGHTS_DIR`) as local directories
  (`config.json` + `model.safetensors`), are verified against the pre-registered
  `manifests/weights.json` (size + sha256) before loading (`manifests/weights.sha256` is a derived
  GNU manifest for `shasum -a 256 -c`) and are loaded with `HF_HUB_OFFLINE=1` set before any import: call
  `kronos_agent.runtime.configure_runtime()` before importing `torch` or `huggingface_hub`.
  Downloads only through `kronos-agent fetch-weights`, which verifies sha256 against the
  pre-registered `manifests/weights.json`.
- Comparable inference: CPU only, `torch.set_num_threads(2)` (the server's vCPU count),
  `device="cpu"` passed explicitly, both models in `.eval()`, deterministic algorithms on. MPS is
  not a reference device (documented nondeterministic ops and an open `multinomial` correctness
  bug); it may only be used for training (step 6) and for informational benchmarks.
- Seeds derive from the sha256 of the decision hour (`kronos_agent.seeds`); every forecast
  records seed, threads, torch version, model name and weight hashes.
- Data lives in `data/` (relocatable with `KRONOS_AGENT_DATA_DIR`). Both folders are gitignored.

## Kronos facts verified at the pinned commit (do not "fix" them from memory)

- Call `.eval()` on the tokenizer and the model yourself: the pinned `huggingface_hub==0.33.1`
  does not call it on the safetensors path, `KronosPredictor` does not either, Kronos-small has
  dropout 0.1/0.25/0.25/0.1 and cross-attention uses `is_causal = self.training`.
- `predict(sample_count=K)` returns the MEAN of K sampled paths. The fan of K trajectories is
  `predict_batch([df]*K, [x_ts]*K, [y_ts]*K, pred_len=N, sample_count=1)`.
- `KronosPredictor(device=None)` selects MPS on Apple silicon: always pass `device="cpu"`.
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
  gap).
- Real 1h history starts 2024-04-27T10:00Z; 1m starts ~2025-07-31 and 5m ~2026-09-14 (rolling
  retention).
- Demo host `https://open-api-vst.bingx.com` only (no `.pro` fallback). Signing: HMAC-SHA256 hex
  of the ASCII-sorted, unencoded `key=value&...` string including `timestamp`; header
  `X-BX-APIKEY`.
- Hedge mode: every order carries `positionSide` (BingX defaults a missing one to LONG); never
  send `reduceOnly`; `clientOrderId` only for LIMIT/MARKET; positions `code 109500` means the
  query failed, not "flat".
- Fees taker 0.05 % / maker 0.02 %; funding every 8 h at 00/08/16 UTC for BTC-USDT (read
  `fundingIntervalHours` from `premiumIndex` at runtime).

## Git

- Exactly one commit (the initial skeleton) until the owner says otherwise. Never push. Show
  `git status` in step summaries.

## Commands

`uv sync --frozen` - `uv run pytest` (markers `weights` and `network` skip with an explicit reason
when prerequisites are absent; network tests are opt-in via `KRONOS_AGENT_NETWORK_TESTS=1` and
touch public hosts only) - `uv run kronos-agent env-report | verify | fetch-weights | bench |
vendor-kronos | params`. The full suite peaks at about 1.25 GiB of RAM (measured 2026-10-31): on
the 1.9 GB server run `uv run pytest -m "not weights"` until the 4 GB swap exists.
