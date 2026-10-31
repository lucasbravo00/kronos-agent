# kronos-agent

Research system that combines **Kronos** (a decoder-only foundation model for candlestick
sequences, [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos), MIT) with the owner's
discretionary criterion (volume discrepancies at volume-profile levels) to trade **BTC-USDT
perpetual on a BingX DEMO account**. The question it tries to answer, one parameter at a time
and with explicit multiple-comparison corrections, is whether that combination has an edge out of
sample and whether the edge survives a live market. A well-measured "no" is a valid result.

> **Research project. Demo account (virtual funds) only. Nothing here is financial advice.**

Status: step 1 of 8 (environment, skeleton, safe clients, tests, weight verification) done on
2026-10-31; the owner's design-question batch is still open and the defaults recorded in
`docs/DECISIONS.md` apply. Measured on the development machine (darwin-arm64, Apple M4, CPU,
2 threads, torch 2.14.1; `reports/environment/`): a Kronos-small fan of 20 trajectories x 8
candles takes 5.86 s (median of 3 after a warm-up), the model loads in 0.23 s, the benchmark
process peaks at 979 MiB, and the same seed reproduces the fan bit for bit. See `docs/SPEC.md`
(the owner's specification, Spanish) and `docs/DECISIONS.md`.

## How it is built

- Offline walk-forward over real history decides WHAT runs; the demo decides whether the system
  BEHAVES as the backtest predicted. Signals always use real-market candles (public endpoint, no
  keys); orders go to the demo host only.
- Kronos never opens a trade on its own; it only modulates what the criterion enables.
- Two exchange clients with no shared signing code (`hmac` exists only in the demo module):
  `exchange/public_market.py` (real market data,
  `open-api.bingx.com`, GET only) and `exchange/demo_account.py` (authenticated, hard-pinned to
  `open-api-vst.bingx.com`: two start-up checks, then one literal-host check at three points of
  every request: pre-send assertion, request hook and pinned transport).
- Kronos is vendored at a pinned commit under `third_party/kronos/` with a sha256 manifest;
  weights are verified against pre-registered sha256 values and loaded offline on CPU with fixed
  threads and seeds derived from the decision hour.

## Layout

See `docs/ARCHITECTURE.md`.

## Reproduce step 1

```
uv sync --frozen
uv run pytest                                   # weights/network tests skip with a reason
uv run kronos-agent vendor-kronos               # already done; re-verifies the manifest
uv run kronos-agent fetch-weights --model small # Kronos-small + Kronos-Tokenizer-base
uv run pytest                                   # now includes the weights tests
KRONOS_AGENT_NETWORK_TESTS=1 uv run pytest -m network  # opt-in; public BingX host only
uv run kronos-agent env-report --network        # one public GET for the clock offset
uv run kronos-agent bench --model small --trajectories 20 --horizon 8 --threads 2 \
    --repeats 3 --threads-sweep 1,2,10 --include-mps
```

The full suite peaks at about 1.25 GiB of RAM (the test process plus a benchmark worker,
measured on darwin-arm64); on a smaller machine run `uv run pytest -m "not weights"`.

Credentials (never needed for step 1) are read only from the environment variables
`BINGX_API_KEY` and `BINGX_API_SECRET`.

## License

MIT proposed for the project's own code, pending the owner's approval. Kronos keeps its own MIT
license and attribution under `third_party/kronos/`; see `third_party/README.md` for the citation.
