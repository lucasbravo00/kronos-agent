# Kronos forecast benchmark: darwin-arm64

Generated 2026-10-31T18:01:56+00:00. Kronos-small with its paired tokenizer, device cpu, 2 threads, torch 2.14.1, deterministic algorithms on. Input: upstream_fixture (512 rows, step 300 s, horizon 8).

| Metric | Value |
| --- | --- |
| Fan 20 x 8 (`predict_batch`, `sample_count=1`), median of 3 run(s) after 1 warm-up | 5.86 s (min 5.85, max 5.86) |
| Seconds per trajectory-candle | 0.0366 |
| Model load (`load_models`) | 0.23 s |
| Peak RSS of the benchmark process | 979 MiB |
| Same seed twice gives an identical fan | yes |
| Different seed gives a different fan | yes |
| `predict(sample_count=20)`, one mean path (not a fan) | 5.85 s |
| Upstream greedy fixture, context 512 (report-only) | max_rel_diff 4.02e-07, rtol 1e-5 pass |
| Upstream greedy fixture, context 256 (report-only) | max_rel_diff 6.19e-07, rtol 1e-5 pass |

## Thread sweep (subprocesses, informational)

| Threads | Median | Min | Max | Identical to the reference fan | Peak RSS | Error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.57 s | 7.56 s | 7.58 s | yes | 925 MiB | - |
| 2 | 5.88 s | 5.87 s | 5.90 s | yes | 959 MiB | - |
| 10 | 4.78 s | 4.77 s | 4.78 s | no | 961 MiB | - |

## MPS comparison (informational; MPS is not a reference device)

| Device | Median | Identical to the CPU fan | Error |
| --- | --- | --- | --- |
| mps | 3.80 s | no | - |

Notes:

- same machine, same build, same thread count and same seeds reproduce a fan bit for bit; other thread counts, devices or architectures are not guaranteed to (recorded, not required).
- predict(sample_count=K) averages K sampled paths and returns ONE path; it is timed for comparison only, the fan is built with predict_batch and sample_count=1.
- offline replica of the upstream greedy regression test (seed 123, top_k=1, top_p=1.0, T=1.0, raw fixture timestamps, predictor.predict); the platform that generated the fixtures is unknown, so the comparison is report-only (decision D-007).
