# Measurement step of the retraining (finetune timing)

Status: **completed**. Design: docs/RETRAINING.md section (a), phase P0. Synthetic data only: no market data and no outcome read; tests consumed: 0. Generated 2026-11-02T21:01:36Z; run 2026-11-02T20:44:30Z to 2026-11-02T21:01:36Z; scratch `.scratch/timing-20261102T204430Z` under the fine-tune root.

## Environment and code

| Item | Value |
| --- | --- |
| platform | darwin-arm64 |
| torch | 2.14.1 |
| numpy | 2.2.6 |
| pandas | 2.2.2 |
| CPU count | 10 |
| RAM GiB | 16.00 |
| MPS available | yes |
| training code sha256 | fb82f2ec48fe06cdee817965473051b2fff4061f83c7640c20617fee96d14be5 |
| forecasting code sha256 | 37c3bf99e48a6aee17a804432e4ae40726310e9d82aef40adbdc4f8d66e593a1 |
| vendored manifest sha256 | 4a3214ae510ccf6cc0fd7836d4e48e54851bf71b579fd852faf7f9089a081f92 |
| source tree sha256 | bb2fad7612c9921adbc98078fbc52dc419c6e5fa58d507bf2703ed84e0dff615 |
| timing plan sha256 | 7408a6f62be55872c13a5e81939ac5f899d5b2266a2b33e2afc34531d96a4359 |
| RETRAINING.md sha256 | 6e5979f561100a08a4ed73e23f607a361e61564bffb11ff0b48e6e2417c44a23 |

A later estimate refuses this record when the chosen device, the torch version, the platform or the training code differ.

## Budget

Estimate (priors) about 2620 CPU-seconds and 25.0 GPU-minutes; spent 2049 CPU-seconds and 1.8 GPU-minutes; limits 3000 CPU-seconds and 45 GPU-minutes, below gate 3 (3600 CPU-seconds, 60 GPU-minutes). Stopped: no.

| Item | Prior CPU-s | Prior GPU-min | Spent CPU-s | Spent GPU-min |
| --- | --- | --- | --- | --- |
| M1 | 1350.000 | 2.000 | 769.260 | 0.000 |
| M2 | 200.000 | 1.000 | 205.343 | 0.000 |
| M3 | 150.000 | 2.000 | 36.689 | 1.777 |
| M4 | 120.000 | 20.000 | 0.000 | 0.000 |
| M5 | 210.000 | 0.000 | 224.831 | 0.000 |
| M6 | 300.000 | 0.000 | 779.009 | 0.000 |
| M7 | 150.000 | 0.000 | 0.000 | 0.000 |
| M8 | 20.000 | 0.000 | 15.222 | 0.000 |
| M9 | 20.000 | 0.000 | 18.348 | 0.000 |
| loads | 100.000 | 0.000 | 0.615 | 0.000 |

## Choice (design 3.4)

| Item | Value |
| --- | --- |
| device | cpu |
| threads | 2 |
| batch | 32 |
| deterministic | yes |
| activation checkpointing | yes |
| MPS watermark ratio | - |
| batch 32 fits | no |
| seconds per step (batch 32) | 9.679 |
| theta | - |
| MPS vs best CPU | - |
| 8 vs 2 threads | 0.976 |

- CPU threads 2: 8 threads only 0.98 times faster at batch 32 (at least 1.3 needed)
- MPS not usable: its preflight (M3) did not pass
- batch 32 does not fit as is: activation checkpointing (M2 passed)
- theta for the CPU comes from the pilot's first pseudo-epoch (no M4)

## M1 training step time and peak memory

| Device | Threads | Batch | Status | Median s/step | Min | Max | Windows/s | Peak GiB | Swap growth GiB | Pressure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | 2 | 8 | completed | 1.531 | 1.506 | 1.647 | 5.227 | 3.77 | 0.00 | 1 |
| cpu | 2 | 16 | completed | 3.155 | 3.032 | 3.377 | 5.071 | 6.80 | 0.00 | 1 |
| cpu | 2 | 32 | completed | 10.359 | 9.770 | 11.893 | 3.089 | 9.23 | 1.52 | 1 |
| cpu | 8 | 8 | completed | 1.723 | 1.693 | 1.835 | 4.643 | 3.77 | 0.00 | 1 |
| cpu | 8 | 16 | completed | 3.434 | 3.379 | 4.357 | 4.659 | 6.80 | 0.00 | 1 |
| cpu | 8 | 32 | completed | 10.612 | 9.818 | 11.362 | 3.015 | 9.16 | 1.34 | 1 |
| mps | 2 | 8 | not_run | - | - | - | - | - | - | - |
| mps | 2 | 16 | not_run | - | - | - | - | - | - | - |
| mps | 2 | 32 | not_run | - | - | - | - | - | - | - |

Peak: the child's peak RSS, plus its largest MPS driver allocation on MPS.

| Device | Threads | Batches | GiB per window |
| --- | --- | --- | --- |
| cpu | 2 | [8, 16, 32] | 0.217 |
| cpu | 8 | [8, 16, 32] | 0.214 |
| mps | 2 | [] | - |

## M2 activation checkpointing

| Device | Threads | Why | Status | Median s/step | Peak GiB | Gradients | Passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | 2 | batch 32 peak 9.23 GiB, swap growth 1.52 GiB | completed | 9.679 | 3.16 | yes | yes |

## M3 MPS preflight

Status memory_abort; passed no.

| Check | Passed | Detail |
| --- | --- | --- |
| multinomial | - |  |
| parity | - |  |
| steps | - |  |
| deterministic | - |  |

## M4 sustained MPS throughput

| Item | Value |
| --- | --- |
| status | not_run |
| reason | not run: the MPS preflight (M3) did not pass |
| timed seconds | - |
| timed steps | - |
| median s/step, first 2 minutes | - |
| median s/step, last 10 minutes | - |
| theta | - |
| theta not computed because | - |
| activation checkpointing | - |
| peak MPS driver GiB | - |

## M5 fan speed (one warm-up and 3 timed fans, 2 threads)

| Length | Status | Seconds per fan | Prior |
| --- | --- | --- | --- |
| 4 | completed | 3.230 | 3.152 |
| 8 | completed | 6.059 | 5.864 |
| 24 | completed | 19.018 | 16.712 |

Fit: t(L) = -0.107 + 0.795 L seconds.

## M6 worker scaling and M7 contention (during M4)

| Test | Workers | Status | Wall s | s per fan | Fans/s | Start-up s | Slowdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M6 | 1 | completed | 51.675 | 4.306 | 0.232 | 5.266 | - |
| M6 | 2 | completed | 41.311 | 6.885 | 0.290 | 8.544 | - |
| M6 | 3 | completed | 36.872 | 9.218 | 0.325 | 11.283 | - |
| M6 | 4 | completed | 36.148 | 12.049 | 0.332 | 14.593 | - |
| M7 | 1 | not_run | - | - | - | - | - |
| M7 | 2 | not_run | - | - | - | - | - |

## M8 NLL4-TF time and M9 identity control

| Item | Value |
| --- | --- |
| M8 status | completed |
| M8 decisions | 48 |
| M8 seconds per decision | 0.158 |
| M8 seconds per fold evaluation | 53.213 |
| M9 status | completed |
| M9 identical fan | yes |
| M9 shipped config id | 62906d91f853a8e5 |
| M9 checkpoint config id | d49ec38a515fe546 |
