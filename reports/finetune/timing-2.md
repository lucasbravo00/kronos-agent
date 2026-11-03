# Measurement step of the retraining (finetune timing)

Status: **completed**. Design: docs/RETRAINING.md section (a), phase P0. Synthetic data only: no market data and no outcome read; tests consumed: 0. Generated 2026-11-03T01:42:36Z; run 2026-11-03T01:00:48Z to 2026-11-03T01:42:36Z; scratch `.scratch/timing-20261103T010048Z` under the fine-tune root.

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
| training code sha256 | f28c0f30d217c24c914fe6b8f8c75aed27aa44115198b4a1821122706a5216d8 |
| forecasting code sha256 | 37c3bf99e48a6aee17a804432e4ae40726310e9d82aef40adbdc4f8d66e593a1 |
| vendored manifest sha256 | 4a3214ae510ccf6cc0fd7836d4e48e54851bf71b579fd852faf7f9089a081f92 |
| source tree sha256 | 50cf9115aab6b973e89e7cb8429b53c3d0ee995c0324155140a8f8f1400514bc |
| timing plan sha256 | 2128b4bb2eb0e2ed947024ea95cffe5cf327657ad6b5f9a2eea5c267e7a17d53 |
| RETRAINING.md sha256 | 7d8b6504b5ec9ff459ccc531205c31546684a4d5b71375b1cb5d12e2a9b9135f |

A later estimate refuses this record when the chosen device, the torch version, the platform or the training code differ.

## Budget

Estimate (priors) about 3578 CPU-seconds and 33.2 GPU-minutes; spent 3038 CPU-seconds and 27.4 GPU-minutes; limits 5400 CPU-seconds, the ceiling the owner approved (gate 3 below), and 45 GPU-minutes. Stopped: no.

Gate 3: the owner approved 1.5 CPU-hours (--approved-by-owner given at 2026-11-03T01:00:47Z, the command's start): a CPU limit of 5400 CPU-seconds instead of the step's own 3000, above the gate-3 limit (3600 CPU-seconds) under the owner's approval, a ceiling the spend never passes: the run is planned and stopped against its stop limit of 4800 CPU-seconds, a reserve of 600 below it for the unit running then; estimate about 3578 CPU-seconds and 33.2 GPU-minutes. The CPU limit is the ceiling the owner approved in chat for this measurement: an estimate above the stop limit is refused before anything runs, a unit or retry whose prior would pass it is not started, each child's deadline and the watch of M4 follow it, and the run stops once its measured spend passes it. The run's spend, 3038 CPU-seconds, stayed within the 5400 CPU-seconds the owner approved and stayed below the gate-3 limit (3600 CPU-seconds).

| Item | Prior CPU-s | Prior GPU-min | Spent CPU-s | Spent GPU-min |
| --- | --- | --- | --- | --- |
| M1 | 1350.000 | 2.000 | 778.723 | 1.982 |
| M2 | 200.000 | 1.000 | 182.700 | 0.000 |
| M3 | 553.760 | 4.615 | 259.169 | 5.021 |
| M4 | 120.000 | 21.000 | 75.124 | 20.360 |
| M5 | 210.000 | 0.000 | 219.708 | 0.000 |
| M6 | 300.000 | 0.000 | 698.023 | 0.000 |
| M7 | 150.000 | 0.000 | 791.564 | 0.000 |
| M8 | 20.000 | 0.000 | 14.916 | 0.000 |
| M9 | 20.000 | 0.000 | 17.015 | 0.000 |
| loads | 100.000 | 0.000 | 0.601 | 0.000 |
| retries | 553.760 | 4.615 | 0.000 | 0.000 |

## Choice (design 3.4)

| Item | Value |
| --- | --- |
| device | mps |
| threads | 2 |
| batch | 32 |
| deterministic | yes |
| activation checkpointing | yes |
| MPS watermark ratio | 1.000 |
| batch 32 fits | no |
| seconds per step (batch 32) | 5.449 |
| theta | 1.004 |
| MPS sustained s/step | 5.445 |
| best CPU configuration | CPU x2 with activation checkpointing, 8.790 s per step |
| MPS vs best CPU | 1.614 |
| 8 vs 2 threads | 1.021 |

- CPU threads 2: 8 threads only 1.02 times faster at batch 32 (at least 1.3 needed)
- device MPS: its sustained step time (with activation checkpointing) is 1.61 times faster than the best CPU configuration's (CPU x2 with activation checkpointing, 8.790 s per step; at least 1.5 needed)
- batch 32 runs with activation checkpointing on MPS: the batch 32 of CPU x2 does not fit as is (M1) and activation checkpointing passed there (M2)
- theta 1.004 from M4 (at least 1)

## Activation checkpointing on MPS and retries

MPS units from batch 32 up ran with activation checkpointing: the batch 32 of CPU x2 does not fit as is (M1) and activation checkpointing passed there (M2).

No MPS unit ended memory_abort without activation checkpointing: none was retried.

## M1 training step time and peak memory

| Device | Threads | Batch | Checkpointing | Attempt | Status | Median s/step | Min | Max | Windows/s | Peak GiB | Swap growth GiB | Pressure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cpu | 2 | 8 | no | 1 | completed | 1.555 | 1.519 | 1.688 | 5.145 | 3.58 | 0.00 | 1 |
| cpu | 2 | 16 | no | 1 | completed | 4.164 | 3.982 | 4.627 | 3.842 | 5.76 | 0.00 | 1 |
| cpu | 2 | 32 | no | 1 | completed | 9.665 | 9.339 | 12.219 | 3.311 | 9.25 | 3.19 | 1 |
| cpu | 8 | 8 | no | 1 | completed | 1.690 | 1.673 | 1.768 | 4.734 | 3.76 | 0.00 | 1 |
| cpu | 8 | 16 | no | 1 | completed | 3.298 | 3.279 | 3.574 | 4.852 | 6.80 | 0.00 | 1 |
| cpu | 8 | 32 | no | 1 | completed | 9.465 | 8.981 | 11.957 | 3.381 | 9.41 | 0.95 | 1 |
| mps | 2 | 8 | no | 1 | completed | 1.090 | 1.065 | 1.162 | 7.338 | 5.11 | 0.00 | 1 |
| mps | 2 | 16 | no | 1 | completed | 2.096 | 2.078 | 2.149 | 7.633 | 8.02 | 0.00 | 1 |
| mps | 2 | 32 | yes | 1 | completed | 5.449 | 5.397 | 5.486 | 5.873 | 5.03 | 0.00 | 1 |

Peak: the child's peak RSS, plus its largest MPS driver allocation on MPS.

| Device | Threads | Batches | GiB per window |
| --- | --- | --- | --- |
| cpu | 2 | [8, 16, 32] | 0.234 |
| cpu | 8 | [8, 16, 32] | 0.225 |
| mps | 2 | [8, 16] | 0.364 |

## M2 activation checkpointing

CPU gradient bit-identity of one step with and without the wrap (dropout active): status completed, passed yes; 127 tensors, mismatches: none.

| Device | Threads | Why | Status | Median s/step | Peak GiB | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| cpu | 2 | batch 32 peak 9.25 GiB, swap growth 3.19 GiB | completed | 8.790 | 3.16 | yes |

## M3 MPS preflight

Status completed; passed yes; activation checkpointing yes.

| Check | Passed | Detail |
| --- | --- | --- |
| multinomial | yes | categories 1024, repetitions 64, rows 16384 |
| parity | yes | gradient_cosine 1, max_relative_logit_difference 2.72072e-06 |
| steps | yes | finite yes, mean_loss_cpu 3.04932, mean_loss_device 3.04742, relative_difference 0.000624295, steps 20 |
| deterministic | yes | op -, refused no |

## M4 sustained MPS throughput

| Item | Value |
| --- | --- |
| status | completed |
| reason | - |
| timed seconds | 1200.766 |
| timed steps | 212 |
| median s/step, first 2 minutes | 5.423 |
| median s/step, last 10 minutes | 5.445 |
| theta | 1.004 |
| theta not computed because | - |
| activation checkpointing | yes |
| peak MPS driver GiB | 4.31 |

## M5 fan speed (one warm-up and 3 timed fans, 2 threads)

| Length | Status | Seconds per fan | Prior |
| --- | --- | --- | --- |
| 4 | completed | 3.062 | 3.152 |
| 8 | completed | 6.035 | 5.864 |
| 24 | completed | 18.361 | 16.712 |

Fit: t(L) = -0.045 + 0.766 L seconds.

## M6 worker scaling and M7 contention (during M4)

| Test | Workers | Status | Wall s | s per fan | Fans/s | Start-up s | Slowdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M6 | 1 | completed | 40.766 | 3.397 | 0.294 | 4.859 | - |
| M6 | 2 | completed | 37.493 | 6.249 | 0.320 | 7.371 | - |
| M6 | 3 | completed | 34.297 | 8.574 | 0.350 | 10.317 | - |
| M6 | 4 | completed | 32.592 | 10.864 | 0.368 | 12.805 | - |
| M7 | 1 | completed | 122.726 | 10.227 | 0.098 | 10.953 | 3.010 |
| M7 | 2 | completed | 136.528 | 22.755 | 0.088 | 27.106 | 3.641 |

## M8 NLL4-TF time and M9 identity control

| Item | Value |
| --- | --- |
| M8 status | completed |
| M8 decisions | 48 |
| M8 seconds per decision | 0.155 |
| M8 seconds per fold evaluation | 52.154 |
| M9 status | completed |
| M9 identical fan | yes |
| M9 shipped config id | 62906d91f853a8e5 |
| M9 checkpoint config id | d49ec38a515fe546 |
