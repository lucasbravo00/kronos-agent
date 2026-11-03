# Coverage of the candle stores

Study 1 of docs/DATA.md: the first and last candle of every store with the date it was observed, the gaps, and the decision hours usable per segment for the 512 h Kronos context. A decision hour D is usable when every candle of the interval opening in [D - 512 h, D) is present; hours at or after the sealed start (2026-08-01T00:00Z) are not counted.

- Study: `coverage`; config sha256 `15e6461de3cba1bd99f8c545c1387d06e0cc7707cc54dc8b876736e87f27a87d`; docs/DATA.md sha256 `de4bea1bd26247abd3c9c9da3b7e1e8a2ad032485a6fa07024db425bed56b5f8`
- Code: commit `3cda6c6b125e573dd764f1dc1bd1510602da7711`, dirty yes, source tree sha256 `d2461f46a3a871dc6300089c99eed75d6a92fc862ec2a563b7173b14c072a7fd`
- Generated at 2026-11-02T14:51:15Z (report time 2026-11-02T14:51:13.865Z)
- Tests consumed: 0; statistics with intervals: 0 (counts of available data)

## Stores

| store | rows | first open | last open | gaps | missing candles | built at | first candle observed at | history days observed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `candles/binance/1m.csv` | 3595680 | 2019-12-31T00:00Z | 2026-11-01T23:59Z | 1 | 1440 | 2026-11-02T14:09:09Z | n/a | n/a |
| `candles/binance/5m.csv` | 719136 | 2019-12-31T00:00Z | 2026-11-01T23:55Z | 1 | 288 | 2026-11-02T14:09:09Z | n/a | n/a |
| `candles/binance/15m.csv` | 239712 | 2019-12-31T00:00Z | 2026-11-01T23:45Z | 1 | 96 | 2026-11-02T14:09:09Z | n/a | n/a |
| `candles/binance/1h.csv` | 59928 | 2019-12-31T00:00Z | 2026-11-01T23:00Z | 1 | 24 | 2026-11-02T14:09:09Z | n/a | n/a |
| `candles/binance_official/1h.csv` | 59928 | 2019-12-31T00:00Z | 2026-11-01T23:00Z | 1 | 24 | 2026-11-02T14:09:10Z | n/a | n/a |
| `candles/bingx/1m.csv` | 660842 | 2025-07-31T16:00Z | 2026-11-02T14:01Z | 0 | 0 | 2026-11-02T14:50:29Z | 2026-11-02T14:03:32.087Z | 458.9 |
| `candles/bingx/5m.csv` | 12960 | 2026-09-18T14:00Z | 2026-11-02T13:55Z | 0 | 0 | 2026-11-02T14:03:30Z | 2026-11-02T14:02:11.410Z | 45 |
| `candles/bingx/15m.csv` | 9599 | 2026-07-25T14:00Z | 2026-11-02T13:30Z | 0 | 0 | 2026-11-02T14:02:09Z | 2026-11-02T14:00:56.283Z | 100 |
| `candles/bingx/1h.csv` | 22059 | 2024-04-27T10:00Z | 2026-11-02T12:00Z | 0 | 0 | 2026-11-02T14:00:54Z | 2026-11-02T13:58:29.101Z | 919.2 |

## Decision hours usable before the sealed start (usable / eligible)

| store | window | before_training | training | embargo_1 | validation | embargo_2 |
| --- | --- | --- | --- | --- | --- | --- |
| `candles/bingx/1h.csv` | context_512h | 0 / 0 | 518 / 38912 | 520 / 520 | 17744 / 17744 | 520 / 520 |
| `candles/bingx/15m.csv` | context_512h | 0 / 0 | 0 / 38912 | 0 / 520 | 0 / 17744 | 0 / 520 |
| `candles/bingx/5m.csv` | context_512h | 0 / 0 | 0 / 38912 | 0 / 520 | 0 / 17744 | 0 / 520 |
| `candles/bingx/1m.csv` | context_512h | 0 / 0 | 0 / 38912 | 0 / 520 | 7736 / 17744 | 520 / 520 |
| `candles/binance/1m.csv` | context_512h | 0 / 24 | 38424 / 38912 | 520 / 520 | 17744 / 17744 | 520 / 520 |
| `candles/binance/1h.csv` | context_512h | 0 / 24 | 38424 / 38912 | 520 / 520 | 17744 / 17744 | 520 / 520 |

## Gaps

- `candles/binance/1m.csv`: [2026-10-28T00:00Z, 2026-10-29T00:00Z)
- `candles/binance/5m.csv`: [2026-10-28T00:00Z, 2026-10-29T00:00Z)
- `candles/binance/15m.csv`: [2026-10-28T00:00Z, 2026-10-29T00:00Z)
- `candles/binance/1h.csv`: [2026-10-28T00:00Z, 2026-10-29T00:00Z)
- `candles/binance_official/1h.csv`: [2026-10-28T00:00Z, 2026-10-29T00:00Z)
- `candles/bingx/1m.csv`: none
- `candles/bingx/5m.csv`: none
- `candles/bingx/15m.csv`: none
- `candles/bingx/1h.csv`: none

## Unusable decision hours before the sealed start

- `candles/bingx/1h.csv` context_512h training: [2020-01-01T00:00Z, 2024-05-18T18:00Z)
- `candles/bingx/1h.csv` context_512h embargo_1: none
- `candles/bingx/1h.csv` context_512h validation: none
- `candles/bingx/1h.csv` context_512h embargo_2: none
- `candles/bingx/15m.csv` context_512h training: [2020-01-01T00:00Z, 2024-06-09T08:00Z)
- `candles/bingx/15m.csv` context_512h embargo_1: [2024-06-09T08:00Z, 2024-07-01T00:00Z)
- `candles/bingx/15m.csv` context_512h validation: [2024-07-01T00:00Z, 2026-07-10T08:00Z)
- `candles/bingx/15m.csv` context_512h embargo_2: [2026-07-10T08:00Z, 2026-08-01T00:00Z)
- `candles/bingx/5m.csv` context_512h training: [2020-01-01T00:00Z, 2024-06-09T08:00Z)
- `candles/bingx/5m.csv` context_512h embargo_1: [2024-06-09T08:00Z, 2024-07-01T00:00Z)
- `candles/bingx/5m.csv` context_512h validation: [2024-07-01T00:00Z, 2026-07-10T08:00Z)
- `candles/bingx/5m.csv` context_512h embargo_2: [2026-07-10T08:00Z, 2026-08-01T00:00Z)
- `candles/bingx/1m.csv` context_512h training: [2020-01-01T00:00Z, 2024-06-09T08:00Z)
- `candles/bingx/1m.csv` context_512h embargo_1: [2024-06-09T08:00Z, 2024-07-01T00:00Z)
- `candles/bingx/1m.csv` context_512h validation: [2024-07-01T00:00Z, 2025-08-22T00:00Z)
- `candles/bingx/1m.csv` context_512h embargo_2: none
- `candles/binance/1m.csv` context_512h before_training: [2019-12-31T00:00Z, 2020-01-01T00:00Z)
- `candles/binance/1m.csv` context_512h training: [2020-01-01T00:00Z, 2020-01-21T08:00Z)
- `candles/binance/1m.csv` context_512h embargo_1: none
- `candles/binance/1m.csv` context_512h validation: none
- `candles/binance/1m.csv` context_512h embargo_2: none
- `candles/binance/1h.csv` context_512h before_training: [2019-12-31T00:00Z, 2020-01-01T00:00Z)
- `candles/binance/1h.csv` context_512h training: [2020-01-01T00:00Z, 2020-01-21T08:00Z)
- `candles/binance/1h.csv` context_512h embargo_1: none
- `candles/binance/1h.csv` context_512h validation: none
- `candles/binance/1h.csv` context_512h embargo_2: none

## Retention of the BingX real market

- no warning
