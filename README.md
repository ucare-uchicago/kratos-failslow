# All Algorithms Runner

This repository runs eight disk slow-failure detection algorithms against the
included cluster data:

- `kratos`
- `perseus`
- `csr`
- `lstm`
- `multi_pred`
- `iaso`
- `patchTST`
- `sliding_window` (also selectable as `threshold`)

The input data comes from the Perseus FAST '23 paper:
https://www.usenix.org/conference/fast23/presentation/lu

## Quick Start

Run every available algorithm against all CSV files under `data/`:

```bash
python3 all_algorithms_runner/run.py
```

The default outputs are:

- `all_algorithms_runner/results.csv` for final positive detections
- `all_algorithms_runner/kratos_scores.csv` for Kratos daily and decision scores
- `all_algorithms_runner/perseus_scores.csv` for Perseus daily and decision scores
- `all_algorithms_runner/aggregate_scores.csv` for the other aggregated algorithms

Run a smaller subset while developing:

```bash
python3 all_algorithms_runner/run.py \
  --clusters cluster_A \
  --hosts host_1 \
  --algorithms kratos,perseus,csr,multi_pred,iaso,sliding_window \
  --output all_algorithms_runner/results_host_1.csv \
  --verbose
```

Run a single algorithm:

```bash
python3 all_algorithms_runner/run.py --algorithms kratos
```

## Data

The runner reads this directory layout by default:

```text
data/
  cluster_A/
    host_1/
      2022-07-18.csv
```

Each CSV must contain these columns:

```csv
ts,disk_id,throughput,latency
```

At load time, the runner normalizes those columns to the schema expected by the
algorithm modules:

- `ts` becomes `time`
- `disk_id` becomes `serialno`
- `throughput` becomes `read_blocks_delta`
- `latency` becomes `read_total_latency` and `read_blk_latency`
- `cluster_X/host_N` becomes `rg_id`

Use `--data-root PATH` to run against another data directory.

## Algorithm Aggregation

Most algorithms produce daily disk scores or predictions that are aggregated
before final results are written. `sliding_window` can be selected with the
`threshold` alias, and its final output algorithm name is `threshold`.

`kratos` computes daily KL rows, applies its score rules, and then aggregates
over the last 15 days. Its default aggregate score threshold is `1000`.

`perseus` computes daily scores and aggregates non-null disk scores over the
last 15 days. Its default aggregate score threshold is `50`.

The other aggregated algorithms use these settings from
`all_algorithms_runner/run.py`:

| Algorithm | Lookback | Threshold | Value |
| --- | ---: | ---: | --- |
| `csr` | 15 days | 0.8 | positive prediction count |
| `multi_pred` | 15 days | 0.8 | positive prediction count |
| `lstm` | 1 day | 1.0 | positive prediction count |
| `iaso` | 7 days | 0.3 | IASO score sum |
| `patchTST` | 5 days | 0.4 | positive prediction count |
| `threshold` | 15 days | 1.0 | positive prediction count |

The thresholds and lookback window are selected based on their best MCC.

## Model Files

`lstm` and `patchTST` need pretrained model files. By default the runner uses
`all_algorithms_runner/model/` when that directory exists; otherwise it falls
back to `model/` relative to the current working directory.

- `lstm_model.pth`
- `lstm_scaler_X.pkl`
- `lstm_scaler_y.pkl`
- `patchTST_model.pth`
- `patchTST_scaler_X.pkl`
- `patchTST_scaler_y.pkl`

If those files are missing, the default `all` run skips those two algorithms and
prints a setup message. Use `--model-dir PATH` to point at the artifacts, or
`--strict-models` to fail immediately when they are missing.

## Output

The final result CSV contains one row per positive detection:

```csv
prefix,day,algorithm,rg_id,serialno,status
cluster_A_host_1,last_15_days,threshold,cluster_A/host_1,disk1,T
```

`status` is only written for positive detections, matching the existing runner
behavior.

Final result rows are aggregate decisions for the registered algorithms, so
`day` is an aggregate label such as `last_15_days`, `last_7_days`,
`last_5_days`, or `last_1_days`. The daily rows used to make those decisions
are written to the score CSVs described above.

You can override output paths with:

```bash
python3 all_algorithms_runner/run.py \
  --output /tmp/results.csv \
  --kratos-scores-output /tmp/kratos_scores.csv \
  --perseus-scores-output /tmp/perseus_scores.csv \
  --aggregate-scores-output /tmp/aggregate_scores.csv
```
