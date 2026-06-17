# All Algorithms Runner

This directory contains one runner package for all eight algorithms in this
repository:

- `kratos`
- `perseus`
- `csr`
- `lstm`
- `multi_pred`
- `iaso`
- `patchTST`
- `sliding_window` (`threshold`)


## Data Format

The default input root is the repository's `data/` directory:

```text
data/
  cluster_A/
    host_1/
      2022-07-18.csv
```

Each CSV must contain:

```csv
ts,disk_id,throughput,latency
```
The data is retrieved from the Perseus paper published at FAST'23
(https://www.usenix.org/conference/fast23/presentation/lu)

The runner normalizes those columns to the benchmark schema used by all local
algorithm modules:

- `ts` -> `time`
- `disk_id` -> `serialno`
- `throughput` -> `read_blocks_delta`
- `latency` -> `read_total_latency` and `read_blk_latency`
- `cluster_X/host_N` -> `rg_id`

## Run

From the repository root, run every available algorithm on all CSV files under
`data/`:

```bash
python3 all_algorithms_runner/run.py --output all_algorithms_runner/results.csv
```

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

`sliding_window` can also be selected with the `threshold` alias. Its output
algorithm column is `threshold`.

`kratos` first computes the daily KL rows, then
applies the score and last-15-day aggregation. Its output day column is
`last_15_days` because those detections are aggregate disk results rather than
single-day rows. The current default aggregate score threshold is `1000`.

When `kratos` runs, the runner also writes the score inputs used for the status
decision to `all_algorithms_runner/kratos_scores.csv` by default. Override that
path with:

```bash
python3 all_algorithms_runner/run.py \
  --algorithms kratos \
  --kratos-scores-output /tmp/kratos_scores.csv
```

`perseus` performs the daily score generation first, then
aggregates non-null disk scores over the last 15 days. Its output day column is also
`last_15_days`. The default aggregate score threshold is `50`. The runner writes
the daily and aggregate Perseus decision fields to
`all_algorithms_runner/perseus_scores.csv`; override it with
`--perseus-scores-output`.

The remaining benchmark algorithms are also aggregated before being written to
the final result CSV, using the lookback and threshold settings in
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

Their daily score rows and aggregate decision fields are written to
`all_algorithms_runner/aggregate_scores.csv` by default. Override that with
`--aggregate-scores-output`.


## Model-Based Algorithms

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

The output CSV contains one row per positive detection:

```csv
prefix,day,algorithm,rg_id,serialno,status
cluster_A_host_1,last_15_days,threshold,cluster_A/host_1,disk1,T
```

`status` is only written for positive detections, matching the existing runner
behavior.

For the registered algorithms, final result rows are aggregate decisions, so
`day` is an aggregate label such as `last_15_days`, `last_7_days`,
`last_5_days`, or `last_1_days`. The daily rows used to make those decisions
are written to the score CSVs described above.
