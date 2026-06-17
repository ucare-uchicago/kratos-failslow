# All Algorithms Runner

## Getting Started Instructions

This artifact package evaluates disk slow-failure detection algorithms on the
included cluster data. The targeted artifact badge is **Artifact Available**.

The package includes implementations or runner wrappers for eight algorithms:
`kratos`, `perseus`, `csr`, `lstm`, `multi_pred`, `iaso`, `patchTST`, and
`sliding_window` (also selectable as `threshold`). The input data comes from
the Perseus FAST '23 paper:
https://www.usenix.org/conference/fast23/presentation/lu

To check basic functionality within a short review window, run a small subset
from the repository root:

```bash
python3 all_algorithms_runner/run.py \
  --clusters cluster_A \
  --hosts host_1 \
  --algorithms kratos,perseus,iaso,sliding_window \
  --output /tmp/kratos_failslow_results.csv \
  --kratos-scores-output /tmp/kratos_scores.csv \
  --perseus-scores-output /tmp/perseus_scores.csv \
  --aggregate-scores-output /tmp/aggregate_scores.csv \
  --verbose
```

This smoke test reads the CSV files under `data/cluster_A/host_1/`, runs four
algorithms that do not require external model downloads, and writes final
positive detections to `/tmp/kratos_failslow_results.csv`.

Expected result format:

```csv
prefix,day,algorithm,rg_id,serialno,status
cluster_A_host_1,last_15_days,threshold,cluster_A/host_1,disk1,T
```

The exact number of positive rows can vary if dependencies or algorithm
libraries change, but the command should finish without runner errors and
produce the requested CSV files.

## Detailed Instructions

**Artifact claims.** This artifact is intended to support the following concrete
claims:

- The repository contains a runnable package for applying the listed algorithms
  to cluster/host/day CSV data.
- The runner normalizes the included Perseus-format CSV files into one shared
  schema before invoking the algorithm modules.
- The runner writes final positive detections in a common CSV format.
- For algorithms that produce daily scores or predictions, the runner also
  writes score files that expose the intermediate values used for final
  aggregate decisions.
- The artifact is packaged for availability and inspection. It is not claimed
  to be a turnkey reproduction of every experiment, figure, or environment from
  the associated paper.

**Data layout and schema.** The default input root is `data/`, organized by
cluster, host, and day:

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

At load time, the runner normalizes those fields as follows:

- `ts` becomes `time`
- `disk_id` becomes `serialno`
- `throughput` becomes `read_blocks_delta`
- `latency` becomes `read_total_latency` and `read_blk_latency`
- `cluster_X/host_N` becomes `rg_id`

Use `--data-root PATH` to evaluate a different data directory with the same
layout and columns.

**Full runner usage.** Run every available algorithm on all CSV files under
`data/`:

```bash
python3 all_algorithms_runner/run.py
```

The default output paths are:

- `all_algorithms_runner/results.csv` for final positive detections
- `all_algorithms_runner/kratos_scores.csv` for Kratos daily and decision scores
- `all_algorithms_runner/perseus_scores.csv` for Perseus daily and decision
  scores
- `all_algorithms_runner/aggregate_scores.csv` for the other aggregated
  algorithms

Run one algorithm:

```bash
python3 all_algorithms_runner/run.py --algorithms kratos
```

Run selected algorithms and write outputs elsewhere:

```bash
python3 all_algorithms_runner/run.py \
  --clusters cluster_A \
  --hosts host_1 \
  --algorithms kratos,perseus,csr,multi_pred,iaso,sliding_window \
  --output /tmp/results.csv \
  --kratos-scores-output /tmp/kratos_scores.csv \
  --perseus-scores-output /tmp/perseus_scores.csv \
  --aggregate-scores-output /tmp/aggregate_scores.csv
```

`sliding_window` can also be selected with the `threshold` alias. Its final
output algorithm name is `threshold`.

**Model-based algorithms.** `lstm` and `patchTST` require pretrained model
artifacts. By default the runner uses `all_algorithms_runner/model/` when that
directory exists; otherwise it falls back to `model/` relative to the current
working directory.

Required files:

- `lstm_model.pth`
- `lstm_scaler_X.pkl`
- `lstm_scaler_y.pkl`
- `patchTST_model.pth`
- `patchTST_scaler_X.pkl`
- `patchTST_scaler_y.pkl`

If those files are missing, the default `all` run skips `lstm` and `patchTST`
and prints a setup message. Use `--model-dir PATH` to point at model artifacts,
or `--strict-models` to fail immediately when model files are missing.

**Aggregation behavior.** Final result rows are aggregate decisions for the
registered algorithms, so the `day` field is an aggregate label such as
`last_15_days`, `last_7_days`, `last_5_days`, or `last_1_days`.

`kratos` computes daily KL rows, applies score rules, and aggregates over the
last 15 days. Its default aggregate score threshold is `1000`.

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

The thresholds and lookback windows are selected based on their best MCC.

**Evaluation notes.** Reviewers should use the smoke test above first to detect
basic packaging or dependency problems. A full run over all included data can
take substantially longer than the smoke test because it processes every
cluster, host, and day and may invoke heavier algorithms such as `csr`,
`multi_pred`, `lstm`, and `patchTST`.

Some systems may print CPU feature or library warnings from numerical
dependencies. Those warnings are not artifact failures when the runner completes
and writes the requested CSV files.
