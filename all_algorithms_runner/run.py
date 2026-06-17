from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from all_algorithms_runner.common import Result, normalize_cluster_csv


PACKAGE_ROOT = Path(__file__).resolve().parent

ALGORITHM_MODULES = {
    "kratos": "all_algorithms_runner.algorithms.kratos",
    "perseus": "all_algorithms_runner.algorithms.perseus",
    "csr": "all_algorithms_runner.algorithms.csr",
    "lstm": "all_algorithms_runner.algorithms.lstm",
    "multi_pred": "all_algorithms_runner.algorithms.multi_pred",
    "iaso": "all_algorithms_runner.algorithms.iaso",
    "patchTST": "all_algorithms_runner.algorithms.patchTST",
    "sliding_window": "all_algorithms_runner.algorithms.sliding_window",
}

MODEL_FILES = {
    "lstm": [
        "lstm_model.pth",
        "lstm_scaler_X.pkl",
        "lstm_scaler_y.pkl",
    ],
    "patchTST": [
        "patchTST_model.pth",
        "patchTST_scaler_X.pkl",
        "patchTST_scaler_y.pkl",
    ],
}

AGGREGATE_SETTINGS = {
    "csr": {"lookback_days": 15, "threshold": 0.8, "score_column": "prediction"},
    "multi_pred": {"lookback_days": 15, "threshold": 0.8, "score_column": "prediction"},
    "lstm": {"lookback_days": 1, "threshold": 1.0, "score_column": "prediction"},
    "iaso": {"lookback_days": 7, "threshold": 0.3, "score_column": "score"},
    "patchTST": {"lookback_days": 5, "threshold": 0.4, "score_column": "prediction"},
    "sliding_window": {"lookback_days": 15, "threshold": 1.0, "score_column": "prediction"},
}


def parse_algorithms(raw: str) -> list[str]:
    aliases = {"all": ",".join(ALGORITHM_MODULES), "threshold": "sliding_window", "patchtst": "patchTST"}
    expanded = aliases.get(raw.strip().lower(), raw)
    selected = []
    for item in expanded.split(","):
        name = item.strip()
        if not name:
            continue
        name = aliases.get(name.lower(), name)
        if name not in ALGORITHM_MODULES:
            raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(ALGORITHM_MODULES)}")
        selected.append(name)
    return selected or list(ALGORITHM_MODULES)


def load_algorithm_modules(selected: list[str]) -> dict[str, object]:
    modules = {}
    for name in selected:
        try:
            modules[name] = importlib.import_module(ALGORITHM_MODULES[name])
        except ImportError as exc:
            print(f"skipping {name}: cannot import dependencies: {exc}", file=sys.stderr)
    return modules


def iter_csvs(data_root: Path, clusters: list[str] | None, hosts: list[str] | None):
    cluster_dirs = sorted(path for path in data_root.glob("cluster_*") if path.is_dir())
    if clusters:
        wanted = set(clusters)
        cluster_dirs = [path for path in cluster_dirs if path.name in wanted]
    for cluster_dir in cluster_dirs:
        host_dirs = sorted(path for path in cluster_dir.glob("host_*") if path.is_dir())
        if hosts:
            wanted_hosts = set(hosts)
            host_dirs = [path for path in host_dirs if path.name in wanted_hosts]
        for host_dir in host_dirs:
            for csv_path in sorted(host_dir.glob("*.csv")):
                yield cluster_dir.name, host_dir.name, csv_path


def _model_available(model_dir: Path, name: str) -> bool:
    return all((model_dir / filename).exists() for filename in MODEL_FILES.get(name, []))


def _missing_model_files(model_dir: Path, name: str) -> list[Path]:
    return [
        model_dir / filename
        for filename in MODEL_FILES.get(name, [])
        if not (model_dir / filename).exists()
    ]


def default_model_dir() -> Path:
    package_model_dir = PACKAGE_ROOT / "model"
    if package_model_dir.exists():
        return package_model_dir
    return Path("model")


def load_model_bundles(
    selected: list[str],
    modules: dict[str, object],
    model_dir: Path,
    strict_models: bool,
) -> dict[str, object]:
    bundles = {}
    for name in ("lstm", "patchTST"):
        if name not in selected or name not in modules:
            continue
        if not _model_available(model_dir, name):
            missing = ", ".join(str(path) for path in _missing_model_files(model_dir, name))
            message = f"{name} model files are missing under {model_dir}: {missing}"
            if strict_models:
                raise FileNotFoundError(message)
            print(f"skipping {name}: {message}", file=sys.stderr)
            modules.pop(name, None)
            continue
        bundles[name] = modules[name].load_model(model_dir)
    return bundles


def _last_n_day_sum(group, value_column: str, lookback_days: int) -> float:
    max_date = group["date_dt"].max()
    return float(group[group["date_dt"] >= (max_date - pd.DateOffset(days=lookback_days))][value_column].sum())


def aggregate_decisions(score_df, algorithm: str, prefix_by_group: dict[tuple[str, str], str]) -> tuple[list[Result], object]:
    if score_df.empty:
        return [], score_df

    settings = AGGREGATE_SETTINGS[algorithm]
    lookback_days = settings["lookback_days"]
    threshold = settings["threshold"]
    value_column = settings["score_column"]
    out_algorithm = "threshold" if algorithm == "sliding_window" else algorithm

    work = score_df.copy()
    work["date_dt"] = pd.to_datetime(work["date"])
    totals = (
        work.groupby(["cluster", "host", "disk"])[[value_column, "date_dt"]]
        .apply(_last_n_day_sum, value_column=value_column, lookback_days=lookback_days)
        .reset_index()
        .rename(columns={0: f"last_{lookback_days}_score"})
    )
    merged = pd.merge(work, totals, how="left", on=["cluster", "host", "disk"])
    score_col = f"last_{lookback_days}_score"
    merged["decision_status"] = np.where(merged[score_col] >= threshold, "T", "F")
    merged["score_threshold"] = threshold
    merged["lookback_days"] = lookback_days
    merged["algorithm"] = out_algorithm

    flagged = merged[merged["decision_status"] == "T"].drop_duplicates(["cluster", "host", "disk"])
    results = []
    for row in flagged.itertuples(index=False):
        prefix = prefix_by_group.get((row.cluster, row.host), f"{row.cluster}_{row.host}")
        rg_id = f"{row.cluster}/{row.host}"
        results.append(Result(prefix, f"last_{lookback_days}_days", out_algorithm, rg_id, str(row.disk)))
    return results, merged


def result_to_score_rows(results: list[Result], source_algorithm: str) -> list[dict[str, object]]:
    rows = []
    for result in results:
        cluster, host = result.rg_id.split("/", 1)
        rows.append(
            {
                "date": result.day,
                "cluster": cluster,
                "host": host,
                "disk": result.serialno,
                "prediction": 1.0,
                "source_algorithm": source_algorithm,
            }
        )
    return rows


def run(args: argparse.Namespace) -> list[Result]:
    selected = parse_algorithms(args.algorithms)
    modules = load_algorithm_modules(selected)
    bundles = load_model_bundles(selected, modules, args.model_dir, args.strict_models)
    results: list[Result] = []
    kratos_scores = []
    kratos_prefixes: dict[tuple[str, str], str] = {}
    perseus_scores = []
    perseus_prefixes: dict[tuple[str, str], str] = {}
    aggregate_scores: dict[str, list] = {name: [] for name in AGGREGATE_SETTINGS}
    aggregate_prefixes: dict[tuple[str, str], str] = {}

    for cluster, host, csv_path in iter_csvs(args.data_root, args.clusters, args.hosts):
        day = csv_path.stem
        rg_id = f"{cluster}/{host}"
        prefix = f"{cluster}_{host}"
        kratos_prefixes[(cluster, host)] = prefix
        perseus_prefixes[(cluster, host)] = prefix
        aggregate_prefixes[(cluster, host)] = prefix
        if args.verbose:
            print(f"processing {rg_id} {day}", file=sys.stderr)
        try:
            df = normalize_cluster_csv(csv_path, cluster, host)
        except Exception as exc:
            print(f"skipping {csv_path}: {exc}", file=sys.stderr)
            continue
        if "kratos" in selected and "kratos" in modules:
            try:
                kratos_scores.append(modules["kratos"].score_daily(df.copy(), day, rg_id))
            except Exception as exc:
                print(f"kratos failed for {csv_path}: {exc}", file=sys.stderr)
        if "perseus" in selected and "perseus" in modules:
            try:
                perseus_scores.append(modules["perseus"].score_daily(df.copy(), day, rg_id))
            except Exception as exc:
                print(f"perseus failed for {csv_path}: {exc}", file=sys.stderr)
        if "iaso" in selected and "iaso" in modules:
            try:
                aggregate_scores["iaso"].append(modules["iaso"].score_daily(df.copy(), day, rg_id))
            except Exception as exc:
                print(f"iaso failed for {csv_path}: {exc}", file=sys.stderr)
        for name in selected:
            if name in {"kratos", "perseus", "iaso"}:
                continue
            if name not in modules:
                continue
            try:
                kwargs = {"model_bundle": bundles[name]} if name in bundles else {}
                daily_results = modules[name].run(df.copy(), day, rg_id, prefix, **kwargs)
                if name in AGGREGATE_SETTINGS:
                    aggregate_scores[name].extend(result_to_score_rows(daily_results, name))
                else:
                    results.extend(daily_results)
            except Exception as exc:
                print(f"{name} failed for {csv_path}: {exc}", file=sys.stderr)
    if "kratos" in selected and "kratos" in modules and kratos_scores:
        import pandas as pd

        kratos_score_df = pd.concat(kratos_scores, ignore_index=True)
        results.extend(modules["kratos"].flag_scores(kratos_score_df, prefix_by_group=kratos_prefixes))
        write_kratos_scores(
            modules["kratos"].decision_scores(kratos_score_df),
            args.kratos_scores_output,
        )
    if "perseus" in selected and "perseus" in modules and perseus_scores:
        import pandas as pd

        perseus_score_df = pd.concat(perseus_scores, ignore_index=True)
        results.extend(modules["perseus"].flag_scores(perseus_score_df, prefix_by_group=perseus_prefixes))
        write_scores(
            modules["perseus"].decision_scores(perseus_score_df),
            args.perseus_scores_output,
        )
    import pandas as pd

    aggregate_debug = []
    for name, rows in aggregate_scores.items():
        if name not in selected or not rows:
            continue
        score_df = pd.concat(rows, ignore_index=True) if name == "iaso" else pd.DataFrame(rows)
        aggregate_results, decision_df = aggregate_decisions(score_df, name, aggregate_prefixes)
        results.extend(aggregate_results)
        aggregate_debug.append(decision_df)
    if aggregate_debug:
        write_scores(pd.concat(aggregate_debug, ignore_index=True), args.aggregate_scores_output)
    return results


def write_results(results: list[Result], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["prefix", "day", "algorithm", "rg_id", "serialno", "status"])
        for result in results:
            writer.writerow(
                [
                    result.prefix,
                    result.day,
                    result.algorithm,
                    result.rg_id,
                    result.serialno,
                    result.status,
                ]
            )


def write_scores(scores, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)


def write_kratos_scores(scores, output_path: Path) -> None:
    write_scores(scores, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all algorithms on cluster/host/day CSV data.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--clusters", nargs="*", help="Cluster directory names, for example cluster_A.")
    parser.add_argument("--hosts", nargs="*", help="Optional host directory names, for example host_23.")
    parser.add_argument(
        "--algorithms",
        default="all",
        help="Comma-separated algorithms or 'all'. Choices: kratos,perseus,csr,lstm,multi_pred,iaso,patchTST,sliding_window.",
    )
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--output", type=Path, default=Path("all_algorithms_runner/results.csv"))
    parser.add_argument(
        "--kratos-scores-output",
        type=Path,
        default=Path("all_algorithms_runner/kratos_scores.csv"),
        help="CSV path for Kratos daily scores and final decision fields.",
    )
    parser.add_argument(
        "--perseus-scores-output",
        type=Path,
        default=Path("all_algorithms_runner/perseus_scores.csv"),
        help="CSV path for Perseus daily scores and final decision fields.",
    )
    parser.add_argument(
        "--aggregate-scores-output",
        type=Path,
        default=Path("all_algorithms_runner/aggregate_scores.csv"),
        help="CSV path for aggregated daily scores for csr,multi_pred,lstm,iaso,patchTST,threshold.",
    )
    parser.add_argument("--strict-models", action="store_true", help="Fail if lstm/patchTST model files are missing.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run(args)
    write_results(results, args.output)
    print(f"wrote {len(results)} flagged rows to {args.output}")


if __name__ == "__main__":
    main()
