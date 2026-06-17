from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"ts", "disk_id", "throughput", "latency"}


@dataclass(frozen=True)
class Result:
    prefix: str
    day: str
    algorithm: str
    rg_id: str
    serialno: str
    status: str = "T"

    def to_csv_row(self) -> str:
        return ",".join(
            [
                self.prefix,
                self.day,
                self.algorithm,
                self.rg_id,
                self.serialno,
                self.status,
            ]
        )


def normalize_cluster_csv(csv_path: Path, cluster: str, host: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    normalized = pd.DataFrame(
        {
            "time": pd.to_datetime(df["ts"], unit="s", utc=True),
            "rg_id": f"{cluster}/{host}",
            "serialno": df["disk_id"].astype(str),
            "read_blocks_delta": pd.to_numeric(df["throughput"], errors="coerce"),
            "read_total_latency": pd.to_numeric(df["latency"], errors="coerce"),
            "read_blk_latency": pd.to_numeric(df["latency"], errors="coerce"),
            "write_total_latency": 0.0,
            "write_blocks_delta": 0.0,
        }
    )
    normalized = normalized.dropna(
        subset=["time", "serialno", "read_blocks_delta", "read_total_latency"]
    )
    normalized.set_index(["time", "rg_id", "serialno"], inplace=True)
    normalized.sort_index(inplace=True)
    return normalized


def indexed_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index()


def ratio_series(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    blocks = work["read_blocks_delta"].replace(0, np.nan).fillna(1e-6)
    latency = work["read_total_latency"].fillna(0)
    return latency / blocks


def robust_threshold(values: Iterable[float], sigma: float = 2.0) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("inf")
    std = arr.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return float("inf")
    return float(arr.mean() + sigma * std)


def day_from_index(df: pd.DataFrame) -> str:
    days = df.index.get_level_values("time").strftime("%Y-%m-%d").unique()
    return str(days[0]) if len(days) else ""
