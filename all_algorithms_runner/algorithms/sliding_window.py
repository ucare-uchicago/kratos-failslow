from __future__ import annotations

import numpy as np
import pandas as pd

from all_algorithms_runner.common import Result, indexed_to_columns


def run(df: pd.DataFrame, day: str, rg_id: str, prefix: str, threshold: float = 50.0) -> list[Result]:
    work = indexed_to_columns(df)
    if work.empty:
        return []

    resampled = (
        work.set_index("time")
        .groupby(["rg_id", "serialno"])["read_blk_latency"]
        .resample("30s")
        .mean()
        .reset_index()
    )
    resampled["rolling_avg"] = (
        resampled.set_index("time")
        .groupby(["rg_id", "serialno"])["read_blk_latency"]
        .rolling("300s", closed="right", min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    flagged = resampled[resampled["rolling_avg"] > threshold]
    serials = sorted(flagged["serialno"].dropna().astype(str).unique())
    return [Result(prefix, day, "threshold", rg_id, serial) for serial in serials]
