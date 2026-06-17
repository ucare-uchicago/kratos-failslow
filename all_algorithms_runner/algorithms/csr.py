from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

from all_algorithms_runner.common import Result, robust_threshold


LARGE_SECONDS = 100 * 24 * 3600


def _labels(df: pd.DataFrame, column: str) -> pd.Series:
    threshold = robust_threshold(df[column], sigma=3.0)
    pieces = []
    for _, group in df.groupby(level=["rg_id", "serialno"], sort=False):
        warn = (group[column] > threshold).astype(int)
        by_time = pd.Series(
            warn.to_numpy(),
            index=group.index.get_level_values("time"),
        )
        rolled = by_time.rolling("3min", closed="right", min_periods=1).mean()
        pieces.append(pd.Series((rolled > 0.5).astype(int).to_numpy(), index=group.index))
    if not pieces:
        return pd.Series(dtype=int, index=df.index)
    return pd.concat(pieces).sort_index()


def _snapshot(df: pd.DataFrame) -> pd.DataFrame:
    snap = df.reset_index().groupby(["rg_id", "serialno"]).agg(
        {
            "read_blocks_delta": ["mean", "min", "std"],
            "read_total_latency": ["mean", "max", "std"],
        }
    )
    snap.columns = ["_".join(col) for col in snap.columns]
    return snap.reset_index()


def _time_until_error(df: pd.DataFrame, label: str) -> pd.DataFrame:
    first_error = df[df[label] == 1].reset_index().groupby(["rg_id", "serialno"])["time"].min()
    first_seen = df.reset_index().groupby(["rg_id", "serialno"])["time"].min()
    out = (first_error - first_seen).dt.total_seconds().to_frame("time_until_err")
    return out.fillna(LARGE_SECONDS)


def run(df: pd.DataFrame, day: str, rg_id: str, prefix: str) -> list[Result]:
    work = df.copy()
    if work.empty or len(work.index.get_level_values("serialno").unique()) < 2:
        return []
    work["ground_truth"] = np.maximum(
        _labels(work, "read_blocks_delta"),
        _labels(work, "read_total_latency"),
    )

    train, test = train_test_split(work, test_size=0.5, random_state=41)
    x_train = _snapshot(train).drop(columns=["rg_id", "serialno"])
    x_test = _snapshot(test).drop(columns=["rg_id", "serialno"])
    if x_train.empty or x_test.empty:
        return []
    y_train = _time_until_error(train, "ground_truth")
    y_test = _time_until_error(test, "ground_truth")
    positives = int((y_train["time_until_err"] < LARGE_SECONDS).sum())

    dtrain = xgb.DMatrix(x_train, label=y_train["time_until_err"])
    dtrain.set_group([len(x_train)])
    model = xgb.train(
        {"objective": "rank:pairwise", "learning_rate": 0.2, "max_depth": 8},
        dtrain,
        num_boost_round=100,
    )
    dtest = xgb.DMatrix(x_test, label=y_test["time_until_err"])
    dtest.set_group([len(x_test)])
    predictions = model.predict(dtest)

    ranked = y_test.copy()
    ranked["rank"] = predictions
    ranked.sort_values("rank", inplace=True)
    threshold_rank = (
        ranked["rank"].iloc[positives - 1]
        if positives < len(ranked)
        else ranked["rank"].iloc[-1]
    )
    flagged = ranked[ranked["rank"] < threshold_rank].reset_index()
    return [
        Result(prefix, day, "csr", rg_id, str(row.serialno))
        for row in flagged.itertuples()
    ]
