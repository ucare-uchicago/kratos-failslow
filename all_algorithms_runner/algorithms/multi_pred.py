from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from all_algorithms_runner.common import Result, robust_threshold


FEATURES = ["read_blocks_delta", "read_total_latency"]


def _labels(df: pd.DataFrame, column: str) -> pd.Series:
    threshold = robust_threshold(df[column], sigma=3.0)
    pieces = []
    for _, group in df.groupby(level=["rg_id", "serialno"], sort=False):
        warn = (group[column] > threshold).astype(int)
        by_time = pd.Series(
            warn.to_numpy(),
            index=group.index.get_level_values("time"),
        )
        rolled = by_time.rolling("1min", closed="right", min_periods=1).mean()
        pieces.append(pd.Series((rolled > 0.5).astype(int).to_numpy(), index=group.index))
    if not pieces:
        return pd.Series(dtype=int, index=df.index)
    return pd.concat(pieces).sort_index()


def run(df: pd.DataFrame, day: str, rg_id: str, prefix: str) -> list[Result]:
    work = df.copy()
    if work.empty:
        return []
    work["ground_truth"] = np.maximum(
        _labels(work, "read_blocks_delta"),
        _labels(work, "read_total_latency"),
    )
    tmp = work.reset_index(level=["rg_id", "serialno"], drop=False).dropna(
        subset=FEATURES + ["ground_truth"]
    )
    x = tmp.groupby(["rg_id", "serialno", pd.Grouper(freq="15min")])[FEATURES].first()
    y = tmp.groupby(["rg_id", "serialno", pd.Grouper(freq="15min")])["ground_truth"].sum()
    y = y.apply(lambda value: 1 if value >= 1 else 0)
    if x.empty or len(set(y)) < 2:
        return []

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=41
    )
    if len(set(y_train)) < 2 or y_train.value_counts().min() < 6:
        return []

    x_train_res, y_train_res = SMOTE(random_state=41).fit_resample(x_train, y_train)
    clf = RandomForestClassifier(n_estimators=100, random_state=41)
    clf.fit(x_train_res, y_train_res)
    pred = clf.predict(x_test)
    flagged = y_test.to_frame("truth")
    flagged["prediction"] = pred
    flagged = flagged[flagged["prediction"] == 1].reset_index()
    serials = sorted(flagged["serialno"].astype(str).unique())
    return [Result(prefix, day, "multi_pred", rg_id, serial) for serial in serials]
