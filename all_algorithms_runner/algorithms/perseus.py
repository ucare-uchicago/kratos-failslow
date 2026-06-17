from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.api as sm

from all_algorithms_runner.common import Result, indexed_to_columns


def _score(avg: float, duration: int) -> int:
    if avg >= 5:
        if duration >= 24:
            return 100
        if duration >= 12:
            return 25
        if duration >= 6:
            return 10
    elif avg >= 2:
        if duration >= 24:
            return 25
        if duration >= 12:
            return 10
        if duration >= 6:
            return 5
    else:
        if duration >= 24:
            return 10
        if duration >= 12:
            return 5
        if duration >= 6:
            return 1
    return 0


def run(
    df: pd.DataFrame,
    day: str,
    rg_id: str,
    prefix: str,
    score_threshold: float = 50.0,
    last_n_days: int = 15,
    eps: float = 0.125,
    alpha: float = 0.05,
    min_windows: int = 2,
) -> list[Result]:
    score_df = score_daily(df, day, rg_id, eps=eps,
                           alpha=alpha, min_windows=min_windows)
    return flag_scores(
        score_df,
        prefix_by_group={
            (score_df.iloc[0].cluster, score_df.iloc[0].host): prefix}
        if not score_df.empty
        else {},
        score_threshold=score_threshold,
        last_n_days=last_n_days,
    )


def score_daily(
    df: pd.DataFrame,
    day: str,
    rg_id: str,
    eps: float = 0.125,
    alpha: float = 0.05,
    min_windows: int = 2,
) -> pd.DataFrame:
    work = indexed_to_columns(df).dropna(
        subset=["read_blocks_delta", "read_total_latency"])
    if work.empty or len(work) < 8 or work["serialno"].nunique() < 2:
        return pd.DataFrame()

    cluster, host = rg_id.split("/", 1)

    x_col = "read_blocks_delta"
    y_col = "read_total_latency"
    scaled = StandardScaler().fit_transform(work[[x_col, y_col]])
    components = PCA(n_components=2).fit_transform(scaled)
    labels = DBSCAN(eps=eps).fit(components).labels_
    if len(labels) == 0:
        return pd.DataFrame()

    normal_label = max(set(labels), key=list(labels).count)
    normal_mask = labels == normal_label
    if normal_mask.sum() < 5:
        return pd.DataFrame()

    poly = PolynomialFeatures(degree=4)
    x_poly = poly.fit_transform(work.loc[normal_mask, [x_col]])
    y = work.loc[normal_mask, y_col]
    results = sm.OLS(y, x_poly).fit()
    predictions = results.get_prediction(x_poly).summary_frame(alpha)

    normal = work.loc[normal_mask, ["time", y_col, x_col, "serialno"]].copy()
    normal.insert(0, "prediction_upper_bound",
                  predictions["obs_ci_upper"].to_numpy())
    outliers = work.loc[~normal_mask, [
        "time", y_col, x_col, "serialno"]].copy()
    outliers["prediction_upper_bound"] = np.nan
    work = pd.concat([normal, outliers]).sort_index()
    work["prediction_upper_bound"] = work["prediction_upper_bound"].ffill().bfill()
    if work["prediction_upper_bound"].isna().any():
        return pd.DataFrame()
    work["slowdown_ratio"] = work[y_col] / \
        work["prediction_upper_bound"].replace(0, np.nan)
    work["slowdown_ratio"] = work["slowdown_ratio"].replace(
        [np.inf, -np.inf], np.nan).fillna(0)

    rows = []
    min_span = 20
    for serial, group in work.sort_values("time").groupby("serialno", sort=False):
        values = group["slowdown_ratio"].to_numpy()
        slow_events = []
        for idx in range(0, len(values), min_span):
            current = values[idx: idx + min_span]
            if current.size == 0:
                continue
            median = float(np.median(current))
            top_half = current[current >= median]
            final_median = float(np.median(top_half)) if top_half.size else 0.0
            if final_median > 1:
                slow_events.append(final_median)
        avg = float(np.mean(slow_events)) if slow_events else np.nan
        duration = len(slow_events)
        score = _score(avg, duration) if len(
            slow_events) >= min_windows else np.nan
        rows.append(
            {
                "date": day,
                "cluster": cluster,
                "host": host,
                "disk": str(serial),
                "score": score if score > 0 else np.nan,
                "avg": avg,
                "duration": duration,
            }
        )
    return pd.DataFrame(rows)


def _last_n_day_sum(group: pd.DataFrame, last_n_days: int) -> float:
    max_date = group["dt"].max()
    return float(group[group["dt"] >= (max_date - pd.DateOffset(days=last_n_days))]["score"].sum())


def flag_scores(
    score_df: pd.DataFrame,
    prefix_by_group: dict[tuple[str, str], str],
    score_threshold: float = 50.0,
    last_n_days: int = 15,
) -> list[Result]:
    decisions = decision_scores(
        score_df, score_threshold=score_threshold, last_n_days=last_n_days)
    if decisions.empty:
        return []
    flagged = decisions[decisions["decision_status"] ==
                        "T"].drop_duplicates(["cluster", "host", "disk"])
    rows = []
    for row in flagged.itertuples(index=False):
        prefix = prefix_by_group.get(
            (row.cluster, row.host), f"{row.cluster}_{row.host}")
        rg_id = f"{row.cluster}/{row.host}"
        rows.append(
            Result(prefix, f"last_{last_n_days}_days", "perseus", rg_id, str(row.disk)))
    return rows


def decision_scores(
    score_df: pd.DataFrame,
    score_threshold: float = 50.0,
    last_n_days: int = 15,
) -> pd.DataFrame:
    if score_df.empty:
        return score_df

    scored = score_df.copy()
    scored["dt"] = pd.to_datetime(scored["date"])
    fired = scored[~scored["score"].isna()]
    if fired.empty:
        scored[f"last_{last_n_days}_score"] = np.nan
        scored["decision_status"] = "F"
        scored["score_threshold"] = score_threshold
        return scored

    totals = (
        fired.groupby(["cluster", "host", "disk"])[["score", "dt"]]
        .apply(_last_n_day_sum, last_n_days=last_n_days)
        .reset_index()
        .rename(columns={0: f"last_{last_n_days}_score"})
    )
    out = pd.merge(scored, totals, how="left", on=["cluster", "host", "disk"])
    score_col = f"last_{last_n_days}_score"
    out["decision_status"] = np.where(
        out[score_col] >= score_threshold, "T", "F")
    out["score_threshold"] = score_threshold
    return out
