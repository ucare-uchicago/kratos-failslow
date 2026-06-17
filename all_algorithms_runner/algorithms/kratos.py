from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance

from all_algorithms_runner.common import Result, indexed_to_columns


NUM_BIN = 10


def _custom_cumulative(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    values = [max(0, int(series.iloc[0]))]
    for idx in range(1, len(series)):
        values.append(max(0, values[-1] + int(series.iloc[idx])))
    return pd.Series(values, index=series.index)


def _distribution(values: pd.Series, bin_range: tuple[float, float]) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.zeros(NUM_BIN)
    if bin_range[0] == bin_range[1]:
        return np.ones(NUM_BIN) / NUM_BIN
    return np.histogram(arr, bins=NUM_BIN, range=bin_range)[0] / (arr.size + 1)


def _divergence(left: pd.Series, right: pd.Series, wsd: bool) -> float:
    left_values = pd.to_numeric(left, errors="coerce").dropna()
    right_values = pd.to_numeric(right, errors="coerce").dropna()
    if left_values.empty or right_values.empty:
        raise ValueError("empty disk series")
    if wsd:
        return float(wasserstein_distance(left_values, right_values))

    bin_range = (
        float(min(left_values.min(), right_values.min())),
        float(max(left_values.max(), right_values.max())),
    )
    left_dist = _distribution(left_values, bin_range)
    right_dist = _distribution(right_values, bin_range)
    return float(entropy(left_dist + 1e-6, right_dist + 1e-6))


def run(
    df: pd.DataFrame,
    day: str,
    rg_id: str,
    prefix: str,
    score_threshold: float = 1000,
    zscore_threshold: float = 1.5,
    pct_threshold: float = 0.10,
    last_n_days: int = 15,
    wsd: bool = False,
) -> list[Result]:
    score_df = score_daily(df, day, rg_id)
    return flag_scores(
        score_df,
        prefix_by_group={
            (score_df.iloc[0].cluster, score_df.iloc[0].host): prefix}
        if not score_df.empty
        else {},
        score_threshold=score_threshold,
        zscore_threshold=zscore_threshold,
        pct_threshold=pct_threshold,
        last_n_days=last_n_days,
    )


def score_daily(
    df: pd.DataFrame,
    day: str,
    rg_id: str,
    wsd: bool = False,
) -> pd.DataFrame:
    work = indexed_to_columns(df)
    if work.empty or work["serialno"].nunique() < 2:
        return pd.DataFrame()

    cluster, host = rg_id.split("/", 1)

    blocks = work["read_blocks_delta"].replace(0, np.nan).fillna(1e-6)
    work["blk_latency"] = work["read_total_latency"] / blocks
    work["blk_latency_stddev"] = work.groupby(
        "time")["blk_latency"].transform("std").fillna(0)
    work["blk_latency_mean"] = work.groupby(
        "time")["blk_latency"].transform("mean")
    work["flagged"] = work["blk_latency"] >= (
        work["blk_latency_mean"] + work["blk_latency_stddev"] * 2
    )
    work["cumulative_flags"] = np.where(work["flagged"], 1, -1)
    work["cumulative_flags"] = work.groupby("serialno", sort=False)["cumulative_flags"].transform(
        _custom_cumulative
    )

    serials = sorted(work["serialno"].dropna().astype(str).unique())
    rows = []
    for serial in serials:
        divergences = []
        left = work[work["serialno"] == serial]["cumulative_flags"]
        for other in serials:
            if serial == other:
                continue
            right = work[work["serialno"] == other]["cumulative_flags"]
            try:
                divergences.append(_divergence(left, right, wsd))
            except ValueError:
                continue
        current = work[work["serialno"] == serial]
        total_cnt = int(current["cumulative_flags"].notna().sum())
        zero_cnt = int((current["cumulative_flags"] == 0).sum())
        max_oc = float(current["cumulative_flags"].max())
        rows.append(
            {
                "date": day,
                "cluster": cluster,
                "host": host,
                "disk": serial,
                "zero_cnt": zero_cnt,
                "total_cnt": total_cnt,
                "max_oc": max_oc,
                "kl": float(np.sum(divergences)) if divergences else 0.0,
                "compared_cnt": len(divergences),
            }
        )
    return pd.DataFrame(rows)


def _last_n_day_sum(group: pd.DataFrame, last_n_days: int) -> float:
    max_date = group["dt"].max()
    return float(group[group["dt"] >= (max_date - pd.DateOffset(days=last_n_days))]["score"].sum())


def _apply_score_rules(
    df: pd.DataFrame,
    zscore_threshold: float,
    pct_threshold: float,
) -> pd.DataFrame:
    work = df[df["total_cnt"] != 0].copy()
    if work.empty:
        return work

    group_cols = ["cluster", "host", "date"]
    work["percent"] = (work["total_cnt"] - work["zero_cnt"]
                       ) / work["total_cnt"]
    work["max_kl"] = work.groupby(group_cols)["kl"].transform("max")
    work["adjusted_mean"] = work[work["max_kl"] != work["kl"]].groupby(group_cols)[
        "kl"].transform("mean")
    work["adjusted_mean"] = work.groupby(
        group_cols)["adjusted_mean"].transform("max")
    work["adjusted_std"] = work[work["max_kl"] != work["kl"]].groupby(group_cols)[
        "kl"].transform("std")
    work["adjusted_std"] = work.groupby(
        group_cols)["adjusted_std"].transform("max")
    work["adjusted_zscore_kl"] = (
        work["kl"] - work["adjusted_mean"]) / work["adjusted_std"]
    work["score"] = np.nan

    high_z = work["adjusted_zscore_kl"] > zscore_threshold
    high_pct = work["percent"] > pct_threshold
    work.loc[high_z, "score"] = 0
    work.loc[(work["kl"] <= 0.5) & high_pct & high_z, "score"] = 1
    work.loc[(work["kl"] <= 1) & (work["kl"] > 0.5)
             & high_pct & high_z, "score"] = 5
    work.loc[(work["kl"] <= 1.5) & (work["kl"] > 1)
             & high_pct & high_z, "score"] = 10
    work.loc[(work["kl"] <= 3) & (work["kl"] > 1.5)
             & high_pct & high_z, "score"] = 25
    work.loc[(work["kl"] > 3) & high_pct & high_z, "score"] = 100
    work["dt"] = pd.to_datetime(work["date"])
    return work


def flag_scores(
    score_df: pd.DataFrame,
    prefix_by_group: dict[tuple[str, str], str],
    score_threshold: float = 1000,
    zscore_threshold: float = 1.5,
    pct_threshold: float = 0.10,
    last_n_days: int = 15,
) -> list[Result]:
    scored = _apply_score_rules(score_df, zscore_threshold, pct_threshold)
    fired = scored[~scored["score"].isna()]
    if fired.empty:
        return []

    totals = (
        fired.groupby(["cluster", "host", "disk"])[["score", "dt"]]
        .apply(_last_n_day_sum, last_n_days=last_n_days)
        .reset_index()
        .rename(columns={0: "total_score"})
    )
    flagged = totals[totals["total_score"] >= score_threshold]
    rows = []
    for row in flagged.itertuples(index=False):
        prefix = prefix_by_group.get(
            (row.cluster, row.host), f"{row.cluster}_{row.host}")
        rg_id = f"{row.cluster}/{row.host}"
        rows.append(
            Result(prefix, f"last_{last_n_days}_days", "kratos", rg_id, str(row.disk)))
    return rows


def decision_scores(
    score_df: pd.DataFrame,
    score_threshold: float = 1000.0,
    zscore_threshold: float = 1.5,
    pct_threshold: float = 0.10,
    last_n_days: int = 15,
) -> pd.DataFrame:
    scored = _apply_score_rules(score_df, zscore_threshold, pct_threshold)
    if scored.empty:
        return scored

    fired = scored[~scored["score"].isna()]
    if fired.empty:
        scored["last_15_score"] = np.nan
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
