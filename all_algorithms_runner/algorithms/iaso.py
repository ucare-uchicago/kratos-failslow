from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN

from all_algorithms_runner.common import Result


class ScoreDB:
    def __init__(
        self,
        n: int,
        epoch: int = 10,
        ratio_thresh: float = 1.0,
        min_ttr: int = 2,
        score_multiplier: float = 1.0,
        window_length: int = 10,
        decision_interval: int = 1,
        rep_percentile: int = 30,
        dbscan_eps: float = 3.0,
        dbscan_min_samples: int = 2,
    ) -> None:
        self.n = n
        self.t = 0
        self.table = [[1.0] for _ in range(n)]
        self.decisions: list[list[bool]] = []
        self.epoch = epoch
        self.ratio_thresh = ratio_thresh
        self.min_ttr = min_ttr
        self.score_multiplier = score_multiplier
        self.window_length = window_length
        self.decision_interval = decision_interval
        self.rep_percentile = rep_percentile
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def process_epoch(self, data: list[float]) -> None:
        for idx in range(self.n):
            prev = self.table[idx][-1]
            ratio = data[idx]
            if ratio <= 0:
                score = prev - (100 * self.epoch / (self.min_ttr * 60))
            else:
                score = prev + (
                    prev
                    * (min(ratio, self.ratio_thresh) / self.ratio_thresh)
                    * self.score_multiplier
                )
            self.table[idx].append(min(max(score, 1), 100))
            if len(self.table[idx]) > self.window_length * 60 / self.epoch:
                self.table[idx].pop(0)
        self.t += 1
        if self.t * self.epoch % (self.decision_interval * 60) == 0:
            self.make_decisions()

    def make_decisions(self) -> list[bool]:
        rep_scores = [
            float(np.percentile(np.asarray(scores), self.rep_percentile))
            for scores in self.table
        ]
        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
        ).fit(np.asarray(rep_scores).reshape(-1, 1)).labels_
        decision = [label == -1 for label in labels]
        self.decisions.append(decision)
        return decision


def run(df: pd.DataFrame, day: str, rg_id: str, prefix: str, mode: str = "mediandiv") -> list[Result]:
    scores = score_daily(df, day, rg_id, mode=mode)
    if scores.empty:
        return []
    flagged = scores[scores["score"] > 0.1]
    return [
        Result(prefix, day, "iaso", rg_id, str(row.disk))
        for row in flagged.itertuples(index=False)
    ]


def score_daily(df: pd.DataFrame, day: str, rg_id: str, mode: str = "mediandiv") -> pd.DataFrame:
    work = df.reset_index()
    cluster, host = rg_id.split("/", 1)
    pivot = (
        work.pivot_table(
            index="time",
            columns="serialno",
            values="read_total_latency",
            aggfunc="mean",
        )
        .sort_index()
        .dropna(axis=1, how="all")
        .ffill()
        .dropna()
    )
    if pivot.empty or len(pivot.columns) < 2:
        return pd.DataFrame()

    scoredb = ScoreDB(len(pivot.columns))
    for _, row in pivot.iterrows():
        if mode == "zscore":
            metrics = stats.zscore(row.fillna(0)).tolist()
        else:
            median = row.median()
            metrics = [float((value / median) - 1) if median else 0.0 for value in row]
        scoredb.process_epoch(metrics)

    if not scoredb.decisions:
        return pd.DataFrame()

    rates = {
        str(disk): sum(decision[idx] for decision in scoredb.decisions)
        / len(scoredb.decisions)
        for idx, disk in enumerate(pivot.columns)
    }
    return pd.DataFrame(
        [
            {
                "date": day,
                "cluster": cluster,
                "host": host,
                "disk": disk,
                "score": rate,
            }
            for disk, rate in sorted(rates.items())
        ]
    )
