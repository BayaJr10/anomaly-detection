"""The five base anomaly detectors (notebook Section 11).

Each detector is fitted on the full training period and applied unchanged to validation
and test. They produce the consensus pseudo-label the meta-model learns; they are not
needed at serving time.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import KDTree, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.seasonal import STL

from airq.preprocess import robust_z

log = logging.getLogger(__name__)
BASE_FLAGS = ["stat_flag", "lof_flag", "svm_flag", "dbscan_flag", "stl_flag"]


def _batched(fn, X, size=20_000):
    return np.concatenate([fn(X[i:i + size]) for i in range(0, len(X), size)])


# --------------------------- 11.5 STL helpers ---------------------------------
def stl_on_segments(g: pd.DataFrame, period: int = 24, min_len: int = 48) -> pd.DataFrame:
    """Robust STL on unbroken segments of one sensor (a gap > 1 day starts a new segment)."""
    g = g.sort_values("DATETIME").copy()
    g["segment"] = (g["DATETIME"].diff() > pd.Timedelta("1D")).cumsum()
    g["stl_resid"] = np.nan
    for _, seg in g.groupby("segment"):
        if len(seg) >= min_len:
            try:
                res = STL(seg["PM25"].astype(float).values, period=period, robust=True).fit()
                g.loc[seg.index, "stl_resid"] = res.resid
            except Exception as e:  # pragma: no cover - numerical edge cases
                log.debug("STL failed on a segment: %s", e)
    return g.drop(columns="segment")


def run_stl(d: pd.DataFrame, period: int = 24, min_len: int = 48) -> pd.Series:
    """STL residual for every row of one split (each split is decomposed on its own)."""
    out = pd.Series(np.nan, index=d.index)
    for _, g in d.groupby("LOCATION", sort=False):
        r = stl_on_segments(g, period, min_len)
        out.loc[r.index] = r["stl_resid"].values
    return out


class DetectorSuite:
    """Statistical, density, kernel, clustering and time-series detectors."""

    def __init__(self, cfg: dict, random_state: int = 42):
        self.c = cfg["detectors"]
        self.rs = random_state
        self.features = self.c["detector_features"]

    # ----- fit on training data only -----
    def fit(self, train: pd.DataFrame) -> "DetectorSuite":
        c = self.c
        # 11.1 statistical: MAD of the residual
        self.res_med_ = float(train["residual"].median())
        self.res_mad_ = float(np.median(np.abs(train["residual"] - self.res_med_)))
        # shared robust scaling for 11.2 - 11.4
        self.fill_ = train[self.features].median()
        self.scaler_ = RobustScaler().fit(train[self.features].fillna(self.fill_))
        Xtr = self._X(train)
        # 11.2 density: LOF in novelty mode
        self.lof_ = LocalOutlierFactor(n_neighbors=c["lof_neighbors"], contamination=c["lof_contamination"],
                                       novelty=True).fit(Xtr)
        # 11.3 kernel: Nystroem RBF + SGD one-class SVM (scales to the full training set)
        gamma = 1.0 / (Xtr.shape[1] * Xtr.var())
        self.ocsvm_ = Pipeline([
            ("nystroem", Nystroem(kernel="rbf", gamma=gamma, n_components=c["ocsvm_components"],
                                  random_state=self.rs)),
            ("svm", SGDOneClassSVM(nu=c["ocsvm_nu"], tol=1e-4, max_iter=1000, random_state=self.rs)),
        ]).fit(Xtr)
        # 11.4 clustering: exact DBSCAN noise rule with KD-trees
        tree = KDTree(Xtr)
        kth = _batched(lambda b: tree.query(b, k=c["dbscan_min_samples"])[0][:, -1], Xtr)
        core = kth <= c["dbscan_eps"]
        self.n_core_ = int(core.sum())
        self.core_tree_ = KDTree(Xtr[core]) if core.any() else None
        # 11.5 time series: per-sensor MAD of training STL residuals
        stl_tr = run_stl(train, c["stl_period"], c["stl_min_len"])
        self.stl_params_ = {}
        for loc, r in stl_tr.groupby(train["LOCATION"]):
            r = r.dropna()
            if len(r) >= 24:
                med = float(r.median())
                mad = float(np.median(np.abs(r - med)))
                if mad > 0:
                    self.stl_params_[loc] = (med, mad)
        self._train_stl = stl_tr
        return self

    def _X(self, d: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(d[self.features].fillna(self.fill_))

    # ----- apply to any split -----
    def flag(self, d: pd.DataFrame, stl_resid: pd.Series | None = None) -> pd.DataFrame:
        c, out = self.c, pd.DataFrame(index=d.index)
        out["stat_flag"] = (np.abs(robust_z(d["residual"], self.res_med_, self.res_mad_))
                            > c["mad_threshold"]).astype(int)
        X = self._X(d)
        out["lof_flag"] = (self.lof_.predict(X) == -1).astype(int)
        out["svm_flag"] = (self.ocsvm_.predict(X) == -1).astype(int)
        if self.core_tree_ is not None:
            dist = _batched(lambda b: self.core_tree_.query(b, k=1)[0][:, 0], X)
            out["dbscan_flag"] = (dist > c["dbscan_eps"]).astype(int)
        else:
            out["dbscan_flag"] = 0
        if stl_resid is None:
            stl_resid = run_stl(d, c["stl_period"], c["stl_min_len"])
        out["stl_resid"] = stl_resid
        flag = np.zeros(len(d), dtype=int)
        r = stl_resid.values
        for loc, pos in d.groupby("LOCATION").indices.items():
            if loc in self.stl_params_:
                med, mad = self.stl_params_[loc]
                flag[pos] = (np.abs(0.6745 * (r[pos] - med) / mad) > c["mad_threshold"]).astype(int)
        out["stl_flag"] = flag
        out["base_votes"] = out[BASE_FLAGS].sum(axis=1)
        out["pseudo_label"] = (out["base_votes"] >= c["consensus_votes"]).astype(int)
        return out

    def flag_train(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.flag(train, stl_resid=self._train_stl)
