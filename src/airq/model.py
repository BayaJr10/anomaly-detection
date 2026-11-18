"""Feature pruning and the SMOTE-balanced XGBoost meta-model (notebook Section 12)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from airq.detectors import BASE_FLAGS

CANDIDATE_FEATURES = [
    "residual", "PM25", "PM10", "HUMIDITY", "TEMPERATURE",
    "pm25_pm10_ratio", "hood_deviation", "hood_hour_mean", "hood_hour_std",
    "pm25_roll_mean_6", "pm25_roll_mean_24", "pm25_roll_mean_72",
    "pm25_roll_std_6", "pm25_roll_std_24", "pm25_roll_std_72",
    "PM1_missing", "HUMIDITY_missing", "TEMPERATURE_missing",
]
LEAKY = set(BASE_FLAGS + ["base_votes", "pseudo_label"])


def prune_features(train: pd.DataFrame, random_state: int = 42, min_keep: int = 6) -> tuple[list, dict, pd.Series]:
    """Exact redundancy -> |r| > 0.95 near-duplicates -> < 1% gain on a probe model. Train only."""
    assert not set(CANDIDATE_FEATURES) & LEAKY
    kept, dropped = list(CANDIDATE_FEATURES), {}
    if np.allclose(train["PM25"] - train["hood_deviation"], train["hood_hour_mean"], equal_nan=True):
        kept.remove("hood_hour_mean")
        dropped["hood_hour_mean"] = "exactly PM25 - hood_deviation"
    corr = train[kept].corr().abs()
    to_drop = set()
    for a in range(len(kept)):
        if kept[a] in to_drop:
            continue
        for b in range(a + 1, len(kept)):
            if kept[b] not in to_drop and corr.loc[kept[a], kept[b]] > 0.95:
                to_drop.add(kept[b])
                dropped[kept[b]] = f"|r| = {corr.loc[kept[a], kept[b]]:.2f} with {kept[a]}"
    kept = [c for c in kept if c not in to_drop]
    y = train["pseudo_label"]
    pos = int(y.sum())
    probe = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=(len(train) - pos) / max(pos, 1),
                              random_state=random_state, eval_metric="logloss", n_jobs=-1)
    probe.fit(SimpleImputer(strategy="median").fit_transform(train[kept]), y)
    gain = pd.Series(probe.feature_importances_, index=kept).sort_values(ascending=False)
    low = list(gain[gain < 0.01].index)
    if len(kept) - len(low) < min_keep:
        low = list(gain.nsmallest(max(len(kept) - min_keep, 0)).index)
    for f in low:
        dropped[f] = f"gain share {gain[f]:.3%} (< 1%)"
    kept = [c for c in kept if c not in low]
    return kept, dropped, gain


def make_meta_model(features: list, cfg: dict, n_estimators: int | None = None, random_state: int = 42):
    m = cfg["meta_model"]
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    return ImbPipeline([
        ("prep", ColumnTransformer([("num", num, features)])),
        ("smote", SMOTE(random_state=random_state)),
        ("clf", xgb.XGBClassifier(
            n_estimators=n_estimators or m["n_estimators"], learning_rate=m["learning_rate"],
            max_depth=m["max_depth"], subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=random_state, eval_metric="logloss", n_jobs=-1)),
    ])


def cross_validate_meta(model, train: pd.DataFrame, features: list, cfg: dict, random_state: int = 42) -> dict:
    """StratifiedGroupKFold: folds keep the anomaly rate (stratified) AND never split a sensor (grouped)."""
    n = min(cfg["meta_model"]["cv_folds"], train["LOCATION"].nunique())
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=random_state)
    s = cross_validate(model, train[features], train["pseudo_label"], groups=train["LOCATION"], cv=cv,
                       scoring=["roc_auc", "average_precision", "f1"])
    return {k.replace("test_", ""): (float(np.mean(v)), float(np.std(v)))
            for k, v in s.items() if k.startswith("test_")}


def choose_threshold(y_true, proba) -> float:
    """Operating threshold = max-F1 point on the validation PR curve (then frozen)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)]) if len(thr) else 0.5
