"""Five-model comparison on shared site-grouped folds + hold-out test.

Usage: python compare_models.py <data_dir> <out_dir>
data_dir must contain train_features.csv, test_features.csv, cv_folds.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 42
TARGET, ID_COL = "ClassLabel", "url"
data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(data_dir / "train_features.csv")
test = pd.read_csv(data_dir / "test_features.csv")
folds = pd.read_csv(data_dir / "cv_folds.csv").set_index("url")["fold"]

features = [c for c in train.columns
            if c not in (ID_COL, TARGET) and train[c].nunique() > 1]   # 21 features
assert len(features) == 21, features
fold_id = train[ID_COL].map(folds).to_numpy()
K = int(fold_id.max()) + 1
splits = [(np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)) for k in range(K)]

MODELS = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "SVM (RBF, Platt)":    make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
    "Random Forest":       RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1),
    "LightGBM":            LGBMClassifier(random_state=SEED, verbose=-1),
    "XGBoost":             XGBClassifier(random_state=SEED, eval_metric="logloss"),
}
METRICS = {
    "Accuracy":  lambda y, p, s: accuracy_score(y, p),
    "Precision": lambda y, p, s: precision_score(y, p),
    "Recall":    lambda y, p, s: recall_score(y, p),
    "F1":        lambda y, p, s: f1_score(y, p),
    "ROC-AUC":   lambda y, p, s: roc_auc_score(y, s),
    "MCC":       lambda y, p, s: matthews_corrcoef(y, p),
}


def evaluate(model, X_tr, y_tr, X_ev, y_ev):
    m = clone(model).fit(X_tr, y_tr)
    s = m.predict_proba(X_ev)[:, 1]
    p = (s >= 0.5).astype(int)
    return {name: fn(y_ev, p, s) for name, fn in METRICS.items()}


def run(feature_set, tag):
    X, y = train[feature_set], train[TARGET]
    rows = []
    for name, model in MODELS.items():
        for k, (tr, va) in enumerate(splits):
            r = evaluate(model, X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
            rows += [{"setting": tag, "model": name, "fold": k, "metric": m, "value": v} for m, v in r.items()]
    return pd.DataFrame(rows)


def corrected_ttest(a, b, n_tr, n_va):
    # Nadeau & Bengio (2003) corrected resampled t-test, paired over folds
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_va / n_tr) * var)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)


cv = pd.concat([run(features, "all 21 features"),
                run([f for f in features if f != "httpsflag"], "without httpsflag")])
cv.to_csv(out_dir / "cv_scores_all_models.csv", index=False)

test_rows = []
for name, model in MODELS.items():
    r = evaluate(model, train[features], train[TARGET], test[features], test[TARGET])
    test_rows.append({"model": name, **r})
test_res = pd.DataFrame(test_rows).set_index("model")
test_res.to_csv(out_dir / "test_scores_all_models.csv")

# Paired comparison of every model against the best mean-MCC model (main setting)
main = cv[cv.setting == "all 21 features"]
mcc = main[main.metric == "MCC"].pivot(index="fold", columns="model", values="value")
best = mcc.mean().idxmax()
n_va = np.mean([len(va) for _, va in splits]); n_tr = len(train) - n_va
tests = []
for m in mcc.columns:
    if m == best:
        continue
    t, p = corrected_ttest(mcc[best], mcc[m], n_tr, n_va)
    tests.append({"comparison": f"{best} vs {m}", "mean_diff_MCC": mcc[best].mean() - mcc[m].mean(), "t": t, "p": p})
pd.DataFrame(tests).to_csv(out_dir / "paired_tests_mcc.csv", index=False)

pd.set_option("display.width", 200)
summary = (main.groupby(["model", "metric"])["value"].agg(["mean", lambda v: v.std(ddof=1)])
           .rename(columns={"<lambda_0>": "sd"}))
print(summary.round(4).unstack("metric").to_string())
print("\nTEST\n", test_res.round(4).to_string())
print("\nTESTS (best =", best, ")\n", pd.DataFrame(tests).round(4).to_string(index=False))
sens = cv.groupby(["setting", "model", "metric"])["value"].mean().unstack("setting")
print("\nSENSITIVITY\n", sens.xs("MCC", level="metric").round(4).to_string())
print("\n", sens.xs("F1", level="metric").round(4).to_string())
