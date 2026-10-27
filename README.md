# Job-Recruitment Phishing URL Detection using Random Forest

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fizzz96/phishing-url-rf/blob/main/phishing_rf_colab.ipynb)

Lexical-feature Random Forest classifier for job-recruitment phishing URLs.

## Run

Click **Open in Colab**, then *Runtime → Run all*. The notebook downloads the data from `data/` in this repository and verifies its SHA-256 checksum. No upload or Google account access is needed.

## Data

| File | Rows | SHA-256 |
|---|---|---|
| `data/train_features.csv` | 6,298 | `b62d19c9d1f07d504cbfac679a06bd07c7b212926e24947cfcb9eb4c10706c2a` |
| `data/test_features.csv` | 1,575 | `9cf35014953f51dac1b249cbaf067f00b40b259480af488a251f8753c214fb4e` |
| `data/cv_folds.csv` | 6,298 | `70533f5a41db53bf7d9c24b2d3c1509fe8ccc371f5272ee448658c0c4cc85ace` |

Comma-delimited. `ClassLabel`: 1 = phishing, 0 = legitimate. 24 lexical URL features (21 used after dropping zero-variance columns).

## Shared cross-validation folds (for all models)

Every classifier in the comparison must use the folds in `data/cv_folds.csv`, not its own `KFold` or `StratifiedGroupKFold`. Each training URL is assigned to one of five folds, grouped by site (registered domain, so `www.indeed.com` and `id.indeed.com` count as one site). No site appears in two folds; each validation fold holds out 5 of the 25 legitimate sites. Identical folds make the per-fold scores of different models directly comparable.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

BASE = "https://raw.githubusercontent.com/fizzz96/phishing-url-rf/main/data"
train = pd.read_csv(f"{BASE}/train_features.csv")
folds = pd.read_csv(f"{BASE}/cv_folds.csv").set_index("url")["fold"]

fold_id = train["url"].map(folds).to_numpy()
cv_splits = [(np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)) for k in range(5)]

# X_train, y_train built from `train` in its original row order
res = cross_validate(model, X_train, y_train, cv=cv_splits, scoring=scoring)
```

Rules:

- Keep `train` in the row order of `train_features.csv`; the split indices refer to that order.
- Put any preprocessing that learns from data (scaling, imputation) inside a `Pipeline` with the model, so it is fitted inside each fold. Fitting a scaler on the full training set before CV leaks validation data.
- Use the same 21 features for every model (the 24 columns minus `hash`, `hasipaddress`, `suspiciousfileextension`, which are constant in train).
- Save per-fold scores to `results/cv_scores_<model>.csv` with columns `model, fold, metric, value` and metrics `accuracy, precision, recall, f1, roc_auc, mcc`. See `results/cv_scores_randomforest.csv`.

## Environment

Python 3, scikit-learn ≥ 1.0 (reference run: 1.9.1), pandas, NumPy, Matplotlib, seaborn. All preinstalled on Colab.
