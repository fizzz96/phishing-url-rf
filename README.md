# Phishing URL Detection using Random Forest (Stage 1)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fizzz96/phishing-url-rf/blob/main/phishing_rf_colab.ipynb)

Random Forest baseline for stage 1 (phishing vs legitimate URL) of the IFLP-ICDSOS 2026 paper *A Comparative Analysis of Machine Learning Algorithms for Job Phishing Detection*. The repository also holds the **shared data and cross-validation folds** that every stage-1 model in the paper must use.

## Run

Click **Open in Colab**, then *Runtime → Run all* (about 10 minutes on a standard runtime). The notebook downloads the data from `data/` in this repository and verifies its SHA-256 checksum. No upload or Google account access is needed.

## Data (v3)

| File | Rows | Source file from the team | SHA-256 |
|---|---|---|---|
| `data/train_features.csv` | 79,998 | `features_24_hybrid.csv` | `4dfb71b28307b9ab34fb19e0595a222031c43b5c8a1dea3bad47ead02fa6a35d` |
| `data/test_features.csv` | 38,387 | `testing_features_24_hybrid.csv` | `c59cd8e238db5b534b48e7fbce7532a9d6d233949efa2e915d1b49f328ea373e` |
| `data/cv_folds.csv` | 79,998 | built from the training file | `7f8a1b8865806c29ad8d01c3882c5bdfedd32c166012366d5d13e5c4c4242f2d` |

- Legitimate URLs from the Tranco list, phishing URLs from PhishTank (verified, online-valid).
- Training set collected 19 Sep 2026: 39,999 legitimate / 39,999 phishing.
- Test set collected 28 Oct 2026, with URLs already in the training set removed: 14,658 legitimate / 23,729 phishing (61.82% phishing).
- Columns: `URL`, `ClassLabel` (1 = phishing, 0 = legitimate), and 24 lexical features. No missing values and no zero-variance features.

The files are byte-identical to the team's versions; the notebook does not remove any rows, so the counts match the dataset described in the paper (sections 2.2–2.3).

## Feature sets

| Set | Features | Used by |
|---|---|---|
| FS-24 | all 24 features | Random Forest, LightGBM, XGBoost |
| FS-20 | FS-24 without `has_http`, `tokencount`, `count_letters`, `numberofdigits` | Logistic Regression, SVM |

The four removed columns come from the pairs with |Pearson r| > 0.8 in the training set (`has_http` = 1 − `httpsflag`; `urllength`–`count_letters` r = 0.997; `urllength`–`numberofdigits` 0.949; `tokencount`–`count_non_alphanumeric` 0.927; `numberofdigits`–`count_letters` 0.926).

## Shared cross-validation folds (for all models)

Every classifier in the comparison must use the folds in `data/cv_folds.csv`, not its own `KFold`, `StratifiedKFold` or `StratifiedGroupKFold`. Each training URL is assigned to one of five folds, grouped by **site** (registered domain, so `www.indeed.com` and `id.indeed.com` count as one site, and every `qrco.de` link stays in one fold). No site appears in two folds; each fold holds about 16,000 URLs with 50% phishing. Identical folds make the per-fold scores of different models paired observations, and they are identical across scikit-learn versions.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

BASE = "https://raw.githubusercontent.com/fizzz96/phishing-url-rf/main/data"
train = pd.read_csv(f"{BASE}/train_features.csv")
folds = pd.read_csv(f"{BASE}/cv_folds.csv").set_index("URL")["fold"]

fold_id = train["URL"].map(folds).to_numpy()
cv_splits = [(np.flatnonzero(fold_id != k), np.flatnonzero(fold_id == k)) for k in range(5)]

# X_train, y_train built from `train` in its original row order (FS-24 or FS-20)
res = cross_validate(model, X_train, y_train, cv=cv_splits, scoring=scoring)
```

Rules:

- Keep `train` in the row order of `train_features.csv`; the split indices refer to that order.
- Put any preprocessing that learns from data (scaling, imputation) inside a `Pipeline` with the model, so it is fitted inside each fold. Fitting a scaler on the full training set before CV leaks validation data.
- Use FS-24 for tree-based models and FS-20 for LR and SVM (table above).
- Save per-fold scores to `results/cv_scores_<model>.csv` with columns `model, fold, metric, value` and metrics `accuracy, precision, recall, f1, roc_auc, mcc`. See `results/cv_scores_randomforest.csv`.
- Evaluate the test set once, and report it overall and split by whether the URL's site occurs in the training set (see notebook section 9).

## Random Forest results (reference run)

`n_estimators=500`, `max_features="sqrt"` (4 of 24 features per split), `class_weight=None`, `random_state=42`, threshold 0.5, FS-24.

| Evaluation | n | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Site-grouped 5-fold CV (mean ± SD) | 79,998 | 0.9788 ± 0.0008 | 0.9795 ± 0.0007 | 0.9780 ± 0.0015 | 0.9787 ± 0.0008 | 0.9575 ± 0.0016 | 0.9950 ± 0.0010 |
| Test, all URLs | 38,387 | 0.9667 | 0.9678 | 0.9786 | 0.9732 | 0.9292 | 0.9929 |
| Test, site seen in training | 19,366 | 0.9933 | 0.9979 | 0.9953 | 0.9966 | 0.5903 | 0.9812 |
| Test, site not seen in training | 19,021 | 0.9395 | 0.8476 | 0.9072 | 0.8764 | 0.8373 | 0.9716 |

Test confusion matrix: TN 13,885, FP 773, FN 507, TP 23,222. The "seen site" subset contains only 132 legitimate URLs, so its MCC is unstable. Path features (`pathlength`, `tokencount`, `count_slashes`) carry much of the signal: a single rule `pathlength > 0` already reaches F1 0.908 on the test set. Notebook section 12 discusses both points.

Output files: `results/cv_scores_randomforest.csv`, `results/test_metrics_randomforest.csv`, `results/feature_importance_randomforest.csv`, `figures/fig_confusion_matrix.png`, `figures/fig_feature_importance.png`.

## Environment

Python 3, scikit-learn ≥ 1.0 (reference run: 1.9.1), pandas, NumPy, Matplotlib, seaborn, tldextract (installed automatically on Colab).
