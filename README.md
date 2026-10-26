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

Comma-delimited. `ClassLabel`: 1 = phishing, 0 = legitimate. 24 lexical URL features (21 used after dropping zero-variance columns).

## Environment

Python 3, scikit-learn ≥ 1.0 (reference run: 1.9.1), pandas, NumPy, Matplotlib, seaborn. All preinstalled on Colab.
