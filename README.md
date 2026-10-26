# Job-Recruitment Phishing URL Detection using Random Forest

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<github-username>/<repo-name>/blob/main/phishing_rf_colab.ipynb)

Lexical-feature Random Forest classifier for job-recruitment phishing URLs.

## Run

Click **Open in Colab**, then *Runtime → Run all*. The notebook downloads the data from `data/` in this repository and verifies its SHA-256 checksum. No upload or Google account access is needed.

## Data

| File | Rows | SHA-256 |
|---|---|---|
| `data/train_features.csv` | 6,298 | `bd41c77df42ddd0eef9c774818f6516118bf86d9220948fb568198b1e9678f83` |
| `data/test_features.csv` | 1,575 | `ca038969f8fe216ba2f5662414fe676a086c0b863dd6d6d0fe362a15e0478ac8` |

Semicolon-delimited. `label`: 1 = phishing, 0 = legitimate.

## Environment

Python 3, scikit-learn ≥ 1.0 (reference run: 1.9.1), pandas, NumPy, Matplotlib, seaborn. All preinstalled on Colab.
