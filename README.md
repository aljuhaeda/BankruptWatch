# BankruptWatch — Corporate Bankruptcy Prediction

Ensemble machine learning models predicting corporate bankruptcy on the Taiwan Economic Journal (TEJ) financial dataset, comparing Random Forest against AdaBoost with SMOTE applied correctly (training fold only, after the train/test split) to avoid leaking synthetic minority-class data into evaluation.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**Problem.** Predicting corporate bankruptcy from financial ratios is a heavily imbalanced classification task — only ~3% of companies in the dataset are bankrupt. A model that never predicts bankruptcy still scores ~97% accuracy while catching zero real cases, which makes accuracy alone a misleading headline number here.

**Approach.**

1. Cleaned and profiled the TEJ dataset (6,819 Taiwan-listed companies, 1999–2009, 96 financial ratios).
2. Split into train/test **before** any resampling or scaling, so the test set stays an untouched sample of the real ~97/3 class distribution.
3. Applied **SMOTE** to the training fold only, then trained and grid-search-tuned two ensemble classifiers: **Random Forest** and **AdaBoost**.
4. Evaluated with precision/recall/F1 on the bankrupt class and a confusion matrix — not just accuracy — since accuracy alone can't tell you whether the model actually catches bankruptcies.
5. Ranked driving financial ratios via Random Forest feature importance.

**Result.** Random Forest catches **57% of actual bankruptcies** (vs. 0% for a naive majority-class baseline), trading a few points of raw accuracy for a model that's actually useful as a screener. See **Results** below.

## Demo

**Live**: [bankruptwtch.streamlit.app](https://bankruptwtch.streamlit.app/) — adjust the 10 most influential financial ratios (sliders, defaulting to the training set's median), the rest default to their median, and it returns a bankruptcy-risk flag with predicted probability.

Run it locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Results

Evaluated on a held-out test set (1,364 companies, 44 actually bankrupt — the real class imbalance, untouched by resampling):

| Model | Accuracy | Bankrupt Precision | Bankrupt Recall | Bankrupt F1 |
|---|---|---|---|---|
| Always predict "not bankrupt" (baseline) | 96.8% | — | 0% | — |
| **Random Forest (deployed)** | 95.9% | 40.3% | **56.8%** | 47.2% |
| AdaBoost | 92.2% | 23.7% | 63.6% | 34.6% |

Hyperparameters for both models were chosen by 5-fold grid search (see notebook) — `n_estimators=200, max_depth=None` for Random Forest, `n_estimators=200, learning_rate=1.0` for AdaBoost.

**Why lower accuracy than the naive baseline is the right trade-off:** always predicting "not bankrupt" scores 96.8% accuracy by doing nothing useful. Random Forest gives up ~1 point of accuracy to catch 57% of actual bankruptcies instead of 0% — for a bankruptcy screener, recall on the minority class is what matters, not overall accuracy. AdaBoost has higher recall (64%) but far worse precision (roughly 3 in 4 flags is a false alarm, vs. 3 in 5 for Random Forest), which is why Random Forest — the better F1 — is the deployed model.

Top predictive features (Random Forest importance): **Borrowing dependency**, **Net Income to Total Assets**, **Persistent EPS (last 4 seasons)**, **After-tax net Interest Rate**, **Liability to Equity**, **Debt ratio**, **Net worth/Assets** — consistent with corporate-finance theory (leverage and profitability ratios dominate).

## Tech Stack

- **Python**, **Jupyter Notebook**
- **scikit-learn** — RandomForestClassifier, AdaBoostClassifier, GridSearchCV
- **imbalanced-learn** — SMOTE
- **Streamlit** — interactive demo
- **pandas / NumPy** — data handling
- **Matplotlib / seaborn** — EDA and feature-importance plots

## Dataset

Source: [Company Bankruptcy Prediction — Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction), originally from the **Taiwan Economic Journal**, covering listed companies from 1999–2009. 96 financial ratios per company (95 features + label) and a binary bankruptcy label defined by Taiwan Stock Exchange regulations.

## Project Structure

```
BankruptWatch/
├── dataset/
│   └── BankruptData.csv              # TEJ financial ratios (6,819 rows)
├── notebooks/
│   └── BankruptWatch.ipynb           # EDA, correct SMOTE pipeline, RF vs AdaBoost, evaluation
├── model/
│   ├── model.pkl                     # Trained Random Forest (deployed in app.py)
│   ├── scaler.pkl                    # StandardScaler fit on the training fold
│   ├── feature_names.pkl             # Column order the model expects
│   ├── feature_medians.pkl           # Training-set medians (demo app defaults)
│   └── results.json                  # Evaluation metrics for both models
├── train.py                          # Standalone training script (regenerates model/)
├── app.py                            # Streamlit demo
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

**1. Clone the repo**

```bash
git clone https://github.com/aljuhaeda/BankruptWatch.git
cd BankruptWatch
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the demo**

```bash
streamlit run app.py
```

**4. (Optional) Retrain the model**

```bash
python train.py
```

**5. (Optional) Explore the full notebook**

```bash
jupyter notebook notebooks/BankruptWatch.ipynb
```

Run all cells to reproduce the EDA, correct SMOTE pipeline, grid-search-tuned model comparison, and evaluation.

## Notes on Fixes Since the Original Version

This project previously reported "92% accuracy with SMOTE" — a number that didn't actually match anything in the notebook. Digging in surfaced several real issues, now fixed:

- **The notebook couldn't run outside the original Kaggle session.** Data was loaded from `../input/company-bankruptcy-prediction/data.csv`, a Kaggle-only mount path, and `pandas` was never actually imported before its first use — anyone cloning the repo hit an immediate error.
- **SMOTE was applied but never used.** An early "Oversampling" section ran SMOTE on the full dataset, but the actual model-training section reloaded the raw, untouched, ~97/3 imbalanced data and never referenced the SMOTE output. The reported ~96% accuracy numbers came from models never exposed to a balanced training set.
- **No precision/recall/F1 was ever computed.** With only accuracy reported on a 97/3 imbalanced test set, there was no way to tell whether the model caught any bankruptcies at all. (It's now confirmed it barely did — see Results.)
- **A `matplotlib` API change broke one of the EDA plots** (`boxplot(labels=...)` was renamed to `tick_labels`).

The notebook, `train.py`, and this README now reflect a pipeline that splits before resampling (avoiding data leakage), applies SMOTE to the training fold only, and reports the metrics that actually matter for an imbalanced classification problem.

## Limitations

- **Small positive class** (44 bankrupt companies in the test set) — precision/recall estimates have real uncertainty at this sample size.
- **34.8% precision** means roughly 2 in 3 bankruptcy flags are false alarms — appropriate for a low-cost screening tool, not for automated decisions.
- **Static financial ratios only**, no time-series trend data — the model can't see a company's trajectory, only a snapshot.
- Not a financial advisory tool; see the in-app disclaimer.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Zul Iflah Al Juhaeda** — [LinkedIn](https://linkedin.com/in/aljuhaeda) · [GitHub](https://github.com/aljuhaeda)
