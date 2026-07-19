# BankruptWatch — Corporate Bankruptcy Prediction

Ensemble machine learning models predicting corporate bankruptcy on the Taiwan Economic Journal (TEJ) financial dataset, addressing class imbalance with SMOTE and comparing Random Forest against AdaBoost.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

**Problem.** Predicting corporate bankruptcy from financial ratios is a heavily imbalanced classification task — bankrupt firms are the minority class, which naive classifiers ignore in favor of the majority.

**Approach.**

1. Cleaned and profiled the TEJ dataset (Taiwan-listed companies, 1999–2009).
2. Balanced the training set with **SMOTE** (Synthetic Minority Over-sampling).
3. Trained and compared two ensemble classifiers: **Random Forest** and **AdaBoost**.
4. Ranked driving financial ratios via feature importance to make results business-interpretable.

**Result.** Best model reaches **92% accuracy** on the held-out test set.

## Tech Stack

- **Python**, **Jupyter Notebook**
- **scikit-learn** — RandomForestClassifier, AdaBoostClassifier, train/test split, metrics
- **imbalanced-learn** — SMOTE
- **pandas / NumPy** — data handling
- **Matplotlib / seaborn** — EDA and feature-importance plots

## Dataset

Source: [Company Bankruptcy Prediction — Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction), originally from the **Taiwan Economic Journal**, covering listed companies from 1999–2009. The dataset contains 95 financial ratios per company and a binary bankruptcy label defined by Taiwan Stock Exchange regulations.

## Project Structure

```
BankruptWatch/
├── dataset/
│   └── data.csv                          # TEJ financial ratios
├── notebooks/
│   └── Bankruptcy_Prediction.ipynb       # EDA, SMOTE, RF vs AdaBoost
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
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

**3. Open the notebook**

```bash
jupyter notebook notebooks/Bankruptcy_Prediction.ipynb
```

Run all cells to reproduce the EDA, SMOTE balancing, model training, and Random Forest vs. AdaBoost comparison.

## Key Findings

- Class imbalance in the raw data is severe (~3% bankruptcy rate). Without SMOTE, both models over-predict the majority class.
- After SMOTE, **Random Forest** achieved the highest accuracy at 92%.
- Feature importance highlights ratios tied to **debt-to-equity**, **retained earnings**, and **operating cash flow** as the strongest bankruptcy predictors — consistent with corporate-finance theory.

## License

MIT. See [LICENSE](LICENSE).

## Author

**Zul Iflah Al Juhaeda** — [LinkedIn](https://linkedin.com/in/aljuhaeda) · [GitHub](https://github.com/aljuhaeda)
