import os

import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

st.set_page_config(page_title="BankruptWatch", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    feature_medians = joblib.load(os.path.join(MODEL_DIR, "feature_medians.pkl"))
    return model, scaler, feature_names, feature_medians


model, scaler, feature_names, feature_medians = load_artifacts()

# The 10 features the model weighs most heavily (see model/results.json).
# Everything else defaults to its training-set median -- 96 financial
# ratios is not a reasonable form to hand anyone.
TOP_FEATURES = [
    "Borrowing dependency",
    "Net Income to Total Assets",
    "Persistent EPS in the Last Four Seasons",
    "After-tax net Interest Rate",
    "Liability to Equity",
    "Debt ratio %",
    "Continuous interest rate (after tax)",
    "Net worth/Assets",
    "Total income/Total expense",
    "Retained Earnings to Total Assets",
]

st.title("📉 BankruptWatch")
st.markdown(
    "Predicts corporate bankruptcy risk from financial ratios, trained on the "
    "[Taiwan Economic Journal dataset](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) "
    "(1999–2009). All ratios below are normalized 0–1 as in the source dataset; "
    "unlisted ratios are held at their training-set median."
)

st.warning(
    "⚠️ This model catches roughly 6 out of 10 real bankruptcies in testing, at the cost of "
    "a meaningful false-positive rate — see **Model Performance** below before reading too much "
    "into any single prediction. It's a demonstration of the modeling approach, not a financial "
    "decision tool.",
    icon="⚠️",
)

st.subheader("Key Financial Ratios")
st.caption("Sliders default to the training dataset's median value for each ratio.")

inputs = {}
col1, col2 = st.columns(2)
for i, feat in enumerate(TOP_FEATURES):
    col = col1 if i % 2 == 0 else col2
    default = float(feature_medians[feat])
    inputs[feat] = col.slider(feat, min_value=0.0, max_value=1.0, value=default, step=0.01)

if st.button("Predict", type="primary"):
    row = feature_medians.copy()
    for feat, val in inputs.items():
        row[feat] = val
    X = pd.DataFrame([row])[feature_names]
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ **Bankruptcy risk flagged** — predicted probability: {probability:.1%}")
    else:
        st.success(f"✅ **No bankruptcy risk flagged** — predicted probability: {probability:.1%}")

st.divider()
st.subheader("Model Performance")
st.markdown(
    """
Evaluated on a held-out test set (1,364 companies, 44 actually bankrupt — the real ~3% class
imbalance, untouched by resampling):

| Model | Accuracy | Bankrupt Precision | Bankrupt Recall | Bankrupt F1 |
|---|---|---|---|---|
| Always predict "not bankrupt" | 96.8% | — | 0% | — |
| Random Forest (deployed here) | 95.9% | 40.3% | **56.8%** | 47.2% |
| AdaBoost | 92.2% | 23.7% | 63.6% | 34.6% |

**Why lower accuracy than the naive baseline is the right trade:** always predicting "not
bankrupt" scores 96.8% accuracy on this dataset by doing nothing useful — it catches zero real
bankruptcies. Random Forest trades ~1 point of accuracy for catching **57% of actual
bankruptcies** instead of 0%. For a bankruptcy screener, recall on the minority class is the
number that matters, not overall accuracy.
"""
)
