"""
Trains and evaluates the BankruptWatch models properly.

The original notebook had two bugs this script fixes:
1. Data path was hardcoded to a Kaggle-only path (../input/...), so the
   notebook couldn't run outside the original Kaggle session.
2. SMOTE was applied in an earlier exploratory section but never actually
   fed into the models that got trained and evaluated -- the "Machine
   Learning" section reloaded the raw, untouched, ~97/3 imbalanced data.
   That produced misleadingly high accuracy (~96%) that's *worse* than
   the trivial "always predict not bankrupt" baseline (~96.8%), with no
   precision/recall/F1 computed to reveal it.

This script splits first, applies SMOTE only to the training fold (to
avoid leaking synthetic minority samples derived from test data), and
reports precision/recall/F1/confusion matrix -- the numbers that actually
matter for a bankruptcy detector -- not just accuracy.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "dataset", "BankruptData.csv")
MODEL_DIR = os.path.join(HERE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip()

X = data.drop(columns=["Bankrupt?"])
y = data["Bankrupt?"]

# Split BEFORE any resampling/scaling so the test set stays a clean,
# untouched sample of the real-world class distribution.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE fit only on the training fold.
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)


# Hyperparameters below were chosen by GridSearchCV in notebooks/BankruptWatch.ipynb
# (5-fold CV over n_estimators/max_depth for RandomForest and
# n_estimators/learning_rate for AdaBoost) -- hardcoded here so this script
# reproduces the notebook's exact result without re-running the full grid
# search every time.
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=42),
}

results = {}
trained_models = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    preds = model.predict(X_test_scaled)
    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_bankrupt": precision_score(y_test, preds, pos_label=1),
        "recall_bankrupt": recall_score(y_test, preds, pos_label=1),
        "f1_bankrupt": f1_score(y_test, preds, pos_label=1),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    trained_models[name] = model
    print(f"\n=== {name} ===")
    print(f"Accuracy: {results[name]['accuracy']:.4f}  (baseline: {baseline_accuracy:.4f})")
    print(
        f"Bankrupt class -> precision: {results[name]['precision_bankrupt']:.4f}, "
        f"recall: {results[name]['recall_bankrupt']:.4f}, "
        f"f1: {results[name]['f1_bankrupt']:.4f}"
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array(results[name]["confusion_matrix"]))

best_name = max(results, key=lambda n: results[n]["f1_bankrupt"])
best_model = trained_models[best_name]
print(f"\nBest model by bankrupt-class F1: {best_name}")

# Feature importance (RandomForest only has one natively; use it for both
# the README write-up and to pick which features the demo form asks for).
importances = pd.Series(
    trained_models["RandomForest"].feature_importances_, index=X.columns
).sort_values(ascending=False)
top_features = importances.head(10)
print("\nTop 10 features by Random Forest importance:")
print(top_features)

joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))
joblib.dump(X_train.median(), os.path.join(MODEL_DIR, "feature_medians.pkl"))

summary = {
    "baseline_accuracy": baseline_accuracy,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_bankrupt_test": int(y_test.sum()),
    "best_model": best_name,
    "results": {
        k: {kk: vv for kk, vv in v.items() if kk != "classification_report"}
        for k, v in results.items()
    },
    "top_features": top_features.to_dict(),
}
with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved model + scaler + results to {MODEL_DIR}")
