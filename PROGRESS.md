# BankruptWatch — Progress

## Status
Deployed, stable. Live at https://bankruptwtch.streamlit.app (Streamlit
Community Cloud free tier — sleeps after inactivity, wakes on first visit).

## Done
- Random Forest vs. AdaBoost bankruptcy classifiers on the TEJ dataset
  (6,819 Taiwan-listed companies, 96 financial ratios, ~3% positive class).
- Pipeline rebuilt correctly: split before resampling, SMOTE applied to
  the training fold only — original version computed SMOTE but never
  actually used it in training.
- Reports precision/recall/F1 on the bankrupt class, not just accuracy —
  original version reported only accuracy on a 97/3 imbalanced test set,
  which hid that the model may have caught zero real bankruptcies.
- Real, honest result: Random Forest catches 57% of actual bankruptcies
  (vs. 0% for a naive baseline), documented alongside its 40.3% precision
  (roughly 3 in 5 flags are false alarms) rather than hiding the trade-off.
- Fixed: notebook Kaggle-only path dependency, missing `pandas` import,
  a `matplotlib` API rename that broke an EDA plot.
- Interactive Streamlit demo with slider inputs for the 10 most
  influential ratios.
- In-app disclaimer that this is not a financial advisory tool.

## In progress
- Nothing currently active.

## Known issues / honest limitations
- **The live demo cannot show a bankruptcy flag (found 2026-11-05, in `app.py` since it was added).** The app holds 85 of the 95 ratios at their median, and with those fixed the model stayed under 0.5. Run locally on `model/`: 20,000 random slider settings peaked at 0.415, none flagged; the 220 real bankrupt companies with their own 10 slider values peaked under 0.5 (median 0.31), none flagged; all sliders at their worst values gave 0.40. The same model on all 95 real ratios does flag (test recall 56.8%, precision 40.3%, both reproduced), so the reported metrics hold, so the app and README now say this plainly (reworded 2026-11-05); the demo itself is unchanged. Random search, not a proof that no setting flags.
- Small positive class in the test set (44 bankrupt companies) — real
  uncertainty in the precision/recall estimates at this sample size.
- Static financial ratios only, no time-series trend data — can't see a
  company's trajectory, only a snapshot.
- 40.3% precision means most flags are false alarms — appropriate only
  as a low-cost screening signal, not for automated decisions.
- Free-tier Streamlit hosting sleeps after inactivity. Cold start is
  slower than previously documented — a live re-check (2026-07-29) was
  still mid-boot past 60-85s, not the ~20-30s this used to say.

## Verification log
- 2026-07-23: git working tree clean, no pending diff. `/security-review`
  skill checked — N/A, diff-based and nothing to review. Live demo woken
  from sleep; slider-widget JS and all core assets loaded 200, no
  console errors.
- 2026-08-11: reviewed commits since last check (README screenshot,
  `.devcontainer/devcontainer.json` added) — dev tooling only, no app
  code changed, status unaffected.
- 2026-11-03 (metrics re-reproduced from code on 2026-11-03): ran
  `train.py`'s exact pipeline as a scratch script (`train.py` itself not
  executed because it overwrites `model/`), repo `.venv` = Python 3.14.7,
  scikit-learn 1.9.0, imbalanced-learn 0.14.2. Data:
  `dataset/BankruptData.csv`, 6,819 rows. Split: stratified
  `train_test_split(test_size=0.2, random_state=42)`, giving 5,455 train
  and 1,364 test rows (44 bankrupt); `StandardScaler` and
  `SMOTE(random_state=42)` fit on the train fold only. **Result matches
  README/app/`results.json` exactly:** baseline 0.9677; Random Forest
  acc 0.9589, bankrupt precision 0.4032, recall 0.5682, F1 0.4717,
  CM [[1283, 37], [19, 25]]; AdaBoost 0.9223 / 0.2373 / 0.6364 / 0.3457,
  CM [[1230, 90], [16, 28]]. The committed `model.pkl` + `scaler.pkl`
  on the same test split give the identical RF numbers. Re-running the
  notebook's `GridSearchCV(cv=5)` picks the same hyperparameters
  (RF `n_estimators=200, max_depth=None`; AdaBoost `n_estimators=200,
  learning_rate=1.0`).
  - **Leak check.** Scaler and SMOTE are fit after the split, train only:
    clean. There are 0 duplicate rows and 0 test feature vectors that
    also appear in train. There's no date column, so the time-ordered
    split pattern can't be checked. Two mild issues remain. (1)
    `train.py` picks the deployed model (RF vs AdaBoost) by **test-set**
    F1 (`best_name`), so the test set does double duty for a
    2-candidate choice. (2) The notebook runs SMOTE *before*
    `GridSearchCV`, so the CV folds contain synthetic points derived from
    their own validation rows. That inflates the CV scores (RF 0.9807,
    AdaBoost 0.9432) and affects only which hyperparameters get picked,
    not the reported test numbers. The notebook's legacy cells 30/35/38
    still fit SMOTE, SelectKBest and the scaler on the full data before
    the split, but nothing downstream uses them for the reported metrics.
  - Nothing claimed was left unverified.

## Next up
- `requirements.txt` pins are live (`91e6de2`, Cloud runs Python 3.14). Checked live 2026-11-05: page renders, no exception, Predict works, and a slider drag reaches the model (Borrowing dependency = 1.00 returns 6.5%, same as the local model).
- Optional: make the demo able to flag (for example, load real example companies). Not scheduled.
- Nothing scheduled.
