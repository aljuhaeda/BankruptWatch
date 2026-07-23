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
  (vs. 0% for a naive baseline), documented alongside its 34.8% precision
  (roughly 2 in 3 flags are false alarms) rather than hiding the trade-off.
- Fixed: notebook Kaggle-only path dependency, missing `pandas` import,
  a `matplotlib` API rename that broke an EDA plot.
- Interactive Streamlit demo with slider inputs for the 10 most
  influential ratios.
- In-app disclaimer that this is not a financial advisory tool.

## In progress
- Nothing currently active.

## Known issues / honest limitations
- Small positive class in the test set (44 bankrupt companies) — real
  uncertainty in the precision/recall estimates at this sample size.
- Static financial ratios only, no time-series trend data — can't see a
  company's trajectory, only a snapshot.
- 34.8% precision means most flags are false alarms — appropriate only
  as a low-cost screening signal, not for automated decisions.
- Free-tier Streamlit hosting sleeps after inactivity (cold start ~20-30s).

## Verification log
- 2026-07-23: git working tree clean, no pending diff. `/security-review`
  skill checked — N/A, diff-based and nothing to review. Live demo woken
  from sleep; slider-widget JS and all core assets loaded 200, no
  console errors.

## Next up
- Nothing scheduled.
