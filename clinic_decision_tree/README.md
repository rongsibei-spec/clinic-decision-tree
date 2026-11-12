# Clinic no-show decision tree

Place the course-provided `clinic_appointments.csv` beside `analyze.py`. The source CSV is supplied through the course and is not included in this package. Use Python 3.11 or newer.

```sh
python -m pip install -r requirements.txt
python analyze.py
```

The script writes `results/summary.json`, `results/tree_rules.txt`, and `results/tree.png`. To change the illustrative annual appointment volume, run `python analyze.py --annual-volume 4000` with your chosen volume. The data do not report annual volume.

## Method

- The supplied data contain 400 historical appointments. The stratified 70/30 split uses seed 42. Fit one classification tree on the 280 training rows and score the baseline and tree on the same 120 held-out rows.
- Offer all seven appointment-time predictors to the tree. Cap depth at three and require at least ten training appointments per leaf to keep the rules readable and reduce sensitivity to very small groups. Only prior no-shows, SMS reminder status, and age appear in the fitted splits.
- No-show is the positive class. The baseline holds every slot and makes no positive predictions, so its precision has a zero denominator. The script reports zero by convention.
- Overbook when the fitted tree predicts no-show. Decision function `decide` returns the action, leaf ID, path, and fraction of training appointments in the leaf that were no-shows.
- Score each false positive as a $60 cost and each false negative as a $120 cost. Correct predictions have zero incremental cost. Annual savings are an illustrative extrapolation from held-out savings per appointment. Reminder status is a predictor and this analysis does not estimate the causal effect of sending a reminder.

The included `analysis_results.txt` records the output for the original data and default annual volume.
