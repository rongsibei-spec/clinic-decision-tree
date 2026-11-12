"""Reproduce the clinic no-show assignment using a single shallow tree."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "lead_time_days", "prior_no_shows", "age", "distance_miles",
    "sms_reminder", "new_patient", "appointment_hour",
]
SEED = 42
TEST_SHARE = 0.30
MAX_DEPTH = 3
MIN_LEAF = 10
OVERBOOK_SHOW_COST = 60
HOLD_NO_SHOW_COST = 120
CASES = {
    "D": dict(zip(FEATURES, [21, 1, 27, 9.5, 0, 1, 15])),
    "E": dict(zip(FEATURES, [3, 0, 61, 2.0, 1, 0, 9])),
}


def decide(model, appointment):
    """Return the action, reached leaf, estimated probability, and path."""
    row = pd.DataFrame([appointment], columns=FEATURES)
    leaf = int(model.apply(row)[0])
    action = "overbook" if model.predict(row)[0] == 1 else "hold"
    path = []
    for node in model.decision_path(row).indices:
        if node == leaf:
            continue
        feature = FEATURES[model.tree_.feature[node]]
        threshold = float(model.tree_.threshold[node])
        relation = "<=" if row.at[0, feature] <= threshold else ">"
        path.append(f"{feature} {relation} {threshold:.6g}")
    return {
        "action": action, "leaf": leaf, "path": path,
        "no_show_probability": float(model.predict_proba(row)[0, 1]),
        "leaf_training_count": int(model.tree_.n_node_samples[leaf]),
    }


def evaluate(actual, predicted):
    """No-show is the positive class. Values are negative costs."""
    tn, fp, fn, tp = map(int, confusion_matrix(actual, predicted, labels=[0, 1]).ravel())
    value = -OVERBOOK_SHOW_COST * fp - HOLD_NO_SHOW_COST * fn
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy_score(actual, predicted),
        "no_show_precision": precision_score(actual, predicted, zero_division=0),
        "no_show_recall": recall_score(actual, predicted, zero_division=0),
        "total_value": value, "value_per_appointment": value / len(actual),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "clinic_appointments.csv")
    parser.add_argument("--annual-volume", type=int, default=4000,
                        help="Illustrative annual appointments, not observed in the data.")
    args = parser.parse_args()
    appointments = pd.read_csv(args.data)
    y = appointments["no_show"].map({"checked-in": 0, "no-show": 1})
    x = appointments[FEATURES]
    if y.isna().any() or x.isna().any().any():
        raise ValueError("Unexpected outcome or missing feature.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SHARE, random_state=SEED, stratify=y
    )
    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=SEED
    ).fit(x_train, y_train)
    baseline = evaluate(y_test, [0] * len(y_test))
    v1 = evaluate(y_test, model.predict(x_test))
    improvement = (v1["total_value"] - baseline["total_value"]) / len(y_test)
    summary = {
        "data": {"rows": len(x), "columns": len(appointments.columns),
                 "no_shows": int(y.sum()), "train": len(x_train), "test": len(x_test),
                 "train_no_shows": int(y_train.sum()), "test_no_shows": int(y_test.sum()),
                 "seed": SEED},
        "model": {"features": FEATURES, "depth": model.get_depth(), "min_leaf": MIN_LEAF,
                  "training_accuracy": model.score(x_train, y_train)},
        "baseline": baseline, "v1": v1,
        "examples": {name: decide(model, values) for name, values in CASES.items()},
        "improvement_total": v1["total_value"] - baseline["total_value"],
        "improvement_per_appointment": improvement,
        "assumed_annual_volume": args.annual_volume,
        "projected_annual_improvement": improvement * args.annual_volume,
    }
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    tree_rules = export_text(model, feature_names=FEATURES)
    (output / "tree_rules.txt").write_text(tree_rules, encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    figure, ax = plt.subplots(figsize=(15, 7))
    plot_tree(model, feature_names=FEATURES, class_names=["Show", "No-show"],
              filled=True, rounded=True, node_ids=True, precision=2, fontsize=10, ax=ax)
    figure.savefig(output / "tree.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print("\nFitted tree\n" + tree_rules)


if __name__ == "__main__":
    main()
