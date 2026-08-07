"""
Stage 3: evaluate
Scores the model on the held-out test set,
This exit code is what stops a bad model from ever reaching the register step.
"""
import sys
import json
import yaml
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    clf = joblib.load("model/model.joblib")
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    preds = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    if metrics["accuracy"] < params["min_accuracy"]:
        print(
            f"FAIL: accuracy {metrics['accuracy']:.4f} "
            f"is below gate {params['min_accuracy']}"
        )
        sys.exit(1)

    print("PASS: model cleared the quality gate")


if __name__ == "__main__":
    main()
