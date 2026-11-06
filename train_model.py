"""Train and evaluate EduPulse's baseline early-warning classifier."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_DIR = ROOT / "models"
FEATURES = [
    "attendance_pct", "study_hours_per_day", "assignment_avg",
    "previous_exam_score", "missed_assignments", "sleep_hours",
    "support_sessions"
]


def main():
    if not DATA_PATH.exists():
        from generate_data import generate_dataset
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)
    if not set(FEATURES + ["at_risk"]).issubset(df.columns):
        raise ValueError("Dataset is missing one or more expected columns.")
    X, y = df[FEATURES], df["at_risk"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=250, max_depth=7, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "features": FEATURES,
        "model": "RandomForestClassifier",
        "data_note": "Synthetic demonstration data; labels come from a composite-risk rule."
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "edupulse_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})\
      .sort_values("importance", ascending=False)\
      .to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()
