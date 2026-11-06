"""Generate a reproducible synthetic student dataset for the EduPulse demo.

IMPORTANT: The dataset is simulated. It is not real student information and the
labels are generated from a transparent composite-risk rule for demonstration.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FEATURES = [
    "attendance_pct", "study_hours_per_day", "assignment_avg",
    "previous_exam_score", "missed_assignments", "sleep_hours",
    "support_sessions"
]


def generate_dataset(n: int = 600, seed: int = 42) -> pd.DataFrame:
    if n < 100:
        raise ValueError("Use at least 100 rows so the train/test split is meaningful.")
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(82, 11, n), 40, 100)
    study = np.clip(rng.normal(3.5, 1.5, n), 0, 9)
    assignment = np.clip(rng.normal(72, 15, n), 20, 100)
    previous = np.clip(rng.normal(68, 16, n), 15, 100)
    missed = np.clip(rng.poisson(1.4, n), 0, 8)
    sleep = np.clip(rng.normal(6.8, 1.0, n), 3, 10)
    support = np.clip(rng.poisson(1.0, n), 0, 5)

    # Transparent composite indicator: lower attendance/marks/study and more
    # missed work increase the simulated risk score. Support sessions provide
    # a small protective adjustment. No real student outcomes are represented.
    risk_score = (
        0.28 * (100 - attendance) / 100
        + 0.13 * (100 - assignment) / 100
        + 0.22 * (100 - previous) / 100
        + 0.17 * (9 - study) / 9
        + 0.13 * missed / 8
        + 0.04 * np.maximum(0, 7 - sleep) / 4
        - 0.03 * support / 5
        + rng.normal(0, 0.025, n)
    )
    cutoff = float(np.quantile(risk_score, 0.70))
    at_risk = (risk_score >= cutoff).astype(int)

    df = pd.DataFrame({
        "student_id": [f"STU{1001+i}" for i in range(n)],
        "attendance_pct": np.round(attendance, 1),
        "study_hours_per_day": np.round(study, 1),
        "assignment_avg": np.round(assignment, 1),
        "previous_exam_score": np.round(previous, 1),
        "missed_assignments": missed.astype(int),
        "sleep_hours": np.round(sleep, 1),
        "support_sessions": support.astype(int),
        "at_risk": at_risk,
        "risk_score_demo_only": np.round(risk_score, 4),
    })
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(ROOT / "data" / "student_performance.csv"))
    args = parser.parse_args()
    df = generate_dataset(args.samples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df)} synthetic student records to {output}")
    print(f"Synthetic at-risk share: {df['at_risk'].mean():.1%}")

if __name__ == "__main__":
    main()
