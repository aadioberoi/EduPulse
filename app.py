"""EduPulse: Student Performance Insights & Early-Warning System (Streamlit demo)."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_PATH = ROOT / "models" / "edupulse_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
FEATURES = [
    "attendance_pct", "study_hours_per_day", "assignment_avg",
    "previous_exam_score", "missed_assignments", "sleep_hours",
    "support_sessions"
]

st.set_page_config(page_title="EduPulse | Student Insights", page_icon="📚", layout="wide")

st.markdown("""
<style>
:root { --ink:#17243A; --muted:#66758A; --teal:#0D9488; --pale:#F1F6F8; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px; }
[data-testid="stMetric"] { background:#F1F6F8; border:1px solid #DCE7EA; padding:14px 16px; border-radius:14px; }
[data-testid="stMetricLabel"] { color:#66758A; }
div[data-testid="stDataFrame"] { border:1px solid #DCE7EA; border-radius:12px; }
.small-note { color:#66758A; font-size:0.88rem; }
.hero { background:linear-gradient(110deg,#102A43,#0F766E); color:white; padding:24px 28px; border-radius:18px; margin-bottom:18px; }
.hero h1 { color:white; margin:0; font-size:2.1rem; }
.hero p { color:#D6F4EF; margin:6px 0 0 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Run python train_model.py first.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error("Dataset not found. Run python generate_data.py first.")
        st.stop()
    return pd.read_csv(DATA_PATH)

if not METRICS_PATH.exists():
    st.error("Evaluation metrics not found. Run python train_model.py first.")
    st.stop()

students = load_data()
model = load_model()
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

st.markdown("<div class='hero'><h1>EduPulse</h1><p>Student Performance Insights & Early-Warning System</p></div>", unsafe_allow_html=True)
st.caption("A decision-support prototype that highlights students who may benefit from timely academic support.")

with st.sidebar:
    st.header("Explore EduPulse")
    page = st.radio("Workspace", ["Overview", "Student risk check", "Model evaluation", "About the prototype"])
    st.divider()
    st.caption("Demo dataset: simulated student records. Do not use for real academic decisions.")

if page == "Overview":
    predictions = model.predict(students[FEATURES])
    probabilities = model.predict_proba(students[FEATURES])[:, 1]
    view = students.copy()
    view["predicted_risk"] = predictions
    view["risk_probability"] = probabilities
    total = len(view)
    at_risk = int((view["predicted_risk"] == 1).sum())
    avg_att = view["attendance_pct"].mean()
    avg_score = view["assignment_avg"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in demo", f"{total:,}")
    c2.metric("Flagged for support", f"{at_risk:,}", f"{at_risk/total:.1%} of sample")
    c3.metric("Average attendance", f"{avg_att:.1f}%")
    c4.metric("Average assignment score", f"{avg_score:.1f}/100")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Predicted support priority")
        counts = view["predicted_risk"].map({0:"Lower priority", 1:"Needs attention"}).value_counts()
        st.bar_chart(counts, color="#0D9488")
        st.markdown("<p class='small-note'>Priority is a model estimate, not a diagnosis or final academic judgment.</p>", unsafe_allow_html=True)
    with right:
        st.subheader("Average profile by group")
        summary = view.groupby("predicted_risk")[["attendance_pct", "assignment_avg", "previous_exam_score"]].mean().rename(index={0:"Lower priority", 1:"Needs attention"})
        st.dataframe(summary.round(1), use_container_width=True)

    st.subheader("Students to review first")
    flagged = view[view["predicted_risk"] == 1].sort_values("risk_probability", ascending=False).head(12)
    display = flagged[["student_id", "attendance_pct", "assignment_avg", "previous_exam_score", "missed_assignments", "risk_probability"]].copy()
    display["risk_probability"] = (display["risk_probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Suggested action: a faculty member reviews the context and speaks with the student before offering support.")

elif page == "Student risk check":
    st.subheader("Check an individual student profile")
    st.write("Enter a small set of academic indicators to see the model's estimated support priority.")
    with st.form("student_form"):
        a, b, c = st.columns(3)
        attendance = a.slider("Attendance (%)", 40, 100, 78)
        study = b.slider("Study hours per day", 0.0, 9.0, 3.5, 0.5)
        assignment = c.slider("Assignment average", 20, 100, 70)
        d, e, f, g = st.columns(4)
        previous = d.slider("Previous exam score", 15, 100, 65)
        missed = e.slider("Missed assignments", 0, 8, 1)
        sleep = f.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5)
        support = g.slider("Support sessions", 0, 5, 1)
        submitted = st.form_submit_button("Assess support priority", type="primary")
    if submitted:
        sample = pd.DataFrame([{
            "attendance_pct": attendance, "study_hours_per_day": study,
            "assignment_avg": assignment, "previous_exam_score": previous,
            "missed_assignments": missed, "sleep_hours": sleep,
            "support_sessions": support
        }], columns=FEATURES)
        probability = float(model.predict_proba(sample)[0, 1])
        prediction = int(model.predict(sample)[0])
        if prediction:
            st.warning(f"Needs attention — estimated model probability: {probability:.1%}")
            st.write("Suggested next step: check for barriers, clarify missed work, and offer targeted academic support.")
        else:
            st.success(f"Lower priority at this time — estimated model probability: {probability:.1%}")
            st.write("Continue routine monitoring and encourage the student to maintain helpful study habits.")
        st.caption("This is an educational prototype trained on simulated records. It must not be used to label or penalize real students.")

elif page == "Model evaluation":
    st.subheader("Baseline model evaluation")
    st.caption("Hold-out test set. Metrics below are generated by the local training script on the included synthetic dataset.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{metrics['precision']:.1%}")
    m3.metric("Recall", f"{metrics['recall']:.1%}")
    m4.metric("F1-score", f"{metrics['f1_score']:.1%}")
    cm = pd.DataFrame(metrics["confusion_matrix"], index=["Actual lower", "Actual at-risk"], columns=["Predicted lower", "Predicted at-risk"])
    st.write("**Confusion matrix**")
    st.dataframe(cm, use_container_width=True)
    importance_path = ROOT / "models" / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).set_index("feature")
        st.write("**Feature importance (baseline model)**")
        st.bar_chart(importance, color="#0D9488")
    st.caption(f"Training rows: {metrics['train_rows']} | Test rows: {metrics['test_rows']} | Model: {metrics['model']}")

else:
    st.subheader("About EduPulse")
    st.write("EduPulse is a small academic decision-support prototype. It organizes student performance indicators, estimates a support-priority score using a baseline machine-learning model, and presents the result in an accessible dashboard.")
    st.markdown("**Prototype workflow**")
    st.write("1. Load synthetic academic records.  2. Prepare selected indicators.  3. Train a Random Forest classifier.  4. Estimate support priority.  5. Review results in the dashboard.")
    st.markdown("**Important limitations**")
    st.write("- The included dataset is synthetic and does not represent real students.\n- The target labels are generated from a transparent composite-risk rule.\n- Metrics show whether the model reproduces this simulated pattern; they do not prove real-world predictive validity.\n- A real deployment would require consent, institutional data governance, fairness checks, human review, and validation on representative data.")
    st.markdown("**Planned improvements**")
    st.write("Connect an approved institutional dataset, include time-based progress trends, validate the model with educators, evaluate fairness across student groups, and add explainable alerts with human-in-the-loop review.")

st.divider()
st.caption("EduPulse | B.Sc. Data Science project prototype | Built for academic demonstration")
