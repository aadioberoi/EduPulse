# EduPulse: Student Performance Insights & Early-Warning System

**Project type:** B.Sc. Data Science prototype  
**Prepared by:** Aadi Oberoi  
**Institution:** AAFT, Noida  
**Project status:** Working prototype using synthetic demonstration data

## Abstract
EduPulse is a lightweight student-performance monitoring prototype. It combines a small set of academic and study-related indicators in a dashboard and uses a baseline Random Forest classifier to estimate whether a simulated profile belongs to a higher support-priority group. The project demonstrates a practical data-science workflow: synthetic data generation, preprocessing, model training, hold-out evaluation, and dashboard-based interaction. The dataset and target labels are simulated; the results are not validated predictions about real students. EduPulse is therefore presented as a proof of concept for responsible early-support decision support, not as an operational student assessment tool.

## 1. Introduction
Educational institutions collect different signals about student progress, including attendance, assignment completion and assessment performance. These indicators can be difficult to review together when they are stored in separate records. A simple dashboard can help present them in one place and draw attention to profiles that may warrant a supportive conversation. EduPulse explores this idea through a compact machine-learning prototype.

## 2. Problem statement
A teacher or academic coordinator may have to review multiple performance indicators before identifying students who could benefit from timely support. The project demonstrates how a unified view and a basic classification model can summarize a group of indicators. The system is not intended to replace teacher judgment or explain the causes of academic difficulty.

## 3. Aim and objectives
**Aim:** Build a working prototype that converts student-performance indicators into an accessible overview and a model-estimated support-priority flag.

**Objectives:**
1. Create a reproducible synthetic dataset for a safe demonstration.
2. Select a concise set of academic and study-related input features.
3. Train a baseline classification model using a reproducible train/test split.
4. Evaluate the model with accuracy, precision, recall, F1-score and a confusion matrix.
5. Build a Streamlit interface for overview, individual profile checks and evaluation.
6. Document limitations and requirements for responsible real-world validation.

## 4. Scope
The prototype uses seven numeric features: attendance percentage, study hours per day, assignment average, previous exam score, missed assignments, sleep hours and support sessions. The output is a binary support-priority estimate. The included dataset contains 600 synthetic profiles. The target label is generated using a composite-risk score and a 70th-percentile threshold. This setup is appropriate for a controlled demonstration but does not provide evidence of real-world predictive performance.

## 5. System architecture
**Data generation → feature selection → train/test split → Random Forest classifier → evaluation → saved model → Streamlit dashboard.**

The dashboard offers an overview of the sample, a table of profiles flagged by the model, an individual profile form, a model-evaluation page and a page describing the prototype's limitations. Model training and dashboard rendering are kept in separate Python files so the workflow can be rerun.

## 6. Tools and technologies
- **Python:** main programming language.
- **Pandas and NumPy:** data generation and tabular processing.
- **scikit-learn:** train/test split, Random Forest and evaluation metrics.
- **Joblib:** saving and loading the trained model.
- **Streamlit:** interactive dashboard.

## 7. Methodology
The data generator uses a fixed random seed and creates plausible ranges for the seven selected indicators. A weighted composite formula assigns higher risk scores to lower attendance, weaker assessment results, less study time and more missed assignments. Support sessions slightly reduce the simulated risk score. The top 30% of the score distribution are assigned the positive demo label. This is a transparent rule for creating training labels, not an observed student outcome.

The dataset is divided into 80% training and 20% test data with stratification. A Random Forest classifier with 250 trees is trained on the training portion. The untouched test portion is used to calculate accuracy, precision, recall, F1-score and a confusion matrix. Feature importance is saved as a diagnostic summary, not as proof of causal influence.

## 8. Dashboard modules
1. **Overview:** sample size, predicted support-priority distribution, average attendance and assignment score, and a short list of profiles flagged for review.
2. **Student risk check:** accepts a single profile and returns a model-estimated probability and a supportive suggested next step.
3. **Model evaluation:** displays hold-out metrics, a confusion matrix and feature importance.
4. **About the prototype:** explains the workflow, limitations and future improvements.

## 9. Evaluation
The training script writes the actual hold-out metrics to \`models/metrics.json\`. The app reads those values directly, so no performance numbers are manually invented. Since the test labels are generated by a rule, metrics only measure agreement with that simulated rule. A meaningful real-world evaluation would require a representative, approved dataset with a defined future outcome and time horizon.

## 10. Ethical considerations and limitations
The demo contains no real personal information. Any real implementation would require institutional approval, appropriate consent and notice, data minimization, secure access, bias and subgroup evaluation, monitoring for drift, and a human review process. Model output should never be the sole basis for academic, disciplinary or financial decisions. Students should have a way to challenge incorrect records and receive support without stigma.

## 11. Future scope
Future work may include longitudinal trends, approved real-world data, calibration, subgroup fairness evaluation, false-positive and false-negative analysis, explanation summaries, and a teacher-reviewed referral workflow. Audio, text feedback or other modalities are outside the current scope and should only be added when there is a justified use case and suitable governance.

## 12. Conclusion
EduPulse demonstrates a complete, small-scale data-science prototype that connects synthetic data generation, baseline classification, model evaluation and an interactive dashboard. Its main contribution is a clear, reproducible workflow and an example of how analytics can support timely human attention. The next step is not simply to add a more complex model, but to validate the concept with suitable, approved data and careful human oversight.

## References
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5-32.
- Streamlit documentation: https://docs.streamlit.io/
- scikit-learn documentation: https://scikit-learn.org/stable/
