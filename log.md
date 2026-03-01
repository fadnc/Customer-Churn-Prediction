Production-Ready Customer Churn Prediction Pipeline with Monitoring & Auto-Retraining

Predict whether a telecom customer will churn, and deploy a production system that:
  Trains models
  Tracks experiments
  Serves predictions via API
  Monitors drift
  Triggers retraining when performance degrades

Data → Validation → Feature Engineering → Training → MLflow → Model Registry → FastAPI → Monitoring → Drift Detection → Auto Retraining

Phase 1 — Strong Classical ML Core
Phase 2 — Experiment Tracking
Phase 3 — Model Serving
Phase 4 — Data Drift Detection
Phase 5 — Automated Retraining

Built a production-grade churn prediction pipeline with end-to-end ML lifecycle management, achieving 0.89 ROC-AUC using XGBoost with engineered behavioral features and stratified CV.
Implemented MLflow-based experiment tracking and model registry, automated drift detection using KS-test & PSI, and deployed a containerized FastAPI inference service with retraining trigger logic.
