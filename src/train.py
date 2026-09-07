import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from src.data_generator import generate_synthetic_data

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False


FEATURE_COLUMNS = ["calls_per_hour", "avg_call_duration", "contact_degree", "spam_report_count"]
TARGET_COLUMN = "is_spam"

def train_and_evaluate(data_path: str = "data/synthetic_calls.csv", model_output_path: str = "models/signaltrust_xgboost.pkl"):
    """
    Loads synthetic call dataset, trains an XGBoost (or RandomForest) classifier,
    evaluates performance metrics, and exports the model pipeline.
    """
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"[+] Loaded existing dataset from '{data_path}'")
    else:
        print("[!] Dataset file not found. Generating synthetic dataset on the fly...")
        df = generate_synthetic_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if HAS_XGBOOST:
        print("[+] Training XGBoost Classifier...")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    else:
        print("[!] XGBoost unavailable, falling back to RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 5-Fold Stratified Cross-Validation (reproducibility check)
    # Run on the training split only — zero leakage from X_test.
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
    print("\n" + "=" * 50)
    print("5-FOLD STRATIFIED CV (TRAINING SPLIT ONLY)")
    print("=" * 50)
    print(f"Fold Scores   : {[float(round(s, 4)) for s in cv_scores]}")
    print(f"Mean ROC-AUC  : {cv_scores.mean():.4f}")
    print(f"Std Dev       : {cv_scores.std():.4f}  (+/- {cv_scores.std():.4f})")
    print("=" * 50)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Custom thresholding prioritizing High Precision (minimizing false positives)
    threshold = 0.70
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"Precision     : {precision:.4f} (at threshold {threshold})")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print("=" * 50)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"[+] Model artifact saved successfully to '{model_output_path}'")

    return {
        "model": model,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_roc_auc_mean": cv_scores.mean(),
        "cv_roc_auc_std": cv_scores.std(),
        "model_path": model_output_path
    }

if __name__ == "__main__":
    train_and_evaluate()
