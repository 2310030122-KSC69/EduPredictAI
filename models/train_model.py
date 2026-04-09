"""
train_model.py - Train and save student performance prediction models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
import joblib
import json
import os

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_synthetic_data(n_samples=2000, random_state=42):
    """Generate realistic synthetic student performance data."""
    np.random.seed(random_state)

    data = {
        "study_hours_per_day": np.clip(np.random.normal(4, 2, n_samples), 0, 12),
        "attendance_rate": np.clip(np.random.normal(75, 15, n_samples), 0, 100),
        "previous_gpa": np.clip(np.random.normal(2.8, 0.8, n_samples), 0, 4.0),
        "assignments_completed": np.clip(np.random.normal(75, 20, n_samples), 0, 100),
        "sleep_hours": np.clip(np.random.normal(7, 1.5, n_samples), 3, 12),
        "extracurricular_activities": np.random.randint(0, 5, n_samples),
        "parental_education": np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.4, 0.2]),
        "internet_access": np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        "tutoring_sessions": np.clip(np.random.poisson(2, n_samples), 0, 10),
        "stress_level": np.random.randint(1, 11, n_samples),
        "motivation_score": np.random.randint(1, 11, n_samples),
        "part_time_job": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }

    df = pd.DataFrame(data)

    # Create a composite performance score
    score = (
        df["study_hours_per_day"] * 4.0 +
        df["attendance_rate"] * 0.3 +
        df["previous_gpa"] * 15 +
        df["assignments_completed"] * 0.2 +
        (df["sleep_hours"] - 3) * 2.0 +
        df["motivation_score"] * 2.5 +
        df["tutoring_sessions"] * 1.5 +
        df["parental_education"] * 3.0 +
        df["internet_access"] * 5.0 -
        df["stress_level"] * 1.5 -
        df["part_time_job"] * 8.0 +
        np.random.normal(0, 10, n_samples)
    )

    # Classify into performance categories
    percentiles = np.percentile(score, [30, 60, 80])
    df["performance"] = pd.cut(
        score,
        bins=[-np.inf, percentiles[0], percentiles[1], percentiles[2], np.inf],
        labels=["Poor", "Average", "Good", "Excellent"]
    )

    return df


def train_and_save_models():
    print("Generating training data...")
    df = generate_synthetic_data(2000)

    feature_cols = [
        "study_hours_per_day", "attendance_rate", "previous_gpa",
        "assignments_completed", "sleep_hours", "extracurricular_activities",
        "parental_education", "internet_access", "tutoring_sessions",
        "stress_level", "motivation_score", "part_time_job"
    ]

    X = df[feature_cols]
    y = df["performance"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    best_model_name = None
    best_accuracy = 0

    for name, model in models.items():
        print(f"Training {name}...")
        if name in ["logistic_regression", "svm"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "accuracy": round(float(acc), 4),
            "report": classification_report(
                y_test, y_pred,
                target_names=le.classes_, output_dict=True
            )
        }
        print(f"  Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    # Feature importance from best tree-based model
    rf_model = models["random_forest"]
    feature_importance = dict(zip(feature_cols, rf_model.feature_importances_.tolist()))

    # Save artifacts
    joblib.dump(models["random_forest"], os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(models["gradient_boosting"], os.path.join(MODELS_DIR, "gb_model.pkl"))
    joblib.dump(models["logistic_regression"], os.path.join(MODELS_DIR, "lr_model.pkl"))
    joblib.dump(models["svm"], os.path.join(MODELS_DIR, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    metadata = {
        "feature_cols": feature_cols,
        "classes": le.classes_.tolist(),
        "model_results": results,
        "best_model": best_model_name,
        "feature_importance": feature_importance,
        "training_samples": 1600,
        "test_samples": 400,
    }

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll models saved! Best: {best_model_name} ({best_accuracy:.4f})")
    return metadata


if __name__ == "__main__":
    train_and_save_models()
