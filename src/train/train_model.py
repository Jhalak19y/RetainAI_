import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.data.load_data import load_data, ROOT
from src.features.preprocessing import create_preprocessor

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def pipeline(estimator):
    return Pipeline([("preprocessor", create_preprocessor()), ("classifier", estimator)])

def main():
    df = load_data()
    X, y = df.drop(columns="Churn"), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=neg / max(pos, 1), eval_metric="logloss",
            random_state=42, n_jobs=1
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy":"accuracy", "precision":"precision", "recall":"recall",
               "f1":"f1", "roc_auc":"roc_auc"}
    comparison = {}
    for name, estimator in models.items():
        scores = cross_validate(pipeline(estimator), X_train, y_train,
                                cv=cv, scoring=scoring, n_jobs=1)
        comparison[name] = {
            metric: float(np.mean(scores[f"test_{metric}"]))
            for metric in scoring
        }
        comparison[name]["roc_auc_std"] = float(np.std(scores["test_roc_auc"]))
    selected_name = max(comparison, key=lambda k: comparison[k]["roc_auc"])
    selected_estimator = models[selected_name]
    if selected_name == "Logistic Regression":
        grid = {"classifier__C":[0.5, 1.0, 2.0]}
    elif selected_name == "Random Forest":
        grid = {"classifier__max_depth":[None, 12],
                "classifier__min_samples_leaf":[1, 3]}
    else:
        grid = {"classifier__max_depth":[3, 5],
                "classifier__learning_rate":[0.05, 0.1]}
    search = GridSearchCV(
        pipeline(selected_estimator), grid, scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=1, refit=True
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # Choose threshold using out-of-fold training predictions only.
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(
        model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    threshold_rows = []
    for t in thresholds:
        pred = (oof >= t).astype(int)
        threshold_rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y_train, pred, zero_division=0)),
            "recall": float(recall_score(y_train, pred, zero_division=0)),
            "f1": float(f1_score(y_train, pred, zero_division=0)),
        })
    best = max(threshold_rows, key=lambda r: (r["f1"], -abs(r["threshold"]-0.5)))
    threshold = best["threshold"]
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
    }
    report = {
        "selected_model": selected_name,
        "best_parameters": search.best_params_,
        "decision_threshold": threshold,
        "threshold_selection": best,
        "test_metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "model_comparison": comparison,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "churn_rate_test": float(y_test.mean()),
        "class_weighting": "balanced class weights / scale_pos_weight",
        "threshold_note": "Threshold maximizes out-of-fold training F1; tune for business costs later.",
    }
    joblib.dump(model, MODEL_DIR / "churn_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([{"model":k, **v} for k,v in comparison.items()]).to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )
    pd.DataFrame({"actual": y_test, "probability": probabilities,
                  "prediction": predictions}).to_csv(
        REPORT_DIR / "test_predictions.csv", index=False
    )
    print(json.dumps(report, indent=2))
    print(f"\nSaved model to {MODEL_DIR / 'churn_model.joblib'}")

if __name__ == "__main__":
    main()
