from pathlib import Path
import numpy as np
import pandas as pd
import shap

FEATURE_LABELS = {
    "Contract": "Contract type",
    "tenure": "Customer tenure (months)",
    "TechSupport": "Technical support",
    "OnlineSecurity": "Online security",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "PaymentMethod": "Payment method",
    "InternetService": "Internet service",
}

class ChurnExplainer:
    def __init__(self, pipeline, background: pd.DataFrame):
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self.background_raw = background.copy()
        self.background = self._transform(background)
        self.feature_names = self.preprocessor.get_feature_names_out().tolist()
        self.masker = shap.maskers.Independent(
            self.background, max_samples=min(50, len(self.background))
        )
        self.explainer = shap.Explainer(
            self._predict, self.masker, algorithm="permutation",
            feature_names=self.feature_names
        )

    def _transform(self, frame):
        transformed = self.preprocessor.transform(frame)
        return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)

    def _predict(self, transformed):
        return self.classifier.predict_proba(transformed)[:, 1]

    def explain(self, customers: pd.DataFrame, max_evals=None):
        transformed = self._transform(customers)
        if max_evals is None:
            max_evals = 2 * transformed.shape[1] + 1
        return self.explainer(transformed, max_evals=max_evals)

    @staticmethod
    def original_feature(name):
        name = name.split("__", 1)[-1]
        known = [
            "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod",
        ]
        for feature in known:
            if name == feature or name.startswith(feature + "_"):
                return feature
        return name

    def local_contributions(self, customer):
        explanation = self.explain(customer)
        values = np.asarray(explanation.values[0])
        aggregated = {}
        for name, value in zip(self.feature_names, values):
            feature = self.original_feature(name)
            aggregated[feature] = aggregated.get(feature, 0.0) + float(value)
        rows = [{"feature": k, "shap_value": v} for k, v in aggregated.items()]
        return pd.DataFrame(rows).sort_values(
            "shap_value", key=lambda s: s.abs(), ascending=False
        )

    def global_importance(self, customers, sample_size=150, random_state=42):
        sample = customers.sample(
            min(sample_size, len(customers)), random_state=random_state
        )
        explanation = self.explain(sample)
        vals = np.asarray(explanation.values)
        aggregated = {}
        for i, name in enumerate(self.feature_names):
            feature = self.original_feature(name)
            aggregated.setdefault(feature, []).append(np.abs(vals[:, i]))
        rows = [
            {"feature": k, "mean_abs_shap": float(np.mean(np.column_stack(v)))}
            for k, v in aggregated.items()
        ]
        return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False), explanation, sample

def plain_language_reasons(customer, contributions, top_n=5):
    positive = contributions[contributions.shap_value > 0].head(top_n)
    negative = contributions[contributions.shap_value < 0].head(top_n)
    def describe(row, direction):
        feature = row.feature
        value = customer.get(feature, "unknown")
        descriptions = {
            ("Contract", "Month-to-month"): "Month-to-month contract is associated with higher modelled churn risk.",
            ("TechSupport", "No"): "No technical support is associated with higher modelled churn risk.",
            ("OnlineSecurity", "No"): "No online security is associated with higher modelled churn risk.",
            ("PaymentMethod", "Electronic check"): "Electronic-check payment is associated with higher modelled churn risk.",
            ("InternetService", "Fiber optic"): "Fiber-optic service is associated with higher modelled churn risk in this model.",
        }
        key = (feature, str(value))
        if key in descriptions and direction == "increases":
            return descriptions[key]
        if feature == "tenure":
            return f"Tenure is {value} months; this feature {direction} the modelled churn score."
        if feature in ("MonthlyCharges", "TotalCharges"):
            return f"{feature} is {value}; this feature {direction} the modelled churn score."
        return f"{feature} = {value}; this feature {direction} the modelled churn score."
    reasons = []
    for _, row in positive.iterrows():
        reasons.append({"feature": row.feature, "direction": "increases",
                        "shap_value": float(row.shap_value),
                        "text": describe(row, "increases")})
    for _, row in negative.iterrows():
        reasons.append({"feature": row.feature, "direction": "decreases",
                        "shap_value": float(row.shap_value),
                        "text": describe(row, "decreases")})
    return reasons
