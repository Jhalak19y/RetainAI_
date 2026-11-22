def recommend_actions(customer, contributions):
    """Rule-based suggestions; these are hypotheses, not guaranteed interventions."""
    rules = {
        "Contract": ("Discuss a longer-term contract with a transparent price comparison.",
                     "Contract flexibility may be contributing to risk."),
        "TechSupport": ("Offer a guided technical-support trial or onboarding session.",
                        "The customer does not currently have technical support."),
        "OnlineSecurity": ("Explain available online-security options and benefits.",
                           "The customer does not currently have online security."),
        "tenure": ("Prioritise a proactive check-in and onboarding review.",
                   "The customer has relatively short tenure."),
        "PaymentMethod": ("Offer help setting up an automatic payment method if suitable.",
                          "Payment method is a modelled churn signal."),
    }
    output = []
    for _, row in contributions.iterrows():
        feature = row["feature"]
        if row["shap_value"] <= 0 or feature not in rules:
            continue
        action, reason = rules[feature]
        if feature == "tenure" and float(customer.get("tenure", 999)) > 12:
            continue
        output.append({
            "feature": feature, "action": action, "reason": reason,
            "priority_score": round(float(row["shap_value"]), 4)
        })
    return sorted(output, key=lambda x: x["priority_score"], reverse=True)
