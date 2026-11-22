def risk_band(probability):
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"

def roi_calculation(top_n, population, offer_cost, lifetime_value, acceptance_rate, save_rate):
    """All financial inputs are user-entered assumptions."""
    targeted = max(0, min(int(top_n), int(population)))
    accepted = targeted * acceptance_rate
    retained = accepted * save_rate
    gross_value = retained * lifetime_value
    offer_spend = targeted * offer_cost
    return {
        "targeted": targeted,
        "expected_acceptances": accepted,
        "expected_retained": retained,
        "gross_value": gross_value,
        "offer_spend": offer_spend,
        "net_expected_value": gross_value - offer_spend,
    }

def threshold_counts(y_true, probabilities, threshold):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    return {"true_positives": int(tp), "false_positives": int(fp),
            "false_negatives": int(fn), "true_negatives": int(tn),
            "offers_to_non_churners": int(fp)}
