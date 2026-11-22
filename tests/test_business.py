from src.business import risk_band, roi_calculation

def test_risk_bands():
    assert risk_band(0.2) == "Low"
    assert risk_band(0.5) == "Medium"
    assert risk_band(0.8) == "High"

def test_roi_math():
    result = roi_calculation(100, 1000, 10, 500, 0.2, 0.5)
    assert result["expected_acceptances"] == 20
    assert result["expected_retained"] == 10
    assert result["offer_spend"] == 1000
    assert result["gross_value"] == 5000
    assert result["net_expected_value"] == 4000
