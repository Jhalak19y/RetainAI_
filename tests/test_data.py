import pandas as pd
from src.data.load_data import REQUIRED

def test_required_columns_are_unique():
    assert len(REQUIRED) == len(set(REQUIRED))
    assert "Churn" in REQUIRED
