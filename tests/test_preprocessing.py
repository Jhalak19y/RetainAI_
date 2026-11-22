import pandas as pd
from src.features.preprocessing import create_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def test_preprocessor_handles_missing_and_unknown():
    row = {c: 1 for c in NUMERIC_FEATURES}
    row.update({c: "Yes" for c in CATEGORICAL_FEATURES})
    row["tenure"] = None
    X = pd.DataFrame([row])
    transformer = create_preprocessor()
    transformed = transformer.fit_transform(X)
    assert transformed.shape[0] == 1
