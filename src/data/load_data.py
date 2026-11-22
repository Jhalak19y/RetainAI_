from pathlib import Path
from urllib.request import urlretrieve
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

REQUIRED = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

def download_dataset(path=DEFAULT_DATA):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the IBM Telco sample dataset...")
    urlretrieve(DATA_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        path.unlink(missing_ok=True)
        raise ValueError("Downloaded dataset checksum did not match the expected IBM sample file.")
    return path

def load_data(path=DEFAULT_DATA):
    path = Path(path)
    if not path.exists():
        download_dataset(path)
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if "customerID" in df.columns:
        df = df.drop_duplicates("customerID").drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"]).copy()
    if df.empty:
        raise ValueError("No usable records remain after cleaning.")
    df["Churn"] = df["Churn"].astype(int)
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)
