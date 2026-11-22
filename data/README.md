# Dataset

The app automatically downloads the IBM Telco Customer Churn sample dataset the first time it is needed if the CSV is absent.

Source: IBM's public sample repository:
https://github.com/IBM/telco-customer-churn-on-icp4d

Direct CSV:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The downloader verifies SHA-256 before accepting the file. If you already have the CSV, save it as `WA_Fn-UseC_-Telco-Customer-Churn.csv` in this folder. The raw data is excluded from Git by `.gitignore`.
