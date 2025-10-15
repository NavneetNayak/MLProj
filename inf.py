import os
import warnings
import pandas as pd
import numpy as np
from joblib import load
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")
DATASET_PATH = "dataset/paad_tcga_gdc_clinical_data.tsv"
MODEL_PATH = "bin/model.joblib"
IMPUTER_PATH = "bin/imputer.joblib"
SCALER_PATH = "bin/scaler.joblib"

df = pd.read_csv(DATASET_PATH, sep="\t")

df = df.rename(columns={
    "Overall Survival (Months)": "duration",
    "Overall Survival Status": "event"
})

df["event"] = df["event"].str.contains("DECEASED").astype(int)

leakage_cols = [
    "Death from Initial Pathologic Diagnosis Date",
    "Last Communication Contact from Initial Pathologic Diagnosis Date",
    "Disease Free (Months)",
    "Disease Free Status",
    "American Joint Committee on Cancer Publication Version Type",
    "Patient's Vital Status"
]
id_cols = ["Study ID", "Patient ID", "Sample ID", "Other Patient ID", "Other Sample ID"]

df = df.drop(columns=leakage_cols + id_cols, errors='ignore')

df = df.dropna(axis=1, thresh=0.6 * len(df))
df = df.loc[:, df.nunique() > 1]

cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in ["duration", "event"]]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

samples = df_encoded.sample(n=5)
sample_info = samples[["duration", "event"]].copy()
X_new = samples.drop(columns=["duration", "event"], errors='ignore')

cph = load(MODEL_PATH)  
imputer = load(IMPUTER_PATH)
scaler = load(SCALER_PATH)

expected_features = list(imputer.feature_names_in_)

X_imputed = imputer.transform(X_new)  
X_imputed_df = pd.DataFrame(X_imputed, columns=expected_features, index=X_new.index)

X_scaled = scaler.transform(X_imputed_df)
X_scaled_df = pd.DataFrame(X_scaled, columns=expected_features, index=X_imputed_df.index)

model_features = list(cph.params_.index)
for feat in model_features:
    if feat not in X_scaled_df.columns:
        X_scaled_df[feat] = 0

X_final = X_scaled_df[model_features].copy()

pred_hazard = cph.predict_partial_hazard(X_final)

print(sample_info)
for idx, val in zip(pred_hazard.index, pred_hazard.values.flatten()):
    print(f"Sample {idx}: {val:.6f}")

