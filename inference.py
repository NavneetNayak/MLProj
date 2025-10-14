import pandas as pd
import warnings
import pandas as pd
import numpy as np
from joblib import load
import numpy as np
from joblib import load
import os

DATASET_PATH = "dataset/paad_tcga_gdc_clinical_data.tsv"
MODEL_PATH = "bin/model.joblib"
IMPUTER_PATH = "bin/imputer.joblib"
SCALER_PATH = "bin/scaler.joblib"

warnings.filterwarnings("ignore")  
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

df_encoded["duration"] = np.log1p(df_encoded["duration"])

cph = load(MODEL_PATH)
imputer = load(IMPUTER_PATH)
scaler = load(SCALER_PATH)

samples = df_encoded.sample(n=5, random_state=2)
sample_info = samples[["duration", "event"]]

X_new = samples.drop(columns=["duration", "event"])

for col in cph.params_.index:
    if col not in X_new.columns:
        X_new[col] = 0
X_new = X_new[cph.params_.index] 

X_new_imputed = imputer.transform(X_new)
X_new_scaled = scaler.transform(X_new_imputed)

X_new_final = pd.DataFrame(X_new_scaled, columns=X_new.columns, index=samples.index)

pred_hazard = cph.predict_partial_hazard(X_new_final)

print(sample_info)

for idx, val in zip(pred_hazard.index, pred_hazard.values.flatten()):
    print(f"Sample {idx}: {val:.4f}")

