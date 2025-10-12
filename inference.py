import pandas as pd
import numpy as np

from joblib import load

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from config import *

cph = load(f"{MODEL_PATH}.joblib")

file_path = f"{DATASET_PATH}.tsv"
df = pd.read_csv(file_path, sep="\t")

sample_df = df.sample(n=5, random_state=42)

sample_original = sample_df.copy()

leakage_cols = [
    "Death from Initial Pathologic Diagnosis Date",
    "Last Communication Contact from Initial Pathologic Diagnosis Date",
    "Disease Free (Months)",
    "Disease Free Status"
]
id_cols = ["Study ID", "Patient ID", "Sample ID", "Other Patient ID", "Other Sample ID"]
sample_df = sample_df.drop(columns=leakage_cols + id_cols, errors='ignore')

sample_df = sample_df.loc[:, sample_df.nunique() > 1]

cat_cols = sample_df.select_dtypes(include="object").columns.tolist()
sample_df_encoded = pd.get_dummies(sample_df, columns=cat_cols, drop_first=True)

model_features = cph.params_.index.tolist()
sample_X = sample_df_encoded.reindex(columns=model_features, fill_value=0)

imputer = IterativeImputer(max_iter=20, random_state=42)
sample_X_imputed = imputer.fit_transform(sample_X)
sample_X = pd.DataFrame(sample_X_imputed, columns=sample_X.columns)

scaler = StandardScaler()
sample_X_scaled = scaler.fit_transform(sample_X)
sample_X = pd.DataFrame(sample_X_scaled, columns=sample_X.columns)

pred_hazard = cph.predict_partial_hazard(sample_X)

key_features = ["Diagnosis Age", "Fraction Genome Altered", "Mutation Count"]
pretty_df = sample_original[key_features + ["Overall Survival (Months)", "Overall Survival Status"]].copy()
pretty_df = pretty_df.rename(columns={
    "Overall Survival (Months)": "duration",
    "Overall Survival Status": "event"
})
pretty_df["Predicted_Hazard"] = pred_hazard.values

pretty_df["event"] = pretty_df["event"].apply(lambda x: 1 if "DECEASED" in str(x) else 0)

print(pretty_df.to_string(index=False))

