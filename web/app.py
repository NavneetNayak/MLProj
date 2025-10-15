"""
Flask Web Application for Pancreatic Cancer Survival Prediction
Uses trained Cox Proportional Hazards model from the training notebook
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from joblib import load
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Global variables for model components
model = None
imputer = None
scaler = None
feature_columns = None
MODEL_CINDEX = None

def load_models():
    """Load the trained model, scaler, and imputer"""
    global model, imputer, scaler, feature_columns, MODEL_CINDEX
    
    try:
        # Paths to saved models
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bin')
        
        print(f"Loading models from {model_dir}...")
        
        # Load model components
        model = load(os.path.join(model_dir, 'model.joblib'))
        imputer = load(os.path.join(model_dir, 'imputer.joblib'))
        scaler = load(os.path.join(model_dir, 'scaler.joblib'))
        
        # Get feature columns from scaler (these are the columns after preprocessing)
        feature_columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
        
        # Get model concordance index
        MODEL_CINDEX = f"{model.concordance_index_:.3f}" if hasattr(model, 'concordance_index_') else "N/A"

        # Build reference processed matrix and compute partial hazards distribution so we can
        # compute percentiles for UI (helps interpret very small/large hazards)
        try:
            # Recreate reference preprocessing (same logic as preprocess_input)
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'paad_tcga_gdc_clinical_data.tsv')
            df_ref = pd.read_csv(dataset_path, sep='\t')
            leakage_cols = [
                "Death from Initial Pathologic Diagnosis Date",
                "Last Communication Contact from Initial Pathologic Diagnosis Date",
                "Disease Free (Months)",
                "Disease Free Status",
                "Patient's Vital Status"
            ]
            id_cols = ["Study ID", "Patient ID", "Sample ID", "Other Patient ID", "Other Sample ID", "American Joint Committee on Cancer Publication Version Type"]
            df_ref = df_ref.drop(columns=leakage_cols + id_cols, errors='ignore')
            df_ref = df_ref.rename(columns={"Overall Survival (Months)": "duration",
                                            "Overall Survival Status": "event"})
            df_ref = df_ref.dropna(axis=1, thresh=0.6 * len(df_ref))
            df_ref = df_ref.loc[:, df_ref.nunique() > 1]

            cat_cols_ref = df_ref.select_dtypes(include=["object"]).columns.tolist()
            cat_cols_ref = [c for c in cat_cols_ref if c not in ["duration", "event"]]

            # One-hot encode reference and get feature columns
            df_ref_encoded = pd.get_dummies(df_ref, columns=cat_cols_ref, drop_first=True)
            REF_COLUMNS = [c for c in df_ref_encoded.columns if c not in ['duration', 'event']]

            # Impute and scale reference features to compute partial hazards
            X_ref = df_ref_encoded[REF_COLUMNS]
            X_ref_imputed = imputer.transform(X_ref)
            X_ref_scaled = scaler.transform(X_ref_imputed)

            # Compute partial hazards array
            ref_ph = model.predict_partial_hazard(pd.DataFrame(X_ref_scaled, columns=REF_COLUMNS))
            MODEL_HAZARDS = ref_ph.values.flatten()
            # compute median hazard for relative risk normalization
            MODEL_MEDIAN_HAZARD = float(np.median(MODEL_HAZARDS)) if len(MODEL_HAZARDS) else None
            # store for global use
            globals().update({'REF_COLUMNS': REF_COLUMNS, 'MODEL_HAZARDS': MODEL_HAZARDS, 'MODEL_MEDIAN_HAZARD': MODEL_MEDIAN_HAZARD})
        except Exception as e:
            print('Warning: could not compute reference hazards for percentiles:', e)
            globals().update({'REF_COLUMNS': None, 'MODEL_HAZARDS': None})
        
        print("✓ Models loaded successfully")
        print(f"  Model C-Index: {MODEL_CINDEX}")
        print(f"  Number of features: {len(feature_columns) if feature_columns is not None else 'Unknown'}")
        
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False

def preprocess_input(input_data):
    """
    Preprocess user input to match the training data format
    
    Steps:
    1. Create a DataFrame with all possible features
    2. Fill in user-provided values and add missing columns with defaults
    3. Apply one-hot encoding for categorical variables
    4. Impute missing values
    5. Scale features
    """
    
    # Load reference data to get all possible column names after one-hot encoding
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'dataset', 'paad_tcga_gdc_clinical_data.tsv')
    df_ref = pd.read_csv(dataset_path, sep='\t')
    
    # Apply same preprocessing as training
    leakage_cols = [
        "Death from Initial Pathologic Diagnosis Date",
        "Last Communication Contact from Initial Pathologic Diagnosis Date",
        "Disease Free (Months)",
        "Disease Free Status",
        "Patient's Vital Status"
    ]
    id_cols = ["Study ID", "Patient ID", "Sample ID", "Other Patient ID", 
               "Other Sample ID", "American Joint Committee on Cancer Publication Version Type"]
    
    df_ref = df_ref.drop(columns=leakage_cols + id_cols, errors='ignore')
    df_ref = df_ref.rename(columns={
        "Overall Survival (Months)": "duration",
        "Overall Survival Status": "event"
    })
    
    # Drop columns with < 60% data and single-value columns (same as training notebook)
    df_ref = df_ref.dropna(axis=1, thresh=0.6 * len(df_ref))
    df_ref = df_ref.loc[:, df_ref.nunique() > 1]
    
    # Fill missing columns with defaults (for columns that were dropped in preprocessing)
    # These are constant columns that don't affect predictions
    default_values = {
        "Biopsy Site": "Pancreas",
        "Cancer Type": "Pancreatic Adenocarcinoma",
        "Cancer Type Detailed": "Pancreatic Adenocarcinoma",
        "Is FFPE": "NO",
        "Oncotree Code": "PAAD",
        "Patient Primary Tumor Site": "Pancreas",
        "Prior Malignancy": False,
        "Prior Treatment": False,
        "Project Identifier": "TCGA-PAAD",
        "Project Name": "Pancreatic Adenocarcinoma",
        "Project State": "released"
    }
    
    # Add default values for missing columns
    for col, default_val in default_values.items():
        if col not in input_data:
            input_data[col] = default_val
    
    # Get categorical columns from reference (after filtering)
    cat_cols = df_ref.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["duration", "event"]]
    
    # Create a single-row DataFrame with user input
    user_df = pd.DataFrame([input_data])
    
    # Only encode categorical columns that exist in the filtered reference
    cat_cols_to_encode = [c for c in cat_cols if c in user_df.columns]
    
    # Apply one-hot encoding
    user_encoded = pd.get_dummies(user_df, columns=cat_cols_to_encode, drop_first=True)
    
    # Get all columns from reference after encoding
    df_ref_encoded = pd.get_dummies(df_ref, columns=cat_cols, drop_first=True)
    ref_columns = [c for c in df_ref_encoded.columns if c not in ['duration', 'event']]
    
    # Ensure user data has all columns (fill missing with 0)
    for col in ref_columns:
        if col not in user_encoded.columns:
            user_encoded[col] = 0
    
    # Keep only the columns that exist in reference
    user_encoded = user_encoded[ref_columns]
    
    # Apply imputation
    user_imputed = imputer.transform(user_encoded)
    user_df_imputed = pd.DataFrame(user_imputed, columns=ref_columns)
    
    # Apply scaling
    user_scaled = scaler.transform(user_df_imputed)
    user_df_scaled = pd.DataFrame(user_scaled, columns=ref_columns)
    
    return user_df_scaled

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    Expects JSON with patient features
    """
    try:
        # Get input data
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Preprocess input
        processed_data = preprocess_input(input_data)

        # Make prediction (partial hazard)
        risk_score = float(model.predict_partial_hazard(processed_data).values[0])

        # Return just the partial hazard prediction
        return jsonify({
            'partial_hazard': float(risk_score),
            'success': True
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/random_sample', methods=['GET'])
def random_sample():
    """
    Return a random patient sample from the dataset for demo purposes
    """
    try:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'dataset', 'paad_tcga_gdc_clinical_data.tsv')
        df = pd.read_csv(dataset_path, sep='\t')
        
        # Get a random row
        random_row = df.sample(n=1).iloc[0].to_dict()
        
        # Map to the fields our form expects (keep only relevant fields)
        sample_data = {}
        
        # Numeric fields
        if 'Diagnosis Age' in random_row and pd.notna(random_row['Diagnosis Age']):
            sample_data['Diagnosis Age'] = int(random_row['Diagnosis Age'])
        
        if 'Fraction Genome Altered' in random_row and pd.notna(random_row['Fraction Genome Altered']):
            sample_data['Fraction Genome Altered'] = float(random_row['Fraction Genome Altered'])
        
        if 'Mutation Count' in random_row and pd.notna(random_row['Mutation Count']):
            sample_data['Mutation Count'] = int(random_row['Mutation Count'])
        
        if 'Birth from Initial Pathologic Diagnosis Date' in random_row and pd.notna(random_row['Birth from Initial Pathologic Diagnosis Date']):
            sample_data['Birth from Initial Pathologic Diagnosis Date'] = int(random_row['Birth from Initial Pathologic Diagnosis Date'])
        
        if 'Year of Diagnosis' in random_row and pd.notna(random_row['Year of Diagnosis']):
            sample_data['Year of Diagnosis'] = int(random_row['Year of Diagnosis'])
        
        if 'Number of Samples Per Patient' in random_row and pd.notna(random_row['Number of Samples Per Patient']):
            sample_data['Number of Samples Per Patient'] = int(random_row['Number of Samples Per Patient'])
        
        if 'Sample type id' in random_row and pd.notna(random_row['Sample type id']):
            sample_data['Sample type id'] = int(random_row['Sample type id'])
        
        # Categorical fields
        categorical_fields = [
            'Sex', 'Race Category', 'Ethnicity Category', 'Alcohol History Documented',
            'Disease Type', 'Primary Diagnosis', 'Morphology', 'ICD-10 Classification',
            'AJCC Pathologic Stage', 'AJCC Pathologic T-Stage', 'AJCC Pathologic N-Stage',
            'AJCC Pathologic M-Stage', 'Sample Type'
        ]
        
        for field in categorical_fields:
            if field in random_row and pd.notna(random_row[field]):
                sample_data[field] = str(random_row[field])
        
        return jsonify({
            'success': True,
            'data': sample_data
        })
        
    except Exception as e:
        print(f"Random sample error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_cindex': MODEL_CINDEX
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Pancreatic Cancer Survival Prediction Web App")
    print("=" * 60)
    
    # Load models before starting server
    if not load_models():
        print("\n⚠️  Warning: Models not loaded. Please train the model first.")
        print("   Run the training.ipynb notebook to generate model files.")
        sys.exit(1)
    
    print("\n🚀 Starting Flask server...")
    print("   Open http://localhost:5000 in your browser")
    print("   Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
