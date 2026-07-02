"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import mlflow

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = "/app/model"


def _find_model_path() -> Path:
    """Locate the trained MLflow model artifact from local development or container paths."""
    candidates = [
        Path(MODEL_DIR),
        ROOT / "artifacts" / "model",
    ]

    if (ROOT / "mlruns").exists():
        for mlmodel_file in sorted((ROOT / "mlruns").rglob("MLmodel")):
            if mlmodel_file.is_file():
                candidates.append(mlmodel_file.parent)

        for experiment_dir in sorted((ROOT / "mlruns").iterdir()):
            if not experiment_dir.is_dir():
                continue
            for run_dir in sorted(experiment_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                artifact_dir = run_dir / "artifacts" / "model"
                if artifact_dir.exists():
                    candidates.append(artifact_dir)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("No trained model artifact was found.")


def _load_feature_columns() -> list:
    """Load the exact feature column order used during training."""
    candidates = [
        ROOT / "artifacts" / "feature_columns.json",
        ROOT / "artifacts" / "feature_columns.txt",
        ROOT / "artifacts" / "preprocessing.pkl",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        if candidate.suffix == ".json":
            with open(candidate, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
                if isinstance(loaded, list):
                    return loaded
                if isinstance(loaded, dict) and "feature_columns" in loaded:
                    return loaded["feature_columns"]

        if candidate.suffix == ".txt":
            with open(candidate, "r", encoding="utf-8") as handle:
                return [line.strip() for line in handle if line.strip()]

        if candidate.suffix == ".pkl":
            import joblib
            payload = joblib.load(candidate)
            if isinstance(payload, dict) and "feature_columns" in payload:
                return payload["feature_columns"]

    raise FileNotFoundError("Could not load feature columns from local artifacts.")


try:
    MODEL_DIR = _find_model_path()
    model = mlflow.sklearn.load_model(str(MODEL_DIR))
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    raise

try:
    FEATURE_COLS = _load_feature_columns()
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    return df

def predict_with_details(input_dict: dict) -> Dict[str, Any]:
    """Return the churn label and probability for a single customer record."""
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(df_enc)[0, 1])
            result = int(probability >= 0.5)
        else:
            result = int(model.predict(df_enc)[0])
            probability = float(result)
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    label = "Likely to churn" if result == 1 else "Not likely to churn"
    return {
        "prediction": label,
        "predicted_class": result,
        "probability": probability,
    }


def predict(input_dict: dict) -> str:
    """Return a human-readable churn prediction for the supplied customer data."""
    return predict_with_details(input_dict)["prediction"]