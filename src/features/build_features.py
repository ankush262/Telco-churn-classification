import pandas as pd
def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Map a binary series to 0 and 1.
    
    Parameters:
    s (pd.Series): A pandas Series containing binary values (e.g., "Yes"/"No").
    
    Returns:
    pd.Series: A pandas Series with values mapped to 0 and 1.
    """
    #get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)
    
    # === DETERMINISTIC BINARY MAPPINGS ===
    # CRITICAL: These exact mappings are hardcoded in serving pipeline
    
    # Yes/No mapping (most common pattern in telecom data)
    if valset == {"Yes", "No"}:
        return s.map({"Yes": 1, "No": 0}).astype("Int64")
    #gender mapping(demographic data)
    if valset == {"Male", "Female"}:
        return s.map({"Male": 1, "Female": 0}).astype("Int64")  
    if len(vals)==2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.map(mapping).astype("Int64")
    return s

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """
    df = df.copy()
    print(f" starting feature engineering on {df.shape[1]} columns...")
    
    #step1 : identify feature types
    obj_cols = ([c for c in df.columns if df[c].dtype == "object" and c != target_col])
    numeric_cols = ([c for c in df.columns if df[c].dtype in ["int64", "float64"] and c != target_col])
    print(f"found {len(obj_cols)} object columns and {len(numeric_cols)} numeric columns")
    # step 2 : split categorical columns into binary and multi-class
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi__cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    print(f"found {len(binary_cols)} binary columns and {len(multi__cols)} multi-class columns")
    if binary_cols:
        print(f" binary columns: {binary_cols}")
    if multi__cols:
        print(f" multi-class columns: {multi__cols}")
        
    
    #step 3 apply binary encoding
    for col in binary_cols:
        df[col] = _map_binary_series(df[col].astype(str))
        print(f"mapped binary column {col} to 0/1")
    
    #convert boolen columns to int
    bool_cols = [c for c in df.columns if df[c].dtype == "bool"]
    for col in bool_cols:
        df[col] = df[col].astype(int)
        print(f"converted boolean column {col} to int")
        
    #step 4 apply one-hot encoding to multi-class columns
    if multi__cols:
        df = pd.get_dummies(df, columns=multi__cols, drop_first=True)
        print(f"applied one-hot encoding to {len(multi__cols)} multi-class columns")
        
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype("Int64")
    
    print(f"feature engineering complete. final dataset has {df.shape[1]} columns")
    return df