import pandas as pd
def preprocess_data(df:pd.DataFrame,target_col:str = "Churn")->pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """
    df.columns = df.columns.str.strip()
    
    for col in ["CustomerID","customerID"]:
        if col in df.columns:
            df.drop(columns=col,inplace=True)
            
    if target_col in df.columns and df[target_col].dtype =="object":
        df[target_col] = df[target_col].map({"Yes":1,"No":0})
        
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
        
    if "seniorcitizen" in df.columns:
        df["seniorcitizen"] = df["seniorcitizen"].fillna(0).astype(int)
    
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    
    return df