from typing import Tuple, List
import pandas as pd

try:
    import great_expectations as ge
except ImportError:  # pragma: no cover - fallback when package is unavailable
    ge = None


def _validate_with_great_expectations(df):
    if ge is None:
        return None

    try:
        if hasattr(ge, "dataset"):
            ge_df = ge.dataset.PandasDataset(df)
        else:
            ge_df = ge.from_pandas(df)
    except Exception:
        return None

    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    ge_df.expect_column_to_exist("gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")

    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Contract", ["Month-to-month", "One year", "Two year"])
    ge_df.expect_column_values_to_be_in_set("InternetService", ["DSL", "Fiber optic", "No"])
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")

    try:
        ge_df.expect_column_pair_values_A_to_be_greater_than_B(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,
        )
    except Exception:
        pass

    try:
        results = ge_df.validate()
    except Exception:
        return None

    failed_expectations = []
    for r in results.get("results", []):
        if not r.get("success", False):
            expectation_type = r.get("expectation_config", {}).get("expectation_type")
            failed_expectations.append(expectation_type)

    success = bool(results.get("success", False))
    return success, failed_expectations


def _validate_with_pandas_fallback(df):
    failures = []
    required_cols = ["customerID", "gender", "Partner", "Dependents", "PhoneService", "InternetService", "Contract", "tenure", "MonthlyCharges", "TotalCharges"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        failures.append(f"missing columns: {missing}")

    if "customerID" in df.columns and df["customerID"].isna().any():
        failures.append("customerID contains nulls")

    for col in ["gender", "Partner", "Dependents", "PhoneService"]:
        if col in df.columns:
            allowed = {"Male", "Female"} if col == "gender" else {"Yes", "No"}
            invalid = set(df[col].dropna().astype(str).unique()) - allowed
            if invalid:
                failures.append(f"{col} has invalid values: {sorted(invalid)}")

    if "Contract" in df.columns:
        allowed = {"Month-to-month", "One year", "Two year"}
        invalid = set(df["Contract"].dropna().astype(str).unique()) - allowed
        if invalid:
            failures.append(f"Contract has invalid values: {sorted(invalid)}")

    if "InternetService" in df.columns:
        allowed = {"DSL", "Fiber optic", "No"}
        invalid = set(df["InternetService"].dropna().astype(str).unique()) - allowed
        if invalid:
            failures.append(f"InternetService has invalid values: {sorted(invalid)}")

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                numeric = pd.Series([pd.NA] * len(df), index=df.index)
            if numeric.lt(0).any():
                failures.append(f"{col} contains negative values")
            if col == "tenure" and numeric.gt(120).any():
                failures.append("tenure exceeds 120 months")
            if col == "MonthlyCharges" and numeric.gt(200).any():
                failures.append("MonthlyCharges exceeds 200")

    return (len(failures) == 0), failures


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation...")

    result = _validate_with_great_expectations(df)
    if result is not None:
        success, failed_expectations = result
        if success:
            print(f"✅ Data validation PASSED using Great Expectations")
        else:
            print(f"❌ Data validation FAILED: {failed_expectations}")
        return success, failed_expectations

    success, failures = _validate_with_pandas_fallback(df)
    if success:
        print("✅ Data validation PASSED using pandas fallback")
    else:
        print(f"❌ Data validation FAILED: {failures}")
    return success, failures