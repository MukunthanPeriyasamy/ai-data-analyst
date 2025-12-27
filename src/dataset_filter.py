import pandas as pd

def fliter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns

    df = df.select_dtypes(exclude=['object', 'datetime'])

    return df 

def get_dataset_metadata(df: pd.DataFrame) -> str:
    """
    Extracts high-level metadata and statistics from the dataset.
    Returns a formatted string summary.
    """
    shape = df.shape
    columns = list(df.columns)
    null_counts = df.isnull().sum().to_dict()
    
    # Statistical summary for numeric columns
    stats = df.describe().to_string()
    
    # Balance analysis for categorical or integer columns with few unique values
    balance_info = ""
    for col in df.columns:
        if df[col].nunique() < 10:  # Potential categorical/target variable
            balance_info += f"\nValue counts for '{col}':\n{df[col].value_counts().to_string()}\n"
            
    summary = (
        f"### Dataset Metadata Summary\n"
        f"- **Shape**: {shape[0]} rows, {shape[1]} columns\n"
        f"- **Columns**: {', '.join(columns)}\n"
        f"- **Missing Values**:\n{null_counts}\n"
        f"\n### Descriptive Statistics\n"
        f"{stats}\n"
        f"\n### Data Balance / Value Counts\n"
        f"{balance_info}"
    )
    
    return summary

