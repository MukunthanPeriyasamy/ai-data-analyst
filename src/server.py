"""mcp_tools.py
Utility helpers for missing-value handling and outlier handling used by the
AI Data Analyst project.

This module is exposed as an MCP (Model Context Protocol) server, providing tools for:
- diagnostics to decide between mean/median imputation for numeric columns,
- simple imputation helpers (`impute_with_mean`, `impute_with_median`),
- a top-level `handling_missing_values` orchestration function, and
- a set of outlier utilities.
"""

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import RobustScaler
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("Data Analyst Tools")

# Global state to keep the active DataFrame in memory on the server side
_active_df = None

def get_df(data: list[dict] = None) -> pd.DataFrame:
    """Helper to get the DataFrame from either provided data or global state."""
    global _active_df
    if data is not None:
        return pd.DataFrame(data)
    if _active_df is not None:
        return _active_df
    # Fallback: try to load from the default path if it exists
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_path = os.path.join(base_dir, "data", "dataset.csv")
    if os.path.exists(default_path):
        _active_df = pd.read_csv(default_path)
        return _active_df
    raise ValueError("No active dataset loaded and no data provided.")

@mcp.tool()
def load_dataset(path: str) -> str:
    """Load a CSV dataset into the server's memory.
    
    Args:
        path: Absolute path to the CSV file.
    """
    global _active_df
    try:
        _active_df = pd.read_csv(path)
        cols = list(_active_df.columns)
        return (f"Successfully loaded dataset from {path}. "
                f"Shape: {_active_df.shape}. "
                f"Columns: {cols}")
    except Exception as e:
        return f"Error loading dataset: {e}"

@mcp.tool()
def get_columns() -> list[str]:
    """Get the column names of the active dataset."""
    global _active_df
    if _active_df is None:
        get_df() 
    return list(_active_df.columns)

@mcp.tool()
def filter_function(data: list[dict] | None = None) -> str:
    """Select numeric (non-object) columns and update the active dataset.
    
    If 'data' is provided, it filters that data. Otherwise, it filters the active dataset.
    """
    global _active_df
    try:
        df = get_df(data)
        df2 = df.select_dtypes(exclude=['object']).copy()
        _active_df = df2
        return f"Filtered dataset to numeric columns only. New shape: {_active_df.shape}. Columns: {list(_active_df.columns)}"
    except Exception as e:
        return f"Error in filter_function: {e}"


# ---------------------------------------------------------------------------
# Imputation diagnostics
# ---------------------------------------------------------------------------
@mcp.tool()
def compute_imputation_decision(column: str, threshold: float = 0.05) -> dict:
    """Compute diagnostics and choose an imputation strategy for a numeric column.

    Args:
        column: The name of the numeric column in the active dataset.
        threshold: Numeric threshold (default 0.05) used to decide strategy.

    Returns:
        dict containing strategy and diagnostic metrics.
    """
    global _active_df
    if _active_df is None:
        get_df() # Trigger fallback loading
    
    if column not in _active_df.columns:
        return {"error": f"Column '{column}' not found in active dataset."}
        
    series = _active_df[column]
    s = series.dropna()
    mean_value = s.mean() if len(s) > 0 else 0.0
    median_value = s.median() if len(s) > 0 else 0.0

    denom = abs(median_value) if abs(median_value) > 0 else (abs(mean_value) if abs(mean_value) > 0 else 1.0)
    mean_median_distance = abs(mean_value - median_value) / denom

    skewness = abs(s.skew()) if len(s) > 2 else 0.0

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or len(s) == 0:
        outlier_percentage = 0.0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = ((s < lower) | (s > upper)).sum()
        outlier_percentage = outlier_count / len(s)

    use_mean = (mean_median_distance < threshold) and (skewness < threshold) and (outlier_percentage < threshold)
    strategy = "mean" if use_mean else "median"
    
    missing_count = int(_active_df[column].isnull().sum())

    return {
        "column": column,
        "missing_count": missing_count,
        "mean": mean_value,
        "median": median_value,
        "mean_median_distance": mean_median_distance,
        "skewness": skewness,
        "outlier_percentage": outlier_percentage,
        "strategy": strategy,
    }


# ---------------------------------------------------------------------------
# Imputation helpers
# ---------------------------------------------------------------------------
@mcp.tool()
def impute_with_mean(col: str) -> str:
    """Impute missing values in `col` of active dataset with the column mean."""
    global _active_df
    if col not in _active_df.columns:
        return f"Error: Column '{col}' not found."
    mean_value = _active_df[col].mean()
    _active_df[col] = _active_df[col].fillna(mean_value)
    return f"Imputed missing values in '{col}' with mean ({mean_value:.4f})."


@mcp.tool()
def impute_with_median(col: str) -> str:
    """Impute missing values in `col` of active dataset with the column median."""
    global _active_df
    if col not in _active_df.columns:
        return f"Error: Column '{col}' not found."
    median_value = _active_df[col].median()
    _active_df[col] = _active_df[col].fillna(median_value)
    return f"Imputed missing values in '{col}' with median ({median_value:.4f})."


# ---------------------------------------------------------------------------
# High-level missing-value handling
# ---------------------------------------------------------------------------
@mcp.tool()
def handling_missing_values(threshold: float = 0.05) -> str:
    """Orchestrate missing-value handling for the entire active dataset."""
    global _active_df
    if _active_df is None:
        return "Error: No dataset loaded."
    
    df_out = _active_df.copy()
    for col in list(df_out.columns):
        missing_percentage = df_out[col].isnull().mean()

        if missing_percentage > 0.3:
            df_out.drop(columns=[col], inplace=True)
            continue

        if missing_percentage == 0:
            continue

        if not is_numeric_dtype(df_out[col].dtype):
            mode_vals = df_out[col].mode()
            if len(mode_vals) > 0:
                df_out[col] = df_out[col].fillna(mode_vals.iloc[0])
            continue

        metrics = compute_imputation_decision(col, threshold=threshold)
        if "error" in metrics: continue
        
        if metrics["strategy"] == "mean":
            df_out[col] = df_out[col].fillna(df_out[col].mean())
        else:
            df_out[col] = df_out[col].fillna(df_out[col].median())

    _active_df = df_out
    return f"Completed high-level missing value handling. New shape: {_active_df.shape}"


# ---------------------------------------------------------------------------
# Outlier utilities
# ---------------------------------------------------------------------------
@mcp.tool()
def outliers_cap(cap_percentiles: list[float] = [0.05, 0.95]) -> str:
    """Cap numeric columns of active dataset at the provided percentiles."""
    global _active_df
    numeric_cols = _active_df.select_dtypes(include=[np.number]).columns
    low_p, high_p = cap_percentiles
    for col in numeric_cols:
        s = _active_df[col].dropna()
        if s.empty: continue
        _active_df[col] = _active_df[col].clip(lower=s.quantile(low_p), upper=s.quantile(high_p))
    return f"Capped outliers at {cap_percentiles} percentiles."


@mcp.tool()
def outliers_remove(iqr_multiplier: float = 1.5) -> str:
    """Remove rows that are IQR-outliers in any numeric column of active dataset."""
    global _active_df
    numeric_cols = _active_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return "No numeric columns found."
    mask = pd.Series(True, index=_active_df.index)
    for col in numeric_cols:
        s = _active_df[col].dropna()
        if s.empty: continue
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
        mask &= ((_active_df[col].isnull()) | ((_active_df[col] >= lower) & (_active_df[col] <= upper)))
    _active_df = _active_df[mask].copy()
    return f"Removed outliers. New shape: {_active_df.shape}"


@mcp.tool()
def outliers_impute(strategy: str = 'median', iqr_multiplier: float = 1.5) -> str:
    """Impute outlier values in active dataset with a robust statistic."""
    global _active_df
    numeric_cols = _active_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        s = _active_df[col]
        non_null = s.dropna()
        if non_null.empty: continue
        Q1, Q3 = non_null.quantile(0.25), non_null.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
        non_outliers = non_null[(non_null >= lower) & (non_null <= upper)]
        fill_val = non_outliers.median() if strategy == 'median' else non_outliers.mean()
        if non_outliers.empty:
             fill_val = non_null.median() if strategy == 'median' else non_null.mean()
        _active_df.loc[(s < lower) | (s > upper), col] = fill_val
    return f"Imputed outliers using {strategy} strategy."


@mcp.tool()
def outliers_robust_scale(with_centering: bool = True, with_scaling: bool = True) -> str:
    """Apply RobustScaler to numeric columns of the active dataset."""
    global _active_df
    numeric_cols = _active_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return "No numeric columns to scale."
    scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
    scaled = scaler.fit_transform(_active_df[numeric_cols].astype(float))
    _active_df[numeric_cols] = pd.DataFrame(scaled, index=_active_df.index, columns=numeric_cols)
    return "Successfully applied robust scaling to numeric columns."


if __name__ == "__main__":
    mcp.run()
