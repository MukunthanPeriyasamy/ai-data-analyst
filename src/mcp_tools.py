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


# ---------------------------------------------------------------------------
# Imputation diagnostics
# ---------------------------------------------------------------------------
@mcp.tool()
def compute_imputation_decision(series: pd.Series, threshold: float = 0.05) -> dict:
    """Compute diagnostics and choose an imputation strategy for a numeric series.

    Short summary:
    - Computes mean, median, relative mean-median distance, absolute skewness,
      and the fraction of IQR-based outliers.
    - Recommends `"mean"` when all diagnostics are below `threshold`,
      otherwise recommends `"median"`.

    Args:
        series: Numeric `pd.Series` to inspect (may include NA values).
        threshold: Numeric threshold (default 0.05) used to decide strategy.

    Returns:
        dict containing `mean`, `median`, `mean_median_distance`, `skewness`,
        `outlier_percentage`, and `strategy` ("mean" or "median").
    """
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

    missing_count = int(series.isnull().sum())

    return {
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
def impute_with_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Impute missing values in `col` with the column mean.

    Returns a shallow-copied DataFrame with the imputation applied.

    Args:
        df: Input DataFrame.
        col: Column name to impute.

    Returns:
        DataFrame copy with `col` NA values replaced by the column mean.
    """
    df_out = df.copy()
    if col not in df_out.columns:
        return df_out
    mean_value = df_out[col].mean()
    df_out[col] = df_out[col].fillna(mean_value)
    return df_out


@mcp.tool()
def impute_with_median(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Impute missing values in `col` with the column median.

    Returns a shallow-copied DataFrame with the imputation applied.

    Args:
        df: Input DataFrame.
        col: Column name to impute.

    Returns:
        DataFrame copy with `col` NA values replaced by the column median.
    """
    df_out = df.copy()
    if col not in df_out.columns:
        return df_out
    median_value = df_out[col].median()
    df_out[col] = df_out[col].fillna(median_value)
    return df_out


# ---------------------------------------------------------------------------
# High-level missing-value handling
# ---------------------------------------------------------------------------
@mcp.tool()
def handling_missing_values(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Orchestrate missing-value handling for each column in a DataFrame.

    Summary of rules:
    - Drop any column with >30% missing values.
    - For non-numeric columns, perform mode imputation when possible.
    - For numeric columns, call `compute_imputation_decision` to choose between
      mean and median imputation, then apply the corresponding helper.

    Args:
        df: Input DataFrame.
        threshold: Threshold passed to diagnostics (default 0.05).

    Returns:
        A new DataFrame with imputed values (or columns dropped where required).
    """
    df_out = df.copy()
    for col in list(df_out.columns):
        missing_count = df_out[col].isnull().sum()
        total_count = len(df_out) if len(df_out) > 0 else 1
        missing_percentage = (missing_count / total_count)

        if missing_percentage > np.float64(0.3):
            print(f"{col}: {missing_percentage:.2%} missing — dropping column")
            df_out.drop(columns=[col], inplace=True)
            continue

        if missing_percentage == 0:
            print(f"{col}: no missing values")
            continue

        # Non-numeric: use mode
        if not is_numeric_dtype(df_out[col].dtype):
            mode_vals = df_out[col].mode()
            if len(mode_vals) > 0:
                fill_val = mode_vals.iloc[0]
                print(f"{col}: {missing_percentage:.2%} missing (non-numeric) — mode imputation")
                df_out[col] = df_out[col].fillna(fill_val)
            else:
                print(f"{col}: {missing_percentage:.2%} missing (non-numeric) — no mode available, leaving NA")
            continue

        # Numeric: compute diagnostics and apply chosen imputation
        metrics = compute_imputation_decision(df_out[col], threshold=threshold)
        strategy = metrics["strategy"]
        mm_dist = metrics["mean_median_distance"]
        skew = metrics["skewness"]
        out_pct = metrics["outlier_percentage"]

        if strategy == "mean":
            print(f"{col}: {missing_percentage:.2%} missing — mean imputation (mean-median {mm_dist:.2%}, skew {skew:.3f}, outliers {out_pct:.2%})")
            df_out = impute_with_mean(df_out, col)
        else:
            print(f"{col}: {missing_percentage:.2%} missing — median imputation (mean-median {mm_dist:.2%}, skew {skew:.3f}, outliers {out_pct:.2%})")
            df_out = impute_with_median(df_out, col)
        print()

    return df_out


# ---------------------------------------------------------------------------
# Outlier utilities
# ---------------------------------------------------------------------------
@mcp.tool()
def outliers_cap(df: pd.DataFrame, cap_percentiles: tuple = (0.05, 0.95)) -> pd.DataFrame:
    """Cap numeric columns at the provided lower/upper percentiles (winsorization-like).

    Args:
        df: Input DataFrame.
        cap_percentiles: Tuple of (low_percentile, high_percentile) used for clipping.

    Returns:
        DataFrame copy with numeric columns clipped to the specified percentiles.
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    low_p, high_p = cap_percentiles
    for col in numeric_cols:
        s = df_out[col].dropna()
        if s.empty:
            continue
        lower_cap = s.quantile(low_p)
        upper_cap = s.quantile(high_p)
        df_out[col] = df_out[col].clip(lower=lower_cap, upper=upper_cap)
    return df_out


@mcp.tool()
def outliers_remove(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Remove rows that are IQR-outliers in any numeric column.

    Args:
        df: Input DataFrame.
        iqr_multiplier: Multiplier for IQR to define outlier bounds (default 1.5).

    Returns:
        Filtered DataFrame with rows removed where any numeric value falls outside the IQR bounds.
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_out
    mask = pd.Series(True, index=df_out.index)
    for col in numeric_cols:
        s = df_out[col].dropna()
        if s.empty:
            continue
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        mask &= ((df_out[col].isnull()) | ((df_out[col] >= lower) & (df_out[col] <= upper)))
    return df_out[mask].copy()


@mcp.tool()
def outliers_impute(df: pd.DataFrame, strategy: str = 'median', iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Impute outlier values detected by IQR with a robust statistic.

    For each numeric column, values outside [Q1 - m*IQR, Q3 + m*IQR] are replaced
    with the chosen statistic computed on non-outliers.

    Args:
        df: Input DataFrame.
        strategy: 'median' or 'mean' to replace outliers.
        iqr_multiplier: Multiplier for IQR to define outlier bounds.

    Returns:
        DataFrame copy with extreme values replaced by the robust statistic.
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        s = df_out[col]
        non_null = s.dropna()
        if non_null.empty:
            continue
        Q1 = non_null.quantile(0.25)
        Q3 = non_null.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        non_outliers = non_null[(non_null >= lower) & (non_null <= upper)]
        if non_outliers.empty:
            # fallback to column median/mean if everything is extreme
            fill_val = non_null.median() if strategy == 'median' else non_null.mean()
        else:
            fill_val = non_outliers.median() if strategy == 'median' else non_outliers.mean()
        mask_outlier = (s < lower) | (s > upper)
        df_out.loc[mask_outlier, col] = fill_val
    return df_out


@mcp.tool()
def outliers_robust_scale(df: pd.DataFrame, with_centering: bool = True, with_scaling: bool = True) -> pd.DataFrame:
    """Apply RobustScaler to numeric columns and return a new DataFrame.

    Uses median and IQR for scaling so outliers have reduced influence.

    Args:
        df: Input DataFrame.
        with_centering: If True, center data before scaling.
        with_scaling: If True, scale data to IQR-based range.

    Returns:
        DataFrame copy with numeric columns robustly scaled.
    """
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_out
    scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
    scaled = scaler.fit_transform(df_out[numeric_cols].astype(float))
    df_out[numeric_cols] = pd.DataFrame(scaled, index=df_out.index, columns=numeric_cols)
    return df_out


if __name__ == "__main__":
    mcp.run()
