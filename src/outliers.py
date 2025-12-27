import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def outliers_cap(df, cap_percentiles=(0.05, 0.95)) -> pd.DataFrame:
    """Cap numeric columns at the provided lower/upper percentiles (winsorization-like).
    Returns a new DataFrame with capped numeric values."""
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

def outliers_remove(df, iqr_multiplier=1.5) -> pd.DataFrame:
    """Remove rows that are IQR-outliers in any numeric column."""
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

def outliers_impute(df, strategy='median', iqr_multiplier=1.5) -> pd.DataFrame:
    """Impute outlier values detected by IQR with a robust statistic (median by default).
    For each numeric column, values outside [Q1 - m*IQR, Q3 + m*IQR] are replaced with the chosen statistic computed on non-outliers."""
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

def outliers_robust_scale(df, with_centering=True, with_scaling=True) -> pd.DataFrame:
    """Apply RobustScaler to numeric columns and return a new DataFrame.
    Uses median and IQR for scaling so outliers have reduced influence."""
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df_out
    scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling)
    scaled = scaler.fit_transform(df_out[numeric_cols].astype(float))
    df_out[numeric_cols] = pd.DataFrame(scaled, index=df_out.index, columns=numeric_cols)
    return df_out
