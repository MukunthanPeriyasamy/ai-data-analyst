# AI Data Analyst — Notebook Function Reference

This README documents the helper functions and recommended workflow used in the notebook `test.ipynb` for filtering columns, handling missing values, detecting/handling outliers, and plotting.

**Overview**
- **Purpose:** Provide concise, reproducible utilities to prepare numeric data (`df2`) for modeling and analysis.
- **Recommended flow:** `df -> filter_function(df) -> handling_missing_values(df2) -> handle_outliers_after_imputation(df_clean)`

**Function Reference**

- **filter_function(df):** Returns a copy of `df` containing only numeric (non-object) columns.
  - Parameters: `df` (pandas.DataFrame)
  - Returns: DataFrame (numeric columns only, explicit copy)
  - Notes: Uses `select_dtypes(exclude=['object']).copy()` to avoid views and SettingWithCopyWarning.

- **filling_missing_values_mean(col) / filling_missing_values_median(col):** Small helpers used in the notebook to impute a single column on the `df2` variable.
  - Parameters: `col` (string) — column name in `df2`.
  - Behavior: Compute mean/median and assign filled series back to `df2[col]` using non-inplace assignment to avoid warnings.
  - Recommendation: Convert these to pure functions (accept and return Series) for reuse.

- **choose_imputation_strategy(df):** Heuristic to choose mean vs median imputation per numeric feature.
  - Parameters: `df` (pandas.DataFrame) — expected numeric-only (output of `filter_function`).
  - Metrics computed per column: skew, mean, median, Q1, Q3, IQR, non-null count, outlier ratio.
  - Decision: Use mean when abs(skew) <= 0.5, mean/median relative difference <= 0.1, and outlier ratio <= 0.05; otherwise use median.
  - Side effects: Calls the `filling_missing_values_*` helpers in the notebook (prints decisions). Consider returning a modified DataFrame instead of mutating globals.

- **handling_missing_values(df):** Returns a new DataFrame after handling missing values using simple rules.
  - Parameters: `df` (pandas.DataFrame)
  - Rules:
    - >30% missing: drop the column.
    - 5–30% missing: median imputation.
    - 0–5% missing: mean imputation.
    - 0% missing: leave unchanged.
  - Returns: `df_out` — a copy with columns dropped or imputed.
  - Notes: Uses non-inplace assignment and prints actions; safe against division-by-zero.

- **handling_outliers(df):** Identifies outliers using IQR (1.5 * IQR), logs a per-column report to `outlier_report.txt`, and removes extreme outlier rows while preserving nulls.
  - Parameters: `df` (pandas.DataFrame)
  - Behavior: For each column compute Q1/Q3/IQR, count outliers, append results to `outlier_report.txt`, then filter rows to remove extreme IQR outliers.
  - Returns: Filtered DataFrame with outliers removed.
  - Notes: Checks non-null counts to avoid ZeroDivisionError.

- **handle_outliers_after_imputation(df, method='cap', iqr_multiplier=1.5, cap_percentiles=(0.01,0.99)):** Post-imputation outlier handler.
  - Parameters:
    - `df` (pandas.DataFrame): DataFrame that has been imputed/cleaned.
    - `method` (str): `'cap'` (default) or `'remove'`.
    - `iqr_multiplier` (float): multiplier for IQR when `method='remove'`.
    - `cap_percentiles` (tuple): (low_percentile, high_percentile) used when `method='cap'`.
  - Behavior:
    - `'cap'`: clip numeric columns to the specified percentiles (limits extremes without dropping rows).
    - `'remove'`: remove any row that is an IQR outlier (abs > `iqr_multiplier * IQR`) in any numeric column.
  - Returns: New DataFrame with outliers handled (no in-place mutation).

**Plotting / Visualization**
- Scatter plot highlighting outliers (example):
  1. Compute Q1, Q3, IQR for target column.
  2. Build `normal` and `outliers` masks/series (dropna for computation but keep original indices).
  3. Plot `sns.scatterplot(x=normal.index, y=normal.values, color='C0')` and `sns.scatterplot(x=outliers.index, y=outliers.values, color='C3')`.
  4. Add labels, title, and legend.

**Best Practices & Suggestions**
- Prefer pure functions (accept and return DataFrames/Series) rather than mutating notebook-global variables — improves reusability and testability.
- Parameterize thresholds (missing % cutoffs, IQR multiplier, cap percentiles) and document choices when running experiments.
- Persist `outlier_report.txt` with timestamps or use dated filenames to avoid ambiguous appends.
- When creating final datasets for modeling, chain functions in the recommended flow and save result to a single variable (e.g., `df_clean`).

**Example workflow (notebook)**
```python
# 1. keep numeric columns
df2 = filter_function(df)

# 2. handle missing values (returns new DataFrame)
formatted_df = handling_missing_values(df2)

# 3. cap extremes (or remove)
final_df = handle_outliers_after_imputation(formatted_df, method='cap')
```

---

If you want, I can also export this README content to a plain text file with a timestamped copy, or add a short example notebook cell that demonstrates the recommended flow and prints before/after stats.