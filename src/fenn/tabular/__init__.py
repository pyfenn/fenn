from typing import Optional

import numpy as np
import pandas as pd


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-shot overview combining shape, dtypes, basic stats,
    missing value counts, and cardinality info for categorical columns.
    """
    result = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "missing": df.isnull().sum(),
            "missing_%": (df.isnull().sum() / len(df) * 100).round(2),
            "unique": df.nunique(),
        }
    )
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return result


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact report of missing values per column, percentage,
    and flags for all-null or almost-all-null columns.
    """
    missing = df.isnull().sum()
    percent = (missing / len(df) * 100).round(2)
    report = pd.DataFrame(
        {
            "missing": missing,
            "missing_%": percent,
            "all_null": missing == len(df),
            "almost_null": percent > 90,
        }
    )
    return report[report["missing"] > 0]


def numeric_profile(
    df: pd.DataFrame, clip_quantile: Optional[float] = None
) -> pd.DataFrame:
    """
    Describe numeric columns only (min, max, mean, std, quantiles)
    with optional clipping of extreme quantiles.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if clip_quantile is not None:
        lower = numeric_df.quantile(clip_quantile)
        upper = numeric_df.quantile(1 - clip_quantile)
        numeric_df = numeric_df.clip(lower=lower, upper=upper, axis=1)
    return numeric_df.describe().T


def quick_sample(
    df: pd.DataFrame,
    n: int = 5,
    columns: Optional[list] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper around head/random sampling,
    with optional column subset and seed.
    """
    subset = df[columns] if columns else df
    return subset.sample(n=min(n, len(subset)), random_state=seed)


def unique_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Show number of unique values per column and, for low-cardinality
    columns, a small frequency table.
    """
    result = pd.DataFrame(
        {
            "unique_count": df.nunique(),
            "unique_%": (df.nunique() / len(df) * 100).round(2),
        }
    )
    return result


def corr_overview(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compute correlations between numeric columns and return
    the strongest pairs as a tidy table.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    # Convert to tidy format
    pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = pd.Index(["feature_1", "feature_2", "correlation"])
    pairs["abs_correlation"] = pairs["correlation"].abs()
    pairs = pairs.dropna(subset=["correlation"])
    return pairs.sort_values("abs_correlation", ascending=False).head(top_n)


def array_summary(arr: np.ndarray) -> pd.DataFrame:
    """
    NumPy-oriented helper for shape, dtype, basic stats,
    and NaN checks on ndarray.
    """
    flat = arr.flatten()
    return pd.DataFrame(
        [
            {
                "shape": arr.shape,
                "dtype": arr.dtype,
                "size": arr.size,
                "mean": float(np.nanmean(flat)),
                "std": float(np.nanstd(flat)),
                "min": float(np.nanmin(flat)),
                "max": float(np.nanmax(flat)),
                "nan_count": int(np.isnan(flat).sum()),
                "nan_%": round(float(np.isnan(flat).mean() * 100), 2),
            }
        ]
    )
