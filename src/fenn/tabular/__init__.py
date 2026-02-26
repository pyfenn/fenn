import pandas as pd
import numpy as np
from typing import Optional

def summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-shot overview combining shape, dtypes, basic stats,
    missing value counts, and cardinality info for categorical columns.
    """
    result = pd.DataFrame({
        "dtype": df.dtypes,
        "missing": df.isnull().sum(),
        "missing_%": (df.isnull().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
    })
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return result


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact report of missing values per column, percentage,
    and flags for all-null or almost-all-null columns.
    """
    missing = df.isnull().sum()
    percent = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({
        "missing": missing,
        "missing_%": percent,
        "all_null": missing == len(df),
        "almost_null": percent > 90,
    })
    return report[report["missing"] > 0]


def numeric_profile(
    df: pd.DataFrame,
    clip_quantile: Optional[float] = None
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