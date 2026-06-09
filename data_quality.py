"""
Enterprise-grade data quality assessment and cleaning utilities.
Provides profiling, scoring, outlier detection, and intelligent imputation.
"""

import pandas as pd
import numpy as np
import re
import html
from typing import Dict, List, Tuple, Optional, Any
import warnings

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class DataQualityProfiler:
    """Comprehensive data quality assessment and profiling."""

    @staticmethod
    def profile_dataframe(df: pd.DataFrame, sample_size: int = 10000) -> Dict[str, Any]:
        """Generate comprehensive data quality profile."""
        sample_df = df.sample(min(len(df), sample_size), random_state=42) if len(df) > sample_size else df
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_cells": len(df) * len(df.columns),
            "duplicate_rows": len(df) - len(df.drop_duplicates()),
            "duplicate_rows_pct": (len(df) - len(df.drop_duplicates())) / len(df) * 100 if len(df) > 0 else 0,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "completeness_score": DataQualityProfiler.calculate_completeness(df),
            "consistency_score": DataQualityProfiler.calculate_consistency(df),
            "validity_score": DataQualityProfiler.calculate_validity(df),
            "unique_columns": len(df.columns) - len([col for col in df.columns if df[col].nunique(dropna=True) <= 1]),
            "missing_columns": len([col for col in df.columns if df[col].isna().all()]),
            "column_types": df.dtypes.value_counts().to_dict(),
            "null_distribution": df.isnull().sum().to_dict(),
        }

    @staticmethod
    def calculate_completeness(df: pd.DataFrame) -> float:
        """Calculate data completeness score (0-100)."""
        total_cells = len(df) * len(df.columns)
        non_null_cells = df.notna().sum().sum()
        return (non_null_cells / total_cells * 100) if total_cells > 0 else 0

    @staticmethod
    def calculate_consistency(df: pd.DataFrame) -> float:
        """Calculate data consistency score based on type uniformity."""
        consistency_scores = []
        for col in df.select_dtypes(include=['object']).columns:
            unique_types = df[col].dropna().apply(lambda x: type(x).__name__).nunique()
            consistency_scores.append(100 - (unique_types - 1) * 10)
        return np.mean(consistency_scores) if consistency_scores else 100

    @staticmethod
    def calculate_validity(df: pd.DataFrame) -> float:
        """Calculate data validity score based on type appropriateness."""
        validity_scores = []
        for col in df.columns:
            valid_count = df[col].notna().sum()
            total_count = len(df)
            validity_scores.append((valid_count / total_count * 100) if total_count > 0 else 0)
        return np.mean(validity_scores) if validity_scores else 100

    @staticmethod
    def calculate_overall_quality_score(profile: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        weights = {
            "completeness_score": 0.35,
            "consistency_score": 0.30,
            "validity_score": 0.25,
            "uniqueness_score": 0.10,
        }
        
        uniqueness_score = min(100, (profile.get("unique_columns", 0) / max(profile.get("total_columns", 1), 1)) * 100)
        
        overall = (
            profile.get("completeness_score", 0) * weights["completeness_score"] +
            profile.get("consistency_score", 0) * weights["consistency_score"] +
            profile.get("validity_score", 0) * weights["validity_score"] +
            uniqueness_score * weights["uniqueness_score"]
        )
        
        return min(100, max(0, overall))


class AdvancedDataTypeDetector:
    """Detect and classify advanced data types."""

    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    PHONE_PATTERN = r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}$'
    URL_PATTERN = r'https?://[^\s]+'
    POSTAL_CODE_PATTERN = r'^\d{5}(-\d{4})?$|^[A-Z]{1,2}\d{1,2}[A-Z]?[\s]?\d[A-Z]{2}$'
    UUID_PATTERN = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    
    @staticmethod
    def detect_emails(series: pd.Series, threshold: float = 0.7) -> bool:
        """Detect if column contains email addresses."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        matches = non_null.astype(str).str.match(AdvancedDataTypeDetector.EMAIL_PATTERN).sum()
        return (matches / len(non_null)) >= threshold

    @staticmethod
    def detect_phone_numbers(series: pd.Series, threshold: float = 0.7) -> bool:
        """Detect if column contains phone numbers."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        matches = non_null.astype(str).str.match(AdvancedDataTypeDetector.PHONE_PATTERN).sum()
        return (matches / len(non_null)) >= threshold

    @staticmethod
    def detect_urls(series: pd.Series, threshold: float = 0.7) -> bool:
        """Detect if column contains URLs."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        matches = non_null.astype(str).str.contains(AdvancedDataTypeDetector.URL_PATTERN, regex=True).sum()
        return (matches / len(non_null)) >= threshold

    @staticmethod
    def detect_postal_codes(series: pd.Series, threshold: float = 0.7) -> bool:
        """Detect if column contains postal codes."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        matches = non_null.astype(str).str.match(AdvancedDataTypeDetector.POSTAL_CODE_PATTERN).sum()
        return (matches / len(non_null)) >= threshold

    @staticmethod
    def detect_uuids(series: pd.Series, threshold: float = 0.7) -> bool:
        """Detect if column contains UUIDs."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        matches = non_null.astype(str).str.match(AdvancedDataTypeDetector.UUID_PATTERN).sum()
        return (matches / len(non_null)) >= threshold


class OutlierDetector:
    """Multiple outlier detection methods."""

    @staticmethod
    def iqr_method(series: pd.Series, multiplier: float = 1.5) -> np.ndarray:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (series < lower_bound) | (series > upper_bound)

    @staticmethod
    def zscore_method(series: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-Score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    @staticmethod
    def isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> np.ndarray:
        """Detect outliers using Isolation Forest."""
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not installed. Skipping Isolation Forest detection.")
            return np.zeros(len(df), dtype=bool)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.zeros(len(df), dtype=bool)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(df[numeric_cols].fillna(df[numeric_cols].mean()))
        return predictions == -1

    @staticmethod
    def local_outlier_factor(df: pd.DataFrame, n_neighbors: int = 20, threshold: float = 1.5) -> np.ndarray:
        """Detect outliers using Local Outlier Factor."""
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn not installed. Skipping LOF detection.")
            return np.zeros(len(df), dtype=bool)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return np.zeros(len(df), dtype=bool)
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof_scores = lof.fit_predict(df[numeric_cols].fillna(df[numeric_cols].mean()))
        return lof_scores == -1


class TextCleaningAdvanced:
    """Advanced text cleaning techniques."""

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Remove HTML tags and decode entities."""
        if not isinstance(text, str):
            return text
        text = re.sub(r'<[^>]+>', '', text)
        text = html.unescape(text)
        return text

    @staticmethod
    def remove_xml_tags(text: str) -> str:
        """Remove XML fragments."""
        if not isinstance(text, str):
            return text
        return re.sub(r'<\?xml[^>]*\?>', '', text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        if not isinstance(text, str):
            return text
        return re.sub(r'https?://[^\s]+', '', text)

    @staticmethod
    def remove_non_printable(text: str) -> str:
        """Remove non-printable characters."""
        if not isinstance(text, str):
            return text
        return ''.join(char for char in text if char.isprintable() or char in '\n\t\r')

    @staticmethod
    def remove_emoji(text: str) -> str:
        """Remove emoji characters."""
        if not isinstance(text, str):
            return text
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F"
            "\U0001F780-\U0001F7FF"
            "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA00-\U0001FA6F"
            "\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)


class ImputationStrategies:
    """Multiple imputation strategies for missing values."""

    @staticmethod
    def mean_imputation(series: pd.Series) -> pd.Series:
        """Impute missing values with mean."""
        numeric_series = pd.to_numeric(series, errors='coerce')
        return numeric_series.fillna(numeric_series.mean())

    @staticmethod
    def median_imputation(series: pd.Series) -> pd.Series:
        """Impute missing values with median."""
        numeric_series = pd.to_numeric(series, errors='coerce')
        return numeric_series.fillna(numeric_series.median())

    @staticmethod
    def mode_imputation(series: pd.Series) -> pd.Series:
        """Impute missing values with mode."""
        mode_value = series.mode()
        if len(mode_value) > 0:
            return series.fillna(mode_value[0])
        return series

    @staticmethod
    def forward_fill_imputation(series: pd.Series) -> pd.Series:
        """Impute missing values with forward fill."""
        return series.fillna(method='ffill').fillna(method='bfill')

    @staticmethod
    def interpolation_imputation(series: pd.Series, method: str = 'linear') -> pd.Series:
        """Impute missing values with interpolation."""
        numeric_series = pd.to_numeric(series, errors='coerce')
        return numeric_series.interpolate(method=method, fill_value='extrapolate')

    @staticmethod
    def group_based_imputation(df: pd.DataFrame, value_col: str, group_col: str, method: str = 'mean') -> pd.Series:
        """Impute missing values based on group statistics."""
        if method == 'mean':
            group_means = df.groupby(group_col)[value_col].transform('mean')
            return df[value_col].fillna(group_means)
        elif method == 'median':
            group_medians = df.groupby(group_col)[value_col].transform('median')
            return df[value_col].fillna(group_medians)
        elif method == 'mode':
            group_modes = df.groupby(group_col)[value_col].transform(lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
            return df[value_col].fillna(group_modes)
        return df[value_col]


class FuzzyDeduplication:
    """Fuzzy duplicate detection and deduplication."""

    @staticmethod
    def find_fuzzy_duplicates(series: pd.Series, threshold: float = 0.85) -> List[List[int]]:
        """Find fuzzy duplicate groups using RapidFuzz."""
        if not RAPIDFUZZ_AVAILABLE:
            warnings.warn("RapidFuzz not installed. Skipping fuzzy deduplication.")
            return []
        
        unique_values = series.dropna().unique()
        duplicate_groups = []
        processed = set()
        
        for idx, value in enumerate(unique_values):
            if idx in processed:
                continue
            
            matches = process.extract(str(value), unique_values, scorer=fuzz.token_set_ratio)
            group_indices = [i for i, (match, score) in enumerate(matches) if score >= (threshold * 100)]
            
            if len(group_indices) > 1:
                duplicate_groups.append(group_indices)
                processed.update(group_indices)
        
        return duplicate_groups

    @staticmethod
    def deduplicate_category(series: pd.Series, method: str = 'most_common') -> pd.Series:
        """Deduplicate categorical values."""
        if not RAPIDFUZZ_AVAILABLE:
            return series
        
        unique_vals = series.dropna().unique()
        mapping = {}
        
        for val in unique_vals:
            matches = process.extract(str(val), unique_vals, scorer=fuzz.token_set_ratio)
            best_match = max(matches, key=lambda x: x[1])[0]
            mapping[val] = best_match
        
        return series.map(mapping)


class DataQualityReport:
    """Generate comprehensive data quality reports."""

    @staticmethod
    def generate_report(df_before: pd.DataFrame, df_after: pd.DataFrame, cleaning_log: Dict[str, Any]) -> str:
        """Generate comprehensive before/after quality report."""
        profile_before = DataQualityProfiler.profile_dataframe(df_before)
        profile_after = DataQualityProfiler.profile_dataframe(df_after)
        
        quality_before = DataQualityProfiler.calculate_overall_quality_score(profile_before)
        quality_after = DataQualityProfiler.calculate_overall_quality_score(profile_after)
        
        report = [
            "=" * 80,
            "DATA QUALITY REPORT - ENTERPRISE GRADE CLEANING",
            "=" * 80,
            "",
            "BEFORE vs AFTER COMPARISON",
            "-" * 80,
            f"Rows: {profile_before['total_rows']} → {profile_after['total_rows']}",
            f"Columns: {profile_before['total_columns']} → {profile_after['total_columns']}",
            f"Duplicate Rows: {profile_before['duplicate_rows']} → {profile_after['duplicate_rows']}",
            f"Missing Cells: {df_before.isna().sum().sum()} → {df_after.isna().sum().sum()}",
            "",
            "QUALITY SCORES",
            "-" * 80,
            f"Overall Quality Score: {quality_before:.1f} → {quality_after:.1f}",
            f"Completeness: {profile_before['completeness_score']:.1f}% → {profile_after['completeness_score']:.1f}%",
            f"Consistency: {profile_before['consistency_score']:.1f}% → {profile_after['consistency_score']:.1f}%",
            f"Validity: {profile_before['validity_score']:.1f}% → {profile_after['validity_score']:.1f}%",
            "",
            "MEMORY USAGE",
            "-" * 80,
            f"Before: {profile_before['memory_usage_mb']:.2f} MB",
            f"After: {profile_after['memory_usage_mb']:.2f} MB",
            "",
            "CLEANING ACTIONS PERFORMED",
            "-" * 80,
        ]
        
        for action, count in cleaning_log.items():
            report.append(f"{action}: {count}")
        
        report.append("=" * 80)
        
        return "\n".join(report)
