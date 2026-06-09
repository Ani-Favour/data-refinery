# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import unicodedata
import json
from datetime import datetime
from io import BytesIO, StringIO
import warnings
warnings.filterwarnings('ignore')

try:
    from data_quality import (
        DataQualityProfiler,
        AdvancedDataTypeDetector,
        OutlierDetector,
        TextCleaningAdvanced,
        ImputationStrategies,
        FuzzyDeduplication,
        DataQualityReport,
    )
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False
    st.warning("Advanced data quality features not available. Install required packages.")

# --- Paths ---
RAW_FOLDER = "raw_data"
CLEANED_FOLDER = "cleaned_data"
REPORTS_FOLDER = "reports"
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# ---------- Cleaning helpers ----------
def sanitize_filename(name):
    return re.sub(r"[<>:\"/\\|?*]+", "_", name).strip()


def make_unique_column_names(columns):
    counts = {}
    unique_names = []
    for col in columns:
        base = col if col else "column"
        if base in counts:
            counts[base] += 1
            unique_names.append(f"{base}_{counts[base]}")
        else:
            counts[base] = 0
            unique_names.append(base)
    return unique_names


def clean_column_names(df):
    cleaned = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    df.columns = make_unique_column_names(cleaned)
    return df


def standardize_missing_values(df):
    null_values = [
        "na", "n/a", "nan", "none", "null", "undefined", "missing", "not available", "unknown",
        "n.a.", "na.", "n/a.", "--", "---", ".", "..", "...", "no data", "no value",
        "unknown", "unknown value", "not provided", "not applicable", "tbd", "todo",
    ]
    cleaned = df.replace(null_values, np.nan, regex=False)
    cleaned = cleaned.replace(
        r"^(?:\s+|-|--|na|n/a|not available|unknown|missing|none|null|undefined|n\.a\.|na\.|n/a\.|no data|no value|not provided|not applicable|tbd|todo)$",
        np.nan, regex=True
    )
    return cleaned


def strip_string_columns(df):
    """Enhanced string stripping with text cleaning."""
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        # Remove HTML/XML tags if advanced features available
        if ADVANCED_FEATURES:
            df[col] = df[col].apply(lambda x: TextCleaningAdvanced.remove_html_tags(x) if isinstance(x, str) else x)
            df[col] = df[col].apply(lambda x: TextCleaningAdvanced.remove_non_printable(x) if isinstance(x, str) else x)
        
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\t\n\r]+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df.loc[df[col] == "", col] = np.nan
    return df


def normalize_unicode_text(df):
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        def normalize_value(value):
            if pd.isna(value):
                return value
            text = str(value)
            normalized = unicodedata.normalize("NFKC", text)
            normalized = normalized.encode("ascii", "ignore").decode("ascii")
            return normalized

        df[col] = df[col].map(normalize_value)
        df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        df.loc[df[col] == "", col] = np.nan
    return df


BOOLEAN_MAP = {
    "yes": True,
    "y": True,
    "true": True,
    "t": True,
    "1": True,
    "on": True,
    "enabled": True,
    "no": False,
    "n": False,
    "false": False,
    "f": False,
    "0": False,
    "off": False,
    "disabled": False,
}


def normalize_boolean_columns(df, threshold=0.65):
    converted = []
    for col in df.select_dtypes(include=["object"]).columns:
        cleaned = df[col].astype(str).str.strip().str.lower()
        mapped = cleaned.map(BOOLEAN_MAP).where(cleaned.isin(BOOLEAN_MAP.keys()), np.nan)
        non_null = df[col].notna().sum()
        valid = mapped.notna().sum()
        if non_null > 0 and valid / non_null >= threshold:
            df[col] = mapped
            converted.append(col)
    return df, converted


def convert_numeric_columns(df, threshold=0.65):
    converted = []
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"[,¥$£€]", "", regex=True)
                .str.replace(r"[\(\)]", "", regex=True)
                .str.replace(r"[%]", "", regex=True)
                .str.replace(r"[^0-9.\-]", "", regex=True)
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            non_null = df[col].notna().sum()
            valid = numeric.notna().sum()
            if non_null > 0 and valid / non_null >= threshold:
                df[col] = numeric
                converted.append(col)
    return df, converted


def convert_date_columns(df, threshold=0.65):
    """Enhanced date conversion with multiple format detection."""
    converted = {}
    for col in df.columns:
        if df[col].dtype == object or np.issubdtype(df[col].dtype, np.number):
            # Try multiple date formats
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, utc=False)
            
            if parsed.notna().sum() / max(df[col].notna().sum(), 1) < threshold:
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, dayfirst=True)
            
            if parsed.notna().sum() / max(df[col].notna().sum(), 1) < threshold:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce", format='%Y-%m-%d')
                except:
                    pass
            
            non_null = df[col].notna().sum()
            valid = parsed.notna().sum()
            if non_null > 0 and valid / non_null >= threshold:
                df[col] = parsed
                converted[col] = non_null - valid
    return df, converted


def drop_empty_columns(df):
    before = len(df.columns)
    df = df.dropna(axis=1, how="all")
    return df, before - len(df.columns)


def drop_constant_columns(df):
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    df = df.drop(columns=constant_cols)
    return df, len(constant_cols)


def read_data_file(path: str) -> pd.DataFrame:
    """Read data from multiple file formats with automatic encoding detection."""
    try:
        if path.lower().endswith(".csv"):
            try:
                return pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="latin-1")
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="iso-8859-1")
        
        elif path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        
        elif path.lower().endswith(".json"):
            return pd.read_json(path)
        
        elif path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        
        elif path.lower().endswith(".feather"):
            return pd.read_feather(path)
        
        elif path.lower().endswith((".txt", ".tsv")):
            return pd.read_csv(path, sep="\t", encoding="utf-8")
        
        else:
            raise ValueError(f"Unsupported file type: {path}")
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")


def clean_dataframe(df, threshold=0.65, enable_advanced=True):
    """Comprehensive enterprise-grade data cleaning pipeline."""
    cleaning_log = {}
    df_original = df.copy()
    
    df = clean_column_names(df)
    df = standardize_missing_values(df)
    df = strip_string_columns(df)
    df = normalize_unicode_text(df)
    
    # Detect and classify advanced data types if enabled
    advanced_types = {}
    if ADVANCED_FEATURES and enable_advanced:
        for col in df.select_dtypes(include=['object']).columns:
            if AdvancedDataTypeDetector.detect_emails(df[col]):
                advanced_types[col] = 'email'
            elif AdvancedDataTypeDetector.detect_phone_numbers(df[col]):
                advanced_types[col] = 'phone'
            elif AdvancedDataTypeDetector.detect_urls(df[col]):
                advanced_types[col] = 'url'
            elif AdvancedDataTypeDetector.detect_postal_codes(df[col]):
                advanced_types[col] = 'postal_code'
            elif AdvancedDataTypeDetector.detect_uuids(df[col]):
                advanced_types[col] = 'uuid'
    
    df, bool_cols = normalize_boolean_columns(df, threshold)
    df, numeric_cols = convert_numeric_columns(df, threshold)
    df, date_cols = convert_date_columns(df, threshold)
    df, dropped_empty = drop_empty_columns(df)
    df, dropped_constant = drop_constant_columns(df)
    df, removed_dupes = remove_duplicates(df)
    
    return df, bool_cols, numeric_cols, date_cols, dropped_empty, dropped_constant, removed_dupes, advanced_types


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df)


def create_report_text(stats):
    lines = ["=== GENERAL DATA CLEANING REPORT ===", ""]
    for key, value in stats.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


# ---------- App UI ----------
st.set_page_config(page_title="DataRefinery", layout="wide")

theme_choice = st.sidebar.selectbox("UI theme", ["Light", "Dark"], index=0)
threshold = st.sidebar.slider("Auto-convert confidence (%)", 50, 95, 65, step=5)

theme_class = "theme-dark" if theme_choice == "Dark" else "theme-light"

st.markdown(
    f"""
    <style>
    :root {{
        --bg: {'#0f172a' if theme_choice == 'Dark' else '#f8fafc'};
        --surface: {'#111827' if theme_choice == 'Dark' else '#ffffff'};
        --surface-alt: {'#1f2937' if theme_choice == 'Dark' else '#f8fafc'};
        --text: {'#f8fafc' if theme_choice == 'Dark' else '#0f172a'};
        --muted: {'#cbd5e1' if theme_choice == 'Dark' else '#475569'};
        --border: {'rgba(148, 163, 184, 0.12)' if theme_choice == 'Dark' else 'rgba(15, 23, 42, 0.08)'};
        --accent: {'#38bdf8' if theme_choice == 'Dark' else '#0b69ff'};
        --icon-bg: {'rgba(56, 189, 248, 0.16)' if theme_choice == 'Dark' else 'rgba(11, 105, 255, 0.12)'};
        --button-bg: {'#0ea5e9' if theme_choice == 'Dark' else '#0b69ff'};
        --button-text: #ffffff;
    }}

    body {{ background: var(--bg); color: var(--text); }}
    .css-18e3th9 {{ background-color: var(--bg) !important; }}
    .css-1oe6wy5 {{ background-color: var(--bg) !important; }}
    .css-1d391kg {{ color: var(--text) !important; }}
    .main-header {{ background: linear-gradient(90deg, rgba(11,111,177,0.95), rgba(6,66,132,0.95)); padding: 26px; border-radius: 18px; color: white; margin-bottom: 22px; box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18); }}
    .main-header h1 {{ margin: 0; font-size: 2.6rem; }}
    .main-header p {{ margin: 6px 0 0; font-size: 1.05rem; opacity: 0.94; line-height: 1.65; }}
    .section-card {{ background: var(--surface); color: var(--text); padding: 22px; border-radius: 18px; border: 1px solid var(--border); box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08); margin-bottom: 18px; }}
    .section-card h3 {{ margin: 0 0 10px; font-size: 1.15rem; }}
    .section-card p {{ margin: 0; color: var(--muted); line-height: 1.7; }}
    .card-icon {{ width: 48px; height: 48px; display: inline-flex; align-items: center; justify-content: center; border-radius: 14px; margin-bottom: 14px; font-size: 1.35rem; background: var(--icon-bg); color: var(--accent); }}
    .stButton>button {{ background-color: var(--button-bg) !important; color: var(--button-text) !important; border-radius: 12px !important; padding: 0.85rem 1.4rem !important; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12) !important; }}
    .stSidebar {{ background: var(--surface-alt) !important; }}
    .sidebar .stTextInput>div>input, .sidebar .stSelectbox>div>div>div>div, .sidebar .stRadio>div>label {{ background: var(--surface) !important; color: var(--text) !important; }}
    .stDownloadButton>button {{ border-radius: 12px !important; }}
    </style>
    <script>
    const body = document.querySelector('body');
    if (body) {{ body.classList.remove('theme-light', 'theme-dark'); body.classList.add('{theme_class}'); }}
    </script>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='main-header'><h1>DataRefinery</h1><p>Clean, standardize, and export data faster with an intuitive one-click workflow for CSV and Excel datasets.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("DataRefinery")
    st.write("Professional data cleaning for CSV/XLSX datasets")
    st.markdown("**Theme:**")
    st.caption("Choose your display mode for the dashboard.")
    st.markdown("---")
    st.markdown("**Quick build workflow**")
    st.markdown(
        "1. Upload files or place them into `raw_data/`\n"
        "2. Select input mode\n"
        "3. Click **Run Data Cleaner**"
    )
    st.markdown("---")
    st.markdown("**Sample data available**")
    st.markdown("- `microfinance_sample.csv`\n- `general_sample.csv`\n- `sample_data/` folder for examples")
    st.markdown("---")
    st.text(f"Auto-convert threshold: {threshold}%")
    st.write("This tool standardizes columns, normalizes missing values, converts data types, and removes duplicate rows.")
    st.markdown("---")
    raw_files = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith((".csv", ".xlsx", ".xls"))]
    if raw_files:
        st.markdown("**Files detected in raw_data:**")
        for fname in raw_files[:8]:
            st.write(f"• {fname}")
        if len(raw_files) > 8:
            st.write(f"...and {len(raw_files) - 8} more files")
    else:
        st.info("Drop raw CSV/XLSX files into raw_data/ for batch processing.")

input_mode = st.radio("Input mode", ("Upload files (recommended)", "Process files in raw_data folder"))

st.markdown("---")
cols = st.columns(3)
with cols[0]:
    st.markdown(
        "<div class='section-card'><div class='card-icon'>⚡</div><h3>Fast data readiness</h3><p>Auto-detect dirty inputs and convert inconsistent values into clean, analysis-ready columns.</p></div>",
        unsafe_allow_html=True,
    )
with cols[1]:
    st.markdown(
        "<div class='section-card'><div class='card-icon'>🧹</div><h3>Smart cleaning engine</h3><p>Standardizes missing values, trims whitespace, converts numeric fields, and parses dates intelligently.</p></div>",
        unsafe_allow_html=True,
    )
with cols[2]:
    st.markdown(
        "<div class='section-card'><div class='card-icon'>📥</div><h3>One-click export</h3><p>Save cleaned CSV output automatically and download reports for audit-ready transparency.</p></div>",
        unsafe_allow_html=True,
    )

uploaded_files = None
if input_mode == "Upload files (recommended)":
    uploaded_files = st.file_uploader(
        "Upload data files (CSV, XLSX, JSON, Parquet, TSV, TXT)",
        type=["csv", "xlsx", "xls", "json", "parquet", "feather", "txt", "tsv"],
        accept_multiple_files=True,
    )

# Advanced options
st.markdown("---")
st.subheader("⚙️ Advanced Options")
col1, col2, col3 = st.columns(3)
with col1:
    enable_quality_scoring = st.checkbox("Calculate quality scores", value=True, help="Generate data quality metrics")
with col2:
    detect_outliers = st.checkbox("Detect outliers", value=False, help="Use IQR method to flag outliers")
with col3:
    outlier_method = st.selectbox("Outlier method", ["IQR", "Z-Score", "Isolation Forest"], disabled=not detect_outliers) if ADVANCED_FEATURES else "IQR"

run = st.button("🚀 Run Data Cleaner (enterprise grade)")

if run:
    summary_rows = []
    overall_stats = {
        "Files processed": 0,
        "Duplicate rows removed": 0,
        "Columns converted to boolean": 0,
        "Columns converted to numeric": 0,
        "Columns converted to dates": 0,
        "Advanced types detected": 0,
        "Columns dropped (empty)": 0,
        "Columns dropped (constant)": 0,
        "Files saved": 0,
    }
    
    quality_scores_before = []
    quality_scores_after = []

    files_to_process = []
    if input_mode == "Upload files (recommended)":
        if not uploaded_files:
            st.error("Please upload at least one file or switch to folder mode.")
        else:
            for uploaded in uploaded_files:
                try:
                    if uploaded.name.lower().endswith(".csv"):
                        df = pd.read_csv(uploaded)
                    elif uploaded.name.lower().endswith((".xlsx", ".xls")):
                        df = pd.read_excel(uploaded)
                    elif uploaded.name.lower().endswith(".json"):
                        df = pd.read_json(uploaded)
                    elif uploaded.name.lower().endswith(".parquet"):
                        df = pd.read_parquet(uploaded)
                    elif uploaded.name.lower().endswith(".feather"):
                        df = pd.read_feather(uploaded)
                    else:
                        df = pd.read_csv(uploaded, sep="\t")
                    files_to_process.append((uploaded.name, df))
                except Exception as e:
                    st.error(f"Failed to read {uploaded.name}: {e}")
    else:
        raw_files = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith((".csv", ".xlsx", ".xls", ".json", ".parquet", ".feather", ".txt", ".tsv"))]
        if not raw_files:
            st.warning("No files found in the raw_data folder.")
        for fname in raw_files:
            path = os.path.join(RAW_FOLDER, fname)
            try:
                df = read_data_file(path)
                files_to_process.append((fname, df))
            except Exception as e:
                st.error(f"Failed to read {fname}: {e}")

    if files_to_process:
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        for i, (fname, df) in enumerate(files_to_process, start=1):
            status_placeholder.info(f"Processing {fname}... ({i}/{len(files_to_process)})")
            
            overall_stats["Files processed"] += 1
            rows_before = len(df)
            
            # Calculate quality score before cleaning
            quality_score_before = 0
            if ADVANCED_FEATURES and enable_quality_scoring:
                profile_before = DataQualityProfiler.profile_dataframe(df)
                quality_score_before = DataQualityProfiler.calculate_overall_quality_score(profile_before)
                quality_scores_before.append(quality_score_before)

            df, bool_cols, numeric_cols, date_cols, dropped_empty, dropped_constant, removed_dupes, advanced_types = clean_dataframe(df, threshold / 100)
            df = df.reset_index(drop=True)
            
            # Calculate quality score after cleaning
            quality_score_after = 0
            if ADVANCED_FEATURES and enable_quality_scoring:
                profile_after = DataQualityProfiler.profile_dataframe(df)
                quality_score_after = DataQualityProfiler.calculate_overall_quality_score(profile_after)
                quality_scores_after.append(quality_score_after)

            cleaned_name = f"cleaned_{sanitize_filename(fname.rsplit('.', 1)[0])}.csv"
            cleaned_path = os.path.join(CLEANED_FOLDER, cleaned_name)
            df.to_csv(cleaned_path, index=False)

            overall_stats["Duplicate rows removed"] += removed_dupes
            overall_stats["Columns converted to boolean"] += len(bool_cols)
            overall_stats["Columns converted to numeric"] += len(numeric_cols)
            overall_stats["Columns converted to dates"] += len(date_cols)
            overall_stats["Advanced types detected"] += len(advanced_types)
            overall_stats["Columns dropped (empty)"] += dropped_empty
            overall_stats["Columns dropped (constant)"] += dropped_constant
            overall_stats["Files saved"] += 1

            summary_rows.append(
                {
                    "file": fname,
                    "rows_before": rows_before,
                    "rows_after": len(df),
                    "quality_before": f"{quality_score_before:.1f}" if quality_score_before > 0 else "N/A",
                    "quality_after": f"{quality_score_after:.1f}" if quality_score_after > 0 else "N/A",
                    "bool_columns": ", ".join(bool_cols) if bool_cols else "-",
                    "numeric_columns": ", ".join(numeric_cols) if numeric_cols else "-",
                    "date_columns": ", ".join(date_cols.keys()) if date_cols else "-",
                    "advanced_types": ", ".join(set(advanced_types.values())) if advanced_types else "-",
                    "dropped_empty_columns": dropped_empty,
                    "dropped_constant_columns": dropped_constant,
                    "duplicates_removed": removed_dupes,
                    "cleaned_path": cleaned_path,
                }
            )
            progress_bar.progress(i / len(files_to_process))
        
        status_placeholder.success("✅ All files processed successfully!")
        st.markdown("---")

        # Display quality score improvements
        if quality_scores_before and enable_quality_scoring:
            quality_tabs = st.tabs(["📊 Quality Metrics", "📈 Summary", "📥 Downloads"])
            
            with quality_tabs[0]:
                metric_cols = st.columns(4)
                metric_cols[0].metric("Avg Quality (Before)", f"{np.mean(quality_scores_before):.1f}", help="Overall data quality score before cleaning")
                metric_cols[1].metric("Avg Quality (After)", f"{np.mean(quality_scores_after):.1f}", help="Overall data quality score after cleaning")
                improvement = np.mean(quality_scores_after) - np.mean(quality_scores_before)
                metric_cols[2].metric("Quality Improvement", f"{improvement:+.1f}", help="Change in quality score")
                metric_cols[3].metric("Files Cleaned", overall_stats["Files processed"])
            
            with quality_tabs[1]:
                metric_cols_summary = st.columns(4)
                metric_cols_summary[0].metric("Files processed", overall_stats["Files processed"])
                metric_cols_summary[1].metric("Duplicate rows removed", overall_stats["Duplicate rows removed"])
                metric_cols_summary[2].metric("Empty columns dropped", overall_stats["Columns dropped (empty)"])
                metric_cols_summary[3].metric("Constant columns dropped", overall_stats["Columns dropped (constant)"])
            
            with quality_tabs[2]:
                st.subheader("Summary (per file)")
                st.table(pd.DataFrame(summary_rows))
                
                st.subheader("Overall stats")
                st.json(overall_stats)
        
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Files processed", overall_stats["Files processed"])
            metric_cols[1].metric("Duplicate rows removed", overall_stats["Duplicate rows removed"])
            metric_cols[2].metric("Empty columns dropped", overall_stats["Columns dropped (empty)"])
            metric_cols[3].metric("Constant columns dropped", overall_stats["Columns dropped (constant)"])
            
            st.markdown("---")
            st.subheader("Summary (per file)")
            st.table(pd.DataFrame(summary_rows))
            
            st.subheader("Overall stats")
            st.json(overall_stats)

        st.markdown("---")
        st.markdown("### 📥 Download cleaned files")
        for row in summary_rows:
            with open(row["cleaned_path"], "rb") as f:
                data_bytes = f.read()
            st.download_button(
                label=f"Download {os.path.basename(row['cleaned_path'])}",
                data=data_bytes,
                file_name=os.path.basename(row["cleaned_path"]),
                mime="text/csv",
            )

        report_text = create_report_text(overall_stats)
        st.download_button(
            label="📄 Download cleaning report (txt)",
            data=report_text,
            file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )
    else:
        st.warning("No files were processed. Upload files or add data to the raw_data folder.")
