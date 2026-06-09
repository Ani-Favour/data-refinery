# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

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


def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def standardize_missing_values(df):
    null_values = ["", " ", "na", "n/a", "nan", "none", "null", "undefined"]
    return df.replace(null_values, np.nan)


def strip_string_columns(df):
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col] == "", col] = np.nan
    return df


def convert_numeric_columns(df):
    converted = []
    for col in df.columns:
        if df[col].dtype == object:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[^0-9.\-]", "", regex=True)
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            non_null = df[col].notna().sum()
            valid = numeric.notna().sum()
            if non_null > 0 and valid / non_null >= 0.65:
                df[col] = numeric
                converted.append(col)
    return df, converted


def convert_date_columns(df):
    converted = {}
    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            non_null = df[col].notna().sum()
            valid = parsed.notna().sum()
            if non_null > 0 and valid / non_null >= 0.65:
                df[col] = parsed
                converted[col] = non_null - valid
    return df, converted


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
st.set_page_config(page_title="General Data Cleaner", layout="wide")
st.title("General Data Cleaner — Single-Click Data Cleaning")

st.markdown(
    "This app cleans raw datasets automatically. Upload CSV/XLSX files or place them into `raw_data/`, then click the button to clean all files in one step."
)

input_mode = st.radio("Input mode", ("Upload files (recommended)", "Process files in raw_data folder"))

uploaded_files = None
if input_mode == "Upload files (recommended)":
    uploaded_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
    )

run = st.button("Run Data Cleaner (single click)")

if run:
    summary_rows = []
    overall_stats = {
        "Files processed": 0,
        "Duplicate rows removed": 0,
        "Columns converted to numeric": 0,
        "Columns converted to dates": 0,
        "Files saved": 0,
    }

    files_to_process = []
    if input_mode == "Upload files (recommended)":
        if not uploaded_files:
            st.error("Please upload at least one file or switch to folder mode.")
        else:
            for uploaded in uploaded_files:
                try:
                    if uploaded.name.lower().endswith(".csv"):
                        df = pd.read_csv(uploaded)
                    else:
                        df = pd.read_excel(uploaded)
                    files_to_process.append((uploaded.name, df))
                except Exception as e:
                    st.error(f"Failed to read {uploaded.name}: {e}")
    else:
        raw_files = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith((".csv", ".xlsx"))]
        if not raw_files:
            st.warning("No files found in the raw_data folder.")
        for fname in raw_files:
            path = os.path.join(RAW_FOLDER, fname)
            try:
                if fname.lower().endswith(".csv"):
                    df = pd.read_csv(path, encoding="latin-1")
                else:
                    df = pd.read_excel(path)
                files_to_process.append((fname, df))
            except Exception as e:
                st.error(f"Failed to read {fname}: {e}")

    if files_to_process:
        progress_bar = st.progress(0)
        for i, (fname, df) in enumerate(files_to_process, start=1):
            overall_stats["Files processed"] += 1
            rows_before = len(df)

            df = clean_column_names(df)
            df = standardize_missing_values(df)
            df = strip_string_columns(df)
            df, numeric_cols = convert_numeric_columns(df)
            df, date_cols = convert_date_columns(df)
            df, removed_dupes = remove_duplicates(df)
            df = df.reset_index(drop=True)

            cleaned_name = f"cleaned_{sanitize_filename(fname.rsplit('.', 1)[0])}.csv"
            cleaned_path = os.path.join(CLEANED_FOLDER, cleaned_name)
            df.to_csv(cleaned_path, index=False)

            overall_stats["Duplicate rows removed"] += removed_dupes
            overall_stats["Columns converted to numeric"] += len(numeric_cols)
            overall_stats["Columns converted to dates"] += len(date_cols)
            overall_stats["Files saved"] += 1

            summary_rows.append(
                {
                    "file": fname,
                    "rows_before": rows_before,
                    "rows_after": len(df),
                    "numeric_columns": ", ".join(numeric_cols) if numeric_cols else "-",
                    "date_columns": ", ".join(date_cols.keys()) if date_cols else "-",
                    "duplicates_removed": removed_dupes,
                    "cleaned_path": cleaned_path,
                }
            )

            progress_bar.progress(i / len(files_to_process))

        st.success("Cleaning finished ✅")
        st.subheader("Summary (per file)")
        st.table(pd.DataFrame(summary_rows))

        st.subheader("Overall stats")
        st.json(overall_stats)

        st.markdown("### Download cleaned files")
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
            label="Download cleaning report (txt)",
            data=report_text,
            file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )
    else:
        st.warning("No files were processed. Upload files or add data to the raw_data folder.")
