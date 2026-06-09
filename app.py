# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import traceback
from datetime import datetime
import io

# --- Paths (optional; used when processing folders) ---
RAW_FOLDER = "raw_data"
CLEANED_FOLDER = "cleaned_data"
REPORTS_FOLDER = "reports"
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# ---------- Cleaning helpers (same logic as your script) ----------
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def clean_customer_id(df):
    df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
    before = len(df)
    df = df.dropna(subset=['customer_id'])
    df['customer_id'] = df['customer_id'].astype(int)
    removed = before - len(df)
    return df, removed

def clean_loan_amount(df):
    if 'loan_amount' not in df.columns:
        return df
    df['loan_amount'] = df['loan_amount'].astype(str).str.replace(r"[^0-9.-]", "", regex=True)
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
    df.loc[df['loan_amount'] <= 0, 'loan_amount'] = np.nan
    return df

def clean_income(df):
    if 'customer_income' not in df.columns:
        return df
    df['customer_income'] = df['customer_income'].astype(str).str.replace(r"[^0-9.-]", "", regex=True)
    df['customer_income'] = pd.to_numeric(df['customer_income'], errors='coerce')
    df.loc[df['customer_income'] < 10000, 'customer_income'] = np.nan
    return df

def clean_loan_status(df):
    if 'loan_status' not in df.columns:
        return df
    valid_status = {
        "paid": "paid", "payed": "paid", "settled": "paid", "completed": "paid",
        "unpaid": "unpaid", "owing": "unpaid", "overdue": "unpaid", "pending": "unpaid"
    }
    df['loan_status'] = df['loan_status'].astype(str).str.strip().str.lower().map(valid_status)
    return df

def clean_dates(df):
    if 'repayment_date' not in df.columns:
        return df, 0
    removed = 0
    def fix_date(x):
        nonlocal removed
        d = pd.to_datetime(x, errors='coerce')
        if pd.isna(d):
            removed += 1
            return pd.NaT
        if d > pd.to_datetime("today"):
            removed += 1
            return pd.NaT
        return d
    df['repayment_date'] = df['repayment_date'].apply(fix_date)
    return df, removed

def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df)

def create_report_text(stats):
    lines = ["=== MICROFINANCE DATA CLEANING REPORT ===", ""]
    for k, v in stats.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)

# ---------- App UI ----------
st.set_page_config(page_title="Microfinance Data Cleaner", layout="wide")
st.title("Microfinance Data Cleaner — Streamlit App")

st.markdown("Upload one or more CSV/Excel files, or let the app clean files in the `raw_data/` folder.")

# Choose input mode
input_mode = st.radio("Input mode", ("Upload files (recommended)", "Process files in raw_data folder"))

uploaded_files = None
if input_mode == "Upload files (recommended)":
    uploaded_files = st.file_uploader("Upload CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True)

# Run button (1-click)
run = st.button("Run Data Cleaner (one click)")

if run:
    summary_rows = []
    overall_stats = {
        "Removed invalid IDs": 0,
        "Invalid date entries removed": 0,
        "Duplicate rows removed": 0,
        "Files processed": 0
    }

    files_to_process = []

    if input_mode == "Upload files (recommended)":
        if not uploaded_files:
            st.error("Please upload at least one file or switch to folder mode.")
        else:
            for f in uploaded_files:
                # read file into DataFrame
                try:
                    if f.name.lower().endswith(".csv"):
                        df = pd.read_csv(f)
                    else:
                        df = pd.read_excel(f)
                    files_to_process.append((f.name, df))
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")
    else:
        # process all csv/xlsx files in RAW_FOLDER
        for fname in os.listdir(RAW_FOLDER):
            if fname.lower().endswith((".csv", ".xlsx")):
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
        def sanitize_filename(name):
            return re.sub(r'[<>:"/\\|?*]', '_', name).strip()

        for i, (fname, df) in enumerate(files_to_process, start=1):
            overall_stats["Files processed"] += 1
            original_rows = len(df)

            try:
                df = clean_column_names(df)

                df, removed_ids = clean_customer_id(df)
                overall_stats["Removed invalid IDs"] += removed_ids

                df = clean_loan_amount(df)
                df = clean_income(df)
                df = clean_loan_status(df)

                df, bad_dates = clean_dates(df)
                overall_stats["Invalid date entries removed"] += bad_dates

                df, removed_dupes = remove_duplicates(df)
                overall_stats["Duplicate rows removed"] += removed_dupes

                df = df.reset_index(drop=True)

                # Save cleaned file
                cleaned_name = f"cleaned_{sanitize_filename(fname.rsplit('.',1)[0])}.csv"
                cleaned_path = os.path.join(CLEANED_FOLDER, cleaned_name)
                df.to_csv(cleaned_path, index=False)

                # Store a small preview
                summary_rows.append({
                    "file": fname,
                    "rows_before": original_rows,
                    "rows_after": len(df),
                    "removed_ids": removed_ids,
                    "bad_dates": bad_dates,
                    "removed_dupes": removed_dupes,
                    "cleaned_path": cleaned_path
                })

            except Exception as e:
                st.error(f"Error processing {fname}: {e}")
                st.error(traceback.format_exc())

            progress_bar.progress(i / len(files_to_process))

        st.success("Cleaning finished ✅")
        st.subheader("Summary (per file)")
        st.table(pd.DataFrame(summary_rows))

        st.subheader("Overall stats")
        st.json(overall_stats)

        # Allow user to download cleaned files and report
        st.markdown("### Download cleaned files")
        for r in summary_rows:
            with open(r["cleaned_path"], "rb") as f:
                data_bytes = f.read()
            st.download_button(label=f"Download {os.path.basename(r['cleaned_path'])}",
                               data=data_bytes,
                               file_name=os.path.basename(r['cleaned_path']),
                               mime="text/csv")

        report_text = create_report_text(overall_stats)
        st.download_button(label="Download cleaning report (txt)",
                           data=report_text,
                           file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                           mime="text/plain")
    else:
        st.warning("No files to process. Upload files or place files into the raw_data folder.")
