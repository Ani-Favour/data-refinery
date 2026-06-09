# Microfinance Data Cleaner

A simple Streamlit app for cleaning microfinance loan datasets.

## Project structure

- `app.py` — Streamlit application entrypoint
- `raw_data/` — input data files (CSV or Excel)
- `cleaned_data/` — cleaned output CSV files
- `reports/` — generated cleaning report files
- `requirements.txt` — Python dependencies

## Setup

1. Open a terminal in this project folder.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

Then open the app in your browser at `http://localhost:8501`.

## How to use

- Upload one or more CSV/XLSX files using the app UI, or place files into `raw_data/`.
- Click **Run Data Cleaner (one click)**.
- Download cleaned CSV files and the cleaning report.

## Notes

- The app creates the `raw_data/`, `cleaned_data/`, and `reports/` folders automatically if they do not exist.
- Data folders are ignored by Git to keep the repository clean.
