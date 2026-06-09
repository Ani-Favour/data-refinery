# DataRefinery

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/streamlit-1.52.0-orange.svg)](https://streamlit.io/)

A Streamlit toolkit for automated data cleaning and preparation. DataRefinery processes CSV and Excel datasets using configurable, one-click cleaning rules to standardize, validate, and export analysis-ready data.

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

## Sample data

Two sample datasets are included in `raw_data/` for quick testing: `microfinance_sample.csv` and `general_sample.csv`.

## Publish to GitHub

If you already created a GitHub repository, connect it and push:

```powershell
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

If you still need to create the repo, do that on GitHub first, then run the commands above.

## How to use

- Upload one or more CSV/XLSX files using the app UI, or place files into `raw_data/`.
- Click **Run Data Cleaner (one click)**.
- Download cleaned CSV files and the cleaning report.

## Notes

- The app creates the `raw_data/`, `cleaned_data/`, and `reports/` folders automatically if they do not exist.
- Data folders are ignored by Git to keep the repository clean.
