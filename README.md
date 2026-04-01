# CAPTCHA Attack Detector (Behavior‑Based)

A mini‑project that detects automated (bot‑like) CAPTCHA abuse by analyzing behavioral data from CAPTCHA attempts.  
**It does NOT solve or break CAPTCHAs** – it only classifies attempts as Human or Suspicious based on patterns.

## Features

- Upload CSV files with CAPTCHA attempt data (required columns: `solve_time`, `retry_count`, `attempts_in_window`, `consistency_score`)
- Manual single‑attempt form for quick testing
- Backend uses **Isolation Forest** (scikit‑learn) to compute anomaly scores and labels
- Stores upload history and per‑row results in **SQLite**
- Clean, responsive web UI with summary cards and result tables
- Download original CSV files from history

## Tech Stack

- Python, Flask, Pandas, NumPy, scikit‑learn, joblib
- SQLite, HTML5, CSS3 (no external UI libraries)

## Setup & Run

1. **Clone or extract** the project folder.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows