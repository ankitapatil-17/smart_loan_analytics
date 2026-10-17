# Smart Loan Eligibility & Customer Risk Analytics

A runnable Flask + SQLite project for a college project/demo. It includes applicant registration, loan applications, data validation, a dashboard, exploratory analytics, demo eligibility/risk scoring, prediction history, CSV upload preview, and CSV report export.

> **Important:** The eligibility score is a transparent rule-based demo baseline, not a trained or validated credit model. It must not be used to approve/deny real loans. Real lending decisions require validated data, fairness testing, explainability, privacy controls, and human review.

## Requirements
- Python 3.10+
- pip

## Run on Windows
1. Extract the ZIP.
2. Open the extracted folder in VS Code or Command Prompt.
3. Create a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt:
   ```bat
   .venv\Scripts\activate.bat
   ```
4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   python app.py
   ```
6. Open http://127.0.0.1:5000

## Demo login
- Email: `admin@loanrisk.local`
- Password: `Admin123!`

Change the development secret and demo credentials before any deployment. The app runs locally by default.

## Included pages
- Login / Register
- Dashboard with KPIs and recent applications
- Applicant management (create and list)
- Loan application creation and list with purpose-based document checklist, file uploads and downloads, editable statuses, and automatic approval for credit scores above 750
- Eligibility/risk prediction and history
- Analytics by status, purpose, employment type and risk segment
- CSV validation, demo eligibility/risk assessment, and import of valid rows into applicants and loan applications
- CSV report export

## CSV upload
Upload any `.csv` file (up to 8 MB and 1,000 rows). The explorer automatically detects common applicant and loan header variations; no column mapping is required. It validates data when all fields needed for a loan application are identifiable, reports invalid rows, and shows the rule-based demo eligibility/risk result for valid rows. Other datasets remain available for preview and descriptive analysis, but cannot be imported as loan applications unless the required information can be detected. You can import valid rows as applicant profiles and loan applications; credit scores above 750 auto-approve the application, while all other scores stay Pending. CSV imports do not include supporting document files. The assessment is for demonstration only, not a real lending decision. Do not upload confidential customer information to a demo instance.

## Database
SQLite database is created automatically as `loan_analytics.db` on first run. Tables: `users`, `applicants`, `loans`, `credit_records`, `predictions`, `loan_documents`, `csv_imports`.

## Reset demo data
Stop the app, delete `loan_analytics.db`, then run `python app.py` again. The demo database will be recreated.

## Notes for viva
- **Normalization:** applicant data, credit records, loans, and predictions are separated into relational tables.
- **Foreign keys:** loans reference applicants; credit records reference applicants; predictions reference loans.
- **CRUD:** applicant and loan forms create records; lists display them.
- **Loan status:** authorized users can set applications to Pending, Approved, Closed, or Rejected. New applications with a credit score above 750 are auto-approved; lower scores remain Pending.
- **Analytics:** counts and distributions are descriptive analytics.
- **Prediction:** the demo score uses credit score, income relative to payment burden, late payments, and existing loans. It is advisory and not a real lending model.
- **Production improvements:** CSRF protection, rate limiting, audit logging, robust migrations, model validation, calibration, fairness review, access controls, encrypted backups, and a production WSGI server.
