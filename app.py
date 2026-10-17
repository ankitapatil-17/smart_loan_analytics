import os
import json
import math
import re
import sqlite3
import uuid
from datetime import datetime
from functools import wraps

import pandas as pd
from flask import Flask, abort, render_template, request, redirect, url_for, flash, session, g, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "change-this-development-secret")
DB_PATH = os.path.join(os.path.dirname(__file__), "loan_analytics.db")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
LOAN_DOCUMENTS_FOLDER = os.path.join(UPLOAD_FOLDER, "loan_documents")
ALLOWED_EXTENSIONS = {"csv"}
DOCUMENT_REQUIREMENTS = {
    "Home": [
        {"key": "identity_proof", "label": "Identity proof", "description": "Aadhaar, PAN or passport copy"},
        {"key": "income_proof", "label": "Income proof", "description": "Last 3 salary slips or income certificate"},
        {"key": "property_documents", "label": "Property documents", "description": "Sale deed, agreement or property valuation"},
    ],
    "Personal": [
        {"key": "identity_proof", "label": "Identity proof", "description": "Aadhaar or PAN"},
        {"key": "income_proof", "label": "Income proof", "description": "Recent payslip or bank statements"},
        {"key": "address_proof", "label": "Address proof", "description": "Utility bill or rental agreement"},
    ],
    "Education": [
        {"key": "identity_proof", "label": "Identity proof", "description": "Aadhaar or PAN"},
        {"key": "education_documents", "label": "Admission documents", "description": "Admission letter and fee receipt"},
        {"key": "income_proof", "label": "Income proof", "description": "Parent/guardian income documents"},
    ],
    "Vehicle": [
        {"key": "identity_proof", "label": "Identity proof", "description": "Aadhaar or PAN"},
        {"key": "income_proof", "label": "Income proof", "description": "Salary slip or ITR copy"},
        {"key": "vehicle_quote", "label": "Vehicle quote", "description": "Quotation or invoice for the vehicle"},
    ],
    "Business": [
        {"key": "identity_proof", "label": "Identity proof", "description": "Aadhaar or PAN"},
        {"key": "business_proof", "label": "Business proof", "description": "GST, registration or business documents"},
        {"key": "financials", "label": "Financial records", "description": "Bank statements or financial statements"},
    ],
    "Medical": [
        {"key": "identity_proof", "label": "Identity proof", "description": "Aadhaar or PAN"},
        {"key": "medical_report", "label": "Medical report", "description": "Doctor prescription or treatment estimate"},
        {"key": "income_proof", "label": "Income proof", "description": "Pay slips or bank statements"},
    ],
    "Other": [
        {"key": "identity_proof", "label": "Identity proof", "description": "Aadhaar or PAN"},
        {"key": "income_proof", "label": "Income proof", "description": "Latest income proof"},
        {"key": "supporting_docs", "label": "Supporting documents", "description": "Any additional proof relevant to the request"},
    ],
}
LOAN_STATUSES = ("Pending", "Approved", "Closed", "Rejected")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOAN_DOCUMENTS_FOLDER, exist_ok=True)

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db

@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'analyst',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS applicants (
        applicant_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        income REAL NOT NULL,
        employment_type TEXT NOT NULL,
        dependents INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS loans (
        loan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_id INTEGER NOT NULL,
        amount REAL NOT NULL,
        tenure INTEGER NOT NULL,
        purpose TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Pending',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(applicant_id) REFERENCES applicants(applicant_id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS credit_records (
        credit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_id INTEGER NOT NULL UNIQUE,
        credit_score INTEGER NOT NULL,
        existing_loans INTEGER NOT NULL DEFAULT 0,
        late_payments INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(applicant_id) REFERENCES applicants(applicant_id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        loan_id INTEGER NOT NULL,
        probability REAL NOT NULL,
        decision TEXT NOT NULL,
        risk_category TEXT NOT NULL,
        model_name TEXT NOT NULL,
        predicted_at TEXT NOT NULL,
        FOREIGN KEY(loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS loan_documents (
        document_id INTEGER PRIMARY KEY AUTOINCREMENT,
        loan_id INTEGER NOT NULL,
        document_type TEXT NOT NULL,
        original_name TEXT NOT NULL,
        stored_name TEXT NOT NULL,
        uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(loan_id) REFERENCES loans(loan_id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS csv_imports (
        import_id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)
    admin = db.execute("SELECT user_id FROM users WHERE email=?", ("admin@loanrisk.local",)).fetchone()
    if not admin:
        db.execute("INSERT INTO users(name,email,password_hash,role) VALUES(?,?,?,?)",
                   ("Demo Admin", "admin@loanrisk.local", generate_password_hash("Admin123!"), "admin"))
    count = db.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]
    if count == 0:
        demo = [
            ("Aarav Sharma", 29, "Male", 72000, "Salaried", 1, 745, 1, 0, 450000, 60, "Home"),
            ("Priya Nair", 34, "Female", 98000, "Salaried", 2, 790, 0, 0, 800000, 120, "Home"),
            ("Karan Mehta", 25, "Male", 32000, "Self-employed", 0, 610, 2, 3, 250000, 36, "Personal"),
            ("Sneha Rao", 41, "Female", 56000, "Salaried", 1, 680, 2, 1, 300000, 48, "Education"),
            ("Imran Khan", 38, "Male", 125000, "Self-employed", 3, 720, 1, 0, 900000, 84, "Business"),
            ("Meera Iyer", 31, "Female", 45000, "Contract", 0, 640, 1, 2, 180000, 24, "Vehicle"),
            ("Dev Patel", 27, "Male", 85000, "Salaried", 0, 770, 0, 0, 500000, 60, "Vehicle"),
            ("Ananya Das", 45, "Female", 68000, "Salaried", 2, 700, 2, 1, 620000, 72, "Home"),
            ("Rahul Verma", 23, "Male", 28000, "Contract", 1, 580, 3, 4, 150000, 24, "Personal"),
            ("Fatima Sheikh", 36, "Female", 91000, "Self-employed", 2, 755, 1, 0, 550000, 60, "Business"),
        ]
        for row in demo:
            name, age, gender, income, emp, deps, score, existing, late, amount, tenure, purpose = row
            cur = db.execute("INSERT INTO applicants(name,age,gender,income,employment_type,dependents) VALUES(?,?,?,?,?,?)",
                             (name, age, gender, income, emp, deps))
            aid = cur.lastrowid
            db.execute("INSERT INTO credit_records(applicant_id,credit_score,existing_loans,late_payments) VALUES(?,?,?,?)",
                       (aid, score, existing, late))
            db.execute("INSERT INTO loans(applicant_id,amount,tenure,purpose,status) VALUES(?,?,?,?,?)",
                       (aid, amount, tenure, purpose, "Approved" if score >= 700 and late <= 1 else "Pending"))
    db.commit()
    db.close()

def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapped

def get_required_documents(purpose="Home"):
    return DOCUMENT_REQUIREMENTS.get(purpose, DOCUMENT_REQUIREMENTS["Other"])


def save_loan_documents(loan_id, purpose, files):
    docs = get_required_documents(purpose)
    saved_docs = []
    for doc in docs:
        field = files.get(doc["key"])
        if field is None or not field.filename:
            raise ValueError(f"Please upload: {doc['label']}")
        filename = secure_filename(field.filename)
        if not filename:
            raise ValueError(f"Please upload a valid file for {doc['label']}.")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        safe_name = f"{loan_id}_{doc['key']}_{timestamp}_{filename}"
        destination = os.path.join(LOAN_DOCUMENTS_FOLDER, safe_name)
        field.save(destination)
        saved_docs.append((loan_id, doc["label"], filename, safe_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    if saved_docs:
        get_db().executemany(
            "INSERT INTO loan_documents(loan_id, document_type, original_name, stored_name, uploaded_at) VALUES(?,?,?,?,?)",
            saved_docs,
        )
        get_db().commit()
    return saved_docs


def get_loan_rows():
    return get_db().execute("""
        SELECT l.*, a.name, a.age, a.gender, a.income, a.employment_type, a.dependents,
               c.credit_score, c.existing_loans, c.late_payments
        FROM loans l JOIN applicants a ON a.applicant_id=l.applicant_id
        LEFT JOIN credit_records c ON c.applicant_id=a.applicant_id
        ORDER BY l.loan_id DESC
    """).fetchall()

def estimate_probability(row):
    # Transparent demo baseline. Replace with a validated model before real-world use.
    score = 0.0
    credit = int(row["credit_score"] or 600)
    income = float(row["income"] or 0)
    amount = float(row["amount"] or 0)
    late = int(row["late_payments"] or 0)
    existing = int(row["existing_loans"] or 0)
    tenure = int(row["tenure"] or 1)
    score += max(0, min(1, (credit - 500) / 350)) * 0.42
    score += max(0, min(1, income / max(amount / max(tenure, 1), 1) / 12)) * 0.28
    score += max(0, 1 - late / 5) * 0.20
    score += max(0, 1 - existing / 5) * 0.10
    return max(0.02, min(0.98, score))

def classify(row):
    probability = estimate_probability(row)
    if probability >= 0.70:
        decision, risk = "Likely Eligible", "Low"
    elif probability >= 0.45:
        decision, risk = "Needs Review", "Medium"
    else:
        decision, risk = "High Risk", "High"
    return probability, decision, risk, "Rule-based demo baseline"


CSV_FIELD_ALIASES = {
    "name": {"name", "applicant_name", "customer_name", "full_name", "borrower_name"},
    "age": {"age", "applicant_age", "customer_age"},
    "gender": {"gender", "sex"},
    "income": {
        "income", "applicant_income", "annual_income", "yearly_income", "monthly_income",
        "salary", "annual_salary", "monthly_salary", "gross_income", "gross_annual_income",
    },
    "employment_type": {"employment_type", "employment", "occupation", "job_type", "work_type"},
    "dependents": {"dependents", "dependent", "number_of_dependents", "num_dependents"},
    "credit_score": {"credit_score", "creditscore", "cibil", "cibil_score", "credit_rating", "bureau_score"},
    "existing_loans": {"existing_loans", "current_loans", "number_of_loans", "num_loans", "loan_count"},
    "late_payments": {"late_payments", "missed_payments", "delayed_payments", "delinquencies"},
    "amount": {"amount", "loan_amount", "requested_amount", "requested_loan_amount"},
    "tenure": {"tenure", "loan_tenure", "loan_term", "term_months", "loan_tenure_months"},
    "purpose": {"purpose", "loan_purpose", "loan_type", "loan_category"},
}
CSV_LOAN_PURPOSES = {"home", "personal", "education", "vehicle", "business", "medical", "other"}


def normalize_csv_column(column):
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def detect_csv_fields(columns):
    normalized = {}
    for column in columns:
        key = normalize_csv_column(column)
        normalized.setdefault(key, []).append(column)
    mapping = {}
    ambiguous = []
    for field, aliases in CSV_FIELD_ALIASES.items():
        candidates = [
            column
            for alias in aliases
            for column in normalized.get(alias, [])
        ]
        if len(candidates) > 1:
            ambiguous.append(field)
        elif candidates:
            mapping[field] = candidates[0]
    if ambiguous:
        raise ValueError(
            "More than one uploaded column could represent: " + ", ".join(ambiguous)
            + ". Rename the duplicate columns so each field is unambiguous."
        )
    return mapping, sorted(set(CSV_FIELD_ALIASES) - set(mapping))


def normalize_loan_purpose(value):
    purpose = normalize_csv_column(value)
    for allowed in CSV_LOAN_PURPOSES:
        if purpose == allowed or purpose == f"{allowed}_loan":
            return allowed.title()
    return None


def parse_csv_number(value, field, row_number, integer=False, minimum=None, maximum=None):
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"{field} is required")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be numeric") from None
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    if integer and not number.is_integer():
        raise ValueError(f"{field} must be a whole number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be no more than {maximum}")
    return int(number) if integer else number


def validate_csv_loan_rows(dataframe):
    field_mapping, missing_fields = detect_csv_fields(dataframe.columns)
    if missing_fields:
        return [], [], field_mapping, missing_fields
    if len(dataframe) > 1000:
        raise ValueError("The CSV may contain no more than 1,000 rows per import.")
    if dataframe.empty:
        raise ValueError("The CSV must contain at least one data row.")

    valid_rows = []
    invalid_rows = []
    for row_number, (_, source) in enumerate(dataframe.iterrows(), start=2):
        try:
            name = "" if pd.isna(source[field_mapping["name"]]) else str(source[field_mapping["name"]]).strip()
            gender = "" if pd.isna(source[field_mapping["gender"]]) else str(source[field_mapping["gender"]]).strip()
            employment_type = "" if pd.isna(source[field_mapping["employment_type"]]) else str(source[field_mapping["employment_type"]]).strip()
            if not name or not gender or not employment_type:
                raise ValueError("name, gender and employment_type are required")
            age = parse_csv_number(source[field_mapping["age"]], "age", row_number, integer=True, minimum=18, maximum=100)
            income = parse_csv_number(source[field_mapping["income"]], "income", row_number, minimum=0.01)
            dependents = parse_csv_number(source[field_mapping["dependents"]], "dependents", row_number, integer=True, minimum=0)
            credit_score = parse_csv_number(source[field_mapping["credit_score"]], "credit_score", row_number, integer=True, minimum=300, maximum=900)
            existing_loans = parse_csv_number(source[field_mapping["existing_loans"]], "existing_loans", row_number, integer=True, minimum=0)
            late_payments = parse_csv_number(source[field_mapping["late_payments"]], "late_payments", row_number, integer=True, minimum=0)
            amount = parse_csv_number(source[field_mapping["amount"]], "amount", row_number, minimum=0.01)
            tenure = parse_csv_number(source[field_mapping["tenure"]], "tenure", row_number, integer=True, minimum=1, maximum=360)
            raw_purpose = "" if pd.isna(source[field_mapping["purpose"]]) else str(source[field_mapping["purpose"]]).strip()
            purpose = normalize_loan_purpose(raw_purpose)
            if not purpose:
                raise ValueError("purpose must be Home, Personal, Education, Vehicle, Business, Medical or Other")
            row = {
                "source_row": row_number,
                "name": name,
                "age": age,
                "gender": gender,
                "income": income,
                "employment_type": employment_type,
                "dependents": dependents,
                "credit_score": credit_score,
                "existing_loans": existing_loans,
                "late_payments": late_payments,
                "amount": amount,
                "tenure": tenure,
                "purpose": purpose,
            }
            probability, decision, risk, model = classify(row)
            row["probability"] = probability
            row["decision"] = decision
            row["risk_category"] = risk
            row["model_name"] = model
            row["status"] = "Approved" if credit_score > 750 else "Pending"
            valid_rows.append(row)
        except ValueError as exc:
            invalid_rows.append({"row_number": row_number, "reason": str(exc)})
    return valid_rows, invalid_rows, field_mapping, missing_fields

@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["user_id"]
            session["name"] = user["name"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 8:
            flash("Enter a name, valid email, and password of at least 8 characters.", "error")
        else:
            try:
                get_db().execute("INSERT INTO users(name,email,password_hash,role) VALUES(?,?,?,?)",
                                 (name, email, generate_password_hash(password), "analyst"))
                get_db().commit()
                flash("Account created. You can now sign in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That email is already registered.", "error")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    db = get_db()
    totals = db.execute("""
        SELECT COUNT(*) total,
        SUM(CASE WHEN status='Approved' THEN 1 ELSE 0 END) approved,
        SUM(CASE WHEN status='Pending' THEN 1 ELSE 0 END) pending,
        AVG(amount) avg_amount
        FROM loans
    """).fetchone()
    applicants = db.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]
    purposes = db.execute("SELECT purpose, COUNT(*) count FROM loans GROUP BY purpose ORDER BY count DESC").fetchall()
    recent = get_loan_rows()[:6]
    predictions = db.execute("SELECT risk_category, COUNT(*) count FROM predictions GROUP BY risk_category").fetchall()
    return render_template("dashboard.html", totals=totals, applicants=applicants, purposes=purposes,
                           recent=recent, predictions=predictions)

@app.route("/applicants", methods=["GET", "POST"])
@login_required
def applicants():
    db = get_db()
    if request.method == "POST":
        try:
            name = request.form["name"].strip()
            age = int(request.form["age"])
            gender = request.form["gender"]
            income = float(request.form["income"])
            employment = request.form["employment_type"]
            dependents = int(request.form.get("dependents", 0))
            credit = int(request.form["credit_score"])
            existing = int(request.form.get("existing_loans", 0))
            late = int(request.form.get("late_payments", 0))
            if not name or age < 18 or age > 100 or income <= 0 or not 300 <= credit <= 900:
                raise ValueError("Check the name, age, income and credit score values.")
            cur = db.execute("INSERT INTO applicants(name,age,gender,income,employment_type,dependents) VALUES(?,?,?,?,?,?)",
                             (name, age, gender, income, employment, dependents))
            aid = cur.lastrowid
            db.execute("INSERT INTO credit_records(applicant_id,credit_score,existing_loans,late_payments) VALUES(?,?,?,?)",
                       (aid, credit, existing, late))
            db.commit()
            flash("Applicant created successfully.", "success")
            return redirect(url_for("applicants"))
        except (ValueError, KeyError):
            flash("Please enter valid applicant data. Age must be 18–100 and credit score 300–900.", "error")
    rows = db.execute("""SELECT a.*, c.credit_score, c.existing_loans, c.late_payments
                         FROM applicants a LEFT JOIN credit_records c ON c.applicant_id=a.applicant_id
                         ORDER BY a.applicant_id DESC""").fetchall()
    return render_template("applicants.html", applicants=rows)

@app.route("/loans", methods=["GET", "POST"])
@login_required
def loans():
    db = get_db()
    selected_purpose = request.form.get("purpose", "Home") if request.method == "POST" else "Home"
    required_documents = get_required_documents(selected_purpose)
    if request.method == "POST":
        try:
            aid = int(request.form["applicant_id"])
            amount = float(request.form["amount"])
            tenure = int(request.form["tenure"])
            purpose = request.form["purpose"]
            if amount <= 0 or tenure < 1 or tenure > 360:
                raise ValueError()
            credit_record = db.execute(
                "SELECT credit_score FROM credit_records WHERE applicant_id=?",
                (aid,),
            ).fetchone()
            initial_status = (
                "Approved"
                if credit_record and credit_record["credit_score"] > 750
                else "Pending"
            )
            cur = db.execute("INSERT INTO loans(applicant_id,amount,tenure,purpose,status) VALUES(?,?,?,?,?)",
                             (aid, amount, tenure, purpose, initial_status))
            loan_id = cur.lastrowid
            save_loan_documents(loan_id, purpose, request.files)
            db.commit()
            if initial_status == "Approved":
                flash("Loan application submitted with supporting documents and auto-approved based on credit score above 750.", "success")
            else:
                flash("Loan application submitted with supporting documents.", "success")
            return redirect(url_for("loans"))
        except (ValueError, KeyError, sqlite3.IntegrityError) as exc:
            flash(str(exc) if str(exc) else "Please check applicant, amount, tenure and required documents.", "error")
    people = db.execute("SELECT applicant_id,name FROM applicants ORDER BY name").fetchall()
    documents = db.execute("""
        SELECT document_id, loan_id, document_type, original_name
        FROM loan_documents
        ORDER BY document_id
    """).fetchall()
    documents_by_loan = {}
    for document in documents:
        documents_by_loan.setdefault(document["loan_id"], []).append(document)
    return render_template("loans.html", loans=get_loan_rows(), applicants=people,
                           required_documents=required_documents, selected_purpose=selected_purpose,
                           documents_by_loan=documents_by_loan)


@app.route("/loans/<int:loan_id>/status", methods=["POST"])
@login_required
def update_loan_status(loan_id):
    status = request.form.get("status", "")
    if status not in LOAN_STATUSES:
        flash("Choose a valid loan status.", "error")
        return redirect(url_for("loans"))

    db = get_db()
    result = db.execute("UPDATE loans SET status=? WHERE loan_id=?", (status, loan_id))
    if result.rowcount == 0:
        flash("Loan application not found.", "error")
    else:
        db.commit()
        flash(f"Loan status updated to {status}.", "success")
    return redirect(url_for("loans"))


@app.route("/loan-documents/<int:document_id>")
@login_required
def download_loan_document(document_id):
    document = get_db().execute(
        "SELECT original_name, stored_name FROM loan_documents WHERE document_id=?",
        (document_id,),
    ).fetchone()
    if not document:
        abort(404)
    return send_from_directory(
        LOAN_DOCUMENTS_FOLDER,
        document["stored_name"],
        as_attachment=True,
        download_name=document["original_name"],
    )

@app.route("/predict/<int:loan_id>", methods=["POST"])
@login_required
def predict(loan_id):
    db = get_db()
    row = db.execute("""SELECT l.*, a.income, a.age, a.gender, a.employment_type, a.dependents,
                               c.credit_score, c.existing_loans, c.late_payments
                        FROM loans l JOIN applicants a ON a.applicant_id=l.applicant_id
                        LEFT JOIN credit_records c ON c.applicant_id=a.applicant_id
                        WHERE l.loan_id=?""", (loan_id,)).fetchone()
    if not row:
        flash("Loan application not found.", "error")
        return redirect(url_for("loans"))
    probability, decision, risk, model = classify(row)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db.execute("INSERT INTO predictions(loan_id,probability,decision,risk_category,model_name,predicted_at) VALUES(?,?,?,?,?,?)",
               (loan_id, probability, decision, risk, model, now))
    # Status is kept separate from the model output: the prediction is advisory only.
    db.commit()
    flash(f"Prediction saved: {decision} ({probability:.0%} estimated score). This is a demo, not a lending decision.", "success")
    return redirect(url_for("history"))

@app.route("/history")
@login_required
def history():
    rows = get_db().execute("""
        SELECT p.*, l.amount, l.purpose, a.name AS applicant_name
        FROM predictions p JOIN loans l ON l.loan_id=p.loan_id
        JOIN applicants a ON a.applicant_id=l.applicant_id
        ORDER BY p.predicted_at DESC
    """).fetchall()
    return render_template("history.html", predictions=rows)

@app.route("/analytics")
@login_required
def analytics():
    db = get_db()
    by_status = db.execute("SELECT status, COUNT(*) count FROM loans GROUP BY status").fetchall()
    by_purpose = db.execute("SELECT purpose, COUNT(*) count, AVG(amount) avg_amount FROM loans GROUP BY purpose").fetchall()
    income = db.execute("SELECT employment_type, AVG(income) avg_income, COUNT(*) count FROM applicants GROUP BY employment_type").fetchall()
    risk = db.execute("SELECT risk_category, COUNT(*) count FROM predictions GROUP BY risk_category").fetchall()
    return render_template("analytics.html", by_status=by_status, by_purpose=by_purpose, income=income, risk=risk)

@app.route("/reports")
@login_required
def reports():
    rows = get_loan_rows()
    return render_template("reports.html", loans=rows)

@app.route("/reports/export.csv")
@login_required
def export_csv():
    rows = get_loan_rows()
    df = pd.DataFrame([dict(r) for r in rows])
    from flask import Response
    return Response(df.to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=loan_applications_report.csv"})

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        f = request.files.get("dataset")
        if not f or not f.filename or not f.filename.lower().endswith(".csv"):
            flash("Upload a CSV file only.", "error")
            return redirect(url_for("upload"))
        try:
            df = pd.read_csv(f.stream)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            flash(f"Could not read CSV: {exc}", "error")
            return render_template(
                "upload.html", preview=None, columns=None, summary=None,
                valid_rows=[], invalid_rows=[], detected_fields={}, missing_fields=[],
                import_id=None,
            )

        import_id = None
        valid_rows = []
        invalid_rows = []
        detected_fields = {}
        missing_fields = []
        try:
            valid_rows, invalid_rows, detected_fields, missing_fields = validate_csv_loan_rows(df)
        except ValueError as exc:
            flash(f"CSV validation failed: {exc}", "error")

        db = get_db()
        db.execute("DELETE FROM csv_imports WHERE user_id=?", (session["user_id"],))
        if valid_rows:
            import_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO csv_imports(import_id,user_id,payload_json) VALUES(?,?,?)",
                (import_id, session["user_id"], json.dumps(valid_rows)),
            )
        db.commit()
        preview = df.head(10).fillna("").to_dict(orient="records")
        columns = list(df.columns)
        summary = df.describe(include="all").fillna("").to_html(classes="data-table", border=0)
        if valid_rows or invalid_rows:
            flash(
                f"Validation complete: {len(valid_rows)} valid row(s), {len(invalid_rows)} invalid row(s). "
                "Only valid rows can be added.",
                "success" if valid_rows else "error",
            )
        elif missing_fields:
            flash(
                "The dataset was analyzed, but automatic loan validation needs more identifiable fields.",
                "error",
            )
        return render_template(
            "upload.html",
            preview=preview,
            columns=columns,
            summary=summary,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            detected_fields=detected_fields,
            missing_fields=missing_fields,
            import_id=import_id,
        )
    return render_template(
        "upload.html",
        preview=None,
        columns=None,
        summary=None,
        valid_rows=[],
        invalid_rows=[],
        detected_fields={},
        missing_fields=[],
        import_id=None,
    )


@app.route("/upload/import", methods=["POST"])
@login_required
def import_csv_loans():
    import_id = request.form.get("import_id", "")
    db = get_db()
    staged = db.execute(
        "SELECT payload_json FROM csv_imports WHERE import_id=? AND user_id=?",
        (import_id, session["user_id"]),
    ).fetchone()
    if not staged:
        flash("This validated CSV import has expired or is unavailable. Upload and validate it again.", "error")
        return redirect(url_for("upload"))

    rows = json.loads(staged["payload_json"])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    imported = 0
    try:
        for row in rows:
            applicant = db.execute(
                """INSERT INTO applicants(name,age,gender,income,employment_type,dependents)
                   VALUES(?,?,?,?,?,?)""",
                (
                    row["name"], row["age"], row["gender"], row["income"],
                    row["employment_type"], row["dependents"],
                ),
            )
            applicant_id = applicant.lastrowid
            db.execute(
                """INSERT INTO credit_records(applicant_id,credit_score,existing_loans,late_payments)
                   VALUES(?,?,?,?)""",
                (applicant_id, row["credit_score"], row["existing_loans"], row["late_payments"]),
            )
            loan = db.execute(
                """INSERT INTO loans(applicant_id,amount,tenure,purpose,status)
                   VALUES(?,?,?,?,?)""",
                (applicant_id, row["amount"], row["tenure"], row["purpose"], row["status"]),
            )
            db.execute(
                """INSERT INTO predictions
                   (loan_id,probability,decision,risk_category,model_name,predicted_at)
                   VALUES(?,?,?,?,?,?)""",
                (
                    loan.lastrowid, row["probability"], row["decision"], row["risk_category"],
                    row["model_name"], now,
                ),
            )
            imported += 1
        db.execute("DELETE FROM csv_imports WHERE import_id=?", (import_id,))
        db.commit()
    except sqlite3.Error:
        db.rollback()
        raise

    eligible = sum(1 for row in rows if row["decision"] == "Likely Eligible")
    approved = sum(1 for row in rows if row["status"] == "Approved")
    flash(
        f"Added {imported} loan application(s). Demo assessment: {eligible} likely eligible; "
        f"{approved} auto-approved based on credit score above 750. Review each result before any real decision.",
        "success",
    )
    return redirect(url_for("loans"))

@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(413)
def upload_too_large(_error):
    flash("The uploaded file is too large. Please use a CSV smaller than 8 MB.", "error")
    return redirect(url_for("upload"))

init_db()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
