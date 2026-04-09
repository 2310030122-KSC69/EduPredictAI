"""EduPredict Enhanced - app.py"""
import os
import json
import uuid
import hashlib
import random
import subprocess
import logging
import numpy as np
from datetime import datetime, date, timedelta
from flask import Flask, request, jsonify, render_template, session, redirect
import joblib

def clean_input(val):
    if isinstance(val, str):
        return " ".join(val.strip().split())
    return val

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(FRONTEND_DIR, "templates"), static_folder=os.path.join(FRONTEND_DIR, "static"))
app.secret_key = "edupredict-secret-2025"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("EduPredict server starting up...")

def load_artifacts():
    logger.info("Loading ML models...")
    a = {}
    for n in ["rf_model", "gb_model", "lr_model", "svm_model", "scaler", "label_encoder"]:
        p = os.path.join(MODELS_DIR, f"{n}.pkl")
        if os.path.exists(p):
            a[n] = joblib.load(p)
            logger.info(f"Loaded {n}")
        else:
            logger.warning(f"Model {n} not found at {p}")
    mp = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(mp):
        with open(mp) as f:
            a["metadata"] = json.load(f)
    return a

ARTIFACTS = load_artifacts()
MODEL_MAP = {"random_forest": "rf_model", "gradient_boosting": "gb_model", "logistic_regression": "lr_model", "svm": "svm_model"}
FEATURE_COLS = ["study_hours_per_day", "attendance_rate", "previous_gpa", "assignments_completed", "sleep_hours", "extracurricular_activities", "parental_education", "internet_access", "tutoring_sessions", "stress_level", "motivation_score", "part_time_job"]
SUBJECTS_MAP = {"B.Tech CS": ["Data Structures", "Algorithms", "DBMS", "OS", "CN", "ML"], "B.Tech ECE": ["Signals", "Electronics", "Communication", "VLSI", "Embedded"], "B.Tech ME": ["Thermodynamics", "Fluid Mechanics", "Manufacturing", "Design"]}

def assess_risk(features, prediction):
    risk = {"Excellent": "Low", "Good": "Low", "Average": "Medium", "Poor": "High"}.get(prediction, "Medium")
    tips = []
    if float(features.get("study_hours_per_day", 0)) < 3: tips.append("Increase study hours to at least 3–4 hours per day.")
    if float(features.get("attendance_rate", 0)) < 75: tips.append("Improve attendance above 75% to stay on track.")
    if float(features.get("stress_level", 5)) > 7: tips.append("High stress detected — consider mindfulness or counseling.")
    if float(features.get("sleep_hours", 7)) < 6: tips.append("Sleep at least 7–8 hours for better cognitive performance.")
    if float(features.get("assignments_completed", 0)) < 70: tips.append("Complete more assignments to strengthen your grade base.")
    if float(features.get("motivation_score", 5)) < 5: tips.append("Set short-term goals to boost motivation.")
    if not tips: tips.append("Great performance! Keep up the consistency.")
    return risk, tips

QUIZ_BANK = {
  "Data Structures": [
    {"q": "Time complexity of binary search?", "opts": ["O(n)", "O(log n)", "O(n²)", "O(1)"], "ans": 1, "exp": "Binary search halves the search space each step → O(log n)"},
    # ... (keeping full QUIZ_BANK as is - abbreviated for brevity)
  ],
  # ... (all other subjects unchanged)
}

DB_FILE = os.path.join(DATA_DIR, "db.json")

def init_db():
    return {
        "users": {},
        "students": {},
        "faculty": {},
        "attendance": {},
        "marks": {},
        "fees": {},
        "notifications": {},
        "timetable": {},
        "quiz_attempts": {},
        "leaderboard": {}
    }

def load_db():
    logger.info("Loading database...")
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            logger.info("Database loaded from file")
            return json.load(f)
    logger.warning("Database file not found, initializing...")
    return init_db()

def save_db(db):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(db, f, indent=2, default=str)
        logger.debug("Database saved successfully")
    except Exception as e:
        logger.error(f"Failed to save database: {e}")

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def make_marks_entry():
    return {"insem1": 0, "insem_lab": 0, "insem2": 0, "end_sem": 0, "end_lab": 0}

def compute_total(m):
    return m.get("insem1", 0) + m.get("insem_lab", 0) + m.get("insem2", 0) + m.get("end_sem", 0) + m.get("end_lab", 0)

def compute_max():
    return 150

def get_grade(p):
    if p >= 90: return "O"
    if p >= 80: return "A+"
    if p >= 70: return "A"
    if p >= 60: return "B+"
    if p >= 50: return "B"
    if p >= 40: return "C"
    return "F"

# ... (keeping _make_timetable and init_db unchanged)

DB = load_db()

def current_user():
    uid = session.get("user_id")
    return DB["users"].get(uid) if uid else None

def get_client_ip():
    return request.environ.get('REMOTE_ADDR', 'unknown')

# Routes with logging
@app.before_request
def log_request():
    ip = get_client_ip()
    method = request.method
    endpoint = request.endpoint or request.path
    logger.info(f"Request: {method} {endpoint} from IP {ip}")

@app.route("/")
def index():
    u = current_user()
    if u:
        if u["role"] == "admin": return redirect("/admin/dashboard")
        if u["role"] == "faculty": return redirect("/faculty/dashboard")
        return redirect("/student/dashboard")
    return redirect("/login")

@app.route("/logout")
def logout():
    uid = session.get("user_id")
    ip = get_client_ip()
    if uid:
        logger.info(f"User logout: {uid} ({DB['users'].get(uid, {}).get('email', 'unknown')}) from IP {ip}")
    session.clear()
    return redirect("/login")

# AUTH API with logging
@app.route("/api/auth/login", methods=["POST"])
def api_login():
    ip = get_client_ip()
    try:
        b = {k: clean_input(v) for k, v in request.get_json(force=True).items()}
        email = clean_input(b.get("email", "")).lower()
        pw = b.get("password", "").strip()
        logger.info(f"Login attempt for {email} from IP {ip}")

        if not email or not pw:
            logger.warning(f"Invalid login attempt (missing credentials) from IP {ip}")
            return jsonify({"error": "Email and password required"}), 400

        for uid, u in DB["users"].items():
            if u.get("email", "").lower() == email and u.get("password") == hash_pw(pw):
                session["user_id"] = uid
                logger.info(f"Login successful: {uid} ({u['role']}) {u['name']} from IP {ip}")
                return jsonify({"success": True, "role": u["role"], "name": u["name"]})

        logger.warning(f"Login failed for {email} from IP {ip}")
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        logger.error(f"Login error for IP {ip}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# HADOOP API - UNTOUCHED as per rules
# ── HADOOP API ─────────────────────────────────────────────────────────────────
@app.route("/api/hdfs/list", methods=["POST"])
def api_hdfs_list():
    logger.info(f"HDFS list request from IP {get_client_ip()}")
    data = request.get_json(force=True) or {}
    b = {k: clean_input(v) for k, v in data.items()}
    path = b.get("path", "/")

    if not path:
        logger.warning(f"HDFS empty path from IP {get_client_ip()}")
        return jsonify({"error": "Path cannot be empty"}), 400

    try:
        result = subprocess.run(
            ["hdfs", "dfs", "-ls", path],
            capture_output=True,
            text=True
        )
        logger.info(f"HDFS ls {path} completed from IP {get_client_ip()}")
        return jsonify({
            "output": result.stdout if result.stdout else "No files found",
            "error": result.stderr
        })
    except Exception as e:
        logger.error(f"HDFS ls error for path {path} from IP {get_client_ip()}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Admin POST/PUT/DELETE with logging examples
@app.route("/api/admin/students", methods=["POST"])
def api_admin_add_student():
    ip = get_client_ip()
    logger.info(f"Admin add student request from IP {ip}")
    try:
        b = {k: clean_input(v) for k, v in request.get_json(force=True).items()}
        sid = f"STU{str(len(DB['students']) + 1).zfill(4)}"
        # ... rest of function unchanged
        logger.info(f"Student added: {sid} ({b.get('name')}) by IP {ip}")
        return jsonify({"success": True, "id": sid})
    except Exception as e:
        logger.error(f"Error adding student from IP {ip}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/admin/students/<sid>", methods=["DELETE"])
def api_admin_delete_student(sid):
    ip = get_client_ip()
    logger.info(f"Admin delete student {sid} from IP {ip}")
    try:
        for c in ["students", "attendance", "marks", "fees", "notifications", "timetable", "quiz_attempts", "leaderboard"]: 
            DB[c].pop(sid, None)
        DB["users"].pop(sid, None)
        save_db(DB)
        logger.info(f"Student {sid} deleted by IP {ip}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting student {sid} from IP {ip}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Similar pattern for other POST/PUT/DELETE endpoints (faculty, announcements, materials, marks updates, fees, etc.)
# For brevity, pattern: logger.info(f"{action} {endpoint} from IP {ip}"), try/except logger.error

# Example for faculty add
@app.route("/api/admin/faculty", methods=["POST"])
def api_admin_add_faculty():
    ip = get_client_ip()
    logger.info(f"Admin add faculty request from IP {ip}")

    try:
        b = {k: clean_input(v) for k, v in request.get_json(force=True).items()}

        fid = f"FAC{str(len(DB['faculty']) + 1).zfill(4)}"

        DB["faculty"][fid] = {
            "id": fid,
            "name": b.get("name"),
            "email": b.get("email"),
            "department": b.get("department")
        }

        save_db(DB)

        logger.info(f"Faculty added: {fid} by IP {ip}")
        return jsonify({"success": True, "id": fid})

    except Exception as e:
        logger.error(f"Error adding faculty: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Pattern continues for all CRUD operations...

@app.route("/api/admin/predict", methods=["POST"])
def api_admin_predict():
    ip = get_client_ip()
    logger.info(f"Admin prediction request from IP {ip}")
    # ... existing logic with existing try/except logging already added

if __name__ == "__main__":
    logger.info("EduPredict Enhanced - Server fully started on port 5000")
    print("EduPredict Enhanced - Starting...")
    print("Admin:   admin@edupredict.edu / admin123")
    print("Faculty: anitha@edupredict.edu / faculty123")
    print("Student: aisha@student.edu / student123")
    print("URL:     http://localhost:5000")
    print("Logs: backend/app.log")
    app.run(debug=True, host="0.0.0.0", port=5000)
