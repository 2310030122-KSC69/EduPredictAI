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

# ── Input sanitizer ──────────────────────────────────────────────────────────
def clean_input(val):
    if isinstance(val, str):
        return " ".join(val.strip().split())
    return val

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "..", "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
DATA_DIR     = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)
app.secret_key = "edupredict-secret-2025"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
logger.info("EduPredict server starting up...")

# ── ML Artifacts ──────────────────────────────────────────────────────────────
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

ARTIFACTS    = load_artifacts()
MODEL_MAP    = {"random_forest": "rf_model", "gradient_boosting": "gb_model",
                "logistic_regression": "lr_model", "svm": "svm_model"}
FEATURE_COLS = ["study_hours_per_day", "attendance_rate", "previous_gpa",
                "assignments_completed", "sleep_hours", "extracurricular_activities",
                "parental_education", "internet_access", "tutoring_sessions",
                "stress_level", "motivation_score", "part_time_job"]
SUBJECTS_MAP = {
    "B.Tech CS":  ["Data Structures", "Algorithms", "DBMS", "OS", "CN", "ML"],
    "B.Tech ECE": ["Signals", "Electronics", "Communication", "VLSI", "Embedded"],
    "B.Tech ME":  ["Thermodynamics", "Fluid Mechanics", "Manufacturing", "Design"],
}

# ── Quiz Bank ─────────────────────────────────────────────────────────────────
QUIZ_BANK = {
    "Data Structures": [
        {"q": "Time complexity of binary search?",                          "opts": ["O(n)", "O(log n)", "O(n²)", "O(1)"],                                                              "ans": 1, "exp": "Binary search halves the search space each step → O(log n)"},
        {"q": "Which data structure uses LIFO order?",                      "opts": ["Queue", "Stack", "Heap", "Tree"],                                                                  "ans": 1, "exp": "Stack = Last In, First Out"},
        {"q": "Which traversal gives sorted output for BST?",               "opts": ["Pre-order", "Post-order", "In-order", "Level-order"],                                              "ans": 2, "exp": "In-order (Left→Root→Right) gives sorted output for a BST"},
        {"q": "Array vs Linked List: random access is O(1) in?",            "opts": ["Linked List", "Both", "Array", "Neither"],                                                         "ans": 2, "exp": "Array supports O(1) random access via index; linked list needs O(n) traversal"},
        {"q": "Worst case of QuickSort?",                                   "opts": ["O(n log n)", "O(n)", "O(n²)", "O(log n)"],                                                         "ans": 2, "exp": "QuickSort worst case O(n²) when pivot is always min/max element"},
        {"q": "Data structure used for BFS?",                               "opts": ["Stack", "Priority Queue", "Queue", "Array"],                                                       "ans": 2, "exp": "BFS uses a Queue (FIFO) to explore nodes level by level"},
        {"q": "Height of balanced BST with n nodes?",                       "opts": ["O(n)", "O(n²)", "O(log n)", "O(1)"],                                                              "ans": 2, "exp": "Balanced BST maintains O(log n) height"},
    ],
    "Algorithms": [
        {"q": "Merge Sort uses which technique?",                           "opts": ["Greedy", "Dynamic Programming", "Divide and Conquer", "Backtracking"],                             "ans": 2, "exp": "Merge Sort divides, sorts halves, then merges — classic Divide & Conquer"},
        {"q": "Dijkstra's algorithm finds?",                                "opts": ["MST", "Shortest path", "Topological order", "Cycles"],                                             "ans": 1, "exp": "Dijkstra finds shortest path from source to all vertices in weighted graph"},
        {"q": "Best case of Bubble Sort?",                                  "opts": ["O(n²)", "O(n log n)", "O(n)", "O(1)"],                                                             "ans": 2, "exp": "Optimized Bubble Sort is O(n) when array is already sorted"},
        {"q": "Floyd-Warshall uses which approach?",                        "opts": ["Greedy", "Divide & Conquer", "Dynamic Programming", "Backtracking"],                              "ans": 2, "exp": "Floyd-Warshall uses dynamic programming to find all-pairs shortest paths"},
    ],
    "DBMS": [
        {"q": "ACID stands for?",                                           "opts": ["Atomicity Consistency Isolation Durability", "Array Cursor Index Data", "Auto Commit Insert Delete", "None"], "ans": 0, "exp": "ACID ensures reliable transaction processing in databases"},
        {"q": "Which normal form removes partial dependencies?",            "opts": ["1NF", "2NF", "3NF", "BCNF"],                                                                      "ans": 1, "exp": "2NF removes partial dependencies on composite primary keys"},
        {"q": "SQL command to permanently remove a table?",                 "opts": ["DELETE", "TRUNCATE", "DROP", "REMOVE"],                                                            "ans": 2, "exp": "DROP TABLE removes the table structure; DELETE removes only rows"},
        {"q": "Which JOIN returns all rows from both tables?",              "opts": ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN"],                                        "ans": 3, "exp": "FULL OUTER JOIN returns all rows; missing matches get NULL values"},
        {"q": "An index in a database is?",                                 "opts": ["Stored procedure", "Fast lookup data structure", "A constraint", "A view"],                        "ans": 1, "exp": "Index improves data retrieval speed at cost of storage and write speed"},
    ],
    "OS": [
        {"q": "Which scheduling minimizes average waiting time?",           "opts": ["FCFS", "SJF", "Round Robin", "Priority"],                                                          "ans": 1, "exp": "SJF (Shortest Job First) provably minimizes average waiting time"},
        {"q": "Deadlock requires which four conditions?",                   "opts": ["Only mutual exclusion", "Mutual excl, Hold&Wait, No preemption, Circular wait", "Only circular wait", "None"], "ans": 1, "exp": "All four Coffman conditions must hold simultaneously for deadlock"},
        {"q": "Virtual memory allows?",                                     "opts": ["Faster CPU", "Programs larger than RAM", "Parallel execution", "Disk compression"],               "ans": 1, "exp": "Virtual memory uses disk as extended RAM for larger program execution"},
        {"q": "Thrashing is caused by?",                                    "opts": ["CPU overheating", "Excessive paging", "Hard disk failure", "Cache miss"],                         "ans": 1, "exp": "Thrashing: system spends more time swapping pages than executing"},
    ],
    "CN": [
        {"q": "IP operates at which OSI layer?",                            "opts": ["Application", "Transport", "Network", "Data Link"],                                                "ans": 2, "exp": "IP (Internet Protocol) works at Layer 3 — the Network layer"},
        {"q": "Which protocol guarantees delivery?",                        "opts": ["UDP", "Both", "TCP", "Neither"],                                                                   "ans": 2, "exp": "TCP is connection-oriented with acknowledgment, retransmission, and ordering"},
        {"q": "IPv4 address size?",                                         "opts": ["16 bits", "32 bits", "64 bits", "128 bits"],                                                       "ans": 1, "exp": "IPv4 = 32 bits (4 octets), IPv6 = 128 bits"},
        {"q": "HTTPS uses which port?",                                     "opts": ["80", "21", "443", "22"],                                                                           "ans": 2, "exp": "HTTPS=443, HTTP=80, FTP=21, SSH=22"},
    ],
    "ML": [
        {"q": "Overfitting means?",                                         "opts": ["Model too simple", "Memorizes training data", "High accuracy always", "Model is fast"],            "ans": 1, "exp": "Overfitting: model learns noise in training data, fails to generalize"},
        {"q": "Gradient Descent minimizes?",                                "opts": ["Data size", "Loss function", "Model complexity", "Feature count"],                                 "ans": 1, "exp": "Gradient Descent iteratively adjusts weights to minimize the loss function"},
        {"q": "Random Forest is an ensemble of?",                           "opts": ["Neural Networks", "SVMs", "Decision Trees", "K-NN"],                                              "ans": 2, "exp": "Random Forest aggregates predictions from multiple decision trees"},
        {"q": "Kernel trick in SVM?",                                       "opts": ["Data compression", "Maps to higher dimensions", "Reduces features", "Speeds training"],           "ans": 1, "exp": "Kernel trick implicitly maps data to higher-dimensional space for non-linear classification"},
        {"q": "Cross-validation is used for?",                              "opts": ["Feature selection", "Model evaluation", "Data cleaning", "Training speed"],                       "ans": 1, "exp": "Cross-validation estimates model performance by rotating train/test splits"},
    ],
    "General": [
        {"q": "CPU stands for?",                                            "opts": ["Central Processing Unit", "Computer Processing Unit", "Core Parallel Unit", "None"],              "ans": 0, "exp": "CPU = Central Processing Unit — primary component executing program instructions"},
        {"q": "What is RAM?",                                               "opts": ["Read Access Memory", "Random Access Memory", "Rapid Array Memory", "None"],                       "ans": 1, "exp": "RAM = Random Access Memory — volatile primary memory for active processes"},
        {"q": "Git is used for?",                                           "opts": ["Database management", "Version control", "Web hosting", "Monitoring"],                            "ans": 1, "exp": "Git is a distributed version control system for tracking code changes"},
        {"q": "HTTP stands for?",                                           "opts": ["HyperText Transfer Protocol", "High Transfer Text Protocol", "HyperText Terminal Protocol", "None"], "ans": 0, "exp": "HTTP = HyperText Transfer Protocol — foundation of web communication"},
        {"q": "Which language styles web pages?",                           "opts": ["JavaScript", "HTML", "CSS", "Python"],                                                             "ans": 2, "exp": "CSS (Cascading Style Sheets) controls visual presentation of HTML content"},
    ],
}

# ── DB helpers ────────────────────────────────────────────────────────────────
DB_FILE = os.path.join(DATA_DIR, "db.json")

def load_db():
    logger.info("Loading database...")
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            logger.info("Database loaded from file")
            return json.load(f)
    logger.warning("Database file not found, initializing fresh database...")
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
    return (m.get("insem1", 0) + m.get("insem_lab", 0) +
            m.get("insem2", 0) + m.get("end_sem", 0) + m.get("end_lab", 0))

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

def _make_timetable(subs):
    days  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    slots = ["9:00-10:00", "10:00-11:00", "11:15-12:15", "12:15-1:00", "2:00-3:00", "3:00-4:00"]
    tt    = {day: {} for day in days}
    random.seed(42)
    for day in days:
        for slot in slots:
            if slot == "12:15-1:00":
                tt[day][slot] = "Lunch Break"
            elif random.random() > 0.3 and subs:
                tt[day][slot] = random.choice(subs)
            else:
                tt[day][slot] = "Free"
    return tt

def init_db():
    db = {
        "users": {
            "admin001": {
                "id": "admin001", "role": "admin", "name": "Dr. Rajesh Kumar",
                "email": "admin@edupredict.edu", "password": hash_pw("admin123"),
                "department": "Computer Science", "phone": "9876543210",
                "created_at": str(datetime.now()), "avatar": "RK",
            },
            "fac001": {
                "id": "fac001", "role": "faculty", "name": "Prof. Anitha Rao",
                "email": "anitha@edupredict.edu", "password": hash_pw("faculty123"),
                "department": "Computer Science", "phone": "9876543211",
                "subjects": ["Data Structures", "Algorithms", "ML"],
                "created_at": str(datetime.now()), "avatar": "AR",
            },
            "fac002": {
                "id": "fac002", "role": "faculty", "name": "Prof. Suresh Babu",
                "email": "suresh@edupredict.edu", "password": hash_pw("faculty123"),
                "department": "Computer Science", "phone": "9876543212",
                "subjects": ["DBMS", "OS", "CN"],
                "created_at": str(datetime.now()), "avatar": "SB",
            },
        },
        "faculty": {
            "fac001": {"id": "fac001", "name": "Prof. Anitha Rao",  "email": "anitha@edupredict.edu",
                       "department": "CS", "subjects": ["Data Structures", "Algorithms", "ML"],
                       "phone": "9876543211", "avatar": "AR", "designation": "Associate Professor"},
            "fac002": {"id": "fac002", "name": "Prof. Suresh Babu", "email": "suresh@edupredict.edu",
                       "department": "CS", "subjects": ["DBMS", "OS", "CN"],
                       "phone": "9876543212", "avatar": "SB", "designation": "Assistant Professor"},
        },
        "students": {}, "attendance": {}, "marks": {}, "fees": {},
        "announcements": [], "notifications": {}, "timetable": {},
        "study_materials": [], "feedback": [], "quiz_attempts": {}, "leaderboard": {},
    }

    samples = [
        {"name": "Aisha Sharma",   "email": "aisha@student.edu",  "roll": "CS2021001", "course": "B.Tech CS",  "semester": 6, "phone": "9000000001", "dob": "2002-05-14", "address": "Hyderabad",   "guardian": "Ramesh Sharma",  "guardian_phone": "9000000010", "avatar": "AS"},
        {"name": "Mohammed Ali",   "email": "mali@student.edu",   "roll": "CS2021002", "course": "B.Tech CS",  "semester": 6, "phone": "9000000002", "dob": "2002-08-22", "address": "Secunderabad","guardian": "Ahmed Ali",       "guardian_phone": "9000000020", "avatar": "MA"},
        {"name": "Priya Reddy",    "email": "priya@student.edu",  "roll": "CS2021003", "course": "B.Tech CS",  "semester": 5, "phone": "9000000003", "dob": "2003-01-10", "address": "Warangal",    "guardian": "Srinivas Reddy",  "guardian_phone": "9000000030", "avatar": "PR"},
        {"name": "Vikram Singh",   "email": "vikram@student.edu", "roll": "EC2022001", "course": "B.Tech ECE", "semester": 4, "phone": "9000000004", "dob": "2003-11-30", "address": "Karimnagar",  "guardian": "Balveer Singh",   "guardian_phone": "9000000040", "avatar": "VS"},
        {"name": "Sneha Patel",    "email": "sneha@student.edu",  "roll": "ME2022002", "course": "B.Tech ME",  "semester": 4, "phone": "9000000005", "dob": "2003-07-19", "address": "Nizamabad",   "guardian": "Ravi Patel",      "guardian_phone": "9000000050", "avatar": "SP"},
    ]

    np.random.seed(42)
    today = date.today()

    for i, s in enumerate(samples):
        sid = f"STU{str(i + 1).zfill(4)}"
        db["users"][sid] = {
            "id": sid, "role": "student", "name": s["name"],
            "email": s["email"], "password": hash_pw("student123"),
            "created_at": str(datetime.now()),
        }
        db["students"][sid] = {
            "id": sid, "roll_no": s["roll"], "name": s["name"], "email": s["email"],
            "phone": s["phone"], "dob": s["dob"], "address": s["address"],
            "course": s["course"], "semester": s["semester"],
            "guardian": s["guardian"], "guardian_phone": s["guardian_phone"],
            "avatar": s["avatar"], "admission_year": 2021 + (i % 2),
            "created_at": str(datetime.now()),
        }
        subs = SUBJECTS_MAP.get(s["course"], SUBJECTS_MAP["B.Tech CS"])

        db["attendance"][sid] = {}
        for sub in subs:
            records = []
            for day_off in range(60, 0, -1):
                d = today - timedelta(days=day_off)
                if d.weekday() < 5:
                    records.append({"date": str(d), "status": "present" if np.random.random() < 0.82 else "absent"})
            present = sum(1 for r in records if r["status"] == "present")
            db["attendance"][sid][sub] = {"total": len(records), "present": present, "records": records}

        db["marks"][sid] = {
            sub: {
                "insem1":    int(np.random.randint(14, 25)),
                "insem_lab": int(np.random.randint(14, 25)),
                "insem2":    int(np.random.randint(14, 25)),
                "end_sem":   int(np.random.randint(28, 50)),
                "end_lab":   int(np.random.randint(14, 25)),
            }
            for sub in subs
        }

        db["fees"][sid] = {"total_due": 85000, "paid": 0, "transactions": []}
        paid = int(np.random.choice([0, 42500, 85000]))
        if paid > 0:
            db["fees"][sid]["paid"] = paid
            db["fees"][sid]["transactions"].append({
                "id": str(uuid.uuid4())[:8], "amount": paid, "date": "2024-08-01",
                "mode": "Online", "receipt": f"RCP{i + 1:04d}001", "status": "Paid",
            })

        db["notifications"][sid] = []
        db["timetable"][sid]     = _make_timetable(subs)

        xp = int(np.random.randint(50, 600))
        db["quiz_attempts"][sid] = {
            "total_answered": int(np.random.randint(10, 50)),
            "total_correct":  int(np.random.randint(5, 40)),
            "streak":         int(np.random.randint(0, 10)),
            "xp": xp, "level": 1 + xp // 200, "badges": [],
            "subject_stats": {sub: {"answered": int(np.random.randint(2, 10)), "correct": int(np.random.randint(1, 8))} for sub in subs},
        }
        db["leaderboard"][sid] = {"name": s["name"], "avatar": s["avatar"], "xp": xp, "level": 1 + xp // 200}

    db["announcements"] = [
        {"id": "ann001", "title": "Mid Semester Exams",     "body": "Mid semester exams from 15th Oct. Prepare well!", "date": "2024-09-25", "priority": "high",   "by": "Admin"},
        {"id": "ann002", "title": "Fee Payment Reminder",   "body": "Last date for fee payment is 30th September.",    "date": "2024-09-20", "priority": "medium", "by": "Admin"},
        {"id": "ann003", "title": "Hackathon Registration", "body": "Register for inter-college hackathon by 5th Oct.","date": "2024-09-18", "priority": "low",    "by": "Admin"},
    ]
    db["study_materials"] = [
        {"id": "mat001", "title": "Data Structures Notes", "subject": "Data Structures", "type": "PDF", "url": "#", "uploaded_by": "Admin", "date": "2024-09-10"},
        {"id": "mat002", "title": "DBMS Lab Manual",       "subject": "DBMS",            "type": "PDF", "url": "#", "uploaded_by": "Admin", "date": "2024-09-12"},
    ]
    db["feedback"] = []
    save_db(db)
    return db

DB = load_db()

def current_user():
    uid = session.get("user_id")
    return DB["users"].get(uid) if uid else None

def get_client_ip():
    return request.environ.get("REMOTE_ADDR", "unknown")

# ── Request logging ───────────────────────────────────────────────────────────
@app.before_request
def log_request():
    ip       = get_client_ip()
    method   = request.method
    endpoint = request.endpoint or request.path
    logger.info(f"Request: {method} {endpoint} from IP {ip}")

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    u = current_user()
    if u:
        if u["role"] == "admin":   return redirect("/admin/dashboard")
        if u["role"] == "faculty": return redirect("/faculty/dashboard")
        return redirect("/student/dashboard")
    # ✅ FIX: render login directly instead of redirect to avoid loop
    try:
        return render_template("login.html")
    except Exception as e:
        logger.error(f"Login template error: {e}")
        return "<h2>Login page unavailable</h2>", 500

@app.route("/login")
def login_page():
    try:
        return render_template("login.html")
    except Exception as e:
        logger.error(f"Login template error: {e}")
        return "<h2>Login page unavailable</h2>", 500

@app.route("/unauthorized")
def unauthorized():
    return "<h2>403 Unauthorized</h2><a href='/login'>Back to Login</a>", 403

# ── Dynamic page routes ───────────────────────────────────────────────────────
# ✅ FIX: unique endpoint name derived from URL path (not template name)
#         and proper closure using default-argument capture of t and r
_PAGE_ROUTES = [
    ("/admin/dashboard",    "admin_dashboard.html",    ["admin"]),
    ("/admin/students",     "admin_students.html",     ["admin"]),
    ("/admin/attendance",   "admin_attendance.html",   ["admin"]),
    ("/admin/marks",        "admin_marks.html",        ["admin"]),
    ("/admin/fees",         "admin_fees.html",         ["admin"]),
    ("/admin/predict",      "admin_predict.html",      ["admin"]),
    ("/admin/announcements","admin_announcements.html",["admin"]),
    ("/admin/reports",      "admin_reports.html",      ["admin"]),
    ("/admin/timetable",    "admin_timetable.html",    ["admin"]),
    ("/admin/faculty",      "admin_faculty.html",      ["admin"]),
    ("/faculty/dashboard",  "faculty_dashboard.html",  ["faculty", "admin"]),
    ("/faculty/attendance", "faculty_attendance.html", ["faculty", "admin"]),
    ("/faculty/marks",      "faculty_marks.html",      ["faculty", "admin"]),
    ("/student/dashboard",  "student_dashboard.html",  ["student"]),
    ("/student/attendance", "student_attendance.html", ["student"]),
    ("/student/marks",      "student_marks.html",      ["student"]),
    ("/student/fees",       "student_fees.html",       ["student"]),
    ("/student/profile",    "student_profile.html",    ["student"]),
    ("/student/timetable",  "student_timetable.html",  ["student"]),
    ("/student/materials",  "student_materials.html",  ["student"]),
    ("/student/quiz",       "student_quiz.html",       ["student"]),
]

def _make_page_view(template, allowed_roles):
    """
    ✅ FIX: uses default-argument capture (template=template, roles=allowed_roles)
    so each closure correctly captures its own template and roles,
    not the last values from the loop.
    """
    def page_view(template=template, roles=allowed_roles):
        u = current_user()
        if not u or u["role"] not in roles:
            logger.warning(f"Unauthorized access attempt to {template} from IP {get_client_ip()}")
            return redirect("/login")
        try:
            return render_template(template, user=u)
        except Exception as e:
            logger.error(f"Template render error for {template}: {e}")
            return f"<h2>Page error: {template}</h2><a href='/'>Home</a>", 500
    return page_view

for _path, _tpl, _roles in _PAGE_ROUTES:
    # ✅ FIX: endpoint name derived from URL path, guaranteed unique
    _endpoint = "page" + _path.replace("/", "_")
    _view_func = _make_page_view(_tpl, _roles)
    _view_func.__name__ = _endpoint          # Flask uses __name__ as endpoint key
    app.add_url_rule(_path, endpoint=_endpoint, view_func=_view_func)

@app.route("/logout")
def logout():
    uid = session.get("user_id")
    ip  = get_client_ip()
    if uid:
        logger.info(f"User logout: {uid} ({DB['users'].get(uid, {}).get('email', 'unknown')}) from IP {ip}")
    session.clear()
    return redirect("/login")

# ── HADOOP API ────────────────────────────────────────────────────────────────
@app.route("/api/hdfs/list", methods=["POST"])
def api_hdfs_list():
    ip = get_client_ip()
    logger.info(f"HDFS list request from IP {ip}")
    data = request.get_json(force=True) or {}
    b    = {k: clean_input(v) for k, v in data.items()}
    path = b.get("path", "/")
    if not path:
        logger.warning(f"HDFS empty path from IP {ip}")
        return jsonify({"error": "Path cannot be empty"}), 400
    try:
        result = subprocess.run(
            ["hdfs", "dfs", "-ls", path],
            capture_output=True, text=True,
        )
        logger.info(f"HDFS ls {path} completed from IP {ip}")
        return jsonify({
            "output": result.stdout if result.stdout else "No files found",
            "error":  result.stderr,
        })
    except FileNotFoundError:
        logger.error(f"HDFS binary not found (Hadoop not installed) from IP {ip}")
        return jsonify({"error": "Hadoop not installed on this server"}), 503
    except Exception as e:
        logger.error(f"HDFS ls error for path {path} from IP {ip}: {e}")
        return jsonify({"error": str(e)}), 500

# ── AUTH API ──────────────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def api_login():
    ip = get_client_ip()
    try:
        raw   = request.get_json(force=True) or {}
        b     = {k: clean_input(v) for k, v in raw.items()}
        email = b.get("email", "").lower().strip()
        pw    = b.get("password", "").strip()
        logger.info(f"Login attempt for {email} from IP {ip}")
        if not email or not pw:
            logger.warning(f"Login attempt with missing credentials from IP {ip}")
            return jsonify({"error": "Email and password required"}), 400
        for uid, u in DB["users"].items():
            if u.get("email", "").lower() == email and u.get("password") == hash_pw(pw):
                session["user_id"] = uid
                logger.info(f"Login successful: {uid} ({u['role']}) {u['name']} from IP {ip}")
                return jsonify({"success": True, "role": u["role"], "name": u["name"]})
        logger.warning(f"Login failed for {email} from IP {ip}")
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        logger.error(f"Login error from IP {ip}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/auth/me")
def api_me():
    u = current_user()
    if not u:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify({"id": session["user_id"], "name": u["name"], "role": u["role"], "email": u.get("email", "")})

# ── ADMIN API ─────────────────────────────────────────────────────────────────
@app.route("/api/admin/stats")
def api_admin_stats():
    total    = len(DB["students"])
    att_rates = [
        sum(d["present"] / d["total"] * 100 for d in subs.values() if d["total"] > 0) /
        max(1, len([d for d in subs.values() if d["total"] > 0]))
        for subs in DB["attendance"].values()
    ]
    avg_att   = round(sum(att_rates) / len(att_rates), 1) if att_rates else 0
    fee_total = sum(f.get("total_due", 0) for f in DB["fees"].values())
    fee_paid  = sum(f.get("paid",      0) for f in DB["fees"].values())
    passing   = sum(
        1 for sid, marks in DB["marks"].items()
        if marks and sum(compute_total(m) for m in marks.values()) / len(marks) / compute_max() * 100 >= 40
    )
    return jsonify({
        "total_students":      total,
        "avg_attendance":      avg_att,
        "fee_collection":      fee_paid,
        "fee_pending":         fee_total - fee_paid,
        "passing_students":    passing,
        "total_faculty":       len(DB.get("faculty", {})),
        "recent_announcements": len(DB["announcements"]),
    })

@app.route("/api/admin/students", methods=["GET"])
def api_admin_students():
    result = []
    for sid, s in DB["students"].items():
        att   = DB["attendance"].get(sid, {})
        fee   = DB["fees"].get(sid, {})
        rates = [d["present"] / d["total"] * 100 for d in att.values() if d["total"] > 0]
        avg_att = round(sum(rates) / len(rates), 1) if rates else 0
        due  = fee.get("total_due", 0)
        paid = fee.get("paid", 0)
        result.append({
            **s,
            "avg_attendance": avg_att,
            "fee_status": "Paid" if paid >= due else ("Partial" if paid > 0 else "Unpaid"),
        })
    return jsonify(result)

@app.route("/api/admin/students", methods=["POST"])
def api_admin_add_student():
    ip = get_client_ip()
    logger.info(f"Admin add student request from IP {ip}")
    try:
        raw  = request.get_json(force=True) or {}
        b    = {k: clean_input(v) for k, v in raw.items()}
        sid  = f"STU{str(len(DB['students']) + 1).zfill(4)}"
        email = b.get("email", f"{sid}@student.edu")
        DB["users"][sid] = {
            "id": sid, "role": "student", "name": b["name"],
            "email": email, "password": hash_pw("student123"),
            "created_at": str(datetime.now()),
        }
        DB["students"][sid] = {
            "id": sid, "roll_no": b.get("roll_no", sid), "name": b["name"],
            "email": email, "phone": b.get("phone", ""), "dob": b.get("dob", ""),
            "address": b.get("address", ""), "course": b.get("course", "B.Tech CS"),
            "semester": int(b.get("semester", 1)),
            "guardian": b.get("guardian", ""), "guardian_phone": b.get("guardian_phone", ""),
            "avatar": "".join(w[0] for w in b["name"].split()[:2]).upper(),
            "admission_year": datetime.now().year, "created_at": str(datetime.now()),
        }
        course = b.get("course", "B.Tech CS")
        subs   = SUBJECTS_MAP.get(course, ["Subject 1"])
        DB["attendance"][sid]   = {s: {"total": 0, "present": 0, "records": []} for s in subs}
        DB["marks"][sid]        = {s: make_marks_entry() for s in subs}
        DB["fees"][sid]         = {"total_due": 85000, "paid": 0, "transactions": []}
        DB["notifications"][sid] = []
        DB["timetable"][sid]    = _make_timetable(subs)
        DB["quiz_attempts"][sid] = {
            "total_answered": 0, "total_correct": 0, "streak": 0,
            "xp": 0, "level": 1, "badges": [],
            "subject_stats": {s: {"answered": 0, "correct": 0} for s in subs},
        }
        DB["leaderboard"][sid] = {"name": b["name"], "avatar": DB["students"][sid]["avatar"], "xp": 0, "level": 1}
        save_db(DB)
        logger.info(f"Student added: {sid} ({b.get('name')}) by IP {ip}")
        return jsonify({"success": True, "id": sid})
    except Exception as e:
        logger.error(f"Error adding student from IP {ip}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/admin/students/<sid>", methods=["PUT"])
def api_admin_update_student(sid):
    if sid not in DB["students"]:
        return jsonify({"error": "Not found"}), 404
    b = {k: clean_input(v) for k, v in (request.get_json(force=True) or {}).items()}
    for k in ["name", "phone", "address", "course", "semester", "guardian", "guardian_phone", "dob"]:
        if k in b:
            DB["students"][sid][k] = b[k]
            if k == "name":
                DB["users"][sid]["name"] = b[k]
                DB["students"][sid]["avatar"] = "".join(w[0] for w in b[k].split()[:2]).upper()
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/admin/students/<sid>", methods=["DELETE"])
def api_admin_delete_student(sid):
    ip = get_client_ip()
    logger.info(f"Admin delete student {sid} from IP {ip}")
    try:
        for c in ["students", "attendance", "marks", "fees", "notifications",
                  "timetable", "quiz_attempts", "leaderboard"]:
            DB[c].pop(sid, None)
        DB["users"].pop(sid, None)
        save_db(DB)
        logger.info(f"Student {sid} deleted by IP {ip}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting student {sid} from IP {ip}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/admin/attendance")
def api_admin_attendance():
    result = {}
    for sid, s in DB["students"].items():
        att     = DB["attendance"].get(sid, {})
        rates   = {sub: round(d["present"] / d["total"] * 100, 1) if d["total"] > 0 else 0 for sub, d in att.items()}
        overall = round(sum(rates.values()) / len(rates), 1) if rates else 0
        result[sid] = {"name": s["name"], "roll_no": s["roll_no"], "avatar": s.get("avatar", ""),
                       "subjects": att, "rates": rates, "overall": overall}
    return jsonify(result)

@app.route("/api/admin/attendance/<sid>", methods=["PUT"])
def api_admin_update_attendance(sid):
    if sid not in DB["attendance"]:
        return jsonify({"error": "Not found"}), 404
    b        = request.get_json(force=True) or {}
    subject  = b.get("subject")
    action   = b.get("action")
    date_str = b.get("date", str(date.today()))
    if subject not in DB["attendance"][sid]:
        DB["attendance"][sid][subject] = {"total": 0, "present": 0, "records": []}
    att = DB["attendance"][sid][subject]
    att["total"] += 1
    if action == "present":
        att["present"] += 1
    att["records"].append({"date": date_str, "status": action})
    rate = att["present"] / att["total"] * 100
    if rate < 75:
        DB["notifications"].setdefault(sid, []).append({
            "id": str(uuid.uuid4())[:8], "type": "warning",
            "msg": f"Attendance in {subject} is {rate:.1f}% — below 75%!",
            "date": str(date.today()), "read": False,
        })
    save_db(DB)
    return jsonify({"success": True, "rate": round(rate, 1)})

@app.route("/api/admin/marks")
def api_admin_marks():
    result = {}
    for sid, s in DB["students"].items():
        marks       = DB["marks"].get(sid, {})
        subjects_out = {}
        for sub, m in marks.items():
            if "mid1" in m and "insem1" not in m:
                m = {"insem1": m.get("mid1", 0), "insem_lab": 0,
                     "insem2": m.get("mid2", 0), "end_sem": m.get("final", 0),
                     "end_lab": m.get("assignment", 0)}
                DB["marks"][sid][sub] = m
            total = compute_total(m)
            pct   = round(total / compute_max() * 100, 1)
            subjects_out[sub] = {**m, "total": total, "max": compute_max(), "percentage": pct, "grade": get_grade(pct)}
        result[sid] = {"name": s["name"], "roll_no": s["roll_no"], "avatar": s.get("avatar", ""), "subjects": subjects_out}
    save_db(DB)
    return jsonify(result)

@app.route("/api/admin/marks/<sid>", methods=["PUT"])
def api_admin_update_marks(sid):
    if sid not in DB["marks"]:
        return jsonify({"error": "Not found"}), 404
    b       = request.get_json(force=True) or {}
    subject = b.get("subject")
    if subject not in DB["marks"][sid]:
        DB["marks"][sid][subject] = make_marks_entry()
    for field in ["insem1", "insem_lab", "insem2", "end_sem", "end_lab"]:
        if field in b:
            DB["marks"][sid][subject][field] = int(b[field])
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/admin/faculty", methods=["GET"])
def api_admin_get_faculty():
    return jsonify(list(DB.get("faculty", {}).values()))

@app.route("/api/admin/faculty", methods=["POST"])
def api_admin_add_faculty():
    ip = get_client_ip()
    logger.info(f"Admin add faculty request from IP {ip}")
    try:
        raw    = request.get_json(force=True) or {}
        b      = {k: clean_input(v) for k, v in raw.items()}
        fid    = f"fac{str(len(DB.get('faculty', {})) + 1).zfill(3)}"
        email  = b.get("email", f"{fid}@edupredict.edu")
        avatar = "".join(w[0] for w in b["name"].split()[:2]).upper()
        DB["users"][fid] = {
            "id": fid, "role": "faculty", "name": b["name"],
            "email": email, "password": hash_pw("faculty123"),
            "created_at": str(datetime.now()),
        }
        DB.setdefault("faculty", {})[fid] = {
            "id": fid, "name": b["name"], "email": email,
            "department": b.get("department", "CS"),
            "subjects":   b.get("subjects", []),
            "phone":      b.get("phone", ""),
            "avatar":     avatar,
            "designation": b.get("designation", "Assistant Professor"),
        }
        save_db(DB)
        logger.info(f"Faculty added: {fid} ({b.get('name')}) by IP {ip}")
        return jsonify({"success": True, "id": fid})
    except Exception as e:
        logger.error(f"Error adding faculty from IP {ip}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/admin/faculty/<fid>", methods=["PUT"])
def api_admin_update_faculty(fid):
    if fid not in DB.get("faculty", {}):
        return jsonify({"error": "Not found"}), 404
    b = {k: clean_input(v) for k, v in (request.get_json(force=True) or {}).items()}
    for k in ["name", "phone", "department", "subjects", "designation"]:
        if k in b:
            DB["faculty"][fid][k] = b[k]
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/admin/faculty/<fid>", methods=["DELETE"])
def api_admin_delete_faculty(fid):
    ip = get_client_ip()
    logger.info(f"Admin delete faculty {fid} from IP {ip}")
    DB.get("faculty", {}).pop(fid, None)
    DB["users"].pop(fid, None)
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/admin/reports/attendance")
def api_reports_attendance():
    period     = request.args.get("period", "day")
    target     = request.args.get("date", str(date.today()))
    sid_filter = request.args.get("sid", "")
    try:
        target_date = datetime.strptime(target, "%Y-%m-%d").date()
    except Exception:
        target_date = date.today()
    if period == "day":
        dates_range = [target_date]
    elif period == "week":
        start       = target_date - timedelta(days=target_date.weekday())
        dates_range = [start + timedelta(days=i) for i in range(7)]
    else:
        start = target_date.replace(day=1)
        end   = start.replace(month=start.month + 1) if start.month < 12 else start.replace(year=start.year + 1, month=1)
        dates_range = [start + timedelta(days=i) for i in range((end - start).days)]
    date_strs = {str(d) for d in dates_range}
    report    = []
    for sid, s in DB["students"].items():
        if sid_filter and sid != sid_filter:
            continue
        att          = DB["attendance"].get(sid, {})
        student_data = {"sid": sid, "name": s["name"], "roll_no": s["roll_no"], "avatar": s.get("avatar", ""), "subjects": {}}
        for sub, d in att.items():
            recs    = [r for r in d.get("records", []) if r["date"] in date_strs]
            present = sum(1 for r in recs if r["status"] == "present")
            total   = len(recs)
            student_data["subjects"][sub] = {
                "records": recs, "present": present, "total": total,
                "rate":    round(present / total * 100, 1) if total > 0 else None,
            }
        report.append(student_data)
    return jsonify({"period": period, "dates": [str(d) for d in dates_range], "report": report})

@app.route("/api/admin/reports/marks")
def api_reports_marks():
    sid_filter = request.args.get("sid", "")
    report     = []
    for sid, s in DB["students"].items():
        if sid_filter and sid != sid_filter:
            continue
        marks        = DB["marks"].get(sid, {})
        subjects_out = {}
        for sub, m in marks.items():
            if "mid1" in m and "insem1" not in m:
                m = {"insem1": m.get("mid1", 0), "insem_lab": 0,
                     "insem2": m.get("mid2", 0), "end_sem": m.get("final", 0),
                     "end_lab": m.get("assignment", 0)}
            total = compute_total(m)
            pct   = round(total / compute_max() * 100, 1)
            subjects_out[sub] = {**m, "total": total, "percentage": pct, "grade": get_grade(pct)}
        report.append({"sid": sid, "name": s["name"], "roll_no": s["roll_no"], "avatar": s.get("avatar", ""), "subjects": subjects_out})
    return jsonify({"report": report})

@app.route("/api/admin/reports/summary")
def api_reports_summary():
    period = request.args.get("period", "week")
    today  = date.today()
    if period == "week":
        start = today - timedelta(days=today.weekday())
        dates = {str(start + timedelta(days=i)) for i in range(7)}
        label = f"Week of {start.strftime('%d %b %Y')}"
    elif period == "month":
        start = today.replace(day=1)
        dates = {str(start + timedelta(days=i)) for i in range(32) if (start + timedelta(days=i)).month == start.month}
        label = today.strftime("%B %Y")
    else:
        dates = {str(today)}
        label = today.strftime("%d %B %Y")
    total_present = total_absent = 0
    low_att = []
    for sid, subs in DB["attendance"].items():
        s_present = s_total = 0
        for sub, d in subs.items():
            recs       = [r for r in d.get("records", []) if r["date"] in dates]
            s_present += sum(1 for r in recs if r["status"] == "present")
            s_total   += len(recs)
        total_present += s_present
        total_absent  += (s_total - s_present)
        if s_total > 0 and s_present / s_total * 100 < 75:
            st = DB["students"].get(sid, {})
            low_att.append({"name": st.get("name", ""), "roll_no": st.get("roll_no", ""),
                            "rate": round(s_present / s_total * 100, 1)})
    return jsonify({
        "period": period, "label": label,
        "total_present":  total_present, "total_absent": total_absent,
        "attendance_rate": round(total_present / (total_present + total_absent) * 100, 1)
                          if (total_present + total_absent) > 0 else 0,
        "low_attendance_students": low_att,
    })

@app.route("/api/admin/fees")
def api_admin_fees():
    result = []
    for sid, s in DB["students"].items():
        fee  = DB["fees"].get(sid, {})
        due  = fee.get("total_due", 0)
        paid = fee.get("paid", 0)
        result.append({
            "id": sid, "name": s["name"], "roll_no": s["roll_no"],
            "avatar": s.get("avatar", ""), "course": s.get("course", ""),
            "total_due": due, "paid": paid, "balance": due - paid,
            "status": "Paid" if paid >= due else ("Partial" if paid > 0 else "Unpaid"),
            "transactions": fee.get("transactions", []),
        })
    return jsonify(result)

@app.route("/api/admin/fees/<sid>", methods=["POST"])
def api_admin_add_payment(sid):
    if sid not in DB["fees"]:
        return jsonify({"error": "Not found"}), 404
    b      = request.get_json(force=True) or {}
    amount = int(b.get("amount", 0))
    if amount <= 0:
        return jsonify({"error": "Invalid amount"}), 400
    txn_id  = str(uuid.uuid4())[:8].upper()
    receipt = f"RCP{sid[-4:]}{len(DB['fees'][sid]['transactions']) + 1:03d}"
    DB["fees"][sid]["paid"] += amount
    DB["fees"][sid]["transactions"].append({
        "id": txn_id, "amount": amount, "date": str(date.today()),
        "mode": b.get("mode", "Cash"), "receipt": receipt, "status": "Paid",
    })
    DB["notifications"].setdefault(sid, []).append({
        "id": str(uuid.uuid4())[:8], "type": "success",
        "msg": f"Fee payment of ₹{amount:,} received. Receipt: {receipt}",
        "date": str(date.today()), "read": False,
    })
    save_db(DB)
    return jsonify({"success": True, "receipt": receipt})

@app.route("/api/admin/announcements", methods=["GET"])
def api_admin_get_announcements():
    return jsonify(DB["announcements"])

@app.route("/api/admin/announcements", methods=["POST"])
def api_admin_add_announcement():
    b = {k: clean_input(v) for k, v in (request.get_json(force=True) or {}).items()}
    u = current_user()
    ann = {
        "id": str(uuid.uuid4())[:8], "title": b.get("title", ""),
        "body": b.get("body", ""), "priority": b.get("priority", "medium"),
        "date": str(date.today()), "by": u["name"] if u else "Admin",
    }
    DB["announcements"].insert(0, ann)
    for sid in DB["students"]:
        DB["notifications"].setdefault(sid, []).append({
            "id": str(uuid.uuid4())[:8], "type": "info",
            "msg": f"New: {ann['title']}", "date": str(date.today()), "read": False,
        })
    save_db(DB)
    return jsonify({"success": True, "id": ann["id"]})

@app.route("/api/admin/announcements/<aid>", methods=["DELETE"])
def api_admin_delete_announcement(aid):
    DB["announcements"] = [a for a in DB["announcements"] if a["id"] != aid]
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/admin/materials", methods=["GET"])
def api_admin_get_materials():
    return jsonify(DB.get("study_materials", []))

@app.route("/api/admin/materials", methods=["POST"])
def api_admin_add_material():
    b = {k: clean_input(v) for k, v in (request.get_json(force=True) or {}).items()}
    u = current_user()
    mat = {
        "id": str(uuid.uuid4())[:8], "title": b.get("title", ""),
        "subject": b.get("subject", ""), "type": b.get("type", "PDF"),
        "url": b.get("url", "#"), "description": b.get("description", ""),
        "uploaded_by": u["name"] if u else "Admin", "date": str(date.today()),
    }
    DB.setdefault("study_materials", []).insert(0, mat)
    save_db(DB)
    return jsonify({"success": True, "id": mat["id"]})

@app.route("/api/admin/materials/<mid>", methods=["DELETE"])
def api_admin_delete_material(mid):
    DB["study_materials"] = [m for m in DB.get("study_materials", []) if m["id"] != mid]
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/admin/timetable/<sid>")
def api_admin_timetable(sid):
    return jsonify(DB.get("timetable", {}).get(sid, {}))

@app.route("/api/admin/timetable/<sid>", methods=["PUT"])
def api_admin_update_timetable(sid):
    DB.setdefault("timetable", {})[sid] = request.get_json(force=True)
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/admin/predict", methods=["POST"])
def api_admin_predict():
    ip = get_client_ip()
    logger.info(f"Admin prediction request from IP {ip}")
    b          = request.get_json(force=True) or {}
    model_name = b.get("model", "random_forest")
    features   = b.get("features", {})
    try:
        row   = [float(features.get(f, 0)) for f in FEATURE_COLS]
        X     = np.array(row).reshape(1, -1)
        if model_name in ["logistic_regression", "svm"]:
            X = ARTIFACTS["scaler"].transform(X)
        model      = ARTIFACTS[MODEL_MAP.get(model_name, "rf_model")]
        le         = ARTIFACTS["label_encoder"]
        pred_idx   = model.predict(X)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        proba      = model.predict_proba(X)[0]
        risk, tips = assess_risk(features, pred_label)
        return jsonify({
            "prediction":    pred_label,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)},
            "risk_level":    risk,
            "tips":          tips,
        })
    except Exception as e:
        logger.error(f"Prediction error from IP {ip}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/model-metrics")
def api_admin_model_metrics():
    meta    = ARTIFACTS.get("metadata", {})
    results = meta.get("model_results", {})
    summary = []
    for name, data in results.items():
        report = data.get("report", {})
        summary.append({
            "model":     name,
            "accuracy":  round(data["accuracy"] * 100, 2),
            "precision": round(report.get("weighted avg", {}).get("precision", 0) * 100, 2),
            "recall":    round(report.get("weighted avg", {}).get("recall",    0) * 100, 2),
            "f1_score":  round(report.get("weighted avg", {}).get("f1-score",  0) * 100, 2),
        })
    return jsonify({"models": summary, "best_model": meta.get("best_model"),
                    "feature_importance": meta.get("feature_importance", {})})

@app.route("/api/admin/feedback")
def api_admin_feedback():
    return jsonify(DB.get("feedback", []))

@app.route("/api/admin/leaderboard")
def api_admin_leaderboard():
    board = []
    for sid, s in DB["students"].items():
        qa = DB.get("quiz_attempts", {}).get(sid, {})
        board.append({
            "sid": sid, "name": s["name"], "roll_no": s["roll_no"],
            "avatar": s.get("avatar", ""), "xp": qa.get("xp", 0),
            "level": qa.get("level", 1), "streak": qa.get("streak", 0),
            "total_answered": qa.get("total_answered", 0),
            "total_correct":  qa.get("total_correct",  0),
        })
    board.sort(key=lambda x: x["xp"], reverse=True)
    return jsonify(board)

# ── FACULTY API ───────────────────────────────────────────────────────────────
def get_fid():
    return session.get("user_id")

@app.route("/api/faculty/profile")
def api_faculty_profile():
    fid = get_fid()
    u   = DB["users"].get(fid, {})
    fac = DB.get("faculty", {}).get(fid, {})
    return jsonify({**u, "password": None, **fac})

@app.route("/api/faculty/stats")
def api_faculty_stats():
    fid      = get_fid()
    fac      = DB.get("faculty", {}).get(fid, {})
    my_subs  = fac.get("subjects", [])
    att_rates = []
    low_att   = 0
    for sid, subs in DB["attendance"].items():
        for sub in my_subs:
            if sub in subs and subs[sub]["total"] > 0:
                r = subs[sub]["present"] / subs[sub]["total"] * 100
                att_rates.append(r)
                if r < 75:
                    low_att += 1
    return jsonify({
        "total_students":       len(DB["students"]),
        "my_subjects":          len(my_subs),
        "avg_attendance":       round(sum(att_rates) / len(att_rates), 1) if att_rates else 0,
        "low_attendance_count": low_att,
    })

@app.route("/api/faculty/students")
def api_faculty_students():
    fid     = get_fid()
    fac     = DB.get("faculty", {}).get(fid, {})
    my_subs = fac.get("subjects", [])
    result  = []
    for sid, s in DB["students"].items():
        course_subs = SUBJECTS_MAP.get(s.get("course", "B.Tech CS"), [])
        rel_subs    = [sub for sub in my_subs if sub in course_subs]
        if rel_subs or not my_subs:
            att   = DB["attendance"].get(sid, {})
            rates = [att[sub]["present"] / att[sub]["total"] * 100
                     for sub in rel_subs if sub in att and att[sub]["total"] > 0]
            result.append({**s, "my_subjects": rel_subs,
                           "avg_attendance": round(sum(rates) / len(rates), 1) if rates else 0})
    return jsonify(result)

@app.route("/api/faculty/attendance", methods=["GET"])
def api_faculty_get_attendance():
    fid      = get_fid()
    fac      = DB.get("faculty", {}).get(fid, {})
    my_subs  = fac.get("subjects", [])
    result   = {}
    for sid, s in DB["students"].items():
        att          = DB["attendance"].get(sid, {})
        subject_data = {sub: att[sub] for sub in my_subs if sub in att}
        if subject_data:
            rates = {sub: round(d["present"] / d["total"] * 100, 1) if d["total"] > 0 else 0
                     for sub, d in subject_data.items()}
            result[sid] = {
                "name": s["name"], "roll_no": s["roll_no"], "avatar": s.get("avatar", ""),
                "subjects": subject_data, "rates": rates,
                "overall": round(sum(rates.values()) / len(rates), 1) if rates else 0,
            }
    return jsonify(result)

@app.route("/api/faculty/attendance/<sid>", methods=["PUT"])
def api_faculty_mark_attendance(sid):
    if sid not in DB["attendance"]:
        return jsonify({"error": "Not found"}), 404
    fid     = get_fid()
    fac     = DB.get("faculty", {}).get(fid, {})
    my_subs = fac.get("subjects", [])
    b        = request.get_json(force=True) or {}
    subject  = b.get("subject")
    action   = b.get("action")
    date_str = b.get("date", str(date.today()))
    if subject not in my_subs:
        return jsonify({"error": "Not authorized for this subject"}), 403
    if subject not in DB["attendance"][sid]:
        DB["attendance"][sid][subject] = {"total": 0, "present": 0, "records": []}
    att = DB["attendance"][sid][subject]
    att["total"] += 1
    if action == "present":
        att["present"] += 1
    att["records"].append({"date": date_str, "status": action, "marked_by": fid})
    rate = att["present"] / att["total"] * 100
    if rate < 75:
        DB["notifications"].setdefault(sid, []).append({
            "id": str(uuid.uuid4())[:8], "type": "warning",
            "msg": f"Attendance in {subject} is {rate:.1f}% — below 75%!",
            "date": date_str, "read": False,
        })
    save_db(DB)
    return jsonify({"success": True, "rate": round(rate, 1)})

@app.route("/api/faculty/attendance/bulk", methods=["POST"])
def api_faculty_bulk_attendance():
    fid      = get_fid()
    fac      = DB.get("faculty", {}).get(fid, {})
    my_subs  = fac.get("subjects", [])
    b        = request.get_json(force=True) or {}
    subject  = b.get("subject")
    date_str = b.get("date", str(date.today()))
    records  = b.get("records", {})
    if subject not in my_subs:
        return jsonify({"error": "Not authorized"}), 403
    updated = 0
    for sid, status in records.items():
        if sid not in DB["attendance"]:
            continue
        if subject not in DB["attendance"][sid]:
            DB["attendance"][sid][subject] = {"total": 0, "present": 0, "records": []}
        att = DB["attendance"][sid][subject]
        att["total"] += 1
        if status == "present":
            att["present"] += 1
        att["records"].append({"date": date_str, "status": status, "marked_by": fid})
        rate = att["present"] / att["total"] * 100
        if rate < 75:
            DB["notifications"].setdefault(sid, []).append({
                "id": str(uuid.uuid4())[:8], "type": "warning",
                "msg": f"Attendance in {subject} dropped to {rate:.1f}%",
                "date": date_str, "read": False,
            })
        updated += 1
    save_db(DB)
    return jsonify({"success": True, "updated": updated})

@app.route("/api/faculty/marks", methods=["GET"])
def api_faculty_get_marks():
    fid     = get_fid()
    fac     = DB.get("faculty", {}).get(fid, {})
    my_subs = fac.get("subjects", [])
    result  = {}
    for sid, s in DB["students"].items():
        marks        = DB["marks"].get(sid, {})
        subjects_out = {}
        for sub in my_subs:
            if sub in marks:
                m     = marks[sub]
                total = compute_total(m)
                pct   = round(total / compute_max() * 100, 1)
                subjects_out[sub] = {**m, "total": total, "max": compute_max(),
                                     "percentage": pct, "grade": get_grade(pct)}
        if subjects_out:
            result[sid] = {"name": s["name"], "roll_no": s["roll_no"],
                           "avatar": s.get("avatar", ""), "course": s.get("course", ""),
                           "subjects": subjects_out}
    return jsonify(result)

@app.route("/api/faculty/marks/<sid>", methods=["PUT"])
def api_faculty_update_marks(sid):
    if sid not in DB["marks"]:
        return jsonify({"error": "Not found"}), 404
    fid     = get_fid()
    fac     = DB.get("faculty", {}).get(fid, {})
    my_subs = fac.get("subjects", [])
    b       = request.get_json(force=True) or {}
    subject = b.get("subject")
    if subject not in my_subs:
        return jsonify({"error": "Not authorized for this subject"}), 403
    if subject not in DB["marks"][sid]:
        DB["marks"][sid][subject] = make_marks_entry()
    for field in ["insem1", "insem_lab", "insem2", "end_sem", "end_lab"]:
        if field in b:
            DB["marks"][sid][subject][field] = int(b[field])
    m     = DB["marks"][sid][subject]
    total = compute_total(m)
    pct   = round(total / compute_max() * 100, 1)
    DB["notifications"].setdefault(sid, []).append({
        "id": str(uuid.uuid4())[:8], "type": "info",
        "msg": f"Marks updated for {subject}: {total}/{compute_max()} ({get_grade(pct)})",
        "date": str(date.today()), "read": False,
    })
    save_db(DB)
    return jsonify({"success": True})

# ── STUDENT API ───────────────────────────────────────────────────────────────
def get_sid():
    return session.get("user_id")

@app.route("/api/student/profile", methods=["GET"])
def api_student_profile():
    sid = get_sid()
    if not sid or sid not in DB["students"]:
        return jsonify({"error": "Not found"}), 404
    return jsonify(DB["students"][sid])

@app.route("/api/student/profile", methods=["PUT"])
def api_student_update_profile():
    sid = get_sid()
    if not sid:
        return jsonify({"error": "Unauthorized"}), 401
    b = request.get_json(force=True) or {}
    for k in ["phone", "address", "guardian", "guardian_phone"]:
        if k in b:
            DB["students"][sid][k] = clean_input(b[k])
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/student/attendance")
def api_student_attendance():
    sid    = get_sid()
    att    = DB["attendance"].get(sid, {})
    result = {}
    for sub, d in att.items():
        rate = round(d["present"] / d["total"] * 100, 1) if d["total"] > 0 else 0
        result[sub] = {**d, "rate": rate, "status": "Safe" if rate >= 75 else "Low"}
    return jsonify(result)

@app.route("/api/student/marks")
def api_student_marks():
    sid    = get_sid()
    marks  = DB["marks"].get(sid, {})
    result = {}
    for sub, m in marks.items():
        if "mid1" in m and "insem1" not in m:
            m = {"insem1": m.get("mid1", 0), "insem_lab": 0,
                 "insem2": m.get("mid2", 0), "end_sem": m.get("final", 0),
                 "end_lab": m.get("assignment", 0)}
        total = compute_total(m)
        pct   = round(total / compute_max() * 100, 1)
        result[sub] = {**m, "total": total, "max": compute_max(), "percentage": pct, "grade": get_grade(pct)}
    return jsonify(result)

@app.route("/api/student/fees")
def api_student_fees():
    sid  = get_sid()
    fee  = DB["fees"].get(sid, {})
    due  = fee.get("total_due", 0)
    paid = fee.get("paid", 0)
    return jsonify({
        "total_due": due, "paid": paid, "balance": due - paid,
        "status": "Paid" if paid >= due else ("Partial" if paid > 0 else "Unpaid"),
        "transactions": fee.get("transactions", []),
    })

@app.route("/api/student/notifications")
def api_student_notifications():
    sid    = get_sid()
    notifs = DB["notifications"].get(sid, [])
    return jsonify(sorted(notifs, key=lambda x: x["date"], reverse=True))

@app.route("/api/student/notifications/<nid>/read", methods=["PUT"])
def api_student_mark_read(nid):
    sid = get_sid()
    for n in DB["notifications"].get(sid, []):
        if n["id"] == nid:
            n["read"] = True
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/student/announcements")
def api_student_announcements():
    return jsonify(DB["announcements"])

@app.route("/api/student/timetable")
def api_student_timetable():
    return jsonify(DB.get("timetable", {}).get(get_sid(), {}))

@app.route("/api/student/materials")
def api_student_materials():
    return jsonify(DB.get("study_materials", []))

@app.route("/api/student/feedback", methods=["POST"])
def api_student_feedback():
    sid = get_sid()
    b   = {k: clean_input(v) for k, v in (request.get_json(force=True) or {}).items()}
    st  = DB["students"].get(sid, {})
    fb  = {
        "id": str(uuid.uuid4())[:8], "sid": sid,
        "name": st.get("name", ""), "roll_no": st.get("roll_no", ""),
        "subject": b.get("subject", ""), "rating": int(b.get("rating", 5)),
        "comment": b.get("comment", ""), "date": str(date.today()),
    }
    DB.setdefault("feedback", []).insert(0, fb)
    save_db(DB)
    return jsonify({"success": True})

@app.route("/api/student/performance")
def api_student_performance():
    sid      = get_sid()
    att      = DB["attendance"].get(sid, {})
    marks    = DB["marks"].get(sid, {})
    rates    = [d["present"] / d["total"] * 100 for d in att.values() if d["total"] > 0]
    avg_att  = sum(rates) / len(rates) if rates else 0
    totals   = [compute_total(m) for m in marks.values()]
    avg_total = sum(totals) / len(totals) if totals else 0
    avg_pct  = round(avg_total / compute_max() * 100, 1)
    gpa      = round(avg_pct / 25, 2)
    features = {
        "study_hours_per_day": 3.5, "attendance_rate": avg_att,
        "previous_gpa": min(gpa, 4.0), "assignments_completed": 80,
        "sleep_hours": 7, "extracurricular_activities": 1,
        "parental_education": 2, "internet_access": 1,
        "tutoring_sessions": 2, "stress_level": 5,
        "motivation_score": 7, "part_time_job": 0,
    }
    try:
        row        = [float(features.get(f, 0)) for f in FEATURE_COLS]
        X          = np.array(row).reshape(1, -1)
        model      = ARTIFACTS["rf_model"]
        le         = ARTIFACTS["label_encoder"]
        pred_idx   = model.predict(X)[0]
        pred_label = le.inverse_transform([pred_idx])[0]
        proba      = model.predict_proba(X)[0]
        risk, tips = assess_risk(features, pred_label)
        return jsonify({
            "prediction": pred_label, "avg_attendance": round(avg_att, 1),
            "avg_percentage": avg_pct, "gpa": gpa,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)},
            "risk_level": risk, "tips": tips,
        })
    except Exception as e:
        logger.error(f"Student performance prediction error: {e}")
        return jsonify({"prediction": "N/A", "avg_attendance": round(avg_att, 1),
                        "avg_percentage": avg_pct, "gpa": gpa, "risk_level": "Medium", "tips": []})

# ── QUIZ API ──────────────────────────────────────────────────────────────────
@app.route("/api/student/quiz/question")
def api_quiz_question():
    sid = get_sid()
    if not sid:
        return jsonify({"error": "Unauthorized"}), 401
    s       = DB["students"].get(sid, {})
    course  = s.get("course", "B.Tech CS")
    subs    = SUBJECTS_MAP.get(course, ["General"])
    subject = request.args.get("subject", "")
    if subject and subject in QUIZ_BANK:
        pool = QUIZ_BANK[subject]
    else:
        pool = []
        for sub in subs:
            pool.extend(QUIZ_BANK.get(sub, QUIZ_BANK.get("General", [])))
        if not pool:
            pool = QUIZ_BANK["General"]
    q = random.choice(pool)
    return jsonify({"question": q["q"], "options": q["opts"],
                    "subject": subject or "Mixed", "total_options": len(q["opts"])})

@app.route("/api/student/quiz/answer", methods=["POST"])
def api_quiz_answer():
    sid = get_sid()
    if not sid:
        return jsonify({"error": "Unauthorized"}), 401
    b             = request.get_json(force=True) or {}
    question_text = b.get("question", "")
    answer_idx    = int(b.get("answer", 0))
    subject       = b.get("subject", "Mixed")
    correct_idx   = None
    explanation   = ""
    for sub_qs in QUIZ_BANK.values():
        for q in sub_qs:
            if q["q"] == question_text:
                correct_idx = q["ans"]
                explanation = q.get("exp", "")
                break
        if correct_idx is not None:
            break
    is_correct = (answer_idx == correct_idx)
    qa = DB.setdefault("quiz_attempts", {}).setdefault(sid, {
        "total_answered": 0, "total_correct": 0, "streak": 0,
        "xp": 0, "level": 1, "badges": [], "subject_stats": {},
    })
    qa["total_answered"] = qa.get("total_answered", 0) + 1
    xp_gain = 2
    if is_correct:
        qa["total_correct"] = qa.get("total_correct", 0) + 1
        qa["streak"]        = qa.get("streak", 0) + 1
        xp_gain = 10 + (10 if qa["streak"] >= 5 else 0) + (10 if qa["streak"] >= 10 else 0)
        qa["xp"] = qa.get("xp", 0) + xp_gain
    else:
        qa["streak"] = 0
        qa["xp"]     = qa.get("xp", 0) + xp_gain
    qa["level"] = 1 + qa.get("xp", 0) // 200
    sub_stats = qa.setdefault("subject_stats", {}).setdefault(subject, {"answered": 0, "correct": 0})
    sub_stats["answered"] += 1
    if is_correct:
        sub_stats["correct"] += 1
    badges_earned = []
    existing      = qa.get("badges", [])
    for threshold, bid, bname, bicon in [(10, "first_10", "Getting Started", "🎯"),
                                          (25, "silver_25", "Silver Scholar",  "🥈"),
                                          (50, "gold_50",   "Gold Master",     "🥇")]:
        if qa["total_correct"] == threshold and bid not in existing:
            qa["badges"] = existing + [bid]
            badges_earned.append({"id": bid, "name": bname, "icon": bicon})
    if qa.get("streak", 0) == 5 and "streak_5" not in existing:
        qa["badges"] = qa.get("badges", []) + ["streak_5"]
        badges_earned.append({"id": "streak_5", "name": "On Fire!", "icon": "🔥"})
    if qa.get("streak", 0) == 10 and "streak_10" not in existing:
        qa["badges"] = qa.get("badges", []) + ["streak_10"]
        badges_earned.append({"id": "streak_10", "name": "Unstoppable", "icon": "⚡"})
    DB.setdefault("leaderboard", {})[sid] = {
        "name":   DB["students"].get(sid, {}).get("name", ""),
        "avatar": DB["students"].get(sid, {}).get("avatar", ""),
        "xp":     qa.get("xp", 0),
        "level":  qa.get("level", 1),
    }
    save_db(DB)
    return jsonify({
        "correct":        is_correct,
        "correct_answer": correct_idx,
        "explanation":    explanation,
        "xp_gained":      xp_gain,
        "total_xp":       qa.get("xp", 0),
        "streak":         qa.get("streak", 0),
        "level":          qa.get("level", 1),
        "badges_earned":  badges_earned,
    })

@app.route("/api/student/quiz/stats")
def api_quiz_stats():
    sid = get_sid()
    if not sid:
        return jsonify({}), 401
    qa    = DB.get("quiz_attempts", {}).get(sid, {})
    s     = DB["students"].get(sid, {})
    board = sorted(DB.get("leaderboard", {}).values(), key=lambda x: x.get("xp", 0), reverse=True)
    rank  = next((i + 1 for i, b in enumerate(board) if b.get("name") == s.get("name", "")), "-")
    return jsonify({
        "total_answered":    qa.get("total_answered", 0),
        "total_correct":     qa.get("total_correct",  0),
        "streak":            qa.get("streak",  0),
        "xp":                qa.get("xp",      0),
        "level":             qa.get("level",   1),
        "badges":            qa.get("badges",  []),
        "subject_stats":     qa.get("subject_stats", {}),
        "leaderboard_rank":  rank,
        "available_subjects": list(QUIZ_BANK.keys()),
    })

@app.route("/api/student/leaderboard")
def api_student_leaderboard():
    board = []
    for sid, s in DB["students"].items():
        qa = DB.get("quiz_attempts", {}).get(sid, {})
        board.append({
            "name":           s["name"],
            "avatar":         s.get("avatar", ""),
            "xp":             qa.get("xp",             0),
            "level":          qa.get("level",          1),
            "streak":         qa.get("streak",         0),
            "total_answered": qa.get("total_answered", 0),
            "total_correct":  qa.get("total_correct",  0),
        })
    board.sort(key=lambda x: x["xp"], reverse=True)
    return jsonify(board[:20])

# ── Risk assessment ───────────────────────────────────────────────────────────
def assess_risk(features, prediction):
    risk = {"Excellent": "Low", "Good": "Low", "Average": "Medium", "Poor": "High"}.get(prediction, "Medium")
    tips = []
    if float(features.get("study_hours_per_day",   0)) < 3:  tips.append("Increase study hours to at least 3–4 hours per day.")
    if float(features.get("attendance_rate",        0)) < 75: tips.append("Improve attendance above 75% to stay on track.")
    if float(features.get("stress_level",           5)) > 7:  tips.append("High stress detected — consider mindfulness or counseling.")
    if float(features.get("sleep_hours",            7)) < 6:  tips.append("Sleep at least 7–8 hours for better cognitive performance.")
    if float(features.get("assignments_completed",  0)) < 70: tips.append("Complete more assignments to strengthen your grade base.")
    if float(features.get("motivation_score",       5)) < 5:  tips.append("Set short-term goals to boost motivation.")
    if not tips:
        tips.append("Great performance! Keep up the consistency.")
    return risk, tips

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("EduPredict Enhanced - Server fully started on port 5000")
    print("EduPredict Enhanced - Starting...")
    print("Admin:   admin@edupredict.edu / admin123")
    print("Faculty: anitha@edupredict.edu / faculty123")
    print("Student: aisha@student.edu / student123")
    print("URL:     http://localhost:5000")
    print("Logs:    backend/app.log")
    app.run(debug=True, host="0.0.0.0", port=5000)