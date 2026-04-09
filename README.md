# EduPredict AI — Enhanced Student Management System

## 🚀 What's New in This Version

### 🔐 Three-Role Authentication
- **Admin** — Full site control, manages users, reports, fees
- **Faculty** — Mark attendance & enter marks for assigned subjects only
- **Student** — View academics, practice with Quiz Arena

---

### 👨‍🏫 Faculty Portal (NEW)
- Separate faculty dashboard with subject-level stats
- **Bulk attendance marking** — Mark entire class Present/Absent at once
- **Marks entry** — Enter InSem1, Lab, InSem2, EndSem, EndLab per subject
- Auto-notification to students when marks are updated
- Restricted to assigned subjects only

---

### ⚡ Quiz Arena (Gamified Learning)
- Questions for DS, Algo, DBMS, OS, CN, ML
- **XP System** — points for correct answers
- **Streaks** — bonus XP (5x = +10 XP, 10x = +20 XP)
- **Levels** — level up every 200 XP
- **Badges** — 🎯 🥈 🥇 🔥 ⚡
- **Leaderboard** — compete with classmates
- Per-subject accuracy tracking

---

### 🧠 Machine Learning Integration
- Models used:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - SVM
- Predicts student performance using:
  - Study hours
  - Attendance
  - GPA
  - Stress level
  - Motivation

---

### 🗂️ Hadoop Integration (NEW 🔥)
- Integrated **HDFS commands using Python subprocess**
- Example:
  ```python
  subprocess.run(["hdfs", "dfs", "-ls", path])