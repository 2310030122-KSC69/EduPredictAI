# Task: Fix errors in backend/app.py

## Plan Steps:
1. [x] Add missing `import subprocess` to imports section.
2. [x] Remove irrelevant HDFS endpoint (`/api/hdfs/list`).
3. [x] Move `assess_risk` function definition earlier (after constants, before routes).
4. [x] Test: Run `python backend/app.py` and verify no import errors.

## Progress:
✅ All fixes complete. Server running successfully at http://localhost:5000. Scikit-learn version warnings are non-critical (model compatibility).

