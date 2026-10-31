\---

name: school-face-attendance

description: Automated student attendance system using webcam face recognition, 1-minute dynamic Excel syncing, and a live school web dashboard.

\---



\# School Attendance \& Truancy System — Coding Guidelines



\## 1. System Architecture

The project is divided into three decoupled modules running concurrently:

1\. \*\*Vision Engine (vision.py)\*\*: Captures webcam frames, detects/recognizes student faces, and logs timestamps.

2\. \*\*Data Sync Worker (data\_sync.py)\*\*: Polls the teacher's master Excel file (.xlsx) every 60 seconds to update the student roster and excused leaves.

3\. \*\*Dashboard Server (app.py)\*\*: Calculates attendance metrics and serves real-time Thai-language UI/API endpoints for the school website.



\## 2. Core Tech Stack

\- \*Language\*: Python 3.10+

\- \*Vision\*: opencv-python, face\_recognition, numpy

\- \*Data \& Scheduling\*: pandas, openpyxl, apscheduler

\- \*Web/API\*: fastapi, uvicorn, jinja2 (or streamlit)



\## 3. Data Flow \& Calculation Logic

Every 60 seconds (or upon a new face match), compute the following metrics:

\- \*\*total\_students (จำนวนนักเรียนทั้งหมด)\*\*: Total active rows in the teacher's Excel sheet.

\- \*\*present\_students (จำนวนนักเรียนที่มาโรงเรียนวันนี้)\*\*: Unique student\_ids verified by the webcam today.

\- \*\*missing\_or\_skipping (จำนวนนักเรียนที่ขาดหายหรือโดด)\*\*: total\_students minus present\_students (excluding students marked as "Excused/Sick Leave" in the teacher's Excel sheet).



\## 4. Coding Rules \& Best Practices

\### A. Non-Blocking Excel Reads (Crucial)

\- Teachers may have the Excel file open while editing. Always read the Excel file in \*read-only mode\* or copy it to a temporary file before calling pandas.read\_excel(..., engine="openpyxl") to prevent PermissionError crashes.

\- Run the 1-minute Excel sync inside a background thread (BackgroundScheduler) so the webcam FPS never drops.



\### B. Webcam Debouncing (Cooldown)

\- When a student stands in front of the camera, do not log them 30 times a second.

\- Implement a \*cooldown dictionary\* (last\_seen\[student\_id]). Once a student is marked present, ignore subsequent detections of the same student\_id for at least 5–10 minutes.



\### C. Frame Optimization

\- Downscale webcam frames to 1/4 size and convert BGR to RGB before running face recognition to keep real-time performance smooth on standard computers.



\### D. Privacy \& Security (PDPA Compliance)

\- Never expose raw student photos or full names on the public school website dashboard—only serve aggregated counts (จำนวนนักเรียน...) publicly, and keep the detailed name list behind a teacher login.

