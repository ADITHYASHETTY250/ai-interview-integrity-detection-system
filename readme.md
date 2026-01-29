🎥 AI Interview & Exam Integrity Detection System
An AI-powered web application that analyzes recorded exam/interview videos to detect cheating, suspicious behavior, and integrity violations using Computer Vision and Audio Analysis.
The system generates:
📊 Integrity scores
🚨 Cheating alerts
🖼️ Visual proof (frames as evidence)
🎬 Video playback for verification

📁 Project Structure

ai-interview-integrity-detection-system/
│
├── config/
│   └── config.yaml               # Detection & performance configuration
│
├── src/
│   ├── dashboard/
│   │   ├── app.py                # Flask web application
│   │   ├── templates/
│   │   │   ├── dashboard.html
│   │   │   ├── upload.html
│   │   │   └── session_report.html
│   │   └── static/
│   │
│   ├── detection/
│   │   ├── face_detection.py
│   │   ├── eye_tracking.py
│   │   ├── mouth_detection.py
│   │   ├── multi_face.py
│   │   └── object_detection.py
│   │
│   ├── audio/
│   │   └── speaker_consistency.py
│   │
│   ├── analysis/
│   │   └── scoring.py
│   │
│   └── offline_processor.py      # Core AI processing engine
│
├── uploads/                      # Uploaded video & audio files
│
├── logs/
│   └── sessions/
│       ├── <session_id>.json     # Final report per session
│       └── evidence/
│           └── <session_id>/
│               └── frame_*.jpg   # Proof frames (cheating evidence)
│
├── requirements.txt
└── README.md

🚀 Features

👁️ Face presence detection
👀 Eye gaze tracking (looking away)
👄 Mouth movement (talking / whispering)
👥 Multiple face detection
📱 Object detection (phone, book, etc.)
🎤 Optional audio speaker consistency check
🖼️ Auto-saved cheating proof frames
⚖️ Final verdict: CLEAN / SUSPICIOUS / CHEATING
🧠 Technologies Used
Python 3.10
Flask (Web framework)
OpenCV (Computer Vision)
MediaPipe
NumPy
PyYAML
WavLM (Audio speaker analysis)
HTML / CSS

▶️ How to Run (VS Code)

1️⃣ Install Required Apps
App	Download Link
Python 3.10	https://www.python.org/downloads/
VS Code	https://code.visualstudio.com/
Git	https://git-scm.com/downloads
Miniconda	https://docs.conda.io/en/latest/miniconda.html

