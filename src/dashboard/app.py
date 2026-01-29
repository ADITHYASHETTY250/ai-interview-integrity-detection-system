from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from uuid import uuid4
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.offline_processor import analyze_recording

app = Flask(__name__, template_folder="templates")
app.secret_key = "secret"

UPLOAD_DIR = PROJECT_ROOT / "uploads"
LOGS_DIR = PROJECT_ROOT / "logs" / "sessions"

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

@app.route("/")
def dashboard():
    sessions = []
    for p in LOGS_DIR.glob("*.json"):
        with open(p) as f:
            data = json.load(f)
        sessions.append(data["session"])
    return render_template("dashboard.html", sessions=sessions)


@app.route("/upload", methods=["GET", "POST"])
def upload_recording():
    if request.method == "POST":
        video = request.files["video"]
        sid = uuid4().hex
        path = UPLOAD_DIR / f"{sid}_video_{secure_filename(video.filename)}"
        video.save(path)

        analyze_recording(
            str(path),
            session_id=sid,
            config_path=str(PROJECT_ROOT / "config" / "config.yaml"),
            logs_dir=str(LOGS_DIR)
        )

        return redirect(url_for("session_report", session_id=sid))
    return render_template("upload.html")


@app.route("/session/<session_id>")
def session_report(session_id):
    data = json.load(open(LOGS_DIR / f"{session_id}.json"))

    video_file = next(f.name for f in UPLOAD_DIR.glob(f"{session_id}_video_*"))

    return render_template(
        "session_report.html",
        session=data["session"],
        alerts=data["alerts"],
        scores=data["scores"],
        verdict=data["verdict"],
        video_file=video_file
    )


@app.route("/video/<filename>")
def serve_video(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/evidence/<session_id>/<filename>")
def serve_evidence(session_id, filename):
    return send_from_directory(LOGS_DIR / "evidence" / session_id, filename)


if __name__ == "__main__":
    app.run(debug=True)
