import cv2
import yaml
import json
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# ----------------- Detection Modules -----------------
from .detection.face_detection import FaceDetector
from .detection.eye_tracking import EyeTracker
from .detection.mouth_detection import MouthMonitor
from .detection.object_detection import ObjectDetector
from .detection.multi_face import MultiFaceDetector

# ----------------- Audio -----------------
from .audio.speaker_consistency import SpeakerConsistencyAnalyzer

# ----------------- Scoring -----------------
from .analysis.scoring import (
    compute_video_score,
    compute_audio_score,
    compute_overall_score,
)

# ----------------- PERFORMANCE -----------------
FRAME_STRIDE = 5
RESIZE_WIDTH = 480
OBJ_DETECT_EVERY = 10
MF_DETECT_EVERY = 4


# ----------------- DATA MODELS -----------------
@dataclass
class AlertEvent:
    session_id: str
    timestamp: float
    frame_index: int
    type: str
    severity: str
    image_path: Optional[str]
    details: Dict[str, Any]


@dataclass
class SessionSummary:
    session_id: str
    duration_seconds: float
    num_frames: int
    fps: float
    alerts: List[AlertEvent]


# ----------------- CONFIG -----------------
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ----------------- ANALYZER -----------------
class OfflineExamAnalyzer:

    def __init__(self, config_path="config/config.yaml", logs_dir="logs/sessions"):
        self.cfg = load_config(config_path)

        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.evidence_root = self.logs_dir / "evidence"
        self.evidence_root.mkdir(exist_ok=True)

        det = self.cfg["detection"]

        self.face_detector = FaceDetector(self.cfg) if det["face"]["enabled"] else None
        self.eye_tracker = EyeTracker(self.cfg) if det["eyes"]["enabled"] else None
        self.mouth_monitor = MouthMonitor(self.cfg) if det["mouth"]["enabled"] else None
        self.multi_face_detector = MultiFaceDetector(self.cfg) if det["multi_face"]["enabled"] else None
        self.object_detector = ObjectDetector(self.cfg) if det["objects"]["enabled"] else None

        self.audio_enabled = self.cfg["audio_monitoring"]["enabled"]
        self.speaker_analyzer = None

        self._gaze_away_frames = 0
        self._mouth_moving_frames = 0
        self._multi_face_frames = 0

    # ----------------- MAIN -----------------
    def analyze_video(self, video_path, audio_path=None, session_id=None):

        session_id = session_id or self._generate_session_id(video_path)

        evidence_dir = self.evidence_root / session_id
        evidence_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        alerts: List[AlertEvent] = []

        frame_idx = 0
        processed_idx = 0
        last_faces = 1
        last_multi = False
        last_objects = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STRIDE != 0:
                frame_idx += 1
                continue

            t = frame_idx / fps

            if frame.shape[1] > RESIZE_WIDTH:
                scale = RESIZE_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * scale)))

            face_present = self.face_detector.detect_face(frame) if self.face_detector else True

            gaze_dir = "center"
            if self.eye_tracker:
                gaze_dir, _ = self.eye_tracker.track_eyes(frame)

            mouth_moving = self.mouth_monitor.monitor_mouth(frame) if self.mouth_monitor else False

            # Object detection
            last_objects = []
            if self.object_detector and processed_idx % OBJ_DETECT_EVERY == 0:
                res = self.object_detector.detect_objects(frame)
                last_objects = res if isinstance(res, list) else []

            # Multi-face
            if self.multi_face_detector and processed_idx % MF_DETECT_EVERY == 0:
                mf = self.multi_face_detector.detect_multiple_faces(frame)
                last_faces = mf if isinstance(mf, int) else 1
                last_multi = last_faces > 1

            alerts.extend(
                self._generate_alerts(
                    session_id,
                    frame_idx,
                    t,
                    frame,
                    evidence_dir,
                    face_present,
                    gaze_dir,
                    mouth_moving,
                    last_multi,
                    last_faces,
                    last_objects,
                )
            )

            processed_idx += 1
            frame_idx += 1

        cap.release()

        summary = SessionSummary(
            session_id=session_id,
            duration_seconds=frame_idx / fps,
            num_frames=frame_idx,
            fps=fps,
            alerts=alerts,
        )

        audio_summary = None
        if self.audio_enabled and audio_path:
            self.speaker_analyzer = SpeakerConsistencyAnalyzer(tmp_dir=self.logs_dir)
            audio_summary = self.speaker_analyzer.analyze(audio_path)

        self._save_session_log(summary, audio_summary)
        return summary

    # ----------------- ALERTS -----------------
    def _generate_alerts(
        self,
        session_id,
        frame_idx,
        t,
        frame,
        evidence_dir,
        face_present,
        gaze_direction,
        mouth_moving,
        multiple_faces,
        num_faces,
        object_list,
    ):

        alerts = []

        def save_evidence(tag):
            path = evidence_dir / f"{tag}_{frame_idx}.jpg"
            cv2.imwrite(str(path), frame)
            return str(path)

        if not face_present:
            alerts.append(AlertEvent(
                session_id, t, frame_idx, "FACE_MISSING", "medium",
                save_evidence("face_missing"), {}
            ))

        if gaze_direction != "center":
            self._gaze_away_frames += 1
        else:
            self._gaze_away_frames = 0

        if self._gaze_away_frames >= 3:
            alerts.append(AlertEvent(
                session_id, t, frame_idx, "GAZE_AWAY", "medium",
                save_evidence("gaze_away"), {}
            ))
            self._gaze_away_frames = 0

        if mouth_moving:
            self._mouth_moving_frames += 1
        else:
            self._mouth_moving_frames = 0

        if self._mouth_moving_frames >= 3:
            alerts.append(AlertEvent(
                session_id, t, frame_idx, "MOUTH_MOVEMENT", "medium",
                save_evidence("mouth"), {}
            ))
            self._mouth_moving_frames = 0

        if multiple_faces:
            self._multi_face_frames += 1
        else:
            self._multi_face_frames = 0

        if self._multi_face_frames >= 5:
            alerts.append(AlertEvent(
                session_id, t, frame_idx, "MULTI_FACE", "high",
                save_evidence("multi_face"), {"num_faces": num_faces}
            ))
            self._multi_face_frames = 0

        for obj in object_list:
            alerts.append(AlertEvent(
                session_id, t, frame_idx, "OBJECT_DETECTED", "high",
                save_evidence("object"), obj
            ))

        return alerts

    # ----------------- SAVE REPORT -----------------
    def _save_session_log(self, summary, audio_summary):

        out = self.logs_dir / f"{summary.session_id}.json"
        alerts_list = [asdict(a) for a in summary.alerts]
        counts = Counter(a["type"] for a in alerts_list)

        video_score = compute_video_score(
            {"total_alerts": len(alerts_list), "by_type": dict(counts)}, alerts_list
        )
        audio_score = compute_audio_score(audio_summary)
        overall_score = compute_overall_score(video_score, audio_score)

        verdict = "CLEAN"
        if counts.get("MULTI_FACE", 0) > 0 or counts.get("OBJECT_DETECTED", 0) > 0:
            verdict = "CHEATING"
        elif overall_score < 0.7 or len(alerts_list) > 5:
            verdict = "SUSPICIOUS"

        with open(out, "w") as f:
            json.dump({
                "session": asdict(summary),
                "verdict": verdict,
                "alerts": alerts_list,
                "scores": {
                    "video": video_score,
                    "audio": audio_score,
                    "overall": overall_score,
                },
                "audio": audio_summary,
                "generated_at": datetime.now().isoformat(),
            }, f, indent=2)

    @staticmethod
    def _generate_session_id(video_path):
        return f"{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ----------------- WRAPPER -----------------
def analyze_recording(
    video_path,
    audio_path=None,
    session_id=None,
    config_path="config/config.yaml",
    logs_dir="logs/sessions",
):
    analyzer = OfflineExamAnalyzer(config_path, logs_dir)
    return analyzer.analyze_video(video_path, audio_path, session_id)
