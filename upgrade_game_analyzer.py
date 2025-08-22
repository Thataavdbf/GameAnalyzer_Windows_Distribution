import os
import subprocess
import json
import sys

# Resolve absolute paths relative to this script so we don't depend on CWD
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
settings_path = os.path.join(script_dir, "settings.json")

# Step 1: Install packages using the same Python interpreter
print("📦 Installing required packages (using this Python interpreter)...")
python_exe = sys.executable or "python"
packages = ["pyqt5", "moviepy", "ultralytics", "playsound"]
try:
    subprocess.check_call([python_exe, "-m", "pip", "install"] + packages)
except Exception as e:
    print(f"Warning: pip install had issues: {e}. Continuing without aborting.")

# Step 2: Rewrite object_detector.py
print("🧠 Replacing object_detector.py with YOLOv8 version...")
object_detector_code = '''
import cv2
from ultralytics import YOLO  # Make sure ultralytics is installed: pip install ultralytics

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        print(f"Loaded model: {model_path}")

    def detect_objects(self, frame):
        results = self.model.predict(source=frame, verbose=False)[0]
        detected_objects = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = self.model.names[class_id]
            detected_objects.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2 - x1, y2 - y1]
            })
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, detected_objects
'''
os.makedirs(src_dir, exist_ok=True)
object_detector_file = os.path.join(src_dir, "object_detector.py")
with open(object_detector_file, "w", encoding="utf-8") as f:
    f.write(object_detector_code.strip())

# Step 3: Rewrite main_gui.py
print("🖥️ Replacing main_gui.py with GUI + MoviePy version...")
main_gui_code = '''
import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QAction, QMenuBar, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from moviepy.editor import VideoFileClip
from object_detector import ObjectDetector

SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "sound_enabled": True,
    "default_model": "yolov8n.pt",
    "show_overlays": True
}

class GameAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Game Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.settings = self.load_settings()
        self.detector = ObjectDetector(self.settings["default_model"])
        self.label = QLabel("Load a video from File > Open Video", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)
        self.setup_menu()

    def setup_menu(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.open_video_dialog)
        file_menu.addAction(open_action)

        settings_menu = menu_bar.addMenu("Settings")
        self.sound_toggle = QAction("Enable Sounds", self, checkable=True)
        self.sound_toggle.setChecked(self.settings["sound_enabled"])
        self.sound_toggle.triggered.connect(self.toggle_sound)
        settings_menu.addAction(self.sound_toggle)

    def open_video_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.analyze_video(file_path)

    def analyze_video(self, video_path):
        clip = VideoFileClip(video_path)
        clip.preview(audio=True)

    def toggle_sound(self):
        self.settings["sound_enabled"] = self.sound_toggle.isChecked()
        self.save_settings()

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        return DEFAULT_SETTINGS.copy()

    def save_settings(self):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(self.settings, f, indent=2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
'''
main_gui_file = os.path.join(src_dir, "main_gui.py")
with open(main_gui_file, "w", encoding="utf-8") as f:
    f.write(main_gui_code.strip())

# Step 4: Create folders if missing
print("📁 Setting up folders...")
os.makedirs(os.path.join(script_dir, "sounds"), exist_ok=True)

# Step 5: Create settings.json if missing
DEFAULT_SETTINGS = {
    "sound_enabled": True,
    "default_model": "yolov8n.pt",
    "show_overlays": True,
}

if not os.path.exists(settings_path):
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SETTINGS, f, indent=2)

print("🚀 Upgrade complete! Launching Game Analyzer...")
main_gui_path = os.path.join(src_dir, "main_gui.py")
try:
    subprocess.call([python_exe, main_gui_path])
except Exception:
    # Fallback to shell call with explicit path
    subprocess.call(f'"{python_exe}" "{main_gui_path}"', shell=True)
