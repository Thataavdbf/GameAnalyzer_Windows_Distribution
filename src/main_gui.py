import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QAction, QMenuBar, QLabel, QMessageBox, QPushButton, QSlider, QProgressBar, QHBoxLayout, QInputDialog, QWidget, QVBoxLayout, QFormLayout, QComboBox, QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDockWidget
# Try importing OpencvAnalysisWindow using importlib with package, relative, or file-based fallbacks
import importlib
import importlib.util
import os
import sys

OpencvAnalysisWindow = None

# 1) Try package import (installed layout)
try:
    mod_name = "src.opencv_analysis_view"
    spec = importlib.util.find_spec(mod_name)
    if spec is not None:
        mod = importlib.import_module(mod_name)
        OpencvAnalysisWindow = getattr(mod, "OpencvAnalysisWindow", None)
except Exception:
    OpencvAnalysisWindow = None

# 2) Try relative import (when running as a module inside the src package)
if OpencvAnalysisWindow is None:
    try:
        # use package-aware import if possible
        rel_name = ".opencv_analysis_view"
        pkg = __package__ or "src"
        spec = importlib.util.find_spec(rel_name, package=pkg)
        if spec is not None:
            mod = importlib.import_module(rel_name)
            OpencvAnalysisWindow = getattr(mod, "OpencvAnalysisWindow", None)
    except Exception:
        OpencvAnalysisWindow = None

# 3) Try loading by file path (when executing the script directly)
if OpencvAnalysisWindow is None:
    try:
        module_path = os.path.join(os.path.dirname(__file__), "opencv_analysis_view.py")
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location("opencv_analysis_view", module_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["opencv_analysis_view"] = mod
                spec.loader.exec_module(mod)
                OpencvAnalysisWindow = getattr(mod, "OpencvAnalysisWindow", None)
    except Exception:
        OpencvAnalysisWindow = None

# 4) Fallback lightweight stub so UI can still run even if the real widget is missing
if OpencvAnalysisWindow is None:
    class OpencvAnalysisWindow(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)


# small clickable label to detect user clicks on the preview
class ClickableLabel(QLabel):
    clicked = pyqtSignal(object)

    def mousePressEvent(self, event):
        try:
            self.clicked.emit(event)
        except Exception:
            pass
        super().mousePressEvent(event)

# Guard import of moviepy (use importlib to avoid static unresolved-import diagnostics)
try:
    import importlib
    spec = importlib.util.find_spec("moviepy.editor")
    if spec is not None:
        moviepy_editor = importlib.import_module("moviepy.editor")
        VideoFileClip = getattr(moviepy_editor, "VideoFileClip", None)
        MOVIEPY_AVAILABLE = VideoFileClip is not None
    else:
        VideoFileClip = None
        MOVIEPY_AVAILABLE = False
except Exception:
    VideoFileClip = None
    MOVIEPY_AVAILABLE = False

try:
    from object_detector import ObjectDetector
except Exception:
    # object_detector may not exist yet; provide a stub so UI still runs
    class ObjectDetector:
        def __init__(self, model_path="yolov8n.pt"):
            print(f"ObjectDetector stub initialized with {model_path}")

# Use settings.json at distribution root (one level above src)
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app_preview.log")
DEFAULT_SETTINGS = {
    "sound_enabled": True,
    "default_model": "yolov8n.pt",
    "show_overlays": True
}


class GameAnalyzerApp(QMainWindow):
    # Signals for thread-safe UI updates
    progress_update = pyqtSignal(int, str)
    info_signal = pyqtSignal(str, str)
    preview_signal = pyqtSignal(str)
    preview_pixmap = pyqtSignal(object)
    def __init__(self):
        # initialize UI and state
        try:
            super().__init__()
            self.log("GUI init: start")
            self.setWindowTitle("Game Analyzer")
            try:
                # use primary screen available geometry to choose a safe default window size
                from PyQt5.QtWidgets import QDesktopWidget
                screen = QApplication.primaryScreen()
                if screen:
                    avail = screen.availableGeometry()
                    w = max(800, int(avail.width() * 0.8))
                    h = max(600, int(avail.height() * 0.8))
                    x = avail.x() + (avail.width() - w) // 2
                    y = avail.y() + (avail.height() - h) // 2
                    self.setGeometry(x, y, w, h)
                else:
                    self.setGeometry(100, 100, 1000, 700)
            except Exception:
                try:
                    self.setGeometry(100, 100, 1000, 700)
                except Exception:
                    pass
            self.settings = self.load_settings()
            self.detector = ObjectDetector(self.settings.get("default_model", "yolov8n.pt"))

            container = QWidget(self)
            layout = QVBoxLayout(container)
            self.log("GUI init: container and layout created")

            # Analysis options group
            options_group = QGroupBox("Analysis Options", self)
            options_layout = QHBoxLayout(options_group)
            self.game_type_combo = QComboBox(self)
            self.game_type_combo.addItems(["Helldivers 2", "Undisputed Boxing"])
            options_layout.addWidget(QLabel("Game Type:", self))
            options_layout.addWidget(self.game_type_combo)
            self.helldivers_checkbox = QCheckBox("Helldivers Analyzer", self)
            self.helldivers_checkbox.setChecked(True)
            self.boxing_checkbox = QCheckBox("Boxing Analyzer", self)
            self.boxing_checkbox.setChecked(True)
            self.stats_checkbox = QCheckBox("Statistics Engine", self)
            self.stats_checkbox.setChecked(True)
            options_layout.addWidget(self.helldivers_checkbox)
            options_layout.addWidget(self.boxing_checkbox)
            options_layout.addWidget(self.stats_checkbox)
            layout.addWidget(options_group, 0)

            self.video_label = ClickableLabel("Load a video from File > Open Video", self)
            self.video_label.setAlignment(Qt.AlignCenter)
            # Prevent label from forcing window resize when pixmap is set
            from PyQt5.QtWidgets import QSizePolicy
            # Make label ignore the pixmap's native size so it won't force layout resizing
            self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.video_label.setMinimumSize(320, 240)
            # cap maximum size to primary screen available geometry to prevent runaway growth
            try:
                screen = QApplication.primaryScreen()
                if screen:
                    avail = screen.availableGeometry()
                    max_w = max(800, int(avail.width() * 0.95))
                    max_h = max(600, int(avail.height() * 0.95))
                    self.video_label.setMaximumSize(max_w, max_h)
            except Exception:
                pass
            # Allow pixmap to be scaled to the label rather than the label resizing to pixmap
            try:
                self.video_label.setScaledContents(True)
            except Exception:
                pass
            layout.addWidget(self.video_label, 1)
            # connect click handler for labeling
            try:
                self.video_label.clicked.connect(self.on_video_clicked)
            except Exception:
                pass

            self.preview_status_label = QLabel("Preview: idle", self)
            self.preview_status_label.setAlignment(Qt.AlignCenter)
            self.preview_status_label.setStyleSheet("color: gray; font-size: 11px;")
            layout.addWidget(self.preview_status_label, 0)

            self.progress_bar = QProgressBar(self)
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(100)

            # Playback controls
            controls_layout = QHBoxLayout()
            self.play_button = QPushButton("Play", self)
            self.pause_button = QPushButton("Pause", self)
            self.ff_button = QPushButton("Fast Forward", self)
            self.rr_button = QPushButton("Rewind", self)
            self.skip_forward_button = QPushButton("+10s", self)
            self.skip_backward_button = QPushButton("-10s", self)
            controls_layout.addWidget(self.play_button)
            controls_layout.addWidget(self.pause_button)
            controls_layout.addWidget(self.rr_button)
            controls_layout.addWidget(self.ff_button)
            controls_layout.addWidget(self.skip_backward_button)
            controls_layout.addWidget(self.skip_forward_button)
            layout.addLayout(controls_layout)

            # Volume slider
            volume_layout = QHBoxLayout()
            volume_label = QLabel("Volume:", self)
            self.volume_slider = QSlider(Qt.Horizontal, self)
            self.volume_slider.setMinimum(0)
            self.volume_slider.setMaximum(100)
            self.volume_slider.setValue(100 if self.settings.get("sound_enabled", True) else 0)
            volume_layout.addWidget(volume_label)
            volume_layout.addWidget(self.volume_slider)
            layout.addLayout(volume_layout)

            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            self.progress_bar.setStyleSheet("QProgressBar { height: 30px; font-size: 16px; margin: 10px; border: 2px solid #555; border-radius: 8px; background: #eee; } QProgressBar::chunk { background-color: #0078d7; width: 20px; }")
            layout.addWidget(self.progress_bar, 0)

            # Visual debug hints
            try:
                container.setStyleSheet("background-color: #e9eef6;")
                self.video_label.setStyleSheet("border: 2px dashed #888; background: #fff;")
            except Exception:
                pass

            self.setCentralWidget(container)
            container.setVisible(True)
            container.update()
            self.log("GUI: central widget set and shown")
            self.setup_menu()
            self.statusBar().showMessage("")
            self.log("GUI: menu and status bar set")

            # connect signals to main-thread handlers
            try:
                self.progress_update.connect(self._on_progress)
                self.info_signal.connect(self._on_info)
                self.preview_signal.connect(self.set_preview_status)
                self.preview_pixmap.connect(self._on_preview_pixmap)
            except Exception:
                pass

            # Connect playback controls
            self.play_button.clicked.connect(self.play_video)
            self.pause_button.clicked.connect(self.pause_video)
            self.ff_button.clicked.connect(self.fast_forward)
            self.rr_button.clicked.connect(self.rewind)
            self.skip_forward_button.clicked.connect(self.skip_forward)
            self.skip_backward_button.clicked.connect(self.skip_backward)
            self.volume_slider.valueChanged.connect(self.set_volume)

            # Playback state
            self.current_video_path = None
            self.current_clip = None
            # OpenCV preview capture (used when moviepy is not available)
            self.preview_cap = None
            self.current_time = 0.0
            self.is_playing = False
            self.playback_timer = None
            self.playback_fps = 24
            self.playback_duration = 0.0
            self.analysis_thread = None
            self.log("GUI: initialization complete")
        except Exception as e:
            import traceback
            error_msg = f"Error in GameAnalyzerApp.__init__: {e}\n{traceback.format_exc()}"
            print(error_msg)
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(error_msg + "\n")
            except Exception:
                pass
            QMessageBox.critical(self, "Startup Error", error_msg)

    # signal handlers (run in main thread)
    def _on_progress(self, percent: int, msg: str):
        try:
            self.progress_bar.setValue(percent)
            self.statusBar().showMessage(msg)
        except Exception:
            pass

    def _on_info(self, title: str, text: str):
        try:
            QMessageBox.information(self, title, text)
        except Exception:
            pass

    def _on_preview_pixmap(self, qimg):
        try:
            # qimg is a QImage emitted from background thread; create QPixmap in main thread
            pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio)
            self.video_label.setPixmap(pix)
            self.video_label.update()
        except Exception:
            pass

    # Playback control methods
    def play_video(self):
        self.log("play_video called")
        if not self.current_clip and getattr(self, "preview_cap", None) is None:
            self.log("play_video: no preview available")
            return
        self.is_playing = True
        if not self.playback_timer:
            from PyQt5.QtCore import QTimer
            self.playback_timer = QTimer(self)
            self.playback_timer.timeout.connect(self.update_frame)
            self.playback_timer.start(int(1000 / self.playback_fps))
        else:
            self.playback_timer.start(int(1000 / self.playback_fps))

    def pause_video(self):
        self.log("pause_video called")
        self.is_playing = False
        if self.playback_timer:
            self.playback_timer.stop()

    def fast_forward(self):
        self.log("fast_forward called")
        self.current_time = min(self.current_time + 5.0, self.playback_duration)
        self.update_frame(force=True)

    def rewind(self):
        self.log("rewind called")
        self.current_time = max(self.current_time - 5.0, 0.0)
        self.update_frame(force=True)

    def skip_forward(self):
        self.log("skip_forward called")
        self.current_time = min(self.current_time + 10.0, self.playback_duration)
        self.update_frame(force=True)

    def skip_backward(self):
        self.log("skip_backward called")
        self.current_time = max(self.current_time - 10.0, 0.0)
        self.update_frame(force=True)

    def set_volume(self, value):
        self.settings["sound_enabled"] = value > 0
        self.save_settings()

    def update_frame(self, force=False):
        if not self.is_playing and not force:
            return
        # If a moviepy clip is available, use it. Otherwise try OpenCV preview_cap.
        if self.current_clip:
            try:
                frame = self.current_clip.get_frame(self.current_time)
                import numpy as np
                frame = np.array(frame)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # emit QImage to main thread for scaling/create QPixmap there (single path)
                try:
                    self.preview_pixmap.emit(qimg)
                except Exception:
                    pass
                self.current_time += 1.0 / self.playback_fps
                if self.current_time >= self.playback_duration:
                    self.pause_video()
            except Exception as e:
                self.log(f"update_frame error: {e}")
                try:
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(f"update_frame error: {e}\n")
                except Exception:
                    pass
            return

        # OpenCV preview path
        try:
            import cv2
            cap = getattr(self, "preview_cap", None)
            if cap is None:
                return
            ret, frame = cap.read()
            if not ret:
                # reached end -> stop or loop
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception:
                    pass
                self.pause_video()
                return
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            try:
                self.preview_pixmap.emit(qimg)
            except Exception:
                self.log("preview_pixmap emit failed in update_frame (opencv preview)")
        except Exception as e:
            self.log(f"update_frame opencv preview error: {e}")
    def setup_menu(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = menu_bar.addMenu("File")
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.open_video_dialog)
        file_menu.addAction(open_action)

        settings_menu = menu_bar.addMenu("Settings")
        self.sound_toggle = QAction("Enable Sounds", self, checkable=True)
        self.sound_toggle.setChecked(self.settings.get("sound_enabled", True))
        self.sound_toggle.triggered.connect(self.toggle_sound)
        settings_menu.addAction(self.sound_toggle)

    def open_video_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)", options=options
        )
        if file_path:
            self.log(f"Selected video: {file_path}")
            # persist last opened video
            try:
                self.settings["last_opened_video"] = file_path
                self.save_settings()
            except Exception:
                pass
            self.current_video_path = file_path
            self.load_video(file_path)
            self.analyze_video_background(file_path)

    def analyze_video(self, video_path):
        # Top-level try to catch unexpected errors in background thread
        try:
            # Import analyzers and stats engine
            from helldivers_analyzer import HellDiversAnalyzer
            from boxing_analyzer import BoxingAnalyzer
            from statistics_engine import StatisticsEngine
            from action_recognition_analyzer import ActionRecognitionAnalyzer
            from scene_classification_analyzer import SceneClassificationAnalyzer
            from game_event_analyzer import GameEventAnalyzer
            from player_tracking_analyzer import PlayerTrackingAnalyzer
            from audio_analysis_analyzer import AudioAnalysisAnalyzer

            selected_game = self.game_type_combo.currentText()
            use_helldivers = self.helldivers_checkbox.isChecked()
            use_boxing = self.boxing_checkbox.isChecked()
            use_stats = self.stats_checkbox.isChecked()

            # Prepare analyzers
            helldivers = HellDiversAnalyzer() if use_helldivers and selected_game == "Helldivers 2" else None
            boxing = BoxingAnalyzer() if use_boxing and selected_game == "Undisputed Boxing" else None
            stats = StatisticsEngine() if use_stats else None
            action_recog = ActionRecognitionAnalyzer()
            scene_class = SceneClassificationAnalyzer()
            game_event = GameEventAnalyzer()
            player_track = PlayerTrackingAnalyzer()
            audio_analyze = AudioAnalysisAnalyzer()

            # notify main thread
            try:
                self.preview_signal.emit("starting")
            except Exception:
                pass

            # Try MoviePy preview when available; otherwise use OpenCV fallback
            if MOVIEPY_AVAILABLE and VideoFileClip is not None:
                try:
                    clip = VideoFileClip(video_path)
                    duration = clip.duration
                    results = []

                    def progress_callback(current_time):
                        percent = int((current_time / duration) * 100)
                        try:
                            self.progress_update.emit(percent, f"Processing: {percent}% ({current_time:.1f}s/{duration:.1f}s)")
                        except Exception:
                            pass

                    t = 0.0
                    while t < duration:
                        frame = clip.get_frame(t)
                        # Run all analyzers
                        if helldivers:
                            events = helldivers.analyze_frame(frame, t)
                            results.extend(events)
                        if boxing:
                            events = boxing.analyze_frame(frame, t)
                            results.extend(events)
                        results.extend(action_recog.analyze_frame(frame, t))
                        results.extend(scene_class.analyze_frame(frame, t))
                        results.extend(game_event.analyze_frame(frame, t))
                        results.extend(player_track.analyze_frame(frame, t))
                        t += 1.0
                        progress_callback(t)

                    # Optional: run audio analysis on the clip's audio track using larger, overlapping windows
                    try:
                        if hasattr(clip, "audio") and clip.audio is not None:
                            try:
                                sr = 16000
                                audio_arr = clip.audio.to_soundarray(fps=sr)
                                import numpy as _np
                                if audio_arr.ndim == 2:
                                    audio_mono = _np.mean(audio_arr, axis=1)
                                else:
                                    audio_mono = _np.array(audio_arr)

                                # window and stride configurable on the app instance
                                window_s = getattr(self, 'audio_window_s', 2.0)
                                stride_s = getattr(self, 'audio_stride_s', 1.0)
                                # sanity checks
                                if window_s <= 0:
                                    window_s = 2.0
                                if stride_s <= 0:
                                    stride_s = 1.0

                                win = int(sr * float(window_s))
                                hop = int(sr * float(stride_s))
                                if win <= 0:
                                    win = int(sr * 2.0)
                                if hop <= 0:
                                    hop = int(sr * 1.0)

                                length = len(audio_mono)
                                # compute number of windows covering the audio with overlap
                                n_windows = max(1, int(np.floor((length - win) / hop)) + 1)
                                for i in range(n_windows):
                                    start_idx = i * hop
                                    end_idx = start_idx + win
                                    if end_idx > length:
                                        end_idx = length
                                    seg = audio_mono[start_idx:end_idx]
                                    ts = float(start_idx) / float(sr)
                                    try:
                                        evs = audio_analyze.analyze_audio((seg, sr), ts)
                                        if evs:
                                            results.extend(evs)
                                    except Exception:
                                        pass
                            except Exception as e:
                                self.log(f"audio analysis (moviepy) failed: {e}")
                    except Exception:
                        pass

                    # Aggregate stats and save
                    if stats:
                        stats.load_events(results, selected_game.lower().replace(" ", "_"))
                        summary = stats.generate_statistical_overlays()
                        results.append({"summary": summary})
                    out_file = f"analysis_results_{selected_game.lower().replace(' ', '_')}.json"
                    import json
                    with open(out_file, "w") as f:
                        json.dump(results, f, indent=2)
                    try:
                        self.preview_signal.emit("finished (analyzed)")
                        self.progress_update.emit(100, "Processing: 100% (done)")
                        self.info_signal.emit("Processing Complete", f"Video analysis is complete! Results saved to {out_file}")
                    except Exception:
                        pass
                    return
                except Exception as e:
                    self.log(f"moviepy could not open file: {e}")

            # OpenCV fallback
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("OpenCV failed to open the video file")
            try:
                self.preview_signal.emit("opencv")
            except Exception:
                pass
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    # Emit QImage to the main thread for conversion to QPixmap there.
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame.shape
                    bytes_per_line = ch * w
                    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    try:
                        self.preview_pixmap.emit(qimg)
                    except Exception:
                        self.log("preview_pixmap emit failed in worker thread (early opencv branch)")
                frame_count += 1
                percent = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
                try:
                    self.progress_update.emit(percent, f"Processing: {percent}% ({frame_count}/{total_frames} frames)")
                except Exception:
                    pass
                QApplication.processEvents()
            cap.release()
            try:
                self.preview_signal.emit("finished (opencv)")
                self.progress_update.emit(100, "Processing: 100% (done)")
                self.info_signal.emit("Processing Complete", "Video processing is complete!")
            except Exception:
                pass
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            err = f"analyze_video top-level exception: {e}\n{tb}"
            print(err)
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(err + "\n")
            except Exception:
                pass
            try:
                self.info_signal.emit("Analysis Error", f"An error occurred during analysis; see log for details")
            except Exception:
                pass
            return

        # OpenCV fallback
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("OpenCV failed to open the video file")
            try:
                self.preview_signal.emit("opencv")
            except Exception:
                pass
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # run a quick UI update for preview via main thread
                try:
                    # emit QImage to main thread; conversion to QPixmap happens there
                    try:
                        self.preview_pixmap.emit(qimg)
                    except Exception:
                        # if signals fail, skip preview update (no QPixmap creation in worker)
                        self.log("preview_pixmap emit failed in worker thread")
                except Exception:
                    pass
                frame_count += 1
                percent = int((frame_count / total_frames) * 100)
                try:
                    self.progress_update.emit(percent, f"Processing: {percent}% ({frame_count}/{total_frames} frames)")
                except Exception:
                    pass
                QApplication.processEvents()
            cap.release()
            try:
                self.preview_signal.emit("finished (opencv)")
                self.progress_update.emit(100, "Processing: 100% (done)")
                self.info_signal.emit("Processing Complete", "Video processing is complete!")
            except Exception:
                pass
        except Exception as e:
            self.set_preview_status("failed")
            self.log(f"All preview methods failed: {e}")
            QMessageBox.critical(self, "Preview error", f"Failed to preview video:\n{e}")

    def toggle_sound(self):
        if hasattr(self, "sound_toggle"):
            self.settings["sound_enabled"] = self.sound_toggle.isChecked()
            self.save_settings()

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_SETTINGS.copy()

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass

    def log(self, msg: str):
        try:
            print(msg)
        except Exception:
            pass
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def set_preview_status(self, text: str):
        try:
            self.preview_status_label.setText(f"Preview: {text}")
            self.statusBar().showMessage(f"Preview: {text}")
        except Exception:
            pass

    def load_video(self, file_path: str):
        """Load a video for preview/playback. Uses moviepy if available, otherwise leaves clip None."""
        try:
            self.current_video_path = file_path
            if MOVIEPY_AVAILABLE and VideoFileClip is not None:
                try:
                    clip = VideoFileClip(file_path)
                    self.current_clip = clip
                    self.playback_duration = clip.duration
                    self.playback_fps = int(clip.fps) if getattr(clip, "fps", None) else 24
                    self.current_time = 0.0
                    self.set_preview_status("loaded (moviepy)")
                    self.update_frame(force=True)
                    return
                except Exception as e:
                    self.log(f"moviepy failed to load clip: {e}")
            # fallback: try OpenCV VideoCapture for preview
            try:
                import cv2
                # release previous preview cap if any
                try:
                    if getattr(self, "preview_cap", None) is not None:
                        try:
                            self.preview_cap.release()
                        except Exception:
                            pass
                except Exception:
                    pass
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    self.preview_cap = cap
                    fps = cap.get(cv2.CAP_PROP_FPS) or 24
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    self.playback_fps = int(fps) if fps and fps > 0 else 24
                    self.playback_duration = (total_frames / self.playback_fps) if self.playback_fps > 0 and total_frames > 0 else 0.0
                    self.current_time = 0.0
                    self.set_preview_status("loaded (opencv preview)")
                    self.update_frame(force=True)
                    return
            except Exception as e:
                self.log(f"opencv preview failed to open clip: {e}")
            # fallback: no clip available yet
            self.current_clip = None
            self.playback_duration = 0.0
            self.set_preview_status("loaded (no preview)")
        except Exception as e:
            self.log(f"load_video error: {e}")

    def on_video_clicked(self, event):
        """Prompt user for a label at the clicked time and save a short audio snippet for training."""
        try:
            # Use current_time as the timestamp for labeling
            ts = float(getattr(self, 'current_time', 0.0))
            label, ok = QInputDialog.getText(self, 'Label segment', f'Enter label for time {ts:.2f}s:')
            if not ok or not label:
                return
            self.save_label_snippet(label.strip(), ts)
            QMessageBox.information(self, 'Saved', f'Saved snippet for label "{label}" at {ts:.2f}s')
        except Exception as e:
            self.log(f'on_video_clicked error: {e}')

    def save_label_snippet(self, label: str, timestamp_s: float, window_s: float = 2.0):
        """Extract and save an audio snippet centered at timestamp_s (window_s length) into training dataset.

        Writes WAV to training_dataset/<label>/segment_<start_ms>.wav and appends a CSV line.
        """
        try:
            out_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_dataset')
            label_dir = os.path.join(out_base, label)
            os.makedirs(label_dir, exist_ok=True)
            start = max(0.0, timestamp_s - (window_s / 2.0))
            end = start + window_s
            # Prefer MoviePy audio extraction for accuracy
            if MOVIEPY_AVAILABLE and self.current_video_path:
                try:
                    clip = VideoFileClip(self.current_video_path)
                    sr = 16000
                    audio_arr = clip.audio.to_soundarray(fps=sr)
                    import numpy as _np
                    if audio_arr.ndim == 2:
                        audio_mono = _np.mean(audio_arr, axis=1)
                    else:
                        audio_mono = _np.array(audio_arr)
                    start_idx = int(start * sr)
                    end_idx = int(end * sr)
                    seg = audio_mono[start_idx:end_idx]
                    fname = f'segment_{int(start*1000):08d}.wav'
                    out_path = os.path.join(label_dir, fname)
                    # write wav (int16)
                    from scipy.io import wavfile
                    clipped = _np.clip(seg, -1.0, 1.0)
                    wavfile.write(out_path, sr, (clipped * 32767).astype('int16'))
                except Exception as e:
                    self.log(f'save_label_snippet (moviepy) failed: {e}')
                    QMessageBox.warning(self, 'Save failed', f'Could not extract audio: {e}')
                    return
            else:
                QMessageBox.warning(self, 'Not available', 'MoviePy audio not available or no video loaded; cannot extract audio snippet')
                return

            # Append CSV entry
            csv_path = os.path.join(out_base, 'labels.csv')
            try:
                line = f"{os.path.relpath(out_path, out_base).replace('\\', '/')},{label},{start:.3f}\n"
                with open(csv_path, 'a', encoding='utf-8') as f:
                    f.write(line)
            except Exception:
                pass
        except Exception as e:
            self.log(f'save_label_snippet error: {e}')

    def analyze_video_background(self, file_path: str):
        """Run analyze_video in a background thread to avoid blocking the GUI."""
        try:
            import threading
            if self.analysis_thread and self.analysis_thread.is_alive():
                self.log("Analysis already running")
                return
            self.analysis_thread = threading.Thread(target=self.analyze_video, args=(file_path,), daemon=True)
            self.analysis_thread.start()
        except Exception as e:
            self.log(f"analyze_video_background error: {e}")

    def apply_settings(self):
        # read settings.json and apply to app state
        try:
            s = self.load_settings()
            # playback
            pb = s.get("playback", {})
            self.auto_play_on_load = pb.get("auto_play_on_load", False)
            self.start_muted = pb.get("start_muted", False)
            self.playback_framerate_cap = pb.get("playback_framerate_cap", 0)

            # audio
            ad = s.get("audio", {})
            self.audio_enabled = ad.get("enable_audio_analysis", True)
            self.audio_volume_default = ad.get("volume", 1.0)
            self.audio_window_s = ad.get("audio_window_s", 2.0)
            self.audio_stride_s = ad.get("audio_stride_s", 1.0)
            audio_model = ad.get("audio_model_path", "") or None
            # if your audio analyzer instance exists, update it
            try:
                if hasattr(self, "audio_analyzer") and audio_model:
                    # if analyzer supports model_path reload, try to reload
                    try:
                        self.audio_analyzer = self.audio_analyzer.__class__(model_path=audio_model)
                        self.log(f"Loaded audio model: {audio_model}")
                    except Exception:
                        self.log("Warning: failed to load audio model, keeping heuristic analyzer.")
                elif hasattr(self, "audio_analyzer") and not audio_model:
                    # ensure analyzer knows heuristics mode (no model)
                    try:
                        self.audio_analyzer = self.audio_analyzer.__class__(model_path=None)
                    except Exception:
                        pass
            except Exception:
                pass

            # analyzers toggles
            an = s.get("analyzers", {})
            self.analyzers_enabled = an

            # models folder
            md = s.get("models", {})
            self.model_folder = md.get("default_model_folder", "models")

            # Attempt to load per-analyzer models (if paths provided)
            try:
                # helper to safely reload or set model path attribute
                def _reload_analyzer(name, attr_name, model_path):
                    if not model_path:
                        return
                    try:
                        # prefer explicit analyzer instance attributes (e.g., self.helldivers_analyzer)
                        inst_attr = f"{name}_analyzer"
                        if hasattr(self, inst_attr):
                            inst = getattr(self, inst_attr)
                            try:
                                # try re-constructing with model_path if supported
                                new_inst = inst.__class__(model_path=model_path)
                                setattr(self, inst_attr, new_inst)
                                self.log(f"Reloaded {name} analyzer with model: {model_path}")
                                return
                            except Exception:
                                self.log(f"Warning: {name} analyzer did not accept model_path on reload.")
                        # otherwise, store model path for later instantiation
                        setattr(self, f"{name}_model_path", model_path)
                        self.log(f"Set {name}_model_path = {model_path}")
                    except Exception as ee:
                        self.log(f"Error loading model for {name}: {ee}")

                _reload_analyzer("helldivers", "helldivers_model", md.get("helldivers_model", "") if md else "")
                _reload_analyzer("boxing", "boxing_model", md.get("boxing_model", "") if md else "")
                _reload_analyzer("action", "action_model", md.get("action_model", "") if md else "")
                _reload_analyzer("scene", "scene_model", md.get("scene_model", "") if md else "")
                _reload_analyzer("tracking", "tracking_model", md.get("tracking_model", "") if md else "")
                # audio model is handled above via ad.get("audio_model_path")
            except Exception:
                pass

            # ui settings
            ui = s.get("ui", {})
            self.ui_theme = ui.get("theme", "dark")
            self.compact_mode = ui.get("compact_mode", False)

            # apply audio window settings to existing attributes used in processing
            try:
                self.audio_window_s = float(self.audio_window_s)
                self.audio_stride_s = float(self.audio_stride_s)
            except Exception:
                pass

            # persist to app log
            try:
                self.log("Settings applied.")
            except Exception:
                pass
        except Exception as e:
            try:
                self.log(f"Failed to apply settings: {e}")
            except Exception:
                pass
        super().apply_settings()
        # ensure playback settings take effect
        self.playback_fps = max(1, int(self.playback_framerate_cap)) if self.playback_framerate_cap > 0 else 24
        self.log(f"Playback framerate cap set to: {self.playback_fps}")

        # update volume slider range and initial value
        try:
            if hasattr(self, "volume_slider"):
                self.volume_slider.setMinimum(0)
                self.volume_slider.setMaximum(100)
                self.volume_slider.setValue(100 if self.settings.get("sound_enabled", True) else 0)
        except Exception:
            pass

        # update UI theme
        try:
            if hasattr(self, "ui_theme"):
                # example: apply dark or light theme stylesheets
                if self.ui_theme == "dark":
                    self.setStyleSheet("background-color: #121212; color: #ffffff;")
                else:
                    self.setStyleSheet("background-color: #ffffff; color: #000000;")
        except Exception:
            pass

        # apply compact mode
        try:
            if hasattr(self, "compact_mode") and self.compact_mode:
                # example: reduce padding/margins, smaller font sizes
                self.setStyleSheet(self.styleSheet() + "QWidget { margin: 2px; padding: 2px; font-size: 10px; }")
        except Exception:
            pass

        # update window title with app version
        try:
            version = "v1.0.0"  # replace with actual version retrieval if available
            self.setWindowTitle(f"Game Analyzer {version}")
        except Exception:
            pass

        # Optional: start with a test video loaded (for demo or testing)
        try:
            if hasattr(self, "settings") and self.auto_play_on_load:
                last_video = self.settings.get("last_opened_video", "")
                if last_video and os.path.exists(last_video):
                    self.log(f"Auto-loading last video: {last_video}")
                    self.open_video(last_video)
                    self.play_video()
        except Exception:
            pass

        # Optional: mute start if configured
        try:
            if hasattr(self, "settings") and self.start_muted:
                self.volume_slider.setValue(0)
                self.settings["sound_enabled"] = False
                self.save_settings()
        except Exception:
            pass

        # Optional: show welcome message or tutorial
        try:
            if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "welcome_shown.flag")):
                QMessageBox.information(self, "Welcome to Game Analyzer", "Thank you for trying Game Analyzer!\n\n"
                    "This application allows you to analyze gameplay videos using advanced AI models.\n\n"
                    "For best results, use high-quality videos and select the appropriate game type.\n\n"
                    "If you have any questions or feedback, feel free to reach out to the developer.\n\n"
                    "Enjoy analyzing your games!\n\n"
                    "Tip: You can toggle sounds in the settings menu.", QMessageBox.Ok)
                # create a flag file to indicate welcome message has been shown
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "welcome_shown.flag"), "w") as f:
                    f.write("")
        except Exception:
            pass

        # Optional: run initial test analysis (if configured)
        try:
            if hasattr(self, "settings") and self.settings.get("run_initial_analysis", False):
                last_video = self.settings.get("last_opened_video", "")
                if last_video and os.path.exists(last_video):
                    self.log(f"Running initial analysis on load: {last_video}")
                    self.analyze_video_background(last_video)
        except Exception:
            pass

        # Optional: set initial focus to video label or another widget
        try:
            if hasattr(self, "video_label"):
                self.video_label.setFocus()
        except Exception:
            pass

        # Optional: adjust window size or position
        try:
            if hasattr(self, "settings"):
                geom = self.settings.get("window_geometry", "")
                if geom:
                    # example: restore from saved geometry string
                    x, y, w, h = map(int, geom.split(","))
                    self.setGeometry(x, y, w, h)
        except Exception:
            pass

        # Optional: enable/disable specific analyzers by default
        try:
            if hasattr(self, "settings"):
                an = self.settings.get("analyzers", {})
                self.helldivers_checkbox.setChecked(an.get("helldivers", True))
                self.boxing_checkbox.setChecked(an.get("boxing", True))
                self.stats_checkbox.setChecked(an.get("statistics", True))
        except Exception:
            pass

        # Optional: restore last video position (if applicable)
        try:
            if hasattr(self, "settings") and "last_video_position" in self.settings:
                pos = self.settings["last_video_position"]
                self.current_time = max(0, min(self.playback_duration, float(pos)))
                self.log(f"Restored last video position: {self.current_time}s")
        except Exception:
            pass

        # Optional: show debug info overlay (if configured)
        try:
            if hasattr(self, "settings") and self.settings.get("show_debug_info", False):
                self.statusBar().showMessage("Debug: " + str(self.settings), 10000)
        except Exception:
            pass

        # Optional: run any custom startup scripts or commands
        try:
            # example: run a Python script or external command
            startup_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup.py")
            if os.path.exists(startup_script):
                self.log(f"Running custom startup script: {startup_script}")
                exec(open(startup_script).read(), {"app": self})
        except Exception as e:
            self.log(f"Error running startup script: {e}")

        # Optional: check for updates or new features
        try:
            # example: simple version check against latest release
            current_version = "1.0.0"  # replace with actual version retrieval
            latest_version = "1.0.1"   # replace with actual check
            if current_version < latest_version:
                QMessageBox.information(self, "Update Available", f"A new version {latest_version} is available!",
                    QMessageBox.Ok)
        except Exception:
            pass

        # Optional: log app startup completion
        self.log("App startup sequence completed.")

        # Optional: additional post-startup tasks
        try:
            # example: pre-load some data or models in
            # (no-op placeholder so the try has a body; replace with real tasks if needed)
            pass
        except Exception as e:
            # Ensure exceptions during optional startup tasks are logged but don't crash the app
            try:
                self.log(f"Post-startup task failed: {e}")
            except Exception:
                pass

        # create the analysis widget and dock (hidden by default)
        try:
            self.opencv_widget = OpencvAnalysisWindow(self)
            self.opencv_dock = QDockWidget("OpenCV Analysis", self)
            self.opencv_dock.setObjectName("OpenCVAnalysisDock")
            self.opencv_dock.setWidget(self.opencv_widget)
            # add dock to the right side; keep hidden until the user opens it
            try:
                self.addDockWidget(Qt.RightDockWidgetArea, self.opencv_dock)
                self.opencv_dock.hide()
            except Exception:
                # fallback: keep references and show when requested
                self.opencv_dock = None
        except Exception:
            self.opencv_widget = None
            self.opencv_dock = None

        # replace/augment previous "open analysis window" wiring to show dock
        try:
            # ensure action/button calls the dock show helper
            if hasattr(self, "action_open_analysis"):
                self.action_open_analysis.triggered.connect(lambda: self._show_opencv_window())
            if hasattr(self, "btn_open_opencv_window"):
                self.btn_open_opencv_window.clicked.connect(lambda: self._show_opencv_window())
        except Exception:
            pass

        # connect background analysis outputs to the docked widget (defensive)
        try:
            def _forward_frame(qimage):
                try:
                    if not self.opencv_widget:
                        return
                    # show the dock on first frame if hidden
                    if self.opencv_dock and not self.opencv_dock.isVisible():
                        self.opencv_dock.show()
                    self.opencv_widget.update_frame(qimage)
                except Exception:
                    pass

            def _forward_event(ev):
                try:
                    if not self.opencv_widget:
                        return
                    if self.opencv_dock and not self.opencv_dock.isVisible():
                        self.opencv_dock.show()
                    self.opencv_widget.add_event(ev)
                except Exception:
                    pass

            # try known signal names; safe-if-missing
            if hasattr(self, "preview_pixmap") and hasattr(self.preview_pixmap, "connect"):
                try:
                    self.preview_pixmap.connect(_forward_frame)
                except Exception:
                    pass
            if hasattr(self, "preview_signal") and hasattr(self.preview_signal, "connect"):
                try:
                    self.preview_signal.connect(_forward_frame)
                except Exception:
                    pass

            if hasattr(self, "analysis_event") and hasattr(self.analysis_event, "connect"):
                try:
                    self.analysis_event.connect(_forward_event)
                except Exception:
                    pass
            if hasattr(self, "analysis_results_signal") and hasattr(self.analysis_results_signal, "connect"):
                try:
                    self.analysis_results_signal.connect(_forward_event)
                except Exception:
                    pass
        except Exception:
            pass

    def _show_opencv_window(self):
        try:
            # ensure dock exists and widget present
            if not getattr(self, "opencv_dock", None) or not getattr(self, "opencv_widget", None):
                try:
                    self.opencv_widget = OpencvAnalysisWindow(self)
                    self.opencv_dock = QDockWidget("OpenCV Analysis", self)
                    self.opencv_dock.setObjectName("OpenCVAnalysisDock")
                    self.opencv_dock.setWidget(self.opencv_widget)
                    self.addDockWidget(Qt.RightDockWidgetArea, self.opencv_dock)
                except Exception:
                    return
            # show and focus the dock
            try:
                self.opencv_dock.show()
                self.opencv_dock.raise_()
                self.opencv_dock.activateWindow()
            except Exception:
                pass
        except Exception:
            pass

    def open_video(self, file_path: str):
        """Open and analyze a video file (wrapper for open_video_dialog)."""
        try:
            self.log(f"open_video called with: {file_path}")
            self.current_video_path = file_path
            self.load_video(file_path)
            self.analyze_video_background(file_path)
            # Optional: auto-play video if configured
            try:
                if self.auto_play_on_load:
                    self.play_video()
            except Exception:
                pass
        except Exception as e:
            self.log(f"open_video error: {e}")
            QMessageBox.critical(self, "Open Video Error", f"Failed to open video:\n{e}")

    def closeEvent(self, event):
        """Handle application close event."""
        try:
            # Optional: prompt to save settings on exit
            if self.settings.get("prompt_save_on_exit", True):
                reply = QMessageBox.question(self, "Save Settings?", "Do you want to save your settings before exiting?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.save_settings()
                elif reply == QMessageBox.Cancel:
                    event.ignore()
                    return
        except Exception:
            pass
        # Ensure any open video or resources are released
        try:
            self.pause_video()
            self.current_clip = None
            if getattr(self, "preview_cap", None) is not None:
                self.preview_cap.release()
                self.preview_cap = None
        except Exception:
            pass
        # Call base class closeEvent
        event.accept()

    def event(self, e):
        """Override event handler to catch and log all events."""
        try:
            # example: log all close events
            if e.type() == e.Close:
                self.log("Close event detected")
        except Exception:
            pass
        return super().event(e)

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseAnalyzer(ABC):
    """Base class for all video/audio analyzers."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model if model_path is provided."""
        pass

    @abstractmethod
    def analyze_frame(self, frame: Any, timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """Analyze a single frame and return events."""
        pass

    @abstractmethod
    def needs_training(self) -> bool:
        """Return True if the analyzer requires training data."""
        return self.model is None

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {"model_path": self.model_path, "loaded": self.model is not None}

import cv2
from typing import Optional, Any
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

class PlaybackController(QObject):
    """Handles video playback, fast-forward, rewind, and frame updates."""

    frame_updated = pyqtSignal(object)  # Emits QImage or frame data
    position_changed = pyqtSignal(float)  # Emits current_time

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_clip = None
        self.preview_cap = None
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_frame)
        self.playing = False
        self.current_time = 0.0
        self.playback_fps = 30.0
        self.playback_duration = 0.0

    def load_video(self, video_path: str) -> bool:
        """Load video using MoviePy or OpenCV fallback."""
        try:
            import moviepy.editor as mpe
            self.current_clip = mpe.VideoFileClip(video_path)
            self.playback_fps = self.current_clip.fps
            self.playback_duration = self.current_clip.duration
            return True
        except ImportError:
            self.preview_cap = cv2.VideoCapture(video_path)
            if self.preview_cap.isOpened():
                self.playback_fps = self.preview_cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.playback_duration = (self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.playback_fps) if self.playback_fps > 0 else 0.0
                return True
        return False

    def play_video(self):
        self.playing = True
        self.playback_timer.start(int(1000 / self.playback_fps))

    def pause_video(self):
        self.playing = False
        self.playback_timer.stop()

    def fast_forward(self, secs: float = 5.0):
        new_time = min(self.current_time + secs, self.playback_duration)
        self.seek_to(new_time)

    def rewind(self, secs: float = 5.0):
        new_time = max(self.current_time - secs, 0.0)
        self.seek_to(new_time)

    def seek_to(self, time: float):
        self.current_time = time
        if self.current_clip:
            # MoviePy seek
            pass  # Implement MoviePy seek
        elif self.preview_cap:
            frame_idx = int(time * self.playback_fps)
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.position_changed.emit(self.current_time)
        self.update_frame()

    def update_frame(self):
        """Update frame based on current_time."""
        if self.current_clip:
            frame = self.current_clip.get_frame(self.current_time)
            # Convert to QImage and emit
            self.frame_updated.emit(frame)
        elif self.preview_cap and self.preview_cap.isOpened():
            ret, frame = self.preview_cap.read()
            if ret:
                # Convert to QImage and emit
                self.frame_updated.emit(frame)
        if self.playing:
            self.current_time += 1 / self.playback_fps
            if self.current_time >= self.playback_duration:
                self.pause_video()