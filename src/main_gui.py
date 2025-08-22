import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QAction, QMenuBar, QLabel, QMessageBox, QPushButton, QSlider, QProgressBar, QHBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

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

            from PyQt5.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QComboBox, QCheckBox, QGroupBox, QHBoxLayout
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

            self.video_label = QLabel("Load a video from File > Open Video", self)
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


if __name__ == "__main__":
    try:
        print("GameAnalyzer: starting application")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write("GameAnalyzer: starting application\n")
    except Exception:
        pass
    try:
        app = QApplication(sys.argv)
        window = GameAnalyzerApp()
        window.show()
        # Optional automated playback test: set TEST_PLAYBACK=1 in environment to exercise play/pause/seek
        try:
            if os.environ.get("TEST_PLAYBACK", "0") == "1":
                import time
                window.log("TEST_PLAYBACK: starting automated playback test")
                # If no preview loaded, try to load last opened video from settings
                last = window.settings.get("last_opened_video")
                if last:
                    window.load_video(last)
                # Start playback, wait, then call controls
                window.play_video()
                app.processEvents()
                time.sleep(1.0)
                window.fast_forward()
                app.processEvents()
                time.sleep(0.5)
                window.rewind()
                app.processEvents()
                time.sleep(0.5)
                window.skip_forward()
                app.processEvents()
                time.sleep(0.5)
                window.skip_backward()
                app.processEvents()
                time.sleep(0.5)
                window.pause_video()
                window.log("TEST_PLAYBACK: finished")
        except Exception:
            pass
        ret = app.exec_()
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"GameAnalyzer: exiting with code {ret}\n")
        except Exception:
            pass
        sys.exit(ret)
    except Exception as e:
        import traceback
        err = f"Unhandled startup error: {e}\n{traceback.format_exc()}"
        print(err)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(err + "\n")
        except Exception:
            pass
        raise