import os
import sys
import time
import yaml
import shutil
import logging
import numpy as np
from functools import partial
import wave
from typing import Optional
from pydub import AudioSegment

from PySide6.QtCore import (
    Qt, QUrl, QThread, QObject, Signal, Slot, QTimer)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QDoubleSpinBox, QSpinBox, QComboBox,
    QSlider,QScrollArea, QTreeWidget, QTreeWidgetItem, QAbstractItemView, 
    QSizePolicy, QFrame, QApplication,QTabWidget, QGroupBox, QMessageBox
)
from PySide6.QtGui import QPainter, QPen, QColor, QLinearGradient
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# Import local modules
import models
from tts_wrapper import KokoroTTSWrapper
import persistence
import error_handler

# --- Constants ---
OUTPUTS_DIR = "outputs"
TEMP_DIR = "temp_audio"
PERSIST_FILENAME = "generations.json"
PERSIST_FILE = os.path.join(OUTPUTS_DIR, PERSIST_FILENAME)
CHUNK_PREFIX = "chunk_"

logger = logging.getLogger(__name__)

# ------------------- Waveform Widget -------------------
class WaveformWidget(QWidget):
    seek_position_signal = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.waveform_data = None
        self.playback_progress = 0.0
        self.audio_player_duration_ms = 0
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def set_waveform(self, data: Optional[np.ndarray]):
        if data is not None and len(data) > 0:
            max_abs = np.max(np.abs(data))
            normalized_data = data / max_abs if max_abs > 0 else data
            target_points = self.width() * 2
            if len(normalized_data) > target_points:
                step = max(1, len(normalized_data) // target_points)
                self.waveform_data = normalized_data[::step]
            else:
                self.waveform_data = normalized_data
        else:
            self.waveform_data = None
        self.update()

    def set_playback_progress(self, progress: float):
        self.playback_progress = max(0.0, min(progress, 1.0))
        self.update()

    def set_audio_duration(self, duration_ms: int):
        self.audio_player_duration_ms = max(0, duration_ms)

    def mousePressEvent(self, event):
        if self.audio_player_duration_ms > 0:
            if event.button() == Qt.MouseButton.LeftButton:
                click_x = event.position().x()
                progress_ratio = click_x / self.width()
                seek_position_ms = int(self.audio_player_duration_ms * progress_ratio)
                self.seek_position_signal.emit(max(0, min(seek_position_ms, self.audio_player_duration_ms)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        # Background
        gradient = QLinearGradient(0, 0, 0, rect.height())
        gradient.setColorAt(0, QColor("#21252b"))
        gradient.setColorAt(1, QColor("#282c34"))
        painter.fillRect(rect, gradient)

        if self.waveform_data is not None and len(self.waveform_data) > 0:
            pen = QPen(QColor("#61afef"))
            pen.setWidth(1)
            painter.setPen(pen)
            
            center_y = rect.height() / 2
            scale_factor = center_y * 0.90
            num_points = len(self.waveform_data)
            
            if rect.width() > 0:
                points_per_pixel = num_points / rect.width()
                for x in range(rect.width()):
                    idx = int(x * points_per_pixel)
                    if idx >= num_points: 
                        break
                    val = self.waveform_data[idx]
                    y1 = int(center_y - val * scale_factor)
                    y2 = int(center_y + val * scale_factor)
                    painter.drawLine(x, y1, x, y2)

        # Progress Line
        progress_x = int(rect.width() * self.playback_progress)
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawLine(progress_x, 0, progress_x, rect.height())
        
        # Overlay
        overlay_rect = rect.adjusted(0, 0, progress_x - rect.width(), 0)
        painter.fillRect(overlay_rect, QColor(255, 255, 255, 20))
        painter.end()

# ------------------- Worker Thread -------------------
class SynthesisWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object) 
    error = Signal(str)
    file_ready = Signal(str, np.ndarray)

    def __init__(self, tts_wrapper: Optional[KokoroTTSWrapper], parent=None):
        super().__init__(parent)
        self.tts_wrapper = tts_wrapper
        self._is_synthesizing = False
        self._stop_requested = False
        self.start_time = 0
    
    @Slot()
    def stop(self):
        """Signals the worker to stop after the current chunk."""
        self._stop_requested = True
        logger.info("Worker stop requested...")

    @Slot(object)
    def synthesize(self, args: dict):
        if self._is_synthesizing: 
            return
        if not self.tts_wrapper:
            self.error.emit("TTS Engine not initialized.")
            return

        self._is_synthesizing = True
        self._stop_requested = False
        self.start_time = time.time()

        try:
            segments = args.get("segments")
            if not segments:
                self.error.emit("No text to synthesize.")
                self._is_synthesizing = False
                return

            # Extract params
            speed = args.get("speed", 1.0)
            pitch = args.get("pitch", 1.0)
            sample_rate = args.get("sample_rate", 24000)
            output_format = args.get("output_format", "WAV")
            
            def check_stop_progress(curr, total):
                if self._stop_requested:
                    raise InterruptedError("Synthesis stopped by user.")
                self.progress.emit(curr, total)

            # Pass to wrapper
            results = self.tts_wrapper.synthesize(
                segments=segments,
                speed=speed,
                pitch=pitch,
                sample_rate=sample_rate,
                output_format=output_format,
                progress_callback=check_stop_progress
            )
            
            synthesis_result_list, combined_filepath = results
            elapsed = time.time() - self.start_time
            logger.info(f"Worker finished in {elapsed:.2f}s")

            # Load waveform if combined file exists
            if combined_filepath and os.path.exists(combined_filepath):
                waveform = self._load_waveform_data(combined_filepath)
                if waveform is not None:
                    self.file_ready.emit(combined_filepath, waveform)
            
            self.finished.emit(results)

        except Exception as e:
            logger.exception("Worker Error")
            self.error.emit(str(e))
        finally:
            self._is_synthesizing = False

    def _load_waveform_data(self, filepath: str) -> Optional[np.ndarray]:
        """Helper to load waveform safely (supports mp3 via pydub fallback)."""
        if not filepath or not os.path.exists(filepath):
            return None
            
        try:
            # Native WAV (Fast)
            if filepath.lower().endswith(".wav"):
                with wave.open(filepath, "rb") as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    dtype = np.int16 if wav_file.getsampwidth() == 2 else np.uint8
                    audio_data = np.frombuffer(frames, dtype=dtype)
                    if dtype == np.int16: 
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    else: 
                        audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
                    if wav_file.getnchannels() > 1:
                        audio_data = audio_data.reshape(-1, wav_file.getnchannels()).mean(axis=1)
                    
                    # Normalize
                    peak = np.max(np.abs(audio_data))
                    return audio_data / peak if peak > 1e-6 else audio_data
            
            # Fallback/MP3 via Pydub
            audio = AudioSegment.from_file(filepath)
            if audio.channels > 1: 
                audio = audio.set_channels(1)
            raw = np.array(audio.get_array_of_samples())
            
            # Normalize
            if audio.sample_width == 2: 
                max_val = 32768.0
            else: 
                max_val = float(2**(8*audio.sample_width - 1))
            
            normalized = raw.astype(np.float32) / max_val
            return normalized
            
        except Exception as e:
            self.error.emit("Waveform loader failed for {filepath}: {e}")
            return None

class FileLoaderWorker(QObject):
    """Worker to load/parse text files in background."""
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            lines = []
            
            # --- EPUB LOGIC ---
            if self.path.lower().endswith(".epub"):
                try:
                    # Импортираме САМО ТУК, в нишката
                    import ebooklib
                    from ebooklib import epub
                    from bs4 import BeautifulSoup
                    import warnings
                    warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
                    warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")
                except ImportError:
                    self.error.emit("EPUB support requires libraries.\nPlease run: pip install EbookLib beautifulsoup4")
                    return

                book = epub.read_epub(self.path)
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        soup = BeautifulSoup(item.get_content(), 'html.parser')
                        text = soup.get_text()
                        chunk_lines = [line.strip() for line in text.splitlines() if line.strip()]
                        lines.extend(chunk_lines)

            # --- TXT LOGIC ---
            else:
                with open(self.path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

            if not lines:
                self.error.emit("File appears empty or could not be parsed.")
            else:
                self.finished.emit(lines)

        except Exception as e:
            self.error.emit(f"Failed to load file: {str(e)}")


# ------------------- Main Window v2.0 -------------------
class MyTTSMainWindow(QMainWindow):
    synthesize_args_signal = Signal(object)

    def __init__(self, config_path="config.yaml"):
        super().__init__()
        self.setWindowTitle("Kokoro Studio v2.0")
        self.resize(1200, 800)
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.synthesis_results = persistence.load_generations(PERSIST_FILE)
        
        # Audio State
        self.current_filepath = None
        self.stored_duration = 0
        self.audio_output = None
        self.media_player = None

        # Init Engine & Thread
        self._init_engine()
        self._init_media_player()

        # --- UI Construction ---
        self._setup_ui()
        
        # Load Data
        self.refresh_voice_list()
        self.populate_history_table()
        self.statusBar().showMessage("Ready.")

    def _init_engine(self):
        try:
            self.tts_wrapper = KokoroTTSWrapper(output_dir=OUTPUTS_DIR, temp_sub_dir=TEMP_DIR, config=self.config)
            self.synthesis_thread = QThread(self)
            self.synthesis_worker = SynthesisWorker(self.tts_wrapper)
            self.synthesis_worker.moveToThread(self.synthesis_thread)
            
            # Connections
            self.synthesis_worker.progress.connect(self.update_progress)
            self.synthesis_worker.finished.connect(self.on_synthesis_finished)
            self.synthesis_worker.error.connect(self.on_synthesis_error)
            self.synthesis_worker.file_ready.connect(self.on_file_ready_for_playback)
            self.synthesize_args_signal.connect(self.synthesis_worker.synthesize)
            
            self.synthesis_thread.start()
        except Exception as e:
            error_handler.show_error(self, f"Engine Init Failed: {e}")

    def _init_media_player(self):
        self.audio_output = QAudioOutput(self)
        self.media_player = QMediaPlayer(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.7)
        
        self.media_player.positionChanged.connect(self.on_player_position_changed)
        self.media_player.durationChanged.connect(self.on_player_duration_changed)
        self.media_player.playbackStateChanged.connect(self.on_player_state_changed)
        self.media_player.errorOccurred.connect(lambda e, s: error_handler.show_error(self,"Player Error: {s}"))
        

    # --- UI SETUP ---
    def _setup_ui(self):
        self.setAcceptDrops(True)
        self._apply_stylesheet()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Sidebar (Left) with Animation support
        self.sidebar_container = QWidget()
        self.sidebar_container.setFixedWidth(300)
        self.sidebar_container.setObjectName("Sidebar")
        self._create_sidebar()
        main_layout.addWidget(self.sidebar_container)

        # Toggle Button (Thin strip)
        self.btn_toggle_sidebar = QPushButton("<<")
        self.btn_toggle_sidebar.setFixedWidth(20)
        self.btn_toggle_sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.btn_toggle_sidebar.setStyleSheet("background-color: #21252b; border: none; color: #5c6370;")
        self.btn_toggle_sidebar.clicked.connect(self.toggle_sidebar)
        main_layout.addWidget(self.btn_toggle_sidebar)

        # 2. Right Side (Content + Footer)
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 2.1 Content Tabs
        self.tabs = QTabWidget()
        self._create_tabs()
        right_layout.addWidget(self.tabs, 1)

        # 2.2 Footer Player
        self.footer = QWidget()
        self.footer.setObjectName("Footer")
        self.footer.setFixedHeight(120) 
        self._create_footer()
        right_layout.addWidget(self.footer)

        main_layout.addWidget(right_container)

    def _create_sidebar(self):
        """Creates the Sidebar with vertical Groups (No Tabs)."""
        # 1. Main Layout for the sidebar container
        main_layout = QVBoxLayout(self.sidebar_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 2. Scroll Area (Safety for small screens)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # 3. Content Widget inside Scroll
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(20) # Space between groups

        # --- GROUP 1: Voice & Mixing ---
        group_voice = QGroupBox("🎙️ Voice & Mixing")
        gv_layout = QVBoxLayout(group_voice)
        gv_layout.setSpacing(10)

        gv_layout.addWidget(QLabel("Primary Voice:"))
        self.voice_combo_1 = QComboBox()
        self.voice_combo_1.setToolTip("Select the main speaker")
        gv_layout.addWidget(self.voice_combo_1)

        # Mixing Checkbox (Styled as a toggle button or just check)
        self.enable_blend_check = QPushButton("Mix with another voice (OFF)")
        self.enable_blend_check.setCheckable(True)
        self.enable_blend_check.setStyleSheet("text-align: left; padding: 5px;")
        self.enable_blend_check.toggled.connect(self._toggle_blend_ui)
        gv_layout.addWidget(self.enable_blend_check)

        # Secondary Voice
        self.lbl_voice_2 = QLabel("Secondary Voice:")
        self.voice_combo_2 = QComboBox()
        self.lbl_voice_2.setVisible(False)
        self.voice_combo_2.setVisible(False)
        gv_layout.addWidget(self.lbl_voice_2)
        gv_layout.addWidget(self.voice_combo_2)
        
        layout.addWidget(group_voice)

        # --- GROUP 2: Audio Properties ---
        group_audio = QGroupBox("🎛️ Audio Properties")
        ga_layout = QFormLayout(group_audio)
        ga_layout.setSpacing(10)

        # Speed
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setToolTip("Speech speed (1.0 = Normal)")
        ga_layout.addRow("Speed:", self.speed_spin)

        # Pitch
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(50, 200)
        self.pitch_slider.setValue(100)
        self.pitch_label = QLabel("1.00")
        self.pitch_slider.valueChanged.connect(lambda v: self.pitch_label.setText(f"{v/100:.2f}"))
        pitch_con = QHBoxLayout()
        pitch_con.addWidget(self.pitch_slider)
        pitch_con.addWidget(self.pitch_label)
        ga_layout.addRow("Pitch:", pitch_con)

        # Hz
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["24000", "22050", "16000", "8000"])
        saved_hz_index = self.config.get('tts_params', {}).get('sample_rate', 0)
        if saved_hz_index < 4:
            self.sample_rate_combo.setCurrentIndex(saved_hz_index)
            self.sample_rate_combo.setToolTip("Sample Rate (Quality)")
        ga_layout.addRow("Hz:", self.sample_rate_combo)

        # Format
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["WAV", "MP3"])
        saved_fmt = self.config.get('tts_params', {}).get('save_format', "WAV")
        self.save_format_combo.setCurrentText(saved_fmt)
        
        ga_layout.addRow("Format:", self.save_format_combo)
        
        layout.addWidget(group_audio)

        # --- GROUP 3: System & Seed ---
        group_sys = QGroupBox("⚙️ System")
        gs_layout = QFormLayout(group_sys)

        # CPU or GPU
        self.lbl_device = QLabel("Checking...")
        
        if self.tts_wrapper and self.tts_wrapper.device == 'cuda':
            self.lbl_device.setText("GPU (CUDA) 🚀")
            self.lbl_device.setToolTip("You are using CUDA")
            self.lbl_device.setStyleSheet("color: #98c379; font-weight: bold; border: none;") 
        else:
            self.lbl_device.setText("CPU (Slow) 🐢")
            self.lbl_device.setToolTip("You are using CPU")
            self.lbl_device.setStyleSheet("color: #e5c07b; font-weight: bold; border: none;")
            
        gs_layout.addRow("Device:", self.lbl_device)
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999999)

        saved_seed = self.config.get('tts_params', {}).get('seed', 0)
        self.seed_spin.setValue(saved_seed)

        self.seed_spin.setSpecialValueText("Random")
        self.seed_spin.setToolTip("Set specific seed for reproducibility. 0 = Random.")
        gs_layout.addRow("Seed:", self.seed_spin)
        
        # Clear Temp Files
        btn_clean = QPushButton("🧹 Clean Temp")
        btn_clean.setToolTip("Delete temporary chunk files to free space")
        btn_clean.clicked.connect(self.clear_temp_files)
        gs_layout.addRow(btn_clean)

        layout.addWidget(group_sys)

        # Spacer at bottom
        layout.addStretch()

        # Refresh Button
        btn_refresh = QPushButton("🔄 Refresh Voices")
        btn_refresh.setToolTip("Reload voice list from folder")
        btn_refresh.clicked.connect(self.refresh_voice_list)
        layout.addWidget(btn_refresh)

        # Finalize Scroll
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        self.alpha_slider = QDoubleSpinBox() 
        self.beta_slider = QDoubleSpinBox()
        self.diffusion_slider = QSlider()
        self.scale_slider = QDoubleSpinBox()

    def _toggle_blend_ui(self, checked):
        """Shows/Hides the secondary voice dropdown."""
        self.lbl_voice_2.setVisible(checked)
        self.voice_combo_2.setVisible(checked)
        
        status = "ON" if checked else "OFF"
        self.enable_blend_check.setText(f"Mix with another voice ({status})")
        
        if hasattr(self, 'sidebar_container'):
            self.sidebar_container.updateGeometry()

    def toggle_sidebar(self):
        visible = self.sidebar_container.isVisible()
        self.sidebar_container.setVisible(not visible)
        self.btn_toggle_sidebar.setText(">>" if visible else "<<")

    def _create_tabs(self):
        # Tab 1: Scratchpad
        self.tab_scratch = QWidget()
        scratch_layout = QVBoxLayout(self.tab_scratch)
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Type or paste text here...")
        btn_synth_scratch = QPushButton("Synthesize Text")
        btn_synth_scratch.setMinimumHeight(50)
        btn_synth_scratch.setStyleSheet("background-color: #61afef; color: white; font-weight: bold; font-size: 14px;")
        btn_synth_scratch.clicked.connect(self.on_synthesize_scratch)
        
        scratch_layout.addWidget(self.text_edit)
        scratch_layout.addWidget(btn_synth_scratch)
        self.tabs.addTab(self.tab_scratch, "Scratchpad")

        # Tab 2: Audiobook
        self.tab_book = QWidget()
        book_layout = QVBoxLayout(self.tab_book)
        
        # Toolbar
        toolbar = QHBoxLayout()
        btn_open_proj = QPushButton("📂 Open Project")
        btn_open_proj.clicked.connect(self.load_project)
        btn_save_proj = QPushButton("💾 Save Project")
        btn_save_proj.clicked.connect(self.save_project)

        btn_load_txt = QPushButton("📄 Import File")
        btn_load_txt.clicked.connect(self.open_file)
        
        # Global Voice Change
        btn_apply = QPushButton("⬇️ Apply Global Voice")
        btn_apply.setToolTip("Sets ALL rows to use the Voice selected in the Sidebar")
        btn_apply.clicked.connect(self.apply_global_settings_to_book)
        
        btn_add = QPushButton("➕ Add Line")
        btn_add.setToolTip("Add New Text Line")

        btn_add.clicked.connect(self.add_empty_row)
        btn_clear_book = QPushButton("🗑️ Clear")
        btn_clear_book.setToolTip("Clear Unwanted Lines")
        btn_clear_book.clicked.connect(self.clear_book_table)
        
        toolbar.addWidget(btn_open_proj)
        toolbar.addWidget(btn_save_proj)
        toolbar.addWidget(btn_load_txt)
        toolbar.addWidget(btn_apply)
        toolbar.addWidget(btn_add)
        toolbar.addWidget(btn_clear_book)
        toolbar.addStretch()
        
        self.book_table = QTableWidget(0, 4)
        self.book_table.setHorizontalHeaderLabels(["Text Segment", "Primary Voice", "Secondary Voice", "Actions"])
        header = self.book_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        
        self.btn_render = QPushButton("🎬 Render Audiobook")
        self.btn_render.setMinimumHeight(50)
        self.btn_render.setStyleSheet("background-color: #98c379; color: black; font-weight: bold; font-size: 14px;")
        self.btn_render.clicked.connect(self.on_synthesize_book)

        book_layout.addLayout(toolbar)
        book_layout.addWidget(self.book_table)
        book_layout.addWidget(self.btn_render)
        self.tabs.addTab(self.tab_book, "Audiobook Mode")

        # Tab 3: History (Tree View)
        self.tab_history = QWidget()
        hist_layout = QVBoxLayout(self.tab_history)
        
        self.history_tree = QTreeWidget()
        self.history_tree.setHeaderLabels(["ID / Content", "Type", "Time Created", "Actions"])
        
        header = self.history_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch) # Content is wide
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(3, 160) # 
        
        self.history_tree.setAlternatingRowColors(True)
        self.history_tree.setRootIsDecorated(True)
        
        btn_clear_hist = QPushButton("🗑️ Clear History")
        btn_clear_hist.clicked.connect(self.clear_history)
        
        hist_layout.addWidget(self.history_tree)
        hist_layout.addWidget(btn_clear_hist)
        self.tabs.addTab(self.tab_history, "History")

    def _create_footer(self):
        layout = QHBoxLayout(self.footer)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Waveform
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.seek_position_signal.connect(self.seek_audio)
        layout.addWidget(self.waveform_widget, 4)

        # Controls
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(5)
        
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl_layout.addWidget(self.lbl_time)

        btns = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(50)
        self.btn_play.clicked.connect(self.on_play_pause)
        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setFixedWidth(50)
        self.btn_stop.clicked.connect(self.stop_audio)
        
        btns.addWidget(self.btn_play)
        btns.addWidget(self.btn_stop)
        ctrl_layout.addLayout(btns)
        
        self.slider_vol = QSlider(Qt.Orientation.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(70)
        self.slider_vol.setFixedWidth(110)
        self.slider_vol.valueChanged.connect(lambda v: self.audio_output.setVolume(v/100) if self.audio_output else None)
        ctrl_layout.addWidget(self.slider_vol)

        layout.addLayout(ctrl_layout, 1)

    # --- LOGIC ---

    def _get_common_args(self):
        return {
            "speed": self.speed_spin.value(),
            "pitch": self.pitch_slider.value() / 100.0,
            "sample_rate": int(self.sample_rate_combo.currentText()),
            "output_format": self.save_format_combo.currentText(),

            # Seed
            "seed": self.seed_spin.value(),

            # Defaults for unused params to prevent worker crash
            "alpha": 0, "beta": 0, "diffusion_steps": 0, "embedding_scale": 1,
            "config": self.config
        }

    @Slot()
    def on_synthesize_scratch(self):
        text = self.text_edit.toPlainText().strip()
        if not text: 
            return
        
        v1 = self.voice_combo_1.currentText()
        v2 = self.voice_combo_2.currentText()
        
        if self.enable_blend_check.isChecked() and v2 and v2 != "None (Single Voice)":
            voice = f"{v1}+{v2}"
        else:
            voice = v1
            
        args = self._get_common_args()
        args["segments"] = [(text, [voice], None)]
        self.statusBar().showMessage("Synthesizing...")
        self.synthesize_args_signal.emit(args)

    @Slot()
    def on_synthesize_book(self):
        """Generates the book. Changes button to STOP after 2 seconds."""
        segments = []
        for row in range(self.book_table.rowCount()):
            text_item = self.book_table.item(row, 0)
            widget_v1 = self.book_table.cellWidget(row, 1)
            widget_v2 = self.book_table.cellWidget(row, 2)
            
            if text_item is None:
                continue
            if not isinstance(widget_v1, QComboBox) or not isinstance(widget_v2, QComboBox):
                continue

            txt = text_item.text().strip()
            if not txt:
                continue
            
            v1 = widget_v1.currentText()
            v2 = widget_v2.currentText()
            voice = f"{v1}+{v2}" if (v2 and v2 != "None (Single Voice)") else v1
            segments.append((txt, [voice], None))
            
        if not segments:
            error_handler.show_error(self, "No segments found.")
            return
            
        args = self._get_common_args()
        args["segments"] = segments
        
        # --- UI STATE CHANGE (PANIC BUTTON) ---
        self.btn_render.setText("⏳ Starting...")
        self.btn_render.setEnabled(False) # Prevent double click
        self.btn_render.setStyleSheet("background-color: #e5c07b; color: black; font-weight: bold; font-size: 14px;")
        
        self.synthesize_args_signal.emit(args)
        
        # Enable STOP button after 2 seconds
        QTimer.singleShot(2000, self._enable_stop_button)

    def _enable_stop_button(self):
        if self.synthesis_worker._is_synthesizing:
            self.btn_render.setText("🛑 STOP GENERATION")
            self.btn_render.setStyleSheet("background-color: #e06c75; color: white; font-weight: bold; font-size: 14px;")
            self.btn_render.setEnabled(True)
            try: 
                self.btn_render.clicked.disconnect()
            except: 
                pass
            self.btn_render.clicked.connect(self.stop_generation)

    def stop_generation(self):
        self.synthesis_worker.stop()
        self.btn_render.setText("Stopping...")
        self.btn_render.setEnabled(False)

    def _reset_render_button(self):
        """Restores the Render button to normal state."""
        self.btn_render.setText("🎬 Render Audiobook")
        self.btn_render.setStyleSheet("background-color: #98c379; color: black; font-weight: bold; font-size: 14px;")
        self.btn_render.setEnabled(True)
        try: 
            self.btn_render.clicked.disconnect()
        except: 
            pass
        self.btn_render.clicked.connect(self.on_synthesize_book)

    def preview_row(self, row):
        """Play specific row in Audiobook table."""
        text_item = self.book_table.item(row, 0)
        widget_v1 = self.book_table.cellWidget(row, 1)
        widget_v2 = self.book_table.cellWidget(row, 2)
        
        if not text_item: 
            return
        if not isinstance(widget_v1, QComboBox) or not isinstance(widget_v2, QComboBox):
            return

        txt = text_item.text().strip()
        v1 = widget_v1.currentText() 
        v2 = widget_v2.currentText() 
        
        if v2 and v2 != "None (Single Voice)": 
            voice = f"{v1}+{v2}"
        else: 
            voice = v1
        
        args = self._get_common_args()
        args["segments"] = [(txt, [voice], None)]
        self.statusBar().showMessage(f"Previewing Row {row+1}...")
        self.synthesize_args_signal.emit(args)

    # --- File Handling ---
    def save_project(self):
        """Saves current table to file using persistence module."""
        if self.book_table.rowCount() == 0: 
            return
        
        data = []
        for row in range(self.book_table.rowCount()):
            text_item = self.book_table.item(row, 0)
            w_v1 = self.book_table.cellWidget(row, 1)
            w_v2 = self.book_table.cellWidget(row, 2)
            
            if text_item and isinstance(w_v1, QComboBox) and isinstance(w_v2, QComboBox):
                data.append({
                    "text": text_item.text(),
                    "v1": w_v1.currentText(),
                    "v2": w_v2.currentText()
                })
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Kokoro Project (*.kproj);;JSON (*.json)")
        
        if path:
            if persistence.save_project_file(path, data):
                self.statusBar().showMessage(f"Project saved: {os.path.basename(path)}", 3000)
            else:
                error_handler.show_error(self, "Failed to save project. Check logs.")

    def load_project(self, path=None):
        """Loads a project using persistence module."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Kokoro Project (*.kproj);;JSON (*.json)")
        
        if not path: 
            return
        
        data = persistence.load_project_file(path)
        
        if data is None:
            error_handler.show_error(self, "Failed to load project file.")
            return

        try:
            self.book_table.setRowCount(0)
            self.book_table.setUpdatesEnabled(False)
            av_voices = self.list_available_voices()
            
            for entry in data:
                row = self.book_table.rowCount()
                self.book_table.insertRow(row)
                
                # Text Anchor
                item_text = QTableWidgetItem(entry.get("text", ""))
                self.book_table.setItem(row, 0, item_text)
                
                # Voice 1
                c1 = QComboBox()
                c1.addItems(av_voices)
                c1.setCurrentText(entry.get("v1", av_voices[0]))
                self.book_table.setCellWidget(row, 1, c1)
                
                # Voice 2
                c2 = QComboBox()
                c2.addItem("None (Single Voice)")
                c2.addItems(av_voices)
                c2.setCurrentText(entry.get("v2", "None (Single Voice)"))
                self.book_table.setCellWidget(row, 2, c2)
                
                # Actions
                wid = QWidget()
                lay = QHBoxLayout(wid)
                lay.setContentsMargins(0,0,0,0)
                btn_play = QPushButton("▶")
                btn_play.setFixedWidth(30)
                btn_play.clicked.connect(lambda _, x=item_text: self.preview_row(self.book_table.row(x)))
                
                btn_del = QPushButton("❌")
                btn_del.setFixedWidth(30)
                btn_del.clicked.connect(lambda _, x=item_text: self.delete_row_by_anchor(x))
                
                lay.addWidget(btn_play)
                lay.addWidget(btn_del)
                self.book_table.setCellWidget(row, 3, wid)
                
            self.book_table.setUpdatesEnabled(True)
            self.tabs.setCurrentIndex(1)
            self.statusBar().showMessage(f"Project loaded: {len(data)} lines.", 3000)
            
        except Exception as e:
            error_handler.show_error(self,"UI Population Error: {e}")

    def load_text_to_table(self, path):
        """Starts background thread to load file."""
        # 1. Show Loading State
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}... please wait.")
        self.book_table.setEnabled(False) # Lock table while loading
        
        # 2. Setup Thread
        self.loader_thread = QThread()
        self.loader_worker = FileLoaderWorker(path)
        self.loader_worker.moveToThread(self.loader_thread)
        
        # 3. Connect Signals
        self.loader_thread.started.connect(self.loader_worker.run)
        self.loader_worker.finished.connect(self.on_file_load_success)
        self.loader_worker.error.connect(self.on_file_load_error)
        
        # Cleanup when done
        self.loader_worker.finished.connect(self.loader_thread.quit)
        self.loader_worker.finished.connect(self.loader_worker.deleteLater)
        self.loader_thread.finished.connect(self.loader_thread.deleteLater)
        
        # 4. Start
        self.loader_thread.start()

    @Slot(list)
    def on_file_load_success(self, lines):
        """Called when file is parsed successfully."""
        try:
            self.book_table.setRowCount(0)
            self.book_table.setUpdatesEnabled(False) # Optimization for rendering
            
            av_voices = self.list_available_voices()
            def_v1 = self.voice_combo_1.currentText()
            
            # --- Optimization: Batch Insert ---
            for line in lines:
                row = self.book_table.rowCount()
                self.book_table.insertRow(row)
                
                # Text Anchor
                item_text = QTableWidgetItem(line)
                item_text.setToolTip(line[:100]) # Tooltip
                self.book_table.setItem(row, 0, item_text)
                
                # Voice 1
                c1 = QComboBox()
                c1.addItems(av_voices)
                c1.setCurrentText(def_v1)
                self.book_table.setCellWidget(row, 1, c1)
                
                # Voice 2
                c2 = QComboBox()
                c2.addItem("None (Single Voice)")
                c2.addItems(av_voices)
                self.book_table.setCellWidget(row, 2, c2)
                
                # Actions
                wid = QWidget()
                lay = QHBoxLayout(wid)
                lay.setContentsMargins(0,0,0,0)
                
                btn_play = QPushButton("▶")
                btn_play.setFixedWidth(30)
                # Use item_text as anchor
                btn_play.clicked.connect(lambda _, x=item_text: self.preview_row(self.book_table.row(x)))
                
                btn_del = QPushButton("❌")
                btn_del.setFixedWidth(30)
                btn_del.clicked.connect(lambda _, x=item_text: self.delete_row_by_anchor(x))
                
                lay.addWidget(btn_play)
                lay.addWidget(btn_del)
                self.book_table.setCellWidget(row, 3, wid)

            self.tabs.setCurrentIndex(1) 
            self.statusBar().showMessage(f"Loaded {len(lines)} lines.")
            
        except Exception as e:
            self.on_file_load_error(f"Rendering error: {e}")
        finally:
            self.book_table.setUpdatesEnabled(True)
            self.book_table.setEnabled(True) # Unlock table

    @Slot(str)
    def on_file_load_error(self, msg):
        """Called if loading fails."""
        self.book_table.setEnabled(True)
        self.statusBar().showMessage("Load failed.")
        error_handler.show_error(self, msg)

    def open_file(self):
        """Opens file dialog and passes path to helper."""
        path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Books (*.txt *.epub);;Text (*.txt);;EPUB (*.epub)")
        if path:
            self.load_text_to_table(path)

    def apply_global_settings_to_book(self):
        """Sets all rows in the book table to match the Sidebar voice settings."""
        if self.book_table.rowCount() == 0: 
            return
        
        # Взимаме настройките от лявото меню
        global_v1 = self.voice_combo_1.currentText()
        global_v2 = self.voice_combo_2.currentText()
        use_mix = self.enable_blend_check.isChecked()
        
        # Спираме обновяването за бързодействие
        self.book_table.setUpdatesEnabled(False)
        
        for row in range(self.book_table.rowCount()):
            widget_v1 = self.book_table.cellWidget(row, 1)
            widget_v2 = self.book_table.cellWidget(row, 2)
            
            if isinstance(widget_v1, QComboBox):
                widget_v1.setCurrentText(global_v1)
            
            if isinstance(widget_v2, QComboBox):
                if use_mix:
                    widget_v2.setCurrentText(global_v2)
                else:
                    widget_v2.setCurrentIndex(0) # None
        
        self.book_table.setUpdatesEnabled(True)
        self.statusBar().showMessage("Global voice settings applied to all rows.", 2000)

    def add_empty_row(self):
        row = self.book_table.rowCount()
        self.book_table.insertRow(row)
        
        item_text = QTableWidgetItem("New Line...")
        self.book_table.setItem(row, 0, item_text)
        
        av_voices = self.list_available_voices()
        
        c1 = QComboBox()
        c1.addItems(av_voices)
        c1.setCurrentText(self.voice_combo_1.currentText())
        self.book_table.setCellWidget(row, 1, c1)
        
        c2 = QComboBox()
        c2.addItem("None (Single Voice)")
        c2.addItems(av_voices)
        self.book_table.setCellWidget(row, 2, c2)
        
        wid = QWidget()
        lay = QHBoxLayout(wid)
        lay.setContentsMargins(0,0,0,0)
        
        # Play Button
        btn_play = QPushButton("▶")
        btn_play.setFixedWidth(30)
        btn_play.clicked.connect(lambda: self.preview_row(self.book_table.row(item_text)))
        
        # Delete Button
        btn_del = QPushButton("❌")
        btn_del.setFixedWidth(30)
        btn_del.clicked.connect(lambda: self.delete_row_by_anchor(item_text))
        
        lay.addWidget(btn_play)
        lay.addWidget(btn_del)
        self.book_table.setCellWidget(row, 3, wid)

    def clear_book_table(self):
        self.book_table.setRowCount(0)
    
    def delete_row_by_anchor(self, item_anchor):
        """Safely deletes the row containing the specific text item."""
        row = self.book_table.row(item_anchor)
        if row >= 0:
            self.book_table.removeRow(row)

    # --- History ---
    def populate_history_table(self):
        """Populates the QTreeWidget with history data (Smart Buttons)."""
        self.history_tree.clear()
        
        # Iterate backwards (newest first)
        for idx, gen in enumerate(reversed(self.synthesis_results)):
            real_idx = len(self.synthesis_results) - 1 - idx
            
            combined_path = gen.get("combined", "")
            chunks = gen.get("chunks", [])
            source_type = "Book" if gen.get("text_source") == "segmented" else "Quick"
            timestamp = time.strftime('%H:%M:%S', time.localtime(gen.get("timestamp", 0)))
            
            # Title logic
            if source_type == "Book":
                title = f"📖 Audiobook Gen #{real_idx+1} ({len(chunks)} segments)"
            else:
                first_text = chunks[0].get("graphemes", "") if chunks else "No Text"
                title = f"📝 {first_text[:40]}..."

            top_item = QTreeWidgetItem([title, source_type, timestamp, ""])
            self.history_tree.addTopLevelItem(top_item)
            top_item.setExpanded(False)

            # --- 1. Actions for Top Item (Combined) ---
            wid = QWidget()
            hlay = QHBoxLayout(wid)
            hlay.setContentsMargins(2, 2, 2, 2)
            hlay.setSpacing(5)
            
            # Play Button (Smart)
            btn_play = QPushButton("▶")
            btn_play.setFixedWidth(30)
            if combined_path and os.path.exists(combined_path):
                btn_play.setToolTip(f"Play: {os.path.basename(combined_path)}")
                btn_play.clicked.connect(partial(self.play_audio_file, combined_path))
                btn_play.setStyleSheet("color: #98c379; font-weight: bold;") 
            else:
                btn_play.setEnabled(False) 
                btn_play.setToolTip("File missing or deleted")
                btn_play.setStyleSheet("color: #5c6370;")

            # Save Button (Smart)
            btn_save = QPushButton("💾")
            btn_save.setFixedWidth(30)
            if combined_path and os.path.exists(combined_path):
                btn_save.clicked.connect(partial(self.save_audio_dialog, f"gen_{real_idx}_full", combined_path))
            else:
                btn_save.setEnabled(False)

            # Delete Button (Винаги активен, за да триеш записа от историята)
            btn_del = QPushButton("❌")
            btn_del.setFixedWidth(30)
            btn_del.setToolTip("Delete History Entry")
            btn_del.clicked.connect(partial(self.delete_history_item, real_idx))
            
            hlay.addWidget(btn_play)
            hlay.addWidget(btn_save)
            hlay.addWidget(btn_del)
            self.history_tree.setItemWidget(top_item, 3, wid)

            # --- 2. Add Children (Chunks) ---
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get("graphemes", "???")
                chunk_path = chunk.get("filepath", "")
                
                child_item = QTreeWidgetItem([f"   🗣️ {chunk_text[:60]}...", "Segment", f"#{i+1}", ""])
                top_item.addChild(child_item)
                
                wid_c = QWidget()
                hlay_c = QHBoxLayout(wid_c)
                hlay_c.setContentsMargins(2, 2, 2, 2)
                hlay_c.setSpacing(5)
                
                # Chunk Play (Smart)
                btn_play_c = QPushButton("▶")
                btn_play_c.setFixedWidth(30)
                if chunk_path and os.path.exists(chunk_path):
                    btn_play_c.setToolTip("Play segment")
                    btn_play_c.clicked.connect(partial(self.play_audio_file, chunk_path))
                else:
                    btn_play_c.setEnabled(False)
                    btn_play_c.setToolTip("File cleaned up (Temp)")
                    btn_play_c.setStyleSheet("color: #5c6370;")

                # Copy Text (Винаги активно!)
                btn_copy_c = QPushButton("📋")
                btn_copy_c.setFixedWidth(30)
                btn_copy_c.setToolTip("Copy text to clipboard")
                btn_copy_c.clicked.connect(lambda _, t=chunk_text: QApplication.clipboard().setText(t))
                
                # Chunk Save (Smart)
                btn_save_c = QPushButton("💾")
                btn_save_c.setFixedWidth(30)
                if chunk_path and os.path.exists(chunk_path):
                    btn_save_c.clicked.connect(partial(self.save_audio_dialog, f"gen_{real_idx}_seg_{i}", chunk_path))
                else:
                    btn_save_c.setEnabled(False)

                hlay_c.addWidget(btn_play_c)
                hlay_c.addWidget(btn_copy_c)
                hlay_c.addWidget(btn_save_c)
                self.history_tree.setItemWidget(child_item, 3, wid_c)

    def clear_history(self):
        self.synthesis_results.clear()
        persistence.save_generations(PERSIST_FILE, [])
        self.populate_history_table()

    def delete_history_item(self, original_index):
        """Deletes an entire generation (Combined + Chunks) from disk and history."""
        
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            "Are you sure you want to delete this entry and all associated audio files?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No: 
            return

        self.stop_audio()
        if self.media_player:
            self.media_player.setSource(QUrl())

        try:
            if original_index < 0 or original_index >= len(self.synthesis_results):
                return
            
            item = self.synthesis_results[original_index]

            if item.get("combined"):
                persistence.delete_file(item["combined"])

            for chunk in item.get("chunks", []):
                chunk_path = chunk.get("filepath")
                if chunk_path:
                    persistence.delete_file(chunk_path)

            del self.synthesis_results[original_index]
            persistence.save_generations(PERSIST_FILE, self.synthesis_results)
            
            self.populate_history_table()
            self.statusBar().showMessage("Entry deleted.", 2000)

        except Exception as e:
            error_handler.show_error(self, "Error deleting entry", exception=e)

    def clear_temp_files(self):
        """Manually clears temp folder."""
        # Питаме потребителя
        reply = QMessageBox.question(
            self, "Clean Temp Files", 
            f"Delete all temporary files in '{TEMP_DIR}'?\n(This might break playback of older history items)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.stop_audio() # Освобождаваме файловете
            
            # Извикваме функцията от persistence (която преместихме там)
            # Задаваме 0 дни retention, за да изтрие ВСИЧКО
            persistence.cleanup_temp_files(os.path.join(OUTPUTS_DIR, TEMP_DIR), retention_days=0)
            
            self.statusBar().showMessage("Temporary files cleared.", 3000)

    # --- Media Player ---
    def on_play_pause(self):
        if not self.media_player:
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.btn_play.setText("▶")
        else:
            self.media_player.play()
            self.btn_play.setText("⏸")

    def stop_audio(self):
        if not self.media_player: 
            return
        self.media_player.stop()
        self.btn_play.setText("▶")
        self.waveform_widget.set_playback_progress(0)

    def play_audio_file(self, filepath):
        self.stop_audio()
        if not os.path.exists(filepath): 
            return
        
        # Защита, ако плеърът не е инициализиран
        if not self.media_player: 
            return

        self.current_filepath = filepath
        self.media_player.setSource(QUrl.fromLocalFile(filepath))
        
        # [FIX] Викаме метода през работника, за да не дублираме код
        if self.synthesis_worker:
            wf = self.synthesis_worker._load_waveform_data(filepath) 
            self.waveform_widget.set_waveform(wf)
        
        self.media_player.play()
        self.btn_play.setText("⏸")

    @Slot(str, np.ndarray)
    def on_file_ready_for_playback(self, filepath, waveform):
        # [FIX] Защита, ако плеърът не е инициализиран
        if not self.media_player: 
            return

        self.current_filepath = filepath
        self.media_player.setSource(QUrl.fromLocalFile(filepath))
        
        # Тук виждам, че се опитваш да ползваш self._load_waveform_data
        # Но този метод май липсва в класа (виж стъпка 3 долу)
        if hasattr(self, 'waveform_widget'):
             self.waveform_widget.set_waveform(waveform)
             
        self.media_player.play()
        self.btn_play.setText("⏸")


    @Slot(int)
    def seek_audio(self, pos_ms):
        if self.media_player and self.media_player.isSeekable():
            self.media_player.setPosition(pos_ms)

    @Slot(int)
    def on_player_position_changed(self, pos):
        if self.stored_duration > 0:
            self.waveform_widget.set_playback_progress(pos / self.stored_duration)
            
            # --- SMART TIME FORMATTING ---
            ch, cm, cs = self._format_time(pos)
            th, tm, ts = self._format_time(self.stored_duration)
            if th > 0:
                cur_str = f"{ch}:{cm:02}:{cs:02}"
                tot_str = f"{th}:{tm:02}:{ts:02}"
            else:
                cur_str = f"{cm:02}:{cs:02}"
                tot_str = f"{tm:02}:{ts:02}"
                
            self.lbl_time.setText(f"{cur_str} / {tot_str}")

    @Slot(int)
    def on_player_duration_changed(self, dur):
        if dur > 0: 
            self.stored_duration = dur
            self.waveform_widget.set_audio_duration(dur)

    @Slot(object)
    def on_player_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.btn_play.setText("▶")
            self.waveform_widget.set_playback_progress(0)

    # --- Helpers ---
    def refresh_voice_list(self):
        """Populates voice combos and restores selection from Config."""
        voices = models.list_available_voices()
        
        curr_v1 = self.voice_combo_1.currentText()
        curr_v2 = self.voice_combo_2.currentText()

        self.voice_combo_1.clear()
        self.voice_combo_1.addItems(voices)
        
        self.voice_combo_2.clear()
        self.voice_combo_2.addItem("None (Single Voice)")
        self.voice_combo_2.addItems(voices)
        self.voice_combo_2.setCurrentIndex(0) # Default

        # 3. (SMART RESTORE)
        
        if curr_v1 and curr_v1 in voices:
            target_voice = curr_v1
        else:
            config_voice = self.config.get('tts_engine', {}).get('voice', '')
            
            if config_voice in voices:
                target_voice = config_voice
            else:
                friendly = models.get_friendly_voice_name(config_voice)
                if friendly and friendly in voices:
                    target_voice = friendly
                else:
                    target_voice = voices[0] if voices else ""

        if target_voice:
            self.voice_combo_1.setCurrentText(target_voice)
        if curr_v2:
            if curr_v2 == "None (Single Voice)" or curr_v2 in voices:
                self.voice_combo_2.setCurrentText(curr_v2)

    def list_available_voices(self):
        return models.list_available_voices()

    def update_progress(self, curr, total):
        """Updates UI during synthesis."""
        if curr == total:
            self.statusBar().showMessage(f"Synthesis finished ({total}/{total}). Merging & Encoding MP3... Please wait!")
            self.setWindowTitle("Kokoro Studio - Finalizing...")
        else:
            self.statusBar().showMessage(f"Processing segment {curr}/{total}...")
            self.setWindowTitle(f"Kokoro Studio - {int((curr/total)*100)}%")

        # 2. AUTO-SCROLL
        if self.tabs.currentIndex() == 1:
            row = curr - 1 
            if 0 <= row < self.book_table.rowCount():
                self.book_table.selectRow(row)
                item = self.book_table.item(row, 0)
                if item:
                    self.book_table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def on_synthesis_finished(self, result):
        self.setWindowTitle("Kokoro Studio v2.0")
        self.statusBar().showMessage("Ready.")
        if result:
            chunks_raw, combined = result
            if combined:
                # [CRITICAL FIX] Convert result Tuples to Dicts for JSON saving
                clean_chunks = []
                for c in chunks_raw:
                    # c is (graphemes, phonemes, numpy_array, filepath)
                    clean_chunks.append({
                        "graphemes": c[0],
                        "phonemes": c[1],
                        "filepath": c[3] # Skip c[2] (numpy)
                    })
                
                self.synthesis_results.append({
                    "timestamp": time.time(),
                    "combined": combined,
                    "text_source": "segmented" if self.tabs.currentIndex() == 1 else "scratch",
                    "chunks": clean_chunks
                })
                persistence.save_generations(PERSIST_FILE, self.synthesis_results)
                self.populate_history_table()
                
        self._reset_render_button()

    def on_synthesis_error(self, msg):
        self.statusBar().showMessage(f"Error: {msg}")
        error_handler.show_error(self, msg)
        self._reset_render_button()
        
    def dragEnterEvent(self, event):
        """Accept dragging files into the window."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle dropping a file."""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in files:
            if f.lower().endswith(".txt") or f.lower().endswith(".epub"):
                self.load_text_to_table(f)
                break
            elif f.lower().endswith(".kproj") or f.lower().endswith(".json"):
                self.load_project(f)
                break

    def _format_time(self, ms):
        """Converts milliseconds to MM:SS or H:MM:SS."""
        seconds = max(0, ms // 1000)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return h, m, s


    def _load_config(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: 
                return yaml.safe_load(f)
        return {}

    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #282c34; color: #abb2bf; font-family: Segoe UI, sans-serif; }
            QWidget#Sidebar { background-color: #21252b; border-right: 1px solid #181a1f; }
            QWidget#Footer { background-color: #21252b; border-top: 1px solid #181a1f; }
            QGroupBox { border: 1px solid #3e4451; margin-top: 1.5em; border-radius: 4px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #61afef; }
            QPushButton { background-color: #3e4451; border: none; padding: 6px; border-radius: 4px; color: white; }
            QPushButton:hover { background-color: #4b5263; }
            QPushButton:checked { background-color: #98c379; color: black; }
            QTableWidget { background-color: #282c34; gridline-color: #3e4451; color: #abb2bf; border: none; }
            QHeaderView::section { background-color: #21252b; padding: 4px; border: none; color: #9da5b4; font-weight: bold; }
            QTextEdit { background-color: #21252b; border: 1px solid #3e4451; color: #abb2bf; padding: 5px; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #21252b; border: 1px solid #3e4451; padding: 4px; color: #abb2bf; border-radius: 3px; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #61afef; margin-right: 5px; }
            QComboBox:on { border: 1px solid #61afef; }
            QSlider::groove:horizontal { height: 4px; background: #3e4451; border-radius: 2px; }
            QSlider::handle:horizontal { background: #61afef; width: 14px; margin: -5px 0; border-radius: 7px; }
            QTabWidget::pane { border: 1px solid #3e4451; }
            QTabBar::tab { background: #21252b; padding: 8px 20px; color: #abb2bf; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #282c34; border-top: 2px solid #61afef; color: white; }
            QStatusBar { background: #21252b; color: #9da5b4; }
        """)
    
    def save_audio_dialog(self, default_name, source_path):
        if not os.path.exists(source_path):
            error_handler.show_error(self, "Source file not found.")
            return
        fmt = self.save_format_combo.currentText().lower()
        fname = f"{default_name}.{fmt}"
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Audio", fname, f"{fmt.upper()} Files (*.{fmt})")
        if save_path:
            try:
                if source_path.lower().endswith(f".{fmt}"):
                    shutil.copy2(source_path, save_path)
                else:
                    
                    audio = AudioSegment.from_file(source_path)
                    audio.export(save_path, format=fmt)
                self.statusBar().showMessage(f"Saved: {os.path.basename(save_path)}", 3000)
            except Exception as e:
                error_handler.show_error(self, f"Save failed: {e}")
    
    def closeEvent(self, event):
        """Handle clean shutdown and save settings."""
        logger.info("Shutting down...")

        # 1. Initialisation
        if 'tts_engine' not in self.config: 
            self.config['tts_engine'] = {}
        if 'tts_params' not in self.config: 
            self.config['tts_params'] = {}

        # 2. Saving the UI
        # Voice
        self.config['tts_engine']['voice'] = self.voice_combo_1.currentText()
        
        # Params
        self.config['tts_params']['speed_default'] = self.speed_spin.value()
        self.config['tts_params']['seed'] = self.seed_spin.value()
        self.config['tts_params']['sample_rate'] = self.sample_rate_combo.currentIndex()
        self.config['tts_params']['save_format'] = self.save_format_combo.currentText() 

        # 3. Saving in the config file
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info("Configuration saved.")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

        # 4. Stops the Audio Player
        if self.media_player:
            self.media_player.stop()
            self.media_player.setSource(QUrl())
            
        if hasattr(self, 'synthesis_thread') and self.synthesis_thread.isRunning():
            self.synthesis_thread.quit()
            self.synthesis_thread.wait(500)
        
        event.accept()

# ----------------------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)
    win = MyTTSMainWindow()
    win.show()
    sys.exit(app.exec())