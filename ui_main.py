import os
import sys
import time
import yaml
import shutil
import logging
import threading

from functools import partial

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QDoubleSpinBox, QGroupBox, QComboBox,
    QSlider, QAbstractItemView, QMessageBox
)
from PySide6.QtCore import (
    Qt, QUrl, QMetaObject, Q_ARG, Signal, Slot
)
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

import numpy as np
import models
from tts_wrapper import KokoroTTSWrapper
import persistence
import temp_cleanup
import error_handler

# Persistent generations file path:
PERSIST_FILE = os.path.join("outputs", "generations.json")

# --------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------
log_file = 'gui.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)

# --------------------------------------------------------------------
# Waveform Widget
# --------------------------------------------------------------------
class WaveformWidget(QWidget):
    @Slot(str)
    def set_file(self, filepath: str):
        self.audio_filepath = filepath
        self.load_audio(filepath)
        self.update()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_filepath = None
        self.audio_data = np.array([])
        self.sample_rate = 22050
        self.last_loaded_filepath = None

    def load_audio(self, filepath):
        import wave
        if not filepath or not os.path.exists(filepath):
            return
        if filepath == self.last_loaded_filepath:
            return
        try:
            with wave.open(filepath, "rb") as wav_file:
                self.sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                if num_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                peak = np.max(np.abs(audio_data))
                self.audio_data = audio_data / peak if peak != 0 else audio_data
            self.last_loaded_filepath = filepath
        except Exception as e:
            logging.exception(f"WaveformWidget error: {e}")

    def paintEvent(self, event):
        if self.audio_data.size == 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(0, 255, 255))
        pen.setWidth(1)
        painter.setPen(pen)

        rect = self.contentsRect()
        width, height = rect.width(), rect.height()
        max_amplitude = np.max(np.abs(self.audio_data))
        if max_amplitude == 0:
            return

        scale_factor = (height / 2) / max_amplitude
        scaled_data = self.audio_data * scale_factor
        x_scale = width / len(scaled_data) if len(scaled_data) else width
        mid_y = height / 2
        for i in range(len(scaled_data) - 1):
            x1, y1 = i * x_scale, mid_y - scaled_data[i]
            x2, y2 = (i + 1) * x_scale, mid_y - scaled_data[i + 1]
            painter.drawLine(x1, y1, x2, y2)


# --------------------------------------------------------------------
# Main Window
# --------------------------------------------------------------------
class MyTTSMainWindow(QMainWindow):
    synthesis_finished_signal = Signal()

    def __init__(self, config_path="config.yaml"):
        super().__init__()
        self.setWindowTitle("Kokoro Local TTS UI")

        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.synthesis_results = persistence.load_generations(PERSIST_FILE)
        self.current_filepath = None

        # Main widget & layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.statusBar().showMessage("Ready", 1000)

        # Row 1: Parameters + Playback side by side
        row1 = QWidget()
        row1_layout = QHBoxLayout(row1)

        # Left: Parameters
        parameters_group = QGroupBox("Parameters")
        param_layout = QFormLayout(parameters_group)
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(self.list_available_voices())
        param_layout.addRow("Voice:", self.voice_combo)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 2.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        param_layout.addRow("Speed:", self.speed_spin)

        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["WAV", "MP3"])
        param_layout.addRow("Save Format:", self.save_format_combo)
        row1_layout.addWidget(parameters_group, 1)

        # Right: Playback
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout(playback_group)

        self.waveform_widget = WaveformWidget()
        self.waveform_widget.setMinimumHeight(50)  # smaller waveform
        playback_layout.addWidget(self.waveform_widget)

        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.on_main_play_clicked)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_audio)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.stop_button)

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.progress_slider, 2)

        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)

        controls_layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.set_volume)
        controls_layout.addWidget(self.volume_slider)

        playback_layout.addWidget(controls_widget)
        row1_layout.addWidget(playback_group, 2)

        main_layout.addWidget(row1)

        # Row 2: Text to Synthesize + Synthesize button on the same row
        row2 = QGroupBox("Text to Synthesize")
        row2_layout = QHBoxLayout(row2)

        self.text_edit = QTextEdit()
        self.text_edit.setMinimumHeight(150)
        row2_layout.addWidget(self.text_edit, 4)

        self.synth_button = QPushButton("Synthesize")
        self.synth_button.clicked.connect(self.on_synthesize_clicked)
        row2_layout.addWidget(self.synth_button, 1)

        main_layout.addWidget(row2)

        # Row 3: Results Table
        results_group = QGroupBox("Previous Generations")
        results_layout = QVBoxLayout(results_group)
        self.generation_label = QLabel("Last Generation Time: N/A")
        results_layout.addWidget(self.generation_label)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Chunk", "Graphemes", "Phonemes", "Play", "Save"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        results_layout.addWidget(self.results_table)

        clear_layout = QHBoxLayout()
        self.clear_results_button = QPushButton("Clear Old Results")
        self.clear_results_button.clicked.connect(self.clear_old_results)
        clear_layout.addWidget(self.clear_results_button)

        self.clear_temp_button = QPushButton("Clear Temp Files")
        self.clear_temp_button.clicked.connect(self.clear_temp_files)
        clear_layout.addWidget(self.clear_temp_button)

        results_layout.addLayout(clear_layout)
        main_layout.addWidget(results_group)

        # TTS & MEDIA SETUP
        output_dir = os.path.join("outputs")
        os.makedirs(output_dir, exist_ok=True)
        self.tts_wrapper = KokoroTTSWrapper(
            output_dir=output_dir,
            config=self.config,
            temp_dir=os.path.join(output_dir, "temp_audio")
        )
        self.tts_wrapper.load_default_voice()

        self.audio_output = QAudioOutput()
        self.media_player = QMediaPlayer()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.stored_duration = 0

        self.synthesis_finished_signal.connect(self.populate_results_table)

        # Cleanup older chunk_ files
        temp_cleanup.cleanup_temp_files(self.tts_wrapper.temp_dir, retention_days=7)

        # Populate the table from persistent data
        self.populate_results_table()

    @Slot(str, int)
    def update_status_message(self, msg, timeout):
        self.statusBar().showMessage(msg, timeout)

    def closeEvent(self, event):
        self.statusBar().showMessage("Cleaning up temporary files...", 2000)
        temp_cleanup.cleanup_temp_files(self.tts_wrapper.temp_dir, retention_days=7)
        self.statusBar().showMessage("Goodbye!", 1000)
        event.accept()

    def clear_temp_files(self):
        temp_dir = self.tts_wrapper.temp_dir
        for filename in os.listdir(temp_dir):
            if filename.startswith("chunk_"):
                filepath = os.path.join(temp_dir, filename)
                try:
                    os.remove(filepath)
                except Exception as e:
                    logging.exception(f"Error removing temp file {filepath}: {e}")
        for gen in self.synthesis_results:
            gen["chunks"] = []
        self.save_generations()
        self.populate_results_table()
        QMetaObject.invokeMethod(self, "update_status_message", Qt.QueuedConnection,
                                 Q_ARG(str, "Temporary files cleared."), Q_ARG(int, 2000))

    def load_config(self, path):
        if not os.path.exists(path):
            return {
                'tts_params': {'speed_default': 1.0},
                'tts_engine': {'lang_code': 'a', 'voice': 'af_bella', 'output_dir': 'outputs'}
            }
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def list_available_voices():
        return models.list_available_voices()

    def save_generations(self):
        persistence.save_generations(PERSIST_FILE, self.synthesis_results)

    def on_synthesize_clicked(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            return
        self.synth_button.setEnabled(False)
        QMetaObject.invokeMethod(self, "update_status_message", Qt.QueuedConnection,
                                 Q_ARG(str, "Synthesizing..."), Q_ARG(int, 0))
        voice = self.voice_combo.currentText()
        if not voice.strip():
            voice = "af_bella"
        speed = float(self.speed_spin.value())

        def run_synthesis():
            start_time = time.time()
            try:
                results = self.tts_wrapper.synthesize(
                    text=text,
                    speed=speed,
                    selected_voice=voice
                )
                if results:
                    chunk_list, combined_path = results
                    gen = {
                        "timestamp": time.time(),
                        "chunks": [],
                        "combined": combined_path if combined_path and os.path.exists(combined_path) else ""
                    }
                    for chunk in chunk_list:
                        gen["chunks"].append({
                            "graphemes": chunk[0],
                            "phonemes": chunk[1],
                            "filepath": chunk[3]
                        })
                    self.synthesis_results.append(gen)
                    self.save_generations()
                elapsed = time.time() - start_time
                QMetaObject.invokeMethod(self, "update_generation_time", Qt.QueuedConnection,
                                         Q_ARG(float, elapsed))
                if results and results[1]:
                    QMetaObject.invokeMethod(self.waveform_widget, "set_file", Qt.QueuedConnection,
                                             Q_ARG(str, results[1]))
                    self.current_filepath = results[1]
                self.synthesis_finished_signal.emit()
                QMetaObject.invokeMethod(self, "update_status_message", Qt.QueuedConnection,
                                         Q_ARG(str, "Synthesis complete."), Q_ARG(int, 2000))
            except Exception as e:
                logging.exception("Synthesis error:")
                QMetaObject.invokeMethod(self, "update_status_message", Qt.QueuedConnection,
                                         Q_ARG(str, "Synthesis error."), Q_ARG(int, 2000))
                error_handler.show_error(self, f"Synthesis error: {e}")
            finally:
                QMetaObject.invokeMethod(self.synth_button, "setEnabled", Qt.QueuedConnection,
                                         Q_ARG(bool, True))
        threading.Thread(target=run_synthesis, daemon=True).start()

    @Slot()
    def populate_results_table(self):
        self.results_table.setRowCount(0)
        for gen_idx, gen in enumerate(self.synthesis_results, start=1):
            # Show chunk rows
            if gen.get("chunks"):
                for i, chunk in enumerate(gen.get("chunks", [])):
                    row = self.results_table.rowCount()
                    self.results_table.insertRow(row)
                    item_id = QTableWidgetItem(f"{gen_idx}-{i}")
                    self.results_table.setItem(row, 0, item_id)
                    self.results_table.setItem(row, 1, QTableWidgetItem(chunk.get("graphemes", "")))
                    self.results_table.setItem(row, 2, QTableWidgetItem(chunk.get("phonemes", "")))
                    btn_play = QPushButton("Play Chunk")
                    btn_play.clicked.connect(partial(self.play_audio, chunk.get("filepath", "")))
                    self.results_table.setCellWidget(row, 3, btn_play)
                    btn_save = QPushButton("Save")
                    btn_save.clicked.connect(partial(self.save_audio, f"{gen_idx}-{i}", chunk.get("filepath", "")))
                    self.results_table.setCellWidget(row, 4, btn_save)

            # Show combined row
            combined = gen.get("combined", "")
            if combined:
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem(f"{gen_idx}-Combined"))
                self.results_table.setItem(row, 1, QTableWidgetItem("N/A"))
                self.results_table.setItem(row, 2, QTableWidgetItem("N/A"))
                btn_play_combined = QPushButton("Play Combined")
                btn_play_combined.clicked.connect(partial(self.play_audio, combined))
                self.results_table.setCellWidget(row, 3, btn_play_combined)
                btn_save_combined = QPushButton("Save Combined")
                btn_save_combined.clicked.connect(partial(self.save_audio, f"{gen_idx}-combined", combined))
                self.results_table.setCellWidget(row, 4, btn_save_combined)

        # Scroll to last row so you see the newest generation
        row_count = self.results_table.rowCount()
        if row_count > 0:
            last_item = self.results_table.item(row_count - 1, 0)
            if last_item:
                self.results_table.scrollToItem(last_item)

    @Slot(float)
    def update_generation_time(self, elapsed):
        self.generation_label.setText(f"Last Generation Time: {elapsed:.2f} sec")

    def clear_old_results(self):
        self.results_table.setRowCount(0)
        self.synthesis_results.clear()
        self.save_generations()
        self.generation_label.setText("Last Generation Time: N/A")
        QMetaObject.invokeMethod(self, "update_status_message", Qt.QueuedConnection,
                                 Q_ARG(str, "Persistent generations cleared."), Q_ARG(int, 2000))

    # MEDIA PLAYER LOGIC
    def on_main_play_clicked(self):
        state = self.media_player.playbackState()
        if state == QMediaPlayer.PlayingState:
            logging.info("Already playing.")
            return
        if state == QMediaPlayer.PausedState:
            logging.info("Resuming from paused.")
            self.media_player.play()
            return
        if self.current_filepath:
            logging.info(f"Playing last file from start: {self.current_filepath}")
            self.media_player.stop()
            self.media_player.setSource(QUrl.fromLocalFile(self.current_filepath))
            self.media_player.play()
        else:
            logging.warning("No file to play.")

    def play_audio(self, filepath=None):
        if not filepath or not os.path.exists(filepath):
            logging.warning("No valid audio file to play.")
            return
        self.media_player.stop()
        self.media_player.setSource(QUrl.fromLocalFile(filepath))
        self.media_player.play()
        self.current_filepath = filepath
        self.waveform_widget.set_file(filepath)

    def pause_audio(self):
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
            logging.info("Audio paused.")

    def stop_audio(self):
        self.media_player.stop()
        self.media_player.setPosition(0)
        logging.info("Audio stopped (source retained).")

    @Slot(int)
    def on_duration_changed(self, duration_ms):
        if duration_ms > 0:
            self.stored_duration = duration_ms
            self.progress_slider.setEnabled(True)
            self.progress_slider.setValue(0)
        else:
            self.stored_duration = 0
            self.progress_slider.setEnabled(False)
            self.progress_slider.setValue(0)

    @Slot(int)
    def on_position_changed(self, position_ms):
        if self.stored_duration > 0:
            total_sec = self.stored_duration // 1000
            current_sec = position_ms // 1000
            total_str = f"{total_sec // 60:02}:{total_sec % 60:02}"
            current_str = f"{current_sec // 60:02}:{current_sec % 60:02}"
            self.time_label.setText(f"{current_str} / {total_str}")
            percent = int((position_ms / self.stored_duration) * 100)
            self.progress_slider.blockSignals(True)
            self.progress_slider.setValue(percent)
            self.progress_slider.blockSignals(False)
        else:
            self.time_label.setText("00:00 / 00:00")
            self.progress_slider.setValue(0)

    def set_position(self, slider_val):
        if self.stored_duration > 0:
            new_pos = int((slider_val / 100) * self.stored_duration)
            self.media_player.setPosition(new_pos)

    def set_volume(self, val):
        self.audio_output.setVolume(val / 100.0)

    def save_audio(self, index, audio_data_or_path):
        logging.info(f"save_audio index={index}")
        default_filename = f"output_{index}"
        selected_format = self.save_format_combo.currentText()
        if selected_format == "MP3":
            default_filename += ".mp3"
            file_filter = "MP3 Files (*.mp3)"
        else:
            default_filename += ".wav"
            file_filter = "WAV Files (*.wav)"
        default_path = os.path.join(self.tts_wrapper.output_dir, default_filename)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Audio", default_path, file_filter)
        if not save_path:
            return
        if isinstance(audio_data_or_path, str):
            shutil.copy(audio_data_or_path, save_path)
            logging.info(f"Saved file copy to {save_path}")
        else:
            saved_filepath = self.tts_wrapper.save_audio(
                audio_data_or_path,
                save_path,
                format=selected_format
            )
            if saved_filepath:
                logging.info(f"Audio saved to {saved_filepath}")
            else:
                logging.error("Failed to save audio.")

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import subprocess

    app = QApplication(sys.argv)

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--upgrade", "--force-reinstall", "PySide6"
        ])
    except subprocess.CalledProcessError:
        print("Failed to reinstall PySide6")

    window = MyTTSMainWindow()
    window.show()
    sys.exit(app.exec())
