# Kokoro Studio v2.0 (Local GPU TTS)

**Kokoro Studio** is a professional-grade, local Text-to-Speech application powered by the **Kokoro-82M** model.

It creates high-quality audio instantly, runs entirely offline, and supports GPU acceleration for blazing-fast synthesis.

> **v2.0 Major Update:** Complete UI overhaul, Audiobook mode, Voice Mixing, Project Saving, and EPUB support!

## 🚀 Key Features

* **⚡ Hyper-Fast Inference:** Generates audio in real-time (or faster) using **NVIDIA GPU (CUDA)**. Falls back to CPU if no GPU is found.
* **📚 Audiobook Mode:** Load **TXT** or **EPUB** books. The app splits them into segments for easy management.
* **🎛️ Voice Mixing:** Combine two voices (e.g., *Alice + George*) to create unique character blends.
* **💾 Project System:** Save your work (`.kproj`). Keep your voice assignments and text segments to continue working later.
* **🛠️ Fine Control:** Adjust **Speed**, **Pitch**, and **Sample Rate** (24kHz, 16kHz, etc.) per project or globally.
* **🖱️ Drag & Drop:** Drop text files or project files directly into the window to load them.
* **🎵 Integrated Player:** Play specific segments, preview audio, or listen to the full combined track with a visual waveform.

## 📦 Installation (Windows)

We have simplified the installation process using `uv` for speed and reliability.

### Option 1: The "One-Click" Method (Recommended)

1.**Clone or Download** this repository.
2.Double-click **`run.bat`**.

* This script will automatically set up a Python environment.
* It will detect your NVIDIA driver version.
* It will install the correct version of **PyTorch (CUDA)** for your hardware.
* It will launch the application.

### Option 2: Manual Installation

If you prefer managing your own environment:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install basic dependencies
pip install -r requirements.txt

# 3. (Important) Install GPU-Accelerated PyTorch
# Run our helper script to fetch the correct CUDA version for your driver:
python install_torch_uv.py

# 4. Run the app
python main.py
```

## 📖 Usage

The interface is divided into two main tabs:

### 1. Scratchpad (Quick Mode)

* Ideal for testing voices or synthesizing short text snippets.
* Type text, select a voice from the sidebar, and click **Synthesize**.

### 2. Audiobook Mode (Batch Processing)

* **Load File:** Drag & Drop a `.txt` or `.epub` file.
* **Table View:** The text is split into segments/lines.
* **Per-Line Control:** Assign different voices to different lines (great for dialogue).
* **Preview:** Click the **▶** button on any row to hear just that sentence.
* **Render:** Click **Render Audiobook** to generate and merge all lines into one MP3/WAV file.

### Sidebar Controls

* **Primary Voice:** The main speaker.
* **Mix Voice:** Check this to blend a secondary voice (50/50 mix).
* **Audio Props:** Change Speed, Pitch, and Target Hz (Sample Rate).
* **System:** View your current device (GPU/CPU) and set a Seed for reproducibility.

## 📂 File Structure

* `main.py`: Entry point.
* `ui_main.py`: The graphical interface (PySide6).
* `tts_wrapper.py`: Connects the UI to the Kokoro model (handling inference, mixing, resampling).
* `models.py`: Definitions of available voices.
* `persistence.py`: Handles saving/loading projects and history.
* `install_torch_uv.py`: Helper script to auto-detect and install CUDA support.
* `run.bat`: Windows launcher script.

## ⚠️ Troubleshooting

* **"Device: CPU (Slow)"**: If the app shows this label in yellow, it means CUDA is not detected. Run `run.bat` again to force a PyTorch reinstall, or ensure you have NVIDIA drivers installed.
* **Voice Download Error**: On the first run, the app downloads model weights (~300MB) from HuggingFace. Ensure you have an internet connection.
* **MP3 Issues**: If saving as MP3 fails, ensure `ffmpeg` is installed on your system (though `pydub` often handles this seamlessly).

## License

Distributed under the MIT License.

---
*Based on the amazing [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model.*
