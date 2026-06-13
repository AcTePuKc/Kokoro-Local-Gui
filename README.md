# Kokoro Studio v2.0.1 (Local GPU TTS)

**Kokoro Studio** is a professional-grade, local Text-to-Speech application powered by the **Kokoro-82M** model.

It creates high-quality audio instantly, runs entirely offline, and supports GPU acceleration for blazing-fast synthesis.

> **v2.0 Major Update:** Complete UI overhaul, Audiobook mode, Voice Mixing, Project Saving, and EPUB support!

> **v2.0.1 Patch:** Improved CUDA/PyTorch detection, safer CPU fallback, cleaner reinstall flow, and startup crash fixes.

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
* If PyTorch is already installed correctly, it will skip reinstalling it.

### Clean Reinstall

If your local environment gets into a bad state, close the app and use one of these:

```bat
run.bat reinstall
```

Or remove the local virtual environment and let the launcher rebuild it:

```bat
rmdir /s /q .venv
run.bat
```

### Option 2: Manual Installation

If you prefer managing your own environment:

```bash
# 1. Create and activate a virtual environment
uv venv .venv --python python3.11
.venv\Scripts\activate

# 2. Install basic dependencies
uv pip install -r requirements.txt

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

* **"Device: CPU (Slow)"**: If the app shows this label in yellow, CUDA is either not available or the installed PyTorch CUDA build is not compatible with your GPU. Run `run.bat reinstall` to force a PyTorch reinstall, or rebuild from a clean `.venv`.
* **New NVIDIA GPUs / unsupported CUDA wheel**: Some very new cards may temporarily fail on GPU if the installed PyTorch build does not include kernels for that architecture yet. Kokoro Studio now falls back to CPU instead of crashing, so the app should still start.
* **Voice Download Error**: On the first run, the app downloads model weights (~300MB) from HuggingFace. Ensure you have an internet connection.
* **MP3 Issues**: If saving as MP3 fails, ensure `ffmpeg` is installed on your system (though `pydub` often handles this seamlessly).

## Contributors

Thanks to everyone who helped build and improve Kokoro Studio:

| Contributor | Contribution |
| --- | --- |
| Shteryan Nikolaev | Project maintainer and core development |
| WilleIshere | Community code contributions |
| dsovven | CUDA/PyTorch startup improvements |

## License

Distributed under the MIT License.

---
*Based on the amazing [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model.*
