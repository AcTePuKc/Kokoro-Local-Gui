# Kokoro Studio v2.0.2 (Local GPU TTS)

**Kokoro Studio** is a professional-grade, local Text-to-Speech application powered by the **Kokoro-82M** model.

It creates high-quality audio instantly, runs entirely offline, and supports GPU acceleration for blazing-fast synthesis.

> **v2.0 Major Update:** Complete UI overhaul, Audiobook mode, Voice Mixing, Project Saving, and EPUB support!

> **v2.0.1 Patch:** Improved CUDA/PyTorch detection, safer CPU fallback, cleaner reinstall flow, and startup crash fixes.
> **v2.0.2 Patch:** Improved synthesis progress reporting, reduced WAV memory usage, optimized waveform rendering, and restored per-segment history actions.

## 🚀 Key Features

* **⚡ GPU-Accelerated Inference:** Uses **NVIDIA CUDA** when available for faster synthesis and automatically falls back to CPU when necessary.
* **📚 Audiobook Mode:** Load **TXT** or **EPUB** books. The app splits them into segments for easy management.
* **🎛️ Voice Mixing:** Combine two voices (e.g., *Alice + George*) to create unique character blends.
* **💾 Project System:** Save your work (`.kproj`). Keep your voice assignments and text segments to continue working later.
* **🛠️ Fine Control:** Adjust **Speed**, **Pitch**, and **Sample Rate** (24kHz, 16kHz, etc.) per project or globally.
* **🖱️ Drag & Drop:** Drop text files or project files directly into the window to load them.
* **🎵 Integrated Player and History:** Preview individual segments, play completed generations, save selected audio, copy segment text, and view a lightweight waveform.
* **📊 Detailed Progress Tracking:** Follow the current segment, generated chunk count, and synthesis progress while longer jobs are running.

## 📦 Installation (Windows)

We have simplified the installation process using `uv` for speed and reliability.

### Option 1: The "One-Click" Method (Recommended)

1. **Clone or Download** this repository.
2. Double-click **`run.bat`**.

* This script will automatically set up a Python environment.
* It will detect your NVIDIA driver and supported CUDA configuration.
* It will install a compatible **PyTorch CUDA** build when available.
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

```bat
# 1. Create and activate a virtual environment
uv venv .venv --python 3.11
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
* **Progress Display:** During synthesis, the interface shows the active segment, generated chunk count, and overall progress.

### Sidebar Controls

* **Primary Voice:** The main speaker.
* **Mix Voice:** Check this to blend a secondary voice (50/50 mix).
* **Audio Properties:** Adjust Speed, Pitch, and Target Sample Rate.
* **System:** View the active processing device (GPU or CPU) and set a seed for repeatable generation.

## 📂 File Structure

* `main.py`: Entry point.
* `ui_main.py`: The graphical interface (PySide6).
* `tts_wrapper.py`: Connects the UI to the Kokoro model and handles inference, voice mixing, resampling, chunk output, and combined audio generation.
* `models.py`: Definitions of available voices.
* `persistence.py`: Handles saving/loading projects and history.
* `install_torch_uv.py`: Helper script to auto-detect and install CUDA support.
* `run.bat`: Windows launcher script.
* `tests/`: Regression tests for synthesis metadata, progress handling, waveform errors, and History controls.

## ⚠️ Troubleshooting

* **"Device: CPU (Slow)"**: If the app shows this label in yellow, CUDA is either not available or the installed PyTorch CUDA build is not compatible with your GPU. Run `run.bat reinstall` to force a PyTorch reinstall, or rebuild from a clean `.venv`.
* **New NVIDIA GPUs / unsupported CUDA wheel**: Some very new cards may temporarily fail on GPU if the installed PyTorch build does not include kernels for that architecture yet. Kokoro Studio now falls back to CPU instead of crashing, so the app should still start.
* **Voice Download Error**: On the first run, the app downloads model weights (~300 MB) from Hugging Face. Ensure you have an internet connection.
* **MP3 Issues**: If MP3 export or playback fails, ensure that `ffmpeg` is installed and available in your system `PATH`.
* **Long MP3 generations**: MP3 chunk merging currently creates one combined `AudioSegment` in memory. For long audiobooks, WAV output generally uses less peak memory because audio is written incrementally during synthesis.

## Contributors

Thanks to everyone who helped build and improve Kokoro Studio:

| Contributor | Contribution |
| --- | --- |
| Shteryan Nikolaev | Project maintainer and core development |
| WilleIshere | Community code contributions |
| dsovven | CUDA/PyTorch startup improvements |
| syedusama5556 | Synthesis progress, memory usage, waveform, and history UI improvements |

## License

Distributed under the Apache License 2.0. See `LICENSE` for more information.

---

*Based on the amazing [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model, also distributed under the Apache License 2.0.*
