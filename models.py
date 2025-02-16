"""Models module for Kokoro TTS Local EDITED VERSION FOR GUI"""
from typing import Optional, Tuple, List
import torch
from kokoro import KPipeline
import os
import json
import codecs
from pathlib import Path
import numpy as np
import shutil

# Set environment variables for proper encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# List of available voice files (can stay the same)
VOICE_FILES = [
    # American Female voices
    "af_alloy.pt", "af_aoede.pt", "af_bella.pt", "af_jessica.pt",
    "af_kore.pt", "af_nicole.pt", "af_nova.pt", "af_river.pt",
    "af_sarah.pt", "af_sky.pt",
    # American Male voices
    "am_adam.pt", "am_echo.pt", "am_eric.pt", "am_fenrir.pt",
    "am_liam.pt", "am_michael.pt", "am_onyx.pt", "am_puck.pt",
    "am_santa.pt",
    # British Female voices
    "bf_alice.pt", "bf_emma.pt", "bf_isabella.pt", "bf_lily.pt",
    # British Male voices
    "bm_daniel.pt", "bm_fable.pt", "bm_george.pt", "bm_lewis.pt",
    # Special voices
    "el_dora.pt", "em_alex.pt", "em_santa.pt",
    "ff_siwis.pt",
    "hf_alpha.pt", "hf_beta.pt",
    "hm_omega.pt", "hm_psi.pt",
    "jf_sara.pt", "jm_nicola.pt",
    "jf_alpha.pt", "jf_gongtsuene.pt", "jf_nezumi.pt", "jf_tebukuro.pt",
    "jm_kumo.pt",
    "pf_dora.pt", "pm_alex.pt", "pm_santa.pt",
    "zf_xiaobei.pt", "zf_xiaoni.pt", "zf_xiaoqiao.pt", "zf_xiaoyi.pt"
]

# Patch KPipeline's load_voice method to use weights_only=False
original_load_voice = KPipeline.load_voice

def patched_load_voice(self, voice_path):
    """Load voice model with weights_only=False for compatibility"""
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    voice_name = Path(voice_path).stem
    voice_model = torch.load(voice_path, weights_only=False)
    if voice_model is None:
        raise ValueError(f"Failed to load voice model from {voice_path}")
    # Ensure device is set
    if not hasattr(self, 'device'):
        self.device = 'cpu'
    # Move model to device and store in voices dictionary
    self.voices[voice_name] = voice_model.to(self.device)
    return self.voices[voice_name]

KPipeline.load_voice = patched_load_voice

def patch_json_load():
    """Patch json.load to handle UTF-8 encoded files with special characters"""
    original_load = json.load

    def custom_load(fp, *args, **kwargs):
        try:
            # Try reading with UTF-8 encoding
            if hasattr(fp, 'buffer'):
                content = fp.buffer.read().decode('utf-8')
            else:
                content = fp.read()
            return json.loads(content)
        except UnicodeDecodeError:
            # If UTF-8 fails, try with utf-8-sig for files with BOM
            fp.seek(0)
            content = fp.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8-sig', errors='replace')
            return json.loads(content)

    json.load = custom_load

def load_config(config_path: str) -> dict:
    """Load configuration file with proper encoding handling"""
    try:
        with codecs.open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fallback to utf-8-sig if regular utf-8 fails
        with codecs.open(config_path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)

# Initialize espeak-ng (Keep this; it's needed for phonemization)
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    from phonemizer import phonemize
    import espeakng_loader

    library_path = espeakng_loader.get_library_path()
    data_path = espeakng_loader.get_data_path()
    espeakng_loader.make_library_available()

    EspeakWrapper.library_path = library_path
    EspeakWrapper.data_path = data_path

    # Verify espeak-ng (optional, but good for debugging)
    try:
        test_phonemes = phonemize('test', language='en-us')
        if not test_phonemes:
            raise Exception("Phonemization returned empty result")
    except Exception as e:
        print(f"Warning: espeak-ng test failed: {e}")

except ImportError as e:
    print(f"Warning: Required packages not found: {e}.  Attempting to install...")
    import subprocess
    try:
        subprocess.check_call(["pip", "install", "espeakng-loader", "phonemizer-fork"])
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        from phonemizer import phonemize
        import espeakng_loader

        library_path = espeakng_loader.get_library_path()
        data_path = espeakng_loader.get_data_path()
        espeakng_loader.make_library_available()
        EspeakWrapper.library_path = library_path
        EspeakWrapper.data_path = data_path
    except subprocess.CalledProcessError:
        print("Error: Failed to install espeakng-loader and phonemizer-fork.  Please install manually.")

# Initialize pipeline globally (and keep it cached)
_pipeline = None

def download_file(repo_id: str, filename: str, local_dir: str) -> str:
    """Downloads a file from Hugging Face Hub, only if it doesn't exist."""
    from huggingface_hub import hf_hub_download

    local_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        try:
            download_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, force_download=False)  # force_download=False is KEY
            print(f"Downloaded {filename} to {download_path}")
            return download_path
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return ""  # Return empty string on failure
    else:
        print(f"{filename} already exists locally.")
        return local_path # Return existing path



def download_voice_files(force_download: bool = False):
    """Download voice files from Hugging Face, only if they don't exist."""
    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)
    downloaded_voices = []

    for voice_file in VOICE_FILES:
        voice_path = voices_dir / voice_file
        if force_download or not voice_path.exists(): # Only download if forced OR file is missing
            if download_file(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{voice_file}", local_dir="."):
                downloaded_voices.append(voice_file)
        else:
            print(f"Voice file {voice_file} already exists")
            downloaded_voices.append(voice_file)
    return downloaded_voices

def build_model(model_path: str = 'kokoro-v1_0.pth', device: str = 'cpu') -> KPipeline:
    """Build and return the Kokoro pipeline (cached)."""
    global _pipeline

    if _pipeline is not None:  # Return cached pipeline if it exists
        return _pipeline

    try:
        patch_json_load()

        # Download model *only if it doesn't exist*
        if not os.path.exists(model_path):
            model_path = download_file(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v1_0.pth", local_dir=".")
            if not model_path:  # Check if download was successful
                raise ValueError("Failed to download model file.")
        else:
             print(f"Model {model_path} already exists locally.")


        # Download config *only if it doesn't exist*
        config_path = "config.json"
        if not os.path.exists(config_path):
            config_path = download_file(repo_id="hexgrad/Kokoro-82M", filename="config.json", local_dir=".")
            if not config_path:
                raise ValueError("Failed to download config file.")
        else:
            print(f"Config {config_path} already exists locally.")


        # Initialize pipeline
        _pipeline = KPipeline(lang_code='a')
        _pipeline.device = device
        _pipeline.voices = {}

        #  Don't load a voice here.  Let load_voice handle it.

        return _pipeline

    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        raise

def list_available_voices() -> List[str]:
    """List available voices (checks the 'voices' directory)."""
    voices_dir = Path("voices")
    if not voices_dir.exists():
        return []  # No voices directory = no voices
    return [f.stem for f in voices_dir.glob("*.pt")]

def load_voice(voice_name: str, device: str):
    """Load a specific voice, downloading if necessary."""
    pipeline = build_model(device=device) # Get the (cached) pipeline
    voice_name = voice_name.replace('.pt', '')  # Ensure no extension
    voice_path = f"voices/{voice_name}.pt"

    # Download the voice file if it doesn't exist.  This is the ONLY place
    # where voice files are downloaded (besides initial setup).
    if not os.path.exists(voice_path):
        print(f"Voice file {voice_path} not found.  Downloading...")
        download_voice_files()  #  Download all missing voices.
        if not os.path.exists(voice_path): # Check again after downloading
            raise ValueError(f"Voice file {voice_path} not found, even after download attempt.")

    return pipeline.load_voice(voice_path)  # Load the voice
