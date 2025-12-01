import os
from pathlib import Path
import logging
import torch
from typing import List, Optional, Dict, Tuple

# Use the logger configured in main.py
logger = logging.getLogger(__name__)

# --- Configuration ---
VOICES_DIR = Path("voices") # Directory for CUSTOM local .pt voice files

# --- Voice Name Mapping ---
# ONLY Valid voices from hexgrad/Kokoro-82M
VOICE_NAME_MAP: Dict[str, str] = {
    # --- American Female ---
    "af_bella": "🇺🇸 Female (Bella)",
    "af_sarah": "🇺🇸 Female (Sarah)",
    "af_nicole": "🇺🇸 Female (Nicole)",
    "af_sky":    "🇺🇸 Female (Sky)",
    "af_heart":  "🇺🇸 Female (Heart)",
    "af_alloy":  "🇺🇸 Female (Alloy)",
    "af_aoede":  "🇺🇸 Female (Aoede)",
    "af_kore":   "🇺🇸 Female (Kore)",
    
    # --- American Male ---
    "am_adam":    "🇺🇸 Male (Adam)",
    "am_michael": "🇺🇸 Male (Michael)",
    "am_onyx":    "🇺🇸 Male (Onyx)",
    "am_puck":    "🇺🇸 Male (Puck)",
    "am_fenrir":  "🇺🇸 Male (Fenrir)",

    # --- British Female ---
    "bf_emma":     "🇬🇧 Female (Emma)",
    "bf_isabella": "🇬🇧 Female (Isabella)",
    "bf_alice":    "🇬🇧 Female (Alice)",
    "bf_lily":     "🇬🇧 Female (Lily)",

    # --- British Male ---
    "bm_george": "🇬🇧 Male (George)",
    "bm_lewis":  "🇬🇧 Male (Lewis)",
    "bm_daniel": "🇬🇧 Male (Daniel)",
    "bm_fable":  "🇬🇧 Male (Fable)",
}

# --- Cached Voices ---
_cached_available_voices: Optional[List[str]] = None
_cached_internal_to_friendly: Optional[Dict[str, str]] = None
_cached_friendly_to_internal: Optional[Dict[str, str]] = None


def _build_voice_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Builds the internal map by combining the hardcoded VOICE_NAME_MAP 
    with any custom .pt files found in the voices directory.
    """
    global _cached_internal_to_friendly, _cached_friendly_to_internal
    
    if _cached_internal_to_friendly is not None and _cached_friendly_to_internal is not None:
        return _cached_internal_to_friendly, _cached_friendly_to_internal

    # 1. Start with the Standard Voices
    internal_to_friendly_map = VOICE_NAME_MAP.copy()

    # 2. Scan local directory for EXTRA custom voices
    if VOICES_DIR.is_dir():
        found_files = list(VOICES_DIR.glob("*.pt"))
        if found_files:
            logger.info(f"Scanning '{VOICES_DIR}' for custom voices... Found {len(found_files)}.")
            for file_path in found_files:
                internal_stem = file_path.stem 
                if internal_stem not in internal_to_friendly_map:
                    friendly_name = f"📁 Custom ({internal_stem})"
                    internal_to_friendly_map[internal_stem] = friendly_name
                    logger.info(f"Added custom voice: {internal_stem}")
    else:
        try:
            os.makedirs(VOICES_DIR, exist_ok=True)
        except Exception:
            pass

    # 3. Build Reverse Map
    f_to_int = {friendly: internal for internal, friendly in internal_to_friendly_map.items()}

    _cached_internal_to_friendly = internal_to_friendly_map
    _cached_friendly_to_internal = f_to_int

    return _cached_internal_to_friendly, _cached_friendly_to_internal


def list_available_voices() -> List[str]:
    """Returns a sorted list of available user-friendly voice names."""
    global _cached_available_voices
    
    if _cached_available_voices is None:
         int_to_f, _ = _build_voice_maps()
         _cached_available_voices = sorted(list(int_to_f.values()))
         logger.info(f"Refreshed available voices list: {len(_cached_available_voices)} voices found.")

    return _cached_available_voices


def get_internal_voice_name(user_friendly_name: str) -> Optional[str]:
    """Converts a user-friendly voice name back to its internal filename stem."""
    _, f_to_int = _build_voice_maps()
    return f_to_int.get(user_friendly_name)


def get_friendly_voice_name(internal_name: str) -> Optional[str]:
    """Converts an internal filename stem to its user-friendly voice name."""
    int_to_f, _ = _build_voice_maps()
    return int_to_f.get(internal_name)


def build_pipeline(lang_code: str = 'a', device: Optional[str] = None):
    """Builds and returns a KPipeline instance."""
    try:
        from kokoro import KPipeline
    except ImportError as e:
        logger.critical("Failed to import KPipeline. Is 'kokoro' installed?", exc_info=True)
        raise ImportError("Kokoro TTS library not found.") from e

    resolved_device = device
    if not resolved_device:
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Building KPipeline (lang='{lang_code}', device='{resolved_device}')...")

    try:
        pipeline = KPipeline(lang_code=lang_code, device=resolved_device)
        logger.info("KPipeline built successfully.")
        return pipeline
    except Exception as e:
        logger.exception(f"Error building KPipeline on {resolved_device}.")
        raise RuntimeError(f"Failed to build KPipeline: {e}") from e