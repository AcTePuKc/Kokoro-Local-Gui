import os
import soundfile as sf
from typing import Optional, List, Tuple
import torch
import numpy as np
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from models import build_model, list_available_voices, load_voice
import time
import logging
import error_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('tts_wrapper.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class KokoroTTSWrapper:
    def __init__(
        self,
        output_dir: str = 'outputs',
        temp_dir: str = 'temp_audio',
        split_pattern: str = r'\n+',
        model_path: str = 'kokoro-v1_0.pth',
        config=None
    ):
        logger.info("KokoroTTSWrapper.__init__ START")
        self.config = config or {}
        self.voice = self.config.get('tts_engine', {}).get('voice', 'af_bella')
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.split_pattern = split_pattern
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Device: {self.device}")
        self.last_synthesis_timestamp = None

        logger.info("Initializing Kokoro TTS Model...")
        try:
            self.pipeline = build_model(self.model_path, self.device)
        except Exception as e:
            logger.exception("Failed to initialize pipeline.")
            raise e
        logger.info("Kokoro TTS Model Initialized.")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        available_voices = list_available_voices()
        if not available_voices:
            print("No voices found. They will be downloaded on first synthesis.")
        logger.info("KokoroTTSWrapper.__init__ END")

    def load_default_voice(self):
        try:
            logger.info(f"Loading default voice: {self.voice}")
            load_voice(self.voice, self.device)
            logger.info(f"Default voice '{self.voice}' loaded.")
        except Exception as e:
            logger.error(f"Could not load default voice '{self.voice}'. Error: {e}")
    
    def synthesize(
            self,
            text: str,
            speed: float = 1.0,
            selected_voice: Optional[str] = None
        ) -> Tuple[List[Tuple[str, str, np.ndarray]], Optional[str]]:
        logger.info(f"Synthesize - Voice: '{selected_voice}', Speed: {speed}, Text: {text[:50]}...")
        self.last_synthesis_timestamp = time.strftime("%Y%m%d_%H%M%S")
        try:
            load_voice(selected_voice, self.device)
            all_audio_tensors = []
            synthesis_result_list = []
            generator = self.pipeline(
                text,
                voice=f"voices/{selected_voice}.pt",
                speed=speed,
                split_pattern=self.split_pattern
            )
            for chunk_index, (gs, ps, audio_tensor) in enumerate(generator):
                if audio_tensor is not None:
                    audio_data_numpy = audio_tensor.cpu().numpy()
                    if audio_data_numpy.ndim == 1:
                        audio_data_numpy = audio_data_numpy.reshape(-1, 1)
                    chunk_filepath = os.path.join(self.temp_dir, f"chunk_{self.last_synthesis_timestamp}_{chunk_index}.wav")
                    self.save_audio(audio_data_numpy, chunk_filepath)
                    synthesis_result_list.append((gs, ps, audio_data_numpy, chunk_filepath))
                    all_audio_tensors.append(audio_tensor)
            if all_audio_tensors:
                combined_audio_tensor = torch.cat(all_audio_tensors, dim=0)
                combined_filepath = os.path.join(self.temp_dir, f"combined_{self.last_synthesis_timestamp}.wav")
                self.save_audio(combined_audio_tensor.cpu().numpy(), combined_filepath)
                return synthesis_result_list, combined_filepath
            else:
                logger.error("No audio chunks generated.")
                return [], None
        except Exception as e:
            logger.exception(f"Synthesis exception: {e}")
            raise e

    def save_audio(self, audio_data_numpy, filepath, format='WAV'):
        logger.info(f"Saving to: {filepath}, Format: {format}")
        try:
            # Convert float32 audio to int16
            audio_int16 = (audio_data_numpy * 32767).astype(np.int16)
            sf.write(
                filepath,
                audio_int16,
                samplerate=24000,
                format=format.upper(),
                subtype='PCM_16'  # Specify 16-bit integer format
            )
            logger.info(f"Audio saved: {filepath}")
        except Exception as e:
            logger.exception(f"Error saving to '{filepath}' as {format}:")
            fallback_path = filepath.rsplit('.', 1)[0] + ".wav"
            # Convert for fallback as well
            audio_int16 = (audio_data_numpy * 32767).astype(np.int16)
            sf.write(
                fallback_path,
                audio_int16,
                samplerate=24000,
                format='WAV',
                subtype='PCM_16'
            )
            logger.info(f"Saved as fallback WAV: {fallback_path}")
            return fallback_path
        return filepath
