import os
import time
import torch
import numpy as np
import soundfile as sf
import logging
import librosa # Essential for pitch shifting!

from typing import Optional, List, Tuple, Callable
from pydub import AudioSegment
from kokoro import KPipeline 
from models import build_pipeline, list_available_voices, get_internal_voice_name

# Use the logger configured in main.py
logger = logging.getLogger(__name__)

# Define constants
OUTPUTS_DIR = "outputs"
TEMP_DIR = "temp_audio"
CHUNK_PREFIX = "chunk_"
DEFAULT_SAMPLERATE = 24000 

class KokoroTTSWrapper:
    """Wraps Kokoro KPipeline, handles voice loading, blending with weights, saving."""

    def __init__(
        self,
        output_dir: str = OUTPUTS_DIR,
        temp_sub_dir: str = TEMP_DIR,
        config: Optional[dict] = None
    ):
        logger.info("KokoroTTSWrapper.__init__ START")
        self.config = config if config else {}
        self.output_dir = output_dir
        self.temp_dir = os.path.join(self.output_dir, temp_sub_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        logger.info(f"TTS Wrapper using device: {self.device}")
        self.pipeline: Optional[KPipeline] = None

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        try:
            # Build the pipeline on initialization
            self.pipeline = build_pipeline(device=self.device)
            logger.info(f"Kokoro Pipeline built successfully on device {self.device}.")
        except Exception as e:
            logger.exception("Failed to initialize Kokoro pipeline.")
            raise RuntimeError(f"Failed to initialize TTS engine: {e}") from e

        logger.info("KokoroTTSWrapper.__init__ END")

    def synthesize(
        self,
        segments: List[Tuple[str, List[str], Optional[str]]],
        speed: float = 1.0,
        pitch: float = 1.0,
        alpha: float = 0.0, 
        beta: float = 0.0,  
        diffusion_steps: int = 0,
        embedding_scale: float = 1.0,
        sample_rate: int = DEFAULT_SAMPLERATE,
        output_format: str = 'WAV',
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[Tuple[str, str, np.ndarray, str]], Optional[str]]:
        
        if not self.pipeline:
            raise RuntimeError("TTS Pipeline is not initialized.")

        logger.info(f"Starting synthesis. Speed: {speed}, Pitch: {pitch}, Rate: {sample_rate}")

        all_audio_tensors: List[torch.Tensor] = []
        synthesis_result_list: List[Tuple[str, str, np.ndarray, str]] = []
        combined_filepath: Optional[str] = None
        total_segments = len(segments)

        try:
            # --- 1. Pre-load Voices (Standardized) ---
            # We must resolve all friendly names to internal names first
            voice_map = {}
            unique_internal_voices_needed = set()

            for _, segment_voices, _ in segments:
                for v_str in segment_voices:
                    # Handle "VoiceA+VoiceB" from UI
                    parts = v_str.split('+')
                    resolved_parts = []
                    for part in parts:
                        clean_part = part.split(':')[0].strip() # Remove weights if any
                        internal = get_internal_voice_name(clean_part)
                        if not internal:
                            # Fallback for custom files
                            internal = clean_part.lower().replace(" ", "_")
                        
                        unique_internal_voices_needed.add(internal)
                        resolved_parts.append(internal)
                    
                    # Map the full UI string to a COMMA-separated internal string
                    # e.g., "Alice+Bob" -> "bf_alice,bm_bob"
                    voice_map[v_str] = ",".join(resolved_parts)

            logger.info(f"Pre-loading {len(unique_internal_voices_needed)} voices: {unique_internal_voices_needed}")
            for internal_name in unique_internal_voices_needed:
                try:
                    self.pipeline.load_voice(internal_name)
                except Exception as load_error:
                    logger.error(f"Failed to load voice {internal_name}: {str(load_error)}")
                    pass 

            # --- 2. Process Segments ---
            for i, (text_chunk, segment_voices, weight_str) in enumerate(segments):
                segment_num = i + 1
                if not text_chunk.strip(): 
                    continue

                # Get the comma-separated spec we built earlier
                # If segment_voices is ['Alice+Bob'], we get 'bf_alice,bm_bob'
                # If multiple entries (rare in your UI logic), join them with comma too
                internal_specs = [voice_map.get(v, v) for v in segment_voices]
                blended_voice_spec = ",".join(internal_specs)
                
                logger.debug(f"Segment {segment_num}: Voice spec passed to pipeline: '{blended_voice_spec}'")
                
                chunk_results_count = 0
                try:
                    # Kwargs for pipeline
                    generate_kwargs = {
                        "voice": blended_voice_spec,
                        "speed": speed,
                    }
                    
                    # Advanced params logging
                    if diffusion_steps > 0 or alpha > 0 or beta > 0:
                        logger.debug("Advanced style params ignored (not supported by current Kokoro build).")

                    for result in self.pipeline(text_chunk, **generate_kwargs):
                        if hasattr(result, 'audio') and result.audio is not None:
                            try:
                                audio_tensor = result.audio.cpu().float().squeeze()
                                if audio_tensor.ndim != 1: 
                                    audio_tensor = audio_tensor.flatten()
                                
                                audio_data_numpy = audio_tensor.numpy()

                                # --- Pitch Shift ---
                                if pitch != 1.0:
                                    try:
                                        n_steps = 12 * np.log2(pitch)
                                        audio_data_numpy = librosa.effects.pitch_shift(
                                            audio_data_numpy, 
                                            sr=DEFAULT_SAMPLERATE, 
                                            n_steps=n_steps
                                        )
                                        audio_tensor = torch.from_numpy(audio_data_numpy)
                                    except Exception as e_pitch:
                                        logger.error(f"Pitch shift failed: {e_pitch}")

                                chunk_timestamp = time.strftime("%Y%m%d_%H%M%S")
                                unique_suffix = f"{segment_num}_{chunk_results_count}_{int(time.time()*1000)}"
                                chunk_filepath = os.path.join(self.temp_dir, f"{CHUNK_PREFIX}{chunk_timestamp}_{unique_suffix}.wav")
                                
                                # Save Chunk (WAV, Resampled)
                                self.save_audio(audio_data_numpy, chunk_filepath, format='WAV', target_sample_rate=sample_rate)

                                graphemes = getattr(result, 'graphemes', None) or ""
                                phonemes = getattr(result, 'phonemes', None) or ""
                                synthesis_result_list.append((graphemes, phonemes, audio_data_numpy, chunk_filepath))
                                all_audio_tensors.append(audio_tensor)
                                chunk_results_count += 1
                                
                            except Exception as proc_err:
                                logger.exception(f"Error processing chunk for seg {segment_num}: {proc_err}")
                                continue

                except Exception as synth_call_err:
                    logger.exception(f"Pipeline error seg {segment_num}: {synth_call_err}")
                    raise

                if progress_callback:
                    progress_callback(segment_num, total_segments)

            # --- 3. Final Combination ---
            if all_audio_tensors:
                logger.info(f"Combining {len(all_audio_tensors)} audio chunks...")
                combined_audio_tensor = torch.cat(all_audio_tensors, dim=0)
                combined_audio_numpy = combined_audio_tensor.cpu().float().numpy()
                
                combined_timestamp = time.strftime("%Y%m%d_%H%M%S")
                combined_filename = f"combined_{combined_timestamp}.{output_format.lower()}"
                combined_filepath = os.path.join(self.output_dir, combined_filename)
                
                # Save Final (User Format, Resampled)
                self.save_audio(combined_audio_numpy, combined_filepath, format=output_format, target_sample_rate=sample_rate)
                logger.info(f"Combined audio saved: {combined_filepath}")
            else:
                combined_filepath = None

        except Exception as e:
            logger.exception(f"Synthesis failed: {e}")
            raise

        logger.info("Synthesis complete.")
        return synthesis_result_list, combined_filepath

    def save_audio(self, audio_data_numpy: np.ndarray, filepath: str, format: str ='WAV', target_sample_rate: int = DEFAULT_SAMPLERATE):
        """Saves audio, resampling if needed, and enforcing PCM_16 for WAV compatibility."""
        
        current_rate = DEFAULT_SAMPLERATE # Kokoro native is 24000
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            target_format_upper = format.upper()

            # --- 1. Resample if needed (Requires librosa) ---
            if target_sample_rate != current_rate:
                # Use librosa to change sample rate (e.g. 24k -> 16k)
                audio_data_numpy = librosa.resample(audio_data_numpy, orig_sr=current_rate, target_sr=target_sample_rate)
            
            # --- 2. Format Conversion (Float32 -> Float32) ---
            if not np.issubdtype(audio_data_numpy.dtype, np.floating):
                 audio_float = audio_data_numpy.astype(np.float32)
            else:
                 audio_float = audio_data_numpy.astype(np.float32)

            # Mono check
            if audio_float.ndim > 1: 
                audio_float_mono = np.mean(audio_float, axis=1)
            else: 
                audio_float_mono = audio_float

            # Clip
            audio_clipped = np.clip(audio_float_mono, -1.0, 1.0)

            # --- 3. Save ---
            if target_format_upper == 'MP3':
                 audio_data_int16 = (audio_clipped * 32767).astype(np.int16)
                 audio_segment = AudioSegment(
                     data=audio_data_int16.tobytes(), 
                     sample_width=2, 
                     frame_rate=target_sample_rate, 
                     channels=1
                 )
                 audio_segment.export(filepath, format='mp3', bitrate="192k")

            elif target_format_upper == 'WAV':
                 # CRITICAL FIX: Use subtype='PCM_16' for UI compatibility
                 sf.write(filepath, audio_clipped, samplerate=target_sample_rate, format='WAV', subtype='PCM_16')

            else: 
                raise ValueError(f"Unsupported audio format: {format}")

        except Exception as e: 
            logger.exception(f"Error saving audio: {e}")
            
    def list_available_voices(self):
        return list_available_voices()