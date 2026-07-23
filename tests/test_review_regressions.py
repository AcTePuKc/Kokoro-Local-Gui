import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import tts_wrapper
from ui_main import SynthesisWorker


ROOT = Path(__file__).resolve().parents[1]


class FakeAudio:
    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=np.float32)

    def cpu(self):
        return self

    def float(self):
        return self

    def squeeze(self):
        return self

    @property
    def ndim(self):
        return self.samples.ndim

    def flatten(self):
        self.samples = self.samples.flatten()
        return self

    def numpy(self):
        return self.samples


class FakePipeline:
    def load_voice(self, _voice):
        return None

    def __call__(self, _text, **_kwargs):
        yield SimpleNamespace(
            audio=FakeAudio([0.0, 0.25, -0.25]),
            graphemes="Hello",
            phonemes="həˈloʊ",
        )


class ReviewRegressionTests(unittest.TestCase):
    def test_synthesis_results_contain_metadata_not_numpy_audio(self):
        wrapper = tts_wrapper.KokoroTTSWrapper.__new__(tts_wrapper.KokoroTTSWrapper)
        wrapper.pipeline = FakePipeline()
        wrapper.output_dir = "outputs"
        wrapper.temp_dir = "outputs/temp_audio"
        wrapper.save_audio = Mock()
        combined_file = Mock()

        with (
            patch.object(tts_wrapper, "get_internal_voice_name", return_value="af_heart"),
            patch.object(tts_wrapper.sf, "SoundFile", return_value=combined_file),
            patch.object(tts_wrapper.os, "makedirs"),
            patch.object(tts_wrapper.os.path, "exists", return_value=True),
        ):
            chunks, combined = wrapper.synthesize(
                [("Hello", ["Heart"], None)],
                output_format="WAV",
            )

        self.assertTrue(combined.endswith(".wav"))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], ("Hello", "həˈloʊ", chunks[0][2]))
        self.assertIsInstance(chunks[0][2], str)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in chunks[0]))
        combined_file.write.assert_called_once()
        combined_file.close.assert_called_once()

    def test_waveform_loader_error_includes_path_and_exception(self):
        worker = SynthesisWorker(None)
        errors = []
        worker.error.connect(errors.append)

        with patch("ui_main.os.path.exists", return_value=True), patch(
            "ui_main.AudioSegment.from_file", side_effect=RuntimeError("decode failed")
        ):
            self.assertIsNone(worker._load_waveform_data("broken.mp3"))

        self.assertEqual(errors, ["Waveform loader failed for broken.mp3: decode failed"])

    def test_detail_progress_does_not_reference_an_undefined_segment_index(self):
        tree = ast.parse((ROOT / "ui_main.py").read_text(encoding="utf-8-sig"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "update_detail_progress"
        )
        loaded_names = {
            node.id
            for node in ast.walk(method)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertNotIn("curr", loaded_names)

    def test_history_children_keep_play_copy_and_save_actions(self):
        source = (ROOT / "ui_main.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "populate_history_table"
        )
        method_source = ast.get_source_segment(source, method)

        self.assertIn('setToolTip("Play segment")', method_source)
        self.assertIn('setToolTip("Copy text to clipboard")', method_source)
        self.assertIn("btn_save_c", method_source)
        self.assertIn("setItemWidget(child_item, 3, wid_c)", method_source)

    def test_readme_documents_mp3_memory_limitation(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8-sig").lower()
        self.assertIn("mp3", readme)
        self.assertIn("memory", readme)


if __name__ == "__main__":
    unittest.main()
