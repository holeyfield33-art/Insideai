"""CPU-only tests for trace serialization and provenance; no model download."""
import sys
import json
import hashlib
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model.telemetry import telemetry_fields
from app.streaming.protocol import step_events
from app.recording import _pkg_commit, RunRecorder

class TelemetryTests(unittest.TestCase):
    def test_real_fields_survive_protocol(self):
        reading = {"zeta_raw": .72, "spectral_gap": .18, "flagged": False}
        detector = SimpleNamespace(calibrated=True, threshold=.3)
        fields = telemetry_fields(reading, detector)
        layers = [dict(layer=i, attention={}, ffn=None, hidden_norm=1., residual_delta=.2, **fields) for i in range(24)]
        trace = dict(tokens=[], seq_len=1, embeddings={}, positional={}, layers=layers, logits={}, sampled={})
        events = [ev for ev, _ in step_events(trace, 7, 24) if ev['type']=='anomaly']
        self.assertEqual(len(events), 24) # backwards-compatible layer coverage
        for ev in events:
            self.assertEqual(ev['step'], 7)
            for key, value in fields.items(): self.assertEqual(ev[key], value)
        self.assertEqual(reading, {"zeta_raw": .72, "spectral_gap": .18, "flagged": False})

    def test_unavailable_does_not_mean_zero_or_healthy(self):
        fields = telemetry_fields(None)
        self.assertTrue(all(v is None for v in fields.values()))

    def test_nonfinite_and_uncalibrated(self):
        fields = telemetry_fields({"zeta_raw": float('nan'), "spectral_gap": float('inf')}, SimpleNamespace(calibrated=False, threshold=float('inf')))
        self.assertIsNone(fields['zeta_proxy'])
        self.assertIsNone(fields['spectral_gap'])
        self.assertIsNone(fields['threshold'])
        self.assertFalse(fields['calibrated'])

    def test_recording_captures_effective_metadata_and_exact_event_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            model = {"model": "test-fixture", "dtype": "float16", "device": "cuda:0", "model_revision": "revision-fixture", "gpu_model": "fixture", "gpu_vram_bytes": 123, "cuda_version": "fixture"}
            recorder = RunRecorder(run_dir, model, {"prompt": "unit test", "model": "test-fixture", "chat_mode": "raw", "speed": 0, "params": {"max_new_tokens": 1}})
            recorder.record({"type": "step_end", "step": 0})
            with patch('app.recording._repo_versions', return_value={}), patch('app.recording._repo_shas', return_value={}):
                recorder.finalize(text="test", steps=1, reason="max_tokens")
            manifest = json.loads((run_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['events_sha256'], hashlib.sha256((run_dir / 'run.jsonl').read_bytes()).hexdigest())
            for key in ('dtype', 'device', 'model_revision', 'gpu_model', 'gpu_vram_bytes', 'cuda_version'):
                self.assertEqual(manifest[key], model[key])

    def test_unknown_dependency_never_claims_declared_sha(self):
        self.assertIsNone(_pkg_commit('insideai-nonexistent-test-distribution'))

if __name__ == '__main__': unittest.main()
