"""Regression checks for research trial validity and shared-step deduplication."""
import unittest
from research_capture import score, summarize


class CaptureTests(unittest.TestCase):
    def events(self):
        return [{'type': 'generation_start', 'truncated': False},
                *[{'type': 'anomaly', 'step': 0, 'layer': i, 'zeta_proxy': .9, 'spectral_gap': 1} for i in range(2)],
                {'type': 'generation_end', 'text': 'red<|im_end|>', 'steps': 1, 'reason': 'eos', 'duration_s': 1}]

    def test_deduplicates_shared_reading(self):
        result = summarize(self.events(), {'id': 'red', 'expected': 'red'}, 2)
        self.assertEqual(len(result['telemetry']), 1)
        self.assertTrue(result['exact_match'])

    def test_missing_layer_is_invalid(self):
        events = self.events()
        del events[1]
        self.assertIsNone(summarize(events, {'id': 'red', 'expected': 'red'}, 2)['exact_match'])

    def test_conflicting_shared_values_raise(self):
        events = self.events()
        events[1]['spectral_gap'] = 2
        with self.assertRaises(ValueError):
            summarize(events, {'id': 'red', 'expected': 'red'}, 2)

    def test_truncation_is_not_scored(self):
        events = self.events()
        events[0]['truncated'] = True
        self.assertIsNone(summarize(events, {'id': 'red', 'expected': 'red'}, 2)['exact_match'])

    def test_substring_is_not_full_answer_match(self):
        self.assertFalse(score('red or blue', {'expected': 'red'}))


if __name__ == '__main__':
    unittest.main()
