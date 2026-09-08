"""Summarize a completed research_capture directory without counting layers as trials."""
import argparse
import json
from pathlib import Path
import statistics


def render(folder):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    trials = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Capture is incomplete; inspect manifest and raw events')
    valid = [t for t in trials if t['valid_trial']]
    steps = [s for t in valid for s in t['telemetry']]
    matches = sum(t['exact_match'] is True for t in valid)
    text = [f"# Live grounded-answer pilot: {manifest['model_info']['model']}", '',
            f"Started UTC: {manifest['started_utc']}", '',
            f"{len(valid)}/{len(trials)} valid trials; {matches}/{len(valid)} full-output exact matches.",
            f"{len(steps)} unique generation-step measurements. Repeated layer events are deduplicated.", '',
            'This is a tiny repeated-prompt pilot, not an estimate of general model accuracy or hallucination rate.', '',
            '| Case | Repeat | Output (including recorded terminators) | Exact match | Flagged steps |',
            '|---|---:|---|---|---:|']
    for t in trials:
        output = t['text'].replace('|', '&#124;').replace('\n', ' ')
        flagged = sum(s['flagged'] is True for s in t['telemetry'])
        text.append(f"| {t['case']} | {t['repeat']} | {output} | {t['exact_match']} | {flagged}/{t['steps']} |")
    text += ['', '## Phase-separated telemetry', '',
             '| Phase | Steps | Mean zeta | Mean spectral gap | Calibrated steps | Flagged steps |',
             '|---|---:|---:|---:|---:|---:|']
    for phase in ['prefill', 'decode']:
        selected = [s for s in steps if s['phase'] == phase]
        def mean(key):
            numbers = [s[key] for s in selected if isinstance(s[key], (int, float))]
            return f'{statistics.mean(numbers):.6g}' if numbers else 'unavailable'
        text.append(f"| {phase} | {len(selected)} | {mean('zeta_proxy')} | {mean('spectral_gap')} | {sum(s['calibrated'] is True for s in selected)} | {sum(s['flagged'] is True for s in selected)} |")
    text += ['', '## Repeated answers', '']
    for case in manifest['cases']:
        same = [t for t in valid if t['case'] == case['id']]
        identical = len({t['text'] for t in same}) == 1 if len(same) > 1 else None
        text.append(f"- {case['id']}: {len(same)} valid repetitions; identical raw output: {identical}.")
    correct_with_flags = sum(t['exact_match'] is True and any(s['flagged'] is True for s in t['telemetry']) for t in valid)
    text += ['', '## Interpretation and limits', '',
             f"Detector calibrated on {sum(s['calibrated'] is True for s in steps)}/{len(steps)} captured steps. Zero alarms during warmup do not establish detector sensitivity or specificity.",
             f'{correct_with_flags} exact-match trials contained at least one telemetry flag. A flag is an activation-change observation, not an answer-error label.',
             'A mismatched answer can also reflect verbosity or formatting; raw text is retained for review.',
             'Calibration belongs to the existing backend session and was not reset. Other clients may interleave inference. Neither calibration nor ordering was experimentally isolated.',
             'The live backend uses its existing telemetry implementation; the experimental spectral-gap correction was not installed. These gap values must not be mixed with corrected-gap baselines.',
             'Manifest disk hashes and live model_info are recorded separately. The server does not attest loaded code hashes or the loaded GGUF digest.',
             'No BOCPD, Chronos, or dual-node measurement is present in this protocol. No tensor intervention was applied.', '',
             '## Artifacts', '',
             '- manifest.json: configuration, live model metadata, disk-source provenance, completion status.',
             '- summary.json: per-trial scoring, layer coverage, phase-separated unique step telemetry.',
             '- *.jsonl: requests and all returned events, including attention, activations, logits, and timing.', '']
    (folder/'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    print('\n'.join(text))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    render(parser.parse_args().folder)
