"""Capture a live grounded-answer pilot without changing server configuration.

Run: backend/.venv/Scripts/python.exe backend/scripts/research_capture.py
Artifacts: results/research/<UTC timestamp>/ (new directory per invocation).
This observes the running backend, including its existing calibration history.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import websockets

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    {"id": "color_red", "prompt": "Fact: The ball is red. What color is the ball? Answer with one word.", "expected": "red"},
    {"id": "color_blue", "prompt": "Fact: The ball is blue. What color is the ball? Answer with one word.", "expected": "blue"},
    {"id": "count", "prompt": "Fact: There are seven keys. How many keys are there? Answer with one word.", "expected": "seven", "aliases": ["7"]},
    {"id": "unknown", "prompt": "Fact: A box is closed. What color is it? If the fact does not say, answer unknown. One word only.", "expected": "unknown"},
]


def score(text, case, truncated=False):
    # Full-output exact match, not substring matching; explicit EOS removal.
    normalized = re.sub(r'<\|(?:im_end|endoftext)\|>', '', text).strip().lower()
    normalized = normalized.strip(' \n\r\t.\"\'')
    return None if truncated else normalized in [case['expected'], *case.get('aliases', [])]


def summarize(events, case, layers):
    start = next(e for e in events if e['type'] == 'generation_start')
    end = next(e for e in events if e['type'] == 'generation_end')
    observations = {}
    keys = ('zeta_proxy', 'spectral_gap', 'flagged', 'calibrated', 'threshold', 'calibration_scope')
    for e in events:
        if e['type'] != 'anomaly':
            continue
        step = e['step']
        values = {k: e.get(k) for k in keys}
        if step not in observations:
            observations[step] = {**values, 'step': step, 'phase': 'prefill' if step == 0 else 'decode', 'layers': []}
        if any(observations[step][k] != values[k] for k in keys):
            raise ValueError(f'Conflicting shared telemetry within step {step}')
        observations[step]['layers'].append(e['layer'])
    coverage = all(len(e['layers']) == layers and set(e['layers']) == set(range(layers)) for e in observations.values())
    complete = set(observations) == set(range(end['steps'])) and coverage
    valid = not start.get('truncated', False) and end['reason'] not in ('cancelled', 'error') and complete
    return {'case': case['id'], 'expected': case['expected'], 'text': end['text'],
            'exact_match': score(end['text'], case) if valid else None,
            'valid_trial': valid, 'truncated_prompt': start.get('truncated', False),
            'generation_params': start.get('params'), 'chat_template_applied': start.get('template'),
            'reason': end['reason'], 'steps': end['steps'], 'duration_s': end['duration_s'],
            'event_counts': dict(Counter(e['type'] for e in events)),
            'layer_coverage_complete': complete, 'telemetry': list(observations.values()),
            'label_scope': 'full-output exact match to supplied fact; mismatch is not automatically hallucination'}


async def run(args):
    folder = ROOT / 'results' / 'research' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'ws_url': args.ws,
                'repeats': args.repeats, 'cases': CASES, 'intervention': 'none; observational baseline',
                'telemetry_version': 'running server; corrected experimental clone is NOT installed',
                'calibration': 'pre-existing shared backend session; not reset or controlled',
                'insideai_disk_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'disk_file_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                    ['backend/app/config.py', 'backend/app/model/engine.py', 'backend/app/model/telemetry.py', 'backend/scripts/research_capture.py']},
                'provenance_limit': 'disk hashes do not prove loaded-process code identity or GGUF blob identity',
                'status': 'running'}
    trials = []
    try:
        async with websockets.connect(args.ws, max_size=32*1024*1024, ping_timeout=120) as ws:
            info = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if info['type'] != 'model_info':
                raise RuntimeError('A live model is required')
            mode = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if mode.get('type') != 'server_mode' or not mode.get('live'):
                raise RuntimeError('Expected live server mode')
            manifest['model_info'] = info
            (folder/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(f"Capturing {info['model']} to {folder}", flush=True)
            for repeat in range(args.repeats):
                # Reverse order on alternate repetitions to expose some order effects.
                for case in (CASES if repeat % 2 == 0 else CASES[::-1]):
                    request = {'type': 'generate', 'prompt': case['prompt'], 'max_new_tokens': args.tokens,
                               'temperature': 0, 'top_k': 0, 'top_p': 1, 'repetition_penalty': 1,
                               'seed': 42, 'chat_mode': 'auto', 'speed': 0}
                    events = []
                    await ws.send(json.dumps(request))
                    with (folder/f"{repeat}_{case['id']}.jsonl").open('w', encoding='utf-8') as stream:
                        stream.write(json.dumps({'request': request})+'\n')
                        while True:
                            event = json.loads(await asyncio.wait_for(ws.recv(), 120))
                            stream.write(json.dumps({'received_monotonic': time.monotonic(), 'event': event})+'\n')
                            stream.flush()
                            events.append(event)
                            if event['type'] == 'error':
                                raise RuntimeError(event.get('message', str(event)))
                            if event['type'] == 'generation_end':
                                break
                    summary = summarize(events, case, info['n_layer'])
                    summary['repeat'] = repeat
                    trials.append(summary)
                    (folder/'summary.json').write_text(json.dumps(trials, indent=2), encoding='utf-8')
                    print(f"repeat={repeat} {case['id']}: {summary['text']!r}, exact={summary['exact_match']}, steps={summary['steps']}", flush=True)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (folder/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Completed {len(trials)} trials; exact matches={sum(t['exact_match'] is True for t in trials)}. Results: {folder}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ws', default='ws://127.0.0.1:8000/ws')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--tokens', type=int, default=8)
    options = parser.parse_args()
    if options.repeats < 1 or not 1 <= options.tokens <= 96:
        parser.error('repeats must be positive and tokens between 1 and 96')
    asyncio.run(run(options))
