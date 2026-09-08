# Experiment desk research

A fresh sibling clone is at `../unitarity-lab-experiments`, pinned at inspection to
`66c004b5f37a5f4a129761ef1d7f87f306b100ac`. The original sibling `unitarity-lab`
checkout and InsideAI's installed telemetry dependency were left intact.

Start with [the experiment report](../../unitarity-lab-experiments/experiments/README.md).
It explains the screen's measurements, records initial synthetic and real-model results,
and lays out integration work for Chronos Lock, BOCPD, and dual-node experiments.

From InsideAI in PowerShell:

```powershell
../unitarity-lab-experiments/experiments/run.ps1 metrics
../unitarity-lab-experiments/experiments/run.ps1 model
../unitarity-lab-experiments/experiments/run.ps1 tests
```

The model pilot runs independently using cached distilgpt2. It compares baseline,
no-op, 90% L3 residual contribution, and L3 residual blocking on four prompts.
Results are stored in the sibling clone's `experiments` directory.

Initial result: interventions change predictions, but effects on target-token probability
are mixed, and raw coherence stays high. This is an experimental control harness,
not a validated hallucination detector. The report also documents a rank-one spectral
gap discrepancy, BOCPD history dependence concerns, Chronos signed-drift cancellation,
and dual-sync policy penalties that must be distinguished from raw measurements.

The follow-up confirmed and fixed short-context spectral-gap calculations in the
experimental clone: 126 reference/related regression tests passed. On the user
transformer prompt, the baseline gap changed from 0.1875 to 638307.5 without
changing model predictions. Original and corrected JSON results are retained.
The running InsideAI dependency has not been upgraded; live integration still
requires resolving the clone's `var_spectral` packaging mismatch and recalibrating
thresholds against the corrected measurement. See the report for exact scope.

## Live Qwen GGUF follow-up

The user's GGUF loader changes were reviewed and retained. The live backend reports
`qwen2.5-coder:1.5b`, 28 layers, 12 heads, hidden dimension 1536, CPU bfloat16,
with a chat template. No server restart or model replacement was performed.

New capture and reporting commands:

```powershell
backend/.venv/Scripts/python.exe backend/scripts/research_capture.py
backend/.venv/Scripts/python.exe backend/scripts/research_report.py results/research/<capture-directory>
backend/.venv/Scripts/python.exe backend/scripts/research_capture_test.py
```

Each capture creates a fresh timestamped directory with requests, complete event
JSONL, model metadata, source hashes, and trial summaries. It tests four short
grounded questions twice, reversing order on the second pass, with greedy decoding
and seed 42. Scoring excludes explicit EOS markers but requires the full remaining
answer to match. Truncated prompts and incomplete layer coverage are not scored.
Five reducer/scoring regression tests passed.

First capture: `results/research/20260908T013518_259966Z/REPORT.md`.
**8/8 exact matches**, including red/blue context substitution and unknown for
missing information. All four repeated outputs were identical. The 896 layer
anomaly events reduce to **32 unique step observations**, with zero calibrated
steps and zero flags. These numbers do not establish hallucination-detection
performance. Prefill and decode are separated, and current live telemetry remains
distinct from the corrected experimental implementation.

Next priority: attest loaded GGUF/code identity in run manifests and expose
calibration progress. Resolve the new clone's VAR packaging mismatch before
upgrading its passive hook, then establish separate corrected prefill/decode
baselines. Broaden the grounded set with independently scored errors before
testing BOCPD-triggered or layer-targeted interventions.
