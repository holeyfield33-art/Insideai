# Instrument workbench

This branch redesigns InsideAI's actual application. It preserves the existing
Hugging Face backend, WebSocket transport, real tensor-derived 3D views,
generation parameters, model selection via environment, and recording/replay.
No generated or synthetic product data is introduced.

## Interaction and layout

The prompt is available before the 3D stage. Five numbered view selectors
control the camera; a focused view isolates its structure. Orbit, pan, zoom,
reset, opt-in tour, and clickable layer selection remain available. The canvas
also accepts keyboard arrow orbit, +/− zoom, and Home reset. Reduced-motion
users get immediate camera navigation instead of glides.

The inspector is a separate dock with Inspect, Output, and Runs panels; all
remain accessible on mobile. Expand stage temporarily hides the dock and the
same control restores it. Tabs support left/right keyboard navigation.
Graphite surfaces, copper processing accents, readable labels, hard borders,
and restrained corners replace the floating glass treatment. The new palette
has not received a formal color-vision/accessibility certification.

## Measurement corrections

The protocol still emits layer coverage events so existing clients/replays
remain compatible. The frontend stores one EKG point per generation step,
replacing a duplicate step and ignoring older steps in the chart. Layer
coverage remains separate. New run events clear trace history. The regression
suite verifies 24 layers produce one point and the history remains bounded.

The backend sends real spectral_gap, detector calibration state, finite
threshold, and backend_session scope. Missing/nonfinite readings are null.
Coherence uses a fixed −1 to +1 scale; spectral gap uses a labeled automatic
scale and a threshold line when known. Flags are not driven by coherence.
No score is presented as evidence of model health or fault localization.
The existing detector and calibration policy are unchanged.

Recordings include effective dtype, checkpoint revision when available,
device/GPU/CUDA metadata, and a SHA-256 hash of the recorded event file.
Unknown installed dependency SHAs remain null rather than silently borrowing
declared source pins. Old recordings still replay; new metadata is optional.
The recorded event stream contains display reductions, not lossless matrices
for the future model-rank experiment.

## Verification

- Production Next.js build and TypeScript validation.
- Frontend reducer regression script: duplicate steps, layer coverage,
  unavailable old-replay signals, nonfinite values, ordering, history bounds,
  new-run reset, replay identity, and camera selection behavior.
- Python unit tests: real field forwarding through the protocol, null/nonfinite
  handling, calibration state, unknown provenance, and recording metadata/hash.
- Python compile check for edited backend modules.

Live browser/visual QA and larger-model GPU inference were not run here. The
workspace has no NVIDIA GPU. Existing screenshot files show the former UI;
they are not screenshots of this change.

## Next experiment gate

1. Recover/version the earlier real-model rank-audit runner and fixed prompts.
2. Review the research repository's existing numerical-hardening branch.
3. On available GPU hardware, verify one passive Qwen2.5-3B-Instruct run and
   recording before scaling. Record effective precision and peak memory.
4. Keep the live compatibility smoke test separate from the frozen scientific
   experiment. Do not change layer comparisons or success gates silently.
5. Add lossless selected-layer capture and baseline comparison as a separate
   measured experiment slice; this UI change does not claim those exist.

No active intervention, dual-node transport change, paid compute, deployment,
or research-branch merge is included in this workbench change.
