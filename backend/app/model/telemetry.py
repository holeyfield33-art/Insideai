"""JSON-safe telemetry metadata; missing readings are never fabricated as zero."""
from __future__ import annotations

import math


def finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def telemetry_fields(reading, detector=None) -> dict:
    reading = reading or {}
    calibrated = getattr(detector, "calibrated", None)
    return {
        "zeta_proxy": finite_number(reading.get("zeta_raw")),
        "spectral_gap": finite_number(reading.get("spectral_gap")),
        "flagged": reading.get("flagged") if isinstance(reading.get("flagged"), bool) else None,
        "calibrated": calibrated if isinstance(calibrated, bool) else None,
        "threshold": finite_number(getattr(detector, "threshold", None)),
        "calibration_scope": "backend_session" if detector is not None else None,
    }
