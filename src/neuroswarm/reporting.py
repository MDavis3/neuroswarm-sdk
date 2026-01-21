"""
Reporting utilities for robustness testing and presentation-ready metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os

import numpy as np
from numpy.typing import NDArray

from .decoding import SignalExtractor
from .noise import AdversarialNoiseGenerator, NoiseParams, generate_noise_sweep
from .physics import NeuroSwarmPhysics
from .types import WavelengthSweepParams


@dataclass
class DetectionSummary:
    """Summary metrics for spike detection."""
    precision: float
    recall: float
    f1_score: float
    ssnr: float
    mean_timing_error_ms: float


def summarize_detection(
    extractor: SignalExtractor,
    detected_spikes: NDArray[np.int64],
    true_spikes: NDArray[np.bool_],
    signal: NDArray[np.float64],
    baseline_photons: float = 1e5,
    tolerance_samples: int = 5,
) -> DetectionSummary:
    """
    Summarize detection performance with precision/recall/F1 and SSNR.
    """
    eval_metrics = extractor.evaluate_reconstruction(
        detected_spikes, true_spikes, tolerance_samples=tolerance_samples
    )
    ssnr = extractor.compute_ssnr(signal, detected_spikes, baseline_photons)

    # Mean timing error (if any matches)
    timing_errors = []
    true_indices = np.where(true_spikes)[0]
    for det in detected_spikes:
        if len(true_indices) == 0:
            break
        closest = true_indices[np.argmin(np.abs(true_indices - det))]
        timing_errors.append(abs(closest - det))
    mean_timing_error = float(np.mean(timing_errors)) if timing_errors else float("nan")

    return DetectionSummary(
        precision=eval_metrics["precision"],
        recall=eval_metrics["recall"],
        f1_score=eval_metrics["f1_score"],
        ssnr=ssnr,
        mean_timing_error_ms=mean_timing_error,
    )


def noise_sweep_report(
    clean_signal: NDArray[np.float64],
    dt_ms: float,
    noise_levels: List[float],
    extractor: SignalExtractor,
    baseline_photons: float = 1e5,
    seed: int = 42,
) -> Dict[str, NDArray[np.float64]]:
    """
    Generate F1/SSNR curves across noise levels.
    """
    sweep_results = generate_noise_sweep(
        clean_signal, dt_ms, noise_levels, baseline=baseline_photons, seed=seed
    )
    f1_scores = []
    ssnrs = []

    for result in sweep_results:
        noisy = result["noisy_signal"]
        processed = extractor.process_batch(noisy, dt_ms=dt_ms, baseline_photons=baseline_photons)
        true_spikes = clean_signal > np.percentile(clean_signal, 99.5)
        summary = summarize_detection(
            extractor,
            processed["spike_indices"],
            true_spikes,
            processed["preprocessed"],
            baseline_photons=baseline_photons,
            tolerance_samples=10,
        )
        f1_scores.append(summary.f1_score)
        ssnrs.append(summary.ssnr)

    return {
        "noise_levels": np.array(noise_levels, dtype=float),
        "f1_scores": np.array(f1_scores, dtype=float),
        "ssnr": np.array(ssnrs, dtype=float),
    }


def wavelength_sweep_report(
    physics: NeuroSwarmPhysics,
    sweep_params: Optional[WavelengthSweepParams] = None
) -> Dict[str, NDArray[np.float64]]:
    """
    Run wavelength sweep and return best wavelength for SSNR.
    """
    return physics.sweep_wavelengths(sweep_params)


def save_report(
    report: Dict[str, NDArray[np.float64]],
    path: str
) -> None:
    """
    Save a report dictionary to JSON (numpy arrays converted to lists).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in report.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
