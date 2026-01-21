"""
Tests for reporting utilities.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuroswarm.reporting import summarize_detection, noise_sweep_report, wavelength_sweep_report
from neuroswarm.decoding import SignalExtractor, DecodingParams
from neuroswarm.physics import NeuroSwarmPhysics
from neuroswarm.types import WavelengthSweepParams


def test_summarize_detection_basic():
    extractor = SignalExtractor()
    signal = np.zeros(1000)
    signal[200] = 1000
    true_spikes = np.zeros(1000, dtype=bool)
    true_spikes[200] = True
    detected = np.array([200], dtype=np.int64)

    summary = summarize_detection(extractor, detected, true_spikes, signal)
    assert summary.precision == pytest.approx(1.0, rel=0.01)
    assert summary.recall == pytest.approx(1.0, rel=0.01)


def test_noise_sweep_report_shapes():
    extractor = SignalExtractor(DecodingParams(spike_threshold=4.0))
    signal = np.zeros(2000)
    signal[500:510] = 5000
    noise_levels = [10, 50, 100]

    report = noise_sweep_report(signal, 0.1, noise_levels, extractor)
    assert len(report["noise_levels"]) == len(noise_levels)
    assert len(report["f1_scores"]) == len(noise_levels)
    assert len(report["ssnr"]) == len(noise_levels)


def test_wavelength_sweep_report():
    physics = NeuroSwarmPhysics()
    report = wavelength_sweep_report(physics, WavelengthSweepParams(step_nm=100.0))
    assert report["optimal_wavelength_nm"] >= 1000
    assert report["optimal_wavelength_nm"] <= 1700
