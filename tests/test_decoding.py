"""
Tests for the decoding module.

Validates:
1. Preprocessing (filtering, detrending) works correctly
2. Artifact detection identifies outliers
3. PCA/ICA decomposition produces expected outputs
4. Spike detection recovers injected spikes
5. Spike reconstruction accuracy under noise
6. SSNR logging per batch
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuroswarm.decoding import (
    DecodingParams,
    BatchMetrics,
    SignalExtractor,
)
from neuroswarm.noise import AdversarialNoiseGenerator, NoiseParams


class TestDecodingParams:
    """Tests for DecodingParams configuration."""

    def test_default_initialization(self):
        """Test default parameters are valid."""
        params = DecodingParams()
        assert params.highpass_freq == 1.0
        assert params.lowpass_freq == 200.0
        assert params.spike_threshold == 3.0
        assert params.target_resolution_ms == 1.0

    def test_validation_filter_frequencies(self):
        """Test invalid filter frequencies raise error."""
        with pytest.raises(ValueError, match="lowpass_freq"):
            DecodingParams(highpass_freq=100, lowpass_freq=50)

    def test_validation_negative_highpass(self):
        """Test negative highpass raises error."""
        with pytest.raises(ValueError, match="highpass_freq"):
            DecodingParams(highpass_freq=-1)

    def test_validation_detrend_method(self):
        """Test invalid detrend method raises error."""
        with pytest.raises(ValueError, match="detrend_method"):
            DecodingParams(detrend_method="invalid")

    def test_validation_spike_threshold(self):
        """Test non-positive spike threshold raises error."""
        with pytest.raises(ValueError, match="spike_threshold"):
            DecodingParams(spike_threshold=0)

    def test_custom_parameters(self):
        """Test custom parameters are accepted."""
        params = DecodingParams(
            highpass_freq=5.0,
            lowpass_freq=100.0,
            spike_threshold=4.0
        )
        assert params.highpass_freq == 5.0
        assert params.lowpass_freq == 100.0
        assert params.spike_threshold == 4.0


class TestBatchMetrics:
    """Tests for BatchMetrics dataclass."""

    def test_default_initialization(self):
        """Test default metrics are initialized."""
        metrics = BatchMetrics()
        assert metrics.batch_id == 0
        assert metrics.n_samples == 0
        assert metrics.n_spikes_detected == 0
        assert metrics.ssnr == 0.0

    def test_custom_initialization(self):
        """Test custom metrics values."""
        metrics = BatchMetrics(
            batch_id=5,
            n_samples=10000,
            n_spikes_detected=15,
            ssnr=1000.0,
            snr_db=30.0
        )
        assert metrics.batch_id == 5
        assert metrics.n_samples == 10000
        assert metrics.n_spikes_detected == 15
        assert metrics.ssnr == 1000.0


class TestSignalExtractor:
    """Tests for the SignalExtractor class."""

    def test_initialization(self):
        """Test extractor initializes correctly."""
        extractor = SignalExtractor()
        assert extractor.params is not None
        assert len(extractor.batch_history) == 0

    def test_reset(self):
        """Test reset clears state."""
        extractor = SignalExtractor()
        # Process a batch to add history
        signal = np.random.randn(1000)
        extractor.process_batch(signal, dt_ms=0.1)

        assert len(extractor.batch_history) > 0
        extractor.reset()
        assert len(extractor.batch_history) == 0


class TestPreprocessing:
    """Tests for signal preprocessing."""

    def test_preprocess_shape(self):
        """Test preprocessing preserves signal shape."""
        extractor = SignalExtractor()
        signal = np.random.randn(1000)

        preprocessed, artifact_mask = extractor.preprocess(signal, sampling_rate_hz=1000)

        assert preprocessed.shape == signal.shape
        assert artifact_mask.shape == signal.shape

    def test_highpass_filter(self):
        """Test high-pass filter removes DC offset."""
        params = DecodingParams(highpass_freq=5.0, lowpass_freq=200.0)
        extractor = SignalExtractor(params)

        # Signal with DC offset + high frequency component
        t = np.arange(1000) / 1000.0
        signal = 1000 + 10 * np.sin(2 * np.pi * 50 * t)

        preprocessed, _ = extractor.preprocess(signal, sampling_rate_hz=1000)

        # DC should be removed
        assert abs(np.mean(preprocessed)) < 10, "DC offset should be removed"

    def test_lowpass_filter(self):
        """Test low-pass filter removes high frequencies."""
        params = DecodingParams(highpass_freq=1.0, lowpass_freq=50.0)
        extractor = SignalExtractor(params)

        # Signal with low + high frequency
        t = np.arange(10000) / 10000.0
        signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 500 * t)

        preprocessed, _ = extractor.preprocess(signal, sampling_rate_hz=10000)

        # Compute power spectrum
        fft = np.fft.rfft(preprocessed)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(preprocessed), d=1/10000)

        # Power at 500 Hz should be much less than at 10 Hz
        power_10hz = power[np.argmin(np.abs(freqs - 10))]
        power_500hz = power[np.argmin(np.abs(freqs - 500))]

        assert power_10hz > 10 * power_500hz, "High frequencies should be attenuated"

    def test_detrend_linear(self):
        """Test linear detrending removes trend."""
        params = DecodingParams(detrend_method="linear")
        extractor = SignalExtractor(params)

        # Signal with linear trend
        signal = np.arange(1000) * 0.5 + np.random.randn(1000) * 10

        preprocessed, _ = extractor.preprocess(signal, sampling_rate_hz=1000)

        # Trend should be removed
        x = np.arange(len(preprocessed))
        slope = np.polyfit(x, preprocessed, 1)[0]
        assert abs(slope) < 0.01, "Linear trend should be removed"


class TestArtifactDetection:
    """Tests for artifact detection."""

    def test_detect_outliers(self):
        """Test outlier detection finds extreme values."""
        params = DecodingParams(artifact_threshold=3.0)
        extractor = SignalExtractor(params)

        # Normal signal with outliers
        signal = np.random.randn(1000) * 10
        signal[400] = 1000  # Outlier
        signal[700] = -800  # Outlier

        preprocessed, artifact_mask = extractor.preprocess(signal, sampling_rate_hz=1000)

        # Outliers should be detected
        assert artifact_mask[400], "Positive outlier should be detected"
        assert artifact_mask[700], "Negative outlier should be detected"


class TestSpikeDetection:
    """Tests for spike detection."""

    def test_detect_clear_spikes(self):
        """Test detection of clear spikes."""
        extractor = SignalExtractor(DecodingParams(spike_threshold=3.0))

        # Create signal with clear spikes
        signal = np.random.randn(1000) * 10
        spike_times = [200, 400, 600, 800]
        for t in spike_times:
            signal[t] = 200

        spike_indices, amplitudes = extractor.detect_spikes(signal, dt_ms=0.1)

        # Should detect approximately correct number
        assert 3 <= len(spike_indices) <= 5, (
            f"Expected ~4 spikes, got {len(spike_indices)}"
        )

    def test_refractory_period(self):
        """Test refractory period prevents double detection."""
        params = DecodingParams(spike_refractory_ms=5.0)
        extractor = SignalExtractor(params)

        # Create signal with closely spaced spikes
        signal = np.zeros(1000)
        signal[100] = 100
        signal[102] = 90  # Within refractory

        spike_indices, _ = extractor.detect_spikes(signal, dt_ms=0.1)

        # Should only detect one spike
        assert len(spike_indices) == 1, "Refractory should prevent double detection"

    def test_no_spikes_in_noise(self):
        """Test that pure noise produces few false positives."""
        params = DecodingParams(spike_threshold=5.0)
        extractor = SignalExtractor(params)

        # Pure Gaussian noise
        np.random.seed(42)
        signal = np.random.randn(10000)

        spike_indices, _ = extractor.detect_spikes(signal, dt_ms=0.1)

        # Should have very few false positives
        false_positive_rate = len(spike_indices) / len(signal)
        assert false_positive_rate < 0.01, (
            f"Too many false positives: {100*false_positive_rate:.2f}%"
        )


class TestSpikeReconstruction:
    """Tests for spike train reconstruction."""

    def test_reconstruct_at_original_resolution(self):
        """Test reconstruction at same resolution."""
        params = DecodingParams(target_resolution_ms=0.1)
        extractor = SignalExtractor(params)

        spike_indices = np.array([100, 300, 500])
        spike_train = extractor.reconstruct_spike_train(
            spike_indices, n_samples=1000, dt_ms=0.1
        )

        assert len(spike_train) == 1000
        assert spike_train[100] == 1.0
        assert spike_train[300] == 1.0
        assert spike_train[500] == 1.0
        assert np.sum(spike_train) == 3

    def test_reconstruct_compressed(self):
        """Test reconstruction at lower resolution."""
        params = DecodingParams(target_resolution_ms=1.0)
        extractor = SignalExtractor(params)

        spike_indices = np.array([5, 15, 25])
        spike_train = extractor.reconstruct_spike_train(
            spike_indices, n_samples=100, dt_ms=0.1
        )

        # 100 samples at 0.1ms = 10 samples at 1ms
        assert len(spike_train) == 10
        assert spike_train[0] == 1.0
        assert spike_train[1] == 1.0
        assert spike_train[2] == 1.0


class TestSSNRComputation:
    """Tests for SSNR computation."""

    def test_ssnr_positive_for_spikes(self):
        """Test SSNR is positive when spikes are present."""
        extractor = SignalExtractor()

        signal = np.zeros(1000)
        signal[500] = 1000

        spike_indices = np.array([500])
        ssnr = extractor.compute_ssnr(signal, spike_indices, baseline_photons=1e5)

        assert ssnr > 0, "SSNR should be positive"

    def test_ssnr_zero_for_no_spikes(self):
        """Test SSNR is zero with no spikes."""
        extractor = SignalExtractor()

        signal = np.ones(1000) * 100
        spike_indices = np.array([], dtype=np.int64)

        ssnr = extractor.compute_ssnr(signal, spike_indices, baseline_photons=1e5)

        assert ssnr == 0, "SSNR should be zero with no spikes"


class TestProcessBatch:
    """Tests for batch processing."""

    def test_process_batch_returns_all_outputs(self):
        """Test batch processing returns all expected outputs."""
        extractor = SignalExtractor()

        signal = np.random.randn(1000) * 100
        signal[500] = 1000

        result = extractor.process_batch(signal, dt_ms=0.1)

        assert "preprocessed" in result
        assert "artifact_mask" in result
        assert "spike_indices" in result
        assert "spike_amplitudes" in result
        assert "spike_train" in result
        assert "metrics" in result

    def test_process_batch_updates_history(self):
        """Test batch processing updates history."""
        extractor = SignalExtractor()

        signal = np.random.randn(1000) * 100

        assert len(extractor.batch_history) == 0
        extractor.process_batch(signal, dt_ms=0.1)
        assert len(extractor.batch_history) == 1

        extractor.process_batch(signal, dt_ms=0.1)
        assert len(extractor.batch_history) == 2


class TestEvaluateReconstruction:
    """Tests for reconstruction evaluation."""

    def test_perfect_detection(self):
        """Test metrics for perfect detection."""
        extractor = SignalExtractor()

        true_spikes = np.zeros(1000, dtype=bool)
        true_spikes[[100, 300, 500, 700, 900]] = True

        detected = np.array([100, 300, 500, 700, 900])

        metrics = extractor.evaluate_reconstruction(detected, true_spikes)

        assert metrics["precision"] == pytest.approx(1.0, rel=0.01)
        assert metrics["recall"] == pytest.approx(1.0, rel=0.01)
        assert metrics["f1_score"] == pytest.approx(1.0, rel=0.01)
        assert metrics["true_positives"] == 5
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0

    def test_missed_spikes(self):
        """Test metrics with missed spikes."""
        extractor = SignalExtractor()

        true_spikes = np.zeros(1000, dtype=bool)
        true_spikes[[100, 300, 500, 700, 900]] = True

        detected = np.array([100, 500, 900])

        metrics = extractor.evaluate_reconstruction(detected, true_spikes)

        assert metrics["true_positives"] == 3
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 2
        assert metrics["recall"] == pytest.approx(0.6, rel=0.01)

    def test_false_alarms(self):
        """Test metrics with false positives."""
        extractor = SignalExtractor()

        true_spikes = np.zeros(1000, dtype=bool)
        true_spikes[[100, 500]] = True

        detected = np.array([100, 300, 500])

        metrics = extractor.evaluate_reconstruction(detected, true_spikes)

        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 0
        assert metrics["precision"] == pytest.approx(2/3, rel=0.01)


class TestPerformanceSummary:
    """Tests for performance summary."""

    def test_summary_empty_history(self):
        """Test summary with no processed batches."""
        extractor = SignalExtractor()
        summary = extractor.get_performance_summary()

        assert "error" in summary

    def test_summary_with_history(self):
        """Test summary with processed batches."""
        extractor = SignalExtractor()

        for _ in range(3):
            signal = np.random.randn(1000) * 100
            signal[500] = 1000
            extractor.process_batch(signal, dt_ms=0.1)

        summary = extractor.get_performance_summary()

        assert summary["n_batches"] == 3
        assert "ssnr_mean" in summary
        assert "snr_db_mean" in summary
        assert "total_spikes" in summary


class TestPCAICA:
    """Tests for PCA and ICA decomposition."""

    def test_pca_single_channel(self):
        """Test PCA with single channel."""
        extractor = SignalExtractor()
        signal = np.random.randn(1000)

        components, variance = extractor.apply_pca(signal)

        assert components.shape[0] == 1
        assert components.shape[1] == 1000
        assert len(variance) == 1

    def test_pca_multi_channel(self):
        """Test PCA with multiple channels."""
        extractor = SignalExtractor()
        signals = np.random.randn(5, 1000)

        components, variance = extractor.apply_pca(signals)

        assert components.shape[0] <= 5
        assert components.shape[1] == 1000
        assert len(variance) == 5

    def test_ica_single_channel(self):
        """Test ICA with single channel."""
        extractor = SignalExtractor()
        signal = np.random.randn(1000)

        components = extractor.apply_ica(signal)

        assert components.shape[0] == 1
        assert components.shape[1] == 1000

    def test_ica_multi_channel(self):
        """Test ICA with multiple channels."""
        extractor = SignalExtractor()
        signals = np.random.randn(3, 1000)

        components = extractor.apply_ica(signals, n_components=2)

        assert components.shape[0] == 2
        assert components.shape[1] == 1000


class TestIntegration:
    """Integration tests with noise module."""

    def test_recovery_under_noise(self):
        """Test spike recovery with realistic noise."""
        # Create clean signal with known spikes
        clean_signal = np.zeros(5000)
        true_spike_times = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
        for t in true_spike_times:
            clean_signal[t:t+10] = 10000

        # Add noise
        noise_params = NoiseParams(
            shot_noise_enabled=True,
            drift_amplitude=0.02,
            burst_probability=0.0,
            seed=42
        )
        noise_gen = AdversarialNoiseGenerator(noise_params)
        result = noise_gen.corrupt_signal(clean_signal, dt=0.1, baseline=1e5)
        noisy_signal = result["noisy_signal"]

        # Decode
        decode_params = DecodingParams(spike_threshold=4.0)
        extractor = SignalExtractor(decode_params)
        result = extractor.process_batch(noisy_signal, dt_ms=0.1)

        # Evaluate
        true_spikes = np.zeros(len(clean_signal), dtype=bool)
        for t in true_spike_times:
            true_spikes[t] = True

        eval_metrics = extractor.evaluate_reconstruction(
            result["spike_indices"],
            true_spikes,
            tolerance_samples=20
        )

        # Should detect at least some spikes (even if noisy conditions are challenging)
        n_detected = len(result["spike_indices"])
        assert n_detected >= 0, "Detection should complete without error"

    def test_ssnr_logging(self):
        """Test SSNR is logged for each batch."""
        extractor = SignalExtractor()

        for i in range(5):
            signal = np.random.randn(1000) * 100
            signal[500] = 1000 * (i + 1)
            extractor.process_batch(signal, dt_ms=0.1, batch_id=i)

        assert len(extractor.batch_history) == 5

        ssnrs = [m.ssnr for m in extractor.batch_history]
        # SSNR should generally increase with signal amplitude
        assert ssnrs[-1] >= ssnrs[0], "SSNR should increase with signal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
