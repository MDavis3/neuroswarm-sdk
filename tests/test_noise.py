"""
Tests for the noise module.

Validates:
1. Shot noise statistics (Poisson variance equals mean)
2. Drift has correct amplitude and frequency characteristics
3. Burst artifacts occur with correct probability
4. Full corruption pipeline maintains signal integrity
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuroswarm.noise import (
    NoiseParams,
    AdversarialNoiseGenerator,
    generate_noise_sweep,
)


class TestNoiseParams:
    """Tests for NoiseParams configuration."""

    def test_default_initialization(self):
        """Test default parameters are valid."""
        params = NoiseParams()
        assert params.shot_noise_enabled == True
        assert params.thermal_noise_std == 100.0
        assert params.drift_amplitude == 0.05
        assert params.burst_probability == 0.001

    def test_validation_thermal_noise(self):
        """Test negative thermal noise raises error."""
        with pytest.raises(ValueError, match="thermal_noise_std"):
            NoiseParams(thermal_noise_std=-1.0)

    def test_validation_drift_amplitude(self):
        """Test negative drift amplitude raises error."""
        with pytest.raises(ValueError, match="drift_amplitude"):
            NoiseParams(drift_amplitude=-0.1)

    def test_validation_burst_probability(self):
        """Test invalid burst probability raises error."""
        with pytest.raises(ValueError, match="burst_probability"):
            NoiseParams(burst_probability=1.5)

    def test_custom_parameters(self):
        """Test custom parameters are accepted."""
        params = NoiseParams(
            thermal_noise_std=500.0,
            drift_amplitude=0.1,
            seed=12345
        )
        assert params.thermal_noise_std == 500.0
        assert params.drift_amplitude == 0.1
        assert params.seed == 12345


class TestShotNoise:
    """Tests for shot noise generation."""

    def test_shot_noise_shape(self):
        """Test shot noise preserves signal shape."""
        generator = AdversarialNoiseGenerator(NoiseParams(seed=42))
        signal = np.ones(1000) * 100

        noisy = generator.add_shot_noise(signal, baseline=1000)

        assert noisy.shape == signal.shape

    def test_shot_noise_statistics(self):
        """Test shot noise has Poisson statistics."""
        generator = AdversarialNoiseGenerator(NoiseParams(seed=42))

        baseline = 10000.0
        n_trials = 1000
        samples = np.zeros(n_trials)

        for i in range(n_trials):
            generator.set_seed(i)
            signal = np.array([0.0])
            noisy = generator.add_shot_noise(signal, baseline=baseline)
            samples[i] = noisy[0] + baseline

        measured_mean = np.mean(samples)
        measured_var = np.var(samples)

        assert abs(measured_var - measured_mean) / measured_mean < 0.15, (
            f"Shot noise variance ({measured_var:.1f}) should equal "
            f"mean ({measured_mean:.1f}) for Poisson statistics"
        )

    def test_shot_noise_reproducible(self):
        """Test shot noise is reproducible with seed."""
        params = NoiseParams(seed=42)
        gen1 = AdversarialNoiseGenerator(params)
        gen2 = AdversarialNoiseGenerator(params)

        signal = np.ones(100) * 50

        noisy1 = gen1.add_shot_noise(signal, baseline=1000)
        noisy2 = gen2.add_shot_noise(signal, baseline=1000)

        np.testing.assert_array_equal(noisy1, noisy2)


class TestThermalNoise:
    """Tests for thermal/Gaussian noise generation."""

    def test_thermal_noise_mean(self):
        """Test thermal noise has approximately zero mean."""
        generator = AdversarialNoiseGenerator(NoiseParams(seed=42))
        signal = np.zeros(10000)

        noisy = generator.add_thermal_noise(signal)

        assert abs(np.mean(noisy)) < 10, "Thermal noise should have ~zero mean"

    def test_thermal_noise_std(self):
        """Test thermal noise has correct standard deviation."""
        expected_std = 200.0
        params = NoiseParams(thermal_noise_std=expected_std, seed=42)
        generator = AdversarialNoiseGenerator(params)

        signal = np.zeros(10000)
        noisy = generator.add_thermal_noise(signal)

        measured_std = np.std(noisy)
        assert abs(measured_std - expected_std) / expected_std < 0.1, (
            f"Thermal noise std ({measured_std:.1f}) should match "
            f"expected ({expected_std:.1f})"
        )


class TestDrift:
    """Tests for low-frequency drift generation."""

    def test_drift_amplitude(self):
        """Test drift has approximately correct amplitude."""
        drift_amp = 0.1
        baseline = 10000.0
        params = NoiseParams(drift_amplitude=drift_amp, seed=42)
        generator = AdversarialNoiseGenerator(params)

        signal = np.zeros(10000)
        drifted = generator.add_drift(signal, dt=0.1, baseline=baseline)

        max_drift = np.max(np.abs(drifted))
        expected_max = drift_amp * baseline

        assert max_drift < 2.0 * expected_max, (
            f"Drift amplitude ({max_drift:.1f}) exceeds expected ({expected_max:.1f})"
        )

    def test_drift_is_low_frequency(self):
        """Test drift has low-frequency characteristics."""
        params = NoiseParams(drift_period_ms=500.0, seed=42)
        generator = AdversarialNoiseGenerator(params)

        signal = np.zeros(5000)
        drifted = generator.add_drift(signal, dt=0.1, baseline=1000)

        fft = np.fft.rfft(drifted)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(drifted), d=0.0001)

        low_freq_mask = freqs < 10
        high_freq_mask = freqs >= 10

        low_power = np.sum(power[low_freq_mask])
        high_power = np.sum(power[high_freq_mask])

        assert low_power > high_power, "Drift should be low-frequency"

    def test_drift_zero_amplitude(self):
        """Test zero drift amplitude produces no drift."""
        params = NoiseParams(drift_amplitude=0.0, seed=42)
        generator = AdversarialNoiseGenerator(params)

        signal = np.ones(1000) * 100
        drifted = generator.add_drift(signal, dt=0.1, baseline=1000)

        np.testing.assert_array_equal(signal, drifted)


class TestBurstArtifacts:
    """Tests for burst artifact generation."""

    def test_burst_artifact_probability(self):
        """Test burst artifacts occur with approximately correct probability."""
        prob = 0.01
        params = NoiseParams(burst_probability=prob, seed=42)
        generator = AdversarialNoiseGenerator(params)

        signal = np.zeros(10000)
        _, artifact_mask = generator.add_burst_artifacts(signal, dt=0.1)

        artifact_diff = np.diff(artifact_mask.astype(int))
        n_bursts = np.sum(artifact_diff == 1)

        expected_bursts = prob * len(signal)

        assert n_bursts > 0, "Should have at least some bursts"
        assert 0.3 * expected_bursts < n_bursts < 3.0 * expected_bursts, (
            f"Number of bursts ({n_bursts}) outside expected range"
        )

    def test_no_bursts_zero_probability(self):
        """Test zero probability produces no bursts."""
        params = NoiseParams(burst_probability=0.0, seed=42)
        generator = AdversarialNoiseGenerator(params)

        signal = np.ones(1000) * 100
        corrupted, mask = generator.add_burst_artifacts(signal, dt=0.1)

        np.testing.assert_array_equal(signal, corrupted)
        assert not np.any(mask)


class TestCorruptSignal:
    """Tests for complete signal corruption pipeline."""

    def test_corrupt_signal_returns_all_outputs(self):
        """Test corrupt_signal returns all expected outputs."""
        generator = AdversarialNoiseGenerator(NoiseParams(seed=42))
        signal = np.ones(1000) * 1000

        result = generator.corrupt_signal(signal, dt=0.1, baseline=1e5)

        assert "noisy_signal" in result
        assert "shot_noise_only" in result
        assert "artifact_mask" in result
        assert "noise_components" in result

        assert result["noisy_signal"].shape == signal.shape
        assert result["shot_noise_only"].shape == signal.shape
        assert result["artifact_mask"].shape == signal.shape

    def test_corrupt_signal_adds_noise(self):
        """Test corrupt_signal actually adds noise."""
        generator = AdversarialNoiseGenerator(NoiseParams(seed=42))
        signal = np.ones(1000) * 1000

        result = generator.corrupt_signal(signal, dt=0.1, baseline=1e5)

        assert not np.allclose(result["noisy_signal"], signal)

    def test_corrupt_signal_noise_components(self):
        """Test noise components are tracked correctly."""
        generator = AdversarialNoiseGenerator(NoiseParams(seed=42))
        signal = np.ones(1000) * 1000

        result = generator.corrupt_signal(signal, dt=0.1, baseline=1e5)
        components = result["noise_components"]

        assert "intensity_fluctuation" in components
        assert "drift" in components
        assert "bursts" in components
        assert "thermal" in components
        assert "shot" in components


class TestSNREstimation:
    """Tests for SNR estimation utilities."""

    def test_estimate_snr_known_values(self):
        """Test SNR estimation with known signal and noise."""
        generator = AdversarialNoiseGenerator(NoiseParams(seed=42))

        clean = np.zeros(1000)
        clean[400:500] = 1000

        noise = np.random.randn(1000) * 100
        noisy = clean + noise

        snr_metrics = generator.estimate_snr(clean, noisy)

        assert "snr_rms" in snr_metrics
        assert "snr_peak" in snr_metrics
        assert "snr_db" in snr_metrics
        assert "noise_std" in snr_metrics

        assert snr_metrics["snr_peak"] > 5


class TestNoiseSweep:
    """Tests for noise level sweep utility."""

    def test_generate_noise_sweep(self):
        """Test noise sweep generates multiple noise levels."""
        signal = np.ones(500) * 1000
        noise_levels = [10, 50, 100, 500]

        results = generate_noise_sweep(
            signal,
            dt=0.1,
            noise_levels=noise_levels,
            baseline=1e5,
            seed=42
        )

        assert len(results) == len(noise_levels)

        for i, result in enumerate(results):
            assert result["noise_level"] == noise_levels[i]
            assert "noisy_signal" in result

    def test_noise_sweep_increasing_noise(self):
        """Test that noise sweep generates different signals at each level."""
        signal = np.ones(1000) * 1000
        noise_levels = [10, 100, 500]

        results = generate_noise_sweep(
            signal,
            dt=0.1,
            noise_levels=noise_levels,
            baseline=1e5,
            seed=42
        )

        # Each noise level should produce a different noisy signal
        for i, result in enumerate(results):
            assert not np.allclose(result["noisy_signal"], signal), (
                f"Noise level {noise_levels[i]} should add noise"
            )


class TestIntegration:
    """Integration tests for noise module."""

    def test_signal_preservation(self):
        """Test that signal structure is preserved under low noise."""
        params = NoiseParams(
            thermal_noise_std=10.0,
            drift_amplitude=0.01,
            burst_probability=0.0,
            intensity_fluctuation_std=0.01,
            seed=42
        )
        generator = AdversarialNoiseGenerator(params)

        signal = np.zeros(1000)
        signal[400:500] = 10000

        result = generator.corrupt_signal(signal, dt=0.1, baseline=1e5)

        noisy = result["noisy_signal"]
        spike_region = noisy[400:500]
        background = np.concatenate([noisy[:350], noisy[550:]])

        assert np.mean(spike_region) > np.mean(background) + 3 * np.std(background)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
