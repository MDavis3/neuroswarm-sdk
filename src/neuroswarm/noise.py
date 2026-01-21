"""
Adversarial noise generator for Neuro-SWARM signal robustness testing.

Implements realistic noise sources:
1. Shot noise (Poisson statistics - fundamental limit)
2. Detector/thermal noise (Gaussian)
3. Low-frequency drift (physiological artifacts)
4. Burst artifacts (motion, EMG)
5. Multiplicative intensity fluctuations

All operations are numpy-vectorized for efficiency.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class NoiseParams:
    """
    Configuration for noise generation.

    Attributes:
        shot_noise_enabled: Enable Poisson shot noise (fundamental limit)
        thermal_noise_std: Gaussian thermal/detector noise std (photon counts)
        drift_amplitude: Amplitude of low-frequency drift (fraction of signal)
        drift_period_ms: Period of drift oscillation (ms)
        burst_probability: Probability of burst artifact per time step
        burst_amplitude: Amplitude of burst artifacts (photon counts)
        burst_duration_ms: Duration of burst artifacts (ms)
        intensity_fluctuation_std: Std of multiplicative intensity noise (fraction)
        seed: Random seed for reproducibility (None for random)
    """
    shot_noise_enabled: bool = True
    thermal_noise_std: float = 100.0      # photon counts
    drift_amplitude: float = 0.05         # 5% of signal
    drift_period_ms: float = 500.0        # ms
    burst_probability: float = 0.001      # per time step
    burst_amplitude: float = 5000.0       # photon counts
    burst_duration_ms: float = 5.0        # ms
    intensity_fluctuation_std: float = 0.02  # 2%
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.thermal_noise_std < 0:
            raise ValueError(
                f"thermal_noise_std must be non-negative, got {self.thermal_noise_std}"
            )
        if self.drift_amplitude < 0:
            raise ValueError(
                f"drift_amplitude must be non-negative, got {self.drift_amplitude}"
            )
        if not (0 <= self.burst_probability <= 1):
            raise ValueError(
                f"burst_probability must be in [0, 1], got {self.burst_probability}"
            )
        if self.intensity_fluctuation_std < 0:
            raise ValueError(
                f"intensity_fluctuation_std must be non-negative, got {self.intensity_fluctuation_std}"
            )


class AdversarialNoiseGenerator:
    """
    Generate realistic noise for stress-testing signal extraction algorithms.

    The fundamental detection limit is shot noise:
        SSNR = (ΔS/S₀) * sqrt(N_ph)

    This generator adds multiple noise sources to test robustness beyond
    the shot-noise limit.
    """

    def __init__(self, params: Optional[NoiseParams] = None) -> None:
        """
        Initialize the noise generator.

        Args:
            params: Noise configuration. Uses defaults if None.
        """
        self.params = params or NoiseParams()
        self._rng = np.random.default_rng(self.params.seed)

        logger.info(
            f"AdversarialNoiseGenerator initialized: "
            f"shot={self.params.shot_noise_enabled}, "
            f"thermal_std={self.params.thermal_noise_std}, "
            f"drift={self.params.drift_amplitude}"
        )

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
        self.params.seed = seed

    def add_shot_noise(
        self,
        signal: NDArray[np.float64],
        baseline: float = 0.0
    ) -> NDArray[np.float64]:
        """
        Add Poisson shot noise to signal.

        Shot noise arises from the discrete nature of photon counting.
        Variance equals the mean: σ² = N_ph

        Args:
            signal: Clean signal (differential photon counts)
            baseline: Baseline photon count to add before noise

        Returns:
            Signal with shot noise
        """
        # Total photon count (baseline + differential signal)
        total_counts = baseline + signal

        # Handle negative values (they shouldn't occur but be safe)
        total_counts = np.maximum(total_counts, 0)

        # Poisson sampling
        noisy_counts = self._rng.poisson(total_counts).astype(np.float64)

        # Return differential (subtract baseline)
        return noisy_counts - baseline

    def add_thermal_noise(
        self,
        signal: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Add Gaussian thermal/detector noise.

        This represents electronic noise in the detector, dark current, etc.

        Args:
            signal: Input signal

        Returns:
            Signal with added Gaussian noise
        """
        noise = self._rng.normal(0, self.params.thermal_noise_std, signal.shape)
        return signal + noise

    def add_drift(
        self,
        signal: NDArray[np.float64],
        dt: float,
        baseline: float = 1.0
    ) -> NDArray[np.float64]:
        """
        Add low-frequency drift (physiological artifacts).

        Models slow variations from:
        - Blood flow changes
        - Temperature fluctuations
        - Slow tissue movement

        Args:
            signal: Input signal
            dt: Time step (ms)
            baseline: Signal baseline for scaling drift amplitude

        Returns:
            Signal with drift
        """
        n_samples = len(signal)
        time_ms = np.arange(n_samples) * dt

        # Multiple drift frequencies for realism
        drift = np.zeros(n_samples)

        # Primary drift frequency
        f1 = 1.0 / self.params.drift_period_ms  # Hz (in ms^-1)
        phase1 = self._rng.uniform(0, 2 * np.pi)
        drift += np.sin(2 * np.pi * f1 * time_ms + phase1)

        # Secondary (slower) drift
        f2 = f1 / 3.0
        phase2 = self._rng.uniform(0, 2 * np.pi)
        drift += 0.5 * np.sin(2 * np.pi * f2 * time_ms + phase2)

        # Add some random walk component
        random_walk = np.cumsum(self._rng.normal(0, 0.01, n_samples))
        random_walk = random_walk - np.mean(random_walk)  # Zero mean
        random_walk = random_walk / (np.std(random_walk) + 1e-10)  # Normalize
        drift += 0.3 * random_walk

        # Scale to desired amplitude
        drift = drift / (np.max(np.abs(drift)) + 1e-10)
        drift = drift * self.params.drift_amplitude * baseline

        return signal + drift

    def add_burst_artifacts(
        self,
        signal: NDArray[np.float64],
        dt: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Add random burst artifacts (motion, EMG, etc.).

        These are sudden, short-duration signal excursions that can
        mask true spikes.

        Args:
            signal: Input signal
            dt: Time step (ms)

        Returns:
            Tuple of (corrupted signal, artifact mask)
        """
        n_samples = len(signal)
        burst_samples = max(1, int(self.params.burst_duration_ms / dt))

        # Determine burst locations
        burst_starts = self._rng.random(n_samples) < self.params.burst_probability
        artifact_mask = np.zeros(n_samples, dtype=bool)

        output = signal.copy()

        for i in np.where(burst_starts)[0]:
            end_idx = min(i + burst_samples, n_samples)

            # Random burst shape (sharp rise, exponential decay)
            burst_len = end_idx - i
            t = np.arange(burst_len) / burst_len
            burst_shape = np.exp(-3 * t) * (1 - np.exp(-10 * t))

            # Random sign and amplitude variation
            sign = self._rng.choice([-1, 1])
            amplitude = self.params.burst_amplitude * self._rng.uniform(0.5, 1.5)

            output[i:end_idx] += sign * amplitude * burst_shape
            artifact_mask[i:end_idx] = True

        n_bursts = np.sum(burst_starts)
        if n_bursts > 0:
            logger.debug(f"Added {n_bursts} burst artifacts")

        return output, artifact_mask

    def add_intensity_fluctuations(
        self,
        signal: NDArray[np.float64],
        dt: float
    ) -> NDArray[np.float64]:
        """
        Add multiplicative intensity fluctuations.

        Models variations in illumination intensity, which cause
        multiplicative (not additive) signal changes.

        Args:
            signal: Input signal
            dt: Time step (ms)

        Returns:
            Signal with intensity fluctuations
        """
        n_samples = len(signal)

        # Correlated intensity variations (not white noise)
        # Use smoothed random process
        raw_noise = self._rng.normal(0, 1, n_samples)

        # Smooth with exponential moving average
        alpha = 0.1  # Smoothing factor
        smoothed = np.zeros(n_samples)
        smoothed[0] = raw_noise[0]
        for i in range(1, n_samples):
            smoothed[i] = alpha * raw_noise[i] + (1 - alpha) * smoothed[i - 1]

        # Scale to desired std
        smoothed = smoothed / (np.std(smoothed) + 1e-10)
        intensity_factor = 1.0 + self.params.intensity_fluctuation_std * smoothed

        return signal * intensity_factor

    def corrupt_signal(
        self,
        signal: NDArray[np.float64],
        dt: float,
        baseline: float = 1e5
    ) -> dict:
        """
        Apply all noise sources to a clean signal.

        Args:
            signal: Clean differential photon count signal
            dt: Time step (ms)
            baseline: Baseline photon count for shot noise calculation

        Returns:
            Dictionary containing:
                - noisy_signal: Fully corrupted signal
                - shot_noise_only: Signal with only shot noise
                - artifact_mask: Boolean mask of burst artifact locations
                - noise_components: Dictionary of individual noise contributions
        """
        logger.debug("Corrupting signal with adversarial noise...")

        # Track intermediate signals
        components = {}

        # Start with clean signal
        working = signal.copy()

        # 1. Intensity fluctuations (multiplicative, applied first)
        working = self.add_intensity_fluctuations(working, dt)
        components["intensity_fluctuation"] = working - signal

        # 2. Low-frequency drift
        pre_drift = working.copy()
        working = self.add_drift(working, dt, baseline)
        components["drift"] = working - pre_drift

        # 3. Burst artifacts
        pre_burst = working.copy()
        working, artifact_mask = self.add_burst_artifacts(working, dt)
        components["bursts"] = working - pre_burst

        # 4. Thermal noise
        pre_thermal = working.copy()
        working = self.add_thermal_noise(working)
        components["thermal"] = working - pre_thermal

        # 5. Shot noise (applied last, fundamental)
        shot_noise_only = signal.copy()
        if self.params.shot_noise_enabled:
            pre_shot = working.copy()
            working = self.add_shot_noise(working, baseline)
            components["shot"] = working - pre_shot

            # Also compute shot-noise-only version for comparison
            shot_noise_only = self.add_shot_noise(signal, baseline)

        logger.info(
            f"Signal corrupted: "
            f"original range [{signal.min():.2e}, {signal.max():.2e}], "
            f"noisy range [{working.min():.2e}, {working.max():.2e}]"
        )

        return {
            "noisy_signal": working,
            "shot_noise_only": shot_noise_only,
            "artifact_mask": artifact_mask,
            "noise_components": components,
        }

    def estimate_snr(
        self,
        clean_signal: NDArray[np.float64],
        noisy_signal: NDArray[np.float64]
    ) -> dict:
        """
        Estimate signal-to-noise ratio metrics.

        Args:
            clean_signal: Original clean signal
            noisy_signal: Corrupted signal

        Returns:
            Dictionary with SNR metrics
        """
        noise = noisy_signal - clean_signal

        # RMS SNR
        signal_rms = np.sqrt(np.mean(clean_signal ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        snr_rms = signal_rms / (noise_rms + 1e-10)

        # Peak SNR
        signal_peak = np.max(np.abs(clean_signal))
        snr_peak = signal_peak / (noise_rms + 1e-10)

        # Shot-noise-limited SNR (theoretical)
        # SSNR = peak_signal / sqrt(baseline)
        # Assuming baseline is around the magnitude of the signal
        baseline_estimate = np.mean(np.abs(clean_signal)) + np.abs(np.median(clean_signal))
        ssnr_theoretical = signal_peak / (np.sqrt(baseline_estimate) + 1e-10)

        return {
            "snr_rms": snr_rms,
            "snr_peak": snr_peak,
            "snr_db": 20 * np.log10(snr_rms + 1e-10),
            "ssnr_theoretical": ssnr_theoretical,
            "noise_std": np.std(noise),
        }


def generate_noise_sweep(
    signal: NDArray[np.float64],
    dt: float,
    noise_levels: List[float],
    baseline: float = 1e5,
    seed: int = 42
) -> List[dict]:
    """
    Generate multiple noisy versions at different intensity levels.

    Useful for characterizing decoder robustness across noise conditions.

    Args:
        signal: Clean signal
        dt: Time step (ms)
        noise_levels: List of thermal noise std values to test
        baseline: Baseline photon count
        seed: Random seed

    Returns:
        List of corruption results, one per noise level
    """
    results = []

    for i, noise_std in enumerate(noise_levels):
        params = NoiseParams(
            thermal_noise_std=noise_std,
            seed=seed + i
        )
        generator = AdversarialNoiseGenerator(params)
        result = generator.corrupt_signal(signal, dt, baseline)
        result["noise_level"] = noise_std
        results.append(result)

    return results
