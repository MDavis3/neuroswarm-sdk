"""
Signal extraction and spike decoding for Neuro-SWARM.

Implements:
1. Preprocessing (filtering, detrending, artifact removal)
2. PCA/ICA for source separation and artifact identification
3. Spike detection and reconstruction at 1 ms resolution
4. Per-batch SSNR logging

Reference:
    Hardy et al., IEEE Photonics Technology Letters, 2021.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import logging

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA, FastICA

logger = logging.getLogger(__name__)


@dataclass
class DecodingParams:
    """
    Configuration for signal decoding.

    Default values are chosen based on neurophysiology and signal processing
    best practices for extracellular spike detection.

    Attributes:
        highpass_freq: High-pass filter cutoff (Hz). Default 1 Hz removes DC drift
            while preserving spike waveforms (typical spike duration ~1-2 ms).
        lowpass_freq: Low-pass filter cutoff (Hz). Default 200 Hz captures spike
            frequency content while rejecting high-frequency detector noise.
        filter_order: Butterworth filter order. Default 4 gives 24 dB/octave rolloff
            with minimal phase distortion and no passband ripple.
        detrend_method: Detrending method ("linear", "constant", "none").
        artifact_threshold: Threshold for artifact detection (std deviations).
            Default 5σ gives P(false positive) ≈ 5.7×10⁻⁷ for Gaussian noise.
        artifact_interpolate: Interpolate over detected artifacts.
        pca_variance_threshold: Cumulative variance for PCA component selection.
            Default 0.95 retains components explaining 95% of variance.
        ica_n_components: Number of ICA components (None for auto).
        spike_threshold: Spike detection threshold (std deviations). Default 3σ
            gives ~0.13% false positive rate, suitable for typical SNR conditions.
        spike_refractory_ms: Minimum inter-spike interval (ms). Default 2 ms matches
            the absolute refractory period of neurons (Hodgkin & Huxley, 1952).
        detect_polarity: Which spike polarity to detect ("positive", "negative", "both").
        target_resolution_ms: Target temporal resolution (ms). Default 1 ms matches
            the integration time in Hardy et al. (2021).
        use_matched_filter: Enable matched filter spike detection.
        matched_filter_window_ms: Spike template window length (ms).
        matched_filter_template: Optional custom spike template for matched filter.
        use_wiener: Enable Wiener deconvolution for noise-optimal filtering.
        wiener_noise_floor: Noise floor added to PSD for numerical stability.
        wiener_highfreq_hz: High-frequency band for noise PSD estimation (Hz).
    """
    # Filter parameters
    highpass_freq: float = 1.0          # Hz - removes slow drift
    lowpass_freq: float = 200.0         # Hz - rejects HF noise
    filter_order: int = 4               # 24 dB/octave, minimal ringing
    detrend_method: str = "linear"
    # Artifact detection
    artifact_threshold: float = 5.0     # std - very conservative
    artifact_interpolate: bool = True
    # Dimensionality reduction
    pca_variance_threshold: float = 0.95
    ica_n_components: Optional[int] = None
    # Spike detection
    spike_threshold: float = 3.0        # std - ~0.13% FP rate
    spike_refractory_ms: float = 2.0    # ms - absolute refractory period
    detect_polarity: str = "positive"
    target_resolution_ms: float = 1.0   # ms - per Hardy et al. (2021)
    # Advanced options
    use_matched_filter: bool = False
    matched_filter_window_ms: float = 6.0
    matched_filter_template: Optional[NDArray[np.float64]] = None
    use_wiener: bool = False
    wiener_noise_floor: float = 1e-6
    wiener_highfreq_hz: float = 300.0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.highpass_freq < 0:
            raise ValueError(
                f"highpass_freq must be non-negative, got {self.highpass_freq}"
            )
        if self.lowpass_freq <= self.highpass_freq:
            raise ValueError(
                f"lowpass_freq ({self.lowpass_freq}) must be greater than "
                f"highpass_freq ({self.highpass_freq})"
            )
        if self.filter_order < 1:
            raise ValueError(
                f"filter_order must be >= 1, got {self.filter_order}"
            )
        if self.detrend_method not in ("linear", "constant", "none"):
            raise ValueError(
                f"detrend_method must be 'linear', 'constant', or 'none', "
                f"got '{self.detrend_method}'"
            )
        if self.artifact_threshold <= 0:
            raise ValueError(
                f"artifact_threshold must be positive, got {self.artifact_threshold}"
            )
        if self.spike_threshold <= 0:
            raise ValueError(
                f"spike_threshold must be positive, got {self.spike_threshold}"
            )
        if self.detect_polarity not in ("positive", "negative", "both"):
            raise ValueError(
                "detect_polarity must be 'positive', 'negative', or 'both', "
                f"got '{self.detect_polarity}'"
            )
        if self.matched_filter_window_ms <= 0:
            raise ValueError(
                f"matched_filter_window_ms must be positive, got {self.matched_filter_window_ms}"
            )
        if self.wiener_noise_floor <= 0:
            raise ValueError(
                f"wiener_noise_floor must be positive, got {self.wiener_noise_floor}"
            )
        if self.wiener_highfreq_hz <= 0:
            raise ValueError(
                f"wiener_highfreq_hz must be positive, got {self.wiener_highfreq_hz}"
            )


@dataclass
class BatchMetrics:
    """
    Metrics for a processed signal batch.

    Attributes:
        batch_id: Identifier for this batch
        n_samples: Number of samples in batch
        n_spikes_detected: Number of spikes detected
        ssnr: Signal-to-shot-noise ratio
        snr_db: Signal-to-noise ratio in dB
        artifact_fraction: Fraction of samples flagged as artifacts
        pca_components_used: Number of PCA components retained
        processing_time_ms: Time taken to process batch
    """
    batch_id: int = 0
    n_samples: int = 0
    n_spikes_detected: int = 0
    ssnr: float = 0.0
    snr_db: float = 0.0
    artifact_fraction: float = 0.0
    pca_components_used: int = 0
    processing_time_ms: float = 0.0


class MatchedFilterDetector:
    """
    Matched filter detector for spike waveforms.

    Uses a spike template to improve detection under heavy noise.
    """

    def __init__(self) -> None:
        self._template_cache: Dict[float, NDArray[np.float64]] = {}
        self._alignment_cache: Dict[float, int] = {}

    @staticmethod
    def _fit_template_length(
        template: NDArray[np.float64],
        target_len: int
    ) -> NDArray[np.float64]:
        if target_len <= 0:
            raise ValueError("target_len must be positive")
        if len(template) == target_len:
            return template
        if len(template) > target_len:
            start = (len(template) - target_len) // 2
            end = start + target_len
            return template[start:end]
        # Pad symmetrically
        pad_total = target_len - len(template)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(template, (pad_left, pad_right), mode="constant")

    @staticmethod
    def _compute_alignment_shift(template: NDArray[np.float64]) -> int:
        center = len(template) // 2
        peak_idx = int(np.argmax(np.abs(template)))
        # Shift detected indices to align to the template peak
        return peak_idx - center

    def set_template(
        self,
        dt_ms: float,
        window_ms: float,
        template: NDArray[np.float64]
    ) -> None:
        """Set a custom matched-filter template for the given window."""
        key = round(window_ms / dt_ms, 4)
        target_len = int(max(1, round(window_ms / dt_ms)))
        fitted = self._fit_template_length(template, target_len)
        fitted = fitted - np.mean(fitted)
        norm = np.linalg.norm(fitted) + 1e-12
        fitted = fitted / norm
        self._template_cache[key] = fitted
        self._alignment_cache[key] = self._compute_alignment_shift(fitted)

    def _generate_template(self, dt_ms: float, window_ms: float) -> NDArray[np.float64]:
        """Generate spike template using Izhikevich neuron dynamics."""
        from .physics import IzhikevichNeuron

        neuron = IzhikevichNeuron()
        n_steps = int(50.0 / dt_ms)
        I_input = np.zeros(n_steps)
        I_input[int(10.0 / dt_ms):int(20.0 / dt_ms)] = 10.0
        v_trace, spikes = neuron.simulate(I_input, dt_ms)

        spike_indices = np.where(spikes)[0]
        if len(spike_indices) == 0:
            # Fallback: use a simple Gaussian pulse
            t = np.linspace(-1, 1, int(window_ms / dt_ms))
            template = np.exp(-0.5 * (t / 0.2) ** 2)
        else:
            center = spike_indices[0]
            half_window = int((window_ms / dt_ms) / 2)
            start = max(0, center - half_window)
            end = min(len(v_trace), center + half_window)
            template = v_trace[start:end]

        # Normalize template
        template = template - np.mean(template)
        norm = np.linalg.norm(template) + 1e-12
        return template / norm

    def apply(
        self,
        signal: NDArray[np.float64],
        dt_ms: float,
        window_ms: float = 6.0
    ) -> Tuple[NDArray[np.float64], int]:
        """Apply matched filter and return correlation output with alignment shift."""
        key = round(window_ms / dt_ms, 4)
        if key not in self._template_cache:
            template = self._generate_template(dt_ms, window_ms)
            self._template_cache[key] = template
            self._alignment_cache[key] = self._compute_alignment_shift(template)

        template = self._template_cache[key]
        # Matched filter: correlate with reversed template
        output = np.convolve(signal, template[::-1], mode="same")
        shift = self._alignment_cache.get(key, 0)
        return output, shift


class WienerFilter:
    """
    Wiener deconvolution for noise-optimal linear filtering.
    """

    def apply(
        self,
        signal: NDArray[np.float64],
        sampling_rate_hz: float,
        noise_floor: float = 1e-6,
        highfreq_hz: float = 300.0
    ) -> NDArray[np.float64]:
        """
        Apply Wiener filter in frequency domain.

        Args:
            signal: Input signal
            sampling_rate_hz: Sampling rate in Hz
            noise_floor: Stabilization noise floor
            highfreq_hz: High frequency band for noise PSD estimation
        """
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
        signal_fft = np.fft.rfft(signal)
        psd_signal = np.abs(signal_fft) ** 2

        # Estimate noise PSD from high-frequency band
        hf_mask = freqs >= highfreq_hz
        if np.any(hf_mask):
            noise_psd = np.mean(psd_signal[hf_mask])
        else:
            noise_psd = np.mean(psd_signal[-10:])

        # Wiener filter H = S / (S + N)
        H = psd_signal / (psd_signal + noise_psd + noise_floor)
        filtered_fft = signal_fft * H
        return np.fft.irfft(filtered_fft, n=n)


class SignalExtractor:
    """
    Extract neural spike signals from noisy Neuro-SWARM data.

    Pipeline:
    1. Bandpass filtering to isolate spike frequency range
    2. Detrending to remove slow drift
    3. Artifact detection and removal
    4. PCA/ICA for source separation (multi-channel case)
    5. Spike detection via thresholding
    6. Spike reconstruction at target resolution

    Logs SSNR per batch for performance monitoring.
    """

    def __init__(self, params: Optional[DecodingParams] = None) -> None:
        """
        Initialize the signal extractor.

        Args:
            params: Decoding configuration. Uses defaults if None.
        """
        self.params = params or DecodingParams()
        self._batch_history: List[BatchMetrics] = []
        self._pca: Optional[PCA] = None
        self._ica: Optional[FastICA] = None
        self._matched_filter = MatchedFilterDetector()
        self._wiener_filter = WienerFilter()

        logger.info(
            f"SignalExtractor initialized: "
            f"bandpass=[{self.params.highpass_freq}, {self.params.lowpass_freq}] Hz, "
            f"spike_threshold={self.params.spike_threshold} std"
        )

    def reset(self) -> None:
        """Reset extractor state."""
        self._batch_history = []
        self._pca = None
        self._ica = None

    @property
    def batch_history(self) -> List[BatchMetrics]:
        """Get history of processed batches."""
        return self._batch_history

    def preprocess(
        self,
        signal: NDArray[np.float64],
        sampling_rate_hz: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """
        Preprocess signal: filter, detrend, detect artifacts.

        Args:
            signal: Raw input signal
            sampling_rate_hz: Sampling rate (Hz)

        Returns:
            Tuple of (preprocessed signal, artifact mask)
        """
        p = self.params

        # 1. Bandpass filter
        # Why Butterworth: Maximally flat passband (no ripple), good for preserving
        # spike waveform morphology. 4th order provides sharp cutoff (~24 dB/octave)
        # without excessive ringing that could distort spike detection.
        # Why 1-200 Hz: Captures action potential frequency content while rejecting
        # DC drift and high-frequency detector noise.
        nyquist = sampling_rate_hz / 2
        low = max(p.highpass_freq / nyquist, 0.001)
        high = min(p.lowpass_freq / nyquist, 0.999)

        if low < high:
            sos = sp_signal.butter(
                p.filter_order,
                [low, high],
                btype="bandpass",
                output="sos"  # Second-order sections for numerical stability
            )
            # sosfiltfilt: Zero-phase filtering (forward-backward) preserves spike timing
            filtered = sp_signal.sosfiltfilt(sos, signal)
        else:
            logger.warning("Invalid filter frequencies, skipping filtering")
            filtered = signal.copy()

        # 2. Detrend
        if p.detrend_method == "linear":
            filtered = sp_signal.detrend(filtered, type="linear")
        elif p.detrend_method == "constant":
            filtered = sp_signal.detrend(filtered, type="constant")

        # 3. Artifact detection
        artifact_mask = self._detect_artifacts(filtered)

        # 4. Artifact interpolation
        if p.artifact_interpolate and np.any(artifact_mask):
            filtered = self._interpolate_artifacts(filtered, artifact_mask)

        return filtered, artifact_mask

    def _detect_artifacts(
        self,
        signal: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        """
        Detect artifact samples based on amplitude threshold.

        Args:
            signal: Input signal

        Returns:
            Boolean mask (True = artifact)
        """
        # Robust statistics using Median Absolute Deviation (MAD)
        # Why MAD instead of std: Standard deviation is heavily influenced by outliers
        # (the very artifacts we're trying to detect). MAD is robust to up to 50%
        # contamination, making it ideal for artifact detection in neural signals.
        # Reference: Rousseeuw & Croux (1993), J. American Statistical Association
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        # 1.4826 is the consistency constant for Gaussian distributions:
        # For N(0,1), MAD ≈ 0.6745, so std ≈ 1.4826 * MAD
        std_estimate = 1.4826 * mad

        # Why 5σ threshold: Balances sensitivity vs false positives.
        # For Gaussian noise: P(|X| > 5σ) ≈ 5.7×10⁻⁷ (extremely rare)
        # Real artifacts (motion, EMG) are typically 10-100× larger than neural signals.
        threshold = self.params.artifact_threshold * std_estimate
        artifact_mask = np.abs(signal - median) > threshold

        # Expand mask slightly to capture artifact edges
        artifact_mask = np.convolve(
            artifact_mask.astype(float),
            np.ones(5) / 5,
            mode="same"
        ) > 0.1

        n_artifacts = np.sum(artifact_mask)
        if n_artifacts > 0:
            logger.debug(
                f"Detected {n_artifacts} artifact samples "
                f"({100 * n_artifacts / len(signal):.1f}%)"
            )

        return artifact_mask

    def _interpolate_artifacts(
        self,
        signal: NDArray[np.float64],
        artifact_mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """
        Interpolate over artifact regions.

        Args:
            signal: Input signal
            artifact_mask: Boolean mask of artifact locations

        Returns:
            Signal with artifacts interpolated
        """
        output = signal.copy()

        # Find artifact regions
        artifact_diff = np.diff(np.concatenate([[0], artifact_mask.astype(int), [0]]))
        starts = np.where(artifact_diff == 1)[0]
        ends = np.where(artifact_diff == -1)[0]

        for start, end in zip(starts, ends):
            if start > 0 and end < len(signal):
                # Linear interpolation
                output[start:end] = np.linspace(
                    output[max(0, start - 1)],
                    output[min(len(signal) - 1, end)],
                    end - start
                )

        return output

    def apply_pca(
        self,
        signals: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Apply PCA for dimensionality reduction.

        Useful for multi-channel recordings to identify principal modes.

        Args:
            signals: Input signals, shape (n_channels, n_samples) or (n_samples,)

        Returns:
            Tuple of (transformed signals, explained variance ratios)
        """
        # Handle single-channel case
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_channels, n_samples = signals.shape

        # Transpose for sklearn (samples x features)
        X = signals.T

        # Fit PCA
        self._pca = PCA()
        components = self._pca.fit_transform(X)

        # Select components by variance threshold
        cumvar = np.cumsum(self._pca.explained_variance_ratio_)
        n_components = np.searchsorted(cumvar, self.params.pca_variance_threshold) + 1
        n_components = min(n_components, n_channels)

        logger.debug(
            f"PCA: {n_components}/{n_channels} components explain "
            f"{cumvar[n_components - 1]:.1%} variance"
        )

        # Return selected components (transpose back)
        return components[:, :n_components].T, self._pca.explained_variance_ratio_

    def apply_ica(
        self,
        signals: NDArray[np.float64],
        n_components: Optional[int] = None
    ) -> NDArray[np.float64]:
        """
        Apply ICA for source separation.

        Separates mixed signals into independent components, useful for
        isolating neural signals from artifacts.

        Args:
            signals: Input signals, shape (n_channels, n_samples) or (n_samples,)
            n_components: Number of ICA components (None for auto)

        Returns:
            Independent components, shape (n_components, n_samples)
        """
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_channels, n_samples = signals.shape
        X = signals.T

        # Determine number of components
        if n_components is None:
            n_components = self.params.ica_n_components or n_channels

        n_components = min(n_components, n_channels)

        # Fit ICA
        self._ica = FastICA(
            n_components=n_components,
            random_state=42,
            max_iter=500
        )

        try:
            components = self._ica.fit_transform(X)
            logger.debug(f"ICA: extracted {n_components} independent components")
        except Exception as e:
            logger.warning(f"ICA failed: {e}. Returning original signals.")
            return signals[:n_components]

        return components.T

    def detect_spikes(
        self,
        signal: NDArray[np.float64],
        dt_ms: float
    ) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Detect spikes using threshold crossing.

        Args:
            signal: Preprocessed signal
            dt_ms: Time step (ms)

        Returns:
            Tuple of (spike indices, spike amplitudes)
        """
        p = self.params

        # Compute threshold using standard deviation
        # Why 3σ threshold (default): For Gaussian noise, P(X > 3σ) ≈ 0.13%
        # This gives ~1.3 false positives per 1000 samples, acceptable for most
        # neural recording scenarios. Increase to 4-5σ for higher specificity.
        if p.detect_polarity == "both":
            working = np.abs(signal)
        elif p.detect_polarity == "negative":
            working = -signal
        else:
            working = signal

        std = np.std(working)
        threshold = p.spike_threshold * std

        # Find threshold crossings (positive peaks in selected polarity)
        above_threshold = working > threshold

        # Find rising edges (0→1 transitions in thresholded signal)
        crossings = np.diff(above_threshold.astype(int))
        spike_starts = np.where(crossings == 1)[0]

        # Apply refractory period constraint
        # Why 2 ms: Matches the absolute refractory period of neurons.
        # During this period, voltage-gated Na+ channels are inactivated and
        # the neuron cannot fire again regardless of stimulus strength.
        # Reference: Hodgkin & Huxley (1952), J. Physiology
        refractory_samples = int(p.spike_refractory_ms / dt_ms)
        spike_indices = []
        spike_amplitudes = []

        last_spike = -refractory_samples

        for start in spike_starts:
            if start - last_spike < refractory_samples:
                continue

            # Find peak within window
            end = min(start + refractory_samples, len(signal))
            window = working[start:end]
            peak_offset = np.argmax(window)
            peak_idx = start + peak_offset

            spike_indices.append(peak_idx)
            spike_amplitudes.append(signal[peak_idx])
            last_spike = peak_idx

        spike_indices = np.array(spike_indices, dtype=np.int64)
        spike_amplitudes = np.array(spike_amplitudes)

        logger.debug(f"Detected {len(spike_indices)} spikes")

        return spike_indices, spike_amplitudes

    def reconstruct_spike_train(
        self,
        spike_indices: NDArray[np.int64],
        n_samples: int,
        dt_ms: float
    ) -> NDArray[np.float64]:
        """
        Reconstruct binary spike train at target resolution.

        Args:
            spike_indices: Indices of detected spikes
            n_samples: Total number of samples
            dt_ms: Original time step (ms)

        Returns:
            Binary spike train at target resolution
        """
        target_dt = self.params.target_resolution_ms
        compression_factor = int(target_dt / dt_ms)

        if compression_factor <= 1:
            # No compression needed
            spike_train = np.zeros(n_samples)
            spike_train[spike_indices] = 1.0
            return spike_train

        # Compute number of output samples
        n_output = n_samples // compression_factor

        # Convert spike indices to output resolution
        output_indices = spike_indices // compression_factor
        output_indices = output_indices[output_indices < n_output]

        spike_train = np.zeros(n_output)
        spike_train[output_indices] = 1.0

        return spike_train

    def compute_ssnr(
        self,
        signal: NDArray[np.float64],
        spike_indices: NDArray[np.int64],
        baseline_photons: float = 1e5
    ) -> float:
        """
        Compute signal-to-shot-noise ratio (SSNR).

        The SSNR is the fundamental detection limit for optical measurements,
        arising from the discrete nature of photon counting (Poisson statistics).

        Formula:
            SSNR = (ΔS / S₀) × √N_ph

        Where:
            ΔS = peak differential signal (photon counts at spike)
            S₀ = baseline signal level (photon counts)
            N_ph = total photon count (determines shot noise σ = √N_ph)

        Target: SSNR ~ 10³ per Hardy et al. (2021), achieved with:
            - 10³ nanoparticle probes
            - 10 mW/mm² illumination at 1050 nm
            - 1 ms integration time

        Args:
            signal: Processed signal (differential photon counts)
            spike_indices: Detected spike locations
            baseline_photons: Baseline photon count for shot noise calculation

        Returns:
            SSNR value (dimensionless). Higher is better; >100 typically required
            for reliable single-spike detection.
        """
        if len(spike_indices) == 0:
            return 0.0

        # Peak signal amplitude (at spikes)
        spike_amplitudes = np.abs(signal[spike_indices])
        peak_signal = np.max(spike_amplitudes)

        # Baseline (signal level away from spikes)
        mask = np.ones(len(signal), dtype=bool)
        for idx in spike_indices:
            start = max(0, idx - 10)
            end = min(len(signal), idx + 10)
            mask[start:end] = False

        if np.sum(mask) > 0:
            baseline = np.mean(np.abs(signal[mask]))
        else:
            baseline = np.mean(np.abs(signal))

        # SSNR calculation
        if baseline > 0:
            delta_s_over_s0 = peak_signal / baseline
        else:
            delta_s_over_s0 = peak_signal

        ssnr = delta_s_over_s0 * np.sqrt(baseline_photons)

        return ssnr

    def process_batch(
        self,
        signal: NDArray[np.float64],
        dt_ms: float,
        batch_id: Optional[int] = None,
        baseline_photons: float = 1e5
    ) -> Dict[str, Any]:
        """
        Process a single batch of signal data.

        Full pipeline: preprocess -> spike detection -> reconstruction -> metrics

        Args:
            signal: Raw input signal
            dt_ms: Time step (ms)
            batch_id: Optional batch identifier
            baseline_photons: Baseline photon count for SSNR calculation

        Returns:
            Dictionary containing:
                - preprocessed: Preprocessed signal
                - artifact_mask: Artifact locations
                - spike_indices: Detected spike indices
                - spike_amplitudes: Spike amplitudes
                - spike_train: Reconstructed spike train
                - metrics: BatchMetrics object
        """
        import time
        start_time = time.time()

        if batch_id is None:
            batch_id = len(self._batch_history)

        sampling_rate_hz = 1000.0 / dt_ms  # Convert ms to Hz

        # Preprocess
        preprocessed, artifact_mask = self.preprocess(signal, sampling_rate_hz)

        # Optional Wiener deconvolution
        if self.params.use_wiener:
            preprocessed = self._wiener_filter.apply(
                preprocessed,
                sampling_rate_hz,
                noise_floor=self.params.wiener_noise_floor,
                highfreq_hz=self.params.wiener_highfreq_hz
            )

        # Detect spikes (matched filter optional)
        if self.params.use_matched_filter:
            if self.params.matched_filter_template is not None:
                self._matched_filter.set_template(
                    dt_ms,
                    self.params.matched_filter_window_ms,
                    self.params.matched_filter_template
                )
            mf_output, shift_samples = self._matched_filter.apply(
                preprocessed,
                dt_ms,
                window_ms=self.params.matched_filter_window_ms
            )
            spike_indices, _ = self.detect_spikes(mf_output, dt_ms)
            # Align matched-filter detections to the template peak
            if shift_samples != 0 and len(spike_indices) > 0:
                spike_indices = spike_indices + shift_samples
                spike_indices = spike_indices[
                    (spike_indices >= 0) & (spike_indices < len(preprocessed))
                ]
            spike_amplitudes = preprocessed[spike_indices] if len(spike_indices) > 0 else np.array([])
        else:
            spike_indices, spike_amplitudes = self.detect_spikes(preprocessed, dt_ms)

        # Reconstruct spike train
        spike_train = self.reconstruct_spike_train(
            spike_indices, len(signal), dt_ms
        )

        # Compute metrics
        ssnr = self.compute_ssnr(preprocessed, spike_indices, baseline_photons)
        noise_std = np.std(preprocessed)
        signal_peak = np.max(np.abs(preprocessed))
        snr_db = 20 * np.log10(signal_peak / (noise_std + 1e-10))

        processing_time = (time.time() - start_time) * 1000  # ms

        metrics = BatchMetrics(
            batch_id=batch_id,
            n_samples=len(signal),
            n_spikes_detected=len(spike_indices),
            ssnr=ssnr,
            snr_db=snr_db,
            artifact_fraction=np.mean(artifact_mask),
            pca_components_used=0,  # Updated if PCA is used
            processing_time_ms=processing_time
        )

        self._batch_history.append(metrics)

        logger.info(
            f"Batch {batch_id}: {len(spike_indices)} spikes, "
            f"SSNR={ssnr:.2e}, SNR={snr_db:.1f}dB, "
            f"artifacts={100 * metrics.artifact_fraction:.1f}%"
        )

        return {
            "preprocessed": preprocessed,
            "artifact_mask": artifact_mask,
            "spike_indices": spike_indices,
            "spike_amplitudes": spike_amplitudes,
            "spike_train": spike_train,
            "metrics": metrics,
        }

    def evaluate_reconstruction(
        self,
        detected_spikes: NDArray[np.int64],
        true_spikes: NDArray[np.bool_],
        tolerance_samples: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate spike detection accuracy against ground truth.

        Args:
            detected_spikes: Indices of detected spikes
            true_spikes: Boolean array of true spike locations
            tolerance_samples: Tolerance for matching (samples)

        Returns:
            Dictionary with precision, recall, F1 score
        """
        true_indices = np.where(true_spikes)[0]

        # Match detected to true
        true_positives = 0
        matched_true = set()

        for det in detected_spikes:
            for true_idx in true_indices:
                if abs(det - true_idx) <= tolerance_samples and true_idx not in matched_true:
                    true_positives += 1
                    matched_true.add(true_idx)
                    break

        false_positives = len(detected_spikes) - true_positives
        false_negatives = len(true_indices) - true_positives

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics across all processed batches.

        Returns:
            Dictionary with aggregate performance metrics
        """
        if not self._batch_history:
            return {"error": "No batches processed yet"}

        ssnrs = [m.ssnr for m in self._batch_history]
        snrs = [m.snr_db for m in self._batch_history]
        spikes = [m.n_spikes_detected for m in self._batch_history]
        times = [m.processing_time_ms for m in self._batch_history]

        return {
            "n_batches": len(self._batch_history),
            "total_spikes": sum(spikes),
            "ssnr_mean": np.mean(ssnrs),
            "ssnr_std": np.std(ssnrs),
            "snr_db_mean": np.mean(snrs),
            "snr_db_std": np.std(snrs),
            "avg_processing_time_ms": np.mean(times),
            "total_processing_time_ms": sum(times),
        }
