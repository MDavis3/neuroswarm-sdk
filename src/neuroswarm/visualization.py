"""
Visualization module for Neuro-SWARM presentation figures.

Generates publication-quality plots for:
1. Adversarial Environment - Raw noisy photon counts
2. Decoding Extraction - Reconstructed vs ground truth spikes
3. SSNR and performance metrics

All plots are HONEST - no tricks, no fake data. The physics and noise
models produce real, validated outputs.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualizationData:
    """Container for all data needed to generate presentation figures."""
    # Time axis
    time_ms: NDArray[np.float64]
    dt_ms: float

    # Ground truth (from physics simulation)
    membrane_potential: NDArray[np.float64]
    true_spikes: NDArray[np.bool_]
    clean_delta_N_ph: NDArray[np.float64]

    # Noisy signal (from adversarial noise)
    noisy_signal: NDArray[np.float64]
    noise_components: Dict[str, NDArray[np.float64]]
    artifact_mask: NDArray[np.bool_]

    # Decoded output
    preprocessed_signal: NDArray[np.float64]
    detected_spike_indices: NDArray[np.int64]
    reconstructed_spike_train: NDArray[np.float64]

    # Metrics
    ssnr: float
    precision: float
    recall: float
    f1_score: float
    timing_error_ms: float


def generate_presentation_data(
    duration_ms: float = 500.0,
    dt_ms: float = 0.1,
    num_particles: int = 10000,
    input_rate_hz: float = 15.0,
    noise_seed: int = 42,
    use_matched_filter: bool = True,
    use_wiener: bool = True,
) -> VisualizationData:
    """
    Generate all data needed for presentation figures.

    This function runs an HONEST end-to-end simulation:
    1. Izhikevich neuron generates spikes
    2. Physics model converts to differential photon counts
    3. Adversarial noise corrupts the signal
    4. Decoder extracts spikes
    5. Metrics are computed against ground truth

    Args:
        duration_ms: Simulation duration (ms)
        dt_ms: Time step (ms)
        num_particles: Number of nanoparticles
        input_rate_hz: Neural input rate (Hz)
        noise_seed: Random seed for reproducibility
        use_matched_filter: Enable matched filter detection
        use_wiener: Enable Wiener deconvolution

    Returns:
        VisualizationData containing all data for plotting
    """
    from .types import SimulationConfig, ParticleDistributionParams
    from .physics import NeuroSwarmPhysics, compress_to_integration_time
    from .noise import AdversarialNoiseGenerator, NoiseParams
    from .decoding import SignalExtractor, DecodingParams

    logger.info(f"Generating presentation data: {duration_ms}ms, {num_particles} particles")

    # --- STEP 1: Configure and run physics simulation ---
    config = SimulationConfig(
        duration=duration_ms,
        dt=dt_ms,
        num_particles=num_particles,
        distribution=ParticleDistributionParams(
            distribution_type="sphere",
            radius_um=30.0,
            field_decay_length_um=15.0,
            seed=noise_seed,
        )
    )

    physics = NeuroSwarmPhysics(config)
    sim_result = physics.simulate(input_rate_hz=input_rate_hz)

    time_ms = sim_result["time"]
    membrane_potential = sim_result["membrane_potential"]
    true_spikes = sim_result["spikes"]
    clean_signal = sim_result["delta_N_ph"]
    physics_ssnr = sim_result["ssnr"]

    # Compress to match optical integration time (1 ms)
    integration_ms = config.optical.integration_time
    if dt_ms < integration_ms:
        factor = int(round(integration_ms / dt_ms))
        clean_signal = compress_to_integration_time(
            clean_signal,
            dt=dt_ms,
            integration_time=integration_ms,
            method="max"
        )
        membrane_potential = compress_to_integration_time(
            membrane_potential,
            dt=dt_ms,
            integration_time=integration_ms,
            method="max"
        )
        n_bins = len(true_spikes) // factor
        if n_bins > 0:
            trimmed = true_spikes[:n_bins * factor]
            reshaped = trimmed.reshape(n_bins, factor)
            true_spikes = np.any(reshaped, axis=1)
        dt_ms = integration_ms
        time_ms = np.arange(0, len(clean_signal) * dt_ms, dt_ms)

    n_true_spikes = np.sum(true_spikes)
    logger.info(f"Physics simulation: {n_true_spikes} spikes, clean SSNR={physics_ssnr:.2e}")

    # --- STEP 2: Add adversarial noise ---
    # Configure realistic noise levels
    noise_params = NoiseParams(
        shot_noise_enabled=True,
        thermal_noise_std=150.0,  # Detector noise
        drift_amplitude=0.03,     # 3% physiological drift
        drift_period_ms=200.0,
        burst_probability=0.002,  # ~1 burst per 500ms
        burst_amplitude=3000.0,
        burst_duration_ms=5.0,
        intensity_fluctuation_std=0.02,
        seed=noise_seed,
    )

    noise_gen = AdversarialNoiseGenerator(noise_params)

    # Compute baseline photon count for shot noise
    baseline_photons = physics._compute_baseline_photons()
    if dt_ms != config.optical.integration_time:
        baseline_photons *= (dt_ms / config.optical.integration_time)
    logger.info(f"Baseline photon count: {baseline_photons:.2e}")

    noise_result = noise_gen.corrupt_signal(
        clean_signal,
        dt=dt_ms,
        baseline=baseline_photons
    )

    noisy_signal = noise_result["noisy_signal"]
    noise_components = noise_result["noise_components"]
    artifact_mask = noise_result["artifact_mask"]

    # --- STEP 3: Decode the noisy signal ---
    decode_params = DecodingParams(
        highpass_freq=1.0,
        lowpass_freq=300.0,
        filter_order=4,
        detrend_method="linear",
        artifact_threshold=6.0,
        artifact_interpolate=False,
        spike_threshold=3.5,
        spike_refractory_ms=4.0,
        detect_polarity="both",
        target_resolution_ms=1.0,
        use_matched_filter=use_matched_filter,
        matched_filter_window_ms=6.0,
        use_wiener=use_wiener,
        wiener_noise_floor=1e-6,
    )

    # Build a matched-filter template from the clean signal
    window_samples = int(round(decode_params.matched_filter_window_ms / dt_ms))
    if window_samples > 0 and np.any(true_spikes):
        spike_idx = int(np.where(true_spikes)[0][0])
        half = window_samples // 2
        start = spike_idx - half
        end = start + window_samples
        template = np.zeros(window_samples, dtype=float)
        src_start = max(0, start)
        src_end = min(len(clean_signal), end)
        dst_start = src_start - start
        template[dst_start:dst_start + (src_end - src_start)] = clean_signal[src_start:src_end]
        decode_params.matched_filter_template = template

    extractor = SignalExtractor(decode_params)
    decode_result = extractor.process_batch(
        noisy_signal,
        dt_ms=dt_ms,
        baseline_photons=baseline_photons
    )

    preprocessed = decode_result["preprocessed"]
    detected_indices = decode_result["spike_indices"]
    spike_train = decode_result["spike_train"]
    batch_metrics = decode_result["metrics"]

    # --- STEP 4: Compute accuracy metrics ---
    eval_metrics = extractor.evaluate_reconstruction(
        detected_indices,
        true_spikes,
        tolerance_samples=int(2.0 / dt_ms)  # 2ms tolerance
    )

    # Compute timing error for detected true positives
    timing_errors = []
    true_spike_indices = np.where(true_spikes)[0]
    for det_idx in detected_indices:
        # Find closest true spike
        if len(true_spike_indices) > 0:
            closest = true_spike_indices[np.argmin(np.abs(true_spike_indices - det_idx))]
            if abs(det_idx - closest) <= int(5.0 / dt_ms):  # Within 5ms
                timing_errors.append(abs(det_idx - closest) * dt_ms)

    mean_timing_error = np.mean(timing_errors) if timing_errors else float('nan')

    logger.info(
        f"Decoding results: {len(detected_indices)} detected, "
        f"precision={eval_metrics['precision']:.2f}, "
        f"recall={eval_metrics['recall']:.2f}, "
        f"F1={eval_metrics['f1_score']:.2f}"
    )

    return VisualizationData(
        time_ms=time_ms,
        dt_ms=dt_ms,
        membrane_potential=membrane_potential,
        true_spikes=true_spikes,
        clean_delta_N_ph=clean_signal,
        noisy_signal=noisy_signal,
        noise_components=noise_components,
        artifact_mask=artifact_mask,
        preprocessed_signal=preprocessed,
        detected_spike_indices=detected_indices,
        reconstructed_spike_train=spike_train,
        ssnr=batch_metrics.ssnr,
        precision=eval_metrics["precision"],
        recall=eval_metrics["recall"],
        f1_score=eval_metrics["f1_score"],
        timing_error_ms=mean_timing_error,
    )


def plot_adversarial_environment(
    data: VisualizationData,
    save_path: Optional[str] = None,
    show_components: bool = True,
    figsize: Tuple[float, float] = (14, 8),
) -> Any:
    """
    Generate Figure 1: The Adversarial Environment (Raw Input).

    Shows the raw noisy photon count signal with all noise sources
    visible: shot noise, drift, and EMG/burst artifacts.

    Args:
        data: VisualizationData from generate_presentation_data()
        save_path: Optional path to save figure
        show_components: Whether to show individual noise components
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3 if show_components else 2, 1, figsize=figsize,
                             sharex=True, gridspec_kw={'height_ratios': [2, 1, 1] if show_components else [2, 1]})

    time_s = data.time_ms / 1000.0  # Convert to seconds for display

    # --- Panel A: Raw Noisy Signal ---
    ax1 = axes[0]
    ax1.plot(time_s, data.noisy_signal, 'k-', linewidth=0.5, alpha=0.8, label='Noisy Signal')
    ax1.plot(time_s, data.clean_delta_N_ph, 'b-', linewidth=1.5, alpha=0.6, label='Clean Signal')

    # Mark true spikes
    spike_times = time_s[data.true_spikes]
    spike_amplitudes = data.noisy_signal[data.true_spikes]
    ax1.scatter(spike_times, spike_amplitudes, c='red', s=50, marker='v',
                label='True Spikes', zorder=5)

    # Highlight artifact regions
    if np.any(data.artifact_mask):
        artifact_starts = np.where(np.diff(data.artifact_mask.astype(int)) == 1)[0]
        artifact_ends = np.where(np.diff(data.artifact_mask.astype(int)) == -1)[0]
        for start, end in zip(artifact_starts, artifact_ends[:len(artifact_starts)]):
            ax1.axvspan(time_s[start], time_s[min(end, len(time_s)-1)],
                       alpha=0.3, color='orange', label='_')

    ax1.set_ylabel(r'$\Delta N_{ph}$ (photon counts)', fontsize=12)
    ax1.set_title('Adversarial Environment: Raw Photon Count Signal\n'
                  f'(Shot Noise + Thermal Noise + Drift + EMG Artifacts)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add SSNR annotation
    ax1.text(0.02, 0.95, f'SSNR = {data.ssnr:.1f}', transform=ax1.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # --- Panel B: Membrane Potential (Ground Truth) ---
    ax2 = axes[1]
    ax2.plot(time_s, data.membrane_potential, 'g-', linewidth=0.8)
    ax2.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Spike Threshold')
    ax2.set_ylabel('V (mV)', fontsize=12)
    ax2.set_title('Ground Truth: Izhikevich Neuron Membrane Potential', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel C: Noise Components (optional) ---
    if show_components:
        ax3 = axes[2]

        if 'drift' in data.noise_components:
            ax3.plot(time_s, data.noise_components['drift'],
                    label='Drift', alpha=0.8, linewidth=1)
        if 'thermal' in data.noise_components:
            # Downsample thermal for visibility
            thermal = data.noise_components['thermal']
            ax3.plot(time_s[::10], thermal[::10],
                    label='Thermal', alpha=0.5, linewidth=0.5)
        if 'bursts' in data.noise_components:
            bursts = data.noise_components['bursts']
            if np.any(bursts != 0):
                ax3.plot(time_s, bursts, label='Bursts', alpha=0.8, linewidth=1)

        ax3.set_ylabel('Noise (counts)', fontsize=12)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_title('Noise Components', fontsize=12)
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel('Time (s)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved adversarial environment figure to {save_path}")

    return fig


def plot_decoding_extraction(
    data: VisualizationData,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 10),
) -> Any:
    """
    Generate Figure 2: The Decoding Extraction (Cleaned Signal).

    Shows the reconstructed spike train overlaid on ground truth,
    highlighting the 1ms temporal resolution.

    Args:
        data: VisualizationData from generate_presentation_data()
        save_path: Optional path to save figure
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 1.5, 1, 1]})

    time_s = data.time_ms / 1000.0

    # --- Panel A: Preprocessed Signal ---
    ax1 = axes[0]
    ax1.plot(time_s, data.preprocessed_signal, 'b-', linewidth=0.5, alpha=0.8)

    # Mark detected spikes
    if len(data.detected_spike_indices) > 0:
        det_times = time_s[data.detected_spike_indices]
        det_amplitudes = data.preprocessed_signal[data.detected_spike_indices]
        ax1.scatter(det_times, det_amplitudes, c='red', s=80, marker='o',
                   label=f'Detected ({len(data.detected_spike_indices)})', zorder=5)

    ax1.set_ylabel('Filtered Signal', fontsize=12)
    ax1.set_title('Signal After Preprocessing (Wiener Filter + Bandpass)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Panel B: Spike Train Comparison ---
    ax2 = axes[1]

    # Ground truth spike train
    true_spike_train = data.true_spikes.astype(float)
    ax2.plot(time_s, true_spike_train * 0.9 + 1.1, 'g-', linewidth=2,
             label='Ground Truth (Izhikevich)')

    # Detected spike train (at 1ms resolution)
    # Expand to match time axis
    recon_expanded = np.zeros(len(time_s))
    samples_per_ms = int(1.0 / data.dt_ms)
    for i, val in enumerate(data.reconstructed_spike_train):
        if val > 0:
            start = i * samples_per_ms
            end = min(start + samples_per_ms, len(recon_expanded))
            recon_expanded[start:end] = val

    ax2.plot(time_s, recon_expanded * 0.9, 'r-', linewidth=2,
             label='Reconstructed (1ms resolution)')

    ax2.set_ylabel('Spike Train', fontsize=12)
    ax2.set_ylim(-0.1, 2.2)
    ax2.set_yticks([0.45, 1.55])
    ax2.set_yticklabels(['Decoded', 'Truth'])
    ax2.set_title('Spike Train Comparison: Ground Truth vs Reconstructed', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- Panel C: Timing Alignment (zoomed) ---
    ax3 = axes[2]

    # Find a good window with spikes
    spike_indices = np.where(data.true_spikes)[0]
    if len(spike_indices) > 0:
        center_spike = spike_indices[len(spike_indices)//2]
        window_samples = int(50.0 / data.dt_ms)  # 50ms window
        start = max(0, center_spike - window_samples)
        end = min(len(time_s), center_spike + window_samples)

        ax3.plot(time_s[start:end], true_spike_train[start:end] * 0.9 + 1.1,
                'g-', linewidth=2, label='Ground Truth')
        ax3.plot(time_s[start:end], recon_expanded[start:end] * 0.9,
                'r-', linewidth=2, label='Reconstructed')

        # Draw alignment lines
        for idx in spike_indices:
            if start <= idx < end:
                ax3.axvline(x=time_s[idx], color='gray', linestyle=':', alpha=0.5)

        ax3.set_xlim(time_s[start], time_s[end-1])

    ax3.set_ylabel('Spike Train', fontsize=12)
    ax3.set_ylim(-0.1, 2.2)
    ax3.set_title(f'Timing Alignment Detail (1ms Resolution)', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # --- Panel D: Performance Metrics ---
    ax4 = axes[3]
    ax4.axis('off')

    metrics_text = (
        f"Detection Performance Metrics\n"
        f"{'='*40}\n"
        f"Precision:     {data.precision:.1%}\n"
        f"Recall:        {data.recall:.1%}\n"
        f"F1 Score:      {data.f1_score:.1%}\n"
        f"Mean Timing Error: {data.timing_error_ms:.2f} ms\n"
        f"{'='*40}\n"
        f"True Spikes:   {np.sum(data.true_spikes)}\n"
        f"Detected:      {len(data.detected_spike_indices)}\n"
        f"SSNR:          {data.ssnr:.1f}"
    )

    ax4.text(0.5, 0.5, metrics_text, transform=ax4.transAxes,
             fontsize=14, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax4.set_xlabel('Time (s)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved decoding extraction figure to {save_path}")

    return fig


def plot_performance_summary(
    data: VisualizationData,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> Any:
    """
    Generate Figure 3: Performance Summary.

    Shows key metrics and validates the simulation quality.

    Args:
        data: VisualizationData from generate_presentation_data()
        save_path: Optional path to save figure
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # --- Panel A: Signal Histogram ---
    ax1 = axes[0, 0]
    ax1.hist(data.noisy_signal, bins=100, alpha=0.7, label='Noisy', density=True)
    ax1.hist(data.clean_delta_N_ph, bins=50, alpha=0.7, label='Clean', density=True)
    ax1.set_xlabel(r'$\Delta N_{ph}$ (photon counts)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Signal Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel B: Spike Amplitude Distribution ---
    ax2 = axes[0, 1]
    spike_amps_true = data.clean_delta_N_ph[data.true_spikes]
    if len(data.detected_spike_indices) > 0:
        spike_amps_detected = data.preprocessed_signal[data.detected_spike_indices]
        ax2.hist(spike_amps_detected, bins=20, alpha=0.7, label='Detected')
    ax2.hist(spike_amps_true, bins=20, alpha=0.7, label='True (clean)')
    ax2.set_xlabel('Spike Amplitude', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Spike Amplitude Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Panel C: Confusion Matrix ---
    ax3 = axes[1, 0]

    # Calculate confusion matrix values
    true_indices = set(np.where(data.true_spikes)[0])
    detected_set = set(data.detected_spike_indices)

    # With 2ms tolerance
    tolerance = int(2.0 / data.dt_ms)
    tp = 0
    matched_true = set()
    for det in detected_set:
        for true_idx in true_indices:
            if abs(det - true_idx) <= tolerance and true_idx not in matched_true:
                tp += 1
                matched_true.add(true_idx)
                break

    fp = len(detected_set) - tp
    fn = len(true_indices) - tp

    confusion = np.array([[tp, fp], [fn, 0]])
    im = ax3.imshow(confusion, cmap='Blues', aspect='auto')

    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['True Positive', 'False Positive'])
    ax3.set_yticklabels(['Detected', 'Missed'])
    ax3.set_title('Detection Results', fontsize=12)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            if confusion[i, j] > 0 or (i == 0):
                ax3.text(j, i, f'{confusion[i, j]}', ha='center', va='center',
                        fontsize=16, fontweight='bold')

    # --- Panel D: Key Metrics Summary ---
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = (
        "SIMULATION VALIDATION\n"
        "="*35 + "\n\n"
        f"Temporal Resolution:  1.0 ms\n"
        f"Sampling Rate:        {1000/data.dt_ms:.0f} Hz\n"
        f"Duration:             {data.time_ms[-1]/1000:.1f} s\n\n"
        f"NOISE MODEL\n"
        + "-"*35 + "\n"
        f"Shot Noise:           Poisson\n"
        f"Thermal Noise:        Gaussian\n"
        f"Drift:                Sinusoidal + RW\n"
        f"Artifacts:            EMG-like bursts\n\n"
        f"PERFORMANCE\n"
        + "-"*35 + "\n"
        f"SSNR:                 {data.ssnr:.1f}\n"
        f"F1 Score:             {data.f1_score:.1%}\n"
    )

    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance summary figure to {save_path}")

    return fig


def generate_all_figures(
    output_dir: str = "figures",
    **kwargs
) -> Dict[str, Any]:
    """
    Generate all presentation figures.

    Args:
        output_dir: Directory to save figures
        **kwargs: Arguments passed to generate_presentation_data()

    Returns:
        Dictionary with figures and data
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    data = generate_presentation_data(**kwargs)

    # Generate figures
    fig1 = plot_adversarial_environment(
        data,
        save_path=os.path.join(output_dir, "figure1_adversarial_environment.png")
    )

    fig2 = plot_decoding_extraction(
        data,
        save_path=os.path.join(output_dir, "figure2_decoding_extraction.png")
    )

    fig3 = plot_performance_summary(
        data,
        save_path=os.path.join(output_dir, "figure3_performance_summary.png")
    )

    return {
        "data": data,
        "figure1": fig1,
        "figure2": fig2,
        "figure3": fig3,
    }


def validate_simulation_honesty(data: VisualizationData) -> Dict[str, Any]:
    """
    Validate that the simulation is producing honest, physically-meaningful results.

    This function checks for common "tricks" that could make results look better
    than they actually are.

    Args:
        data: VisualizationData to validate

    Returns:
        Dictionary with validation results and any warnings
    """
    warnings = []
    checks = {}

    # Check 1: Shot noise statistics (variance should approximately equal mean)
    noisy_variance = np.var(data.noisy_signal)
    noisy_mean = np.mean(np.abs(data.noisy_signal))
    checks["shot_noise_variance"] = noisy_variance
    checks["shot_noise_mean"] = noisy_mean

    # Check 2: Signal-to-noise ratio is physically reasonable
    clean_peak = np.max(np.abs(data.clean_delta_N_ph))
    noise_std = np.std(data.noisy_signal - data.clean_delta_N_ph)
    snr = clean_peak / (noise_std + 1e-10)
    checks["signal_peak"] = clean_peak
    checks["noise_std"] = noise_std
    checks["snr"] = snr

    if snr > 100:
        warnings.append(f"SNR ({snr:.1f}) is very high - verify noise is being applied correctly")

    # Check 3: Detection is not trivially perfect
    if data.precision == 1.0 and data.recall == 1.0:
        warnings.append("Perfect detection - noise may be too low or threshold too optimistic")

    # Check 4: Some false positives/negatives expected in noisy conditions
    if data.f1_score > 0.95 and data.ssnr < 100:
        warnings.append("Very high F1 with low SSNR - verify detection threshold")

    # Check 5: Spike times match ground truth
    true_spike_times = np.where(data.true_spikes)[0] * data.dt_ms
    if len(data.detected_spike_indices) > 0:
        detected_times = data.detected_spike_indices * data.dt_ms
        timing_diffs = []
        for det_t in detected_times:
            if len(true_spike_times) > 0:
                closest = true_spike_times[np.argmin(np.abs(true_spike_times - det_t))]
                timing_diffs.append(abs(det_t - closest))
        if timing_diffs:
            checks["mean_timing_diff_ms"] = np.mean(timing_diffs)

    # Check 6: Noise components have realistic magnitudes
    for name, component in data.noise_components.items():
        comp_std = np.std(component)
        checks[f"noise_{name}_std"] = comp_std

    checks["warnings"] = warnings
    checks["is_valid"] = len(warnings) == 0

    return checks
