"""
Generate presentation-ready visuals for the Neuro-SWARM SDK.

Outputs:
  - assets/visuals/pipeline_diagram.png
  - assets/visuals/wavelength_sweep.png
  - assets/visuals/noise_robustness.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from neuroswarm.physics import NeuroSwarmPhysics
from neuroswarm.reporting import noise_sweep_report, wavelength_sweep_report
from neuroswarm.decoding import SignalExtractor, DecodingParams
from neuroswarm.noise import NoiseParams, AdversarialNoiseGenerator
from neuroswarm.types import WavelengthSweepParams


OUTPUT_DIR = os.path.join("assets", "visuals")


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    boxes = [
        ("Izhikevich\nNeuron", 0.05),
        ("E-field\nTransduction", 0.27),
        ("Scattering\nΔQ_sca", 0.49),
        ("Photon Count\nΔN_ph", 0.71),
        ("Noise +\nDecoder", 0.88),
    ]

    for label, x in boxes:
        ax.add_patch(plt.Rectangle((x, 0.3), 0.18, 0.4, fill=False, linewidth=2))
        ax.text(x + 0.09, 0.5, label, ha="center", va="center", fontsize=9)

    # Arrows
    for i in range(len(boxes) - 1):
        x_start = boxes[i][1] + 0.18
        x_end = boxes[i + 1][1]
        ax.annotate("", xy=(x_end, 0.5), xytext=(x_start, 0.5),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pipeline_diagram.png"), dpi=200)
    plt.close(fig)


def plot_wavelength_sweep() -> None:
    physics = NeuroSwarmPhysics()
    sweep = wavelength_sweep_report(
        physics, WavelengthSweepParams(step_nm=10.0)
    )

    wavelengths = sweep["wavelengths_nm"]
    ssnr = sweep["ssnr"]
    optimal = sweep["optimal_wavelength_nm"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(wavelengths, ssnr, label="SSNR", linewidth=2)
    ax.axvline(optimal, color="red", linestyle="--", label=f"Optimal λ = {optimal:.0f} nm")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("SSNR")
    ax.set_title("NIR-II Wavelength Sweep (SSNR)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "wavelength_sweep.png"), dpi=200)
    plt.close(fig)


def plot_noise_robustness() -> None:
    # Generate a clean signal
    physics = NeuroSwarmPhysics()
    sim = physics.simulate(input_rate_hz=10.0)
    clean_signal = sim["delta_N_ph"]

    # Noise sweep
    noise_levels = [10, 50, 100, 200, 400, 800]

    baseline_extractor = SignalExtractor(DecodingParams(spike_threshold=4.0))
    enhanced_extractor = SignalExtractor(DecodingParams(
        spike_threshold=4.0,
        use_wiener=True,
        use_matched_filter=True,
        matched_filter_window_ms=6.0
    ))

    baseline_report = noise_sweep_report(
        clean_signal, physics.config.dt, noise_levels, baseline_extractor
    )
    enhanced_report = noise_sweep_report(
        clean_signal, physics.config.dt, noise_levels, enhanced_extractor
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noise_levels, baseline_report["f1_scores"], label="Baseline F1", marker="o")
    ax.plot(noise_levels, enhanced_report["f1_scores"], label="Matched+Wiener F1", marker="o")
    ax.set_xlabel("Thermal Noise Std (photon counts)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Noise Robustness: Spike Recovery")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "noise_robustness.png"), dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_dir()
    plot_pipeline_diagram()
    plot_wavelength_sweep()
    plot_noise_robustness()
    print(f"Saved visuals to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
