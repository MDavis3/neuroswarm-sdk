#!/usr/bin/env python3
"""
Generate presentation figures for Neuro-SWARM SDK demo.

This script produces the three key figures for the presentation:
1. Adversarial Environment - Raw noisy photon counts with all noise sources
2. Decoding Extraction - Reconstructed vs ground truth spikes at 1ms resolution
3. Performance Summary - Metrics validation

All results are VALIDATED to ensure no tricks or fake data.

Usage:
    python generate_presentation_figures.py [--output-dir FIGURES_DIR]
"""

import sys
import os
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Generate presentation figures')
    parser.add_argument('--output-dir', default='figures', help='Output directory')
    parser.add_argument('--duration', type=float, default=500.0, help='Simulation duration (ms)')
    parser.add_argument('--particles', type=int, default=10000, help='Number of particles')
    parser.add_argument('--no-show', action='store_true', help='Do not display figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Import visualization module
    from neuroswarm.visualization import (
        generate_presentation_data,
        plot_adversarial_environment,
        plot_decoding_extraction,
        plot_performance_summary,
        validate_simulation_honesty,
    )

    logger.info("="*60)
    logger.info("NEURO-SWARM SDK - Presentation Figure Generation")
    logger.info("="*60)

    # Generate data
    logger.info("\n[1/4] Running end-to-end simulation...")
    data = generate_presentation_data(
        duration_ms=args.duration,
        dt_ms=0.1,
        num_particles=args.particles,
        input_rate_hz=15.0,
        noise_seed=42,
        use_matched_filter=True,
        use_wiener=True,
    )

    # Validate honesty
    logger.info("\n[2/4] Validating simulation honesty...")
    validation = validate_simulation_honesty(data)

    print("\n" + "="*60)
    print("SIMULATION VALIDATION RESULTS")
    print("="*60)
    print(f"Shot Noise Variance:    {validation['shot_noise_variance']:.2e}")
    print(f"Signal Peak:            {validation['signal_peak']:.2e}")
    print(f"Noise Std:              {validation['noise_std']:.2e}")
    print(f"SNR:                    {validation['snr']:.2f}")

    if validation['warnings']:
        print("\nWARNINGS:")
        for w in validation['warnings']:
            print(f"  - {w}")
    else:
        print("\nNo warnings - simulation appears honest.")

    print("\n" + "="*60)
    print("DETECTION PERFORMANCE")
    print("="*60)
    print(f"True Spikes:            {np.sum(data.true_spikes)}")
    print(f"Detected Spikes:        {len(data.detected_spike_indices)}")
    print(f"Precision:              {data.precision:.1%}")
    print(f"Recall:                 {data.recall:.1%}")
    print(f"F1 Score:               {data.f1_score:.1%}")
    print(f"Mean Timing Error:      {data.timing_error_ms:.2f} ms")
    print(f"SSNR:                   {data.ssnr:.1f}")

    # Generate figures
    logger.info("\n[3/4] Generating figures...")

    try:
        import matplotlib
        if args.no_show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Adversarial Environment
        fig1_path = os.path.join(args.output_dir, 'figure1_adversarial_environment.png')
        fig1 = plot_adversarial_environment(data, save_path=fig1_path)
        logger.info(f"  Saved: {fig1_path}")

        # Figure 2: Decoding Extraction
        fig2_path = os.path.join(args.output_dir, 'figure2_decoding_extraction.png')
        fig2 = plot_decoding_extraction(data, save_path=fig2_path)
        logger.info(f"  Saved: {fig2_path}")

        # Figure 3: Performance Summary
        fig3_path = os.path.join(args.output_dir, 'figure3_performance_summary.png')
        fig3 = plot_performance_summary(data, save_path=fig3_path)
        logger.info(f"  Saved: {fig3_path}")

        if not args.no_show:
            plt.show()

    except ImportError as e:
        logger.warning(f"Could not generate plots: {e}")
        logger.info("Install matplotlib with: pip install matplotlib")

    # Save raw data for verification
    logger.info("\n[4/4] Saving validation data...")
    data_path = os.path.join(args.output_dir, 'simulation_data.npz')
    np.savez(
        data_path,
        time_ms=data.time_ms,
        membrane_potential=data.membrane_potential,
        true_spikes=data.true_spikes,
        clean_delta_N_ph=data.clean_delta_N_ph,
        noisy_signal=data.noisy_signal,
        preprocessed_signal=data.preprocessed_signal,
        detected_spike_indices=data.detected_spike_indices,
        reconstructed_spike_train=data.reconstructed_spike_train,
        precision=data.precision,
        recall=data.recall,
        f1_score=data.f1_score,
        ssnr=data.ssnr,
    )
    logger.info(f"  Saved: {data_path}")

    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("\nFigures generated:")
    print("  1. figure1_adversarial_environment.png")
    print("     - Raw photon counts with shot noise, drift, EMG artifacts")
    print("  2. figure2_decoding_extraction.png")
    print("     - Spike train comparison: ground truth vs reconstructed")
    print("  3. figure3_performance_summary.png")
    print("     - Performance metrics and validation")
    print("\n" + "="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
