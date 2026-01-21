"""
Generate benchmark numbers for README.
"""
import sys
sys.path.insert(0, "src")

import numpy as np
from neuroswarm.physics import NeuroSwarmPhysics
from neuroswarm.noise import AdversarialNoiseGenerator, NoiseParams
from neuroswarm.decoding import SignalExtractor, DecodingParams
from neuroswarm.types import SimulationConfig

# Generate clean signal with stronger spikes
# Use longer duration to get more spikes
config = SimulationConfig(duration=1000, num_particles=1000, dt=0.1)
physics = NeuroSwarmPhysics(config)
sim = physics.simulate(input_rate_hz=15)  # Higher rate for more spikes
clean = sim["delta_N_ph"]
true_spikes = sim["spikes"]

# Scale signal to have meaningful amplitudes for detection
# The clean signal from physics may be small - amplify for testing
signal_max = np.max(np.abs(clean))
if signal_max > 0:
    clean = clean * (10000 / signal_max)  # Scale to ~10000 max

n_true = int(np.sum(true_spikes))
print(f"Generated signal: {len(clean)} samples, {n_true} true spikes")
print(f"Signal range: [{clean.min():.1f}, {clean.max():.1f}]")
print()

noise_levels = [100, 500, 1000, 2000, 3000]
print("| Noise (std) | Standard F1 | Robust F1 | Improvement |")
print("|-------------|-------------|-----------|-------------|")

for noise_std in noise_levels:
    # Corrupt signal
    noise_params = NoiseParams(thermal_noise_std=noise_std, seed=42)
    ng = AdversarialNoiseGenerator(noise_params)
    noisy = ng.corrupt_signal(clean, 0.1)["noisy_signal"]
    
    # Standard decoder with adjusted threshold
    std_dec = SignalExtractor(DecodingParams(
        use_matched_filter=False, 
        use_wiener=False,
        spike_threshold=3.0
    ))
    std_res = std_dec.process_batch(noisy, dt_ms=0.1)
    std_n_det = len(std_res["spike_indices"])
    std_eval = std_dec.evaluate_reconstruction(std_res["spike_indices"], true_spikes, tolerance_samples=50)
    
    # Robust decoder  
    rob_dec = SignalExtractor(DecodingParams(
        use_matched_filter=True, 
        use_wiener=True,
        spike_threshold=3.0
    ))
    rob_res = rob_dec.process_batch(noisy, dt_ms=0.1)
    rob_n_det = len(rob_res["spike_indices"])
    rob_eval = rob_dec.evaluate_reconstruction(rob_res["spike_indices"], true_spikes, tolerance_samples=50)
    
    improvement = rob_eval["f1_score"] - std_eval["f1_score"]
    sign = "+" if improvement >= 0 else ""
    print(f"| {noise_std:>4} | {std_eval['f1_score']:.3f} ({std_n_det:>2} det) | {rob_eval['f1_score']:.3f} ({rob_n_det:>2} det) | {sign}{improvement:.3f} |")

print(f"\nTest conditions: {config.num_particles} particles, {config.duration}ms duration, 15 Hz input rate")
