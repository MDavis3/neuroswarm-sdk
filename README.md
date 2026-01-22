# Neuro-SWARM3 SDK

A Python toolkit for simulating and decoding backscattered NIR-II signals from electro-plasmonic nanoparticle probes for wireless neural activity readout.

This repository focuses on **clarity first**: how the forward model is built, how noise corrupts the signal, and how the decoder recovers spikes.

## What this repo includes

- Forward model (`src/neuroswarm/physics.py`)
  - Izhikevich neuron dynamics
  - Drude-Lorentz dielectric response (PEDOT:PSS)
  - Simplified Mie scattering
  - Differential photon count (Equation 1)
- Inverse model (`src/neuroswarm/decoding.py`)
  - Filtering, artifact handling, spike detection
  - Optional matched filter and Wiener deconvolution
- Noise model (`src/neuroswarm/noise.py`)
  - Shot noise, thermal noise, drift, burst artifacts
- Visualization utilities (`src/neuroswarm/visualization.py`)
  - Generate presentation-ready figures from real simulations
- Streamlit dashboard (`app.py`)
  - Interactive exploration and plots

## Installation

```bash
git clone https://github.com/MDavis3/neuroswarm-sdk.git
cd neuroswarm-sdk
pip install -r requirements.txt
```

## Quick start (forward -> noise -> decode)

```python
from neuroswarm import NeuroSwarmPhysics, SignalExtractor, AdversarialNoiseGenerator

# Forward model (clean signal)
physics = NeuroSwarmPhysics()
sim = physics.simulate(input_rate_hz=10)
clean = sim["delta_N_ph"]

# Add realistic noise
noise = AdversarialNoiseGenerator()
noisy = noise.corrupt_signal(clean, dt=physics.config.dt)["noisy_signal"]

# Decode spikes
extractor = SignalExtractor()
decoded = extractor.process_batch(noisy, dt_ms=physics.config.dt)

print(f"Detected spikes: {len(decoded['spike_indices'])}")
print(f"SSNR: {decoded['metrics'].ssnr:.2f}")
```

## Wavelength sweep

```python
from neuroswarm import NeuroSwarmPhysics, WavelengthSweepParams

physics = NeuroSwarmPhysics()
report = physics.sweep_wavelengths(WavelengthSweepParams(step_nm=25.0))
print("Optimal wavelength:", report["optimal_wavelength_nm"], "nm")
```

## Dashboard

```bash
streamlit run app.py
```

Tabs:
- Pipeline Overview
- Wavelength Sweep
- Noise Robustness
- Live Simulation

## Presentation figures

Generate presentation-ready plots from real simulations:

```bash
python scripts/generate_presentation_figures.py --output-dir figures --no-show
```

This produces:
- `figure1_adversarial_environment.png`
- `figure2_decoding_extraction.png`
- `figure3_performance_summary.png`

## Key defaults (from paper/patent context)

- NIR-II window: 1000 to 1700 nm
- Nominal resonance around 1050 nm
- Particle geometry: 63 nm core, 5 nm shell, 15 nm coating
- Target SSNR: around 10^3 under typical conditions

## Tests

```bash
pytest tests/ -v
```

## Project structure

```
neuroswarm-sdk/
|-- app.py
|-- src/
|   `-- neuroswarm/
|       |-- __init__.py
|       |-- types.py
|       |-- physics.py
|       |-- decoding.py
|       |-- noise.py
|       |-- reporting.py
|       `-- visualization.py
|-- scripts/
|   `-- generate_presentation_figures.py
|-- assets/
|   `-- visuals/
|-- tests/
|-- requirements.txt
`-- README.md
```

## References

1. Hardy, N., Habib, A., Ivanov, T., Yanik, A. A. (2021). Neuro-SWARM3: System-on-a-Nanoparticle for Wireless Recording of Brain Activity. IEEE Photonics Technology Letters, 33(16), 900-903.
2. Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569-1572.

## License

MIT
