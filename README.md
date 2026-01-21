# Neuro-SWARM³ SDK

A production-grade Python library for simulating and decoding backscattered NIR-II signals from electro-plasmonic nanoparticle probes for wireless neural activity recording.

## Overview

Neuro-SWARM³ (Neurophotonic Solution-dispersible Wireless Activity Reporter for Massively Multiplexed Measurements) enables remote detection of in vivo bioelectric signals using near-infrared light (NIR-II, 1000-1700 nm). This SDK provides:

1. **Forward Model (`physics.py`)**: Simulates the biological and physical ground truth
   - Izhikevich neuron model for realistic membrane potential dynamics
   - Drude-Lorentz dielectric function for PEDOT:PSS electrochromic response
   - Mie theory-based scattering calculations for core-shell nanoparticles
   - Equation (1) implementation for differential photon count (ΔN_ph)

2. **Inverse Model (`decoding.py`)**: Production-adjacent signal extraction
   - Bandpass filtering and artifact removal
   - PCA/ICA-based neural signal separation
   - Spike detection at 1 ms temporal resolution
   - Per-batch SSNR validation logging

3. **Adversarial Testing (`noise.py`)**: Robustness validation
   - Shot noise injection (Poisson statistics)
   - Low-frequency drift simulation
   - Burst artifact generation

4. **Robustness Enhancements**
   - Matched filter spike detection for heavy noise
   - Wiener deconvolution for noise-optimal recovery
   - Wavelength sweep for NIR-II optimization
   - Spatial particle distribution modeling
   - Stress-test reporting outputs

## Technical Specifications

### Nanoparticle Geometry
| Layer | Material | Dimension |
|-------|----------|-----------|
| Core | SiOx (silica) | 63 nm radius |
| Shell | Au (gold) | 5 nm thickness |
| Coating | PEDOT:PSS | 15 nm thickness |

### PEDOT:PSS Drude-Lorentz Parameters (300K, NIR-II)
| Parameter | Symbol | Value |
|-----------|--------|-------|
| High-frequency permittivity | ε∞ | 2.75 |
| Plasma frequency | ωp | 1.325 eV |
| Drude damping | γ | 0.271 eV |
| Lorentz oscillator strength | f₁ | 0.098 |
| Lorentz resonance frequency | ωL1 | 1.505 eV |
| Lorentz damping | γL1 | 1.048 eV |

### Key Performance Targets
- **SSNR**: ~10³ (signal-to-shot noise ratio)
- **Temporal Resolution**: 1 ms
- **Field Sensitivity**: Up to 40% modulation at 12 mV/nm
- **Scattering Cross-section**: ~10⁴ nm²

## Installation

```bash
git clone https://github.com/MDavis3/neuroswarm-sdk.git
cd neuroswarm-sdk
pip install -r requirements.txt
```

## Quick Start

```python
from neuroswarm import NeuroSwarmPhysics, SignalExtractor, AdversarialNoiseGenerator

# 1. Generate synthetic neural signal (Forward Model)
physics = NeuroSwarmPhysics()
result = physics.simulate()

clean_signal = result["delta_N_ph"]
time_ms = result["time_ms"]

# 2. Inject realistic noise for testing
noise_gen = AdversarialNoiseGenerator()
noisy_signal = noise_gen.inject(clean_signal, dt_ms=0.1)

# 3. Extract spikes from noisy signal (Inverse Model)
extractor = SignalExtractor()
extraction = extractor.extract(noisy_signal, dt_ms=0.1)

print(f"Detected {len(extraction.spike_times_ms)} spikes")
print(f"SSNR: {extraction.ssnr:.1f}")
```

## Robustness Enhancements

Example using Wiener + matched filter for heavy noise:

```python
from neuroswarm.decoding import SignalExtractor, DecodingParams

params = DecodingParams(
    use_wiener=True,
    use_matched_filter=True,
    matched_filter_window_ms=6.0,
)
extractor = SignalExtractor(params)
result = extractor.process_batch(noisy_signal, dt_ms=0.1)
```

## Project Structure

```
neuroswarm-sdk/
├── app.py                  # Streamlit interactive dashboard
├── src/
│   └── neuroswarm/
│       ├── __init__.py
│       ├── types.py        # Type-safe dataclasses (geometry, Drude-Lorentz params)
│       ├── physics.py      # Forward model (Izhikevich + Drude-Lorentz + Eq. 1)
│       ├── decoding.py     # Inverse model (SignalExtractor with PCA/ICA)
│       ├── noise.py        # Adversarial noise generators
│       └── reporting.py    # Stress-test reporting utilities
├── scripts/
│   └── generate_visuals.py # Generate presentation plots
├── assets/
│   └── visuals/            # Pre-generated presentation plots
├── tests/
│   ├── test_physics.py     # Equation (1) validation
│   ├── test_decoding.py    # Spike recovery under noise
│   ├── test_noise.py       # Noise injection statistics
│   └── test_reporting.py   # Reporting utilities tests
├── requirements.txt
└── README.md
```

## Core Equations

### Equation (1): Differential Photon Count
From Hardy et al. (2021):

$$\Delta N_{ph} = I_{inc} \cdot (\Delta Q_{sca} \cdot \pi r^2) \cdot \frac{\lambda}{hc} \cdot \eta \cdot T \cdot t_{int}$$

Where:
- $I_{inc}$: Incident light intensity (10 mW/mm²)
- $\Delta Q_{sca}$: Change in scattering cross section
- $r$: Nanoparticle radius
- $\lambda$: Probing wavelength (1050 nm)
- $\eta$: Solid angle fraction (NA = 0.9)
- $T$: Detection efficiency (0.5)
- $t_{int}$: Integration time (1 ms)

### Signal-to-Shot Noise Ratio
$$SSNR = \frac{\Delta S}{S_0} \cdot \sqrt{N_{ph}}$$

Target: SSNR ~ 10³ with 10³ probes at 10 mW/mm² illumination.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Forward Model (physics.py)              │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Izhikevich  │──▶│ Drude-Lorentz│──▶│  Equation (1)  │  │
│  │   Neuron     │   │   ε(ω, E)    │   │    ΔN_ph       │  │
│  └──────────────┘   └──────────────┘   └───────┬────────┘  │
└───────────────────────────────────────────────┼────────────┘
                                                │ Clean Signal
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 Noise Injection (noise.py)                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐│
│  │ Shot Noise │  │   Drift    │  │   Burst Artifacts     ││
│  │  (Poisson) │  │ (0.1 Hz)   │  │   (EMG, movement)     ││
│  └────────────┘  └────────────┘  └────────────────────────┘│
└───────────────────────────────────────────────┬────────────┘
                                                │ Noisy Signal
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│                  Inverse Model (decoding.py)                │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Bandpass    │──▶│   PCA/ICA    │──▶│    Spike       │  │
│  │  Filter      │   │  Separation  │   │   Detection    │  │
│  └──────────────┘   └──────────────┘   └───────┬────────┘  │
│                                                │            │
│                      SSNR Logging ◀────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Interactive Dashboard

Launch the Streamlit-based visualization dashboard:

```bash
streamlit run app.py
```

Features:
- 📊 **Pipeline Overview**: Visual diagram of the forward/inverse model pipeline
- 🌈 **Wavelength Sweep**: Optimize detection wavelength in the NIR-II window
- 📉 **Noise Robustness**: Test decoder performance across noise levels
- 🔬 **Live Simulation**: Run end-to-end simulations with configurable parameters

## Presentation Visuals

Pre-generated plots are available in `assets/visuals/`:
- `pipeline_diagram.png` — End-to-end signal processing pipeline
- `wavelength_sweep.png` — Optimal wavelength in NIR-II range
- `noise_robustness.png` — Decoder F1 vs noise level

Regenerate with:
```bash
PYTHONPATH=src python scripts/generate_visuals.py
```

## Testing

```bash
pytest tests/ -v
```

Tests validate:
- Equation (1) numerics against published values
- Drude-Lorentz output in NIR-II wavelength range
- Spike recovery accuracy under injected noise
- SSNR targets from Section III of the paper

## Dependencies

- `numpy>=1.24.0` — Array operations, Drude-Lorentz math
- `scipy>=1.10.0` — Signal filtering, numerical methods
- `scikit-learn>=1.3.0` — PCA/ICA decomposition
- `matplotlib>=3.7.0` — Visualization and plotting
- `streamlit>=1.28.0` — Interactive dashboard
- `pytest>=7.4.0` — Testing framework

## References

1. Hardy, N., Habib, A., Ivanov, T., & Yanik, A. A. (2021). Neuro-SWARM³: System-on-a-Nanoparticle for Wireless Recording of Brain Activity. *IEEE Photonics Technology Letters*, 33(16), 900-903.

2. Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on Neural Networks*, 14(6), 1569-1572.

3. Du, Y., et al. (2018). Dielectric properties of DMSO-doped-PEDOT:PSS at THz frequencies. *Physica Status Solidi*, 255(4), 1700547.

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request
