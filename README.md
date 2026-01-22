# Neuro-SWARM3 SDK

End-to-end demo of the Neuro-SWARM3 signal pipeline: simulate spikes → generate NIR-II backscatter → add noise → decode spikes.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What you’ll see

- Pipeline overview (forward → inverse)
- Live simulation (clean + noisy + decoded)
- Noise robustness sweep
- Wavelength sweep (presentation mode + optional attenuation/QE)

## Short usage (Python)

```python
from neuroswarm import NeuroSwarmPhysics, SignalExtractor, AdversarialNoiseGenerator

physics = NeuroSwarmPhysics()
sim = physics.simulate(input_rate_hz=10)
clean = sim["delta_N_ph"]

noise = AdversarialNoiseGenerator()
noisy = noise.corrupt_signal(clean, dt=physics.config.dt)["noisy_signal"]

decoded = SignalExtractor().process_batch(noisy, dt_ms=physics.config.dt)
print("Detected spikes:", len(decoded["spike_indices"]))
```

## Notes

- This is a **simplified, presentation-focused** model that mirrors the published pipeline.
- Attribution is listed in `CITATION.md`.

## License

MIT
