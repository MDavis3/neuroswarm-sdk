"""
Neuro-SWARM SDK
===============

A production-style Python library for simulating and decoding backscattered NIR
signals from electro-plasmonic nanoparticles (Neuro-SWARM3).

Core modules:
- physics: Forward model (Izhikevich neuron + Drude-Lorentz + Equation 1)
- noise: Adversarial noise generation for robustness testing
- decoding: Signal extraction with PCA/ICA and spike reconstruction

Reference:
    Hardy et al., "Neuro-SWARM3: System-on-a-Nanoparticle for Wireless Recording
    of Brain Activity," IEEE Photonics Technology Letters, 2021.
"""

__version__ = "0.1.0"
__author__ = "Neuro-SWARM SDK Team"

from .types import (
    DrudeLorenzParams,
    NanoparticleGeometry,
    OpticalSystemParams,
    IzhikevichParams,
    SimulationConfig,
    WavelengthSweepParams,
    ParticleDistributionParams,
)
from .physics import NeuroSwarmPhysics
from .noise import AdversarialNoiseGenerator
from .decoding import SignalExtractor
from .reporting import summarize_detection, noise_sweep_report, wavelength_sweep_report

__all__ = [
    # Types
    "DrudeLorenzParams",
    "NanoparticleGeometry",
    "OpticalSystemParams",
    "IzhikevichParams",
    "SimulationConfig",
    "WavelengthSweepParams",
    "ParticleDistributionParams",
    # Core classes
    "NeuroSwarmPhysics",
    "AdversarialNoiseGenerator",
    "SignalExtractor",
    # Reporting
    "summarize_detection",
    "noise_sweep_report",
    "wavelength_sweep_report",
]
