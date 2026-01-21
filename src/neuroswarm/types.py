"""
Typed configuration classes for Neuro-SWARM SDK.

All parameters include units in docstrings and enforce value sanity checks.
Default values are based on the reference paper and patent specifications.

Reference:
    Hardy et al., IEEE Photonics Technology Letters, 2021.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


@dataclass
class DrudeLorenzParams:
    """
    Drude-Lorentz model parameters for PEDOT:PSS at 300K.

    The dielectric function is modeled as:
        epsilon(omega) = epsilon_inf - omega_p^2 / (omega^2 + i*gamma*omega)
                         + sum_j [ f_j * omega_Lj^2 / (omega_Lj^2 - omega^2 - i*gamma_Lj*omega) ]

    Attributes:
        epsilon_inf: High-frequency dielectric constant (dimensionless)
        omega_p: Plasma frequency (eV)
        gamma: Drude damping rate (eV)
        f1: Lorentz oscillator strength (dimensionless)
        omega_L1: Lorentz resonance frequency (eV)
        gamma_L1: Lorentz damping rate (eV)
    """
    epsilon_inf: float = 2.75
    omega_p: float = 1.325  # eV
    gamma: float = 0.271    # eV
    f1: float = 0.098       # dimensionless
    omega_L1: float = 1.505 # eV
    gamma_L1: float = 1.048 # eV

    def __post_init__(self) -> None:
        """Validate parameters are physically meaningful."""
        if self.epsilon_inf < 1.0:
            raise ValueError(f"epsilon_inf must be >= 1.0, got {self.epsilon_inf}")
        if self.omega_p <= 0:
            raise ValueError(f"omega_p must be positive, got {self.omega_p}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.f1 < 0:
            raise ValueError(f"f1 must be non-negative, got {self.f1}")
        if self.omega_L1 <= 0:
            raise ValueError(f"omega_L1 must be positive, got {self.omega_L1}")
        if self.gamma_L1 <= 0:
            raise ValueError(f"gamma_L1 must be positive, got {self.gamma_L1}")


@dataclass
class NanoparticleGeometry:
    """
    Geometry parameters for the core-shell nanoparticle structure.

    Structure: SiOx core / Au shell / PEDOT:PSS coating

    Attributes:
        core_radius: SiOx core radius (nm)
        shell_thickness: Au shell thickness (nm)
        coating_thickness: PEDOT:PSS coating thickness (nm)
    """
    core_radius: float = 63.0        # nm (SiOx)
    shell_thickness: float = 5.0     # nm (Au)
    coating_thickness: float = 15.0  # nm (PEDOT:PSS)

    def __post_init__(self) -> None:
        """Validate geometry parameters."""
        if self.core_radius <= 0:
            raise ValueError(f"core_radius must be positive, got {self.core_radius}")
        if self.shell_thickness <= 0:
            raise ValueError(f"shell_thickness must be positive, got {self.shell_thickness}")
        if self.coating_thickness <= 0:
            raise ValueError(f"coating_thickness must be positive, got {self.coating_thickness}")
        # Total diameter should be < 200 nm for BBB crossing
        total_diameter = 2 * (self.core_radius + self.shell_thickness + self.coating_thickness)
        if total_diameter > 200:
            import warnings
            warnings.warn(
                f"Total nanoparticle diameter ({total_diameter:.1f} nm) exceeds 200 nm. "
                "This may impair blood-brain barrier crossing."
            )

    @property
    def total_radius(self) -> float:
        """Total radius of the nanoparticle (nm)."""
        return self.core_radius + self.shell_thickness + self.coating_thickness

    @property
    def total_diameter(self) -> float:
        """Total diameter of the nanoparticle (nm)."""
        return 2 * self.total_radius

    @property
    def geometric_cross_section(self) -> float:
        """Geometric cross-sectional area (nm^2)."""
        return np.pi * self.total_radius ** 2


@dataclass
class OpticalSystemParams:
    """
    Parameters for the optical detection system.

    Default values assume NIR-II operation at 1050 nm with a 20x objective.

    Attributes:
        wavelength: Probing wavelength (nm), NIR-II range: 1000-1700 nm
        incident_intensity: Light intensity at target (mW/mm^2)
        numerical_aperture: Objective NA for collection efficiency
        quantum_yield: Detector quantum efficiency (dimensionless, 0-1)
        integration_time: Signal integration time (ms)
    """
    wavelength: float = 1050.0       # nm
    incident_intensity: float = 10.0  # mW/mm^2
    numerical_aperture: float = 0.9   # 20x objective
    quantum_yield: float = 0.5        # detection efficiency
    integration_time: float = 1.0     # ms

    def __post_init__(self) -> None:
        """Validate optical parameters."""
        if not (1000 <= self.wavelength <= 1700):
            raise ValueError(
                f"wavelength must be in NIR-II range (1000-1700 nm), got {self.wavelength}"
            )
        if self.incident_intensity <= 0:
            raise ValueError(
                f"incident_intensity must be positive, got {self.incident_intensity}"
            )
        if not (0 < self.numerical_aperture <= 1.5):
            raise ValueError(
                f"numerical_aperture must be in (0, 1.5], got {self.numerical_aperture}"
            )
        if not (0 < self.quantum_yield <= 1):
            raise ValueError(
                f"quantum_yield must be in (0, 1], got {self.quantum_yield}"
            )
        if self.integration_time <= 0:
            raise ValueError(
                f"integration_time must be positive, got {self.integration_time}"
            )

    @property
    def solid_angle_fraction(self) -> float:
        """
        Fraction of scattered light collected by the objective.

        For a microscope objective: eta = 0.5 * (1 - cos(arcsin(NA)))
        """
        theta_max = np.arcsin(min(self.numerical_aperture, 1.0))
        return 0.5 * (1 - np.cos(theta_max))

    @property
    def photon_energy(self) -> float:
        """Energy per photon at the probing wavelength (J)."""
        h = 6.62607015e-34  # Planck's constant (J*s)
        c = 2.99792458e8    # Speed of light (m/s)
        wavelength_m = self.wavelength * 1e-9  # Convert nm to m
        return h * c / wavelength_m


@dataclass
class IzhikevichParams:
    """
    Parameters for the Izhikevich neuron model.

    Model equations:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)
        if v >= 30 mV: v = c, u = u + d

    Default parameters produce phasic spiking behavior as used in the paper.

    Attributes:
        a: Time scale of recovery variable u (1/ms)
        b: Sensitivity of u to subthreshold v (1/ms)
        c: Post-spike reset value of v (mV)
        d: Post-spike increment of u (mV/ms)
        v_thresh: Spike threshold (mV)
        v_init: Initial membrane potential (mV)
        u_init: Initial recovery variable (mV/ms)
    """
    a: float = 0.02   # Phasic spiking
    b: float = 0.25
    c: float = -65.0  # mV
    d: float = 6.0
    v_thresh: float = 30.0   # mV
    v_init: float = -65.0    # mV
    u_init: Optional[float] = None  # Will be set to b*v_init if None

    def __post_init__(self) -> None:
        """Set default u_init and validate parameters."""
        if self.u_init is None:
            self.u_init = self.b * self.v_init
        if self.a <= 0:
            raise ValueError(f"a must be positive, got {self.a}")
        if self.v_thresh <= self.c:
            raise ValueError(
                f"v_thresh ({self.v_thresh}) must be greater than c ({self.c})"
            )


@dataclass
class NeuralCellParams:
    """
    Parameters for modeling neural cell electrical properties.

    Used to calculate extracellular field strength from membrane potential.

    Attributes:
        cell_diameter: Neural cell diameter (um)
        membrane_capacitance: Specific membrane capacitance (uF/cm^2)
        csf_dielectric: Dielectric constant of cerebrospinal fluid
        resting_potential: Resting membrane potential (mV)
        peak_potential: Peak depolarization potential (mV)
    """
    cell_diameter: float = 20.0           # um
    membrane_capacitance: float = 1.0     # uF/cm^2
    csf_dielectric: float = 88.9          # dimensionless
    resting_potential: float = -65.0      # mV
    peak_potential: float = 45.0          # mV (approximate)

    def __post_init__(self) -> None:
        """Validate neural cell parameters."""
        if self.cell_diameter <= 0:
            raise ValueError(
                f"cell_diameter must be positive, got {self.cell_diameter}"
            )
        if self.membrane_capacitance <= 0:
            raise ValueError(
                f"membrane_capacitance must be positive, got {self.membrane_capacitance}"
            )
        if self.csf_dielectric <= 0:
            raise ValueError(
                f"csf_dielectric must be positive, got {self.csf_dielectric}"
            )

    @property
    def cell_surface_area(self) -> float:
        """Cell membrane surface area (cm^2)."""
        radius_cm = (self.cell_diameter / 2) * 1e-4  # um to cm
        return 4 * np.pi * radius_cm ** 2

    @property
    def delta_vm(self) -> float:
        """Transmembrane potential variation during spike (mV)."""
        return self.peak_potential - self.resting_potential

    @property
    def charge_transfer(self) -> float:
        """
        Total charge transferred across membrane during spike (C).

        Q = Cm * Acell * delta_Vm
        """
        # Convert: uF/cm^2 * cm^2 * mV = uF * mV = pC (1e-12 C)
        return (self.membrane_capacitance * 1e-6 *   # uF to F
                self.cell_surface_area *              # cm^2
                self.delta_vm * 1e-3)                 # mV to V


@dataclass
class SimulationConfig:
    """
    Master configuration for a Neuro-SWARM simulation.

    Aggregates all sub-configurations with sensible defaults.

    Attributes:
        drude_lorentz: PEDOT:PSS dielectric parameters
        geometry: Nanoparticle structure parameters
        optical: Optical detection system parameters
        neuron: Izhikevich model parameters
        cell: Neural cell electrical parameters
        dt: Simulation time step (ms)
        duration: Total simulation duration (ms)
        max_field_strength: Maximum extracellular field (mV/nm)
        num_particles: Number of nanoparticles in detection volume
        distribution: Particle distribution parameters
    """
    drude_lorentz: DrudeLorenzParams = field(default_factory=DrudeLorenzParams)
    geometry: NanoparticleGeometry = field(default_factory=NanoparticleGeometry)
    optical: OpticalSystemParams = field(default_factory=OpticalSystemParams)
    neuron: IzhikevichParams = field(default_factory=IzhikevichParams)
    cell: NeuralCellParams = field(default_factory=NeuralCellParams)
    dt: float = 0.1           # ms (simulation step)
    duration: float = 1000.0  # ms (total simulation time)
    max_field_strength: float = 3.0  # mV/nm (conservative estimate)
    num_particles: int = 1000  # Number of nanoparticles
    distribution: "ParticleDistributionParams" = field(
        default_factory=lambda: ParticleDistributionParams()
    )

    def __post_init__(self) -> None:
        """Validate simulation parameters."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.max_field_strength <= 0:
            raise ValueError(
                f"max_field_strength must be positive, got {self.max_field_strength}"
            )
        if self.num_particles <= 0:
            raise ValueError(
                f"num_particles must be positive, got {self.num_particles}"
            )

    @property
    def num_steps(self) -> int:
        """Number of simulation time steps."""
        return int(self.duration / self.dt)

    @property
    def time_vector(self) -> np.ndarray:
        """Time vector for simulation (ms)."""
        return np.arange(0, self.duration, self.dt)


@dataclass
class WavelengthSweepParams:
    """
    Parameters for wavelength sweep optimization in NIR-II range.

    Attributes:
        min_wavelength: Minimum wavelength to scan (nm)
        max_wavelength: Maximum wavelength to scan (nm)
        step_nm: Wavelength step size (nm)
        electric_field: Fixed field strength for sweep (mV/nm)
    """
    min_wavelength: float = 1000.0
    max_wavelength: float = 1700.0
    step_nm: float = 10.0
    electric_field: float = 3.0

    def __post_init__(self) -> None:
        if self.min_wavelength < 1000 or self.max_wavelength > 1700:
            raise ValueError(
                "Wavelength sweep must remain within NIR-II range (1000-1700 nm)"
            )
        if self.step_nm <= 0:
            raise ValueError(f"step_nm must be positive, got {self.step_nm}")
        if self.min_wavelength >= self.max_wavelength:
            raise ValueError("min_wavelength must be less than max_wavelength")


@dataclass
class ParticleDistributionParams:
    """
    Parameters describing spatial distribution of nanoparticle probes.

    Attributes:
        distribution_type: 'sphere', 'slab', or 'clustered'
        radius_um: Sphere radius (um) for distribution
        slab_thickness_um: Slab thickness (um) if using slab
        field_decay_length_um: Decay length for E-field (um)
        seed: Random seed for reproducibility
    """
    distribution_type: Literal["sphere", "slab", "clustered"] = "sphere"
    radius_um: float = 50.0
    slab_thickness_um: float = 20.0
    field_decay_length_um: float = 15.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.radius_um <= 0:
            raise ValueError(f"radius_um must be positive, got {self.radius_um}")
        if self.slab_thickness_um <= 0:
            raise ValueError(
                f"slab_thickness_um must be positive, got {self.slab_thickness_um}"
            )
        if self.field_decay_length_um <= 0:
            raise ValueError(
                f"field_decay_length_um must be positive, got {self.field_decay_length_um}"
            )


# Physical constants (module-level for convenience)
PLANCK_CONSTANT = 6.62607015e-34      # J*s
SPEED_OF_LIGHT = 2.99792458e8         # m/s
ELECTRON_CHARGE = 1.602176634e-19     # C
EV_TO_JOULES = 1.602176634e-19        # J/eV
VACUUM_PERMITTIVITY = 8.8541878128e-12  # F/m
