"""
Tests for the physics module.

Validates:
1. Izhikevich neuron model produces expected spiking behavior
2. Drude-Lorentz dielectric function has correct properties in NIR-II range
3. Equation (1) photon count calculation is numerically reasonable
4. Full simulation pipeline produces expected SSNR values
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuroswarm.types import (
    SimulationConfig,
    DrudeLorenzParams,
    NanoparticleGeometry,
    OpticalSystemParams,
    IzhikevichParams,
)
from neuroswarm.physics import (
    IzhikevichNeuron,
    DrudeLorenzModel,
    MieScattering,
    NeuroSwarmPhysics,
    compress_to_integration_time,
)


class TestIzhikevichNeuron:
    """Tests for the Izhikevich neuron model."""

    def test_initialization(self):
        """Test neuron initializes with correct state."""
        neuron = IzhikevichNeuron()
        assert neuron.v == -65.0
        assert neuron.u == neuron.params.b * neuron.params.v_init

    def test_reset(self):
        """Test neuron reset restores initial state."""
        neuron = IzhikevichNeuron()
        neuron.v = 0.0
        neuron.u = 100.0
        neuron.reset()
        assert neuron.v == neuron.params.v_init
        assert neuron.u == neuron.params.u_init

    def test_single_step(self):
        """Test single time step updates state."""
        neuron = IzhikevichNeuron()
        v_init = neuron.v
        v_new, spiked = neuron.step(I=0.0, dt=0.1)
        assert isinstance(v_new, float)
        assert isinstance(spiked, bool)

    def test_spike_generation(self):
        """Test that sufficient input produces spikes."""
        neuron = IzhikevichNeuron()

        n_steps = 1000
        dt = 0.1
        I_input = np.ones(n_steps) * 10.0

        v_trace, spikes = neuron.simulate(I_input, dt)

        assert np.sum(spikes) > 0, "Neuron should spike with strong input"

    def test_phasic_spiking_behavior(self):
        """Test that default params produce phasic spiking."""
        neuron = IzhikevichNeuron()

        n_steps = 2000
        dt = 0.1
        I_input = np.zeros(n_steps)
        I_input[500:1500] = 5.0

        v_trace, spikes = neuron.simulate(I_input, dt)

        spike_times = np.where(spikes)[0]
        spikes_during_input = np.sum((spike_times >= 500) & (spike_times < 1500))

        assert 0 < spikes_during_input < 10, (
            f"Phasic spiking expected few spikes, got {spikes_during_input}"
        )

    def test_custom_parameters(self):
        """Test neuron with custom parameters."""
        params = IzhikevichParams(a=0.02, b=0.2, c=-65, d=8)
        neuron = IzhikevichNeuron(params)

        assert neuron.params.a == 0.02
        assert neuron.params.d == 8


class TestDrudeLorenzModel:
    """Tests for the Drude-Lorentz dielectric model."""

    def test_initialization(self):
        """Test model initializes with default parameters."""
        model = DrudeLorenzModel()
        assert model.params.epsilon_inf == 2.75
        assert model.params.omega_p == 1.325

    def test_dielectric_function_shape(self):
        """Test dielectric function returns correct shape."""
        model = DrudeLorenzModel()
        energies = np.linspace(0.5, 2.0, 100)
        epsilon = model.dielectric_function(energies)

        assert epsilon.shape == energies.shape
        assert np.iscomplexobj(epsilon)

    def test_dielectric_function_nir_range(self):
        """Test dielectric function in NIR-II range."""
        model = DrudeLorenzModel()

        h_eV_s = 4.135667696e-15
        c_nm_s = 299792458e9
        wavelengths_nm = np.array([1000, 1050, 1200, 1500, 1700])
        energies_eV = h_eV_s * c_nm_s / wavelengths_nm

        epsilon = model.dielectric_function(energies_eV)

        assert np.all(np.isfinite(epsilon)), "Dielectric function should be finite"

    def test_dielectric_modulation(self):
        """Test that electric field causes dielectric modulation."""
        model = DrudeLorenzModel()
        energy = np.array([1.18])

        eps_base = model.dielectric_function(energy)
        delta_omega_p = model.plasma_frequency_shift(12.0)
        eps_modulated = model.dielectric_function(energy, delta_omega_p)

        assert not np.allclose(eps_base, eps_modulated), (
            "Electric field should modulate dielectric function"
        )

    def test_plasma_frequency_shift(self):
        """Test plasma frequency shift scales with field."""
        model = DrudeLorenzModel()

        shift_0 = model.plasma_frequency_shift(0.0)
        shift_6 = model.plasma_frequency_shift(6.0)
        shift_12 = model.plasma_frequency_shift(12.0)

        assert shift_0 == 0.0
        assert shift_6 > 0
        assert shift_12 > shift_6, "Shift should increase with field"


class TestMieScattering:
    """Tests for the Mie scattering model."""

    def test_initialization(self):
        """Test model initializes with default geometry."""
        model = MieScattering()
        assert model.geometry.core_radius == 63.0
        assert model.geometry.shell_thickness == 5.0
        assert model.geometry.coating_thickness == 15.0

    def test_scattering_efficiency_positive(self):
        """Test scattering efficiency is positive."""
        model = MieScattering()
        epsilon_coating = 2.5 + 0.1j

        Q_sca = model.scattering_efficiency(1050.0, epsilon_coating)

        assert Q_sca > 0, "Scattering efficiency must be positive"
        assert np.isfinite(Q_sca), "Scattering efficiency must be finite"

    def test_scattering_cross_section(self):
        """Test scattering cross-section is in expected range."""
        model = MieScattering()
        epsilon_coating = 2.5 + 0.1j

        C_sca = model.scattering_cross_section(1050.0, epsilon_coating)

        assert 1e2 < C_sca < 1e7, f"Cross-section {C_sca} nm^2 out of expected range"


class TestNeuroSwarmPhysics:
    """Tests for the complete forward model."""

    def test_initialization(self):
        """Test model initializes correctly."""
        physics = NeuroSwarmPhysics()
        assert physics.config is not None
        assert physics.neuron is not None
        assert physics.dielectric is not None
        assert physics.scattering is not None

    def test_simulation_runs(self):
        """Test that simulation completes without error."""
        config = SimulationConfig(duration=100.0, dt=0.1)
        physics = NeuroSwarmPhysics(config)

        result = physics.simulate()

        assert "time" in result
        assert "membrane_potential" in result
        assert "spikes" in result
        assert "delta_N_ph" in result
        assert "ssnr" in result

    def test_simulation_output_shapes(self):
        """Test that simulation outputs have correct shapes."""
        config = SimulationConfig(duration=100.0, dt=0.1)
        physics = NeuroSwarmPhysics(config)

        result = physics.simulate()

        n_expected = int(100.0 / 0.1)
        assert len(result["time"]) == n_expected
        assert len(result["membrane_potential"]) == n_expected
        assert len(result["spikes"]) == n_expected
        assert len(result["delta_N_ph"]) == n_expected

    def test_spikes_produce_signal(self):
        """Test that spikes produce differential photon signal."""
        config = SimulationConfig(duration=500.0, dt=0.1)
        physics = NeuroSwarmPhysics(config)

        n_steps = config.num_steps
        I_input = np.ones(n_steps) * 10.0

        result = physics.simulate(input_current=I_input)

        assert np.sum(result["spikes"]) > 0, "Should produce spikes"

        delta_N = result["delta_N_ph"]
        assert np.std(delta_N) > 0, "Signal should vary with spikes"

    def test_ssnr_reasonable(self):
        """Test that SSNR is in reasonable range."""
        config = SimulationConfig(
            duration=1000.0,
            dt=0.1,
            num_particles=1000
        )
        physics = NeuroSwarmPhysics(config)

        result = physics.simulate(input_rate_hz=10.0)

        ssnr = result["ssnr"]
        assert ssnr > 0, "SSNR should be positive"


class TestCompressToIntegrationTime:
    """Tests for the data compression utility."""

    def test_max_compression(self):
        """Test max compression preserves peaks."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        compressed = compress_to_integration_time(data, dt=0.1, integration_time=0.5)

        assert len(compressed) == 2
        assert compressed[0] == 5
        assert compressed[1] == 10

    def test_mean_compression(self):
        """Test mean compression."""
        data = np.array([2, 4, 6, 8, 10, 12], dtype=float)
        compressed = compress_to_integration_time(
            data, dt=0.1, integration_time=0.2, method="mean"
        )

        # 6 samples at 0.1ms, 2 samples per 0.2ms = 3 bins
        assert len(compressed) == 3
        assert compressed[0] == 3.0  # mean(2, 4)
        assert compressed[1] == 7.0  # mean(6, 8)
        assert compressed[2] == 11.0  # mean(10, 12)

    def test_no_compression_needed(self):
        """Test when integration time equals dt."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        compressed = compress_to_integration_time(data, dt=1.0, integration_time=1.0)

        assert len(compressed) == 5
        np.testing.assert_array_equal(compressed, data)


class TestDrudeLorenzParams:
    """Tests for DrudeLorenzParams validation."""

    def test_default_values(self):
        """Test default parameter values."""
        params = DrudeLorenzParams()
        assert params.epsilon_inf == 2.75
        assert params.omega_p == 1.325
        assert params.gamma == 0.271

    def test_invalid_epsilon_inf(self):
        """Test validation rejects non-positive epsilon_inf."""
        with pytest.raises(ValueError, match="epsilon_inf"):
            DrudeLorenzParams(epsilon_inf=0.5)

    def test_invalid_omega_p(self):
        """Test validation rejects non-positive omega_p."""
        with pytest.raises(ValueError, match="omega_p"):
            DrudeLorenzParams(omega_p=0.0)


class TestNanoparticleGeometry:
    """Tests for NanoparticleGeometry."""

    def test_default_values(self):
        """Test default geometry values."""
        geo = NanoparticleGeometry()
        assert geo.core_radius == 63.0
        assert geo.shell_thickness == 5.0
        assert geo.coating_thickness == 15.0

    def test_total_radius(self):
        """Test total radius calculation."""
        geo = NanoparticleGeometry()
        expected = 63.0 + 5.0 + 15.0
        assert geo.total_radius == expected

    def test_total_diameter(self):
        """Test total diameter calculation."""
        geo = NanoparticleGeometry()
        expected = 2.0 * (63.0 + 5.0 + 15.0)
        assert geo.total_diameter == expected

    def test_invalid_core_radius(self):
        """Test validation rejects non-positive core radius."""
        with pytest.raises(ValueError):
            NanoparticleGeometry(core_radius=0.0)


class TestIntegration:
    """Integration tests for the complete physics pipeline."""

    def test_full_pipeline(self):
        """Test complete simulation and analysis pipeline."""
        config = SimulationConfig(
            duration=200.0,
            dt=0.1,
            num_particles=1000
        )
        physics = NeuroSwarmPhysics(config)

        result = physics.simulate(input_rate_hz=10.0)

        compressed = compress_to_integration_time(
            result["delta_N_ph"],
            dt=config.dt,
            integration_time=1.0
        )

        assert len(compressed) == int(config.duration)
        assert np.std(compressed) > 0

    def test_parameter_sensitivity(self):
        """Test that model responds to parameter changes."""
        config1 = SimulationConfig(num_particles=1000)
        physics1 = NeuroSwarmPhysics(config1)
        result1 = physics1.simulate(input_rate_hz=10.0)

        config2 = SimulationConfig(num_particles=10000)
        physics2 = NeuroSwarmPhysics(config2)
        result2 = physics2.simulate(input_rate_hz=10.0)

        max1 = np.max(np.abs(result1["delta_N_ph"]))
        max2 = np.max(np.abs(result2["delta_N_ph"]))

        assert max2 > max1, "More particles should give stronger signal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
