"""
Neuro-SWARM³ SDK Interactive Dashboard

A Streamlit-based visualization for the Neuro-SWARM³ nanoparticle
neural recording system.

Run with: streamlit run app.py
"""

import sys
sys.path.insert(0, "src")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches

# Import neuroswarm modules
from neuroswarm.physics import NeuroSwarmPhysics, IzhikevichNeuron
from neuroswarm.noise import AdversarialNoiseGenerator, NoiseParams
from neuroswarm.decoding import SignalExtractor, DecodingParams
from neuroswarm.types import (
    SimulationConfig,
    WavelengthSweepParams,
    OpticalSystemParams,
    NanoparticleGeometry,
)
from neuroswarm.reporting import summarize_detection

# Page config
st.set_page_config(
    page_title="Neuro-SWARM³ SDK",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🧠 Neuro-SWARM³ SDK</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">System-on-a-Nanoparticle for Wireless Neural Recording</p>',
    unsafe_allow_html=True
)

# Sidebar configuration
st.sidebar.header("⚙️ Simulation Parameters")

# Simulation duration
duration_ms = st.sidebar.slider(
    "Simulation Duration (ms)",
    min_value=100,
    max_value=2000,
    value=500,
    step=100
)

# Number of particles
num_particles = st.sidebar.slider(
    "Number of Nanoparticles",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

# Input rate
input_rate = st.sidebar.slider(
    "Neural Input Rate (Hz)",
    min_value=1,
    max_value=50,
    value=10
)

# Noise parameters
st.sidebar.subheader("Noise Settings")
thermal_noise = st.sidebar.slider(
    "Thermal Noise (σ)",
    min_value=0,
    max_value=500,
    value=100
)
drift_amplitude = st.sidebar.slider(
    "Drift Amplitude (%)",
    min_value=0,
    max_value=20,
    value=5
) / 100.0

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Pipeline Overview",
    "🌈 Wavelength Sweep",
    "📉 Noise Robustness",
    "🔬 Live Simulation"
])

# ============================================================================
# TAB 1: Pipeline Overview
# ============================================================================
with tab1:
    st.header("End-to-End Signal Processing Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create pipeline diagram
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        
        # Pipeline stages
        stages = [
            ("Izhikevich\nNeuron", "#4ECDC4", "Membrane\nPotential"),
            ("Drude-Lorentz\nDielectric", "#45B7D1", "ε(ω, E)"),
            ("Mie\nScattering", "#96CEB4", "ΔQ_sca"),
            ("Equation (1)\nPhoton Count", "#FFEAA7", "ΔN_ph"),
            ("Noise\nCorruption", "#DDA0DD", "Noisy Signal"),
            ("Signal\nExtractor", "#FF6B6B", "Decoded\nSpikes"),
        ]
        
        box_width = 0.12
        box_height = 0.25
        y_center = 0.5
        spacing = 0.15
        start_x = 0.05
        
        for i, (name, color, output) in enumerate(stages):
            x = start_x + i * spacing
            
            # Draw box
            rect = plt.Rectangle(
                (x, y_center - box_height/2), box_width, box_height,
                facecolor=color, edgecolor='white', linewidth=2,
                transform=ax.transAxes, zorder=2
            )
            ax.add_patch(rect)
            
            # Stage name
            ax.text(
                x + box_width/2, y_center,
                name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='#1a1a2e',
                transform=ax.transAxes, zorder=3
            )
            
            # Output label
            ax.text(
                x + box_width/2, y_center - box_height/2 - 0.08,
                output, ha='center', va='top',
                fontsize=8, color='white', style='italic',
                transform=ax.transAxes
            )
            
            # Arrow to next stage
            if i < len(stages) - 1:
                ax.annotate(
                    '', xy=(x + spacing, y_center),
                    xytext=(x + box_width + 0.01, y_center),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2),
                    transform=ax.transAxes
                )
        
        # Title
        ax.text(
            0.5, 0.92, "Neuro-SWARM³ Forward Model → Inverse Model Pipeline",
            ha='center', va='top', fontsize=14, fontweight='bold',
            color='white', transform=ax.transAxes
        )
        
        # Separator line (use plot instead of axhline for transform support)
        ax.plot([0.52, 0.58], [0.5, 0.5], color='#FF6B6B', 
                linestyle='--', linewidth=2, transform=ax.transAxes)
        ax.text(0.55, 0.75, "Forward Model", ha='center', fontsize=10, 
                color='#4ECDC4', transform=ax.transAxes)
        ax.text(0.85, 0.75, "Inverse Model", ha='center', fontsize=10, 
                color='#FF6B6B', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Key Specifications")
        
        st.metric("Particle Size", "~150 nm", "Core-Shell Structure")
        st.metric("Wavelength", "1050 nm", "NIR-II Window")
        st.metric("Target SSNR", "~10³", "Shot-Noise Limited")
        st.metric("Resolution", "1 ms", "Temporal")
        
        st.markdown("---")
        st.markdown("""
        **Nanoparticle Structure:**
        - 🔵 SiOx Core: 63 nm radius
        - 🟡 Au Shell: 5 nm thickness
        - 🟣 PEDOT:PSS: 15 nm coating
        """)

# ============================================================================
# TAB 2: Wavelength Sweep
# ============================================================================
with tab2:
    st.header("Wavelength Optimization in NIR-II Window")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Sweep Parameters")
        min_wl = st.number_input("Min Wavelength (nm)", value=1000, min_value=900, max_value=1200)
        max_wl = st.number_input("Max Wavelength (nm)", value=1700, min_value=1300, max_value=1800)
        step_wl = st.number_input("Step Size (nm)", value=25, min_value=5, max_value=50)
        e_field = st.number_input("Electric Field (mV/nm)", value=3.0, min_value=0.5, max_value=12.0)
    
    if st.button("🔍 Run Wavelength Sweep", key="sweep_btn"):
        with st.spinner("Sweeping wavelengths..."):
            # Run sweep
            config = SimulationConfig(duration=100, num_particles=1000)
            physics = NeuroSwarmPhysics(config)
            
            sweep_params = WavelengthSweepParams(
                min_wavelength=float(min_wl),
                max_wavelength=float(max_wl),
                step_nm=float(step_wl),
                electric_field=float(e_field)
            )
            
            sweep_result = physics.sweep_wavelengths(sweep_params)
            
            with col1:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0e1117')
                
                for ax in [ax1, ax2]:
                    ax.set_facecolor('#1a1a2e')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_color('white')
                
                wavelengths = sweep_result["wavelengths_nm"]
                delta_n_ph = sweep_result["delta_N_ph"]
                ssnr = sweep_result["ssnr"]
                optimal_wl = sweep_result["optimal_wavelength_nm"]
                
                # Plot 1: ΔN_ph
                ax1.plot(wavelengths, delta_n_ph, 'o-', color='#4ECDC4', linewidth=2, markersize=6)
                ax1.axvline(optimal_wl, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Optimal: {optimal_wl:.0f} nm')
                ax1.set_ylabel('ΔN_ph (photons)', color='white', fontsize=12)
                ax1.set_title('Differential Photon Count vs Wavelength', color='white', fontsize=14, fontweight='bold')
                ax1.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
                ax1.grid(True, alpha=0.3, color='gray')
                
                # Plot 2: SSNR
                ax2.plot(wavelengths, ssnr, 's-', color='#FFEAA7', linewidth=2, markersize=6)
                ax2.axvline(optimal_wl, color='#FF6B6B', linestyle='--', linewidth=2)
                ax2.axhline(1000, color='#96CEB4', linestyle=':', linewidth=2, label='Target SSNR = 10³')
                ax2.set_xlabel('Wavelength (nm)', color='white', fontsize=12)
                ax2.set_ylabel('SSNR', color='white', fontsize=12)
                ax2.set_title('Signal-to-Shot-Noise Ratio', color='white', fontsize=14, fontweight='bold')
                ax2.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
                ax2.grid(True, alpha=0.3, color='gray')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Display optimal wavelength
            st.success(f"✅ Optimal wavelength: **{optimal_wl:.0f} nm** with SSNR = {ssnr[np.argmax(ssnr)]:.2e}")

# ============================================================================
# TAB 3: Noise Robustness
# ============================================================================
with tab3:
    st.header("Decoder Robustness Under Noise")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Test Parameters")
        noise_min = st.number_input("Min Noise σ", value=50, min_value=0)
        noise_max = st.number_input("Max Noise σ", value=500, min_value=100)
        noise_steps = st.number_input("Number of Steps", value=10, min_value=3, max_value=20)
        use_matched = st.checkbox("Use Matched Filter", value=True)
        use_wiener = st.checkbox("Use Wiener Filter", value=True)
    
    if st.button("🧪 Run Robustness Test", key="robust_btn"):
        with st.spinner("Running noise sweep..."):
            # Generate clean signal
            config = SimulationConfig(duration=500, num_particles=1000, dt=0.1)
            physics = NeuroSwarmPhysics(config)
            sim_result = physics.simulate(input_rate_hz=10)
            
            clean_signal = sim_result["delta_N_ph"]
            true_spikes = sim_result["spikes"]
            dt_ms = config.dt
            
            noise_levels = np.linspace(noise_min, noise_max, noise_steps)
            
            # Results storage
            f1_standard = []
            f1_robust = []
            
            for noise_std in noise_levels:
                # Corrupt signal
                noise_params = NoiseParams(thermal_noise_std=noise_std, seed=42)
                noise_gen = AdversarialNoiseGenerator(noise_params)
                corrupted = noise_gen.corrupt_signal(clean_signal, dt_ms)
                noisy = corrupted["noisy_signal"]
                
                # Standard decoder
                decoder_std = SignalExtractor(DecodingParams(
                    use_matched_filter=False,
                    use_wiener=False
                ))
                result_std = decoder_std.process_batch(noisy, dt_ms)
                metrics_std = summarize_detection(true_spikes, result_std["spike_indices"], dt_ms)
                f1_standard.append(metrics_std["f1_score"])
                
                # Robust decoder
                decoder_robust = SignalExtractor(DecodingParams(
                    use_matched_filter=use_matched,
                    use_wiener=use_wiener
                ))
                result_robust = decoder_robust.process_batch(noisy, dt_ms)
                metrics_robust = summarize_detection(true_spikes, result_robust["spike_indices"], dt_ms)
                f1_robust.append(metrics_robust["f1_score"])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0e1117')
                ax.set_facecolor('#1a1a2e')
                
                ax.plot(noise_levels, f1_standard, 'o-', color='#FF6B6B', linewidth=2, 
                        markersize=8, label='Standard Decoder')
                ax.plot(noise_levels, f1_robust, 's-', color='#4ECDC4', linewidth=2, 
                        markersize=8, label='Robust Decoder (MF + Wiener)')
                
                ax.fill_between(noise_levels, f1_standard, f1_robust, 
                               alpha=0.3, color='#4ECDC4')
                
                ax.axhline(0.8, color='#FFEAA7', linestyle='--', linewidth=2, 
                          label='Acceptable F1 = 0.8')
                
                ax.set_xlabel('Thermal Noise σ (photon counts)', color='white', fontsize=12)
                ax.set_ylabel('F1 Score', color='white', fontsize=12)
                ax.set_title('Spike Detection Performance vs Noise Level', 
                            color='white', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 1.05)
                ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
                ax.grid(True, alpha=0.3, color='gray')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Summary metrics
            improvement = np.mean(np.array(f1_robust) - np.array(f1_standard))
            st.info(f"📈 Average F1 improvement with robust decoder: **+{improvement:.3f}**")

# ============================================================================
# TAB 4: Live Simulation
# ============================================================================
with tab4:
    st.header("Interactive Forward/Inverse Simulation")
    
    if st.button("▶️ Run Full Simulation", key="sim_btn", type="primary"):
        with st.spinner("Running simulation..."):
            # Configure simulation
            config = SimulationConfig(
                duration=duration_ms,
                num_particles=num_particles,
                dt=0.1
            )
            
            # Forward model
            physics = NeuroSwarmPhysics(config)
            sim_result = physics.simulate(input_rate_hz=input_rate)
            
            # Add noise
            noise_params = NoiseParams(
                thermal_noise_std=thermal_noise,
                drift_amplitude=drift_amplitude,
                seed=42
            )
            noise_gen = AdversarialNoiseGenerator(noise_params)
            corrupted = noise_gen.corrupt_signal(
                sim_result["delta_N_ph"], 
                config.dt,
                baseline=1e5
            )
            
            # Inverse model (decoder)
            decoder = SignalExtractor(DecodingParams(
                use_matched_filter=True,
                use_wiener=True
            ))
            decoded = decoder.process_batch(corrupted["noisy_signal"], config.dt)
            
            # Evaluation
            metrics = summarize_detection(
                sim_result["spikes"],
                decoded["spike_indices"],
                config.dt
            )
            
            # Display results
            st.subheader("Simulation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Spikes", int(np.sum(sim_result["spikes"])))
            col2.metric("Detected Spikes", len(decoded["spike_indices"]))
            col3.metric("F1 Score", f"{metrics['f1_score']:.3f}")
            col4.metric("SSNR", f"{decoded['metrics'].ssnr:.2e}")
            
            # Plot results
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), facecolor='#0e1117')
            time = sim_result["time"]
            
            for ax in axes:
                ax.set_facecolor('#1a1a2e')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')
            
            # Plot 1: Membrane potential
            axes[0].plot(time, sim_result["membrane_potential"], color='#4ECDC4', linewidth=0.8)
            spike_times = time[sim_result["spikes"]]
            axes[0].scatter(spike_times, np.ones_like(spike_times) * 30, 
                           color='#FF6B6B', s=50, marker='v', zorder=5)
            axes[0].set_ylabel('V (mV)', color='white')
            axes[0].set_title('Membrane Potential (Izhikevich Model)', color='white', fontweight='bold')
            
            # Plot 2: Clean photon signal
            axes[1].plot(time, sim_result["delta_N_ph"], color='#FFEAA7', linewidth=0.8)
            axes[1].set_ylabel('ΔN_ph', color='white')
            axes[1].set_title('Clean Differential Photon Count', color='white', fontweight='bold')
            
            # Plot 3: Noisy signal
            axes[2].plot(time, corrupted["noisy_signal"], color='#DDA0DD', linewidth=0.5, alpha=0.8)
            axes[2].set_ylabel('ΔN_ph (noisy)', color='white')
            axes[2].set_title('Noisy Signal (After Adversarial Corruption)', color='white', fontweight='bold')
            
            # Plot 4: Decoded spikes
            axes[3].plot(time[:len(decoded["preprocessed"])], decoded["preprocessed"], 
                        color='#96CEB4', linewidth=0.8)
            detected_times = time[decoded["spike_indices"]]
            axes[3].scatter(detected_times, 
                           decoded["preprocessed"][decoded["spike_indices"]],
                           color='#FF6B6B', s=80, marker='*', zorder=5, label='Detected')
            axes[3].set_xlabel('Time (ms)', color='white')
            axes[3].set_ylabel('Processed', color='white')
            axes[3].set_title('Decoded Signal with Detected Spikes', color='white', fontweight='bold')
            axes[3].legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
            
            for ax in axes:
                ax.grid(True, alpha=0.2, color='gray')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Performance summary
            st.subheader("Detection Performance")
            perf_col1, perf_col2 = st.columns(2)
            
            with perf_col1:
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Precision | {metrics['precision']:.3f} |
                | Recall | {metrics['recall']:.3f} |
                | F1 Score | {metrics['f1_score']:.3f} |
                | True Positives | {metrics['true_positives']} |
                | False Positives | {metrics['false_positives']} |
                | False Negatives | {metrics['false_negatives']} |
                """)
            
            with perf_col2:
                st.markdown(f"""
                | Timing | Value |
                |--------|-------|
                | Mean Error | {metrics['mean_timing_error_ms']:.2f} ms |
                | Std Error | {metrics['std_timing_error_ms']:.2f} ms |
                | Processing Time | {decoded['metrics'].processing_time_ms:.1f} ms |
                | Artifact Fraction | {decoded['metrics'].artifact_fraction*100:.1f}% |
                """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #5A6C7D;'>
        <p>Neuro-SWARM³ SDK | NIR-II Nanoparticle Neural Recording System</p>
        <p>Reference: Hardy et al., IEEE Photonics Technology Letters, 2021</p>
    </div>
    """,
    unsafe_allow_html=True
)
