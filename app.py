"""
Neuro-SWARM3 SDK Interactive Dashboard

A Streamlit-based visualization for the Neuro-SWARM3 nanoparticle
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
    page_title="Neuro-SWARM3 SDK",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-0: #0b1020;
        --bg-1: #111c33;
        --panel: rgba(15, 22, 38, 0.92);
        --panel-2: rgba(18, 28, 48, 0.92);
        --panel-border: rgba(148, 163, 184, 0.18);
        --ink: #e6edf9;
        --muted: #9fb0c7;
        --accent: #2ec4b6;
        --accent-2: #f6c453;
        --accent-3: #ff6b6b;
        --accent-4: #8bd3ff;
        --glow: rgba(46, 196, 182, 0.35);
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
        color: var(--ink);
    }

    h1, h2, h3, h4 {
        font-family: 'Fraunces', 'Space Grotesk', serif;
        letter-spacing: -0.02em;
    }

    .stApp {
        background: linear-gradient(160deg, var(--bg-0) 0%, #0e1629 35%, var(--bg-1) 100%);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: -20vh;
        background:
            radial-gradient(35% 40% at 12% 10%, rgba(139, 211, 255, 0.25), transparent 60%),
            radial-gradient(35% 40% at 85% 12%, rgba(246, 196, 83, 0.2), transparent 60%),
            radial-gradient(40% 50% at 70% 85%, rgba(46, 196, 182, 0.18), transparent 60%);
        z-index: 0;
        pointer-events: none;
    }

    div[data-testid="stAppViewContainer"] {
        background: transparent;
    }

    section.main > div {
        position: relative;
        z-index: 1;
    }

    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0f1b31 0%, #0b1426 100%);
        border-right: 1px solid var(--panel-border);
    }

    section[data-testid="stSidebar"] h2 {
        color: var(--ink);
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .hero {
        background: radial-gradient(120% 120% at 10% 0%, rgba(46, 196, 182, 0.2), transparent 55%),
                    radial-gradient(100% 100% at 90% 10%, rgba(246, 196, 83, 0.22), transparent 60%),
                    var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        padding: 2.2rem 2.4rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
        animation: fadeUp 0.8s ease both;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(46, 196, 182, 0.18);
        color: var(--accent);
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: 0.8rem 0 0.6rem;
        font-size: 2.6rem;
        line-height: 1.05;
    }

    .hero p {
        margin: 0;
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 54rem;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-top: 1.6rem;
    }

    .hero-card {
        background: var(--panel-2);
        border: 1px solid var(--panel-border);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }

    .hero-card span {
        display: block;
        color: var(--muted);
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-card strong {
        display: block;
        font-size: 1.4rem;
        margin-top: 0.35rem;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .ns-card {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.25);
    }

    .ns-card h4 {
        margin: 0 0 0.75rem;
        font-size: 1.1rem;
    }

    .ns-list {
        margin: 0;
        padding-left: 1.1rem;
        color: var(--muted);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(15, 22, 38, 0.55);
        padding: 0.4rem;
        border-radius: 999px;
        border: 1px solid var(--panel-border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: transparent;
        color: var(--muted);
        padding: 0.45rem 1.1rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, rgba(46, 196, 182, 0.18), rgba(139, 211, 255, 0.2));
        color: var(--ink);
        border: 1px solid rgba(46, 196, 182, 0.5);
        box-shadow: 0 0 16px rgba(46, 196, 182, 0.25);
    }

    .stButton > button {
        background: linear-gradient(120deg, var(--accent), #3a86ff);
        color: #0b1020;
        border: none;
        border-radius: 999px;
        padding: 0.55rem 1.3rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        box-shadow: 0 10px 20px rgba(46, 196, 182, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 14px 26px rgba(46, 196, 182, 0.4);
    }

    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 8px 16px rgba(46, 196, 182, 0.3);
    }

    .stAlert {
        background: var(--panel);
        border: 1px solid var(--panel-border);
    }

    .stCaption {
        color: var(--muted) !important;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <section class="hero">
        <div class="hero-badge">Neuro-SWARM3 SDK</div>
        <h1>Wireless Neural Activity Readout in the NIR-II Window</h1>
        <p>
            Simulate electro-plasmonic nanoparticle probes, stress-test the decoding
            pipeline, and explore wavelength optimization for far-field neural detection.
        </p>
        <div class="hero-grid">
            <div class="hero-card">
                <span>Optical Window</span>
                <strong>1000-1700 nm</strong>
            </div>
            <div class="hero-card">
                <span>Probe Diameter</span>
                <strong>&lt; 200 nm</strong>
            </div>
            <div class="hero-card">
                <span>Target SSNR</span>
                <strong>~ 10^3</strong>
            </div>
            <div class="hero-card">
                <span>Model Stack</span>
                <strong>Izhikevich + Mie</strong>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)

# Sidebar configuration
st.sidebar.header("Simulation Controls")
st.sidebar.caption("Tweak the in silico experiment and noise conditions.")

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
    "Thermal Noise (sigma)",
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
    "Pipeline Overview",
    "Wavelength Sweep",
    "Noise Robustness",
    "Live Simulation"
])

# ============================================================================
# TAB 1: Pipeline Overview
# ============================================================================
with tab1:
    st.header("End-to-End Signal Processing Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create pipeline diagram
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0b1020')
        ax.set_facecolor('#0b1020')
        
        # Pipeline stages
        stages = [
            ("Izhikevich\nNeuron", "#2ec4b6", "Membrane\nPotential"),
            ("Drude-Lorentz\nDielectric", "#8bd3ff", "epsilon(w, E)"),
            ("Mie\nScattering", "#a7d49b", "Delta Q_sca"),
            ("Equation (1)\nPhoton Count", "#f6c453", "Delta N_ph"),
            ("Noise\nCorruption", "#f49db3", "Noisy Signal"),
            ("Signal\nExtractor", "#ff6b6b", "Decoded\nSpikes"),
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
                fontsize=9, fontweight='bold', color='#0b1020',
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
            0.5, 0.92, "Neuro-SWARM3 Forward Model -> Inverse Model Pipeline",
            ha='center', va='top', fontsize=14, fontweight='bold',
            color='white', transform=ax.transAxes
        )
        
        # Separator line (use plot instead of axhline for transform support)
        ax.plot([0.52, 0.58], [0.5, 0.5], color='#FF6B6B', 
                linestyle='--', linewidth=2, transform=ax.transAxes)
        ax.text(0.55, 0.75, "Forward Model", ha='center', fontsize=10,
                color='#2ec4b6', transform=ax.transAxes)
        ax.text(0.85, 0.75, "Inverse Model", ha='center', fontsize=10,
                color='#ff6b6b', transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Key Specifications")

        st.metric("Particle Size", "~150 nm", "Core-shell structure")
        st.metric("Wavelength", "1050 nm", "NIR-II window")
        st.metric("Target SSNR", "~10^3", "Shot-noise limited")
        st.metric("Resolution", "1 ms", "Temporal")

        st.markdown(
            """
            <div class="ns-card">
                <h4>Nanoparticle Structure</h4>
                <ul class="ns-list">
                    <li>SiOx core: 63 nm radius</li>
                    <li>Au shell: 5 nm thickness</li>
                    <li>PEDOT:PSS: 15 nm coating</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================================
# TAB 2: Wavelength Sweep
# ============================================================================
with tab2:
    st.header("Wavelength Optimization in the NIR-II Window")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Sweep Parameters")
        min_wl = st.number_input("Min Wavelength (nm)", value=1000, min_value=900, max_value=1200)
        max_wl = st.number_input("Max Wavelength (nm)", value=1700, min_value=1300, max_value=1800)
        step_wl = st.number_input("Step Size (nm)", value=25, min_value=5, max_value=50)
        e_field = st.number_input("Electric Field (mV/nm)", value=3.0, min_value=0.5, max_value=12.0)
    
    if st.button("Run Wavelength Sweep", key="sweep_btn"):
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
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#0b1020')
                
                for ax in [ax1, ax2]:
                    ax.set_facecolor('#121a2d')
                    ax.tick_params(colors='white')
                    for spine in ax.spines.values():
                        spine.set_color('white')
                
                wavelengths = sweep_result["wavelengths_nm"]
                delta_n_ph = sweep_result["delta_N_ph"]
                ssnr = sweep_result["ssnr"]
                optimal_wl = sweep_result["optimal_wavelength_nm"]
                
                # Plot 1: Delta N_ph
                ax1.plot(wavelengths, delta_n_ph, 'o-', color='#2ec4b6', linewidth=2, markersize=6)
                ax1.axvline(optimal_wl, color='#FF6B6B', linestyle='--', linewidth=2, label=f'Optimal: {optimal_wl:.0f} nm')
                ax1.set_ylabel('Delta N_ph (photons)', color='white', fontsize=12)
                ax1.set_title('Differential Photon Count vs Wavelength', color='white', fontsize=14, fontweight='bold')
                ax1.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
                ax1.grid(True, alpha=0.3, color='gray')
                
                # Plot 2: SSNR
                ax2.plot(wavelengths, ssnr, 's-', color='#f6c453', linewidth=2, markersize=6)
                ax2.axvline(optimal_wl, color='#FF6B6B', linestyle='--', linewidth=2)
                ax2.axhline(1000, color='#a7d49b', linestyle=':', linewidth=2, label='Target SSNR = 10^3')
                ax2.set_xlabel('Wavelength (nm)', color='white', fontsize=12)
                ax2.set_ylabel('SSNR', color='white', fontsize=12)
                ax2.set_title('Signal-to-Shot-Noise Ratio', color='white', fontsize=14, fontweight='bold')
                ax2.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
                ax2.grid(True, alpha=0.3, color='gray')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            # Display optimal wavelength
            st.success(f"Optimal wavelength: **{optimal_wl:.0f} nm** with SSNR = {ssnr[np.argmax(ssnr)]:.2e}")

# ============================================================================
# TAB 3: Noise Robustness
# ============================================================================
with tab3:
    st.header("Decoder Robustness Under Noise")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Test Parameters")
        noise_min = st.number_input("Min Noise (% of signal std)", value=5, min_value=0)
        noise_max = st.number_input("Max Noise (% of signal std)", value=50, min_value=5)
        noise_steps = st.number_input("Number of Steps", value=10, min_value=3, max_value=20)
        noise_trials = st.number_input("Trials per Level", value=5, min_value=1, max_value=20)
        use_matched = st.checkbox("Use Matched Filter", value=True)
        use_wiener = st.checkbox("Use Wiener Filter", value=True)
    
    if st.button("Run Robustness Test", key="robust_btn"):
        with st.spinner("Running noise sweep..."):
            # Generate clean signal
            config = SimulationConfig(duration=500, num_particles=1000, dt=0.1)
            physics = NeuroSwarmPhysics(config)
            sim_result = physics.simulate(input_rate_hz=10)

            clean_signal = sim_result["delta_N_ph"]
            true_spikes = sim_result["spikes"]
            dt_ms = config.dt
            tolerance_samples = max(1, int(5.0 / dt_ms))
            if use_matched:
                tolerance_samples = max(
                    tolerance_samples,
                    int(0.5 * DecodingParams().matched_filter_window_ms / dt_ms)
                )
            
            noise_levels = np.linspace(noise_min, noise_max, noise_steps)
            signal_scale = np.std(clean_signal)
            
            # Results storage
            f1_standard = []
            f1_robust = []
            
            for noise_std in noise_levels:
                f1_trials_std = []
                f1_trials_robust = []
                for trial in range(int(noise_trials)):
                # Corrupt signal
                    noise_params = NoiseParams(
                        shot_noise_enabled=False,
                        thermal_noise_std=(noise_std / 100.0) * signal_scale,
                        drift_amplitude=0.0,
                        burst_probability=0.0,
                        intensity_fluctuation_std=0.0,
                        seed=42 + trial
                    )
                    noise_gen = AdversarialNoiseGenerator(noise_params)
                    corrupted = noise_gen.corrupt_signal(clean_signal, dt_ms)
                    noisy = corrupted["noisy_signal"]
                
                # Standard decoder
                    decoder_std = SignalExtractor(DecodingParams(
                        spike_threshold=4.0,
                        spike_refractory_ms=6.0,
                        use_matched_filter=False,
                        use_wiener=False
                    ))
                    result_std = decoder_std.process_batch(noisy, dt_ms)
                    metrics_std = summarize_detection(
                        decoder_std,
                        result_std["spike_indices"],
                        true_spikes,
                        result_std["preprocessed"],
                        dt_ms=dt_ms,
                        tolerance_samples=tolerance_samples
                    )
                    f1_trials_std.append(metrics_std.f1_score)
                
                # Robust decoder
                    decoder_robust = SignalExtractor(DecodingParams(
                        spike_threshold=4.0,
                        spike_refractory_ms=6.0,
                        use_matched_filter=use_matched,
                        use_wiener=use_wiener
                    ))
                    result_robust = decoder_robust.process_batch(noisy, dt_ms)
                    metrics_robust = summarize_detection(
                        decoder_robust,
                        result_robust["spike_indices"],
                        true_spikes,
                        result_robust["preprocessed"],
                        dt_ms=dt_ms,
                        tolerance_samples=tolerance_samples
                    )
                    f1_trials_robust.append(metrics_robust.f1_score)

                f1_standard.append(float(np.mean(f1_trials_std)))
                f1_robust.append(float(np.mean(f1_trials_robust)))
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0b1020')
                ax.set_facecolor('#121a2d')
                
                ax.plot(noise_levels, f1_standard, 'o-', color='#ff6b6b', linewidth=2,
                        markersize=8, label='Standard Decoder')
                ax.plot(noise_levels, f1_robust, 's-', color='#2ec4b6', linewidth=2,
                        markersize=8, label='Robust Decoder (MF + Wiener)')
                
                ax.fill_between(noise_levels, f1_standard, f1_robust,
                               alpha=0.3, color='#2ec4b6')
                
                ax.axhline(0.8, color='#f6c453', linestyle='--', linewidth=2,
                          label='Acceptable F1 = 0.8')
                
                ax.set_xlabel('Thermal Noise (% of signal std)', color='white', fontsize=12)
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
            st.info(f"Average F1 improvement with robust decoder: **+{improvement:.3f}**")

# ============================================================================
# TAB 4: Live Simulation
# ============================================================================
with tab4:
    st.header("Interactive Forward/Inverse Simulation")
    
    if st.button("Run Full Simulation", key="sim_btn", type="primary"):
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
                spike_threshold=4.0,
                spike_refractory_ms=6.0,
                use_matched_filter=True,
                use_wiener=True
            ))
            decoded = decoder.process_batch(corrupted["noisy_signal"], config.dt)
            
            # Evaluation
            summary = summarize_detection(
                decoder,
                decoded["spike_indices"],
                sim_result["spikes"],
                decoded["preprocessed"],
                dt_ms=config.dt,
                tolerance_samples=max(
                    1,
                    int(5.0 / config.dt),
                    int(0.5 * decoder.params.matched_filter_window_ms / config.dt)
                )
            )

            tolerance_samples = max(
                1,
                int(5.0 / config.dt),
                int(0.5 * decoder.params.matched_filter_window_ms / config.dt)
            )
            eval_metrics = decoder.evaluate_reconstruction(
                decoded["spike_indices"],
                sim_result["spikes"],
                tolerance_samples=tolerance_samples
            )
            true_indices = np.where(sim_result["spikes"])[0]
            timing_errors = []
            for det in decoded["spike_indices"]:
                if len(true_indices) == 0:
                    break
                closest = true_indices[np.argmin(np.abs(true_indices - det))]
                timing_errors.append(abs(closest - det) * config.dt)
            timing_errors = np.array(timing_errors, dtype=float)
            std_timing_error_ms = float(np.std(timing_errors)) if timing_errors.size else float("nan")
            
            # Display results
            st.subheader("Simulation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("True Spikes", int(np.sum(sim_result["spikes"])))
            col2.metric("Detected Spikes", len(decoded["spike_indices"]))
            col3.metric("F1 Score", f"{summary.f1_score:.3f}")
            col4.metric("SSNR", f"{decoded['metrics'].ssnr:.2e}")
            
            # Plot results
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), facecolor='#0b1020')
            time = sim_result["time"]
            
            for ax in axes:
                ax.set_facecolor('#121a2d')
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
            axes[1].set_ylabel('Delta N_ph', color='white')
            axes[1].set_title('Clean Differential Photon Count', color='white', fontweight='bold')
            
            # Plot 3: Noisy signal
            axes[2].plot(time, corrupted["noisy_signal"], color='#DDA0DD', linewidth=0.5, alpha=0.8)
            axes[2].set_ylabel('Delta N_ph (noisy)', color='white')
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
                | Precision | {summary.precision:.3f} |
                | Recall | {summary.recall:.3f} |
                | F1 Score | {summary.f1_score:.3f} |
                | True Positives | {eval_metrics['true_positives']} |
                | False Positives | {eval_metrics['false_positives']} |
                | False Negatives | {eval_metrics['false_negatives']} |
                """)
            
            with perf_col2:
                st.markdown(f"""
                | Timing | Value |
                |--------|-------|
                | Mean Error | {summary.mean_timing_error_ms:.2f} ms |
                | Std Error | {std_timing_error_ms:.2f} ms |
                | Processing Time | {decoded['metrics'].processing_time_ms:.1f} ms |
                | Artifact Fraction | {decoded['metrics'].artifact_fraction*100:.1f}% |
                """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #5A6C7D;'>
        <p>Neuro-SWARM3 SDK | NIR-II Nanoparticle Neural Recording System</p>
        <p>Reference: Hardy et al., IEEE Photonics Technology Letters, 2021</p>
    </div>
    """,
    unsafe_allow_html=True
)
